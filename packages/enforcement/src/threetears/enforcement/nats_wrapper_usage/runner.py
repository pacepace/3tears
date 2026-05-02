"""pytest-friendly orchestration for nats-wrapper-usage enforcement.

a single :func:`run_nats_enforcement` entry point lets each consumer's
thin shell invoke either of the two walkers (production / tests) or
both. the runner is the policy point: it resolves src roots for the
production scan, applies the whole-file exemption list parsed from
``_nats_exemptions.txt``, emits the standardised report, and either
raises :func:`pytest.fail` or returns silently according to the
configured mode.

deviation from sibling domains' exemption handling: this domain's
exemption file format is **whole-file path entries**, not the
``file:line:symbol`` triple format that
:func:`~threetears.enforcement.common.exemptions.parse_exemptions_with_rationale`
parses. preserving the existing files unchanged across the bot-trio
repos is a hard requirement (the plan calls out that exemption files
are not migrated as part of this shard), so the runner uses a
domain-local parser that still enforces the same rationale-discipline
rules — non-empty rationale, blanket-phrase rejection, minimum
length — by delegating those checks to the shared module's helpers
where possible.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pytest

from threetears.enforcement.common import (
    Exemption,
    ExemptionError,
    MODE_REPORT,
    MODE_STRICT,
    Violation,
    discover_src_roots,
    emit_report,
    relative_posix_path,
    resolve_mode,
)

from threetears.enforcement.nats_wrapper_usage.config import (
    NatsWrapperConfig,
)
from threetears.enforcement.nats_wrapper_usage.walkers import (
    find_direct_nats_imports,
    find_test_nats_imports,
)

__all__ = ["run_nats_enforcement"]


_VALID_WALKERS: frozenset[str] = frozenset({"production", "tests", "all"})

_MIN_RATIONALE_LENGTH = 30

_BLANKET_RATIONALE_PHRASES: frozenset[str] = frozenset({
    "internal access needed",
    "tests need access",
    "tests need this",
    "same-file colocation",
})


@dataclass(frozen=True)
class _PathExemption:
    """one parsed whole-file exemption entry.

    :ivar path: forward-slash repo-relative path of the exempted file
    :ivar rationale: text from the ``# rationale: <reason>`` line, prefix-stripped
    """

    path: str
    rationale: str


def run_nats_enforcement(
    config: NatsWrapperConfig, walker: str = "all",
) -> None:
    """run the configured walker(s), emit report, fail if strict.

    accepted ``walker`` values:

    - ``"production"``: only the production import walker runs;
      ``tests_root`` is ignored.
    - ``"tests"``: only the tests import walker runs;
      ``src_roots`` / discovery are ignored. when
      :attr:`~NatsWrapperConfig.tests_root` is ``None`` no work is
      done.
    - ``"all"`` (default): both walkers run.

    src roots for the production walker come from
    :attr:`NatsWrapperConfig.src_roots` when set, else from
    :func:`discover_src_roots
    <threetears.enforcement.common.pyproject_discovery.discover_src_roots>`
    so the walker sees every transitively-reachable path-dep src
    tree.

    exemptions live in
    :attr:`NatsWrapperConfig.exemptions_path`. the file format is the
    canonical whole-file form: one repo-relative path per line,
    preceded by a ``# rationale: <reason>`` comment line. blanket
    rationales (e.g. ``"internal access needed"``, ``"tests need
    this"``) are rejected, as are rationales shorter than the
    sibling-domain threshold (30 characters). a violation is exempted
    iff its file's relative posix path is listed in the exemption
    file.

    in :data:`~threetears.enforcement.common.modes.MODE_REPORT` the
    runner returns normally regardless of violations; in
    :data:`~threetears.enforcement.common.modes.MODE_STRICT` (the
    default) it calls :func:`pytest.fail` with the rendered report.

    :param config: per-repo enforcement config
    :ptype config: NatsWrapperConfig
    :param walker: which walker(s) to invoke; one of ``"production"``,
        ``"tests"``, ``"all"``
    :ptype walker: str
    :raises ValueError: ``walker`` is not in the accepted set
    :raises ExemptionError: the exemption file violates the
        rationale-discipline contract
    :raises pytest.fail.Exception: in strict mode with violations
    """
    if walker not in _VALID_WALKERS:
        raise ValueError(
            f"walker must be one of {sorted(_VALID_WALKERS)}, got {walker!r}"
        )

    src_roots = _resolve_src_roots(config)
    path_exemptions = _load_path_exemptions(config.exemptions_path)
    exempt_paths = {entry.path for entry in path_exemptions}

    violations = _run_walkers(config, walker, src_roots)
    filtered = _apply_path_exemptions(
        violations, exempt_paths, config.repo_root,
    )

    mode = resolve_mode(config.mode_env_var, default=MODE_STRICT)

    scanned_roots = _scanned_roots(config, walker, src_roots)
    report = emit_report(
        filtered,
        scanned_roots,
        _to_common_exemptions(path_exemptions),
        mode,
        config.repo_root,
        domain=f"nats_wrapper_usage.{walker}",
    )
    print(report, file=sys.stderr)

    if mode == MODE_REPORT:
        return
    if filtered:
        pytest.fail(
            f"nats-wrapper-usage enforcement found {len(filtered)} "
            f"violation(s):\n{report}"
        )


def _resolve_src_roots(config: NatsWrapperConfig) -> tuple[Path, ...]:
    """pick the production src roots to scan.

    explicit ``config.src_roots`` wins; otherwise we walk the path-dep
    graph so the walker sees siblings declared as workspace members
    or path-deps. this is the option-B behaviour described in the
    task plan.

    :param config: per-repo enforcement config
    :ptype config: NatsWrapperConfig
    :return: src roots to scan for the production walker
    :rtype: tuple[Path, ...]
    """
    if config.src_roots is not None:
        return config.src_roots
    return discover_src_roots(config.repo_root)


def _run_walkers(
    config: NatsWrapperConfig,
    walker: str,
    src_roots: tuple[Path, ...],
) -> list[Violation]:
    """invoke the requested walker(s) and return aggregated violations.

    :param config: per-repo enforcement config
    :ptype config: NatsWrapperConfig
    :param walker: walker selector (``"production"``, ``"tests"``,
        ``"all"``)
    :ptype walker: str
    :param src_roots: production src roots resolved by
        :func:`_resolve_src_roots`
    :ptype src_roots: tuple[Path, ...]
    :return: combined violation list, production first then tests
    :rtype: list[Violation]
    """
    violations: list[Violation] = []
    if walker in {"production", "all"}:
        violations.extend(
            find_direct_nats_imports(
                src_roots, config.repo_root, config.forbidden_module,
            )
        )
    if walker in {"tests", "all"}:
        violations.extend(
            find_test_nats_imports(
                config.tests_root, config.repo_root, config.forbidden_module,
            )
        )
    return violations


def _scanned_roots(
    config: NatsWrapperConfig,
    walker: str,
    src_roots: tuple[Path, ...],
) -> tuple[Path, ...]:
    """compute the inventory of roots actually scanned for the report.

    :param config: per-repo enforcement config
    :ptype config: NatsWrapperConfig
    :param walker: walker selector
    :ptype walker: str
    :param src_roots: production src roots resolved by
        :func:`_resolve_src_roots`
    :ptype src_roots: tuple[Path, ...]
    :return: src roots inspected (tests root included only when
        scanning tests and the path is set)
    :rtype: tuple[Path, ...]
    """
    roots: list[Path] = []
    if walker in {"production", "all"}:
        roots.extend(src_roots)
    if walker in {"tests", "all"} and config.tests_root is not None:
        roots.append(config.tests_root)
    return tuple(roots)


def _apply_path_exemptions(
    violations: list[Violation],
    exempt_paths: set[str],
    repo_root: Path,
) -> list[Violation]:
    """drop violations whose file is in ``exempt_paths``.

    matches the canonical's whole-file exemption semantics: an entry
    in the exemption file allows every direct-nats import inside that
    file (because the integration-test files are migrated as a unit,
    not import-by-import).

    :param violations: walker output to filter
    :ptype violations: list[Violation]
    :param exempt_paths: forward-slash relative paths of exempted files
    :ptype exempt_paths: set[str]
    :param repo_root: repo root for relative-path comparison
    :ptype repo_root: Path
    :return: violations whose file is not exempted, in original order
    :rtype: list[Violation]
    """
    if not exempt_paths:
        return list(violations)
    result: list[Violation] = []
    for violation in violations:
        rel = relative_posix_path(violation.file, repo_root)
        if rel in exempt_paths:
            continue
        result.append(violation)
    return result


def _load_path_exemptions(path: Path | None) -> list[_PathExemption]:
    """parse a whole-file exemption file with rationale discipline.

    the file format mirrors the canonical's: one repo-relative path
    per line, preceded by a ``# rationale: <reason>`` line. comments
    that are not rationales are allowed and pass through silently.
    blanket rationales (the same set the common
    :func:`~threetears.enforcement.common.exemptions.parse_exemptions_with_rationale`
    rejects) and rationales shorter than 30 characters fail loudly so
    the exemption list cannot become a silent test-disabler.

    :param path: exemption file path; ``None`` returns ``[]``
    :ptype path: Path | None
    :return: parsed entries in file order
    :rtype: list[_PathExemption]
    :raises ExemptionError: missing rationale, blanket rationale,
        rationale too short, or unreadable file content
    :raises FileNotFoundError: ``path`` is set but does not exist
    """
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"exemption file not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ExemptionError(f"{path}: not valid utf-8: {exc}") from exc
    except OSError as exc:
        raise ExemptionError(f"cannot read {path}: {exc}") from exc

    entries: list[_PathExemption] = []
    pending: str | None = None
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            stripped = line.lstrip("#").strip()
            if stripped.lower().startswith("rationale:"):
                rationale = stripped[len("rationale:"):].strip()
                _validate_rationale(rationale, path, lineno)
                pending = rationale
            continue
        if pending is None:
            raise ExemptionError(
                f"{path}:{lineno}: entry {line!r} has no preceding "
                f"'# rationale: ...' line"
            )
        entries.append(_PathExemption(path=line, rationale=pending))
        pending = None
    return entries


def _validate_rationale(rationale: str, path: Path, lineno: int) -> None:
    """raise :class:`ExemptionError` for empty / blanket / too-short rationales.

    mirrors the discipline of
    :func:`~threetears.enforcement.common.exemptions.parse_exemptions_with_rationale`:
    a rationale must be non-empty, at least 30 characters, and must
    not match any blanket phrase in
    :data:`_BLANKET_RATIONALE_PHRASES`.

    :param rationale: text of the rationale (already prefix-stripped)
    :ptype rationale: str
    :param path: exemption file path (for error context)
    :ptype path: Path
    :param lineno: line number in the exemption file (for error context)
    :ptype lineno: int
    :raises ExemptionError: rationale fails any of the contract checks
    """
    if not rationale:
        raise ExemptionError(
            f"{path}:{lineno}: '# rationale:' must be followed by a "
            f"non-empty reason"
        )
    if len(rationale) < _MIN_RATIONALE_LENGTH:
        raise ExemptionError(
            f"{path}:{lineno}: rationale must be at least "
            f"{_MIN_RATIONALE_LENGTH} characters; got {len(rationale)}: "
            f"{rationale!r}"
        )
    lower = rationale.lower()
    if lower in _BLANKET_RATIONALE_PHRASES:
        raise ExemptionError(
            f"{path}:{lineno}: rationale {rationale!r} matches blanket "
            f"phrase; rationales must be specific"
        )


def _to_common_exemptions(
    entries: Iterable[_PathExemption],
) -> list[Exemption]:
    """adapt parsed path-exemptions to the shared :class:`Exemption` shape.

    the report renderer prints the exemption count, so we project each
    whole-file entry to a common :class:`Exemption` with ``line=0``
    (file-wide) and a synthetic ``"*"`` symbol — the report only uses
    the count, not the field details, but exposing the entries via
    the common type keeps the report renderer's contract uniform.

    :param entries: parsed whole-file exemptions
    :ptype entries: Iterable[_PathExemption]
    :return: adapted entries
    :rtype: list[Exemption]
    """
    return [
        Exemption(file=e.path, line=0, symbol="*", rationale=e.rationale)
        for e in entries
    ]
