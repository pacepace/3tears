"""pytest-friendly orchestration for codebase-conventions enforcement.

a single :func:`run_codebase_conventions_enforcement` entry point lets
each consumer's thin shell invoke any one of the four walkers or all
of them. the runner is the policy point: it resolves src roots,
runs the selected walker(s), applies the rationale-required exemption
file, emits the standardised report, and either raises
:func:`pytest.fail` or returns silently according to the configured
mode.

accepted ``walker`` values:

- ``"print"``: only :func:`find_print_calls`.
- ``"stdlib_getlogger"``: only :func:`find_stdlib_getlogger_calls`.
- ``"future_annotations"``: only :func:`find_missing_future_annotations`.
- ``"return_type"``: only :func:`find_missing_return_types`.
- ``"all"`` (default): run every walker; aggregated violations share
  the same report.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from threetears.enforcement.common import (
    Exemption,
    MODE_REPORT,
    MODE_STRICT,
    Violation,
    apply_exemptions,
    discover_src_roots,
    emit_report,
    parse_exemptions_with_rationale,
    resolve_mode,
)

from threetears.enforcement.codebase_conventions.config import (
    CodebaseConventionsConfig,
)
from threetears.enforcement.codebase_conventions.walkers import (
    find_missing_future_annotations,
    find_missing_return_types,
    find_print_calls,
    find_stdlib_getlogger_calls,
)

__all__ = ["run_codebase_conventions_enforcement"]


_VALID_WALKERS: frozenset[str] = frozenset({
    "print",
    "stdlib_getlogger",
    "future_annotations",
    "return_type",
    "all",
})


def run_codebase_conventions_enforcement(
    config: CodebaseConventionsConfig, walker: str = "all",
) -> None:
    """run the configured walker(s), apply exemptions, fail if strict.

    accepted ``walker`` values: ``"print"``, ``"stdlib_getlogger"``,
    ``"future_annotations"``, ``"return_type"``, ``"all"``. raises
    :class:`ValueError` for any other value so a typo doesn't silently
    no-op.

    src roots come from :attr:`CodebaseConventionsConfig.src_roots`
    when set, else from :func:`discover_src_roots
    <threetears.enforcement.common.pyproject_discovery.discover_src_roots>`
    (which walks path-deps so cross-repo source is visible to the
    AST scanner).

    file-level exemptions live in the four per-walker dicts on the
    config and are applied inside each walker. additionally, when
    :attr:`CodebaseConventionsConfig.exemptions_path` is set and
    exists, that file is parsed via
    :func:`~threetears.enforcement.common.exemptions.parse_exemptions_with_rationale`
    and the resulting entries are applied via
    :func:`~threetears.enforcement.common.exemptions.apply_exemptions`
    — kept for symmetry with sibling domains.

    in :data:`~threetears.enforcement.common.modes.MODE_REPORT` the
    runner returns normally regardless of violations; in
    :data:`~threetears.enforcement.common.modes.MODE_STRICT` (the
    default) it calls :func:`pytest.fail` with the rendered report.

    :param config: per-repo enforcement config
    :ptype config: CodebaseConventionsConfig
    :param walker: which walker(s) to invoke; one of ``"print"``,
        ``"stdlib_getlogger"``, ``"future_annotations"``,
        ``"return_type"``, ``"all"``
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
    violations = _run_walkers(config, walker, src_roots)
    exemptions = _load_exemptions(config.exemptions_path)
    filtered = apply_exemptions(violations, exemptions, config.repo_root)

    mode = resolve_mode(config.mode_env_var, default=MODE_STRICT)

    report = emit_report(
        filtered,
        src_roots,
        exemptions,
        mode,
        config.repo_root,
        domain=f"codebase_conventions.{walker}",
    )
    print(report, file=sys.stderr)

    if mode == MODE_REPORT:
        return
    if filtered:
        pytest.fail(
            f"codebase-conventions enforcement found {len(filtered)} "
            f"violation(s):\n{report}"
        )


def _resolve_src_roots(
    config: CodebaseConventionsConfig,
) -> tuple[Path, ...]:
    """pick the src roots to scan.

    explicit ``config.src_roots`` wins; otherwise we walk the path-dep
    graph so the walker sees siblings declared as workspace members
    or path-deps.

    :param config: per-repo enforcement config
    :ptype config: CodebaseConventionsConfig
    :return: src roots to scan
    :rtype: tuple[Path, ...]
    """
    if config.src_roots is not None:
        return config.src_roots
    return discover_src_roots(config.repo_root)


def _run_walkers(
    config: CodebaseConventionsConfig,
    walker: str,
    src_roots: tuple[Path, ...],
) -> list[Violation]:
    """invoke the requested walker(s) and return aggregated violations.

    :param config: per-repo enforcement config
    :ptype config: CodebaseConventionsConfig
    :param walker: walker selector
    :ptype walker: str
    :param src_roots: src roots resolved by :func:`_resolve_src_roots`
    :ptype src_roots: tuple[Path, ...]
    :return: combined violation list, in walker order
    :rtype: list[Violation]
    """
    violations: list[Violation] = []
    if walker in {"print", "all"}:
        violations.extend(
            find_print_calls(
                src_roots,
                config.repo_root,
                config.print_exempt_files,
            )
        )
    if walker in {"stdlib_getlogger", "all"}:
        violations.extend(
            find_stdlib_getlogger_calls(
                src_roots,
                config.repo_root,
                config.getlogger_exempt_files,
                config.getlogger_marker,
            )
        )
    if walker in {"future_annotations", "all"}:
        violations.extend(
            find_missing_future_annotations(
                src_roots,
                config.repo_root,
                config.future_annotations_exempt_files,
                config.skip_basenames,
            )
        )
    if walker in {"return_type", "all"}:
        violations.extend(
            find_missing_return_types(
                src_roots,
                config.repo_root,
                config.return_type_exempt_files,
                config.skip_basenames,
            )
        )
    return violations


def _load_exemptions(path: Path | None) -> list[Exemption]:
    """load exemptions from ``path``, or return ``[]`` when ``path`` is None.

    :param path: exemption file path, or ``None`` to skip loading
    :ptype path: Path | None
    :return: parsed exemption entries (empty when ``path`` is None)
    :rtype: list[Exemption]
    :raises FileNotFoundError: ``path`` is set but does not exist
    """
    if path is None:
        return []
    return parse_exemptions_with_rationale(path)
