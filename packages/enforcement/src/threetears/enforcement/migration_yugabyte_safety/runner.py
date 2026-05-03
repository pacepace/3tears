"""pytest-friendly orchestration for migration-yugabyte-safety enforcement.

a single :func:`run_migration_enforcement` entry point lets each
consumer's thin shell invoke the walker with one line. the runner is
the policy point: it resolves the existing migration directories,
loads the exemption file (delegating to core's
:func:`load_exemptions` so the on-disk format is identical to what
the canonical hub harness already accepts), invokes the walker via
the adapter in :mod:`.walkers`, emits the standardised report, and
either calls :func:`pytest.fail` or returns silently according to
the configured mode.

deviation from sibling domains' exemption handling: this domain's
exemption file format is ``<file_path>:<rule_name> # rationale: <reason>``
per line — the format that
:func:`threetears.core.data.migrations.enforcement.load_exemptions`
expects, NOT the ``file:line:symbol`` triple that
:func:`~threetears.enforcement.common.exemptions.parse_exemptions_with_rationale`
parses. preserving the existing files unchanged across the bot-trio
repos is a hard requirement, so the runner uses core's parser
directly. core's parser surfaces "lines lacking a rationale" instead
of raising; the runner converts those to an
:class:`ExemptionError` so the contract matches sibling domains.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from threetears.core.data.migrations.enforcement import load_exemptions
from threetears.enforcement.common import (
    Exemption,
    ExemptionError,
    MODE_REPORT,
    emit_report,
    resolve_mode,
)
from threetears.enforcement.migration_yugabyte_safety.config import (
    MigrationYugabyteConfig,
)
from threetears.enforcement.migration_yugabyte_safety.walkers import (
    find_migration_violations,
)

__all__ = ["run_migration_enforcement"]


_VALID_WALKERS: frozenset[str] = frozenset({"all"})


def run_migration_enforcement(
    config: MigrationYugabyteConfig,
    walker: str = "all",
) -> None:
    """run the walker, emit report, fail if strict.

    accepted ``walker`` values:

    - ``"all"`` (default): walks every existing migration directory.

    only one walker exists for this domain (a thin adapter onto the
    core walker), but the parameter is preserved for shape-parity
    with sibling domains' runners. an unknown value raises
    :class:`ValueError`.

    migration directories that do not exist on disk are silently
    skipped before invoking the walker — preserving the canonical
    ``_existing_dirs()`` filter so a repo can declare its future
    migrations path in :class:`MigrationYugabyteConfig.migration_dirs`
    without a probe in shell code.

    exemptions live in :attr:`MigrationYugabyteConfig.exemptions_path`.
    the file format is ``<file_path>:<rule_name> # rationale: <reason>``
    per line (the canonical hub format). missing rationales raise
    :class:`ExemptionError` so the exemption list cannot become a
    silent test-disabler.

    in :data:`~threetears.enforcement.common.modes.MODE_REPORT` (the
    default for this domain — see :class:`MigrationYugabyteConfig`)
    the runner returns normally regardless of violations; in
    :data:`~threetears.enforcement.common.modes.MODE_STRICT` it calls
    :func:`pytest.fail` with the rendered report.

    :param config: per-repo enforcement config
    :ptype config: MigrationYugabyteConfig
    :param walker: which walker to invoke; the only valid value is
        ``"all"`` for this domain.
    :ptype walker: str
    :raises ValueError: ``walker`` is not in the accepted set
    :raises ExemptionError: the exemption file violates the
        rationale-required contract or cannot be read
    :raises pytest.fail.Exception: in strict mode with violations
    """
    if walker not in _VALID_WALKERS:
        raise ValueError(f"walker must be one of {sorted(_VALID_WALKERS)}, got {walker!r}")

    existing_dirs = tuple(d for d in config.migration_dirs if d.exists())
    exemptions, exemption_entries = _load_exemptions(config.exemptions_path)

    violations = find_migration_violations(
        existing_dirs,
        config.repo_root,
        exemptions,
    )

    mode = resolve_mode(config.mode_env_var, default=config.default_mode)

    report = emit_report(
        violations,
        existing_dirs,
        exemption_entries,
        mode,
        config.repo_root,
        domain="migration_yugabyte_safety",
    )
    print(report, file=sys.stderr)

    if mode == MODE_REPORT:
        return
    if violations:
        pytest.fail(f"migration-yugabyte-safety enforcement found {len(violations)} violation(s):\n{report}")


def _load_exemptions(
    path: Path | None,
) -> tuple[frozenset[tuple[str, str]], list[Exemption]]:
    """load core's exemption file, surfacing missing rationales as errors.

    delegates parsing to
    :func:`threetears.core.data.migrations.enforcement.load_exemptions`
    so the on-disk format is identical to the canonical hub harness.
    core's parser returns ``(exemptions, missing)`` where ``missing``
    is the list of raw lines lacking a non-blank ``# rationale: ...``
    suffix; the canonical's meta-test asserted ``missing == []`` and
    we convert that to an :class:`ExemptionError` so the contract
    matches sibling domains (which raise from inside the parser).

    the returned :class:`Exemption` entries are synthetic: core's
    parser only retains the ``(path, rule)`` pair, not the rationale
    text. for the report's exemption count we project each pair into
    an :class:`Exemption` with ``line=0`` and a placeholder rationale.
    the report renderer only uses ``len(exemptions)``, not the
    individual rationale text, so this projection is information-
    preserving for the rendered output.

    :param path: exemption file path; ``None`` returns empty results.
    :ptype path: Path | None
    :return: ``(exemption_pairs, common_exemption_records)``
    :rtype: tuple[frozenset[tuple[str, str]], list[Exemption]]
    :raises ExemptionError: missing rationale on at least one entry,
        or unreadable file content surfaced as a parser error.
    :raises FileNotFoundError: ``path`` is set but does not exist.
    """
    if path is None:
        return frozenset(), []
    if not path.exists():
        raise FileNotFoundError(f"exemption file not found: {path}")
    pairs, missing = load_exemptions(path)
    if missing:
        rendered = "\n".join(f"  {line!r}" for line in missing)
        raise ExemptionError(
            f"{path}: {len(missing)} exemption entry/entries lack a non-blank '# rationale: ...' suffix:\n{rendered}"
        )
    entries = [
        Exemption(
            file=file_path,
            line=0,
            symbol=rule_name,
            rationale="(rationale text not retained by core parser)",
        )
        for file_path, rule_name in sorted(pairs)
    ]
    return pairs, entries
