"""walker for the migration-yugabyte-safety enforcement domain.

thin adapter around
:func:`threetears.core.data.migrations.enforcement.walk_migration_directory`.
the underlying walker AST-scans every ``*.py`` module under each
configured migration directory and emits ``MigrationViolation``
records keyed by rule name (M-1 .. M-5). the adapter's job is to
project those records into the shared :class:`Violation
<threetears.enforcement.common.violations.Violation>` shape so the
common runner machinery (report formatter, mode resolver, exemption
plumbing) works identically to every other enforcement domain.

walker logic deliberately lives in 3tears core, not here: every repo
that consumes this domain already pulls 3tears as a dependency for
the migration runner itself, so co-locating the rule-evaluator with
the migration helpers keeps the truth-source single. this module is
the pytest-friendly facade.

mapping:

- core's ``MigrationViolation.rule`` (``"M-1"`` etc.) becomes the
  trailing component of the common :class:`Violation`'s ``category``
  (``"migration_yugabyte_safety.M-1"``). exemption-file matching keys
  off ``(file_path_str, rule_name)`` exactly as the canonical did, so
  exemption files round-trip unchanged.
- core's ``literal_excerpt`` is appended to the common ``reason``
  field so the rendered report includes the offending SQL excerpt
  (limited to 200 chars by core).
- core's ``lineno`` becomes ``Violation.line``.
- core's exemption application happens inside core's
  :func:`find_migration_violations` — the adapter passes the parsed
  exemptions and trusts core's filter. we deliberately do not also
  apply exemptions via the common :func:`apply_exemptions` machinery,
  because the domain's exemption format
  (``path:rule # rationale: ...``) is not the
  ``path:line:symbol`` triple the common parser expects.
"""

from __future__ import annotations

from pathlib import Path

from threetears.core.data.migrations.enforcement import (
    WalkerConfig,
    walk_migration_directory,
)
from threetears.enforcement.common import Violation

__all__ = ["find_migration_violations"]


_CATEGORY_PREFIX = "migration_yugabyte_safety"


def find_migration_violations(
    migration_dirs: tuple[Path, ...],
    repo_root: Path,
    exemptions: frozenset[tuple[str, str]],
) -> list[Violation]:
    """walk migration directories and return common-shape violations.

    delegates to
    :func:`threetears.core.data.migrations.enforcement.walk_migration_directory`
    and projects the emitted :class:`MigrationViolation
    <threetears.core.data.migrations.enforcement.MigrationViolation>`
    records into the shared :class:`Violation` shape used across all
    enforcement domains. directories that do not exist are silently
    skipped by the underlying walker (canonical's
    ``_existing_dirs()`` behaviour) — the runner is responsible for
    filtering ``migration_dirs`` if it wants different semantics.

    rule-name mapping: each core ``MigrationViolation.rule`` becomes
    ``"migration_yugabyte_safety.<rule>"`` in the emitted
    :class:`Violation.category`, so report output groups violations
    by rule (M-1, M-2, ...) within the domain.

    exemptions are applied inside the underlying walker, keyed by
    ``(repo-relative POSIX path, rule_name)``. the runner is the only
    caller that should ever pass a non-empty set; tests call this
    function directly with ``frozenset()`` to exercise raw walker
    output.

    :param migration_dirs: absolute paths to migration directories.
        non-existent paths are skipped.
    :ptype migration_dirs: tuple[Path, ...]
    :param repo_root: repo root used to compute repo-relative
        exemption-match keys; rendered into the common report at
        format time.
    :ptype repo_root: Path
    :param exemptions: ``(path_str, rule_name)`` pairs to suppress;
        ``frozenset()`` for an unfiltered scan.
    :ptype exemptions: frozenset[tuple[str, str]]
    :return: violations in source order (file-major, line-major,
        rule-tiebreak), one per non-exempted core violation.
    :rtype: list[Violation]
    """
    walker_config = WalkerConfig(
        migration_dirs=migration_dirs,
        exemptions=exemptions,
        repo_root=repo_root,
    )
    raw = walk_migration_directory(walker_config)
    result: list[Violation] = []
    for hit in raw:
        result.append(
            Violation(
                category=f"{_CATEGORY_PREFIX}.{hit.rule}",
                file=hit.path,
                line=hit.lineno,
                symbol=hit.rule,
                reason=f"{hit.message} -- {hit.literal_excerpt!r}",
            )
        )
    return result
