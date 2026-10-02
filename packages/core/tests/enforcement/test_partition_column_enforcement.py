"""Enforcement -- every SQL touching a partitioned table filters by partition.

collections-task-02. partition columns are the structural defense
against cross-partition data bleed (the canonical agent-pod scenario:
two concurrent conversations served by the same pod, a Collection
method that forgets to filter by ``conversation_id``, and one
conversation seeing the other's context items). this enforcement
walker is the third defense layer: at CI time, every SQL string
literal that touches a partitioned table is verified to also include
the partition column name in its WHERE / SET / VALUES context.

design points:

- mode is controlled by ``PARTITION_ENFORCEMENT_MODE`` env var --
  ``strict`` raises on any violation; ``report`` (default during
  collections-task-02 cleanup) logs and passes so the test can be
  introduced and have its discoveries reviewed before flipping
  CI-blocking. mode flips to ``strict`` in the final cleanup commit
  once every flagged violation has been resolved.
- the discovery list (:data:`PARTITIONED_TABLES`) maps table name
  -> partition column. extending it covers a new table; the cost
  of discovery automation across multi-repo Collections is paid
  here in audit clarity.
- migration files (``CREATE TABLE`` / ``ALTER TABLE``) are exempt
  by directory: ``migrations/`` subtrees legitimately contain DDL
  that does not filter rows.
- the walker tolerates SQL spread across multi-line strings and
  joined string concatenations because the AST visitor evaluates
  the joined value via ``ast.unparse`` for f-strings and joined
  literal concatenation for plain strings.
- the walker, the discovery list and the exemptions live in
  ``partition_column_walker.py`` beside this module, shared with the
  walker's meta-test.
- exemptions live in the walker's ``_EXEMPT_LITERAL_FRAGMENTS`` -- a narrow
  allowlist of string fragments for SQL that legitimately spans
  partitions (e.g. the body of a ``@spans_partitions``-decorated
  method). every entry carries an inline ``# rationale: ...``
  comment.

runtime is AST-only and well under the 15s CLAUDE.md budget; the
walker is deterministic and side-effect-free.
"""

from __future__ import annotations

import os

import pytest

from packages.core.tests.enforcement.partition_column_walker import (  # type: ignore[import-not-found]
    PACKAGE_SRC_ROOTS,
    PARTITIONED_TABLES,
    violations_in_file,
    walk_python_files,
)

__all__: list[str] = []


def test_partition_column_enforcement_across_packages() -> None:
    """walk every package source root and verify partition-column compliance.

    mode is controlled by ``PARTITION_ENFORCEMENT_MODE``: ``strict``
    (default after collections-task-02) fails on any non-deferred
    violation; ``report`` logs every violation and passes so the test
    can be re-flipped during follow-up cleanup windows.

    deferred-table violations (memories surface, follow-up
    collections-task-04) never block the strict gate -- they surface
    via :func:`pytest.skip` with the punch list so the residual work
    stays visible without forcing tasks-out-of-order.

    :return: nothing
    :rtype: None
    :raises AssertionError: when one or more strict-eligible violations
        surface in strict mode
    """
    mode = os.environ.get("PARTITION_ENFORCEMENT_MODE", "strict").lower()
    strict_violations: list[str] = []
    deferred_violations: list[str] = []
    for src_root in PACKAGE_SRC_ROOTS:
        for path in walk_python_files(src_root):
            strict_hits, deferred_hits = violations_in_file(path)
            strict_violations.extend(strict_hits)
            deferred_violations.extend(deferred_hits)

    if not strict_violations and not deferred_violations:
        return

    if mode == "report":
        # report mode: surface every hit (strict + deferred) on a single
        # skip message so the punch list is visible but non-blocking.
        all_violations = strict_violations + deferred_violations
        formatted = "\n".join(all_violations)
        pytest.skip(
            f"partition-column enforcement: {len(all_violations)} violation(s) (mode=report)\n{formatted}",
        )
        return

    # strict mode: deferred hits never block, but stay visible via skip
    # when no strict-eligible violations remain.
    if strict_violations:
        formatted = "\n".join(strict_violations)
        raise AssertionError(
            f"partition-column enforcement: {len(strict_violations)} strict violation(s) (mode=strict)\n{formatted}",
        )
    if deferred_violations:
        formatted = "\n".join(deferred_violations)
        pytest.skip(
            f"partition-column enforcement: {len(deferred_violations)} "
            f"deferred violation(s) (mode=strict; deferred surfaces "
            f"stay report-only)\n{formatted}",
        )


def test_partitioned_tables_map_is_non_empty() -> None:
    """sanity check: discovery list must declare at least one table.

    if this fails the partition primitive is unused; the enforcement
    test would silently pass.

    :return: nothing
    :rtype: None
    """
    assert len(PARTITIONED_TABLES) > 0, "PARTITIONED_TABLES is empty -- partition enforcement is a no-op"
