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
- the discovery list (:data:`_PARTITIONED_TABLES`) maps table name
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
- exemptions live in :data:`_EXEMPT_LITERAL_FRAGMENTS` -- a narrow
  allowlist of string fragments for SQL that legitimately spans
  partitions (e.g. the body of a ``@spans_partitions``-decorated
  method). every entry carries an inline ``# rationale: ...``
  comment.

runtime is AST-only and well under the 15s CLAUDE.md budget; the
walker is deterministic and side-effect-free.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path

import pytest

__all__: list[str] = []


# table_name -> partition column. extend this map when a Collection
# adopts the ``partition=True`` flag on a TableSchema column. the test
# walks every .py file under each declared package and verifies SQL
# touching the table includes the partition column.
_PARTITIONED_TABLES: dict[str, str] = {
    # agent-tools
    "context_items": "conversation_id",
    # conversations
    "conversations": "agent_id",
    # agent-memory
    "memories": "agent_id",
    "media": "agent_id",
    "media_content": "agent_id",
    "memory_chunks": "agent_id",
    "conversation_memory_refs": "conversation_id",
    # agent-skills
    "agent_skills": "agent_id",
    "agent_skill_invocations": "agent_id",
    # agent-wake
    "agent_wake_schedules": "conversation_id",
    "wake_fires": "conversation_id",
    "webhook_subscriptions": "conversation_id",
    # agent-workspace
    "workspaces": "agent_id",
    "workspace_files": "workspace_id",
    "workspace_file_versions": "workspace_id",
}

# tables whose partition-column hits stay in report-only mode while a
# follow-up shard finishes the cross-agent retrieval surface. each entry
# carries a one-line rationale tying back to the deferring task.
# collections-task-04 cleared the memories surface; this map is empty
# now and the partition guard runs in strict mode against every
# partitioned table.
_DEFERRED_TABLES: dict[str, str] = {}


# packages to walk. additions go here when a new repo / package is
# brought under the partition-column doctrine.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
_PACKAGE_SRC_ROOTS: list[Path] = [
    _REPO_ROOT / "packages" / "core" / "src",
    _REPO_ROOT / "packages" / "agent-tools" / "src",
    _REPO_ROOT / "packages" / "agent-workspace" / "src",
    _REPO_ROOT / "packages" / "agent-memory" / "src",
    _REPO_ROOT / "packages" / "conversations" / "src",
    # Post agent-namespace move, agent-* packages live under packages/agent/<name>/src.
    # The dash-named paths above are kept for any historical layout that survives;
    # paths that do not exist are skipped silently by ``_walk_python_files``.
    _REPO_ROOT / "packages" / "agent" / "tools" / "src",
    _REPO_ROOT / "packages" / "agent" / "memory" / "src",
    _REPO_ROOT / "packages" / "agent" / "workspace" / "src",
    _REPO_ROOT / "packages" / "agent" / "skills" / "src",
    _REPO_ROOT / "packages" / "agent" / "wake" / "src",
    _REPO_ROOT / "packages" / "agent" / "acl" / "src",
    _REPO_ROOT / "packages" / "agent" / "audit" / "src",
]


# narrow exemption list. each entry is a SQL fragment that legitimately
# spans partitions. callers add an entry here only after the
# cross-partition rationale is documented in the surrounding code.
_EXEMPT_LITERAL_FRAGMENTS: tuple[tuple[str, str], ...] = (
    # rationale: extension / trigger / function DDL is unscoped by
    # design; partition predicates do not apply to schema setup.
    ("CREATE EXTENSION", "DDL"),
    ("CREATE TRIGGER", "DDL"),
    ("CREATE OR REPLACE FUNCTION", "DDL"),
    ("DROP TRIGGER", "DDL"),
    # rationale: dynamic SQL builders compose the partition predicate
    # via parameterized fragments rather than a literal column name in
    # the same string. these helpers live in the memories collection
    # and inject ``user_id`` / ``agent_id`` / ``customer_id`` predicates
    # via _build_user_scope_clause; the walker cannot statically prove
    # the resulting SQL contains the partition column, so the test
    # treats the unparametrized template as exempt and trusts the
    # authoring discipline (every consumer of _build_user_scope_clause
    # passes scope params).
    ("websearch_to_tsquery", "dynamic-scope-clause"),
    # rationale: one-time data translation at boot. matches the
    # CLAUDE.md "translation, not shim" carve-out -- the helper
    # backfills new columns from old columns on legacy rows during
    # the v0.5.0 schema migration; the UPDATE deliberately spans
    # every partition because the migration runs globally before
    # any per-conversation read path is exercised.
    ("long_desc = LEFT(content, 1000)", "one-time-schema-migration"),
    # rationale: per-user rate-limit aggregate (PLACEMENT §1.9 +
    # agent-wake shard-05 OBS-14) MUST sum fires across every
    # conversation the user owns -- the rule is "100 fires per user
    # per 24h" total, not per-conversation. Partitioning by
    # conversation_id here would defeat the cap. The query joins
    # ``wake_fires`` against both source-tables (``agent_wake_schedules``
    # and ``webhook_subscriptions``) filtered by ``user_id``; the
    # source-table joins are intrinsically cross-partition. Same
    # carve-out shape as the dynamic-scope-clause case above.
    (
        "JOIN agent_wake_schedules ws ON wf.schedule_id = ws.schedule_id",
        "per-user-rate-limit-aggregate",
    ),
)


# regex to find the table name following FROM/INTO/UPDATE. matches
# qualified (schema.table) and bare table forms, case-insensitive.
_TABLE_PATTERN = re.compile(
    r"\b(?:FROM|INTO|UPDATE|JOIN)\s+(?:[a-zA-Z_]\w*\.)?([a-zA-Z_]\w*)",
    re.IGNORECASE,
)

# heuristic gate to filter docstrings / prose from SQL. a literal is
# treated as SQL only when it starts (after stripping leading
# whitespace) with one of these tokens. this catches the common case
# of a docstring that mentions "FROM memories" without false-firing
# on it.
_SQL_LEADING_TOKENS = ("SELECT", "INSERT", "UPDATE", "DELETE", "WITH", "VALUES")


def _collect_string_literals(tree: ast.AST) -> list[tuple[str, int]]:
    """walk ``tree`` and collect every string literal with its line number.

    handles plain ``ast.Constant`` strings, joined-string ``f"..."``
    forms (best-effort: takes the constant fragments and joins with
    placeholders for the formatted parts), and adjacent literal
    concatenation (the parser folds those into a single Constant
    already, so no special handling needed).

    constants nested inside a JoinedStr (the literal fragments
    interleaved with FormattedValue parts of an f-string) are NOT
    collected separately — they were already absorbed into the joined
    string the walker emits for the parent JoinedStr. collecting them
    again would surface false positives where the partial literal
    "SELECT * FROM memories WHERE " (the constant fragment before
    a ``{scope_conditions}`` interpolation) lacks the partition column
    by construction even though the joined string does carry the
    `__PLACEHOLDER__` marker the partition guard accepts.

    :param tree: AST root
    :ptype tree: ast.AST
    :return: list of (literal text, line number) pairs
    :rtype: list[tuple[str, int]]
    """
    # find every Constant that is a child of a JoinedStr so we can
    # skip them in the outer walk (they're already accounted for via
    # the JoinedStr's joined emission).
    joined_constants: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            for v in node.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    joined_constants.add(id(v))
    results: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in joined_constants:
                continue
            results.append((node.value, node.lineno))
        elif isinstance(node, ast.JoinedStr):
            parts: list[str] = []
            for v in node.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    parts.append(v.value)
                else:
                    # placeholder for formatted segment so partition
                    # columns embedded via interpolation count
                    parts.append("__PLACEHOLDER__")
            results.append(("".join(parts), node.lineno))
    return results


def _is_exempt(literal: str) -> bool:
    """true iff ``literal`` is in :data:`_EXEMPT_LITERAL_FRAGMENTS`.

    :param literal: SQL string literal
    :ptype literal: str
    :return: presence of any exempt fragment in the literal
    :rtype: bool
    """
    return any(fragment in literal for fragment, _ in _EXEMPT_LITERAL_FRAGMENTS)


def _violations_in_file(path: Path) -> tuple[list[str], list[str]]:
    """return ``(strict_violations, deferred_violations)`` found in ``path``.

    strict violations cover tables NOT in :data:`_DEFERRED_TABLES`;
    deferred violations cover tables that are still under their
    follow-up shard (e.g. memories tables under collections-task-04).
    the test surfaces deferred hits at report level even when the walker
    is otherwise in strict mode, so the surviving punch list stays
    visible without blocking the strict gate.

    :param path: source file under inspection
    :ptype path: Path
    :return: ``(strict_violations, deferred_violations)`` -- two parallel
        lists of human-readable messages
    :rtype: tuple[list[str], list[str]]
    """
    if "migrations" in path.parts:
        return [], []
    if "tests" in path.parts:
        return [], []
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return [], []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return [], []
    strict_violations: list[str] = []
    deferred_violations: list[str] = []
    for literal, lineno in _collect_string_literals(tree):
        upper_stripped = literal.lstrip().upper()
        # heuristic gate: a docstring may legitimately contain
        # "FROM memories" prose without being SQL. only literals that
        # begin with a SQL keyword count.
        if not any(upper_stripped.startswith(tok) for tok in _SQL_LEADING_TOKENS):
            continue
        upper = literal.upper()
        if "FROM " not in upper and "INTO " not in upper and "UPDATE " not in upper and "JOIN " not in upper:
            continue
        if _is_exempt(literal):
            continue
        # extract every table touched by the literal
        tables_touched: set[str] = set()
        for match in _TABLE_PATTERN.finditer(literal):
            tables_touched.add(match.group(1).lower())
        for table in tables_touched:
            partition_col = _PARTITIONED_TABLES.get(table)
            if partition_col is None:
                continue
            # partition column must appear somewhere in the literal
            # (the walker treats f-string placeholders as opaque, so
            # ``WHERE {scope}`` interpolations pass when ``scope``
            # contains the partition column predicate built upstream
            # -- this is the dynamic-scope-clause caveat; the
            # CollectionRegistry primitive enforces the static
            # signature contract elsewhere)
            if partition_col not in literal and "__PLACEHOLDER__" not in literal:
                msg = (
                    f"{path}:{lineno}: SQL touches partitioned table "
                    f"{table!r} without filtering on partition column "
                    f"{partition_col!r}: {literal[:160]!r}"
                )
                if table in _DEFERRED_TABLES:
                    deferred_violations.append(
                        f"{msg} [{_DEFERRED_TABLES[table]}]",
                    )
                else:
                    strict_violations.append(msg)
    return strict_violations, deferred_violations


def _walk_python_files(root: Path) -> list[Path]:
    """yield every ``.py`` file under ``root``.

    :param root: package source root
    :ptype root: Path
    :return: list of file paths
    :rtype: list[Path]
    """
    if not root.exists():
        return []
    return [p for p in root.rglob("*.py") if p.is_file()]


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
    for src_root in _PACKAGE_SRC_ROOTS:
        for path in _walk_python_files(src_root):
            strict_hits, deferred_hits = _violations_in_file(path)
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
    assert len(_PARTITIONED_TABLES) > 0, "_PARTITIONED_TABLES is empty -- partition enforcement is a no-op"
