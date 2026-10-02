"""the partition-column walker, shared by the enforcement test and its meta-test.

every SQL string literal touching a partitioned table must carry that table's partition column.
the rules and their rationale are documented on
:mod:`packages.core.tests.enforcement.test_partition_column_enforcement`; this module holds the
walker itself so both test modules import it under a public name instead of one test module
reaching into the other's private helpers.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

__all__ = [
    "PACKAGE_SRC_ROOTS",
    "PARTITIONED_TABLES",
    "violations_in_file",
    "walk_python_files",
]


# table_name -> partition column. extend this map when a Collection
# adopts the ``partition=True`` flag on a TableSchema column. the test
# walks every .py file under each declared package and verifies SQL
# touching the table includes the partition column.
PARTITIONED_TABLES: dict[str, str] = {
    # agent-tools
    "context_items": "conversation_id",
    # conversations
    "conversations": "agent_id",
    "folders": "agent_id",
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
    # scheduled-jobs (generic store)
    "scheduled_jobs": "partition_key",
    "job_fires": "partition_key",
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
PACKAGE_SRC_ROOTS: list[Path] = [
    _REPO_ROOT / "packages" / "core" / "src",
    _REPO_ROOT / "packages" / "agent-tools" / "src",
    _REPO_ROOT / "packages" / "agent-workspace" / "src",
    _REPO_ROOT / "packages" / "agent-memory" / "src",
    _REPO_ROOT / "packages" / "conversations" / "src",
    _REPO_ROOT / "packages" / "scheduled-jobs" / "src",
    # Post agent-namespace move, agent-* packages live under packages/agent/<name>/src.
    # The dash-named paths above are kept for any historical layout that survives;
    # paths that do not exist are skipped silently by ``walk_python_files``.
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
    # rationale: the active-schedule cap is per agent (agent-wake 0.57.0).
    # An agent's wakes live in several wake conversations, so the count
    # under the agent's advisory lock must span them; a per-conversation
    # count would let an agent exceed its cap one conversation at a time.
    # The fragment covers both the create count and the resume count.
    (
        "FROM agent_wake_schedules WHERE agent_id = $1 AND status = 'active' AND NOT protected",
        "per-agent-active-schedule-cap",
    ),
    # rationale: sizing the operator-facing scheduled-job listing, which
    # is cross-partition by construction -- a seeder mints one partition
    # per schedule, so a partition-scoped count would report 1 for a
    # deployment of any size. The listing it sizes
    # (ScheduledJobCollection.list_jobs) spans partitions for the same
    # reason and passes the walker only because count(*) is the one
    # projection with no column list to carry ``partition_key`` in. The
    # count MUST apply the same filters as that listing or it mis-pages
    # whoever trusts it.
    ("SELECT count(*) FROM scheduled_jobs", "admin-listing-page-size"),
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


def violations_in_file(path: Path) -> tuple[list[str], list[str]]:
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
            partition_col = PARTITIONED_TABLES.get(table)
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


def walk_python_files(root: Path) -> list[Path]:
    """yield every ``.py`` file under ``root``.

    :param root: package source root
    :ptype root: Path
    :return: list of file paths
    :rtype: list[Path]
    """
    if not root.exists():
        return []
    return [p for p in root.rglob("*.py") if p.is_file()]
