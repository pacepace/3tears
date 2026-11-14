"""the partition-column rule: every read or write of a partitioned table pins its partition column.

A statement written as SQL text touching a partitioned table must carry the table's partition column (or a column a
declared foreign key ties to it), and so must a keyed helper call (``threetears.core.keyed``: ``read_by``,
``keyset_page``, ``update_keyed``, ...) naming the table, in its ``by=`` (or a whole unique key of the table there: at
most one row, so one partition; :func:`unique_keys`). A read across partitions by design says so: an
exempt SQL fragment (:data:`EXEMPT_LITERAL_FRAGMENTS`, or a repo's own), or for a helper call a
``__SPANS_PARTITIONS__`` comment above its statement with the reason.

One walker for every repo: a repo's gate builds a :class:`PartitionRule` (its partitioned tables, the columns tied to
them, its exempt fragments and deferred tables, any literal check of its own) and runs :func:`violations_in_file` over
its sources (:func:`walk_python_files`). 3tears' own tables are :data:`PARTITIONED_TABLES`.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from threetears.enforcement.no_whole_reads.schema import SchemaCatalog

__all__ = [
    "EXEMPT_LITERAL_FRAGMENTS",
    "PARTITIONED_TABLES",
    "SPANS_PARTITIONS_MARKER",
    "PartitionRule",
    "tied_columns",
    "unique_keys",
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

# narrow exemption list. each entry is a SQL fragment that legitimately
# spans partitions. callers add an entry here only after the
# cross-partition rationale is documented in the surrounding code.
EXEMPT_LITERAL_FRAGMENTS: tuple[tuple[str, str], ...] = (
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
    # via _OWNER_SCOPE / _OWNER_SCOPE_MC; the walker cannot statically prove
    # the resulting SQL contains the partition column, so the test
    # treats the unparametrized template as exempt and trusts the
    # authoring discipline (every consumer of _OWNER_SCOPE
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


def tied_columns(catalog: SchemaCatalog, partitioned: Mapping[str, str] | None = None) -> dict[str, frozenset[str]]:
    """each partitioned table's columns whose bound value fixes its partition column, through a declared foreign key.

    A key whose referencing columns hold both ``c`` and the partition column, onto a table where ``c``'s referenced
    column alone is unique, ties ``c`` to the partition: every row with one ``c`` references the one parent row, so it
    holds that row's partition value. A read bound by ``c`` reads one partition. The keys are read from the catalog
    the no_whole_reads gate builds from the migrations (the same one an index's ``fk:`` purpose is judged against).

    :param catalog: the schema catalog
    :ptype catalog: SchemaCatalog
    :param partitioned: table onto its partition column; None takes :data:`PARTITIONED_TABLES`
    :ptype partitioned: Mapping[str, str] | None
    :return: table onto its tied columns (tables with none left out)
    :rtype: dict[str, frozenset[str]]
    """
    tables = PARTITIONED_TABLES if partitioned is None else partitioned
    tied = {table: catalog.tied_columns(table, column) for table, column in tables.items()}
    return {table: columns for table, columns in tied.items() if columns}


def unique_keys(
    catalog: SchemaCatalog, partitioned: Mapping[str, str] | None = None
) -> dict[str, frozenset[frozenset[str]]]:
    """each partitioned table's declared unique keys (its primary key and every whole-table ``UNIQUE``): a helper call
    binding every column of one reads or writes at most one row, so one partition, whatever its partition value.

    A partial unique index is no key of the table (two rows outside its predicate may share the columns), and the
    catalog leaves it out; an expression key is never a ``by=`` column, so it never matches.

    :param catalog: the schema catalog
    :ptype catalog: SchemaCatalog
    :param partitioned: table onto its partition column; None takes :data:`PARTITIONED_TABLES`
    :ptype partitioned: Mapping[str, str] | None
    :return: table onto its unique keys (tables with none left out)
    :rtype: dict[str, frozenset[frozenset[str]]]
    """
    tables = PARTITIONED_TABLES if partitioned is None else partitioned
    found: dict[str, frozenset[frozenset[str]]] = {}
    for table in tables:
        entry = catalog.table(table)
        if entry is None:
            continue
        keys = frozenset(frozenset(key) for key in [*entry.primary_keys, *entry.unique_keys] if key)
        if keys:
            found[table] = keys
    return found


def _binds(literal: str, column: str) -> bool:
    """whether ``literal`` binds ``column`` to a value: ``column = ...`` (``= $1``, ``= ANY(...)``) or ``column IN``.

    :param literal: SQL string literal
    :ptype literal: str
    :param column: the column
    :ptype column: str
    :return: whether it is bound
    :rtype: bool
    """
    return re.search(rf"(?<![\w]){re.escape(column)}\s*(?:=(?!=)|IN\b)", literal, re.IGNORECASE) is not None


def _is_exempt(literal: str, fragments: tuple[tuple[str, str], ...]) -> bool:
    """true iff ``literal`` holds one of ``fragments``.

    :param literal: SQL string literal
    :ptype literal: str
    :param fragments: the exempt fragments, each with its rationale
    :ptype fragments: tuple[tuple[str, str], ...]
    :return: presence of any exempt fragment in the literal
    :rtype: bool
    """
    return any(fragment in literal for fragment, _ in fragments)


#: the keyed helpers (``threetears.core.keyed``) that read or write their table by ``by=``: ``(conn, table, ..., by=...)``
_BY_HELPERS: frozenset[str] = frozenset(
    {
        "aggregate_keyed",
        "count_at_most",
        "counts_by",
        "delete_keyed",
        "holds_any",
        "keyset_page",
        "read_after",
        "read_by",
        "read_row",
        "read_value",
        "update_keyed",
    }
)


#: the comment a read across partitions by design carries in the comment block above its statement (or a statement
#: enclosing it in the same function), saying why; a helper call has no literal to carry an exemption fragment
SPANS_PARTITIONS_MARKER = "__SPANS_PARTITIONS__"


def _declared_spanning(node: ast.AST, parents: Mapping[ast.AST, ast.AST], lines: list[str]) -> bool:
    """whether ``node`` sits under a statement whose comment block (the comment lines right above it) carries
    :data:`SPANS_PARTITIONS_MARKER`, looking out through its enclosing statements up to its function.

    :param node: the helper call
    :ptype node: ast.AST
    :param parents: each node's parent
    :ptype parents: Mapping[ast.AST, ast.AST]
    :param lines: the module's source lines
    :ptype lines: list[str]
    :return: whether it is declared
    :rtype: bool
    """
    current: ast.AST | None = node
    while current is not None and not isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
        if isinstance(current, ast.stmt):
            index = current.lineno - 2
            while index >= 0 and lines[index].strip().startswith("#"):
                if SPANS_PARTITIONS_MARKER in lines[index]:
                    return True
                index -= 1
        current = parents.get(current)
    return False


def _helper_calls(tree: ast.AST, source: str = "") -> list[tuple[str, frozenset[str] | None, frozenset[str], int]]:
    """each keyed helper call naming its table as a literal: ``(table, the columns its literal by= pins, the ones it
    pins to a value other than a literal None, line)``; the columns are None when ``by=`` is built elsewhere (a name, a
    call, a spread), which the walker treats as opaque, as it does an f-string placeholder. A call with no ``by=`` pins
    nothing (an empty set). A column pinned to None reads ``IS NULL``, which a unique key does not bound.

    A call under a statement declared :data:`SPANS_PARTITIONS_MARKER` (its comment block) is left out: it reads
    across partitions by design, and says why.

    :param tree: the module
    :ptype tree: ast.AST
    :param source: the module's source, for the comment blocks
    :ptype source: str
    :return: the calls
    :rtype: list[tuple[str, frozenset[str] | None, frozenset[str], int]]
    """
    found: list[tuple[str, frozenset[str] | None, frozenset[str], int]] = []
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    lines = source.splitlines()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
        if name not in _BY_HELPERS or len(node.args) < 2:
            continue
        table = node.args[1]
        if not (isinstance(table, ast.Constant) and isinstance(table.value, str)):
            continue
        if _declared_spanning(node, parents, lines):
            continue
        by = next((keyword.value for keyword in node.keywords if keyword.arg == "by"), None)
        pinned: frozenset[str] | None
        valued: frozenset[str] = frozenset()
        if by is None or (isinstance(by, ast.Constant) and by.value is None):
            pinned = frozenset()
        elif isinstance(by, ast.Dict) and all(isinstance(key, ast.Constant) for key in by.keys):
            pinned = frozenset(str(key.value) for key in by.keys if isinstance(key, ast.Constant))
            valued = frozenset(
                str(key.value)
                for key, value in zip(by.keys, by.values, strict=True)
                if isinstance(key, ast.Constant) and not (isinstance(value, ast.Constant) and value.value is None)
            )
        else:
            pinned = None
        found.append((table.value.lower(), pinned, valued, node.lineno))
    return found


@dataclass(frozen=True)
class PartitionRule:
    """one repo's partition-column rule.

    :ivar partitioned: table onto its partition column
    :ivar tied: each table's columns a foreign key ties to its partition column (:func:`tied_columns`): a statement or
        call binding one reads one partition
    :ivar exempt_fragments: SQL fragments that span partitions by design, each with its rationale
    :ivar deferred: tables whose hits are reported, not failed, each with the follow-up that clears them
    :ivar exempt_literal: a repo's own check that a ``(literal, table)`` reads one bounded row set (a by-unique-id
        lookup), or None
    :ivar unique: each table's unique keys (:func:`unique_keys`): a helper call binding every column of one reads or
        writes at most one row, in one partition
    """

    partitioned: Mapping[str, str] = field(default_factory=lambda: dict(PARTITIONED_TABLES))
    tied: Mapping[str, frozenset[str]] = field(default_factory=dict)
    exempt_fragments: tuple[tuple[str, str], ...] = EXEMPT_LITERAL_FRAGMENTS
    deferred: Mapping[str, str] = field(default_factory=dict)
    exempt_literal: Callable[[str, str], bool] | None = None
    unique: Mapping[str, frozenset[frozenset[str]]] = field(default_factory=dict)


def violations_in_file(path: Path, rule: PartitionRule | None = None) -> tuple[list[str], list[str]]:
    """return ``(strict_violations, deferred_violations)`` found in ``path``.

    strict violations cover tables NOT in the rule's ``deferred``;
    deferred violations cover tables that are still under their
    follow-up shard (e.g. memories tables under collections-task-04).
    the test surfaces deferred hits at report level even when the walker
    is otherwise in strict mode, so the surviving punch list stays
    visible without blocking the strict gate.

    :param path: source file under inspection
    :ptype path: Path
    :param rule: the repo's rule; None is 3tears' tables with no tied columns
    :ptype rule: PartitionRule | None
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
    rule = PartitionRule() if rule is None else rule
    tied = rule.tied
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
        if _is_exempt(literal, rule.exempt_fragments):
            continue
        # extract every table touched by the literal
        tables_touched: set[str] = set()
        for match in _TABLE_PATTERN.finditer(literal):
            tables_touched.add(match.group(1).lower())
        for table in tables_touched:
            partition_col = rule.partitioned.get(table)
            if partition_col is None:
                continue
            if rule.exempt_literal is not None and rule.exempt_literal(literal, table):
                continue
            # partition column must appear somewhere in the literal
            # (the walker treats f-string placeholders as opaque, so
            # ``WHERE {scope}`` interpolations pass when ``scope``
            # contains the partition column predicate built upstream
            # -- this is the dynamic-scope-clause caveat; the
            # CollectionRegistry primitive enforces the static
            # signature contract elsewhere)
            if any(_binds(literal, column) for column in tied.get(table, ())):
                # bound by a column a declared foreign key ties to the partition column: one partition's rows
                continue
            if partition_col not in literal and "__PLACEHOLDER__" not in literal:
                msg = (
                    f"{path}:{lineno}: SQL touches partitioned table "
                    f"{table!r} without filtering on partition column "
                    f"{partition_col!r}: {literal[:160]!r}"
                )
                if table in rule.deferred:
                    deferred_violations.append(
                        f"{msg} [{rule.deferred[table]}]",
                    )
                else:
                    strict_violations.append(msg)
    # a keyed helper call is a read or write of its table as much as a statement written as text is
    for table, pinned, valued, lineno in _helper_calls(tree, source):
        partition_col = rule.partitioned.get(table)
        if partition_col is None or pinned is None or ({partition_col} | tied.get(table, frozenset())) & pinned:
            continue
        if any(key <= valued for key in rule.unique.get(table, ())):
            # bound by a whole unique key: at most one row, so one partition
            continue
        msg = (
            f"{path}:{lineno}: keyed helper call on partitioned table {table!r} without its partition column "
            f"{partition_col!r} in by= (pins {sorted(pinned)})"
        )
        if table in rule.deferred:
            deferred_violations.append(f"{msg} [{rule.deferred[table]}]")
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
