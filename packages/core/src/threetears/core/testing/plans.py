"""a plan check for SQL that must scale with the rows it answers, not the table it reads.

A statement can be right and still read a whole table each time it runs: a keyset page ordered by one
key and found through another index re-reads and re-sorts every row the filters leave, page after page;
a table's notes read with no index leading on the table scan every note. Neither shows on an empty test
table. This module runs a statement's real plan, ``EXPLAIN ANALYZE`` on seeded data, and names what does
not scale:

- **a sequential scan of a large table on the inner side of a nested loop**: the scan repeats once per
  outer row. Large is judged by the rows the scan actually read, not by the table's name, so a test on
  an empty table says nothing either way;
- **rows scanned beyond rows returned**: a read that touches about its own rows scans no more than a
  small factor of what it answers.

Rows scanned are counted per relation scan (a sequential, index, index-only or bitmap heap scan): on
YugabyteDB from ``EXPLAIN (ANALYZE, DIST)``'s storage counts (the larger of the table's and the index's
rows scanned, which an index scan reads both of), on Postgres from the rows the scan answered plus those
its filters and rechecks removed. Each per loop, times the loops.

Every ``EXPLAIN ANALYZE`` runs inside a transaction that is rolled back, so a checked write changes
nothing; the statement itself then runs as the caller asked.

Usage, around the code under test (anything that reads through ``fetch`` / ``fetchrow`` / ``fetchval``):

.. code-block:: python

    checked = PlanRecorder(conn)
    rows = await read_l3_rows(checked, "results", columns, key, where={"state": "TX"})
    assert not plan_problems(checked.plans, scan_ratio=1.1, together=True)
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from threetears.core.keyed._filters import binds_native_arrays

__all__ = [
    "PlanRecorder",
    "WholeScanGuard",
    "explain_write",
    "on_yugabyte",
    "plan_problems",
    "plan_uses_index",
]

#: the rows a sequential scan reads, per loop, from which it counts as a scan of a large table
DEFAULT_LARGE_ROWS = 1_000

#: the YugabyteDB storage counts of a relation scan; the larger is what it read
_YB_SCANNED = ("Storage Table Rows Scanned", "Storage Index Rows Scanned")

#: what a Postgres scan read beyond what it answered
_PG_REMOVED = ("Rows Removed by Filter", "Rows Removed by Index Recheck")


@dataclass(frozen=True)
class StatementPlan:
    """one statement's analysed plan, and what it read against what it answered.

    :ivar sql: the statement
    :ivar rows_returned: the rows the statement answered
    :ivar rows_scanned: the rows its relation scans read (:func:`rows_scanned`)
    :ivar looped: each large table it scans sequentially on the inner side of a nested loop, with the rows
        each loop read
    :ivar seconds: the execution time the plan reports
    :ivar plan: the plan's root node, as ``EXPLAIN (FORMAT JSON)`` gives it
    """

    sql: str
    rows_returned: int
    rows_scanned: int
    looped: tuple[str, ...]
    seconds: float
    plan: Mapping[str, Any] = field(repr=False)


def _read_a_loop(node: Mapping[str, Any]) -> int:
    """the rows one relation scan read in one loop (both engines report a loop's average), 0 for any other node.

    A write's ``ModifyTable`` node names the relation it writes, and answers its RETURNING rows, but reads nothing:
    the scan under it found the rows.
    """
    if "Relation Name" not in node or node.get("Node Type") == "ModifyTable":
        return 0
    if any(name in node for name in _YB_SCANNED):
        return max(int(node.get(name, 0)) for name in _YB_SCANNED)
    return int(node.get("Actual Rows", 0)) + sum(int(node.get(name, 0)) for name in _PG_REMOVED)


def rows_scanned(plan: Mapping[str, Any]) -> int:
    """the rows every relation scan under ``plan`` read, over all its loops.

    :param plan: a plan node, as ``EXPLAIN (ANALYZE, FORMAT JSON)`` gives it
    :ptype plan: Mapping[str, Any]
    :return: the rows read
    :rtype: int
    """
    total = int(plan.get("Actual Loops", 1) or 1) * _read_a_loop(plan)
    for child in plan.get("Plans", ()):
        total += rows_scanned(child)
    return total


def looped_seq_scans(
    plan: Mapping[str, Any], *, large_rows: int = DEFAULT_LARGE_ROWS, in_loop: bool = False
) -> Iterator[str]:
    """each sequential scan under ``plan`` on the inner side of a nested loop that read ``large_rows`` or more a loop.

    :param plan: a plan node, analysed
    :ptype plan: Mapping[str, Any]
    :param large_rows: the rows a loop of the scan read from which it counts
    :ptype large_rows: int
    :param in_loop: whether ``plan`` itself runs once per outer row of a nested loop above it
    :ptype in_loop: bool
    :return: ``"<relation>: <rows> rows x <loops> loops"`` for each
    :rtype: Iterator[str]
    """
    each = _read_a_loop(plan)
    if in_loop and plan.get("Node Type") == "Seq Scan" and each >= large_rows:
        yield f"{plan.get('Relation Name')}: {each} rows x {int(plan.get('Actual Loops', 1) or 1)} loops"
    looping = plan.get("Node Type") == "Nested Loop"
    for child in plan.get("Plans", ()):
        inner = looping and child.get("Parent Relationship") == "Inner"
        yield from looped_seq_scans(child, large_rows=large_rows, in_loop=in_loop or inner)


async def on_yugabyte(conn: Any) -> bool:
    """whether ``conn`` is a YugabyteDB connection (its ``version()`` says ``-YB-``).

    The plan check's verdicts are YugabyteDB's, the database the platform runs on: a suite that also runs on
    Postgres asserts the ratio and the budget only where this is true, and correctness everywhere.

    :param conn: an asyncpg connection
    :ptype conn: Any
    :return: whether it is YugabyteDB
    :rtype: bool
    """
    version = await conn.fetchval("SELECT version()")
    return "-YB-" in str(version)


async def explain_statement(
    conn: Any, sql: str, *args: Any, large_rows: int = DEFAULT_LARGE_ROWS, yugabyte: bool | None = None
) -> StatementPlan:
    """``EXPLAIN ANALYZE`` one statement inside a transaction that is rolled back, and read its plan.

    :param conn: an asyncpg connection (anything with ``transaction()`` and ``fetchval``)
    :ptype conn: Any
    :param sql: the statement
    :ptype sql: str
    :param args: its parameters
    :ptype args: Any
    :param large_rows: the rows from which a looped sequential scan counts (:func:`looped_seq_scans`)
    :ptype large_rows: int
    :param yugabyte: whether ``conn`` is YugabyteDB (``EXPLAIN`` asks for its storage counts); asked of
        the server when None
    :ptype yugabyte: bool | None
    :return: the analysed plan
    :rtype: StatementPlan
    """
    is_yugabyte = await on_yugabyte(conn) if yugabyte is None else yugabyte
    if _writes(sql):
        raise ValueError(f"a write is never EXPLAIN ANALYZE-d (it would run); use explain_write: {sql[:120]}")
    options = "ANALYZE, DIST, FORMAT JSON" if is_yugabyte else "ANALYZE, FORMAT JSON"
    transaction = conn.transaction()
    await transaction.start()
    try:
        explained = await conn.fetchval(f"EXPLAIN ({options}) {sql}", *args)
    finally:
        await transaction.rollback()
    document = json.loads(explained) if isinstance(explained, str) else explained
    root = document[0]
    plan = root["Plan"]
    return StatementPlan(
        sql=sql,
        rows_returned=int(plan.get("Actual Rows", 0)) * int(plan.get("Actual Loops", 1) or 1),
        rows_scanned=rows_scanned(plan),
        looped=tuple(looped_seq_scans(plan, large_rows=large_rows)),
        seconds=float(root.get("Execution Time", 0.0)) / 1000,
        plan=plan,
    )


def _writes(sql: str) -> bool:
    """whether ``sql`` writes: its first word, or a data-modifying ``WITH``'s, is not a read."""
    words = sql.upper().replace("(", " ").split()
    return bool(words) and (
        words[0] in _WRITE_WORDS or (words[0] == "WITH" and any(word in _WRITE_WORDS for word in words))
    )


#: the first words of a statement that writes
_WRITE_WORDS = frozenset({"INSERT", "UPDATE", "DELETE", "MERGE", "TRUNCATE", "COPY", "CREATE", "DROP", "ALTER"})


async def explain_write(conn: Any, sql: str, *args: Any, large_rows: int = DEFAULT_LARGE_ROWS) -> StatementPlan:
    """the planner's plan of a write, never run: ``EXPLAIN`` without ``ANALYZE``, judged on its estimates.

    Only a looped sequential scan can be judged from estimates (its rows estimated a loop against
    ``large_rows``); the rows fields are the planner's, so a ratio over them means little.

    :param conn: an asyncpg connection
    :ptype conn: Any
    :param sql: the write
    :ptype sql: str
    :param args: its parameters
    :ptype args: Any
    :param large_rows: the estimated rows from which a looped sequential scan counts
    :ptype large_rows: int
    :return: the plan, from estimates
    :rtype: StatementPlan
    """
    explained = await conn.fetchval(f"EXPLAIN (FORMAT JSON) {sql}", *args)
    document = json.loads(explained) if isinstance(explained, str) else explained
    plan = _estimated(document[0]["Plan"])
    return StatementPlan(
        sql=sql,
        rows_returned=int(plan.get("Actual Rows", 0)),
        rows_scanned=rows_scanned(plan),
        looped=tuple(looped_seq_scans(plan, large_rows=large_rows)),
        seconds=0.0,
        plan=plan,
    )


def _estimated(node: Mapping[str, Any]) -> dict[str, Any]:
    """``node`` with the planner's estimates standing in for the actual counts it has not got."""
    copied = dict(node)
    copied["Actual Rows"] = int(node.get("Plan Rows", 0))
    copied["Actual Loops"] = 1
    copied["Plans"] = [_estimated(child) for child in node.get("Plans", ())]
    return copied


class PlanRecorder:
    """an asyncpg connection whose every read is ``EXPLAIN ANALYZE``-d (rolled back) before it runs.

    Hand it to the code under test in place of the connection; :attr:`plans` then holds each statement's
    plan, in order, for :func:`plan_problems`. ``fetch``, ``fetchrow`` and ``fetchval`` are checked; every
    other attribute is the connection's own.

    :param conn: the connection
    :ptype conn: asyncpg.Connection
    :param large_rows: the rows from which a looped sequential scan counts
    :ptype large_rows: int
    """

    def __init__(self, conn: Any, *, large_rows: int = DEFAULT_LARGE_ROWS) -> None:
        self._conn = conn
        self._large_rows = large_rows
        self._yugabyte: bool | None = None
        self.plans: list[StatementPlan] = []
        #: wall time the statements themselves took, without their EXPLAINs
        self.seconds = 0.0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)

    @property
    def binds_native_arrays(self) -> bool:
        """whether the connection it records binds typed arrays natively: a keyed helper then sends it the statement
        it would send the connection itself.

        :return: whether it does
        :rtype: bool
        """
        return binds_native_arrays(self._conn)

    async def _explain(self, sql: str, args: Sequence[Any]) -> None:
        if self._yugabyte is None:
            self._yugabyte = await on_yugabyte(self._conn)
        self.plans.append(
            await explain_statement(self._conn, sql, *args, large_rows=self._large_rows, yugabyte=self._yugabyte)
        )

    async def fetch(self, sql: str, *args: Any) -> Any:
        """``fetch``, after its plan is recorded."""
        await self._explain(sql, args)
        began = time.perf_counter()
        try:
            return await self._conn.fetch(sql, *args)
        finally:
            self.seconds += time.perf_counter() - began

    async def fetchrow(self, sql: str, *args: Any) -> Any:
        """``fetchrow``, after its plan is recorded."""
        await self._explain(sql, args)
        began = time.perf_counter()
        try:
            return await self._conn.fetchrow(sql, *args)
        finally:
            self.seconds += time.perf_counter() - began

    async def fetchval(self, sql: str, *args: Any) -> Any:
        """``fetchval``, after its plan is recorded."""
        await self._explain(sql, args)
        began = time.perf_counter()
        try:
            return await self._conn.fetchval(sql, *args)
        finally:
            self.seconds += time.perf_counter() - began


def plan_problems(
    plans: Sequence[StatementPlan], *, scan_ratio: float | None = 1.1, together: bool = False
) -> list[str]:
    """what in ``plans`` does not scale: each looped sequential scan of a large table, and rows scanned past
    ``scan_ratio`` times the rows returned.

    :param plans: the analysed statements
    :ptype plans: Sequence[StatementPlan]
    :param scan_ratio: the most rows scanned per row returned; None checks only the loops. A statement
        answering no row may scan one (a probe past the end)
    :ptype scan_ratio: float | None
    :param together: judge the ratio over every statement at once (one read, many pages) rather than each
        statement alone
    :ptype together: bool
    :return: one line for each problem; empty when there are none
    :rtype: list[str]
    """
    problems = [
        f"looped sequential scan of {scan} in: {' '.join(p.sql.split())[:200]}" for p in plans for scan in p.looped
    ]
    if scan_ratio is None:
        return problems
    if together:
        returned = sum(p.rows_returned for p in plans)
        scanned = sum(p.rows_scanned for p in plans)
        if scanned > scan_ratio * max(returned, 1):
            problems.append(
                f"{len(plans)} statements scanned {scanned} rows for {returned} returned "
                f"({scanned / max(returned, 1):.2f}x, over {scan_ratio}x)"
            )
        return problems
    for plan in plans:
        if plan.rows_scanned > scan_ratio * max(plan.rows_returned, 1):
            problems.append(
                f"scanned {plan.rows_scanned} rows for {plan.rows_returned} returned "
                f"({plan.rows_scanned / max(plan.rows_returned, 1):.2f}x, over {scan_ratio}x): "
                f"{' '.join(plan.sql.split())[:200]}"
            )
    return problems


#: the rows from which a table read whole fails a :class:`WholeScanGuard`
_DEFAULT_WHOLE_ROWS = 1_000

#: the share of a table's rows a block may read by key, page after page, before it counts as reading it whole
_DEFAULT_PAGED_SHARE = 0.5


#: rows past its ``Limit``'s an index walk under it may read and still be a page: the one a pager reads to learn a
#: next page follows, and a row or two the walk passes at its edge
_LIMIT_SLACK_ROWS: Final = 2

#: an index walk with no index condition, which reads the index whole unless a ``Limit`` stops it
_INDEX_WALK = "with no index condition"


def _whole_scans(plan: Mapping[str, Any], limit: int | None = None) -> Iterator[tuple[str, str]]:
    """each relation ``plan`` reads whole: a sequential scan (with or without a filter), an index scan with no
    index condition (every entry of the index, in its order), or a bitmap heap scan over a bitmap index scan with no
    index condition (every entry of a partial index, the filter applied after) -- unless, in an analysed plan, a ``Limit`` above it
    stopped the walk: it read at most the rows the ``Limit`` answered plus :data:`_LIMIT_SLACK_ROWS` (a keyset
    listing's first page). A walk whose filter made it read far past its ``Limit`` stays whole.

    :param plan: a plan node, as ``EXPLAIN (VERBOSE, FORMAT JSON)`` gives it; with ``ANALYZE``, a walk under a
        ``Limit`` is judged by the rows it read
    :ptype plan: Mapping[str, Any]
    :param limit: the rows the nearest ``Limit`` above answered (analysed), over its loops; None when none is above or
        the plan has no actuals
    :ptype limit: int | None
    :return: ``(schema-qualified relation, how)`` for each
    :rtype: Iterator[tuple[str, str]]
    """
    node = plan.get("Node Type", "")
    relation = plan.get("Relation Name")
    if relation is not None:
        qualified = f"{plan.get('Schema', 'public')}.{relation}"
        if node == "Seq Scan":
            yield qualified, "sequential scan" + (" with a filter" if "Filter" in plan else "")
        elif node in {"Index Scan", "Index Only Scan", "Index Scan Backward"} and "Index Cond" not in plan:
            read = int(plan.get("Actual Loops", 1) or 1) * _read_a_loop(plan)
            if limit is None or read > limit + _LIMIT_SLACK_ROWS:
                yield qualified, f"{node.lower()} {_INDEX_WALK}"
        elif node == "Bitmap Heap Scan" and any(_bitmap_walks(child) for child in plan.get("Plans", ())):
            # a bitmap is built whole before its first row is answered, so no Limit above it stops the walk
            yield qualified, f"bitmap index scan {_INDEX_WALK}"
    if node == "Limit":
        limit = int(plan["Actual Rows"]) * int(plan.get("Actual Loops", 1) or 1) if "Actual Rows" in plan else None
    for child in plan.get("Plans", ()):
        yield from _whole_scans(child, limit)


def _bitmap_walks(plan: Mapping[str, Any]) -> bool:
    """whether a bitmap heap scan's input walks a whole index: a ``Bitmap Index Scan`` with no index condition (every
    entry of the index; of a partial index, every row its predicate holds), alone or under a ``BitmapAnd`` /
    ``BitmapOr``.

    :param plan: a node under a ``Bitmap Heap Scan``
    :ptype plan: Mapping[str, Any]
    :return: whether it does
    :rtype: bool
    """
    if plan.get("Node Type") == "Bitmap Index Scan":
        return "Index Cond" not in plan
    return any(_bitmap_walks(child) for child in plan.get("Plans", ()))


def _under_a_limit(plan: Mapping[str, Any], limited: bool = False) -> bool:
    """whether an index walk with no index condition lies under a ``Limit``: the plan to analyse before judging it.

    :param plan: a plan node
    :ptype plan: Mapping[str, Any]
    :param limited: whether a ``Limit`` is above it
    :ptype limited: bool
    :return: whether one does
    :rtype: bool
    """
    node = plan.get("Node Type", "")
    if limited and node in {"Index Scan", "Index Only Scan", "Index Scan Backward"} and "Index Cond" not in plan:
        return True
    return any(_under_a_limit(child, limited or node == "Limit") for child in plan.get("Plans", ()))


def plan_uses_index(
    plan: StatementPlan | Mapping[str, Any] | Sequence[Any], index: str, *, partial: bool = False
) -> bool:
    """whether a statement's plan finds its rows through ``index``: by an index condition, or walking it in order
    under a ``Limit`` (a keyset page's first rows). An index walked whole with nothing to stop it is no keyed read.

    The plan test an index's ``plan_test`` names (:class:`threetears.core.collections.schema_backed.IndexDef`) calls
    this on the plan of the read its ``purpose`` names, seeded past a size where the planner would rather scan.
    Postgres and YugabyteDB give a plan (:func:`explain_statement`, or its root node); SQLite gives the rows of
    ``EXPLAIN QUERY PLAN`` (each a mapping with ``detail``, or a tuple ending in it), where the index is read when a
    row says ``USING INDEX <index>`` or ``USING COVERING INDEX <index>``.

    :param plan: the statement's plan, its root node, or SQLite's ``EXPLAIN QUERY PLAN`` rows
    :ptype plan: StatementPlan | Mapping[str, Any] | Sequence[Any]
    :param index: the index name
    :ptype index: str
    :param partial: ``index`` is a partial index (``CREATE INDEX ... WHERE ring_read``) the read states the predicate
        of: read whole, it holds only the rows the predicate keeps, so a scan of it with no condition (the planner
        drops a predicate the index implies) is its keyed read too
    :ptype partial: bool
    :return: whether the plan reads ``index`` as a keyed read
    :rtype: bool
    """
    if isinstance(plan, StatementPlan):
        return _reads_index(plan.plan, index, limited=partial)
    if isinstance(plan, Mapping):
        return _reads_index(plan, index, limited=partial)
    pattern = re.compile(rf"USING (?:COVERING )?INDEX {re.escape(index)}\b")
    details = [
        row.get("detail", "") if isinstance(row, Mapping) else row[-1] if isinstance(row, tuple) else row
        for row in plan
    ]
    return any(pattern.search(str(detail)) for detail in details)


def _reads_index(node: Mapping[str, Any], index: str, *, limited: bool) -> bool:
    """whether a plan node, or one under it, reads ``index`` by a condition or in order under a ``Limit``.

    :param node: a plan node
    :ptype node: Mapping[str, Any]
    :param index: the index name
    :ptype index: str
    :param limited: whether a ``Limit`` node is above it
    :ptype limited: bool
    :return: whether it does
    :rtype: bool
    """
    limited = limited or node.get("Node Type") == "Limit"
    if node.get("Index Name") == index and ("Index Cond" in node or limited):
        return True
    return any(_reads_index(child, index, limited=limited) for child in node.get("Plans", ()))


def _relations(plan: Mapping[str, Any]) -> Iterator[str]:
    """every schema-qualified relation ``plan`` scans.

    :param plan: a plan node
    :ptype plan: Mapping[str, Any]
    :return: the relations
    :rtype: Iterator[str]
    """
    if plan.get("Relation Name") is not None:
        yield f"{plan.get('Schema', 'public')}.{plan['Relation Name']}"
    for child in plan.get("Plans", ()):
        yield from _relations(child)


class WholeScanGuard:
    """an asyncpg connection that fails a load-path test on any statement reading a whole table.

    The runtime half of the no-whole-reads rule (:mod:`threetears.enforcement.no_whole_reads`): hand it to the code
    under test in place of the connection, run a steady-state load (or a follower's catch-up, a cache's refresh),
    and call :meth:`check`, or use it as ``async with WholeScanGuard(conn) as guarded:``. Every statement through
    ``fetch``, ``fetchrow``, ``fetchval``, ``execute`` and ``executemany`` is ``EXPLAIN (VERBOSE, FORMAT JSON)``-ed
    first (a write is planned, never done), then run as asked. It fails on:

    - a plan that scans a table of ``threshold_rows`` or more whole: a sequential scan, or an index scan with no
      index condition (the table counted once, by the guard, per relation). A read whose index walk lies under a
      ``Limit`` (a keyset listing's first page) is analysed too, in a transaction rolled back: the walk is a page
      when it read at most the rows the ``Limit`` answered plus two, whole when its filter read far past them;
    - a ``TRUNCATE``;
    - reads by key that, page after page, return ``paged_share`` or more of such a table: a whole read through a
      keyset pager, whose every page alone looks keyed.

    Seed the tables past ``threshold_rows`` and run the analyses the database would have (``ANALYZE``), so the
    planner chooses as it would in production; an empty table scans whole for free and proves nothing.

    :param conn: an asyncpg connection (or anything with its ``fetch`` / ``fetchval`` / ``execute``)
    :ptype conn: Any
    :param threshold_rows: the rows from which a table read whole fails
    :ptype threshold_rows: int
    :param paged_share: the share of such a table that keyed reads may return before they count as reading it whole
    :ptype paged_share: float
    """

    def __init__(
        self, conn: Any, *, threshold_rows: int = _DEFAULT_WHOLE_ROWS, paged_share: float = _DEFAULT_PAGED_SHARE
    ) -> None:
        self._conn = conn
        self._threshold = threshold_rows
        self._share = paged_share
        self._sizes: dict[str, int] = {}
        self._returned: dict[str, int] = {}
        self._problems: list[str] = []
        self._yugabyte: bool | None = None
        #: every statement seen, in order
        self.statements: list[str] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)

    async def __aenter__(self) -> WholeScanGuard:
        return self

    async def __aexit__(self, *exc: object) -> None:
        if exc[0] is None:
            self.check()

    @property
    def problems(self) -> list[str]:
        """every whole read seen so far, one line each.

        :return: the problems, paged reads included
        :rtype: list[str]
        """
        paged = [
            f"keyed reads returned {returned} of {relation}'s {self._sizes[relation]} rows "
            f"(over {self._share:.0%}): the table was read whole, a page at a time"
            for relation, returned in sorted(self._returned.items())
            if self._sizes.get(relation, 0) >= self._threshold and returned >= self._share * self._sizes[relation]
        ]
        return [*self._problems, *paged]

    def check(self) -> None:
        """fail when any statement read a table whole.

        :return: nothing
        :rtype: None
        :raises AssertionError: naming each whole read and its statement
        """
        found = self.problems
        if found:
            raise AssertionError(f"{len(found)} whole read(s) on a load path:\n" + "\n".join(found))

    async def _size(self, relation: str) -> int:
        if relation not in self._sizes:
            schema, _, table = relation.partition(".")
            self._sizes[relation] = int(
                await self._conn.fetchval(f'SELECT count(*) FROM "{schema}"."{table}"')  # noqa: S608 - plan's names
            )
        return self._sizes[relation]

    async def _plan(self, sql: str, args: Sequence[Any]) -> list[str]:
        """judge one statement's plan before it runs.

        :param sql: the statement
        :ptype sql: str
        :param args: its parameters
        :ptype args: Sequence[Any]
        :return: the relations it reads (for the paged count)
        :rtype: list[str]
        """
        self.statements.append(sql)
        words = sql.lstrip().split(None, 1)
        first = words[0].upper() if words else ""
        relations: list[str] = []
        if first == "TRUNCATE":
            self._problems.append(f"TRUNCATE on a load path: {' '.join(sql.split())[:200]}")
        elif first in {"SELECT", "WITH", "INSERT", "UPDATE", "DELETE", "VALUES", "TABLE"}:
            explained = await self._conn.fetchval(f"EXPLAIN (VERBOSE, FORMAT JSON) {sql}", *args)
            document = json.loads(explained) if isinstance(explained, str) else explained
            plan = document[0]["Plan"]
            judged = plan
            if _under_a_limit(plan) and not _writes(sql):
                # an index walk a Limit may stop: a read is analysed (rolled back) to see how far the walk went. A
                # write is never run to learn it, so its walk stays whole
                if self._yugabyte is None:
                    self._yugabyte = await on_yugabyte(self._conn)
                judged = (await explain_statement(self._conn, sql, *args, yugabyte=self._yugabyte)).plan
            for relation, how in _whole_scans(judged):
                size = await self._size(relation)
                if size >= self._threshold:
                    self._problems.append(
                        f"{how} of {relation} ({size} rows, from {self._threshold}): {' '.join(sql.split())[:200]}"
                    )
            relations = list(dict.fromkeys(_relations(plan)))
        return relations

    async def _count(self, relations: list[str], rows: int) -> None:
        if len(relations) == 1:
            await self._size(relations[0])
            self._returned[relations[0]] = self._returned.get(relations[0], 0) + rows

    async def fetch(self, sql: str, *args: Any) -> Any:
        """``fetch``, after its plan is judged."""
        relations = await self._plan(sql, args)
        rows = await self._conn.fetch(sql, *args)
        await self._count(relations, len(rows))
        return rows

    async def fetchrow(self, sql: str, *args: Any) -> Any:
        """``fetchrow``, after its plan is judged."""
        await self._plan(sql, args)
        return await self._conn.fetchrow(sql, *args)

    async def fetchval(self, sql: str, *args: Any) -> Any:
        """``fetchval``, after its plan is judged."""
        await self._plan(sql, args)
        return await self._conn.fetchval(sql, *args)

    async def execute(self, sql: str, *args: Any) -> Any:
        """``execute``, after its plan is judged."""
        await self._plan(sql, args)
        return await self._conn.execute(sql, *args)

    async def executemany(self, sql: str, args: Sequence[Sequence[Any]]) -> Any:
        """``executemany``, after its plan (for the first row of parameters) is judged."""
        if args:
            await self._plan(sql, args[0])
        return await self._conn.executemany(sql, args)
