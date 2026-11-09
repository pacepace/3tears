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
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "DEFAULT_LARGE_ROWS",
    "PlanRecorder",
    "StatementPlan",
    "explain_statement",
    "explain_write",
    "looped_seq_scans",
    "on_yugabyte",
    "plan_problems",
    "rows_scanned",
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
    """the rows one relation scan read in one loop (both engines report a loop's average), 0 for any other node."""
    if "Relation Name" not in node:
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
