"""how fast this database does the bare operation a path performs, so a path's budget holds on any machine.

A wall-time budget in seconds is a bound on the machine as much as the code: generous enough for a busy
laptop, it misses a path ten times slower than it should be on a fast one. These time the bare operation a
path performs on the same row count, on the same database and connection, against a plain table: reading
N rows, copying N rows in, updating N rows by key, deleting N rows by key. A path's budget is then a
multiple of that baseline (:func:`budget_problem`), computed in the test, not a figure in its source.

Each baseline is the best of a few runs, so a stall in one run does not inflate it. The plain tables are
created in the connection's current schema under a unique name and dropped after.

Usage:

.. code-block:: python

    baseline = await plain_read_seconds(conn, "SELECT * FROM results WHERE state = $1", "TX")
    took = await best_seconds(lambda: read_l3_rows(conn, "results", columns, order, where={"state": "TX"}))
    assert not budget_problem("TX read", took, baseline)
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any

__all__ = [
    "DEFAULT_BUDGET_FACTOR",
    "best_seconds",
    "budget_problem",
    "plain_copies",
    "plain_copy_seconds",
    "plain_keyed_delete_seconds",
    "plain_keyed_update_seconds",
    "plain_read_seconds",
]

#: how many times its calibrated baseline a path may take
DEFAULT_BUDGET_FACTOR = 2.0

#: runs a baseline or a path is timed over; the best is kept
_RUNS = 3


async def best_seconds(run: Callable[[], Awaitable[Any]], *, runs: int = _RUNS) -> float:
    """the fastest of ``runs`` awaits of ``run()``, in seconds.

    :param run: makes the awaitable to time, once a run
    :ptype run: Callable[[], Awaitable[Any]]
    :param runs: how many runs
    :ptype runs: int
    :return: the best run's seconds
    :rtype: float
    """
    best = float("inf")
    for _ in range(runs):
        began = time.perf_counter()
        await run()
        best = min(best, time.perf_counter() - began)
    return best


def _plain_name() -> str:
    return f"calibration_{uuid.uuid4().hex[:12]}"


@asynccontextmanager
async def plain_copies(conn: Any, sources: Mapping[str, tuple[str, Sequence[Any]]]) -> AsyncIterator[dict[str, str]]:
    """plain tables holding only the rows each source names, for timing the bare reads a path must make.

    For a path whose least work is several reads (a ring row and two tables' rows, say): the caller times
    one statement over the copies, shaped as the path's own statement is, and that is its baseline.

    :param conn: an asyncpg connection
    :ptype conn: Any
    :param sources: a name for each, -> a ``SELECT`` naming its rows and that statement's parameters
    :ptype sources: Mapping[str, tuple[str, Sequence[Any]]]
    :return: each name's plain table, dropped on exit
    :rtype: AsyncIterator[dict[str, str]]
    """
    made: dict[str, str] = {}
    try:
        for name, (rows_sql, args) in sources.items():
            plain = _plain_name()
            # copied over rather than CREATE TABLE AS, which takes no bound parameters
            statement = await conn.prepare(rows_sql)
            records = await statement.fetch(*args)
            shape = ", ".join(f'"{a.name}" {a.type.schema}.{a.type.name}' for a in statement.get_attributes())
            await conn.execute(f"CREATE TABLE {plain} ({shape})")
            made[name] = plain
            await conn.copy_records_to_table(plain, records=[tuple(r) for r in records])
        yield made
    finally:
        for plain in made.values():
            await conn.execute(f"DROP TABLE IF EXISTS {plain}")


async def plain_read_seconds(conn: Any, rows_sql: str, *args: Any, runs: int = _RUNS) -> float:
    """the best time to read, in one statement, the rows ``rows_sql`` names, from a plain table holding only them.

    :param conn: an asyncpg connection
    :ptype conn: Any
    :param rows_sql: a ``SELECT`` naming the rows the path reads, with the columns it reads
    :ptype rows_sql: str
    :param args: its parameters
    :ptype args: Any
    :param runs: how many timed reads
    :ptype runs: int
    :return: seconds
    :rtype: float
    """
    async with plain_copies(conn, {"rows": (rows_sql, args)}) as plain:
        return await best_seconds(lambda: conn.fetch(f"SELECT * FROM {plain['rows']}"), runs=runs)  # noqa: S608


async def plain_copy_seconds(
    conn: Any, columns: Sequence[tuple[str, str]], records: Sequence[Sequence[Any]], *, runs: int = _RUNS
) -> float:
    """the best time to ``COPY`` ``records`` into a fresh plain table of ``columns`` (no key, no index).

    :param conn: an asyncpg connection
    :ptype conn: Any
    :param columns: each column's name and SQL type
    :ptype columns: Sequence[tuple[str, str]]
    :param records: the rows, each in ``columns``' order
    :ptype records: Sequence[Sequence[Any]]
    :param runs: how many timed copies, each into a table of its own
    :ptype runs: int
    :return: seconds
    :rtype: float
    """
    best = float("inf")
    shape = ", ".join(f"{name} {kind}" for name, kind in columns)
    for _ in range(runs):
        plain = _plain_name()
        await conn.execute(f"CREATE TABLE {plain} ({shape})")
        try:
            began = time.perf_counter()
            await conn.copy_records_to_table(plain, records=list(records), columns=[name for name, _ in columns])
            best = min(best, time.perf_counter() - began)
        finally:
            await conn.execute(f"DROP TABLE IF EXISTS {plain}")
    return best


async def _keyed_plain(conn: Any, rows: int) -> str:
    """a plain table of ``rows`` rows keyed by ``k`` (0 to rows - 1), with one column to change."""
    plain = _plain_name()
    await conn.execute(f"CREATE TABLE {plain} (k BIGINT PRIMARY KEY, v BIGINT)")
    await conn.execute(f"INSERT INTO {plain} SELECT g, g FROM generate_series(0, $1::bigint - 1) g", rows)
    return plain


async def plain_keyed_update_seconds(conn: Any, rows: int, *, runs: int = _RUNS) -> float:
    """the best time to update ``rows`` rows of a plain keyed table by their keys, in one statement.

    :param conn: an asyncpg connection
    :ptype conn: Any
    :param rows: how many rows
    :ptype rows: int
    :param runs: how many timed updates
    :ptype runs: int
    :return: seconds
    :rtype: float
    """
    plain = await _keyed_plain(conn, rows)
    keys = list(range(rows))
    try:
        return await best_seconds(
            lambda: conn.execute(f"UPDATE {plain} SET v = v + 1 WHERE k = ANY($1::bigint[])", keys), runs=runs
        )
    finally:
        await conn.execute(f"DROP TABLE IF EXISTS {plain}")


async def plain_keyed_delete_seconds(conn: Any, rows: int, *, runs: int = _RUNS) -> float:
    """the best time to delete ``rows`` rows of a plain keyed table by their keys, in one statement.

    :param conn: an asyncpg connection
    :ptype conn: Any
    :param rows: how many rows
    :ptype rows: int
    :param runs: how many timed deletes, each of a table filled afresh
    :ptype runs: int
    :return: seconds
    :rtype: float
    """
    best = float("inf")
    keys = list(range(rows))
    for _ in range(runs):
        plain = await _keyed_plain(conn, rows)
        try:
            began = time.perf_counter()
            await conn.execute(f"DELETE FROM {plain} WHERE k = ANY($1::bigint[])", keys)
            best = min(best, time.perf_counter() - began)
        finally:
            await conn.execute(f"DROP TABLE IF EXISTS {plain}")
    return best


def budget_problem(what: str, took: float, baseline: float, *, factor: float = DEFAULT_BUDGET_FACTOR) -> str | None:
    """why ``took`` is over ``factor`` times its calibrated ``baseline``, or None when it is within.

    :param what: the path, for the message
    :ptype what: str
    :param took: the path's seconds
    :ptype took: float
    :param baseline: the bare operation's seconds on the same database
    :ptype baseline: float
    :param factor: the most times the baseline the path may take
    :ptype factor: float
    :return: the problem, or None
    :rtype: str | None
    """
    if took <= factor * baseline:
        return None
    return f"{what} took {took * 1000:.1f} ms, {took / baseline:.1f}x its calibrated {baseline * 1000:.1f} ms (over {factor}x)"
