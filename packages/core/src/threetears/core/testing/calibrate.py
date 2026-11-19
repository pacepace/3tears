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

import asyncio
import gc
import hashlib
import threading
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from types import TracebackType
from typing import Any, Self

from threetears.core.testing.loop_lateness import HostTicker, Tick, trusted

__all__ = [
    "CpuStopwatch",
    "best_cpu_seconds",
    "best_seconds_interleaved",
    "budget_problem",
    "plain_copy_seconds",
    "plain_keyed_delete_seconds",
    "thread_cpu_seconds",
]

#: how many times its calibrated baseline a path may take
DEFAULT_BUDGET_FACTOR = 2.0

#: runs a baseline or a path is timed over; the best is kept
_RUNS = 3

#: the most the host's own idle loop may be late while a bare and a path are timed side by side for their ratio to
#: judge the path (:func:`best_seconds_interleaved`)
HOST_MAY_BE_LATE = 0.020


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


async def best_seconds_interleaved(
    bare: Callable[[], Awaitable[Any]],
    path: Callable[[], Awaitable[Any]],
    *,
    runs: int = _RUNS,
    host_may_be_late: float = HOST_MAY_BE_LATE,
    threads: int | None = None,
) -> tuple[float, float]:
    """the fastest of ``runs`` awaits of each side, the two run turn about: the bare, then the path, ``runs`` times.

    A busy machine's noise then lands on both sides alike, so their ratio holds where two best-of-N runs taken one
    after the other would each catch a different moment's load. A machine busy for the whole measurement does not
    land alike: a core-wide load stretched one real-size read's bare 485 ms against 74-89 ms alone, and its ratio
    said nothing about the code. So the host's own idle loop ticks beside the runs (:class:`HostTicker`, in a process
    of its own), and the runs are judged only when it was late by at most ``host_may_be_late`` over them; else they
    are retaken, and when no measurement is trusted the call raises
    :class:`~threetears.core.testing.loop_lateness.CouldNotMeasure` -- never a pass, never a red
    (:func:`~threetears.core.testing.loop_lateness.trusted`).

    :param bare: makes the baseline's awaitable, once a run
    :ptype bare: Callable[[], Awaitable[Any]]
    :param path: makes the timed path's awaitable, once a run
    :ptype path: Callable[[], Awaitable[Any]]
    :param runs: how many runs of each
    :ptype runs: int
    :param host_may_be_late: the most the host's idle loop may be late over the runs
    :ptype host_may_be_late: float
    :param threads: the threads the two sides run on when more than one (a DuckDB point query: 2):
        the runs are then judged only when that many cores were free before and after them
        (:func:`~threetears.core.testing.loop_lateness.cpu_crowding`) -- a many-core host can be busy enough to
        stretch a multi-thread path while a 1 ms idle loop stays on time
    :ptype threads: int | None
    :return: the bare's best seconds, and the path's
    :rtype: tuple[float, float]
    :raises CouldNotMeasure: when the host was too late over every measurement
    """

    async def measure() -> tuple[tuple[float, float], list[Tick], list[Tick]]:
        best_bare = best_path = float("inf")
        with HostTicker() as host:
            first = time.perf_counter()
            for _ in range(runs):
                began = time.perf_counter()
                await bare()
                best_bare = min(best_bare, time.perf_counter() - began)
                began = time.perf_counter()
                await path()
                best_path = min(best_path, time.perf_counter() - began)
            last = time.perf_counter()
        return (best_bare, best_path), [(first, last)], host.ticks

    measured, _, _ = await trusted(
        measure, lambda taken: (taken[1], taken[2]), host_may_be_late=host_may_be_late, threads=threads
    )
    return measured


# -- the CPU clock: what a test of an operation's own cost bounds (moved from survey's tests/support/cpu_budget.py,
# beside the wall-clock helpers above)
#
# A unit test that bounds wall-clock time bounds how busy the machine is. Measured 2026-09-26 in survey, three ways
# with one result: stalled iterations spent 0.14-5 ms of CPU across ALL the process's threads while 101-201 ms of wall
# clock passed at load averages of 100-185 on 10 cores; 3000 production config parses beside 120 CPU burners went 82
# over a 50 ms wall bound, the worst at 244 ms of wall clock and 0.505 ms of CPU; five middleware requests took
# 37-184 ms of wall clock and the same in CPU alone, and 2.9-4.3 s of wall clock on 226-275 ms of CPU beside 120
# burners. The process was waiting for a CPU, not doing work, and under ``pytest -n auto`` the suite itself is the
# load. So a bound production needs ("this operation costs under N ms") is a bound on CPU: ``time.process_time``
# counts every thread of the process (a ``TestClient`` runs the app on a worker thread) and garbage collection, and
# does not advance while the process waits to be scheduled. It cannot measure work in ANOTHER process (a database, a
# broker) or time waiting on I/O: a test about either is about latency, and wall clock is right for it.
#
# One measurement of process CPU is still not the code's cost. A cyclic collection the block's allocations trigger
# walks the WHOLE heap (late in a pytest worker, every earlier test's leftovers), any other thread of the process adds
# its CPU, and a process moved to an efficiency core spends more on the same instructions. Measured 2026-11-17: one
# config validation read 149 ms of CPU inside a full run beside another builder, against 0.38 ms warm and 1.26 ms cold
# alone. Each adds to a run and none takes from it, so the cost is the FASTEST of a few runs, each from a collected
# heap (:func:`best_cpu_seconds`); an operation wholly on the calling thread is timed on that thread's own clock
# (:func:`thread_cpu_seconds`), which no other thread can add to.

#: runs :func:`best_cpu_seconds` takes by default: enough that one collection or one stalled slice cannot be the best,
#: few enough that a millisecond operation stays a fast test
_BEST_OF_RUNS = 5


def _cpu_seconds() -> float:
    """the CPU this process has spent, across all its threads, in seconds.

    :return: process CPU seconds since an arbitrary fixed point
    :rtype: float
    """
    return time.process_time()


def thread_cpu_seconds() -> float:
    """the CPU the calling thread alone has spent, in seconds.

    For an operation that runs wholly on the caller's thread: nothing another thread of the process does is counted.
    An operation that hands work to another thread is under-counted by it; time that with :func:`_cpu_seconds`.

    :return: this thread's CPU seconds since an arbitrary fixed point
    :rtype: float
    """
    return time.thread_time()


class CpuStopwatch:
    """the process CPU spent inside a ``with`` block, in seconds and in milliseconds.

    .. code-block:: python

        with CpuStopwatch() as spent:
            do_the_work()
        assert spent.ms < 50
    """

    def __init__(self) -> None:
        self._start = 0.0
        self.seconds = 0.0

    @property
    def ms(self) -> float:
        """the CPU spent, in milliseconds.

        :return: milliseconds of CPU
        :rtype: float
        """
        return self.seconds * 1000

    def __enter__(self) -> Self:
        self._start = _cpu_seconds()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.seconds = _cpu_seconds() - self._start


async def best_cpu_seconds(
    operation: Callable[[], Awaitable[object]],
    runs: int = _BEST_OF_RUNS,
    clock: Callable[[], float] = _cpu_seconds,
) -> float:
    """the CPU the fastest of ``runs`` awaits of ``operation()`` spent, each from a collected heap.

    The first await is run and discarded before the timed ones: a cold call's one-time costs (lazy imports, schema
    builds, first-use caches) are the process's start-up, not the operation's per-call cost. ``gc.collect()`` before
    each timed run means a collection the run itself triggers walks only what is live, the same in every worker,
    rather than whatever earlier tests left behind.

    :param operation: makes the operation's awaitable, once a run
    :ptype operation: Callable[[], Awaitable[object]]
    :param runs: how many timed runs
    :ptype runs: int
    :param clock: the CPU clock: :func:`_cpu_seconds` (every thread of the process, the default) or
        :func:`thread_cpu_seconds` (the calling thread only)
    :ptype clock: Callable[[], float]
    :return: the fastest run's CPU seconds on ``clock``
    :rtype: float
    :raises ValueError: when ``runs`` is below one
    """
    if runs < 1:
        raise ValueError(f"runs must be at least 1, got {runs}")
    await operation()
    best = float("inf")
    for _ in range(runs):
        gc.collect()
        began = clock()
        await operation()
        best = min(best, clock() - began)
    return best


def _plain_name() -> str:
    return f"calibration_{uuid.uuid7().hex[-12:]}"  # the random tail: the head is the clock


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


def budget_problem(
    what: str, took: float, baseline: float, *, factor: float = DEFAULT_BUDGET_FACTOR, plus: float = 0.0
) -> str | None:
    """why ``took`` is over ``factor`` times its calibrated ``baseline`` plus ``plus``, or None when it is within.

    :param what: the path, for the message
    :ptype what: str
    :param took: the path's seconds
    :ptype took: float
    :param baseline: the bare operation's seconds on the same database
    :ptype baseline: float
    :param factor: the most times the baseline the path may take
    :ptype factor: float
    :param plus: seconds of a fixed cost the path pays once beside its reads, measured on its own and budgeted on its
        own (a restore's worker start); 0 for none
    :ptype plus: float
    :return: the problem, or None
    :rtype: str | None
    """
    if took <= factor * baseline + plus:
        return None
    if plus:
        return (
            f"{what} took {took * 1000:.1f} ms, over {factor}x its calibrated {baseline * 1000:.1f} ms plus "
            f"{plus * 1000:.1f} ms ({(took - plus) / baseline:.1f}x once that is taken off)"
        )
    return f"{what} took {took * 1000:.1f} ms, {took / baseline:.1f}x its calibrated {baseline * 1000:.1f} ms (over {factor}x)"


#: what each of :func:`bare_loop_lateness`'s threads hashes at a time: past 2 KiB ``hashlib`` lets the interpreter
#: lock go while it hashes, so the threads load the cores and never the lock
_GIL_FREE_CHUNK = b"\x5a" * (1 << 20)


async def bare_loop_lateness(threads: int, seconds: float, *, every: float = 0.002) -> float:
    """the most a ticker on the running loop wakes late while ``threads`` threads keep the cores busy without the
    interpreter lock, for ``seconds``: this machine's own scheduling delay at that load, right now.

    A loop-timing test bounds how long work on other threads holds the interpreter lock. What a ticker measures is
    that hold plus the time the operating system kept the loop's thread off a core, which on a shared runner (other
    test batches beside this one) is no property of the code. This is the second part, measured with the same
    number of busy threads that hold the lock for no time (``hashlib`` over 1 MiB releases it), so the test's
    bound is its allowance for the hold plus this.

    :param threads: the busy threads, as many as the work under test runs
    :ptype threads: int
    :param seconds: how long to measure: as long as the work under test runs
    :ptype seconds: float
    :param every: the ticker's sleep, as the test's own ticker sleeps
    :ptype every: float
    :return: the most the ticker woke late, in seconds
    :rtype: float
    """
    stop = threading.Event()

    def busy() -> None:
        while not stop.is_set():
            hashlib.sha256(_GIL_FREE_CHUNK).digest()

    late = 0.0
    workers = [asyncio.create_task(asyncio.to_thread(busy)) for _ in range(threads)]
    try:
        ends = time.perf_counter() + seconds
        while time.perf_counter() < ends:
            asked = time.perf_counter()
            await asyncio.sleep(every)
            late = max(late, time.perf_counter() - asked - every)
    finally:
        stop.set()
        await asyncio.gather(*workers)
    return late
