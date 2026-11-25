"""Queries over the copy in query worker processes: a query runs DuckDB over a published version of the copy, mapped
read-only in a worker, and its answer comes back as shared Arrow, never pickled.

**Why** (``design-copy-worker-backend.md``, decision 4). Reports' tool reads queried the copy with DuckDB on threads of
the serving process (the "engine slots", ``shared_arrow.SharedArrowReader``): CPU beside the event loop, and three
slots leaked in replay 9 and starved every read. Here every query runs in a worker of a
:class:`~threetears.core.shared_arrow.SharedArrowPool` (named ``copy-query``). The serving process does IO alone: it
leases the version, hands the query's tables over (``SharedArrowStore.hand``: Arrow IPC in shared memory), and maps the
answer the worker wrote into a result slot in pieces. A query's rows never become Python objects or a pickle in either
process; only its SQL, its scalar binds and the worker's pid cross pickled.

**A query** is SQL with ``?`` binds (:meth:`CopyQueries.run`), or a declared query (:meth:`CopyQueries.run_named`: a
module-level function, named at construction, pickled to the worker by name, given the connection and the binds,
answering a ``pyarrow.Table`` or ``RecordBatchReader``). Either sees every table of its version as a view (the
locator tables a split table keeps, :data:`~threetears.core.collections.shared_copy.LOCATED_PREFIX`, dropped), the
tables it was handed (``tables=``: keys to join, say) under their own names, and what the copy's ``prepare`` hook made
(a module-level function run on each query's connection: a product's views). Binds are scalars, at most
:data:`_MAX_BINDS`: a list of keys is rows, and goes as a handed table.

**Version pinning.** A query reads the version it names, or the one current as it is submitted, leased before it waits
for a worker and held until its answer is mapped: a publish while it waits or runs never changes its answer, and the
answer says which version it read (:attr:`QueryAnswer.version`).

**Bounded, in order.** At most ``workers + max_waiting`` queries are admitted at once (each holds a lease, which keeps
its version's files in the store); a caller past that waits in :meth:`CopyQueries.run` for room, in the order it came,
never a queue grown without bound (``wait=False`` refuses at once with :class:`QueryPoolFull`). Admitted queries reach
the workers in the order their callers called (a baton passed caller to caller: each asks for its worker only after
the one before it asked), so a burst is answered first come first served.

**Timeouts and cancellation stop the worker's query.** A caller that stops waiting (cancelled, or past its
``timeout``: :class:`QueryTimeout`) marks its query's ticket on the pool's cancel board, a small shared array every
worker maps from its start. The worker running it watches the board on a thread of its own and interrupts DuckDB
(``DuckDBPyConnection.interrupt``); one not yet started refuses it at its start. Either way the worker is free for the
next query within milliseconds, never held by an answer nobody will read.

**A dead worker** (killed, out of memory) fails the queries it and its pool's other workers were running with
:class:`QueryWorkerDied`, and the pool is replaced for the next query (``SharedArrowPool``'s own replacement). A query
is never given again: its caller decides, since a query that killed one worker would kill the next.

**Not yet here** (later chunks of the design): the inline keyed lookup below a measured cut-off (3T-6,
``INLINE_LOOKUP_KEYS``); the follower's reads (3T-7) and reports' tool reads (R-3) moving onto this; the engine slots'
removal (3T-8).
"""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import datetime
import decimal
import multiprocessing
import os
import pickle
import threading
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from typing import Any, Final

import pyarrow as pa

from threetears.core.collections.shared_copy import LOCATED_PREFIX
from threetears.core.frozen_state import EMPTY, frozen
from threetears.core.shared_arrow import JobContext, MappedSegment, SharedArrowPool, SharedArrowStore, SharedLease
from threetears.core.utils.cpu_quota import bound_native_threads, effective_cpu_count
from threetears.observe import contain, get_logger

__all__ = [
    "CopyQueries",
    "QueryAnswer",
    "QueryPoolFull",
    "QueryStatus",
    "QueryTimeout",
    "QueryWorkerDied",
    "query_workers",
]

log = get_logger(__name__)

#: the CPUs the query workers leave to the rest of the pod, one each: the serving process (its event loop, its IO
#: threads, the store's sweeps), the copy's apply worker (``shared_copy``: CPU-bound while a change applies, and
#: election night's changes come back to back) and its reload worker (``copy_reload``: a whole table's rows at a time,
#: kept apart so a reload never delays an apply, decision 3). Decision 5: the query workers get the pod's CPU allowance
#: after those. Reports' pods are limited to 6 CPUs (``.devops/chart/values-prod.yaml:37``, ``values-dev.yaml:34``),
#: and its replay pod of 2026-11-25 read 6 effective CPUs on a 10-core machine (``effective_cpus`` in its start log),
#: so 6 - 3 = 3 query workers, each running DuckDB on one thread (:data:`_QUERY_THREADS`): the workers' threads never
#: pass the allowance left, so none is throttled. A product with workers of its own (reports' compute workers, 2) holds
#: them back too (``query_workers(held_back=)``). The count is measured at replay size before it is fixed (R-5).
_HELD_BACK_CPUS: Final = 3

#: DuckDB's threads in each query worker. One: the pod's queries come many at once (every tool call a client makes,
#: every report's digest), so its CPUs serve more queries side by side, one worker each, rather than one query faster;
#: a query on more threads than its share would take CPU from the apply worker beside it
_QUERY_THREADS: Final = 1

#: the queries admitted beyond those running. Each admitted query holds a lease on its version, which keeps that
#: version's files in the store (memory only, design section 5: held bytes peak 1.27x one version at replay); a queue of
#: 16 holds at most that many leases, most on the current version, and is several rounds of 3 workers' heavy reads
#: (seconds each at prod size, inventory section 4): a caller past it waits for room, never the queue growing
_MAX_WAITING: Final = 16

#: the longest a query may wait and run, by default. The heaviest read over the copy is a report build: 12-17 s on
#: cobalt prod (reports' ``compute_workers`` docstring); a query still running at 120 s is runaway, and stopping it
#: frees its worker for the queries behind it
_TIMEOUT_SECONDS: Final = 120.0

#: a query worker holding more private memory than this after a query is replaced before its next (reports' compute
#: workers' measured bound, ``COMPUTE_WORKER_PRIVATE_BYTES``: 1 GiB replaced its pool 61 times in 70 minutes of the
#: 2026-11-01 replay). Private (anonymous) memory, never the store's pages the worker maps
_QUERY_PRIVATE_BYTES: Final = 2 << 30

#: the most one piece of an answer may take (``SharedArrowStore.reserve_pieces``), reserved as the one before fills
_ANSWER_PIECE_BYTES: Final = 256 << 20

#: an answer's rows are written in files of about this many bytes, each one table of the result (``answer-<n>``):
#: under a piece, so a file never passes its piece, and the worker holds one file's batches at a time, never the
#: whole answer
_ANSWER_FILE_BYTES: Final = 64 << 20

#: rows per batch read from DuckDB's answer
_BATCH_ROWS: Final = 64 * 1024

#: the name each answer file's table starts with; the files are numbered in order
_ANSWER: Final = "answer-"

#: the most scalar binds a query takes: a query's own values (a state, a race, an instant). Keys by the hundred are
#: rows, handed as a table (``tables=``) so they cross as Arrow, never pickled
_MAX_BINDS: Final = 64

#: the bind values a query takes: scalars DuckDB binds as they are
_SCALARS: Final = (
    str,
    int,
    float,
    bool,
    bytes,
    decimal.Decimal,
    datetime.date,
    datetime.datetime,
    datetime.time,
    datetime.timedelta,
    type(None),
)

#: the cancel board's slots: a ticket ``t`` is cancelled when ``board[t % _BOARD_SLOTS] == t``. Tickets start at 1
#: (the board starts zeroed). A slot is reused only by a ticket this many later, and at most ``workers + max_waiting``
#: queries are in flight (checked at construction), so a live query's slot is never another's
_BOARD_SLOTS: Final = 4096

#: how often a running query's watcher reads the board: a cancelled query stops within this and DuckDB's own check
_WATCH_SECONDS: Final = 0.005

#: the name each query worker gives itself (``/proc/<pid>/comm``, 15 bytes at most)
_QUERY_PROCESS_NAME: Final = "copy-query"


class QueryPoolFull(Exception):
    """refused at once (``wait=False``): the pool has admitted all the queries it holds."""


class QueryTimeout(TimeoutError):
    """the query waited and ran past its timeout; its worker's query was stopped."""


class _QueryStopped(RuntimeError):
    """raised in the worker for a query its caller stopped (cancelled, or past its timeout): DuckDB was interrupted,
    or the query never started."""


class QueryWorkerDied(RuntimeError):
    """the worker running the query died (killed, or out of memory). The pool is replaced; the query is not given
    again."""


def query_workers(cpus: int | None = None, *, held_back: int = _HELD_BACK_CPUS) -> int:
    """the query workers a pod runs: its CPU allowance less what the rest of it holds (:data:`_HELD_BACK_CPUS`), at
    least one.

    :param cpus: the CPUs the pod may use; None for this process's (its cgroup's quota within its affinity:
        ``cpu_quota.effective_cpu_count``)
    :ptype cpus: int | None
    :param held_back: the CPUs left to the serving process and the copy's other workers
    :ptype held_back: int
    :return: the worker count
    :rtype: int
    """
    allowed = effective_cpu_count(cpu_count=getattr(os, "process_cpu_count", os.cpu_count)) if cpus is None else cpus
    return max(1, allowed - held_back)


@dataclass(frozen=True, slots=True)
class _Setup:
    """what a query worker is given as it starts, set once and read only after (:func:`_worker_start`).

    :ivar board: the pool's cancel board, shared with the serving process
    :ivar threads: DuckDB's threads per query
    """

    board: Any
    threads: int


#: the worker's setup, set once by :func:`_worker_start` as the worker starts and read only after; None in every
#: process but a query worker
_SETUP: _Setup | None = None


def _worker_start(board: Any, threads: int) -> None:
    """a query worker's start: its setup kept, and pyarrow's pools sized to its DuckDB threads.

    :param board: the cancel board
    :ptype board: multiprocessing.sharedctypes.RawArray
    :param threads: DuckDB's threads per query
    :ptype threads: int
    :return: nothing
    :rtype: None
    """
    global _SETUP
    _SETUP = _Setup(board=board, threads=threads)
    bound_native_threads(threads)


def _cancelled(board: Any, ticket: int) -> bool:
    """whether the query's caller stopped it.

    :param board: the cancel board
    :ptype board: multiprocessing.sharedctypes.RawArray
    :param ticket: the query's ticket
    :ptype ticket: int
    :return: whether it is marked
    :rtype: bool
    """
    return bool(board[ticket % _BOARD_SLOTS] == ticket)


def _watch(board: Any, ticket: int, connection: Any, done: threading.Event) -> None:
    """a running query's watcher, on a thread of the worker: interrupts DuckDB once the query is marked stopped, again
    each read until the query is done (an interrupt before DuckDB starts executing is cleared by its start).

    :param board: the cancel board
    :ptype board: multiprocessing.sharedctypes.RawArray
    :param ticket: the query's ticket
    :ptype ticket: int
    :param connection: the query's connection
    :ptype connection: duckdb.DuckDBPyConnection
    :param done: set once the query is done
    :ptype done: threading.Event
    :return: nothing
    :rtype: None
    """
    while not done.wait(_WATCH_SECONDS):
        if _cancelled(board, ticket):
            connection.interrupt()


@dataclass(frozen=True, slots=True)
class _Query:
    """a query as its worker is given it (pickled: its SQL or declared function by name, and its scalar binds).

    :ivar sql: the SQL; None for a declared query
    :ivar declared: the declared query's function; None for SQL
    :ivar binds: the scalar binds
    :ivar prepare: the copy's prepare hook; None for none
    """

    sql: str | None
    declared: Callable[[Any, Sequence[Any]], Any] | None
    binds: tuple[Any, ...]
    prepare: Callable[[Any], object] | None


def _connected(context: JobContext, threads: int, prepare: Callable[[Any], object] | None) -> Any:
    """the query's connection: every table of its version a view (the locators dropped), the tables handed to it
    registered under their names, and the prepare hook run.

    :param context: the job's context
    :ptype context: JobContext
    :param threads: DuckDB's threads
    :ptype threads: int
    :param prepare: the copy's prepare hook; None for none
    :ptype prepare: Callable[[duckdb.DuckDBPyConnection], object] | None
    :return: the connection, the caller's to close
    :rtype: duckdb.DuckDBPyConnection
    :raises ValueError: for a handed table named as a table of the version
    """
    version = context.version
    if version is None:
        raise ValueError("a copy query reads a version")
    handed = EMPTY if context.tables is None else context.tables.tables
    clash = sorted(set(handed) & set(version.tables))
    if clash:
        raise ValueError(f"a query's tables are named as the copy's: {clash}")
    connection = version.connect(config={"threads": threads})
    try:
        for name in version.tables:
            if name.startswith(LOCATED_PREFIX):
                connection.execute(f'DROP VIEW "{name}"')
        for name, table in handed.items():
            connection.register(name, table)
        if prepare is not None:
            prepare(connection)
    except BaseException:
        connection.close()
        raise
    return connection


def _batches(answer: Any) -> pa.RecordBatchReader:
    """a query's answer as a reader of its batches.

    :param answer: a DuckDB relation or result, a ``pyarrow.Table`` or a ``RecordBatchReader``
    :ptype answer: Any
    :return: the reader
    :rtype: pa.RecordBatchReader
    :raises TypeError: for anything else
    """
    if isinstance(answer, pa.RecordBatchReader):
        return answer
    if isinstance(answer, pa.Table):
        return answer.to_reader(max_chunksize=_BATCH_ROWS)
    # to_arrow_reader from DuckDB 1.4; fetch_record_batch before it (deprecated since)
    reader = getattr(answer, "to_arrow_reader", None) or getattr(answer, "fetch_record_batch", None)
    if reader is None:
        raise TypeError(f"a query answers a pyarrow Table or RecordBatchReader, not {type(answer).__name__}")
    batches: pa.RecordBatchReader = reader(_BATCH_ROWS)
    return batches


def _write_answer(context: JobContext, batches: pa.RecordBatchReader) -> int:
    """write the answer into the job's result slot, a file at a time (:data:`_ANSWER_FILE_BYTES`): the worker holds
    one file's batches, never the whole answer. An answer with no rows is one file of its columns.

    :param context: the job's context
    :ptype context: JobContext
    :param batches: the answer
    :ptype batches: pa.RecordBatchReader
    :return: the rows written
    :rtype: int
    """
    if context.result is None:
        raise ValueError("a copy query writes its answer into a result slot")
    rows = 0
    files = 0
    with context.result.writing() as step:
        held: list[pa.RecordBatch] = []
        held_bytes = 0
        for batch in batches:
            held.append(batch)
            held_bytes += batch.nbytes
            rows += batch.num_rows
            if held_bytes >= _ANSWER_FILE_BYTES:
                step({f"{_ANSWER}{files:06d}": pa.Table.from_batches(held, batches.schema)})
                files += 1
                held = []
                held_bytes = 0
        if held or files == 0:
            step({f"{_ANSWER}{files:06d}": pa.Table.from_batches(held, batches.schema)})
    return rows


def query_job(context: JobContext, ticket: int, query: _Query) -> tuple[int, int]:
    """a query in its worker: DuckDB over the leased version, the answer written into the result slot as Arrow.

    :param context: the job's context: the version, the handed tables, the result slot
    :ptype context: JobContext
    :param ticket: the query's ticket on the cancel board
    :ptype ticket: int
    :param query: the query
    :ptype query: _Query
    :return: the rows answered and this worker's pid
    :rtype: tuple[int, int]
    :raises _QueryStopped: when its caller stopped it, before it started or while it ran
    """
    setup = _SETUP
    if setup is None:
        raise RuntimeError("a copy query runs in a CopyQueries worker")
    if _cancelled(setup.board, ticket):
        raise _QueryStopped("the query was stopped before it started: its caller cancelled it or its timeout passed")
    connection = _connected(context, setup.threads, query.prepare)
    done = threading.Event()
    watcher = threading.Thread(
        target=_watch, args=(setup.board, ticket, connection, done), name="copy-query-watch", daemon=True
    )
    watcher.start()
    try:
        if query.declared is not None:
            answer = query.declared(connection, query.binds)
        else:
            answer = connection.execute(query.sql, list(query.binds))
        rows = _write_answer(context, _batches(answer))
    except Exception as exc:
        if _cancelled(setup.board, ticket):
            # DuckDB's interrupt surfaces as its InterruptException, or as an Arrow error from the reader mid-stream
            raise _QueryStopped("the query was stopped: its caller cancelled it or its timeout passed") from exc
        raise
    finally:
        done.set()
        watcher.join()
        connection.close()
    return rows, os.getpid()


class QueryAnswer:
    """a query's answer: its rows mapped from the shared result the worker wrote, zero-copy, until :meth:`close`.

    A table kept past the close (a slice, a column) keeps its files (``MappedSegment``'s pin): the store counts them
    until the last buffer goes.

    :ivar table: the rows
    :ivar version: the version the query read
    :ivar worker: the pid of the worker that answered
    """

    def __init__(self, table: pa.Table, version: str, worker: int, segment: MappedSegment) -> None:
        self.table = table
        self.version = version
        self.worker = worker
        self._segment = segment

    def close(self) -> None:
        """let the answer's files go (removed once no table taken from them is held). Blocking: removes files; call
        it on a thread (:meth:`aclose`), never the event loop.

        :return: nothing
        :rtype: None
        """
        self._segment.close()

    async def aclose(self) -> None:
        """:meth:`close`, on a thread.

        :return: nothing
        :rtype: None
        """
        await asyncio.to_thread(self._segment.close)

    async def __aenter__(self) -> QueryAnswer:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


@dataclass(frozen=True, slots=True)
class QueryStatus:
    """a pool's counts, each written only by its pool on the event loop.

    :ivar workers: the worker processes
    :ivar admitted: queries admitted now (waiting for a worker, or running)
    :ivar answered: queries answered
    :ivar failed: queries that raised (their SQL, their declared function, or their worker's death)
    :ivar stopped: queries stopped by their caller (cancelled, or past their timeout)
    :ivar timed_out: of those, the ones past their timeout
    :ivar died: queries failed by their worker's death
    :ivar refused: queries refused at once (``wait=False``)
    """

    workers: int
    admitted: int
    answered: int
    failed: int
    stopped: int
    timed_out: int
    died: int
    refused: int


def _picklable(what: str, value: object) -> None:
    """raise unless ``value`` pickles (a module-level function, sent to the worker by name).

    :param what: what it is, for the error
    :ptype what: str
    :param value: the function
    :ptype value: object
    :return: nothing
    :rtype: None
    :raises ValueError: when it does not pickle
    """
    try:
        pickle.dumps(value)
    except (pickle.PicklingError, TypeError, AttributeError) as exc:
        raise ValueError(f"{what} must be a module-level function, sent to the worker by name: {exc}") from exc


def _checked_binds(binds: Sequence[Any]) -> tuple[Any, ...]:
    """a query's binds, each a scalar, at most :data:`_MAX_BINDS`.

    :param binds: the binds
    :ptype binds: Sequence[Any]
    :return: them, as a tuple
    :rtype: tuple[Any, ...]
    :raises ValueError: for too many, or one that is not a scalar
    """
    if len(binds) > _MAX_BINDS:
        raise ValueError(f"a query takes at most {_MAX_BINDS} binds; hand rows as a table (tables=), not binds")
    for value in binds:
        if not isinstance(value, _SCALARS):
            raise ValueError(f"a query's bind is a scalar, not {type(value).__name__}; hand rows as a table (tables=)")
    return tuple(binds)


def _released(lease: SharedLease) -> None:
    """let a lease go on a thread, never the event loop (its release sweeps the store), logging a failure.

    :param lease: the lease
    :ptype lease: SharedLease
    :return: nothing
    :rtype: None
    """
    releasing = asyncio.get_running_loop().run_in_executor(None, lease.release)
    releasing.add_done_callback(_said_if_failed)


def _said_if_failed(done: asyncio.Future[Any]) -> None:
    """contain what a lease's release raised, so it is logged and counted.

    :param done: the release, done
    :ptype done: asyncio.Future[Any]
    :return: nothing
    :rtype: None
    """
    failed = None if done.cancelled() else done.exception()
    if isinstance(failed, Exception):
        contain(failed, log=log, what="a copy query's version lease could not be released")


def _release_when_leased(leasing: asyncio.Future[SharedLease]) -> None:
    """a lease its caller stopped waiting for: let go once taken.

    :param leasing: the lease being taken, done
    :ptype leasing: asyncio.Future[SharedLease]
    :return: nothing
    :rtype: None
    """
    if not leasing.cancelled() and leasing.exception() is None:
        _released(leasing.result())


def _pass_on(baton: asyncio.Future[None]) -> None:
    """give the next caller its turn to ask for a worker.

    :param baton: this caller's baton
    :ptype baton: asyncio.Future[None]
    :return: nothing
    :rtype: None
    """
    if not baton.done():
        baton.set_result(None)


class CopyQueries:
    """queries over a copy's versions, each run in a query worker process and answered in shared Arrow.

    Make it as the pod starts, or on a thread: its pool starts the fork server its workers are forked by. Every
    method is called on one event loop.

    :param store: the store the copy's versions are published in (``SharedCopyBackend.store``)
    :ptype store: SharedArrowStore
    :param workers: the worker processes; None for :func:`query_workers`
    :ptype workers: int | None
    :param prepare: run on each query's connection after its tables are views: a module-level function taking the
        connection (a product's views over the copy); None for none
    :ptype prepare: Callable[[duckdb.DuckDBPyConnection], object] | None
    :param declared: the declared queries (:meth:`run_named`), by name: module-level functions taking the connection
        and the binds, answering a ``pyarrow.Table`` or ``RecordBatchReader``
    :ptype declared: Mapping[str, Callable[[duckdb.DuckDBPyConnection, Sequence[Any]], Any]]
    :param max_waiting: the queries admitted beyond those running (:data:`_MAX_WAITING`)
    :ptype max_waiting: int
    :param timeout: each query's timeout by default, waiting and running (:data:`_TIMEOUT_SECONDS`)
    :ptype timeout: float
    :param private_bytes: a worker holding more private memory than this after a query is replaced
    :ptype private_bytes: int
    :param threads: DuckDB's threads in each worker (:data:`_QUERY_THREADS`)
    :ptype threads: int
    :param preload: modules the workers hold from their start (the prepare hook's and the declared queries')
    :ptype preload: Sequence[str]
    :raises ValueError: for a prepare hook or a declared query that is not a module-level function, or a bound past the
        cancel board's slots
    """

    def __init__(
        self,
        store: SharedArrowStore,
        *,
        workers: int | None = None,
        prepare: Callable[[Any], object] | None = None,
        declared: Mapping[str, Callable[[Any, Sequence[Any]], Any]] = EMPTY,
        max_waiting: int = _MAX_WAITING,
        timeout: float = _TIMEOUT_SECONDS,
        private_bytes: int = _QUERY_PRIVATE_BYTES,
        threads: int = _QUERY_THREADS,
        preload: Sequence[str] = (),
    ) -> None:
        count = query_workers() if workers is None else workers
        if count < 1 or max_waiting < 0 or count + max_waiting >= _BOARD_SLOTS:
            raise ValueError(f"a copy query pool holds 1 to {_BOARD_SLOTS - 1} queries: {count} + {max_waiting}")
        if prepare is not None:
            _picklable("a copy query's prepare hook", prepare)
        for name, function in declared.items():
            _picklable(f"the declared query {name!r}", function)
        self._store = store
        self._workers = count
        self._prepare = prepare
        self._declared = frozen(declared)
        self._timeout = timeout
        #: the cancel board, shared with every worker the pool forks (its initializer's argument)
        self._board = multiprocessing.RawArray(ctypes.c_int64, _BOARD_SLOTS)
        self._pool = SharedArrowPool(
            store,
            workers=count,
            private_bytes=private_bytes,
            name="copy query",
            initializer=_worker_start,
            initargs=(self._board, threads),
            preload=(__name__, *preload),
            process_name=_QUERY_PROCESS_NAME,
        )
        #: the queries admitted at once; its waiters are served in the order they came
        self._room = asyncio.Semaphore(count + max_waiting)
        #: the last caller's baton: the next caller asks for a worker once it is passed (:meth:`_answer`)
        self._baton: asyncio.Future[None] | None = None
        self._ticket = 0
        self._admitted_now = 0
        self._answered = 0
        self._failed = 0
        self._stopped = 0
        self._timed_out = 0
        self._died = 0
        self._refused = 0

    @property
    def workers(self) -> int:
        """the worker processes.

        :return: the count
        :rtype: int
        """
        return self._workers

    def status(self) -> QueryStatus:
        """the pool's counts.

        :return: them
        :rtype: QueryStatus
        """
        return QueryStatus(
            workers=self._workers,
            admitted=self._admitted_now,
            answered=self._answered,
            failed=self._failed,
            stopped=self._stopped,
            timed_out=self._timed_out,
            died=self._died,
            refused=self._refused,
        )

    async def run(
        self,
        sql: str,
        binds: Sequence[Any] = (),
        *,
        tables: Mapping[str, pa.Table] | None = None,
        version: str | None = None,
        timeout: float | None = None,
        reader: bool = True,
        wait: bool = True,
    ) -> QueryAnswer:
        """run SQL over a version of the copy in a query worker.

        :param sql: the query, with ``?`` binds
        :ptype sql: str
        :param binds: its binds, scalars (at most :data:`_MAX_BINDS`)
        :ptype binds: Sequence[Any]
        :param tables: tables handed to it, registered under their names (keys to join): cross as shared Arrow
        :ptype tables: Mapping[str, pa.Table] | None
        :param version: the version it reads; None for the one current now
        :ptype version: str | None
        :param timeout: the longest it may wait and run; None for the pool's
        :ptype timeout: float | None
        :param reader: a client waits on it (given the next free worker ahead of background queries); False for
            background work
        :ptype reader: bool
        :param wait: wait for room when the pool has admitted all it holds; False to refuse at once
        :ptype wait: bool
        :return: the answer; the caller closes it
        :rtype: QueryAnswer
        :raises LookupError: when nothing is published, or the version named is gone
        :raises QueryTimeout: past its timeout (its worker's query stopped)
        :raises QueryWorkerDied: when its worker died
        :raises QueryPoolFull: refused at once (``wait=False``)
        :raises Exception: what the query raised in its worker (DuckDB's error for bad SQL)
        """
        query = _Query(sql=sql, declared=None, binds=_checked_binds(binds), prepare=self._prepare)
        return await self._answer(query, tables, version, timeout, reader, wait)

    async def run_named(
        self,
        name: str,
        binds: Sequence[Any] = (),
        *,
        tables: Mapping[str, pa.Table] | None = None,
        version: str | None = None,
        timeout: float | None = None,
        reader: bool = True,
        wait: bool = True,
    ) -> QueryAnswer:
        """run a declared query over a version of the copy in a query worker. As :meth:`run`, the query named.

        :param name: the declared query
        :ptype name: str
        :param binds: its binds, scalars
        :ptype binds: Sequence[Any]
        :param tables: as :meth:`run`
        :ptype tables: Mapping[str, pa.Table] | None
        :param version: as :meth:`run`
        :ptype version: str | None
        :param timeout: as :meth:`run`
        :ptype timeout: float | None
        :param reader: as :meth:`run`
        :ptype reader: bool
        :param wait: as :meth:`run`
        :ptype wait: bool
        :return: the answer; the caller closes it
        :rtype: QueryAnswer
        :raises KeyError: for a query not declared
        :raises Exception: as :meth:`run`
        """
        declared = self._declared.get(name)
        if declared is None:
            raise KeyError(f"no copy query is declared as {name!r}: {sorted(self._declared)}")
        query = _Query(sql=None, declared=declared, binds=_checked_binds(binds), prepare=self._prepare)
        return await self._answer(query, tables, version, timeout, reader, wait)

    async def _answer(
        self,
        query: _Query,
        tables: Mapping[str, pa.Table] | None,
        version: str | None,
        timeout: float | None,
        reader: bool,
        wait: bool,
    ) -> QueryAnswer:
        """admit the query, lease its version, ask for a worker in its caller's turn, and map its answer; a caller
        that stops waiting marks its ticket stopped.

        Nothing here waits before the ticket and the baton are taken, so they are in the order callers called.
        """
        if not wait and self._room.locked():
            self._refused += 1
            raise QueryPoolFull(
                f"the copy's query pool has admitted all it holds ({self._workers} running and waiting)"
            )
        self._ticket += 1
        ticket = self._ticket
        before = self._baton
        baton: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._baton = baton
        limit = self._timeout if timeout is None else timeout
        asked = False
        clock = asyncio.timeout(limit)
        try:
            async with clock, self._admitted():
                lease = await self._leased(version)
                try:
                    if before is not None:
                        await asyncio.shield(before)
                    # passed before the ask, which the pool's run makes before it first waits: the next caller wakes
                    # only after this one has asked for its worker
                    _pass_on(baton)
                    asked = True
                    answered = await self._pool.run(
                        query_job,
                        ticket,
                        query,
                        read=lease.version.name,
                        tables=tables,
                        result_pieces=_ANSWER_PIECE_BYTES,
                        reader=reader,
                    )
                finally:
                    _released(lease)
        except TimeoutError as exc:
            if not clock.expired():
                self._failed += 1
                raise
            self._stop(ticket, asked)
            self._timed_out += 1
            raise QueryTimeout(
                f"the copy query waited and ran past its {limit} s timeout; its worker's query was stopped"
            ) from exc
        except asyncio.CancelledError:
            self._stop(ticket, asked)
            raise
        except BrokenProcessPool as exc:
            self._died += 1
            self._failed += 1
            raise QueryWorkerDied(
                "the copy query's worker died (killed, or out of memory) while it ran; the worker is replaced and "
                "the query was not run again"
            ) from exc
        except Exception:
            self._failed += 1
            raise
        finally:
            if not baton.done():
                # stopped before its turn to ask: the turn passes on once the caller before it has asked
                if before is None or before.done():
                    _pass_on(baton)
                else:
                    before.add_done_callback(lambda _: _pass_on(baton))
        return self._mapped(answered.value, answered.segment, lease.version.name)

    def _mapped(self, value: tuple[int, int], segment: MappedSegment | None, version: str) -> QueryAnswer:
        """the answer the worker wrote, its files' tables side by side (no buffer copied).

        :param value: the job's rows and pid
        :ptype value: tuple[int, int]
        :param segment: the answer's files, mapped
        :ptype segment: MappedSegment | None
        :param version: the version read
        :ptype version: str
        :return: the answer
        :rtype: QueryAnswer
        """
        if segment is None:
            raise RuntimeError("a copy query's worker wrote no answer")
        files = sorted(name for name in segment.tables if name.startswith(_ANSWER))
        table = pa.concat_tables([segment.tables[name] for name in files])
        _, worker = value
        self._answered += 1
        return QueryAnswer(table, version, worker, segment)

    def _stop(self, ticket: int, asked: bool) -> None:
        """mark a query its caller stopped on the cancel board: its worker interrupts it, or refuses it at its start.

        :param ticket: the query's ticket
        :ptype ticket: int
        :param asked: whether it asked for a worker (a query that never did needs no mark)
        :ptype asked: bool
        :return: nothing
        :rtype: None
        """
        self._stopped += 1
        if asked:
            self._board[ticket % _BOARD_SLOTS] = ticket

    @contextlib.asynccontextmanager
    async def _admitted(self) -> AsyncIterator[None]:
        """the query's room in the pool, held while it waits for a worker and runs; callers waiting for room are
        served in the order they came.

        :return: a context manager holding the room
        :rtype: AsyncIterator[None]
        """
        await self._room.acquire()
        self._admitted_now += 1
        try:
            yield
        finally:
            self._admitted_now -= 1
            self._room.release()

    async def _leased(self, version: str | None) -> SharedLease:
        """a lease on the version named, or the current one, taken on a thread (it locks the version's directory). A
        caller that stops waiting meanwhile has it let go once taken.

        :param version: the version; None for the current one
        :ptype version: str | None
        :return: the lease
        :rtype: SharedLease
        :raises LookupError: when nothing is published, or the version named is gone
        """
        leasing = asyncio.ensure_future(asyncio.to_thread(self._store.lease, version))
        try:
            return await asyncio.shield(leasing)
        except asyncio.CancelledError:
            leasing.add_done_callback(_release_when_leased)
            raise

    async def close(self) -> None:
        """shut the workers down, waiting for them.

        :return: nothing
        :rtype: None
        """
        await self._pool.close()
