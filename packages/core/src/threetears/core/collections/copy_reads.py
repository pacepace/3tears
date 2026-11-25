"""The copy follower's own reads over its copy: keyed lookups, a version pinned with its mark, and jobs over it.

**Why** (``design-copy-worker-backend.md``, decision 4; inventory chunk 3T-7). The follower read its copy on DuckDB
cursors of the serving process: the rows a pass's keys hold (``_held_arrow``, ``_copy_rows_of_keys``), a checkpoint's
read of one applied state (``reading_marked``: a cursor and the engine slot it holds), its parts (a table whole, a
scope, a patched part) and a drift check's or a reload's rows of a state (``copy_rows_of_state``). CPU beside the event
loop, and on a pod's shared Arrow copy each such read held an engine slot. Here the follower reads through
:class:`CopyReads`, and over a copy published as shared Arrow (:class:`QueriedReads`) none of it opens a DuckDB
connection or a cursor in the serving process:

- **a keyed lookup** is :meth:`~threetears.core.collections.copy_queries.CopyQueries.lookup` (Arrow's hash kernels,
  in a query worker at and above its measured cut-off): keyed, never a scan by SQL;
- **the current version** is :meth:`~threetears.core.collections.copy_queries.CopyQueries.current`: a lease and the
  mark of the change it holds, read off its mark file on a thread, no cursor (:class:`PinnedCopy`);
- **a job** (a checkpoint's parts, a state's rows, a drift check's fingerprint) is
  :meth:`~threetears.core.collections.copy_queries.CopyQueries.run_job` over that version in a query worker, its
  answer shared Arrow (a checkpoint's parts come back encoded, one row a part), never pickled rows.

**The in-process copy's own reads** (:class:`CursorReads`): a follower whose copy is the in-process DuckDB
(``cache.duckdb.DuckDBBackend``) holds its rows in the serving process already, and reads them on a cursor of its
snapshot, on a thread, running the same jobs there; a drift check's fingerprint runs in the follower's worker processes,
as before. It goes with that backend as a follower's backend (chunk 3T-8): it is not a path a shared copy takes.

**One state.** A :class:`PinnedCopy` reads one applied state for as long as it is open: a lease on the version over a
shared copy (no publish retires it meanwhile), a snapshot cursor over the in-process one.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, ExitStack
from typing import TYPE_CHECKING, Any, Final, Protocol

import pyarrow as pa
import pyarrow.ipc as pa_ipc

from threetears.core.arrow_keys import stored_instants
from threetears.core.cache.duckdb_binds import KEYS, run_joined
from threetears.core.collections.copy_queries import CopyQueries, CopyVersion
from threetears.core.collections.volume_fulls import fingerprint_ipc, rows_fingerprint
from threetears.core.sql_fragments import quote_identifier
from threetears.observe import contain, get_logger

if TYPE_CHECKING:
    from threetears.core.collections.copy_follower import FollowedTable
    from threetears.core.process_workers import ProcessWorkers

__all__ = [
    "CopyReads",
    "CursorReads",
    "PinnedCopy",
    "QueriedReads",
    "keyed_job",
]

log = get_logger(__name__)

#: the name a keyed read's keys are handed to its job under: one row a key, ``k1`` … ``kn`` its key columns as text
_KEYS_TABLE: Final = "__copy_keys"

#: the fingerprint a fingerprinting job answers, one row
_FINGERPRINT: Final = "fingerprint"


class PinnedCopy(Protocol):
    """one applied state of the copy, held for a reader until :meth:`close`: the mark of the change it holds, and
    jobs over it."""

    @property
    def mark(self) -> int | None:
        """the mark of the change this state holds; None before any."""
        ...

    async def run_job(
        self,
        job: Callable[[Any, Sequence[Any]], Any],
        args: Sequence[Any] = (),
        *,
        tables: Mapping[str, Any] | None = None,
    ) -> Any:
        """``job(connection, args)`` over this state, ``tables`` registered on its connection: its answer as a
        ``pyarrow.Table``."""
        ...

    async def fingerprint(
        self,
        job: Callable[[Any, Sequence[Any]], Any],
        args: Sequence[Any],
        kinds: tuple[tuple[str, str], ...],
        *,
        tables: Mapping[str, Any] | None = None,
    ) -> str:
        """the fingerprint (``volume_fulls.rows_fingerprint`` by ``kinds``) of the rows ``job`` answers over this
        state, taken in a worker process."""
        ...

    async def close(self) -> None:
        """let the state go."""
        ...


class CopyReads(Protocol):
    """the follower's reads over its copy (module docstring)."""

    async def current(self) -> PinnedCopy:
        """the state applied now, pinned; the caller closes it."""
        ...

    async def rows_of_keys(self, table: FollowedTable, columns: Sequence[str], keys: Any, out: Sequence[str]) -> Any:
        """the rows of ``table`` the copy holds now whose ``columns`` (a leading part of its key, or all of it) hold
        one of ``keys`` (those columns as text, a row a key): ``out`` of them, each row once."""
        ...

    async def run_job(
        self,
        job: Callable[[Any, Sequence[Any]], Any],
        args: Sequence[Any] = (),
        *,
        tables: Mapping[str, Any] | None = None,
    ) -> Any:
        """``job(connection, args)`` over the state applied now: its answer as a ``pyarrow.Table``."""
        ...


class CursorCopy(Protocol):
    """an in-process copy as :class:`CursorReads` reads it (``FollowerBackend``'s reads)."""

    def read_snapshot(self) -> AbstractContextManager[Any]:
        """a cursor over one applied state of the copy, for as long as the block runs."""
        ...

    def staged_mark(self, cursor: Any) -> int | None:
        """the mark of the change the state ``cursor`` reads was applied with; None before any."""
        ...


def _table_of(answer: Any) -> pa.Table:
    """a job's answer as one table.

    :param answer: a ``pyarrow.Table``, a ``RecordBatchReader``, or a DuckDB result or relation
    :ptype answer: Any
    :return: the table
    :rtype: pa.Table
    :raises TypeError: for anything else
    """
    if isinstance(answer, pa.Table):
        return answer
    if isinstance(answer, pa.RecordBatchReader):
        return answer.read_all()
    for name in ("to_arrow_table", "fetch_arrow_table"):
        read = getattr(answer, name, None)
        if read is not None:
            table: pa.Table = read()
            return table
    raise TypeError(f"a copy job answers a pyarrow Table or RecordBatchReader, not {type(answer).__name__}")


def _keyed_statement(table: FollowedTable, columns: Sequence[str], out: Sequence[str]) -> str:
    """the semi-join that reads ``out`` of the copy's rows of ``table`` whose ``columns`` hold a key of
    :data:`_KEYS_TABLE` (:func:`_handed_keys`): each key column compared as the copy's read by key compares it
    (``FollowedTable.held_key``), its text cast to the column's type.

    :param table: the table
    :ptype table: FollowedTable
    :param columns: the key columns named, a leading part of the key in key order
    :ptype columns: Sequence[str]
    :param out: the columns read
    :ptype out: Sequence[str]
    :return: the statement, over ``t``, naming :data:`~threetears.core.cache.duckdb_binds.KEYS` for the keys
    :rtype: str
    """
    selected = ", ".join(f"t.{quote_identifier(column)}" for column in out)
    named = ", ".join(table.held_key(column, "t") for column in columns)
    cast = ", ".join(f"CAST(k{at + 1} AS {table.key_cast(column)})" for at, column in enumerate(columns))
    return (
        f"SELECT {selected} FROM {quote_identifier(table.name)} AS t "  # noqa: S608 - trusted identifiers
        f"WHERE ({named}) IN (SELECT {cast} FROM ({KEYS}))"
    )


def _handed_keys(keys: Any, columns: Sequence[str]) -> pa.Table:
    """``keys``' ``columns`` as :func:`_keyed_statement`'s handed keys: ``k1`` … ``kn``, as text (no buffer copied for
    a text column).

    :param keys: the keys
    :ptype keys: pyarrow.Table
    :param columns: the key columns, in key order
    :ptype columns: Sequence[str]
    :return: the keys
    :rtype: pa.Table
    """
    return pa.table({f"k{at + 1}": keys.column(column).cast(pa.string()) for at, column in enumerate(columns)})


def _joined(statement: str, columns: int) -> str:
    """:func:`_keyed_statement` over the keys handed as :data:`_KEYS_TABLE`."""
    names = ", ".join(f"k{at + 1}" for at in range(columns))
    return statement.replace(KEYS, f'SELECT {names} FROM "{_KEYS_TABLE}"')


def _sql_rows(connection: Any, args: Sequence[Any]) -> Any:
    """a job: SQL over the state, with scalar binds.

    :param connection: the state's connection
    :ptype connection: duckdb.DuckDBPyConnection
    :param args: the SQL, then its binds
    :ptype args: Sequence[Any]
    :return: its result
    :rtype: duckdb.DuckDBPyConnection
    """
    return connection.execute(args[0], list(args[1:]))


def keyed_job(
    table: FollowedTable, columns: Sequence[str], out: Sequence[str], keys: Any
) -> tuple[Any, tuple[Any, ...], Mapping[str, Any]]:
    """the job, its arguments and its handed tables that read ``out`` of the rows ``keys`` name by ``columns``
    (:func:`_keyed_statement`).

    :param table: the table
    :ptype table: FollowedTable
    :param columns: the key columns named
    :ptype columns: Sequence[str]
    :param out: the columns read
    :ptype out: Sequence[str]
    :param keys: the keys, ``columns`` as text
    :ptype keys: pyarrow.Table
    :return: the job, its arguments, its tables
    :rtype: tuple[Callable, tuple[Any, ...], Mapping[str, pyarrow.Table]]
    """
    statement = _joined(_keyed_statement(table, columns, out), len(columns))
    return _sql_rows, (statement,), {_KEYS_TABLE: _handed_keys(keys, columns)}


def _fingerprinted(connection: Any, args: Sequence[Any]) -> pa.Table:
    """a job: the fingerprint of the rows another job answers (``volume_fulls.rows_fingerprint``), taken where they
    are read, so the rows never cross.

    :param connection: the state's connection
    :ptype connection: duckdb.DuckDBPyConnection
    :param args: the job, its arguments, the columns' fingerprint kinds
    :ptype args: Sequence[Any]
    :return: one row, its fingerprint
    :rtype: pa.Table
    """
    job, job_args, kinds = args
    rows = _table_of(job(connection, job_args))
    return pa.table({_FINGERPRINT: pa.array([rows_fingerprint(rows, kinds)], pa.string())})


def _ipc_bytes(rows: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    with pa_ipc.new_stream(sink, rows.schema) as writer:
        writer.write_table(rows)
    return bytes(sink.getvalue().to_pybytes())


def _spelled(table: FollowedTable, columns: Sequence[str], keys: Any) -> pa.Table:
    """``keys``' ``columns``, an instant key column's values in the copy's one stored spelling
    (``arrow_keys.stored_instants``, as the copy keeps its rows'), so a key in any ISO spelling finds its row by
    value. Arrow's kernels alone.

    :param table: the table
    :ptype table: FollowedTable
    :param columns: the key columns named
    :ptype columns: Sequence[str]
    :param keys: the keys, as text
    :ptype keys: pyarrow.Table
    :return: the keys
    :rtype: pa.Table
    """
    named = keys.select(list(columns))
    instants = table.instant_keys
    return pa.table(
        {
            column: stored_instants(named.column(column)) if column in instants else named.column(column)
            for column in columns
        }
    )


class _QueriedPinned:
    """a version of a shared copy, leased, with its mark (:meth:`CopyQueries.current`): its jobs run in a query
    worker over it."""

    def __init__(self, queries: CopyQueries, version: CopyVersion) -> None:
        self._queries = queries
        self._version = version

    @property
    def mark(self) -> int | None:
        return self._version.mark

    @property
    def version(self) -> str:
        """the version held.

        :return: its name
        :rtype: str
        """
        return self._version.name

    async def run_job(
        self,
        job: Callable[[Any, Sequence[Any]], Any],
        args: Sequence[Any] = (),
        *,
        tables: Mapping[str, Any] | None = None,
    ) -> Any:
        return await _answered(self._queries, job, args, tables, self._version.name)

    async def fingerprint(
        self,
        job: Callable[[Any, Sequence[Any]], Any],
        args: Sequence[Any],
        kinds: tuple[tuple[str, str], ...],
        *,
        tables: Mapping[str, Any] | None = None,
    ) -> str:
        answer = await _answered(self._queries, _fingerprinted, (job, tuple(args), kinds), tables, self._version.name)
        found: str = answer.column(_FINGERPRINT)[0].as_py()
        return found

    async def close(self) -> None:
        await self._version.aclose()


async def _answered(
    queries: CopyQueries,
    job: Callable[[Any, Sequence[Any]], Any],
    args: Sequence[Any],
    tables: Mapping[str, Any] | None,
    version: str | None,
) -> pa.Table:
    """a job's answer in a query worker, its table kept mapped (its files pinned until its last buffer goes) and the
    answer closed."""
    answer = await queries.run_job(job, args, tables=tables, version=version)
    table = answer.table
    await answer.aclose()
    return table


class QueriedReads:
    """the follower's reads over a copy published as shared Arrow: lookups and jobs in the query workers of
    ``queries``, the current version leased with its mark; no DuckDB and no cursor in the serving process.

    :param queries: the copy's query pool, over the store its versions are published in
    :ptype queries: CopyQueries
    """

    def __init__(self, queries: CopyQueries) -> None:
        self._queries = queries

    async def current(self) -> PinnedCopy:
        return _QueriedPinned(self._queries, await self._queries.current())

    async def rows_of_keys(self, table: FollowedTable, columns: Sequence[str], keys: Any, out: Sequence[str]) -> Any:
        spelled = await asyncio.to_thread(_spelled, table, columns, keys)
        # a client waits on a pass's keys: given the next free worker ahead of a checkpoint's or a drift check's job
        answer = await self._queries.lookup(table.name, spelled, columns, reader=True)
        found = answer.table.select(list(out))
        await answer.aclose()
        return found

    async def run_job(
        self,
        job: Callable[[Any, Sequence[Any]], Any],
        args: Sequence[Any] = (),
        *,
        tables: Mapping[str, Any] | None = None,
    ) -> Any:
        return await _answered(self._queries, job, args, tables, None)


def _close_contained(stack: ExitStack) -> None:
    """close a cursor's read. Blocking (IO): run it on a thread.

    :param stack: the stack holding the read open
    :ptype stack: ExitStack
    :return: nothing
    :rtype: None
    """
    try:
        stack.close()
    except Exception as exc:  # noqa: BLE001 - the read's state is let go either way; the copy is unaffected
        contain(exc, log=log, what="a copy read's cursor could not be closed")


def _close_landed(opening: asyncio.Future[tuple[ExitStack, Any, int | None]]) -> None:
    """close, on a thread, the read a cancelled caller was opening, once its open lands: never left open.

    :param opening: the open, done
    :ptype opening: asyncio.Future[tuple[ExitStack, Any, int | None]]
    :return: nothing
    :rtype: None
    """
    if opening.cancelled() or opening.exception() is not None:
        return  # nothing opened: a failed open closes what it opened itself
    asyncio.get_running_loop().run_in_executor(None, _close_contained, opening.result()[0])


def _on_cursor(
    cursor: Any, job: Callable[[Any, Sequence[Any]], Any], args: Sequence[Any], tables: Mapping[str, Any] | None
) -> pa.Table:
    """a job on a cursor, its tables registered while it runs. Blocking: run it on a thread."""
    named = tuple(() if tables is None else tables.items())
    for name, table in named:
        cursor.register(name, table)
    try:
        return _table_of(job(cursor, args))
    finally:
        for name, _ in named:
            cursor.unregister(name)


class _CursorPinned:
    """one applied state of the in-process copy: its snapshot cursor, held open until :meth:`close`."""

    def __init__(self, stack: ExitStack, cursor: Any, mark: int | None, workers: ProcessWorkers) -> None:
        self._stack = stack
        self._cursor = cursor
        self._mark = mark
        self._workers = workers

    @property
    def mark(self) -> int | None:
        return self._mark

    async def run_job(
        self,
        job: Callable[[Any, Sequence[Any]], Any],
        args: Sequence[Any] = (),
        *,
        tables: Mapping[str, Any] | None = None,
    ) -> Any:
        return await asyncio.to_thread(_on_cursor, self._cursor, job, args, tables)

    async def fingerprint(
        self,
        job: Callable[[Any, Sequence[Any]], Any],
        args: Sequence[Any],
        kinds: tuple[tuple[str, str], ...],
        *,
        tables: Mapping[str, Any] | None = None,
    ) -> str:
        rows = await self.run_job(job, args, tables=tables)
        found: str = await self._workers.run(fingerprint_ipc, await asyncio.to_thread(_ipc_bytes, rows), kinds)
        return found

    async def close(self) -> None:
        await asyncio.shield(asyncio.ensure_future(asyncio.to_thread(_close_contained, self._stack)))


class CursorReads:
    """the in-process copy's own reads, on a cursor of its snapshot on a thread (module docstring); a drift check's
    fingerprint in ``workers``.

    :param backend: the copy
    :ptype backend: CursorCopy
    :param workers: the worker processes a fingerprint is taken in
    :ptype workers: ProcessWorkers
    """

    def __init__(self, backend: CursorCopy, workers: ProcessWorkers) -> None:
        self._backend = backend
        self._workers = workers

    def _open(self) -> tuple[ExitStack, Any, int | None]:
        """open a snapshot and read its change's mark. Blocking (IO)."""
        stack = ExitStack()
        try:
            cursor = stack.enter_context(self._backend.read_snapshot())
            return stack, cursor, self._backend.staged_mark(cursor)
        except BaseException:
            stack.close()
            raise

    async def current(self) -> PinnedCopy:
        # shielded: a cancel while it opens lets the open finish and closes what it opened as it lands
        opening = asyncio.ensure_future(asyncio.to_thread(self._open))
        try:
            stack, cursor, mark = await asyncio.shield(opening)
        except asyncio.CancelledError:
            opening.add_done_callback(_close_landed)
            raise
        return _CursorPinned(stack, cursor, mark, self._workers)

    def _rows_of_keys(self, table: FollowedTable, columns: Sequence[str], keys: Any, out: Sequence[str]) -> Any:
        with self._backend.read_snapshot() as cursor:
            # the keys one Arrow table the statement semi-joins (run_joined): bound as Python lists, DuckDB 1.5 looks
            # for pandas once a value, 49 s of the 50 s staging 50,000 keys took (2026-11-23); 40 ms so
            return run_joined(
                cursor,
                _keyed_statement(table, columns, out),
                [keys.column(column) for column in columns],
                lambda result: result.fetch_arrow_table(),
            )

    async def rows_of_keys(self, table: FollowedTable, columns: Sequence[str], keys: Any, out: Sequence[str]) -> Any:
        return await asyncio.to_thread(self._rows_of_keys, table, columns, keys, out)

    def _run(
        self, job: Callable[[Any, Sequence[Any]], Any], args: Sequence[Any], tables: Mapping[str, Any] | None
    ) -> pa.Table:
        with self._backend.read_snapshot() as cursor:
            return _on_cursor(cursor, job, args, tables)

    async def run_job(
        self,
        job: Callable[[Any, Sequence[Any]], Any],
        args: Sequence[Any] = (),
        *,
        tables: Mapping[str, Any] | None = None,
    ) -> Any:
        return await asyncio.to_thread(self._run, job, args, tables)
