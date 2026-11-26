"""The copy follower's own reads over its copy: keyed lookups, a version pinned with its mark, and jobs over it.

**Why** (``design-copy-worker-backend.md``, decision 4; inventory chunk 3T-7). The follower read its copy on DuckDB
cursors of the serving process: the rows a pass's keys hold (``_held_arrow``, ``_copy_rows_of_keys``), a checkpoint's
read of one applied state (``reading_marked``: a cursor and the engine slot it holds), its parts (a table whole, a
scope, a patched part) and a drift check's or a reload's rows of a state (``copy_rows_of_state``). CPU beside the event
loop, and on a pod's shared Arrow copy each such read held an engine slot. Here the follower reads through
:class:`CopyReads`, and over a copy published as shared Arrow (:class:`QueriedReads`) none of it opens a DuckDB
connection or a cursor in the serving process:

- **a keyed lookup** is :meth:`~threetears.core.collections.copy_queries.CopyQueries.lookup` (Arrow's hash kernels,
  in a query worker, however few the keys): keyed, never a scan by SQL;
- **the current version** is :meth:`~threetears.core.collections.copy_queries.CopyQueries.current`: a lease and the
  mark of the change it holds, read off its mark file on a thread, no cursor (:class:`PinnedCopy`);
- **a job** (a checkpoint's parts, a state's rows, a drift check's fingerprint) is
  :meth:`~threetears.core.collections.copy_queries.CopyQueries.run_job` over that version in a query worker, its
  answer shared Arrow (a checkpoint's parts come back encoded, one row a part), never pickled rows.

**No cursor read** (chunk 3T-8). The in-process copy's reads on a snapshot cursor went with the in-process DuckDB as a
follower's backend: every follower's copy is published as shared Arrow and read here, in query workers.
``tests/enforcement/test_no_cursor_read_of_the_copy.py`` keeps any serving module from importing one back.

**One state.** A :class:`PinnedCopy` reads one applied state for as long as it is open: a lease on the version (no
publish retires it meanwhile).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, Protocol

import pyarrow as pa

from threetears.core.cache.duckdb_binds import KEYS
from threetears.core.collections.copy_queries import CopyQueries, CopyVersion
from threetears.core.collections.schema_backed import rail_text
from threetears.core.collections.volume_fulls import rows_fingerprint
from threetears.core.sql_fragments import quote_identifier
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.core.collections.copy_follower import FollowedTable

__all__ = [
    "CopyReads",
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


def _key_casts(table: FollowedTable, columns: Sequence[str]) -> dict[str, str]:
    """the SQL type of each of ``columns`` the copy holds as text though its type is not text
    (:func:`~threetears.core.collections.schema_backed.rail_text`: a ``NUMERIC``, a UUID, an instant), which a lookup
    compares by value (``CopyQueries.lookup``'s ``casts``), as the copy's read by key compared them cast in SQL
    (``FollowedTable.held_key``).

    :param table: the table
    :ptype table: FollowedTable
    :param columns: the key columns named
    :ptype columns: Sequence[str]
    :return: column -> its SQL type
    :rtype: dict[str, str]
    """
    return {
        column: table.key_cast(column)
        for column in columns
        if table.key_cast(column) != "text" and rail_text(table.kind(column))
    }


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
        # a client waits on a pass's keys: given the next free worker ahead of a checkpoint's or a drift check's job.
        # Each key column compared as the copy holds it; a NUMERIC, a UUID or an instant held as text by value, spelled
        # one way on both sides in the worker (an instant's asked text in the copy's stored spelling)
        answer = await self._queries.lookup(table.name, keys, columns, reader=True, casts=_key_casts(table, columns))
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
