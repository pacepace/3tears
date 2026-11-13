"""a DuckDB or SQLite connection as the asyncpg-shaped reader the keyed helpers take: each statement on a thread.

The helpers in :mod:`threetears.core.keyed` write one SQL for PostgreSQL, YugabyteDB and DuckDB (``$n``
placeholders, ``unnest``, ``= ANY``, row comparisons) and await ``fetch`` / ``execute`` on what they are given.
DuckDB's own API is synchronous; :class:`ThreadedDuckDB` runs each statement on a cursor of its own in a worker thread
(the IO half of the process rule: DuckDB releases the interpreter lock while it runs), so the event loop never waits
on it. Rows come back as dicts; a write's answer is the status asyncpg would give (``DELETE 3``).

SQLite (an L1 cache) has no arrays and no ``::`` casts: :class:`ThreadedSQLite` numbers each ``$n`` as SQLite's own
``?n``, and a caller binds sets as a list (``AnyOf(..., as_list=True)``, ``KeyBatch(..., as_list=True)``) and pages
on ``Sort(..., cast=None)``. The helpers that read by ``by``, a list of keys, a bound or an id (``holds_any``,
``count_at_most``, ``counts_by``, ``read_keyed``, ``read_after``, ``keyset_page``, ``delete_before``,
``update_before``) then run there unchanged.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

from threetears.core.cache.duckdb import _ARROW_BATCH_ROWS, fetch_arrow
from threetears.core.cache.duckdb_binds import run_bound
from threetears.interfaces.keyed.filters import sqlite_numbered

try:
    import pyarrow as pa

    _PYARROW = True
except ImportError:  # NOSILENT: the snapshot extra is optional; fetch_arrow refuses without it, saying so
    _PYARROW = False

__all__ = ["InlineSQLite", "ThreadedDuckDB", "ThreadedSQLite"]


def _fetch(conn: Any, query: str, params: Sequence[Any]) -> list[dict[str, Any]]:
    cursor = conn.cursor()
    try:
        return run_bound(cursor, query, params, _dict_rows)
    finally:
        cursor.close()


def _dict_rows(result: Any) -> list[dict[str, Any]]:
    names = [column[0] for column in result.description or ()]
    return [dict(zip(names, row, strict=True)) for row in result.fetchall()]


def _execute(conn: Any, query: str, params: Sequence[Any]) -> str:
    cursor = conn.cursor()
    try:
        return run_bound(cursor, query, params, lambda result: _status(result, query))
    finally:
        cursor.close()


def _status(result: Any, query: str) -> str:
    verb = query.lstrip().split(None, 1)[0].upper()
    answered = result.fetchall() if result.description else []
    if verb == "CREATE" and answered:
        # ``CREATE TABLE ... AS`` answers its rows' count, as asyncpg spells it
        return f"SELECT {int(answered[0][0])}"
    touched = int(answered[0][0]) if answered and verb in {"INSERT", "UPDATE", "DELETE"} else 0
    return f"INSERT 0 {touched}" if verb == "INSERT" else f"{verb} {touched}"


def duckdb_arrow(
    conn: Any, query: str, params: Sequence[Any], types: Mapping[str, Any] | None, *, size: int | None = None
) -> Any:
    """a DuckDB SELECT's rows as one Arrow table typed by ``types``, never one Python object a row: DuckDB's own
    Arrow answer, its columns taken in ``types``' order and cast to theirs (the table
    :func:`~threetears.core.backends.arrow_reply.l3_fetch_arrow` answers for any other L3); with no ``types``,
    DuckDB's own answer as it is, its columns and types the statement's.

    :param conn: the DuckDB connection or cursor the statement runs on
    :ptype conn: duckdb.DuckDBPyConnection
    :param query: the SELECT, ``$n`` placeholders
    :ptype query: str
    :param params: its parameters
    :ptype params: Sequence[Any]
    :param types: column name -> its Arrow type, every column selected, in order; None for DuckDB's own
    :ptype types: Mapping[str, pyarrow.DataType] | None
    :param size: the most rows one Arrow batch is sized for (:func:`threetears.core.cache.duckdb.fetch_arrow`): the
        rows the read holds at most, so a small read never carries a million-row batch's buffers; DuckDB's own
        default when None
    :ptype size: int | None
    :return: the rows
    :rtype: pyarrow.Table
    :raises ImportError: without pyarrow
    :raises ValueError: when the statement selects other columns than ``types`` names
    """
    if not _PYARROW:
        raise ImportError("a DuckDB read as Arrow needs pyarrow: install 3tears[snapshot]")
    rows = _ARROW_BATCH_ROWS if size is None else size
    answered = run_bound(conn, query, params, lambda result: fetch_arrow(result, rows))
    if types is None:
        return answered
    if sorted(answered.column_names) != sorted(types):
        raise ValueError(f"the statement selects {answered.column_names}, the types name {list(types)}")
    return answered.select(list(types)).cast(pa.schema([pa.field(name, kind) for name, kind in types.items()]))


def _cursor_arrow(
    conn: Any, query: str, params: Sequence[Any], types: Mapping[str, Any] | None, size: int | None
) -> Any:
    cursor = conn.cursor()
    try:
        return duckdb_arrow(cursor, query, params, types, size=size)
    finally:
        cursor.close()


class ThreadedDuckDB:
    """a DuckDB connection read and written through ``fetch`` / ``execute``, each statement on a thread.

    :param conn: a ``duckdb.DuckDBPyConnection``; each statement runs on a cursor of its own
    :ptype conn: duckdb.DuckDBPyConnection
    """

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    async def fetch(self, query: str, /, *params: Any) -> list[dict[str, Any]]:
        """the statement's rows, as dicts.

        :param query: the statement, ``$n`` placeholders
        :ptype query: str
        :param params: its parameters
        :ptype params: Any
        :return: the rows
        :rtype: list[dict[str, Any]]
        """
        return await asyncio.to_thread(_fetch, self._conn, query, params)

    async def fetch_arrow(
        self, query: str, /, *params: Any, types: Mapping[str, Any] | None, size: int | None = None
    ) -> Any:
        """the statement's rows as one Arrow table typed by ``types`` (:func:`duckdb_arrow`), on a thread.

        :param query: the SELECT, ``$n`` placeholders
        :ptype query: str
        :param params: its parameters
        :ptype params: Any
        :param types: column name -> its Arrow type, every column selected, in order; None for DuckDB's own
        :ptype types: Mapping[str, pyarrow.DataType] | None
        :param size: the most rows one Arrow batch is sized for, as :func:`duckdb_arrow` takes it
        :ptype size: int | None
        :return: the rows
        :rtype: pyarrow.Table
        """
        return await asyncio.to_thread(_cursor_arrow, self._conn, query, params, types, size)

    async def execute(self, query: str, /, *params: Any) -> str:
        """run a write and answer its status as asyncpg spells it (``DELETE 3``).

        :param query: the statement, ``$n`` placeholders
        :ptype query: str
        :param params: its parameters
        :ptype params: Any
        :return: the status
        :rtype: str
        """
        return await asyncio.to_thread(_execute, self._conn, query, params)


def _sqlite_fetch(conn: Any, query: str, params: Sequence[Any]) -> list[dict[str, Any]]:
    cursor = conn.execute(sqlite_numbered(query), list(params))
    try:
        names = [column[0] for column in cursor.description or ()]
        return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]
    finally:
        cursor.close()


def _sqlite_execute(conn: Any, query: str, params: Sequence[Any]) -> str:
    cursor = conn.execute(sqlite_numbered(query), list(params))
    try:
        verb = query.lstrip().split(None, 1)[0].upper()
        touched = max(cursor.rowcount, 0)
        return f"INSERT 0 {touched}" if verb == "INSERT" else f"{verb} {touched}"
    finally:
        cursor.close()


class ThreadedSQLite:
    """a SQLite connection read and written through ``fetch`` / ``execute``, each statement on a thread.

    :param conn: a ``sqlite3.Connection`` opened with ``check_same_thread=False``, its writes committed by its owner
    :ptype conn: sqlite3.Connection
    """

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    async def fetch(self, query: str, /, *params: Any) -> list[dict[str, Any]]:
        """the statement's rows, as dicts.

        :param query: the statement, ``$n`` placeholders
        :ptype query: str
        :param params: its parameters
        :ptype params: Any
        :return: the rows
        :rtype: list[dict[str, Any]]
        """
        return await asyncio.to_thread(_sqlite_fetch, self._conn, query, params)

    async def execute(self, query: str, /, *params: Any) -> str:
        """run a write and answer its status as asyncpg spells it (``DELETE 3``).

        :param query: the statement, ``$n`` placeholders
        :ptype query: str
        :param params: its parameters
        :ptype params: Any
        :return: the status
        :rtype: str
        """
        return await asyncio.to_thread(_sqlite_execute, self._conn, query, params)


class InlineSQLite:
    """a SQLite connection read and written through ``fetch`` / ``execute``, each statement run at once on the
    caller's thread: for synchronous code that already holds the cache's write path (an L1's cap trim inside its
    write), driven by :func:`~threetears.core.keyed.run_inline`.

    .. code-block:: python

        db = InlineSQLite(l1)  # an L1 SQLiteBackend, or a sqlite3.Connection
        run_inline(keyed.delete_oldest(db, "turn_dedup", count=over, key="key", order="date_updated"))

    :param conn: a ``sqlite3.Connection``, or a backend whose ``get_connection()`` answers one (each statement asks
        for it, so a backend that rebinds its connection is followed)
    :ptype conn: Any
    """

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def _connection(self) -> Any:
        found = getattr(self._conn, "get_connection", None)
        return found() if found is not None else self._conn

    async def fetch(self, query: str, /, *params: Any) -> list[dict[str, Any]]:
        """the statement's rows, as dicts.

        :param query: the statement, ``$n`` placeholders
        :ptype query: str
        :param params: its parameters
        :ptype params: Any
        :return: the rows
        :rtype: list[dict[str, Any]]
        """
        return _sqlite_fetch(self._connection(), query, params)

    async def execute(self, query: str, /, *params: Any) -> str:
        """run a write and answer its status as asyncpg spells it (``DELETE 3``).

        :param query: the statement, ``$n`` placeholders
        :ptype query: str
        :param params: its parameters
        :ptype params: Any
        :return: the status
        :rtype: str
        """
        return _sqlite_execute(self._connection(), query, params)
