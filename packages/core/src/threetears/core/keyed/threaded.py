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
import re
from collections.abc import Mapping, Sequence
from typing import Any, Final

try:
    import pyarrow as pa

    _PYARROW = True
except ImportError:  # NOSILENT: the snapshot extra is optional; fetch_arrow refuses without it, saying so
    _PYARROW = False

__all__ = ["ThreadedDuckDB", "ThreadedSQLite"]

#: a positional placeholder, as the helpers write it
_PLACEHOLDER: Final[re.Pattern[str]] = re.compile(r"\$(\d+)")


def _fetch(conn: Any, query: str, params: Sequence[Any]) -> list[dict[str, Any]]:
    cursor = conn.cursor()
    try:
        result = cursor.execute(query, list(params))
        names = [column[0] for column in result.description or ()]
        return [dict(zip(names, row, strict=True)) for row in result.fetchall()]
    finally:
        cursor.close()


def _execute(conn: Any, query: str, params: Sequence[Any]) -> str:
    cursor = conn.cursor()
    try:
        result = cursor.execute(query, list(params))
        verb = query.lstrip().split(None, 1)[0].upper()
        answered = result.fetchall() if result.description else []
        touched = int(answered[0][0]) if answered and verb in {"INSERT", "UPDATE", "DELETE"} else 0
        return f"INSERT 0 {touched}" if verb == "INSERT" else f"{verb} {touched}"
    finally:
        cursor.close()


def duckdb_arrow(conn: Any, query: str, params: Sequence[Any], types: Mapping[str, Any] | None) -> Any:
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
    :return: the rows
    :rtype: pyarrow.Table
    :raises ImportError: without pyarrow
    :raises ValueError: when the statement selects other columns than ``types`` names
    """
    if not _PYARROW:
        raise ImportError("a DuckDB read as Arrow needs pyarrow: install 3tears[snapshot]")
    answered = conn.execute(query, list(params)).fetch_arrow_table()
    if types is None:
        return answered
    if sorted(answered.column_names) != sorted(types):
        raise ValueError(f"the statement selects {answered.column_names}, the types name {list(types)}")
    return answered.select(list(types)).cast(pa.schema([pa.field(name, kind) for name, kind in types.items()]))


def _cursor_arrow(conn: Any, query: str, params: Sequence[Any], types: Mapping[str, Any] | None) -> Any:
    cursor = conn.cursor()
    try:
        return duckdb_arrow(cursor, query, params, types)
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

    async def fetch_arrow(self, query: str, /, *params: Any, types: Mapping[str, Any] | None) -> Any:
        """the statement's rows as one Arrow table typed by ``types`` (:func:`duckdb_arrow`), on a thread.

        :param query: the SELECT, ``$n`` placeholders
        :ptype query: str
        :param params: its parameters
        :ptype params: Any
        :param types: column name -> its Arrow type, every column selected, in order; None for DuckDB's own
        :ptype types: Mapping[str, pyarrow.DataType] | None
        :return: the rows
        :rtype: pyarrow.Table
        """
        return await asyncio.to_thread(_cursor_arrow, self._conn, query, params, types)

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
    cursor = conn.execute(_PLACEHOLDER.sub(r"?\1", query), list(params))
    try:
        names = [column[0] for column in cursor.description or ()]
        return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]
    finally:
        cursor.close()


def _sqlite_execute(conn: Any, query: str, params: Sequence[Any]) -> str:
    cursor = conn.execute(_PLACEHOLDER.sub(r"?\1", query), list(params))
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
