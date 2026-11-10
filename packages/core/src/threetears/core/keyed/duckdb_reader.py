"""a DuckDB connection as the asyncpg-shaped reader the keyed helpers take: each statement on a thread.

The helpers in :mod:`threetears.core.keyed` write one SQL for PostgreSQL, YugabyteDB and DuckDB (``$n``
placeholders, ``unnest``, ``= ANY``, row comparisons) and await ``fetch`` / ``execute`` on what they are given.
DuckDB's own API is synchronous; :class:`ThreadedDuckDB` runs each statement on a cursor of its own in a worker thread
(the IO half of the process rule: DuckDB releases the interpreter lock while it runs), so the event loop never waits
on it. Rows come back as dicts; a write's answer is the status asyncpg would give (``DELETE 3``).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

__all__ = ["ThreadedDuckDB"]


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
