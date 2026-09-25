"""
the one-connection store a migration runs on.

a migration run holds a SESSION-level advisory lock across every statement it
issues, and a session lock belongs to exactly one database connection: it is
released only by an unlock on that same connection, or when the connection's
session ends. so everything the runner does has to happen on one connection.

a pool cannot promise that. asyncpg's ``Pool.execute`` borrows a connection per
statement and, on returning it, runs the pool's reset query -- which begins
``SELECT pg_advisory_unlock_all()``. a lock taken through a pool is therefore
dropped the moment the statement that took it returns, and the run it was meant
to serialise proceeds unguarded; the unlock at the end lands on whichever
connection the pool hands out next and releases nothing.

:class:`MigrationSession` is the surface the runner and
:func:`~threetears.core.data.migrations.ddl_lock.database_ddl_lock` consume,
and :class:`ConnectionSession` is the concrete one for a caller holding a
plain connection: a thin wrapper over it that refuses anything shaped like a
pool. a :class:`~threetears.core.data.store.DataStore` is the other way in:
it pins itself to one connection (``DataStore.ddl_session``).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from threetears.core.data.migrations.errors import SessionRequiredError

__all__ = [
    "ConnectionSession",
    "MigrationSession",
    "SqlConnection",
]


@runtime_checkable
class MigrationSession(Protocol):
    """
    the statement surface a migration run consumes, bound to ONE database session.

    every statement the runner issues -- the lock, the bookkeeping, every
    migration body's DDL -- goes through one of these two methods, and all of
    them must reach the same connection. the protocol cannot express that, so
    the runner pins a pool-backed :class:`~threetears.core.data.store.DataStore`
    to one connection it acquires, and :class:`ConnectionSession` refuses a
    pool at construction. a caller that passes any other implementation
    promises it is one session; the lock's release reports loudly when it
    was not.
    """

    async def execute(self, sql: str, *params: Any) -> str:
        """
        execute ``sql`` on the session.

        :param sql: SQL statement text with ``$N`` placeholders
        :ptype sql: str
        :param params: positional parameter values
        :ptype params: Any
        :return: status tag from the engine
        :rtype: str
        """
        ...

    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        """
        run ``sql`` on the session and return its rows.

        :param sql: SQL query text with ``$N`` placeholders
        :ptype sql: str
        :param params: positional parameter values
        :ptype params: Any
        :return: rows as column-name-keyed dicts
        :rtype: list[dict[str, Any]]
        """
        ...


class SqlConnection(Protocol):
    """
    the asyncpg ``Connection`` shape :class:`ConnectionSession` wraps.

    an asyncpg ``Pool`` has these two methods too, which is exactly why
    :class:`ConnectionSession` checks for ``acquire`` rather than trusting the
    type: a pool satisfies this protocol and is still not one session.
    """

    async def execute(self, query: str, *args: Any) -> str:
        """
        execute ``query`` on the connection.

        :param query: SQL statement text
        :ptype query: str
        :param args: positional parameter values
        :ptype args: Any
        :return: status tag
        :rtype: str
        """
        ...

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        """
        fetch the rows ``query`` returns.

        :param query: SQL query text
        :ptype query: str
        :param args: positional parameter values
        :ptype args: Any
        :return: driver row objects supporting ``dict(row)``
        :rtype: list[Any]
        """
        ...


class ConnectionSession:
    """
    a :class:`MigrationSession` over exactly one database connection.

    the caller owns the connection: it opens it (or acquires it from a pool
    for the length of the run), sets its ``search_path``, and closes or
    releases it afterwards. this class only routes statements to it.

    :param connection: one connection, never a pool -- an asyncpg
        ``Connection``, or a connection acquired from a pool and held for the
        whole run
    :ptype connection: SqlConnection
    :raises SessionRequiredError: when ``connection`` has an ``acquire``
        method, which is what a pool (asyncpg's, or a 3tears ``L3Backend``)
        has and a single connection does not
    """

    def __init__(self, connection: SqlConnection) -> None:
        """
        wrap one connection, refusing a pool.

        :param connection: one connection, never a pool
        :ptype connection: SqlConnection
        :raises SessionRequiredError: when ``connection`` is shaped like a pool
        """
        if hasattr(connection, "acquire"):
            msg = (
                f"ConnectionSession needs one connection, and was given {type(connection).__name__}, which has "
                "acquire() and so hands each statement to whichever connection is free. a session-level lock "
                "taken that way is released as soon as the connection returns to the pool. acquire one "
                "connection and hold it for the whole run: async with pool.acquire() as conn: "
                "ConnectionSession(conn)"
            )
            raise SessionRequiredError(msg)
        self._connection = connection

    async def execute(self, sql: str, *params: Any) -> str:
        """
        execute ``sql`` on the wrapped connection.

        :param sql: SQL statement text with ``$N`` placeholders
        :ptype sql: str
        :param params: positional parameter values
        :ptype params: Any
        :return: status tag from the engine
        :rtype: str
        """
        result: str = await self._connection.execute(sql, *params)
        return result

    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        """
        run ``sql`` on the wrapped connection and return its rows as dicts.

        :param sql: SQL query text with ``$N`` placeholders
        :ptype sql: str
        :param params: positional parameter values
        :ptype params: Any
        :return: rows as column-name-keyed dicts
        :rtype: list[dict[str, Any]]
        """
        rows = await self._connection.fetch(sql, *params)
        # convert at border: asyncpg Records iterate values, not keys
        result = [dict(row) for row in rows]
        return result
