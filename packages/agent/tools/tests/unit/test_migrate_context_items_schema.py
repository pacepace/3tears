"""
``migrate_context_items_schema`` runs its DDL under the database-wide DDL lock.

it renames and adds columns outside a registered migration, at startup, so it
takes the same lock every migration run takes: on the one connection that runs
the ALTERs, before the first and released after the last.
"""

from __future__ import annotations

from types import TracebackType
from typing import Any

from threetears.core.testing.migrations import uncontended_ddl_lock_rows

from threetears.agent.tools.collections import migrate_context_items_schema


class _Transaction:
    """a no-op ``conn.transaction()`` context manager."""

    async def __aenter__(self) -> None:
        """open nothing."""

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """
        close nothing.

        :param exc_type: exception type, ignored
        :ptype exc_type: type[BaseException] | None
        :param exc: exception, ignored
        :ptype exc: BaseException | None
        :param tb: traceback, ignored
        :ptype tb: TracebackType | None
        """


# parity-with: threetears.core.data.migrations.session.SqlConnection
class _FakeConnection:
    """one recording session that answers the DDL lock as an uncontended database would."""

    def __init__(self) -> None:
        """start with nothing recorded."""
        self.statements: list[str] = []

    async def execute(self, query: str, *args: Any) -> str:
        """
        record a statement.

        :param query: SQL text
        :ptype query: str
        :param args: positional parameters
        :ptype args: Any
        :return: status tag
        :rtype: str
        """
        self.statements.append(query)
        return "EXECUTE"

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        """
        record a query and answer the lock's statements.

        :param query: SQL text
        :ptype query: str
        :param args: positional parameters
        :ptype args: Any
        :return: rows
        :rtype: list[Any]
        """
        self.statements.append(query)
        rows = uncontended_ddl_lock_rows(query)
        return rows if rows is not None else []

    def transaction(self) -> _Transaction:
        """
        open a transaction.

        :return: a no-op transaction
        :rtype: _Transaction
        """
        return _Transaction()


class _Acquire:
    """``async with pool.acquire() as conn``."""

    def __init__(self, connection: _FakeConnection) -> None:
        """
        remember the connection to hand out.

        :param connection: the connection
        :ptype connection: _FakeConnection
        """
        self._connection = connection

    async def __aenter__(self) -> _FakeConnection:
        """
        hand out the connection.

        :return: the connection
        :rtype: _FakeConnection
        """
        return self._connection

    async def __aexit__(self, *exc_info: object) -> None:
        """
        take it back.

        :param exc_info: the exception triple, ignored
        :ptype exc_info: object
        """


# parity-exempt: the two pool calls migrate_context_items_schema makes -- a column probe and acquire()
class _FakePool:
    """a pool whose probe reports the legacy column set."""

    def __init__(self) -> None:
        """own one connection."""
        self.connection = _FakeConnection()

    async def fetch(self, query: str, *args: Any) -> list[dict[str, str]]:
        """
        answer the column probe with the legacy columns.

        :param query: SQL text
        :ptype query: str
        :param args: positional parameters
        :ptype args: Any
        :return: legacy column rows
        :rtype: list[dict[str, str]]
        """
        return [{"column_name": name} for name in ("id", "summary", "value")]

    def acquire(self) -> _Acquire:
        """
        hand out the one connection.

        :return: async context manager yielding it
        :rtype: _Acquire
        """
        return _Acquire(self.connection)


async def test_the_alters_run_under_the_ddl_lock_on_their_own_connection() -> None:
    """try-lock before the first ALTER, unlock after the last, all on one connection."""
    pool = _FakePool()

    assert await migrate_context_items_schema(pool) is True

    statements = pool.connection.statements
    alters = [i for i, sql in enumerate(statements) if sql.startswith("ALTER TABLE")]
    take = next(i for i, sql in enumerate(statements) if "pg_try_advisory_lock" in sql)
    give = next(i for i, sql in enumerate(statements) if "pg_advisory_unlock" in sql)
    assert alters
    assert take < min(alters)
    assert max(alters) < give
