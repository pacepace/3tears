"""Integration-test fixtures for 3tears-conversations.

Mirrors the agent-skills and agent-memory integration conftests: the
session-scoped ``db_container`` fixture comes from the canonical harness in
:mod:`threetears.core.testing.fixtures` (wired via the workspace-root
``conftest.py`` ``pytest_plugins`` line); the per-test ``pg_schema`` fixture
creates a fresh schema so each test starts clean.

The conversations migrations need no extension, so the default
``postgres:16`` image from the canonical fixture is sufficient.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import asyncpg
import pytest

__all__ = ["AsyncpgStore", "pg_schema", "pg_url"]


@pytest.fixture(scope="module")
def pg_url(db_container: str) -> str:
    """alias for :func:`threetears.core.testing.fixtures.db_container`.

    :param db_container: canonical session-scoped DB URL
    :ptype db_container: str
    :return: asyncpg-compatible PostgreSQL connection URL
    :rtype: str
    """
    return db_container


class AsyncpgStore:
    """``DataStore``-shaped wrapper over one asyncpg connection.

    exposes :meth:`execute` and :meth:`query`, the two calls the migration
    runner makes.
    """

    def __init__(self, conn: asyncpg.Connection) -> None:
        """wrap ``conn``.

        :param conn: live asyncpg connection whose search_path selects the schema
        :ptype conn: asyncpg.Connection
        """
        self.conn = conn

    async def execute(self, sql: str, *params: Any) -> str:
        """execute SQL on the wrapped connection.

        :param sql: SQL text
        :ptype sql: str
        :param params: positional parameters
        :ptype params: Any
        :return: asyncpg status tag
        :rtype: str
        """
        result: str = await self.conn.execute(sql, *params)
        return result

    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        """fetch rows as a list of dicts.

        :param sql: SQL text
        :ptype sql: str
        :param params: positional parameters
        :ptype params: Any
        :return: list of row dicts
        :rtype: list[dict[str, Any]]
        """
        rows = await self.conn.fetch(sql, *params)
        return [dict(r) for r in rows]


@pytest.fixture
async def pg_schema(pg_url: str) -> AsyncIterator[tuple[str, str]]:
    """create a fresh schema per test and yield ``(pg_url, schema_name)``.

    the schema is dropped on teardown so each test gets a clean slate.

    :param pg_url: testcontainer URL
    :ptype pg_url: str
    :return: tuple of (pg url, fresh schema name)
    :rtype: tuple[str, str]
    """
    schema = f"conv_it_{uuid4().hex}"
    conn = await asyncpg.connect(pg_url)
    try:
        await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
    finally:
        await conn.close()
    yield (pg_url, schema)
    conn = await asyncpg.connect(pg_url)
    try:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
    finally:
        await conn.close()
