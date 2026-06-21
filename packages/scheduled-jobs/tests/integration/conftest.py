"""Integration-test fixtures for scheduled-jobs.

Mirrors :mod:`packages.agent.wake.tests.integration.conftest`: the
session-scoped ``db_container`` fixture comes from the canonical harness
in :mod:`threetears.core.testing.fixtures` (wired via the workspace-root
``conftest.py`` ``pytest_plugins`` line); the per-test ``pg_schema``
fixture creates a fresh schema so each test starts clean.

The scheduled-jobs default store does not exercise pgvector, so the
default ``postgres:16`` image from the canonical fixture is sufficient.
No ``db_image`` override is needed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest


# parity-with: threetears.core.data.store.DataStore
class AsyncpgStore:
    """``DataStore``-shape wrapper over an asyncpg connection.

    Exposes :meth:`execute` and :meth:`query` matching what the migration
    runner expects. Same role agent-wake's integration conftest fills.
    """

    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn

    async def execute(self, sql: str, *params: Any) -> str:
        """Execute SQL via the underlying connection."""
        result: str = await self._conn.execute(sql, *params)
        return result

    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        """Fetch rows as a list of dicts."""
        rows = await self._conn.fetch(sql, *params)
        return [dict(r) for r in rows]


@pytest.fixture(scope="module")
def pg_url(db_container: str) -> str:
    """Alias for :func:`threetears.core.testing.fixtures.db_container`.

    :param db_container: canonical session-scoped DB URL
    :ptype db_container: str
    :return: asyncpg-compatible PostgreSQL connection URL
    :rtype: str
    """
    return db_container


@pytest.fixture
async def pg_schema(pg_url: str) -> AsyncIterator[tuple[str, str]]:
    """Create a fresh schema per test and yield ``(pg_url, schema_name)``.

    The schema is dropped on teardown so each test gets a clean slate.

    :param pg_url: testcontainer URL
    :ptype pg_url: str
    :return: tuple of (pg url, fresh schema name)
    :rtype: tuple[str, str]
    """
    schema = f"sj_it_{id(object())}".lower().replace("-", "_")
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
