"""Integration-test fixtures for the channels package.

Mirrors :mod:`packages.agent.wake.tests.integration.conftest`: the
session-scoped ``db_container`` fixture comes from the canonical
harness in :mod:`threetears.core.testing.fixtures` (wired via the
workspace-root ``conftest.py`` ``pytest_plugins`` line); the per-test
``pg_schema`` fixture creates a fresh schema so each test starts
clean.

The channels package's webhook receiver integration tests need the
wake + skills + conversations schemas applied (the receiver delegates
to :func:`webhook_receive` which reads from ``webhook_subscriptions``
and writes to ``wake_fires``); the per-test ``_apply_schema`` helper
in ``test_webhook_e2e.py`` registers all three migration packs
against the schema this fixture creates.

No ``__init__.py`` under ``tests/integration/`` (pytest is run with
``--import-mode=importlib`` so the conftest module name is the
rootdir-relative path); test files use the ``from .conftest import
AsyncpgStore`` relative import pattern the agent-wake package uses,
which works without an ``__init__.py`` under importlib mode.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest


@pytest.fixture(scope="module")
def pg_url(db_container: str) -> str:
    """Alias for :func:`threetears.core.testing.fixtures.db_container`.

    :param db_container: canonical session-scoped DB URL
    :ptype db_container: str
    :return: asyncpg-compatible PostgreSQL connection URL
    :rtype: str
    """
    return db_container


class AsyncpgStore:
    """``DataStore``-shape wrapper over an asyncpg connection.

    Mirrors the helper in :mod:`packages.agent.wake.tests.integration.conftest`
    so the migration runner has its ``execute`` + ``query`` surface
    when called against a raw asyncpg connection. Shared via
    ``from .conftest import AsyncpgStore`` from integration test
    modules in this package (the wake package uses the same pattern).
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


@pytest.fixture
async def pg_schema(pg_url: str) -> AsyncIterator[tuple[str, str]]:
    """Create a fresh schema per test and yield ``(pg_url, schema_name)``.

    The schema is dropped on teardown so each test gets a clean slate.

    :param pg_url: testcontainer URL
    :ptype pg_url: str
    :return: tuple of (pg url, fresh schema name)
    :rtype: tuple[str, str]
    """
    schema = f"ch_it_{id(object())}".lower().replace("-", "_")
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
