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

from threetears.core.collections.asyncpg_init import init_connection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner
from threetears.scheduled_jobs.collections import JobFireCollection, ScheduledJobCollection
from threetears.scheduled_jobs.migrations import register as register_scheduled_jobs


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


async def apply_schema(url: str, schema: str) -> asyncpg.Pool:
    """Apply the scheduled-jobs migration into ``schema`` and return a pool set up as a host's is.

    :param url: the Postgres URL
    :ptype url: str
    :param schema: the schema to migrate into
    :ptype schema: str
    :return: a pool whose search path is ``schema``
    :rtype: asyncpg.Pool
    """
    setup_conn = await asyncpg.connect(url)
    try:
        await setup_conn.execute(f'SET search_path TO "{schema}", public')
        runner = MigrationRunner()
        register_scheduled_jobs(runner)
        store = AsyncpgStore(setup_conn)
        await runner.apply_for_platform_schema(store)  # type: ignore[arg-type]
    finally:
        await setup_conn.close()
    pool = await asyncpg.create_pool(
        url,
        min_size=2,
        max_size=8,
        server_settings={"search_path": f"{schema}, public"},
        init=init_connection,
    )
    assert pool is not None
    return pool


def build_stores(pool: asyncpg.Pool) -> tuple[ScheduledJobCollection, JobFireCollection]:
    """Build both stores over ``pool``, with no L1 or L2.

    :param pool: the migrated pool
    :ptype pool: asyncpg.Pool
    :return: the job store and the fire store
    :rtype: tuple[ScheduledJobCollection, JobFireCollection]
    """
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    cfg = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return (
        ScheduledJobCollection(registry=registry, config=cfg, nats_client=None),
        JobFireCollection(registry=registry, config=cfg, nats_client=None),
    )
