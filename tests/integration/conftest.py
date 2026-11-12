"""one migrated database per engine for the use-the-helper rewrite proof (``test_use_the_helper_rewrites.py``).

The rewrite tool's generated test (``python -m threetears.enforcement.no_whole_reads.rewrite test-module``) holds every
rewritten statement to its helper call on a database holding every rewritten table, through two fixtures: ``conn``
(PostgreSQL) and ``yugabyte_conn`` (YugabyteDB). A session's database is one engine (``db_container``, the
``THREETEARS_TEST_POSTGRES_URL`` a run points at), so each fixture serves the engine the session's server is and skips
on the other: a PostgreSQL run proves the PostgreSQL half, a YugabyteDB run the YugabyteDB half.

The schema is built once a session, by each package's own migrations: the agent-scope packages (conversations, memory,
skills, workspace, langgraph) and the platform-scope ones (core's coordination tables, epoch, mcp,
scheduled-jobs). The datasources tables are created by
the hub, not by a 3tears migration, so they are built from their declared :class:`TableSchema` (the hub's source too).
The schema is dropped when the session ends.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import asyncpg
import pytest
from sqlalchemy import MetaData
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex, CreateTable

from threetears.agent.memory.migrations import register as register_memory
from threetears.agent.skills.migrations import register as register_skills
from threetears.agent.workspace.migrations import register as register_workspace
from threetears.conversations.migrations import register as register_conversations
from threetears.core.coordination.migrations import register as register_coordination
from threetears.core.data.migrations import MigrationRunner
from threetears.datasources.collections import (
    CapabilitySourceCollection,
    DataSourceColumnCollection,
    DataSourceRelationCollection,
    DataSourceSchemaDigestCollection,
    DataSourceTableCollection,
    TableTemplateCollection,
)
from threetears.epoch.migrations import register as register_epoch
from threetears.langgraph.migrations import register as register_langgraph
from threetears.mcp.migrations import register as register_mcp
from threetears.scheduled_jobs.migrations import register as register_scheduled_jobs

#: the tables the hub creates from their declarations (no 3tears migration builds them)
_DECLARED = (
    CapabilitySourceCollection.schema,
    DataSourceTableCollection.schema,
    DataSourceColumnCollection.schema,
    DataSourceSchemaDigestCollection.schema,
    DataSourceRelationCollection.schema,
    TableTemplateCollection.schema,
)

#: what marks a server's version string as YugabyteDB's
_YUGABYTE_MARK = "-YB-"

_HERE = Path(__file__).resolve().parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """mark every test under this directory ``integration`` (the generated module carries no marker of its own)."""
    for item in items:
        if _HERE in Path(str(item.path)).resolve().parents:
            item.add_marker(pytest.mark.integration)


class _Store:
    """the migration runner's store over one connection."""

    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn

    async def execute(self, sql: str, *args: Any) -> str:
        return str(await self._conn.execute(sql, *args))

    async def query(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        return [dict(row) for row in await self._conn.fetch(sql, *args)]


async def _build(url: str, schema: str) -> bool:
    """migrate ``schema`` with every rewritten table; whether the server is YugabyteDB."""
    conn = await asyncpg.connect(url)
    try:
        yugabyte = _YUGABYTE_MARK in str(await conn.fetchval("SELECT version()"))
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(f'SET search_path TO "{schema}", public')
        agent = MigrationRunner()
        register_conversations(agent)
        register_memory(agent)
        register_skills(agent)
        register_workspace(agent)
        register_langgraph(agent)
        await agent.apply_for_agent_schema(_Store(conn))  # type: ignore[arg-type]
        platform = MigrationRunner()
        register_coordination(platform)
        register_epoch(platform)
        register_mcp(platform)
        register_scheduled_jobs(platform)
        await platform.apply_for_platform_schema(_Store(conn))  # type: ignore[arg-type]
        metadata = MetaData()
        for declared in _DECLARED:
            table = declared.to_sqlalchemy_table(metadata)
            await conn.execute(str(CreateTable(table).compile(dialect=postgresql.dialect())))
            for index in table.indexes:
                await conn.execute(str(CreateIndex(index).compile(dialect=postgresql.dialect())))
        return yugabyte
    finally:
        await conn.close()


async def _drop(url: str, schema: str) -> None:
    conn = await asyncpg.connect(url)
    try:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
    finally:
        await conn.close()


@pytest.fixture(scope="session")
def _rewrites_schema(db_container: str) -> Iterator[tuple[str, str, bool]]:
    """the session's migrated schema: (url, schema, whether the server is YugabyteDB); built and dropped in their own
    event loops, so every test's connection is its own."""
    schema = f"rewrites_{uuid.uuid4().hex[:12]}"
    yugabyte = asyncio.run(_build(db_container, schema))
    try:
        yield db_container, schema, yugabyte
    finally:
        asyncio.run(_drop(db_container, schema))


async def _connect(url: str, schema: str) -> asyncpg.Connection:
    conn = await asyncpg.connect(url)
    await conn.execute(f'SET search_path TO "{schema}", public')
    return conn


@pytest.fixture
async def conn(_rewrites_schema: tuple[str, str, bool]) -> AsyncIterator[asyncpg.Connection]:
    """a superuser connection to the migrated PostgreSQL schema; skipped when the session's server is YugabyteDB."""
    url, schema, yugabyte = _rewrites_schema
    if yugabyte:
        pytest.skip("the session's server is YugabyteDB: the PostgreSQL half runs on a PostgreSQL session")
    connection = await _connect(url, schema)
    try:
        yield connection
    finally:
        await connection.close()


@pytest.fixture
async def yugabyte_conn(_rewrites_schema: tuple[str, str, bool]) -> AsyncIterator[asyncpg.Connection]:
    """a superuser connection to the migrated YugabyteDB schema; skipped when the session's server is PostgreSQL."""
    url, schema, yugabyte = _rewrites_schema
    if not yugabyte:
        pytest.skip("the session's server is PostgreSQL: the YugabyteDB half runs on a YugabyteDB session")
    connection = await _connect(url, schema)
    try:
        yield connection
    finally:
        await connection.close()
