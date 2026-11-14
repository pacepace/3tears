"""one migrated database per engine for the use-the-helper rewrite proof (``test_use_the_helper_rewrites.py``).

The rewrite tool's generated test (``python -m threetears.enforcement.no_whole_reads.rewrite test-module``) holds every
rewritten statement to its helper call on a database holding every rewritten table, through two fixtures: ``conn``
(PostgreSQL) and ``yugabyte_conn`` (YugabyteDB). A session's database is one engine (``db_container``, the
``THREETEARS_TEST_POSTGRES_URL`` a run points at), so each fixture serves the engine the session's server is and skips
on the other: a PostgreSQL run proves the PostgreSQL half, a YugabyteDB run the YugabyteDB half. This directory is in
the integration suite (``.testament.toml``'s ``[suites.integration]``, CI's integration job), so the pool and CI prove
the PostgreSQL half on every run; the YugabyteDB half runs where a YugabyteDB server is
(``THREETEARS_TEST_ENGINE=yugabyte`` with ``THREETEARS_TEST_POSTGRES_URL`` naming it). A run deselects the half of the
engine it does not name (PostgreSQL when it names none). Each module marks itself ``integration``.

The schema is built once a session, by each package's own migrations: the agent-scope packages (conversations, memory,
skills, tools, workspace, langgraph, wake, intention) and the platform-scope ones (acl, audit and the tools columns on its namespaces,
core's coordination tables, epoch, mcp, scheduled-jobs). The datasources tables are created by
the hub, not by a 3tears migration, so they are built from their declared :class:`TableSchema` (the hub's source too).
The schema is dropped when the session ends.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any

import asyncpg
import pytest
from sqlalchemy import MetaData
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex, CreateTable

from threetears.agent.acl.migrations import register as register_acl
from threetears.agent.audit.migrations import register as register_audit
from threetears.agent.intention.migrations import register as register_intention
from threetears.agent.memory.migrations import register as register_memory
from threetears.agent.skills.migrations import register as register_skills
from threetears.agent.tools.migrations import register as register_tools
from threetears.agent.tools.platform_migrations import register as register_tools_platform
from threetears.agent.wake.migrations import register as register_wake
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

#: the engine a run names (``threetears.core.testing.fixtures`` checks the server against it); unnamed, PostgreSQL
_ENGINE_ENV = "THREETEARS_TEST_ENGINE"

#: each engine's connection fixture
_ENGINE_FIXTURES = {"conn": "postgres", "yugabyte_conn": "yugabyte"}


def _engine_of(item: pytest.Item) -> str | None:
    """the engine a test proves: the one its connection fixture serves, named directly or as a fixture parameter;
    None for a test that needs no database."""
    names = [*getattr(item, "fixturenames", ())]
    callspec = getattr(item, "callspec", None)
    if callspec is not None:
        names += [value for value in callspec.params.values() if isinstance(value, str)]
    engines = {_ENGINE_FIXTURES[name] for name in names if name in _ENGINE_FIXTURES}
    return engines.pop() if len(engines) == 1 else None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """deselect the other engine's half: a run proves the engine it names (``THREETEARS_TEST_ENGINE``, PostgreSQL
    when unnamed), so the other half is not run at all rather than skipped (a skip fails the pool's integration
    suite, ``THREETEARS_REQUIRE_SERVICES=1``)."""
    engine = os.environ.get(_ENGINE_ENV) or "postgres"
    other = [item for item in items if _engine_of(item) not in (None, engine)]
    if other:
        config.hook.pytest_deselected(items=other)
        items[:] = [item for item in items if _engine_of(item) in (None, engine)]


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
        register_tools(agent)
        register_workspace(agent)
        register_langgraph(agent)
        register_wake(agent)
        register_intention(agent)
        await agent.apply_for_agent_schema(_Store(conn))  # type: ignore[arg-type]
        platform = MigrationRunner()
        register_acl(platform)
        register_audit(platform)
        register_tools_platform(platform, depends_on=("agent_acl",))
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
    # a vector column (pgvector's, or YugabyteDB's built-in one) has no asyncpg codec: the proof seeds and compares
    # it as its text, the form the memory code reads it in (``embedding::text``)
    vector = await conn.fetchval(
        "SELECT n.nspname FROM pg_type t JOIN pg_namespace n ON n.oid = t.typnamespace WHERE t.typname = 'vector'"
        " ORDER BY n.nspname = 'pg_catalog' DESC LIMIT 1"
    )
    if vector is not None:
        await conn.set_type_codec("vector", schema=vector, encoder=str, decoder=str, format="text")
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
