"""Integration-test fixtures for agent-skills.

Mirrors :mod:`packages.agent.memory.tests.integration.conftest`: the
session-scoped ``db_container`` fixture comes from the canonical
harness in :mod:`threetears.core.testing.fixtures` (wired via the
workspace-root ``conftest.py`` ``pytest_plugins`` line); the per-test
``pg_schema`` fixture creates a fresh schema so each test starts
clean.

The skills package does not exercise pgvector, so the default
``postgres:16`` image from the canonical fixture is sufficient. No
``db_image`` override is needed.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from threetears.agent.skills.migrations import register as register_skills
from threetears.conversations.migrations import register as register_conversations
from threetears.core.data.migrations import MigrationRunner

from tests.pg_template import migrated_copy


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

    Exposes :meth:`execute` and :meth:`query` matching what the
    migration runner expects.
    """

    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn

    async def execute(self, sql: str, *params: Any) -> str:
        """Execute SQL via the underlying connection.

        :param sql: SQL text
        :ptype sql: str
        :param params: positional parameters
        :ptype params: Any
        :return: asyncpg status tag
        :rtype: str
        """
        result: str = await self._conn.execute(sql, *params)
        return result

    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        """Fetch rows as a list of dicts.

        :param sql: SQL text
        :ptype sql: str
        :param params: positional parameters
        :ptype params: Any
        :return: list of row dicts
        :rtype: list[dict[str, Any]]
        """
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
    schema = f"sk_it_{uuid.uuid4().hex}".lower().replace("-", "_")
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


async def _migrate_skills(conn: asyncpg.Connection, _schema: str) -> None:
    """apply conversations + skills migrations, their held contract steps included, into the schema ``conn``'s
    search path names.

    :param conn: a connection whose ``search_path`` is the schema
    :ptype conn: asyncpg.Connection
    :param _schema: the schema's name (the search path already says it)
    :ptype _schema: str
    :return: None
    :rtype: None
    """
    runner = MigrationRunner()
    # the held contract step included: the tests read the tables as they are once v007 drops v002's index
    register_conversations(runner, held_contract=True)
    register_skills(runner, held_contract=True)
    await runner.apply_for_agent_schema(AsyncpgStore(conn))  # type: ignore[arg-type]


@pytest.fixture
async def applied_schema(pg_url: str) -> AsyncIterator[tuple[str, str]]:
    """a database of the test's own with conversations + skills migrated; yields ``(url, schema)``.

    the migrations run once per session, into a template database the test's database is copied from
    (:mod:`tests.pg_template`). tests about migrating take ``pg_schema`` and migrate it themselves.

    :param pg_url: the session's Postgres
    :ptype pg_url: str
    :yield: the test's database URL and the migrated schema's name
    :rtype: AsyncIterator[tuple[str, str]]
    """
    async with migrated_copy(pg_url, key="skills", migrate=_migrate_skills) as copied:
        yield copied
