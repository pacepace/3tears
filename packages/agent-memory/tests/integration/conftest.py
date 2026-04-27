"""integration-test fixtures for agent-memory reconciliation.

post-test-harness-task-01 the bare testcontainer setup
(``PostgresContainer`` lifecycle + docker-skip + asyncpg URL
normalisation) lives in :mod:`threetears.core.testing.fixtures`
as the canonical ``db_container`` fixture. this module's
``pytest_plugins`` line below pulls it in; the local ``pg_url``
fixture is just a per-package alias that pins ``pgvector/pgvector
:pg16`` (instead of the canonical default ``postgres:16``) via the
``db_image`` parametrize hook.

per-test ``pg_schema`` fixture stays here because it is
agent-memory-specific (creates the ``vector`` extension + a fresh
schema per test, drops on teardown).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

# canonical testcontainer fixtures from the central harness;
# a single ``pytest_plugins`` line pulls in ``db_container`` /
# ``db_image`` / ``nats_container`` / ``nats_jetstream`` -- this
# package only uses ``db_container`` (via its ``pg_url`` alias).
pytest_plugins = ["threetears.core.testing.fixtures"]


@pytest.fixture(scope="session")
def db_image() -> str:
    """override the canonical ``db_image`` to pin pgvector/pg16.

    agent-memory exercises the ``vector`` extension; the canonical
    default ``postgres:16`` does not ship pgvector. overriding
    here at session scope means every test in the package picks
    up the pgvector image without per-test indirect parametrize
    boilerplate.

    :return: docker image reference
    :rtype: str
    """
    return "pgvector/pgvector:pg16"


@pytest.fixture(scope="module")
def pg_url(db_container: str) -> str:
    """alias for :func:`threetears.core.testing.fixtures.db_container`.

    every existing agent-memory integration test takes ``pg_url``
    as the testcontainer URL. this one-line alias keeps those
    sites working without renaming each call site to
    ``db_container``. new tests should pull ``db_container``
    directly.

    :param db_container: canonical session-scoped DB URL
    :ptype db_container: str
    :return: asyncpg-compatible PostgreSQL connection URL
    :rtype: str
    """
    return db_container


class AsyncpgStore:
    """
    DataStore-shape wrapper over an asyncpg connection.

    exposes :meth:`execute` and :meth:`query` matching what the
    migration runner expects.

    :param conn: asyncpg connection with search_path pre-set
    :ptype conn: asyncpg.Connection
    """

    def __init__(self, conn: asyncpg.Connection) -> None:
        """
        initialize wrapper.

        :param conn: asyncpg connection with search_path set
        :ptype conn: asyncpg.Connection
        """
        self._conn = conn

    async def execute(self, sql: str, *params: Any) -> str:
        """
        execute SQL via the underlying connection.

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
        """
        fetch rows as a list of dicts.

        :param sql: SQL text
        :ptype sql: str
        :param params: positional parameters
        :ptype params: Any
        :return: list of row dicts
        :rtype: list[dict[str, Any]]
        """
        rows = await self._conn.fetch(sql, *params)
        result = [dict(r) for r in rows]
        return result


@pytest.fixture
async def pg_schema(pg_url: str) -> AsyncIterator[tuple[str, str]]:
    """
    create a fresh schema per test and yield ``(pg_url, schema_name)``.

    the schema is dropped on teardown so each test gets a clean slate.
    also installs the ``vector`` extension at the database level (shared
    across schemas).

    :return: tuple of (pg url, fresh schema name)
    :rtype: tuple[str, str]
    """
    schema = f"mem_it_{id(object())}".lower().replace("-", "_")
    conn = await asyncpg.connect(pg_url)
    try:
        await conn.execute('CREATE EXTENSION IF NOT EXISTS "vector"')
        await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
    finally:
        await conn.close()
    yield (pg_url, schema)
    conn = await asyncpg.connect(pg_url)
    try:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
    finally:
        await conn.close()
