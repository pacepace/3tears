"""integration test for the agent-tools-platform v001 migration.

agent-tools-eligibility shard 01 (TE-04 / TE-10 / TE-11): real
Postgres via testcontainers to prove:

- the migration applies cleanly against a pre-existing
  ``namespaces`` table (mirrors how the deploying app actually
  hosts the table -- created by the platform DDL, then extended by
  this 3tears migration),
- the two new BOOLEAN columns land with the documented defaults,
- replay is a no-op (``ADD COLUMN IF NOT EXISTS`` semantics),
- pre-existing rows pick up the DB-side defaults without a
  backfill (backwards compatibility for TE-08).

requires docker; gated by ``pytest.mark.integration``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from threetears.agent.tools.platform_migrations import (
    PACKAGE_NAME,
    register,
)
from threetears.core.data.migrations import MigrationRunner

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------
# DataStore-shape wrapper for the canonical migration runner
# ---------------------------------------------------------------------


# parity-with: threetears.core.data.store.DataStore
class _AsyncpgStore:
    """DataStore-shape wrapper -- the subset every 3tears integration
    suite uses to drive :class:`MigrationRunner` over an asyncpg
    connection bound to the target schema."""

    def __init__(self, conn: asyncpg.Connection) -> None:
        """capture the connection.

        :param conn: connection with search_path set
        :ptype conn: asyncpg.Connection
        :return: nothing
        :rtype: None
        """
        self._conn = conn

    async def execute(self, sql: str, *params: Any) -> str:
        """proxy execute.

        :param sql: SQL statement
        :ptype sql: str
        :param params: positional parameters
        :ptype params: Any
        :return: asyncpg status string
        :rtype: str
        """
        result: str = await self._conn.execute(sql, *params)
        return result

    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        """proxy fetch -> dict rows.

        :param sql: SQL query
        :ptype sql: str
        :param params: positional parameters
        :ptype params: Any
        :return: list of row dicts
        :rtype: list[dict[str, Any]]
        """
        rows = await self._conn.fetch(sql, *params)
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------
# Fixture: fresh schema with a pre-existing platform-shaped
# ``namespaces`` table that the agent-tools platform migration
# extends. We do NOT run the canonical platform DDL here -- the
# minimal column set the eligibility migration needs is the bare
# table; the canonical platform Alembic adds many more columns
# orthogonal to this shard.
# ---------------------------------------------------------------------


_CREATE_PLATFORM_NAMESPACES_SQL = """
CREATE TABLE IF NOT EXISTS namespaces (
    row_scope TEXT NOT NULL DEFAULT 'platform',
    namespace_id UUID NOT NULL,
    name TEXT NOT NULL,
    namespace_type TEXT NOT NULL,
    owner_agent_id UUID,
    customer_id UUID,
    schema_name TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    date_created TIMESTAMPTZ NOT NULL DEFAULT now(),
    date_updated TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (row_scope, namespace_id)
)
"""


@pytest.fixture
async def pg_schema(db_container: str) -> AsyncIterator[tuple[str, str]]:
    """fresh schema with a minimal ``namespaces`` table pre-created.

    yields ``(url, schema)``. teardown drops the schema cascade.

    :param db_container: shared Postgres testcontainer DSN
    :ptype db_container: str
    :return: async iterator yielding the connection URL + schema name
    :rtype: AsyncIterator[tuple[str, str]]
    """
    schema = f"elig_it_{id(object())}".lower().replace("-", "_")
    conn = await asyncpg.connect(db_container)
    try:
        await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
        await conn.execute(f'SET search_path TO "{schema}"')
        await conn.execute(_CREATE_PLATFORM_NAMESPACES_SQL)
    finally:
        await conn.close()
    yield (db_container, schema)
    conn = await asyncpg.connect(db_container)
    try:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
    finally:
        await conn.close()


class TestPlatformMigrationAddsColumns:
    """v001 lands ``tool_eligible`` + ``skill_eligible`` on the table."""

    async def test_migration_applies_columns_with_defaults(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """fresh apply: both columns appear with the documented defaults."""
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}"')
            runner = MigrationRunner()
            register(runner)
            store = _AsyncpgStore(conn)
            count = await runner.apply_for_platform_schema(store)  # type: ignore[arg-type]
            assert count == 2, f"expected v001 + v002 to apply, applied {count}"
            cols = await conn.fetch(
                "SELECT column_name, data_type, is_nullable, column_default "
                "FROM information_schema.columns "
                "WHERE table_schema = $1 AND table_name = 'namespaces' "
                "AND column_name IN ('tool_eligible', 'skill_eligible')",
                schema,
            )
            assert len(cols) == 2
            by_name = {row["column_name"]: row for row in cols}
            assert by_name["tool_eligible"]["data_type"] == "boolean"
            assert by_name["tool_eligible"]["is_nullable"] == "NO"
            assert "true" in (by_name["tool_eligible"]["column_default"] or "").lower()
            assert by_name["skill_eligible"]["data_type"] == "boolean"
            assert by_name["skill_eligible"]["is_nullable"] == "NO"
            assert "false" in (by_name["skill_eligible"]["column_default"] or "").lower()
        finally:
            await conn.close()

    async def test_replay_is_noop(self, pg_schema: tuple[str, str]) -> None:
        """running v001 twice is a no-op -- ``ADD COLUMN IF NOT EXISTS``."""
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}"')
            runner = MigrationRunner()
            register(runner)
            store = _AsyncpgStore(conn)
            first = await runner.apply_for_platform_schema(store)  # type: ignore[arg-type]
            assert first == 2
            # second apply with a fresh runner: the bookkeeping table
            # remembers the prior apply so the runner does nothing.
            runner_two = MigrationRunner()
            register(runner_two)
            second = await runner_two.apply_for_platform_schema(store)  # type: ignore[arg-type]
            assert second == 0

            # forcibly re-execute the raw SQL -- ``ADD COLUMN IF NOT
            # EXISTS`` should ride through without raising.
            await conn.execute(
                "ALTER TABLE namespaces ADD COLUMN IF NOT EXISTS tool_eligible BOOLEAN NOT NULL DEFAULT TRUE"
            )
            await conn.execute(
                "ALTER TABLE namespaces ADD COLUMN IF NOT EXISTS skill_eligible BOOLEAN NOT NULL DEFAULT FALSE"
            )
            await conn.execute(
                "ALTER TABLE namespaces ADD COLUMN IF NOT EXISTS face_platform_tool BOOLEAN NOT NULL DEFAULT TRUE"
            )
            await conn.execute(
                "ALTER TABLE namespaces ADD COLUMN IF NOT EXISTS face_api BOOLEAN NOT NULL DEFAULT FALSE"
            )
            await conn.execute(
                "ALTER TABLE namespaces ADD COLUMN IF NOT EXISTS face_mcp BOOLEAN NOT NULL DEFAULT FALSE"
            )
        finally:
            await conn.close()

    async def test_v002_applies_face_columns_with_defaults(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """gu-task-02b: v002 lands the three face columns with defaults."""
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}"')
            runner = MigrationRunner()
            register(runner)
            store = _AsyncpgStore(conn)
            await runner.apply_for_platform_schema(store)  # type: ignore[arg-type]
            cols = await conn.fetch(
                "SELECT column_name, data_type, is_nullable, column_default "
                "FROM information_schema.columns "
                "WHERE table_schema = $1 AND table_name = 'namespaces' "
                "AND column_name IN ('face_api', 'face_mcp', 'face_platform_tool')",
                schema,
            )
            assert len(cols) == 3
            by_name = {row["column_name"]: row for row in cols}
            for name in ("face_api", "face_mcp", "face_platform_tool"):
                assert by_name[name]["data_type"] == "boolean"
                assert by_name[name]["is_nullable"] == "NO"
            assert "true" in (by_name["face_platform_tool"]["column_default"] or "").lower()
            assert "false" in (by_name["face_api"]["column_default"] or "").lower()
            assert "false" in (by_name["face_mcp"]["column_default"] or "").lower()
        finally:
            await conn.close()

    async def test_pre_existing_rows_pick_up_defaults(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """rows inserted before the migration backfill via DB defaults."""
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}"')
            # insert a tool-type row BEFORE the migration adds the
            # columns; rely on the platform DDL the fixture pre-created.
            await conn.execute(
                "INSERT INTO namespaces (row_scope, namespace_id, name, namespace_type) "
                "VALUES ('platform', gen_random_uuid(), 'tools.legacy.1-0', 'tool')",
            )
            runner = MigrationRunner()
            register(runner)
            store = _AsyncpgStore(conn)
            count = await runner.apply_for_platform_schema(store)  # type: ignore[arg-type]
            assert count == 2
            row = await conn.fetchrow(
                "SELECT tool_eligible, skill_eligible, face_platform_tool, face_api, face_mcp "
                "FROM namespaces WHERE name = 'tools.legacy.1-0'",
            )
            assert row is not None
            assert row["tool_eligible"] is True
            assert row["skill_eligible"] is False
            assert row["face_platform_tool"] is True
            assert row["face_api"] is False
            assert row["face_mcp"] is False
        finally:
            await conn.close()
