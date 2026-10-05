"""integration: the agent-audit package builds ``audit_events`` once, as the persisting consumer would."""

from __future__ import annotations

import uuid
from typing import Any

import asyncpg
import pytest

from threetears.agent.audit.migrations import register
from threetears.agent.audit.persist import AUDIT_EVENTS_DDL
from threetears.core.data.migrations import MigrationRunner

pytestmark = pytest.mark.integration


class _Store:
    """The DataStore surface the runner drives, over one connection."""

    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn

    async def execute(self, sql: str, *params: Any) -> str:
        return await self._conn.execute(sql, *params)

    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        return [dict(r) for r in await self._conn.fetch(sql, *params)]


async def test_the_package_builds_the_table_the_consumer_ensures_and_a_rerun_does_nothing(db_container: str) -> None:
    schema = f"audit_{uuid.uuid4().hex[:12]}"
    conn = await asyncpg.connect(db_container)
    try:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(f'SET search_path TO "{schema}"')
        runner = MigrationRunner()
        register(runner)

        assert await runner.apply_for_platform_schema(_Store(conn)) == 1  # type: ignore[arg-type]
        assert await runner.apply_for_platform_schema(_Store(conn)) == 0  # type: ignore[arg-type]
        indexes = {
            r["indexname"] for r in await conn.fetch("SELECT indexname FROM pg_indexes WHERE schemaname = $1", schema)
        }
        # the consumer's own ensure finds nothing to do on the package's table
        for statement in AUDIT_EVENTS_DDL:
            await conn.execute(statement)
    finally:
        await conn.execute(f'DROP SCHEMA "{schema}" CASCADE')
        await conn.close()

    assert {
        "idx_audit_events_time",
        "idx_audit_events_customer_time",
        "idx_audit_events_type",
        "idx_audit_events_actor",
    } <= indexes
