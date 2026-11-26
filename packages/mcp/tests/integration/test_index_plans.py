"""integration: each mcp index serves the keyed read its purpose names.

The no-whole-reads rule (:mod:`threetears.enforcement.no_whole_reads`) keys a read by an index only when the index
names that read as its ``purpose``; these are the ``plan_test`` each names. Each seeds its table past the size where
the planner would rather scan, runs the purpose's own read through a :class:`~threetears.core.testing.plans.PlanRecorder`
(``EXPLAIN ANALYZE`` of each statement it sends), and asks the plan whether it found its rows through the index by an
index condition (:func:`~threetears.core.testing.plans.plan_uses_index`).
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner
from threetears.core.ddl.row_epoch import next_epoch
from threetears.core.testing.plans import PlanRecorder, plan_uses_index
from threetears.mcp import McpToolGrantCollection
from threetears.mcp.migrations import register as register_mcp

pytestmark = pytest.mark.integration

#: the grants a live page keeps: its plan reads about its own rows, never the revoked ones before them
_LIVE_PAGE = 50


class _Store:
    """The DataStore surface the runner drives, over one connection."""

    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn

    async def execute(self, sql: str, *params: Any) -> str:
        status: str = await self._conn.execute(sql, *params)
        return status

    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        return [dict(r) for r in await self._conn.fetch(sql, *params)]


@pytest.fixture
async def conn(db_container: str) -> AsyncIterator[asyncpg.Connection]:
    """A connection on a fresh schema the mcp package's migrations built."""
    schema = f"mcpplan_{uuid.uuid4().hex[:12]}"
    connection = await asyncpg.connect(db_container)
    try:
        await connection.execute(f'CREATE SCHEMA "{schema}"')
        await connection.execute(f'SET search_path TO "{schema}"')
        runner = MigrationRunner()
        register_mcp(runner)
        await runner.apply_for_platform_schema(_Store(connection))
        yield connection
    finally:
        await connection.execute(f'DROP SCHEMA "{schema}" CASCADE')
        await connection.close()


async def test_grants_live_page(conn: asyncpg.Connection) -> None:
    """Flips if an operator's page of grants (McpToolGrantCollection.list_grants_page) stops reading
    mcp_tool_grants_live_by_id (mcp v004): through the primary key it walks the revoked grants, the first 2,000 by id
    here, before its first live one."""
    await conn.execute(
        "INSERT INTO mcp_tool_grants (grant_id, principal_type, principal_id, tool_name, permission, epoch)"
        " SELECT gen_random_uuid(), 'user', gen_random_uuid(), 'tool' || g, 'allow', $1"
        " FROM generate_series(1, 5000) g",
        next_epoch(),
    )
    await conn.execute(
        "UPDATE mcp_tool_grants SET gone = true WHERE grant_id IN (SELECT grant_id FROM mcp_tool_grants"
        " ORDER BY grant_id LIMIT 2000)"
    )
    await conn.execute("ANALYZE mcp_tool_grants")
    recorder = PlanRecorder(conn)
    grants = McpToolGrantCollection(registry=CollectionRegistry(), config=DefaultCoreConfig())  # type: ignore[call-arg]
    grants.l3_pool = recorder  # type: ignore[assignment]
    page = await grants.list_grants_page(limit=_LIVE_PAGE)
    assert len(page) == _LIVE_PAGE
    (plan,) = recorder.plans
    assert plan_uses_index(plan, "mcp_tool_grants_live_by_id"), json.dumps(plan.plan)[:2500]
    assert plan.rows_scanned <= 2 * (_LIVE_PAGE + 1), json.dumps(plan.plan)[:2500]
