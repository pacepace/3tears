"""Integration: the HNSW index on ``intentions.embedding`` serves the dedup read of a heavy owner.

``ix_intentions_embedding_hnsw`` was built for the log-time dedup lookup (839e06ae4, "pgvector embedding for log-time
dedup"): :meth:`~threetears.agent.intention.collections.IntentionsCollection.find_similar_for_dedup`. A small owner's
open wants are few, and the planner sorts them exactly off ``idx_intentions_open_ranked``; an owner with many open
wants is where the planner walks the HNSW index instead, so that is the plan tested here, on a platform pool (its
sessions start with ``hnsw.iterative_scan = relaxed_order``), with a small owner beside it still getting every row.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

import threetears.agent.intention.collections as intention_collections
from threetears.agent.intention.collections import IntentionsCollection
from threetears.agent.intention.migrations import register as register_intention
from threetears.conversations.migrations import register as register_conversations
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner
from threetears.core.keyed import read_nearest
from threetears.core.testing.plans import explain_statement, plan_uses_index
from threetears.core.utils.pg_pool_kwargs import create_pool_with_startup_timeout
from threetears.interfaces.keyed import read_nearest_statement

from .conftest import AsyncpgStore

pytestmark = pytest.mark.integration

_DIM = 16
_HEAVY = 20_000
_SMALL = 50
_K = 10


def _wants(count: int, agent_id: uuid.UUID, user_id: uuid.UUID) -> str:
    """an INSERT of ``count`` open wants of one owner, each with a random ``_DIM``-wide embedding (seeded first)."""
    vector = "('[' || array_to_string(ARRAY(SELECT random() + g * 0 FROM generate_series(1, %d)), ',') || ']')" % _DIM
    return (
        "INSERT INTO intentions (intention_id, agent_id, customer_id, user_id, status, content, embedding, salience, "
        "date_created, date_updated) "
        f"SELECT gen_random_uuid(), '{agent_id}'::uuid, '{agent_id}'::uuid, '{user_id}'::uuid, 'open', 'want ' || g, "
        f"{vector}::public.vector, 0.5, now(), now() FROM generate_series(1, {count}) AS g"
    )


@pytest.fixture
async def owners(pg_schema: tuple[str, str]) -> AsyncIterator[tuple[asyncpg.Pool, uuid.UUID, uuid.UUID, uuid.UUID]]:
    """a migrated schema with one agent's heavy owner (20,000 open wants) and small owner (50), analysed; a platform
    pool on it, the agent, the heavy owner and the small one."""
    url, schema = pg_schema
    runner = MigrationRunner()
    register_conversations(runner)
    register_intention(runner, embedding_dim=_DIM)
    agent_id, heavy, small = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    conn = await asyncpg.connect(url)
    try:
        await conn.execute(f'SET search_path TO "{schema}", public')
        await runner.apply_for_agent_schema(AsyncpgStore(conn))
        await conn.execute("SELECT setseed(0.3)")
        await conn.execute(_wants(_HEAVY, agent_id, heavy))
        await conn.execute(_wants(_SMALL, agent_id, small))
        await conn.execute("ANALYZE intentions")
    finally:
        await conn.close()
    pool = await create_pool_with_startup_timeout(
        url,
        pool_name="intentions_plans",
        min_size=1,
        max_size=2,
        server_settings={"search_path": f'"{schema}", public'},
    )
    try:
        yield pool, agent_id, heavy, small
    finally:
        await pool.close()


def _collection(pool: asyncpg.Pool) -> IntentionsCollection:
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return IntentionsCollection(registry=registry, config=config, embedding_dim=_DIM)


async def test_ix_intentions_embedding_hnsw_serves_a_heavy_owners_dedup_read(
    owners: tuple[asyncpg.Pool, uuid.UUID, uuid.UUID, uuid.UUID], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if find_similar_for_dedup's read of an owner with 20,000 open wants stops walking
    ix_intentions_embedding_hnsw (on PostgreSQL and YugabyteDB), or a 50-want owner beside it gets fewer than the 10
    nearest wants it asked for."""
    pool, agent_id, heavy, small = owners
    sent: list[dict[str, Any]] = []

    async def spy(conn: Any, table: str, **kwargs: Any) -> list[Any]:
        sent.append({"table": table, **kwargs})
        return await read_nearest(conn, table, **kwargs)

    monkeypatch.setattr(intention_collections, "read_nearest", spy)
    query = [1.0, *[0.0] * (_DIM - 1)]
    collection = _collection(pool)
    for owner in (heavy, small):
        found = await collection.find_similar_for_dedup(
            user_id=owner, agent_id=agent_id, embedding=query, top_k=_K, threshold=-1.0
        )
        assert len(found) == _K
    heavy_read = sent[0]
    sql, params = read_nearest_statement(
        heavy_read["table"],
        by=heavy_read["by"],
        nearest=heavy_read["nearest"],
        limit=heavy_read["limit"],
        columns=heavy_read["columns"],
        where=heavy_read["where"],
    )
    async with pool.acquire() as conn:
        plan = await explain_statement(conn, sql, *params)
    assert plan_uses_index(plan, "ix_intentions_embedding_hnsw"), plan.plan
