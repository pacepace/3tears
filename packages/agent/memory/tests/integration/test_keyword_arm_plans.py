"""integration: a hybrid search's keyword arm turns only the rows it keeps into text, not every row it ranks.

No index serves a full-text rank, so a keyword arm ranks every row its owner pins and its match leave, then keeps
``candidate_limit`` of them. Each hybrid search (memories, file contents, chunks) selects its rows' embeddings as text
(``t.embedding::text``: the pool decodes no ``vector``), and written beside the rank that cast ran on every ranked row:
for an owner of 20,000 memories (1024-dim) all matching the query, PostgreSQL spent 330 of the arm's 357 ms on it, and
the arm took 26 ms without it (MS1, scratchpad ms1/m3-pg.log). ``read_nearest_statement`` now selects a ranked read's
columns over the kept rows alone (WHY at it). This holds it on both engines by the plan, not the clock: every plan node
that outputs the embedding as text outputs at most the rows the arm keeps, for an owner whose matching rows are many
times that.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Mapping
from typing import Any
from unittest.mock import MagicMock

import asyncpg
import pytest
import pytest_asyncio
from threetears.agent.memory.authorize import MemoryAuthorizerDependencies
from threetears.agent.memory.collections import MediaContentCollection, MemoriesCollection, MemoryChunkCollection
from threetears.agent.memory.migrations import register as register_memory
from threetears.conversations.migrations import register as register_conversations
from threetears.core.collections import init_connection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner
from threetears.core.testing.captured import CapturedL3
from threetears.core.utils.pg_pool_kwargs import create_pool_with_startup_timeout

from .conftest import AsyncpgStore

pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="module")]

#: a narrow embedding: what is counted is the rows cast, not the cost of each
_DIM = 16
#: the owner's memories (and a file content and a chunk for each), every one matching the query
_OWNER = 2_000
#: the rows each arm keeps
_KEEP = 20
_QUERY = [1.0, *[0.0] * (_DIM - 1)]
_TEXT = "memory number"
_WEIGHTS = {"semantic": 1.0, "keyword": 1.0, "recency": 1.0}


class _World:
    def __init__(self, pool: asyncpg.Pool, agent: uuid.UUID, customer: uuid.UUID, owner: uuid.UUID) -> None:
        self.pool = pool
        self.agent = agent
        self.customer = customer
        self.owner = owner

    def collection(self, kind: type[Any], captured: CapturedL3) -> Any:
        registry = CollectionRegistry()
        registry.configure(l3_pool=captured)
        config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
        if kind is MemoriesCollection:
            return kind(registry, config, authorizer=MagicMock(spec=MemoryAuthorizerDependencies), embedding_dim=_DIM)
        return kind(registry, config, embedding_dim=_DIM)


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def world(pg_url: str) -> AsyncIterator[_World]:
    """a fresh schema: one owner's 2,000 memories, each with a file, its content and a chunk, all reading "memory
    number <n>"; analysed, and a platform pool on it."""
    schema = f"mem_kw_plans_{uuid.uuid4().hex[:12]}"
    conn = await asyncpg.connect(pg_url)
    try:
        await conn.execute('CREATE EXTENSION IF NOT EXISTS "vector"')
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(f'SET search_path TO "{schema}", public')
        runner = MigrationRunner()
        register_conversations(runner)
        register_memory(runner, embedding_dim=_DIM)
        await runner.apply_for_agent_schema(AsyncpgStore(conn))
        agent, customer, owner = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        vector = f"(SELECT array_agg(random())::real[] FROM generate_series(1, {_DIM}) WHERE g > 0)::public.vector"
        await conn.execute(
            "INSERT INTO memories (memory_id, agent_id, customer_id, user_id, conversation_id, type_memory, content,"
            " date_created, date_updated, tags, embedding)"
            " SELECT gen_random_uuid(), $1, $2, $3, gen_random_uuid(), 'fact', 'memory number ' || g,"
            f" now() - g * interval '1 hour', now(), '[]'::jsonb, {vector} FROM generate_series(1, {_OWNER}) g",
            agent,
            customer,
            owner,
        )
        await conn.execute(
            "INSERT INTO media (media_id, memory_id, agent_id, customer_id, user_id, date_created, date_updated)"
            " SELECT gen_random_uuid(), memory_id, agent_id, customer_id, user_id, date_created, now() FROM memories"
        )
        await conn.execute(
            "INSERT INTO media_content (content_id, media_id, agent_id, customer_id, user_id, content_type, content,"
            " date_created, embedding)"
            " SELECT gen_random_uuid(), m.media_id, m.agent_id, m.customer_id, m.user_id, 'text', mem.content,"
            " m.date_created, mem.embedding FROM media m JOIN memories mem"
            " ON mem.agent_id = m.agent_id AND mem.memory_id = m.memory_id"
        )
        await conn.execute(
            "INSERT INTO memory_chunks (chunk_id, memory_id, agent_id, customer_id, user_id, chunk_index, content,"
            " date_created, embedding, message_id_start)"
            " SELECT gen_random_uuid(), memory_id, agent_id, customer_id, user_id, 0, content, date_created,"
            " embedding, gen_random_uuid() FROM memories"
        )
        for table in ("memories", "media", "media_content", "memory_chunks"):
            await conn.execute(f"ANALYZE {table}")
        pool = await create_pool_with_startup_timeout(
            pg_url,
            pool_name="memory_keyword_plans",
            min_size=1,
            max_size=5,
            init=init_connection,
            server_settings={"search_path": f'"{schema}", public'},
        )
        try:
            yield _World(pool, agent, customer, owner)
        finally:
            await pool.close()
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


def _casting(node: Mapping[str, Any]) -> list[tuple[str, int]]:
    """each node of a plan that outputs an embedding as text, and how many rows it output.

    :param node: a plan node, as ``EXPLAIN (ANALYZE, VERBOSE, FORMAT JSON)`` gives it
    :ptype node: Mapping[str, Any]
    :return: (node type, actual rows times loops) for each such node
    :rtype: list[tuple[str, int]]
    """
    found: list[tuple[str, int]] = []
    if any("embedding)::text" in item or "embedding::text" in item for item in node.get("Output", ())):
        found.append((str(node.get("Node Type")), int(node.get("Actual Rows", 0)) * int(node.get("Actual Loops", 1))))
    for child in node.get("Plans", ()):
        found.extend(_casting(child))
    return found


async def _keyword_arm(world: _World, kind: type[Any]) -> tuple[str, tuple[Any, ...]]:
    """the keyword arm a ``kind`` hybrid search sends for the owner: the statement ranking by ``ts_rank_cd``."""
    captured = CapturedL3(real=world.pool)
    searched = world.collection(kind, captured)
    common = {
        "user_id": world.owner,
        "agent_id": world.agent,
        "customer_id": world.customer,
        "embedding": _QUERY,
        "user_text": _TEXT,
        "similarity_threshold": -1.0,
    }
    if kind is MemoryChunkCollection:
        await searched.hybrid_search(
            **common, candidate_k=_KEEP, chunk_signal_weights={"semantic": 1.0, "keyword": 1.0}
        )
    else:
        await searched.hybrid_search(
            **common, top_k=10, candidate_limit=_KEEP, recency_half_life_hours=24.0, signal_weights=_WEIGHTS
        )
    [arm] = [(sql, tuple(args)) for sql, args in captured.asked if "ts_rank_cd" in sql]
    return arm


@pytest.mark.parametrize("kind", [MemoriesCollection, MediaContentCollection, MemoryChunkCollection])
async def test_a_keyword_arm_casts_only_the_rows_it_keeps(world: _World, kind: type[Any]) -> None:
    """Flips if a hybrid search's keyword arm (memories, file contents, chunks) turns more embeddings into text than
    the rows it keeps, for an owner whose 2,000 rows all match, on PostgreSQL or YugabyteDB: the cast ran on every
    ranked row while the select list sat beside the rank."""
    sql, args = await _keyword_arm(world, kind)
    async with world.pool.acquire() as conn:
        raw = await conn.fetchval(f"EXPLAIN (ANALYZE, VERBOSE, FORMAT JSON) {sql}", *args)
    plan = (json.loads(raw) if isinstance(raw, str) else raw)[0]["Plan"]
    casting = _casting(plan)
    assert casting, (sql, plan)
    assert plan.get("Actual Rows", 0) == _KEEP, plan.get("Actual Rows")
    wide = [(node, rows) for node, rows in casting if rows > _KEEP]
    assert not wide, (wide, sql)
