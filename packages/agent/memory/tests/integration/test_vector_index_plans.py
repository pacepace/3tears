"""integration: the vector and tag indexes of agent memory serve the reads of an owner too large to sort exactly.

``test_index_plans`` seeds twenty memories an owner, and at that size every nearest read sorts the owner's rows
exactly off the owner index. The HNSW indexes (``ix_memories_embedding_hnsw``, ``ix_media_content_embedding``,
``ix_memory_chunks_embedding``) and the tags GIN (``idx_memories_tags``) are for the other end: one owner with 20,000
memories (and a file content and a chunk for each), where both engines walk the vector index nearest-first, and a
scoped search's rare tag (twenty of the owner's rows) is found by containment. Each vector test also reads a
50-memory owner beside it and asks for every row it wants: the platform pool's sessions start with
``hnsw.iterative_scan = relaxed_order`` (``pg_pool_kwargs``), so a filtered HNSW walk does not come back short.

Measured on this world (K1, 2026-11-15; PostgreSQL 16 + pgvector 0.8.7, YugabyteDB with vector 0.8.0-yb): the heavy
owner's dedup, search-tool and ambient (current and history pool) reads walked ix_memories_embedding_hnsw on both
engines, the file-content and chunk searches their HNSW indexes, and every ``tags @>`` scoped search idx_memories_tags.
find_same_event (its message pin) and hybrid_search_within_memory (one memory's chunks) never read a vector index on
either engine, so neither is a purpose. The small owner's reads stayed on the owner indexes on both.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any
from unittest.mock import MagicMock

import asyncpg
import pytest
import pytest_asyncio
from threetears.agent.memory.authorize import MemoryAuthorizerDependencies
from threetears.agent.memory.collections import MediaContentCollection, MemoriesCollection, MemoryChunkCollection
from threetears.agent.memory.migrations import register as register_memory
from threetears.agent.memory.retrieval_scope import RetrievalScope
from threetears.conversations.migrations import register as register_conversations
from threetears.core.collections import init_connection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner
from threetears.core.testing.captured import CapturedL3
from threetears.core.testing.plans import StatementPlan, explain_statement, plan_uses_index
from threetears.core.utils.pg_pool_kwargs import create_pool_with_startup_timeout

from .conftest import AsyncpgStore

#: one seeded schema for the module (each plan is explained in a transaction rolled back): 20,050 memories, files
#: and chunks through three HNSW indexes are built once
pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="module")]

#: a narrow embedding: the plan turns on the owner's size, not the vector's width
_DIM = 16
#: the heavy owner's memories: past the size where both engines walk the vector index rather than sort the owner
_HEAVY = 20_000
#: the small owner's memories, sorted exactly off the owner index
_SMALL = 50
#: the rows each read asks for
_K = 10
#: a tag twenty of the heavy owner's memories carry (each memory carries ``t<n % 1000>``)
_RARE_TAG = "t7"
#: the query vector
_QUERY = [1.0, *[0.0] * (_DIM - 1)]
_WEIGHTS = {"semantic": 1.0, "keyword": 1.0, "recency": 1.0}
_CHUNK_WEIGHTS = {"semantic": 1.0, "keyword": 1.0}


class _World:
    """the seeded schema's platform pool, its agent and customer, and the heavy and small owners."""

    def __init__(
        self, pool: asyncpg.Pool, agent: uuid.UUID, customer: uuid.UUID, heavy: uuid.UUID, small: uuid.UUID
    ) -> None:
        self.pool = pool
        self.agent = agent
        self.customer = customer
        self.heavy = heavy
        self.small = small

    def collection(self, kind: type[Any], captured: CapturedL3) -> Any:
        """``kind`` over ``captured``, its embeddings ``_DIM`` wide.

        :param kind: the collection class
        :ptype kind: type[Any]
        :param captured: the recording pool
        :ptype captured: CapturedL3
        :return: the collection
        :rtype: Any
        """
        registry = CollectionRegistry()
        registry.configure(l3_pool=captured)
        config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
        if kind is MemoriesCollection:
            return kind(registry, config, authorizer=MagicMock(spec=MemoryAuthorizerDependencies), embedding_dim=_DIM)
        return kind(registry, config, embedding_dim=_DIM)

    async def run(
        self, kind: type[Any], read: Callable[[Any], Awaitable[list[Any]]]
    ) -> tuple[int, list[StatementPlan]]:
        """run ``read`` on a ``kind`` collection over the platform pool, recording each statement it sends.

        :param kind: the collection class
        :ptype kind: type[Any]
        :param read: the read, given the collection
        :ptype read: Callable[[Any], Awaitable[list[Any]]]
        :return: the rows it answered, and the plan of each statement it sent, in order
        :rtype: tuple[int, list[StatementPlan]]
        """
        captured = CapturedL3(real=self.pool)
        rows = await read(self.collection(kind, captured))
        async with self.pool.acquire() as conn:
            plans = [await explain_statement(conn, sql, *args) for sql, args in captured.asked]
        return len(rows), plans


def _vector() -> str:
    """a random ``_DIM``-wide vector, one per seeded row (``g`` makes the sub-select correlated)."""
    return f"(SELECT array_agg(random())::real[] FROM generate_series(1, {_DIM}) WHERE g > 0)::public.vector"


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def world(pg_url: str) -> AsyncIterator[_World]:
    """a fresh schema with the memory tables built and seeded: one agent's heavy owner (20,000 memories) and small
    owner (50), a day apart (so each has memories older than ambient retrieval's recent week), each memory with a
    file, its content and a chunk, each tagged ``t<n % 1000>``; analysed, and a platform pool on it."""
    schema = f"mem_vec_plans_{uuid.uuid4().hex[:12]}"
    conn = await asyncpg.connect(pg_url)
    try:
        await conn.execute('CREATE EXTENSION IF NOT EXISTS "vector"')
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(f'SET search_path TO "{schema}", public')
        runner = MigrationRunner()
        register_conversations(runner)
        register_memory(runner, embedding_dim=_DIM)
        await runner.apply_for_agent_schema(AsyncpgStore(conn))
        agent, customer, heavy, small = uuid.uuid4(), uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        for user, count in ((heavy, _HEAVY), (small, _SMALL)):
            await conn.execute(
                "INSERT INTO memories (memory_id, agent_id, customer_id, user_id, conversation_id, type_memory,"
                " content, date_created, date_updated, tags, embedding)"
                " SELECT gen_random_uuid(), $1, $2, $3, gen_random_uuid(), 'fact', 'memory number ' || g,"
                " now() - g * interval '1 day', now(), jsonb_build_array('t' || g % 1000),"
                f" {_vector()} FROM generate_series(1, {count}) g",
                agent,
                customer,
                user,
            )
        await conn.execute(
            "INSERT INTO media (media_id, memory_id, agent_id, customer_id, user_id, date_created, date_updated)"
            " SELECT gen_random_uuid(), memory_id, agent_id, customer_id, user_id, date_created, now() FROM memories"
        )
        await conn.execute(
            "INSERT INTO media_content (content_id, media_id, agent_id, customer_id, user_id, content_type, content,"
            " date_created, embedding)"
            " SELECT gen_random_uuid(), m.media_id, m.agent_id, m.customer_id, m.user_id, 'text', 'content of a file',"
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
        # the platform pool: its sessions start with hnsw.iterative_scan = relaxed_order; a hybrid search sends its
        # arms at once, so it holds a connection for each and one for the plans
        pool = await create_pool_with_startup_timeout(
            pg_url,
            pool_name="memory_vector_plans",
            min_size=1,
            max_size=5,
            init=init_connection,
            server_settings={"search_path": f'"{schema}", public'},
        )
        try:
            yield _World(pool, agent, customer, heavy, small)
        finally:
            await pool.close()
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


def _shape(plan: StatementPlan) -> str:
    """the plan's nodes, one a line, each with the relation and index it reads: what a failing test shows.

    :param plan: the plan
    :ptype plan: StatementPlan
    :return: the outline
    :rtype: str
    """
    lines: list[str] = []

    def walk(node: Mapping[str, Any], depth: int) -> None:
        named = " ".join(f"{key}={node[key]}" for key in ("Relation Name", "Index Name") if key in node)
        lines.append(f"{'  ' * depth}{node.get('Node Type')} {named}")
        for child in node.get("Plans", ()):
            walk(child, depth + 1)

    walk(plan.plan, 0)
    return plan.sql[:200] + "\n" + "\n".join(lines)


def _nearest(plans: list[StatementPlan]) -> list[StatementPlan]:
    """the statements ordered by a vector distance: a hybrid search's vector arms, not its keyword arm.

    :param plans: a read's statements
    :ptype plans: list[StatementPlan]
    :return: those ordered by ``<=>``; at least one
    :rtype: list[StatementPlan]
    """
    found = [plan for plan in plans if "<=>" in plan.sql]
    assert found, [plan.sql[:120] for plan in plans]
    return found


async def test_ix_memories_embedding_hnsw_serves_a_heavy_owners_nearest_reads(world: _World) -> None:
    """Flips if a 20,000-memory owner's dedup lookup, search-tool read or ambient retrieval (current and history
    pool) stops walking ix_memories_embedding_hnsw (on PostgreSQL and YugabyteDB), or a 50-memory owner beside it gets
    fewer rows than it asked for."""
    plans: list[StatementPlan] = []
    for owner in (world.heavy, world.small):

        def dedup(memories: MemoriesCollection, owner: uuid.UUID = owner) -> Awaitable[list[Any]]:
            return memories.find_similar_for_dedup(
                user_id=owner, agent_id=world.agent, embedding=_QUERY, top_k=_K, threshold=-1.0
            )

        def semantic(memories: MemoriesCollection, owner: uuid.UUID = owner) -> Awaitable[list[Any]]:
            return memories.search_by_semantic(
                user_id=owner, agent_id=world.agent, embedding=_QUERY, max_results=_K, similarity_threshold=-1.0
            )

        def ambient(memories: MemoriesCollection, owner: uuid.UUID = owner) -> Awaitable[list[Any]]:
            return memories.hybrid_search(
                user_id=owner,
                agent_id=world.agent,
                customer_id=world.customer,
                embedding=_QUERY,
                user_text="memory number",
                top_k=_K,
                candidate_limit=20,
                similarity_threshold=-1.0,
                recency_half_life_hours=24.0,
                signal_weights=_WEIGHTS,
                recent_cap=5,
            )

        for read in (dedup, semantic, ambient):
            rows, sent = await world.run(MemoriesCollection, read)
            assert rows == _K, (read.__name__, owner == world.heavy, rows)
            if owner == world.heavy:
                plans.extend(_nearest(sent))
    # dedup, the search tool, and ambient retrieval's current and history pools
    assert len(plans) == 4, [plan.sql[:120] for plan in plans]
    missed = [_shape(plan) for plan in plans if not plan_uses_index(plan, "ix_memories_embedding_hnsw")]
    assert not missed, "\n\n".join(missed)


async def test_ix_media_content_embedding_serves_a_heavy_owners_nearest_reads(world: _World) -> None:
    """Flips if a 20,000-file owner's content search or hybrid search's vector arm stops walking
    ix_media_content_embedding (on PostgreSQL and YugabyteDB), or a 50-file owner beside it gets fewer rows than it
    asked for."""
    plans: list[StatementPlan] = []
    for owner in (world.heavy, world.small):

        def semantic(contents: MediaContentCollection, owner: uuid.UUID = owner) -> Awaitable[list[Any]]:
            return contents.search_by_semantic(
                user_id=owner, agent_id=world.agent, embedding=_QUERY, max_results=_K, similarity_threshold=-1.0
            )

        def hybrid(contents: MediaContentCollection, owner: uuid.UUID = owner) -> Awaitable[list[Any]]:
            return contents.hybrid_search(
                user_id=owner,
                agent_id=world.agent,
                customer_id=world.customer,
                embedding=_QUERY,
                user_text="content of",
                top_k=_K,
                candidate_limit=20,
                similarity_threshold=-1.0,
                recency_half_life_hours=24.0,
                signal_weights=_WEIGHTS,
            )

        for read in (semantic, hybrid):
            rows, sent = await world.run(MediaContentCollection, read)
            assert rows == _K, (read.__name__, owner == world.heavy, rows)
            if owner == world.heavy:
                plans.extend(_nearest(sent))
    missed = [_shape(plan) for plan in plans if not plan_uses_index(plan, "ix_media_content_embedding")]
    assert not missed, "\n\n".join(missed)


async def test_ix_memory_chunks_embedding_serves_a_heavy_owners_nearest_reads(world: _World) -> None:
    """Flips if a 20,000-chunk owner's chunk search or hybrid search's vector arm stops walking
    ix_memory_chunks_embedding (on PostgreSQL and YugabyteDB), or a 50-chunk owner beside it gets fewer rows than it
    asked for."""
    plans: list[StatementPlan] = []
    for owner in (world.heavy, world.small):

        def semantic(chunks: MemoryChunkCollection, owner: uuid.UUID = owner) -> Awaitable[list[Any]]:
            return chunks.search_by_semantic(
                user_id=owner, agent_id=world.agent, embedding=_QUERY, max_results=_K, similarity_threshold=-1.0
            )

        def hybrid(chunks: MemoryChunkCollection, owner: uuid.UUID = owner) -> Awaitable[list[Any]]:
            return chunks.hybrid_search(
                user_id=owner,
                agent_id=world.agent,
                customer_id=world.customer,
                embedding=_QUERY,
                user_text="memory number",
                candidate_k=_K,
                similarity_threshold=-1.0,
                chunk_signal_weights=_CHUNK_WEIGHTS,
            )

        rows, sent = await world.run(MemoryChunkCollection, semantic)
        assert rows == _K, ("semantic", owner == world.heavy, rows)
        if owner == world.heavy:
            plans.extend(_nearest(sent))
        # the hybrid search answers its vector and keyword arms' candidates merged: at least the vector arm's k
        rows, sent = await world.run(MemoryChunkCollection, hybrid)
        assert rows >= _K, ("hybrid", owner == world.heavy, rows)
        if owner == world.heavy:
            plans.extend(_nearest(sent))
    missed = [_shape(plan) for plan in plans if not plan_uses_index(plan, "ix_memory_chunks_embedding")]
    assert not missed, "\n\n".join(missed)


async def test_idx_memories_tags_serves_a_scoped_searchs_every_tag_test(world: _World) -> None:
    """Flips if a search scoped to a tag twenty of a 20,000-memory owner's memories carry (``tags @>``: the search
    tool's vector and keyword reads, ambient retrieval's vector and keyword arms) stops finding them through
    idx_memories_tags, on PostgreSQL and YugabyteDB."""
    scope = RetrievalScope(tags_all=(_RARE_TAG,))
    owner = world.heavy

    def semantic(memories: MemoriesCollection) -> Awaitable[list[Any]]:
        return memories.search_by_semantic(
            user_id=owner,
            agent_id=world.agent,
            embedding=_QUERY,
            max_results=_K,
            similarity_threshold=-1.0,
            scope=scope,
        )

    def keyword(memories: MemoriesCollection) -> Awaitable[list[Any]]:
        return memories.search_by_fts(
            user_id=owner, agent_id=world.agent, fts_text="memory", max_results=_K, scope=scope
        )

    def ambient(memories: MemoriesCollection) -> Awaitable[list[Any]]:
        return memories.hybrid_search(
            user_id=owner,
            agent_id=world.agent,
            customer_id=world.customer,
            embedding=_QUERY,
            user_text="memory number",
            top_k=_K,
            candidate_limit=20,
            similarity_threshold=-1.0,
            recency_half_life_hours=24.0,
            signal_weights=_WEIGHTS,
            scope=scope,
        )

    plans: list[StatementPlan] = []
    for read in (semantic, keyword, ambient):
        rows, sent = await world.run(MemoriesCollection, read)
        assert rows == _K, (read.__name__, rows)
        plans.extend(sent)
    # the search tool's two reads and ambient retrieval's vector and keyword arms
    assert len(plans) == 4, [plan.sql[:120] for plan in plans]
    missed = [_shape(plan) for plan in plans if not plan_uses_index(plan, "idx_memories_tags")]
    assert not missed, "\n\n".join(missed)
