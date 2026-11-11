"""integration: each agent-memory index serves the keyed reads its purpose names.

The no-whole-reads rule (:mod:`threetears.enforcement.no_whole_reads`) keys a read by an index only when the index
names that read as its ``purpose``; these are the ``plan_test`` each names. Each seeds the tables past the size where
the planner would rather scan (three agents, thirty customers, three hundred users, 1024-wide embeddings), takes the
statement the code under test really writes (its collection run over a pool that records statements), and asks the
plan whether it found its rows through the index (:func:`threetears.core.testing.plans.plan_uses_index`).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import asyncpg
import pytest
import pytest_asyncio

from threetears.agent.memory.authorize import MemoryAuthorizerDependencies
from threetears.agent.memory.collections import (
    MediaContentCollection,
    MemoriesCollection,
    MemoryChunkCollection,
    MemoryConsolidationsCollection,
    MemoryRefsCollection,
)
from threetears.agent.memory.migrations import register as register_memory
from threetears.conversations.migrations import register as register_conversations
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner
from threetears.core.testing.plans import StatementPlan, explain_statement, plan_uses_index

from .conftest import AsyncpgStore

#: every test reads the one seeded schema (each plan is explained in a transaction rolled back), so the seed --
#: thousands of 1024-wide vectors through three HNSW indexes -- is built once a module
pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="module")]

#: rows seeded per table: past the size where the planner would rather scan than use an index
_ROWS = 6000
#: the embedding width the package registers by default
_DIM = 1024


class _Captured:
    """an L3 pool that runs nothing and records each statement asked of it, so a test explains the statement the
    code under test really writes."""

    def __init__(self) -> None:
        self.asked: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, sql: str, *args: Any) -> list[Any]:
        self.asked.append((sql, args))
        return []

    async def fetchrow(self, sql: str, *args: Any) -> None:
        self.asked.append((sql, args))

    async def fetchval(self, sql: str, *args: Any) -> int:
        self.asked.append((sql, args))
        return 0

    async def execute(self, sql: str, *args: Any) -> str:
        self.asked.append((sql, args))
        return "UPDATE 0"


class _World:
    """the seeded schema's connection and the owners its reads name."""

    def __init__(self, conn: asyncpg.Connection, agent: uuid.UUID, customer: uuid.UUID, user: uuid.UUID) -> None:
        self.conn = conn
        self.agent = agent
        self.customer = customer
        self.user = user
        self.memory = uuid.UUID(int=0)
        self.chunk = uuid.UUID(int=0)
        self.conversation = uuid.UUID(int=0)
        #: the one conversation holding 3,000 surfaced refs, where reading one agent's newest needs their index
        self.busy_conversation = uuid.UUID(int=0)

    def _collection(self, kind: type[Any], captured: _Captured) -> Any:
        registry = CollectionRegistry()
        registry.configure(l3_pool=captured)
        config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
        if kind is MemoriesCollection:
            return kind(registry, config, authorizer=MagicMock(spec=MemoryAuthorizerDependencies))
        return kind(registry, config)

    def memories(self, captured: _Captured) -> MemoriesCollection:
        """the memories collection over ``captured``.

        :param captured: the recording pool
        :ptype captured: _Captured
        :return: the collection
        :rtype: MemoriesCollection
        """
        collection: MemoriesCollection = self._collection(MemoriesCollection, captured)
        return collection

    def chunks(self, captured: _Captured) -> MemoryChunkCollection:
        """the memory chunks collection over ``captured``.

        :param captured: the recording pool
        :ptype captured: _Captured
        :return: the collection
        :rtype: MemoryChunkCollection
        """
        collection: MemoryChunkCollection = self._collection(MemoryChunkCollection, captured)
        return collection

    def contents(self, captured: _Captured) -> MediaContentCollection:
        """the media content collection over ``captured``.

        :param captured: the recording pool
        :ptype captured: _Captured
        :return: the collection
        :rtype: MediaContentCollection
        """
        collection: MediaContentCollection = self._collection(MediaContentCollection, captured)
        return collection

    def refs(self, captured: _Captured) -> MemoryRefsCollection:
        """the surfaced-refs collection over ``captured``.

        :param captured: the recording pool
        :ptype captured: _Captured
        :return: the collection
        :rtype: MemoryRefsCollection
        """
        collection: MemoryRefsCollection = self._collection(MemoryRefsCollection, captured)
        return collection

    def consolidations(self, captured: _Captured) -> MemoryConsolidationsCollection:
        """the consolidation edges collection over ``captured``.

        :param captured: the recording pool
        :ptype captured: _Captured
        :return: the collection
        :rtype: MemoryConsolidationsCollection
        """
        collection: MemoryConsolidationsCollection = self._collection(MemoryConsolidationsCollection, captured)
        return collection

    async def plans(self, captured: _Captured) -> list[StatementPlan]:
        """the plan of every statement ``captured`` recorded.

        :param captured: the recording pool
        :ptype captured: _Captured
        :return: one plan per statement, in order
        :rtype: list[StatementPlan]
        """
        return [await explain_statement(self.conn, sql, *args) for sql, args in captured.asked]


def _vector() -> list[float]:
    return [0.5] * _DIM


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def world(pg_url: str) -> AsyncIterator[_World]:
    """A fresh schema with the memory tables built and seeded: 6000 memories over three agents, thirty customers and
    three hundred users (each user in one customer), every tenth one named, each tagged and embedded."""
    schema = f"mem_plans_{uuid.uuid4().hex[:12]}"
    conn = await asyncpg.connect(pg_url)
    await conn.execute('CREATE EXTENSION IF NOT EXISTS "vector"')
    await conn.execute(f'CREATE SCHEMA "{schema}"')
    await conn.execute(f'SET search_path TO "{schema}", public')
    runner = MigrationRunner()
    register_conversations(runner)
    register_memory(runner)
    await runner.apply_for_agent_schema(AsyncpgStore(conn))  # type: ignore[arg-type]
    agents = [uuid.uuid4() for _ in range(3)]
    customers = [uuid.uuid4() for _ in range(30)]
    users = [uuid.uuid4() for _ in range(300)]
    await conn.execute(
        "INSERT INTO memories (memory_id, agent_id, customer_id, user_id, conversation_id, type_memory, content,"
        " date_created, date_updated, tags, alias, embedding)"
        " SELECT gen_random_uuid(), ($1::uuid[])[1 + g % 3], ($2::uuid[])[1 + (g % 300) % 30],"
        " ($3::uuid[])[1 + g % 300], gen_random_uuid(), 'fact', 'memory number ' || g,"
        " now() - g * interval '1 minute', now(), jsonb_build_array('t' || g % 50),"
        " CASE WHEN g % 10 = 0 THEN 'a' || g END,"
        f" (SELECT array_agg(random())::real[] FROM generate_series(1, {_DIM}) WHERE g > 0)::public.vector"
        f" FROM generate_series(1, {_ROWS}) g",
        agents,
        customers,
        users,
    )
    # one file per memory, one content row per file, one chunk per memory, and each memory consolidated into the next
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
        " SELECT gen_random_uuid(), memory_id, agent_id, customer_id, user_id, 0, content, date_created, embedding,"
        " gen_random_uuid() FROM memories"
    )
    await conn.execute(
        "INSERT INTO memory_consolidations (agent_id, consolidated_memory_id, source_memory_id)"
        " SELECT a.agent_id, a.memory_id, b.memory_id FROM"
        " (SELECT agent_id, memory_id, row_number() OVER (PARTITION BY agent_id ORDER BY memory_id) r FROM memories) a"
        " JOIN (SELECT agent_id, memory_id, row_number() OVER (PARTITION BY agent_id ORDER BY memory_id) r"
        " FROM memories) b ON b.agent_id = a.agent_id AND b.r = a.r + 1"
    )
    # 300 conversations, each holding 20 surfaced refs by each of the three agents
    await conn.execute(
        "INSERT INTO conversation_memory_refs (conversation_id, item_id, item_type, short_desc, date_created,"
        " date_updated, agent_id) SELECT c.id, gen_random_uuid(), 'memory', 'a ref', now() - n * interval '1 second',"
        " now(), ($1::uuid[])[1 + n % 3] FROM (SELECT gen_random_uuid() id FROM generate_series(1, 300)) c,"
        " generate_series(1, 60) n",
        agents,
    )
    # and one busy conversation: 3,000 refs, 1,000 by each agent (with 60 a conversation, reading them all by the
    # pk and sorting is the cheaper plan, so only a busy one shows whether the agent's newest read their index)
    busy = uuid.uuid4()
    await conn.execute(
        "INSERT INTO conversation_memory_refs (conversation_id, item_id, item_type, short_desc, date_created,"
        " date_updated, agent_id) SELECT $1, gen_random_uuid(), 'memory', 'a ref', now() - n * interval '1 second',"
        " now(), ($2::uuid[])[1 + n % 3] FROM generate_series(1, 3000) n",
        busy,
        agents,
    )
    tables = (
        "memories",
        "media",
        "media_content",
        "memory_chunks",
        "memory_consolidations",
        "conversation_memory_refs",
    )
    for table in tables:
        await conn.execute(f"ANALYZE {table}")
    # the owner the reads name: user 1 of customer 1, whose memories are all under agent 1 + 1 % 3
    world = _World(conn, agents[1], customers[1], users[1])
    mine = await conn.fetchrow(
        "SELECT m.memory_id, m.conversation_id, c.chunk_id FROM memories m JOIN memory_chunks c"
        " ON c.agent_id = m.agent_id AND c.memory_id = m.memory_id WHERE m.agent_id = $1 AND m.user_id = $2 LIMIT 1",
        world.agent,
        world.user,
    )
    world.memory, world.conversation, world.chunk = mine["memory_id"], mine["conversation_id"], mine["chunk_id"]
    world.busy_conversation = busy
    try:
        yield world
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

    def walk(node: dict[str, Any], depth: int) -> None:
        named = " ".join(
            f"{key}={node[key]}" for key in ("Relation Name", "Index Name", "Index Cond", "Actual Rows") if key in node
        )
        lines.append(f"{'  ' * depth}{node.get('Node Type')} {named}")
        for child in node.get("Plans", ()):
            walk(child, depth + 1)

    walk(plan.plan, 0)
    return plan.sql[:300] + "\n" + "\n".join(lines)


async def _only_plan(world: _World, captured: _Captured) -> StatementPlan:
    (plan,) = await world.plans(captured)
    return plan


async def test_memories_agent_user(world: _World) -> None:
    """Flips if one user's reads by agent and user stop reading ix_memories_agent_user_date."""
    captured = _Captured()
    memories = world.memories(captured)
    await memories.count_by_user(world.user, agent_id=world.agent)
    await memories.find_similar_for_dedup(
        user_id=world.user, agent_id=world.agent, embedding=_vector(), top_k=5, threshold=0.9
    )
    await memories.find_same_event(
        user_id=world.user, agent_id=world.agent, embedding=_vector(), message_ids=[uuid.uuid4()], threshold=0.9
    )
    await memories.search_by_semantic(
        user_id=world.user, agent_id=world.agent, embedding=_vector(), max_results=10, similarity_threshold=0.0
    )
    await memories.search_by_fts(user_id=world.user, agent_id=world.agent, fts_text="memory", max_results=10)
    await memories.list_for_user(user_id=world.user, agent_id=world.agent)
    await memories.find_by_scope(world.agent, user_id=world.user)
    await memories.find_for_user_in_agents(user_id=world.user, agent_ids=(world.agent, uuid.uuid4()))
    missed = [
        _shape(plan) for plan in await world.plans(captured) if not plan_uses_index(plan, "ix_memories_agent_user_date")
    ]
    assert not missed, "\n\n".join(missed)


async def test_memories_agent_customer_user(world: _World) -> None:
    """Flips if one owner's reads by the whole owner triple stop reading idx_memories_agent_customer_user."""
    captured = _Captured()
    memories = world.memories(captured)
    await memories.find_by_user(world.user, agent_id=world.agent, customer_id=world.customer)
    await memories.find_ids_by_tags(
        agent_id=world.agent, customer_id=world.customer, user_id=world.user, tags_any=("t1",)
    )
    await memories.find_by_scope(world.agent, world.customer, world.user)
    await memories.find_active_for_consolidation(world.agent, customer_id=world.customer, user_id=world.user)
    await memories.find_for_user_in_agents(
        user_id=world.user, agent_ids=(world.agent, uuid.uuid4()), customer_id=world.customer
    )
    await _hybrid_search(world, memories)
    missed = [
        _shape(plan)
        for plan in await world.plans(captured)
        if not _history_pool(plan) and not plan_uses_index(plan, "idx_memories_agent_customer_user")
    ]
    assert not missed, "\n\n".join(missed)


async def _hybrid_search(world: _World, memories: MemoriesCollection) -> None:
    await memories.hybrid_search(
        user_id=world.user,
        agent_id=world.agent,
        customer_id=world.customer,
        embedding=_vector(),
        user_text="memory number",
        top_k=5,
        candidate_limit=20,
        similarity_threshold=0.0,
        recency_half_life_hours=24.0,
        signal_weights={"semantic": 1.0, "keyword": 1.0, "recency": 1.0},
        recent_cap=5,
    )


def _history_pool(plan: StatementPlan) -> bool:
    """whether the statement is hybrid_search's history pool: the nearest memories older than the recent ones."""
    return "date_created <" in plan.sql and "similarity" in plan.sql


async def test_memories_user_date(world: _World) -> None:
    """Flips if hybrid_search's history pool (one owner's memories older than its recent ones, nearest first) stops
    reading ix_memories_agent_user_date: the user's memories by date, the range bounding them."""
    captured = _Captured()
    await _hybrid_search(world, world.memories(captured))
    history = [plan for plan in await world.plans(captured) if _history_pool(plan)]
    assert history, [sql for sql, _ in captured.asked]
    missed = [_shape(plan) for plan in history if not plan_uses_index(plan, "ix_memories_agent_user_date")]
    assert not missed, "\n\n".join(missed)


async def test_memories_user_alias(world: _World) -> None:
    """Flips if a memory by its name stops reading ix_memories_user_alias."""
    captured = _Captured()
    await world.memories(captured).find_by_alias(user_id=world.user, agent_id=world.agent, alias="a10")
    plan = await _only_plan(world, captured)
    assert plan_uses_index(plan, "ix_memories_user_alias"), _shape(plan)


_CHUNK_WEIGHTS = {"semantic": 1.0, "keyword": 1.0}


async def test_media_content_agent_user(world: _World) -> None:
    """Flips if one user's file contents stop reading idx_media_content_agent_user."""
    captured = _Captured()
    contents = world.contents(captured)
    await contents.hybrid_search(
        user_id=world.user,
        agent_id=world.agent,
        customer_id=world.customer,
        embedding=_vector(),
        user_text="content of",
        top_k=5,
        candidate_limit=20,
        similarity_threshold=0.0,
        recency_half_life_hours=24.0,
        signal_weights={"semantic": 1.0, "keyword": 1.0, "recency": 1.0},
    )
    await contents.search_by_semantic(
        user_id=world.user, agent_id=world.agent, embedding=_vector(), max_results=10, similarity_threshold=0.0
    )
    await contents.search_by_fts(user_id=world.user, agent_id=world.agent, fts_text="content", max_results=10)
    missed = [
        _shape(plan)
        for plan in await world.plans(captured)
        if not plan_uses_index(plan, "idx_media_content_agent_user")
    ]
    assert not missed, "\n\n".join(missed)


async def test_memory_chunks_agent_user(world: _World) -> None:
    """Flips if one user's chunk searches stop reading idx_memory_chunks_agent_user."""
    captured = _Captured()
    chunks = world.chunks(captured)
    await chunks.hybrid_search(
        user_id=world.user,
        agent_id=world.agent,
        customer_id=world.customer,
        embedding=_vector(),
        user_text="memory number",
        candidate_k=20,
        similarity_threshold=0.0,
        chunk_signal_weights=_CHUNK_WEIGHTS,
    )
    await chunks.search_by_semantic(
        user_id=world.user, agent_id=world.agent, embedding=_vector(), max_results=10, similarity_threshold=0.0
    )
    missed = [
        _shape(plan)
        for plan in await world.plans(captured)
        if not plan_uses_index(plan, "idx_memory_chunks_agent_user")
    ]
    assert not missed, "\n\n".join(missed)


async def test_memory_chunks_memory(world: _World) -> None:
    """Flips if a memory's chunks by index, its chunk count, or the count a recalled chunk carries stop reading
    ix_memory_chunks_memory."""
    captured = _Captured()
    chunks = world.chunks(captured)
    owner = {"user_id": world.user, "agent_id": world.agent, "customer_id": world.customer}
    await chunks.find_by_chunk_indexes(world.memory, chunk_indexes=[0, 1], **owner)
    await chunks.count_by_memory_id(world.memory, **owner)
    await chunks.find_for_recall(chunk_id=world.chunk, user_id=world.user, agent_id=world.agent)
    await chunks.hybrid_search_within_memory(
        memory_id=world.memory,
        embedding=_vector(),
        user_text="memory number",
        candidate_k=20,
        similarity_threshold=0.0,
        chunk_signal_weights=_CHUNK_WEIGHTS,
        **owner,
    )
    await chunks.find_by_conversation_id(world.conversation, **owner)
    missed = [
        _shape(plan) for plan in await world.plans(captured) if not plan_uses_index(plan, "ix_memory_chunks_memory")
    ]
    assert not missed, "\n\n".join(missed)


async def test_chunks_memory_id_chunk_id(world: _World) -> None:
    """Flips if a memory's chunks in order stop reading idx_chunks_memory_id_chunk_id."""
    captured = _Captured()
    await world.chunks(captured).find_by_memory_id(
        world.memory, user_id=world.user, agent_id=world.agent, customer_id=world.customer
    )
    plan = await _only_plan(world, captured)
    assert plan_uses_index(plan, "idx_chunks_memory_id_chunk_id"), _shape(plan)


async def test_mem_conversation(world: _World) -> None:
    """Flips if a conversation's chunks stop finding its memories through idx_mem_conversation."""
    captured = _Captured()
    await world.chunks(captured).find_by_conversation_id(
        world.conversation, user_id=world.user, agent_id=world.agent, customer_id=world.customer
    )
    plan = await _only_plan(world, captured)
    assert plan_uses_index(plan, "idx_mem_conversation"), _shape(plan)


async def test_media_memory_id(world: _World) -> None:
    """Flips if a chunk's file, joined by its memory, stops being found through ix_media_agent_memory."""
    captured = _Captured()
    chunks = world.chunks(captured)
    await chunks.search_by_ids([world.chunk], world.user, agent_id=world.agent)
    await chunks.find_for_recall(chunk_id=world.chunk, user_id=world.user, agent_id=world.agent)
    await chunks.hybrid_search(
        user_id=world.user,
        agent_id=world.agent,
        customer_id=world.customer,
        embedding=_vector(),
        user_text="memory number",
        candidate_k=20,
        similarity_threshold=0.0,
        chunk_signal_weights=_CHUNK_WEIGHTS,
    )
    await chunks.search_by_semantic(
        user_id=world.user, agent_id=world.agent, embedding=_vector(), max_results=10, similarity_threshold=0.0
    )
    await chunks.hybrid_search_within_memory(
        memory_id=world.memory,
        user_id=world.user,
        agent_id=world.agent,
        customer_id=world.customer,
        embedding=_vector(),
        user_text="memory number",
        candidate_k=20,
        similarity_threshold=0.0,
        chunk_signal_weights=_CHUNK_WEIGHTS,
    )
    missed = [
        _shape(plan) for plan in await world.plans(captured) if not plan_uses_index(plan, "ix_media_agent_memory")
    ]
    assert not missed, "\n\n".join(missed)


async def test_memory_consolidations_source(world: _World) -> None:
    """Flips if what a source was merged into stops reading idx_memory_consolidations_source."""
    captured = _Captured()
    await world.consolidations(captured).find_consolidated_into(world.agent, world.memory)
    await world.memories(captured).search_by_ids([world.memory], world.user, agent_id=world.agent)
    missed = [
        _shape(plan)
        for plan in await world.plans(captured)
        if not plan_uses_index(plan, "idx_memory_consolidations_source")
    ]
    assert not missed, "\n\n".join(missed)


async def test_conv_mem_refs_conversation_agent_date(world: _World) -> None:
    """Flips if the newest refs one agent surfaced in a busy conversation stop reading
    idx_conv_mem_refs_conversation_agent_date."""
    captured = _Captured()
    await world.refs(captured).find_by_conversation(world.busy_conversation, world.agent)
    plan = await _only_plan(world, captured)
    assert plan_uses_index(plan, "idx_conv_mem_refs_conversation_agent_date"), _shape(plan)


async def test_media_content_media_type(world: _World) -> None:
    """Flips if the contents a search names by file id stop reading ix_media_content_media_type (the content-id
    branch of the same OR reads the unique content id)."""
    file = await world.conn.fetchval(
        "SELECT media_id FROM media_content WHERE agent_id = $1 AND user_id = $2 LIMIT 1", world.agent, world.user
    )
    captured = _Captured()
    await world.contents(captured).search_by_ids([file], world.user, agent_id=world.agent)
    plan = await _only_plan(world, captured)
    assert plan_uses_index(plan, "ix_media_content_media_type"), _shape(plan)


async def test_media_cloud_connection_file(world: _World) -> None:
    """Flips if two media rows of one cloud connection may name the same provider file: uq_media_cloud_connection_file
    refuses the second."""
    connection = uuid.uuid4()
    insert = (
        "INSERT INTO media (media_id, memory_id, agent_id, customer_id, user_id, date_created, date_updated,"
        " cloud_connection_id, cloud_file_id) SELECT gen_random_uuid(), memory_id, agent_id, customer_id, user_id,"
        " now(), now(), $1, 'file-1' FROM media LIMIT 1"
    )
    transaction = world.conn.transaction()
    await transaction.start()
    try:
        await world.conn.execute(insert, connection)
        with pytest.raises(asyncpg.UniqueViolationError, match="uq_media_cloud_connection_file"):
            await world.conn.execute(insert, connection)
    finally:
        await transaction.rollback()
