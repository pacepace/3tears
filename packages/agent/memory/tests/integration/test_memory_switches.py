"""integration: the three automatic-memory switches on :class:`MemoryConfig`.

``extraction_enabled``, ``ambient_recall_enabled`` and ``consolidation_enabled``
each turn one automatic path off, against a real pgvector database:

- off, :meth:`MemoryExtractor.extract` writes no row and answers
  ``SKIPPED`` / ``ExtractionGate.DISABLED``;
- off, :class:`MemoryRetriever` (and through it ``retrieve_memories``, which
  ``MemoryInjectionMiddleware`` calls) returns nothing for a stored memory
  that would match;
- off, :meth:`DreamService.run_consolidation` writes no gist and supersedes
  no source.

Each is run with the switch on as well, on the same fixture, so the off case
cannot pass on a fixture that would have produced nothing anyway. The explicit
tools are then driven with all three off, on the same collections.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

import asyncpg
import pytest

from threetears.agent.memory.authorize import MemoryAuthorizerDependencies
from threetears.agent.memory.collections import (
    MediaContentCollection,
    MemoriesCollection,
    MemoryChunkCollection,
)
from threetears.agent.memory.extraction import ExtractionGate, ExtractionOutcome, MemoryExtractor
from threetears.agent.memory.integration import MemoryIntegration, retrieve_memories
from threetears.agent.memory.migrations import register as register_memory
from threetears.agent.memory.retrieval import MemoryRetriever
from threetears.agent.memory.revisions import RETRACTED_TAG
from threetears.agent.memory.tools import (
    load_memory_add_tool,
    load_memory_forget_tool,
    load_memory_recall_tool,
    load_memory_search_tool,
)
from threetears.agent.memory.types import MemoryConfig
from threetears.conversations.migrations import register as register_conversations
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner

from .conftest import AsyncpgStore
from .memory_support import PurposeChatModelFactory, StubEmbeddings, insert_source, make_pool, make_service, vec


pytestmark = pytest.mark.integration

_ALL_OFF = MemoryConfig(extraction_enabled=False, ambient_recall_enabled=False, consolidation_enabled=False)


@pytest.fixture
async def applied_schema(pg_schema: tuple[str, str]) -> tuple[str, str]:
    """apply conversations + memory migrations into the per-test schema."""
    url, schema = pg_schema
    runner = MigrationRunner()
    register_conversations(runner)
    register_memory(runner)
    conn = await asyncpg.connect(url)
    try:
        await conn.execute(f'SET search_path TO "{schema}", public')
        await runner.apply_for_agent_schema(AsyncpgStore(conn))  # type: ignore[arg-type]
    finally:
        await conn.close()
    return url, schema


def _collections(
    pool: asyncpg.Pool, authorizer: MemoryAuthorizerDependencies
) -> tuple[MemoriesCollection, MediaContentCollection, MemoryChunkCollection]:
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return (
        MemoriesCollection(registry=registry, config=config, authorizer=authorizer, nats_client=None),
        MediaContentCollection(registry=registry, config=config, nats_client=None),
        MemoryChunkCollection(registry=registry, config=config, nats_client=None),
    )


def _extractor(config: MemoryConfig, memories: MemoriesCollection, authorizer: Any) -> MemoryExtractor:
    factory = PurposeChatModelFactory(
        worthiness=json.dumps({"worthy": True, "reason": "has facts"}),
        extraction=json.dumps([{"type": "fact", "content": "User lives in Seattle"}]),
    )
    return MemoryExtractor(
        config=config,
        embedding_provider=StubEmbeddings(vec(0.1)),
        chat_model_factory=factory,
        authorizer=authorizer,
        memories_collection=memories,
        nats_client=None,
    )


def _retriever(
    config: MemoryConfig,
    collections: tuple[MemoriesCollection, MediaContentCollection, MemoryChunkCollection],
    authorizer: Any,
) -> MemoryRetriever:
    memories, media_content, chunks = collections
    return MemoryRetriever(
        config,
        StubEmbeddings(vec(0.1)),
        authorizer,
        memories_collection=memories,
        media_content_collection=media_content,
        memory_chunk_collection=chunks,
    )


async def _seed_memory(pool: asyncpg.Pool, *, agent_id: uuid.UUID, customer_id: uuid.UUID, user_id: uuid.UUID) -> None:
    now = datetime.now(UTC)
    await pool.execute(
        "INSERT INTO memories ("
        "memory_id, agent_id, customer_id, user_id, conversation_id, message_id_source, "
        "type_memory, content, embedding, date_created, date_updated"
        ") VALUES ($1, $2, $3, $4, $5, $6, 'preference', $7, $8::text::public.vector, $9, $9)",
        uuid.uuid4(),
        agent_id,
        customer_id,
        user_id,
        uuid.uuid4(),
        uuid.uuid4(),
        "User prefers Rust programming language",
        json.dumps(vec(0.1)),
        now,
    )


class TestExtractionSwitch:
    @pytest.mark.parametrize("enabled", [True, False])
    async def test_switch_decides_whether_a_turn_writes_a_row(
        self,
        enabled: bool,
        applied_schema: tuple[str, str],
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        url, schema = applied_schema
        pool = await make_pool(url, schema)
        try:
            memories, _, _ = _collections(pool, permissive_memory_authorizer)
            extractor = _extractor(MemoryConfig(extraction_enabled=enabled), memories, permissive_memory_authorizer)
            user_id = uuid.uuid4()
            result = await extractor.extract(
                user_id=user_id,
                conversation_id=uuid.uuid4(),
                message_id_source=uuid.uuid4(),
                user_message="x" * 50,
                assistant_response="y" * 200,
                turn_count=10,
                agent_id=uuid.uuid4(),
                customer_id=uuid.uuid4(),
            )
            rows = await pool.fetch("SELECT content FROM memories WHERE user_id = $1", user_id)
            if enabled:
                assert result.outcome is ExtractionOutcome.STORED
                assert [r["content"] for r in rows] == ["User lives in Seattle"]
            else:
                assert result.outcome is ExtractionOutcome.SKIPPED
                assert result.gate is ExtractionGate.DISABLED
                assert result.stored == 0
                assert rows == []
        finally:
            await pool.close()


class TestAmbientRecallSwitch:
    @pytest.mark.parametrize("enabled", [True, False])
    async def test_switch_decides_whether_a_stored_memory_is_handed_over(
        self,
        enabled: bool,
        applied_schema: tuple[str, str],
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        url, schema = applied_schema
        pool = await make_pool(url, schema)
        try:
            agent_id, customer_id, user_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
            await _seed_memory(pool, agent_id=agent_id, customer_id=customer_id, user_id=user_id)
            collections = _collections(pool, permissive_memory_authorizer)
            retriever = _retriever(
                MemoryConfig(ambient_recall_enabled=enabled), collections, permissive_memory_authorizer
            )
            query = "What does user prefer programming?"

            result = await retriever.retrieve_with_candidates(
                user_id, query, agent_id=agent_id, customer_id=customer_id
            )
            # the path MemoryInjectionMiddleware takes
            injected = await retrieve_memories(
                MemoryIntegration(retriever=retriever),
                agent_id=agent_id,
                customer_id=customer_id,
                user_id=user_id,
                query=query,
            )
            if enabled:
                assert [m["content"] for m in result.memories] == ["User prefers Rust programming language"]
                assert result.context is not None and "Rust" in result.context
                assert len(injected) == 1 and "Rust" in injected[0]
            else:
                assert result.memories == []
                assert result.media_content == []
                assert result.memory_chunks == []
                assert result.context is None
                assert injected == []
        finally:
            await pool.close()


class TestConsolidationSwitch:
    @pytest.mark.parametrize("enabled", [True, False])
    async def test_switch_decides_whether_near_duplicates_merge(
        self, enabled: bool, applied_schema: tuple[str, str]
    ) -> None:
        url, schema = applied_schema
        pool = await make_pool(url, schema)
        conn = await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            agent_id, customer_id, user_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
            sources = []
            for i, age in enumerate((3, 1)):
                mid, _ = await insert_source(
                    conn,
                    agent_id=agent_id,
                    customer_id=customer_id,
                    user_id=user_id,
                    seed=0.05,
                    content=f"prefers postgres reason {i}",
                    age_days=age,
                )
                sources.append(mid)
            service = make_service(
                pool, gist_vector=vec(0.9), memory_config=MemoryConfig(consolidation_enabled=enabled)
            )
            result = await service.run_consolidation(agent_id, customer_id=customer_id, user_id=user_id)

            total = await conn.fetchval("SELECT count(*) FROM memories WHERE agent_id = $1", agent_id)
            superseded = await conn.fetchval(
                "SELECT count(*) FROM memories WHERE agent_id = $1 AND superseded_by IS NOT NULL", agent_id
            )
            if enabled:
                assert result.gists_created == 1
                assert total == 3
                assert superseded == 2
            else:
                assert (result.memories_considered, result.gists_created, result.gist_ids) == (0, 0, [])
                assert total == 2
                assert superseded == 0
        finally:
            await conn.close()
            await pool.close()


class TestExplicitToolsWithEverySwitchOff:
    async def test_add_search_recall_forget_still_work(
        self,
        applied_schema: tuple[str, str],
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        url, schema = applied_schema
        pool = await make_pool(url, schema)
        try:
            agent_id, customer_id, user_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
            conv_id, msg_id = uuid.uuid4(), uuid.uuid4()
            collections = _collections(pool, permissive_memory_authorizer)
            memories, media_content, chunks = collections
            embed = StubEmbeddings(vec(0.1))

            # the automatic paths over the same collections are all off ...
            assert (
                await _retriever(_ALL_OFF, collections, permissive_memory_authorizer).retrieve(
                    user_id, "anything at all", agent_id=agent_id, customer_id=customer_id
                )
                is None
            )

            class _Ctx:
                conversation_id = conv_id
                correlation_id = msg_id

            # ... and the tools still save, find, read and retire.
            [add] = await load_memory_add_tool(
                user_id,
                embed,
                agent_id,
                customer_id,
                permissive_memory_authorizer,
                memories,
                context_resolver=lambda: _Ctx(),
            )
            added = await add.ainvoke({"content": "User prefers Python", "memory_type": "preference"})
            assert "Stored as [memory:" in added
            row = await pool.fetchrow(
                "SELECT memory_id FROM memories WHERE user_id = $1 AND content = $2", user_id, "User prefers Python"
            )
            assert row is not None
            memory_id = row["memory_id"]

            [search] = await load_memory_search_tool(
                user_id, embed, agent_id, customer_id, permissive_memory_authorizer, memories, media_content, chunks
            )
            assert "User prefers Python" in await search.ainvoke({"query": "What does the user prefer?"})

            [recall] = await load_memory_recall_tool(
                user_id, agent_id, customer_id, permissive_memory_authorizer, memories, media_content, chunks
            )
            recalled = await recall.ainvoke({"memory_id": str(memory_id)})
            assert f"[memory:{memory_id}]" in recalled and "User prefers Python" in recalled

            [forget] = load_memory_forget_tool(user_id, agent_id, customer_id, permissive_memory_authorizer, memories)
            await forget.ainvoke({"memory_id": str(memory_id), "reason": "the user said it no longer holds"})
            retired = await pool.fetchval(
                "SELECT tags ? $2 FROM memories WHERE memory_id = $1", memory_id, RETRACTED_TAG
            )
            assert retired is True
        finally:
            await pool.close()
