"""End-to-end coverage for the new MemoryChunkCollection methods.

Pinned in this file:

- ``find_by_memory_id`` returns the chunks parented to a memory in
  ``chunk_id ASC`` order with cursor paging.
- ``find_by_conversation_id`` walks the chunks across all memories
  in one conversation, ordered by ``message_id_start ASC``.
- ``hybrid_search_within_memory`` restricts the candidate pool to
  one parent memory.
- The cursor pair (``chunk_id_after`` / ``chunk_id_before``) is
  mutually exclusive — passing both raises ValueError.
- Auth scoping: mismatching any of ``(user_id, agent_id,
  customer_id)`` returns empty. The static auth test
  (``tests/unit/test_chunk_collection_auth.py``) catches signature
  drops; this test exercises the SQL with real rows.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import asyncpg
import pytest

from threetears.agent.memory.authorize import MemoryAuthorizerDependencies
from threetears.agent.memory.collections import (
    MemoriesCollection,
    MemoryChunkCollection,
)
from threetears.agent.memory.migrations import register as register_memory
from threetears.conversations.migrations import register as register_conversations
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner

from .conftest import AsyncpgStore


@pytest.fixture
async def applied_schema(pg_schema: tuple[str, str]) -> tuple[str, str]:
    """Apply conversations + memory migrations into the per-test schema."""
    url, schema = pg_schema
    runner = MigrationRunner()
    register_conversations(runner)
    register_memory(runner)
    conn = await asyncpg.connect(url)
    try:
        await conn.execute(f'SET search_path TO "{schema}", public')
        store = AsyncpgStore(conn)
        await runner.apply_for_agent_schema(store)  # type: ignore[arg-type]
    finally:
        await conn.close()
    return url, schema


async def _make_pool(url: str, schema: str) -> asyncpg.Pool:
    from threetears.core.collections import init_connection

    pool: asyncpg.Pool = await asyncpg.create_pool(
        dsn=url,
        min_size=1,
        max_size=4,
        server_settings={"search_path": f"{schema}, public"},
        init=init_connection,
    )
    return pool


def _build_collections(
    pool: asyncpg.Pool,
    authorizer: MemoryAuthorizerDependencies,
) -> tuple[MemoriesCollection, MemoryChunkCollection]:
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    config = DefaultCoreConfig(
        collection_flush="ALWAYS",
        collection_flush_tables="",
    )
    memories = MemoriesCollection(
        registry=registry,
        config=config,
        authorizer=authorizer,
    )
    chunks = MemoryChunkCollection(
        registry=registry,
        config=config,
    )
    return memories, chunks


async def _seed_memory(
    memories: MemoriesCollection,
    *,
    agent_id: uuid.UUID,
    customer_id: uuid.UUID,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID,
    content: str = "parent memory",
) -> uuid.UUID:
    """Insert a parent memory and return its id."""
    memory_id = uuid.uuid4()
    now = datetime.now(UTC)
    entity = memories.create(
        {
            "memory_id": memory_id,
            "agent_id": agent_id,
            "customer_id": customer_id,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "message_id_source": uuid.uuid4(),
            "type_memory": "topical_context",
            "content": content,
            "embedding": [0.1] * 1024,
            "date_created": now,
            "date_updated": now,
        }
    )
    await memories.save_entity(entity)
    return memory_id


async def _seed_chunk(
    chunks: MemoryChunkCollection,
    *,
    memory_id: uuid.UUID,
    agent_id: uuid.UUID,
    customer_id: uuid.UUID,
    user_id: uuid.UUID,
    content: str,
    message_id_start: uuid.UUID | None = None,
    message_id_end: uuid.UUID | None = None,
) -> uuid.UUID:
    """Insert one chunk parented to ``memory_id``; returns its chunk_id."""
    import uuid_utils

    # Use uuid7 so chunk_id is byte-ordered with creation time; cursor
    # paging assertions depend on this.
    chunk_id = uuid.UUID(str(uuid_utils.uuid7()))
    now = datetime.now(UTC)
    entity = chunks.create(
        {
            "chunk_id": chunk_id,
            "memory_id": memory_id,
            "agent_id": agent_id,
            "customer_id": customer_id,
            "user_id": user_id,
            "content": content,
            "summary": content[:40],
            "heading_context": None,
            "page_number": None,
            "embedding": [0.1] * 1024,
            "message_id_start": message_id_start,
            "message_id_end": message_id_end,
            "date_created": now,
        }
    )
    await chunks.save_entity(entity)
    return chunk_id


class TestFindByMemoryId:
    async def test_returns_chunks_for_parent_memory(
        self,
        applied_schema: tuple[str, str],
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            memories, chunks = _build_collections(
                pool, permissive_memory_authorizer
            )
            agent_id = uuid.uuid4()
            customer_id = uuid.uuid4()
            user_id = uuid.uuid4()
            mem = await _seed_memory(
                memories,
                agent_id=agent_id,
                customer_id=customer_id,
                user_id=user_id,
                conversation_id=uuid.uuid4(),
            )
            for i in range(3):
                await _seed_chunk(
                    chunks,
                    memory_id=mem,
                    agent_id=agent_id,
                    customer_id=customer_id,
                    user_id=user_id,
                    content=f"chunk {i}",
                )

            result = await chunks.find_by_memory_id(
                mem,
                user_id=user_id,
                agent_id=agent_id,
                customer_id=customer_id,
            )
            assert len(result) == 3
            # ORDER BY chunk_id ASC — UUIDv7 is monotonic within burst.
            ids_in_order = [row["chunk_id"] for row in result]
            assert ids_in_order == sorted(ids_in_order)
        finally:
            await pool.close()

    async def test_cursor_after_advances_forward(
        self,
        applied_schema: tuple[str, str],
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            memories, chunks = _build_collections(
                pool, permissive_memory_authorizer
            )
            agent_id = uuid.uuid4()
            customer_id = uuid.uuid4()
            user_id = uuid.uuid4()
            mem = await _seed_memory(
                memories,
                agent_id=agent_id,
                customer_id=customer_id,
                user_id=user_id,
                conversation_id=uuid.uuid4(),
            )
            for i in range(5):
                await _seed_chunk(
                    chunks,
                    memory_id=mem,
                    agent_id=agent_id,
                    customer_id=customer_id,
                    user_id=user_id,
                    content=f"chunk {i}",
                )

            first_two = await chunks.find_by_memory_id(
                mem,
                user_id=user_id,
                agent_id=agent_id,
                customer_id=customer_id,
                limit=2,
            )
            assert len(first_two) == 2
            cursor = first_two[-1]["chunk_id"]
            next_page = await chunks.find_by_memory_id(
                mem,
                user_id=user_id,
                agent_id=agent_id,
                customer_id=customer_id,
                limit=2,
                chunk_id_after=cursor,
            )
            assert len(next_page) == 2
            assert all(row["chunk_id"] > cursor for row in next_page)
        finally:
            await pool.close()

    async def test_both_cursors_raises_value_error(
        self,
        applied_schema: tuple[str, str],
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            _, chunks = _build_collections(pool, permissive_memory_authorizer)
            with pytest.raises(ValueError, match="at most one"):
                await chunks.find_by_memory_id(
                    uuid.uuid4(),
                    user_id=uuid.uuid4(),
                    agent_id=uuid.uuid4(),
                    customer_id=uuid.uuid4(),
                    chunk_id_after=uuid.uuid4(),
                    chunk_id_before=uuid.uuid4(),
                )
        finally:
            await pool.close()

    async def test_mismatched_user_id_returns_empty(
        self,
        applied_schema: tuple[str, str],
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        """A chunk seeded under user_id A must NOT surface for user_id B."""
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            memories, chunks = _build_collections(
                pool, permissive_memory_authorizer
            )
            agent_id = uuid.uuid4()
            customer_id = uuid.uuid4()
            owner_user = uuid.uuid4()
            mem = await _seed_memory(
                memories,
                agent_id=agent_id,
                customer_id=customer_id,
                user_id=owner_user,
                conversation_id=uuid.uuid4(),
            )
            await _seed_chunk(
                chunks,
                memory_id=mem,
                agent_id=agent_id,
                customer_id=customer_id,
                user_id=owner_user,
                content="owned",
            )

            result = await chunks.find_by_memory_id(
                mem,
                user_id=uuid.uuid4(),  # different user
                agent_id=agent_id,
                customer_id=customer_id,
            )
            assert result == []
        finally:
            await pool.close()

    async def test_mismatched_agent_id_returns_empty(
        self,
        applied_schema: tuple[str, str],
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            memories, chunks = _build_collections(
                pool, permissive_memory_authorizer
            )
            agent_id = uuid.uuid4()
            customer_id = uuid.uuid4()
            user_id = uuid.uuid4()
            mem = await _seed_memory(
                memories,
                agent_id=agent_id,
                customer_id=customer_id,
                user_id=user_id,
                conversation_id=uuid.uuid4(),
            )
            await _seed_chunk(
                chunks,
                memory_id=mem,
                agent_id=agent_id,
                customer_id=customer_id,
                user_id=user_id,
                content="owned",
            )

            result = await chunks.find_by_memory_id(
                mem,
                user_id=user_id,
                agent_id=uuid.uuid4(),  # different partition
                customer_id=customer_id,
            )
            assert result == []
        finally:
            await pool.close()

    async def test_mismatched_customer_id_returns_empty(
        self,
        applied_schema: tuple[str, str],
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            memories, chunks = _build_collections(
                pool, permissive_memory_authorizer
            )
            agent_id = uuid.uuid4()
            customer_id = uuid.uuid4()
            user_id = uuid.uuid4()
            mem = await _seed_memory(
                memories,
                agent_id=agent_id,
                customer_id=customer_id,
                user_id=user_id,
                conversation_id=uuid.uuid4(),
            )
            await _seed_chunk(
                chunks,
                memory_id=mem,
                agent_id=agent_id,
                customer_id=customer_id,
                user_id=user_id,
                content="owned",
            )

            result = await chunks.find_by_memory_id(
                mem,
                user_id=user_id,
                agent_id=agent_id,
                customer_id=uuid.uuid4(),  # different sub-scope
            )
            assert result == []
        finally:
            await pool.close()


class TestFindByConversationId:
    async def test_returns_transcript_chunks_only(
        self,
        applied_schema: tuple[str, str],
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        """Document chunks (message_id_start IS NULL) must be excluded."""
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            memories, chunks = _build_collections(
                pool, permissive_memory_authorizer
            )
            agent_id = uuid.uuid4()
            customer_id = uuid.uuid4()
            user_id = uuid.uuid4()
            conv_id = uuid.uuid4()
            mem = await _seed_memory(
                memories,
                agent_id=agent_id,
                customer_id=customer_id,
                user_id=user_id,
                conversation_id=conv_id,
            )
            # Two transcript chunks with message_id_start populated
            await _seed_chunk(
                chunks,
                memory_id=mem,
                agent_id=agent_id,
                customer_id=customer_id,
                user_id=user_id,
                content="transcript A",
                message_id_start=uuid.uuid4(),
                message_id_end=uuid.uuid4(),
            )
            await _seed_chunk(
                chunks,
                memory_id=mem,
                agent_id=agent_id,
                customer_id=customer_id,
                user_id=user_id,
                content="transcript B",
                message_id_start=uuid.uuid4(),
                message_id_end=uuid.uuid4(),
            )
            # One document chunk (no backlinks)
            await _seed_chunk(
                chunks,
                memory_id=mem,
                agent_id=agent_id,
                customer_id=customer_id,
                user_id=user_id,
                content="document chunk",
                message_id_start=None,
                message_id_end=None,
            )

            result = await chunks.find_by_conversation_id(
                conv_id,
                user_id=user_id,
                agent_id=agent_id,
                customer_id=customer_id,
            )
            assert len(result) == 2
            assert all(row["message_id_start"] is not None for row in result)
        finally:
            await pool.close()

    async def test_mismatched_user_id_returns_empty(
        self,
        applied_schema: tuple[str, str],
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            memories, chunks = _build_collections(
                pool, permissive_memory_authorizer
            )
            agent_id = uuid.uuid4()
            customer_id = uuid.uuid4()
            owner_user = uuid.uuid4()
            conv_id = uuid.uuid4()
            mem = await _seed_memory(
                memories,
                agent_id=agent_id,
                customer_id=customer_id,
                user_id=owner_user,
                conversation_id=conv_id,
            )
            await _seed_chunk(
                chunks,
                memory_id=mem,
                agent_id=agent_id,
                customer_id=customer_id,
                user_id=owner_user,
                content="owned transcript",
                message_id_start=uuid.uuid4(),
                message_id_end=uuid.uuid4(),
            )

            result = await chunks.find_by_conversation_id(
                conv_id,
                user_id=uuid.uuid4(),  # different user
                agent_id=agent_id,
                customer_id=customer_id,
            )
            assert result == []
        finally:
            await pool.close()


class TestHybridSearchWithinMemory:
    async def test_restricts_to_one_parent_memory(
        self,
        applied_schema: tuple[str, str],
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            memories, chunks = _build_collections(
                pool, permissive_memory_authorizer
            )
            agent_id = uuid.uuid4()
            customer_id = uuid.uuid4()
            user_id = uuid.uuid4()
            mem_a = await _seed_memory(
                memories,
                agent_id=agent_id,
                customer_id=customer_id,
                user_id=user_id,
                conversation_id=uuid.uuid4(),
                content="memory A",
            )
            mem_b = await _seed_memory(
                memories,
                agent_id=agent_id,
                customer_id=customer_id,
                user_id=user_id,
                conversation_id=uuid.uuid4(),
                content="memory B",
            )
            await _seed_chunk(
                chunks,
                memory_id=mem_a,
                agent_id=agent_id,
                customer_id=customer_id,
                user_id=user_id,
                content="chunk under A",
            )
            await _seed_chunk(
                chunks,
                memory_id=mem_b,
                agent_id=agent_id,
                customer_id=customer_id,
                user_id=user_id,
                content="chunk under B",
            )

            result = await chunks.hybrid_search_within_memory(
                memory_id=mem_a,
                user_id=user_id,
                agent_id=agent_id,
                customer_id=customer_id,
                embedding=[0.1] * 1024,
                user_text="chunk",
                candidate_k=10,
                similarity_threshold=0.0,
                chunk_signal_weights={"semantic": 0.7, "keyword": 0.3},
            )
            # Only the chunk under mem_a should surface.
            assert len(result) == 1
            assert "chunk under A" in result[0]["content"]
        finally:
            await pool.close()

    async def test_mismatched_user_id_returns_empty(
        self,
        applied_schema: tuple[str, str],
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            memories, chunks = _build_collections(
                pool, permissive_memory_authorizer
            )
            agent_id = uuid.uuid4()
            customer_id = uuid.uuid4()
            owner_user = uuid.uuid4()
            mem = await _seed_memory(
                memories,
                agent_id=agent_id,
                customer_id=customer_id,
                user_id=owner_user,
                conversation_id=uuid.uuid4(),
            )
            await _seed_chunk(
                chunks,
                memory_id=mem,
                agent_id=agent_id,
                customer_id=customer_id,
                user_id=owner_user,
                content="owned",
            )

            result = await chunks.hybrid_search_within_memory(
                memory_id=mem,
                user_id=uuid.uuid4(),  # different user
                agent_id=agent_id,
                customer_id=customer_id,
                embedding=[0.1] * 1024,
                user_text="anything",
                candidate_k=10,
                similarity_threshold=0.0,
                chunk_signal_weights={"semantic": 0.7, "keyword": 0.3},
            )
            assert result == []
        finally:
            await pool.close()
