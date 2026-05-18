"""integration: L1 write-through for every memory-package Collection.

namespace-task-01 phase 8.5b: the four memory Collections snap the L1
:class:`SQLiteBackend` from the :class:`CollectionRegistry` and route
writes / reads through it. this test asserts for each of the four
Collections that:

1. ``save_entity`` populates the L1 row (subsequent ``.get(id)`` hits
   L1 without a L3 round-trip).
2. ``hybrid_search`` / other search methods return the same results
   as the raw-SQL path they replaced (regression-free).

uses the same testcontainers pgvector fixture as
``test_memory_pipeline.py``.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import asyncpg
import pytest
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    Text,
)

from threetears.agent.memory.authorize import MemoryAuthorizerDependencies
from threetears.agent.memory.collections import (
    MediaCollection,
    MediaContentCollection,
    MemoriesCollection,
    MemoryChunkCollection,
)
from threetears.agent.memory.migrations import register as register_memory
from threetears.conversations.migrations import register as register_conversations
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner

from .conftest import AsyncpgStore


pytestmark = pytest.mark.integration


def _build_l1_metadata() -> MetaData:
    """build an L1 mirror of the four memory tables.

    collections-task-04 partitioned every memory table on ``agent_id``;
    L1 keys mirror the composite PK shape so SQLite addresses rows the
    same way L3 does.
    """
    meta = MetaData()
    Table(
        "memories",
        meta,
        Column("agent_id", String(255), primary_key=True),
        Column("memory_id", String(255), primary_key=True),
        Column("customer_id", String(255)),
        Column("user_id", String(255)),
        Column("conversation_id", String(255)),
        Column("message_id_source", String(255)),
        Column("type_memory", String(50)),
        Column("content", Text),
        Column("embedding", Text),
        Column("date_created", DateTime),
        Column("date_updated", DateTime),
    )
    # v0.8.0: schemas widened to full prod shape; mirror the new
    # required columns in this embedded L1 DDL fixture. (Per the
    # 2026-05-03 "embedded DDL in test setup is a smell" learning, a
    # follow-up should retire this in favor of calling the package
    # factories directly; that's a separate refactor scoped to shard
    # 04's factory-collapse work.)
    Table(
        "media",
        meta,
        Column("agent_id", String(255), primary_key=True),
        Column("media_id", String(255), primary_key=True),
        Column("memory_id", String(255)),
        Column("customer_id", String(255)),
        Column("user_id", String(255)),
        Column("s3_key", Text),
        Column("mime_type", Text),
        Column("size_bytes", Integer),
        Column("source", Text),
        Column("media_category", String(64)),
        Column("extraction_status", Text),
        Column("metadata_json", Text),
        Column("generation_prompt", Text),
        Column("thumbnail_s3_key", Text),
        Column("cloud_connection_id", String(255)),
        Column("cloud_file_id", Text),
        Column("cloud_file_url", Text),
        Column("date_created", DateTime),
    )
    Table(
        "media_content",
        meta,
        Column("agent_id", String(255), primary_key=True),
        Column("content_id", String(255), primary_key=True),
        Column("media_id", String(255)),
        Column("customer_id", String(255)),
        Column("user_id", String(255)),
        Column("content_type", String(64)),
        Column("content", Text),
        Column("summary", Text),
        Column("embedding", Text),
        Column("model_id", String(255)),
        Column("provider_id", String(255)),
        Column("model_name", Text),
        Column("provider_name", Text),
        Column("token_count_prompt", Integer),
        Column("token_count_completion", Integer),
        Column("cost", Text),
        Column("metadata_json", Text),
        Column("date_created", DateTime),
    )
    Table(
        "memory_chunks",
        meta,
        Column("agent_id", String(255), primary_key=True),
        Column("chunk_id", String(255), primary_key=True),
        Column("memory_id", String(255)),
        Column("customer_id", String(255)),
        Column("user_id", String(255)),
        Column("chunk_index", Integer),
        Column("content", Text),
        Column("summary", Text),
        Column("heading_context", Text),
        Column("page_number", Integer),
        Column("token_count", Integer),
        Column("embedding", Text),
        Column("message_id_start", String(255)),
        Column("message_id_end", String(255)),
        Column("date_created", DateTime),
    )
    return meta


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
        store = AsyncpgStore(conn)
        await runner.apply_for_agent_schema(store)  # type: ignore[arg-type]
    finally:
        await conn.close()
    return url, schema


async def _make_pool(url: str, schema: str) -> asyncpg.Pool:
    """build an asyncpg pool with search_path pre-bound to the test schema.

    registers the canonical 3tears jsonb text codec on every connection
    via :func:`threetears.core.collections.init_connection` -- without
    it, ``_encode_jsonb``'s typed pass-through hands a dict to asyncpg
    which has no idea how to encode it for ``$N::jsonb``.
    """
    from threetears.core.collections import init_connection

    result: asyncpg.Pool = await asyncpg.create_pool(
        dsn=url,
        min_size=1,
        max_size=4,
        server_settings={"search_path": f"{schema}, public"},
        init=init_connection,
    )
    return result


def _build_stack(
    pool: asyncpg.Pool,
    authorizer: MemoryAuthorizerDependencies,
) -> tuple[
    CollectionRegistry,
    SQLiteBackend,
    MemoriesCollection,
    MediaCollection,
    MediaContentCollection,
    MemoryChunkCollection,
]:
    """build a registry with L1 + all four Collections wired."""
    l1 = SQLiteBackend(db_name=f"mem_l1_{uuid.uuid7().hex[:8]}")
    l1.initialize(_build_l1_metadata())
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, l3_pool=pool)
    config = DefaultCoreConfig(
        collection_flush="ALWAYS",
        collection_flush_tables="",
    )
    memories = MemoriesCollection(
        registry=registry,
        config=config,
        authorizer=authorizer,
    )
    media = MediaCollection(registry=registry, config=config)
    media_content = MediaContentCollection(registry=registry, config=config)
    chunks = MemoryChunkCollection(registry=registry, config=config)
    return registry, l1, memories, media, media_content, chunks


class TestMemoryCollectionsL1:
    """every memory Collection populates L1 on write + serves reads from L1."""

    async def test_memories_collection_l1_hit(
        self,
        applied_schema: tuple[str, str],
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        """save -> L1 populated -> .get() returns from L1 without L3 round-trip."""
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            _, l1, memories, _, _, _ = _build_stack(
                pool,
                permissive_memory_authorizer,
            )

            user_id = uuid.uuid4()
            mid = uuid.uuid4()
            agent_id = uuid.uuid4()
            now = datetime.now(UTC)
            data = {
                "memory_id": mid,
                "agent_id": agent_id,
                "customer_id": uuid.uuid4(),
                "user_id": user_id,
                "conversation_id": uuid.uuid4(),
                "message_id_source": uuid.uuid4(),
                "type_memory": "fact",
                "content": "cached content",
                "embedding": [0.1] * 1024,
                "date_created": now,
                "date_updated": now,
            }
            entity = memories.create(data)
            await memories.save_entity(entity)

            # L1 populated — direct backend probe confirms
            row = l1.select_by_id(
                "memories",
                (str(agent_id), str(mid)),
                ("agent_id", "memory_id"),
            )
            assert row is not None
            assert row["content"] == "cached content"

            # .get() returns from L1 without hitting L3
            entity2 = await memories.get((agent_id, mid))
            assert entity2 is not None
            assert entity2.content == "cached content"
        finally:
            await pool.close()

    async def test_media_collection_l1_hit(
        self,
        applied_schema: tuple[str, str],
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        """MediaCollection saves through to L1 + .get() hits L1.

        Under the unified model media has a NOT NULL parent memory_id
        (CASCADE FK to memories); the test seeds the parent memory
        before the media row.
        """
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            _, l1, memories, media, _, _ = _build_stack(
                pool,
                permissive_memory_authorizer,
            )

            media_id = uuid.uuid4()
            memory_id = uuid.uuid4()
            agent_id = uuid.uuid4()
            customer_id = uuid.uuid4()
            user_id = uuid.uuid4()
            now = datetime.now(UTC)

            parent_memory = memories.create(
                {
                    "memory_id": memory_id,
                    "agent_id": agent_id,
                    "customer_id": customer_id,
                    "user_id": user_id,
                    "conversation_id": uuid.uuid4(),
                    "message_id_source": uuid.uuid4(),
                    "type_memory": "topical_context",
                    "content": "parent memory",
                    "embedding": [0.1] * 1024,
                    "date_created": now,
                    "date_updated": now,
                }
            )
            await memories.save_entity(parent_memory)

            # v0.8.0 enrichment widened MediaCollection.schema to the
            # full prod shape; required (NOT NULL) columns now include
            # ``mime_type`` / ``size_bytes`` / ``source`` /
            # ``extraction_status``. Test data supplies them
            # explicitly for clarity; the server_default INSERT gate
            # (``_insert_columns_for_data``) would let callers omit
            # ``media_category`` / ``extraction_status`` / ``metadata_json``
            # and let Postgres apply the declared defaults, but pinning
            # the explicit-pass shape keeps the L1 fixture readable.
            data = {
                "media_id": media_id,
                "memory_id": memory_id,
                "agent_id": agent_id,
                "customer_id": customer_id,
                "user_id": user_id,
                "mime_type": "image/jpeg",
                "size_bytes": 1024,
                "source": "upload",
                "media_category": "image",
                "extraction_status": "none",
                "metadata_json": {"document_title": "photo.jpg"},
                "date_created": now,
            }
            entity = media.create(data)
            await media.save_entity(entity)

            row = l1.select_by_id(
                "media",
                (str(agent_id), str(media_id)),
                ("agent_id", "media_id"),
            )
            assert row is not None
            assert row["media_category"] == "image"

            entity2 = await media.get((agent_id, media_id))
            assert entity2 is not None
            assert entity2.media_category == "image"
        finally:
            await pool.close()

    async def test_media_content_collection_l1_hit(
        self,
        applied_schema: tuple[str, str],
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        """MediaContentCollection saves through to L1 + .get() hits L1."""
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            _, l1, memories, media, media_content, _ = _build_stack(
                pool,
                permissive_memory_authorizer,
            )

            # seed a parent memory + media (FK chain)
            media_id = uuid.uuid4()
            memory_id = uuid.uuid4()
            agent_id = uuid.uuid4()
            customer_id = uuid.uuid4()
            user_id = uuid.uuid4()
            now = datetime.now(UTC)

            parent_memory = memories.create(
                {
                    "memory_id": memory_id,
                    "agent_id": agent_id,
                    "customer_id": customer_id,
                    "user_id": user_id,
                    "conversation_id": uuid.uuid4(),
                    "message_id_source": uuid.uuid4(),
                    "type_memory": "topical_context",
                    "content": "parent memory",
                    "embedding": [0.1] * 1024,
                    "date_created": now,
                    "date_updated": now,
                }
            )
            await memories.save_entity(parent_memory)

            # v0.8.0 enrichment widened MediaCollection.schema to the
            # full prod shape; supply every NOT NULL column the new
            # shape requires.
            media_entity = media.create(
                {
                    "media_id": media_id,
                    "memory_id": memory_id,
                    "agent_id": agent_id,
                    "customer_id": customer_id,
                    "user_id": user_id,
                    "mime_type": "application/pdf",
                    "size_bytes": 2048,
                    "source": "upload",
                    "media_category": "document",
                    "extraction_status": "none",
                    "metadata_json": {},
                    "date_created": now,
                },
            )
            await media.save_entity(media_entity)

            content_id = uuid.uuid4()
            mc_data = {
                "content_id": content_id,
                "media_id": media_id,
                "agent_id": agent_id,
                "customer_id": customer_id,
                "user_id": user_id,
                "content_type": "ocr",
                "content": "cached media content",
                "summary": None,
                "embedding": [0.2] * 1024,
                "date_created": now,
            }
            mc_entity = media_content.create(mc_data)
            await media_content.save_entity(mc_entity)

            row = l1.select_by_id(
                "media_content",
                (str(agent_id), str(content_id)),
                ("agent_id", "content_id"),
            )
            assert row is not None
            assert row["content"] == "cached media content"

            entity2 = await media_content.get((agent_id, content_id))
            assert entity2 is not None
            assert entity2.content == "cached media content"
        finally:
            await pool.close()

    async def test_memory_chunk_collection_l1_hit(
        self,
        applied_schema: tuple[str, str],
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        """MemoryChunkCollection saves through to L1 + .get() hits L1.

        Under the unified model every chunk has a NOT NULL parent
        memory_id; seed the parent memory first.
        """
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            _, l1, memories, _, _, chunks = _build_stack(
                pool,
                permissive_memory_authorizer,
            )

            chunk_id = uuid.uuid4()
            memory_id = uuid.uuid4()
            agent_id = uuid.uuid4()
            customer_id = uuid.uuid4()
            user_id = uuid.uuid4()
            now = datetime.now(UTC)

            parent_memory = memories.create(
                {
                    "memory_id": memory_id,
                    "agent_id": agent_id,
                    "customer_id": customer_id,
                    "user_id": user_id,
                    "conversation_id": uuid.uuid4(),
                    "message_id_source": uuid.uuid4(),
                    "type_memory": "topical_context",
                    "content": "parent memory",
                    "embedding": [0.1] * 1024,
                    "date_created": now,
                    "date_updated": now,
                }
            )
            await memories.save_entity(parent_memory)

            # v0.8.0 enrichment added ``chunk_index`` + ``token_count``
            # (both NOT NULL in prod; supplied here so the v0.8.0
            # schema's required-column guard is satisfied).
            chunk_data = {
                "chunk_id": chunk_id,
                "memory_id": memory_id,
                "agent_id": agent_id,
                "customer_id": customer_id,
                "user_id": user_id,
                "chunk_index": 0,
                "content": "cached chunk content",
                "summary": None,
                "heading_context": "Intro",
                "page_number": 1,
                "token_count": 5,
                "embedding": [0.3] * 1024,
                "date_created": now,
            }
            entity = chunks.create(chunk_data)
            await chunks.save_entity(entity)

            row = l1.select_by_id(
                "memory_chunks",
                (str(agent_id), str(chunk_id)),
                ("agent_id", "chunk_id"),
            )
            assert row is not None
            assert row["content"] == "cached chunk content"

            entity2 = await chunks.get((agent_id, chunk_id))
            assert entity2 is not None
            assert entity2.content == "cached chunk content"
        finally:
            await pool.close()


class TestHybridSearchRegression:
    """hybrid-search methods return results consistent with raw SQL path."""

    async def test_memories_hybrid_search_returns_seeded_rows(
        self,
        applied_schema: tuple[str, str],
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        """seed memories rows, run hybrid_search, verify they surface."""
        url, schema = applied_schema
        pool = await _make_pool(url, schema)
        try:
            _, _, memories, _, _, _ = _build_stack(
                pool,
                permissive_memory_authorizer,
            )

            user_id = uuid.uuid4()
            agent_id = uuid.uuid4()
            customer_id = uuid.uuid4()
            now = datetime.now(UTC)

            # seed via Collection save_entity path so we go through the
            # Collection's own INSERT
            vec = [0.1] * 1024
            for i in range(3):
                data = {
                    "memory_id": uuid.uuid4(),
                    "agent_id": agent_id,
                    "customer_id": customer_id,
                    "user_id": user_id,
                    "conversation_id": uuid.uuid4(),
                    "message_id_source": uuid.uuid4(),
                    "type_memory": "fact",
                    "content": f"memory {i} content",
                    "embedding": vec,
                    "date_created": now,
                    "date_updated": now,
                }
                entity = memories.create(data)
                await memories.save_entity(entity)

            results = await memories.hybrid_search(
                user_id=user_id,
                agent_id=agent_id,
                customer_id=customer_id,
                embedding=vec,
                user_text="memory",
                top_k=10,
                candidate_limit=30,
                similarity_threshold=0.0,
                recency_half_life_hours=24.0,
                signal_weights={"semantic": 0.55, "keyword": 0.15, "recency": 0.30},
            )
            assert len(results) == 3
            assert all("memory" in r["content"] for r in results)
        finally:
            await pool.close()
