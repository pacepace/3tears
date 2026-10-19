"""integration: agent-memory's embedding width, configured, verified and resized on real PostgreSQL.

- a schema migrated with ``embedding_dim=1536`` has ``vector(1536)`` columns on memories,
  media_content and memory_chunks, and a 1536-dimensional memory saves and is found by
  ``search_by_semantic``;
- a schema built at 1024 and verified against 1536 raises :class:`EmbeddingWidthMismatch` naming
  the table;
- ``resize_embedding_width`` clears the stored embeddings and says how many, makes every column the
  new width, and rebuilds each HNSW index under its own name and definition; a second run, or a
  run after a stop between dropping an index and creating it, finishes the job;
- a write of another width is refused with nothing written, before PostgreSQL could refuse it;
- PostgreSQL refuses a width above 2,000 before any DDL.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import asyncpg
import pytest

from threetears.agent.memory.authorize import MemoryAuthorizerDependencies
from threetears.agent.memory.collections import MemoriesCollection
from threetears.agent.memory.embedding_width import (
    EmbeddingWidthMismatch,
    resize_embedding_width,
    verify_embedding_width,
)
from threetears.agent.memory.migrations import register as register_memory
from threetears.conversations.migrations import register as register_conversations
from threetears.core.collections import init_connection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationFailedError, MigrationRunner

from .conftest import AsyncpgStore

pytestmark = pytest.mark.integration

_EMBEDDING_TABLES = ("memories", "media_content", "memory_chunks")
_EMBEDDING_INDEXES = {"ix_memories_embedding_hnsw", "ix_media_content_embedding", "ix_memory_chunks_embedding"}


async def _migrate(url: str, schema: str, **register_kwargs: Any) -> None:
    runner = MigrationRunner()
    register_conversations(runner)
    register_memory(runner, **register_kwargs)
    conn = await asyncpg.connect(url)
    try:
        await conn.execute(f'SET search_path TO "{schema}", public')
        await runner.apply_for_agent_schema(AsyncpgStore(conn))  # type: ignore[arg-type]
    finally:
        await conn.close()


async def _connect(url: str, schema: str) -> asyncpg.Connection:
    conn: asyncpg.Connection = await asyncpg.connect(url)
    await conn.execute(f'SET search_path TO "{schema}", public')
    return conn


async def _pool(url: str, schema: str) -> asyncpg.Pool:
    pool: asyncpg.Pool = await asyncpg.create_pool(
        dsn=url, min_size=1, max_size=4, server_settings={"search_path": f"{schema}, public"}, init=init_connection
    )
    return pool


def _memories(pool: asyncpg.Pool, embedding_dim: int) -> MemoriesCollection:
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return MemoriesCollection(
        registry=registry,
        config=config,
        authorizer=MagicMock(spec=MemoryAuthorizerDependencies),
        embedding_dim=embedding_dim,
    )


def _memory_row(agent_id: uuid.UUID, user_id: uuid.UUID, embedding: list[float] | None) -> dict[str, Any]:
    now = datetime.now(UTC)
    return {
        "memory_id": uuid.uuid4(),
        "agent_id": agent_id,
        "customer_id": uuid.uuid4(),
        "user_id": user_id,
        "conversation_id": uuid.uuid4(),
        "type_memory": "fact",
        "content": "prefers green tea",
        "embedding": embedding,
        "date_created": now,
        "date_updated": now,
    }


async def _column_types(conn: asyncpg.Connection, schema: str) -> dict[str, str]:
    rows = await conn.fetch(
        """
        SELECT c.relname AS table_name, format_type(a.atttypid, a.atttypmod) AS column_type
          FROM pg_catalog.pg_attribute a
          JOIN pg_catalog.pg_class c ON c.oid = a.attrelid
          JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
         WHERE n.nspname = $1 AND a.attname = 'embedding' AND c.relkind = 'r' AND NOT a.attisdropped
        """,
        schema,
    )
    return {r["table_name"]: r["column_type"] for r in rows}


async def _embedding_index_definitions(conn: asyncpg.Connection, schema: str) -> dict[str, str]:
    rows = await conn.fetch(
        "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = $1 AND indexdef LIKE '%(embedding%'",
        schema,
    )
    return {r["indexname"]: r["indexdef"] for r in rows}


async def _embedding_index_options(conn: asyncpg.Connection, schema: str) -> dict[str, list[str] | None]:
    rows = await conn.fetch(
        """
        SELECT ic.relname, ic.reloptions
          FROM pg_catalog.pg_class ic
          JOIN pg_catalog.pg_namespace n ON n.oid = ic.relnamespace
          JOIN pg_catalog.pg_indexes x ON x.schemaname = n.nspname AND x.indexname = ic.relname
         WHERE n.nspname = $1 AND x.indexdef LIKE '%(embedding%'
        """,
        schema,
    )
    return {r["relname"]: r["reloptions"] for r in rows}


async def _insert_memory(conn: asyncpg.Connection, embedding: str | None) -> None:
    await conn.execute(
        "INSERT INTO memories (memory_id, agent_id, customer_id, user_id, conversation_id, type_memory, content,"
        " embedding, date_created, date_updated)"
        " VALUES ($1, $2, $3, $4, $5, 'fact', 'held', $6::text::public.vector, now(), now())",
        uuid.uuid4(),
        uuid.uuid4(),
        uuid.uuid4(),
        uuid.uuid4(),
        uuid.uuid4(),
        embedding,
    )


class TestAConfiguredWidth:
    async def test_a_schema_migrated_at_1536_holds_and_finds_1536_embeddings(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        await _migrate(url, schema, embedding_dim=1536)
        conn = await _connect(url, schema)
        try:
            types = await _column_types(conn, schema)
            assert {table: types.get(table) for table in _EMBEDDING_TABLES} == dict.fromkeys(
                _EMBEDDING_TABLES, "vector(1536)"
            )
            await verify_embedding_width(AsyncpgStore(conn), 1536)  # type: ignore[arg-type]
        finally:
            await conn.close()
        pool = await _pool(url, schema)
        try:
            memories = _memories(pool, 1536)
            agent_id, user_id = uuid.uuid4(), uuid.uuid4()
            row = _memory_row(agent_id, user_id, [0.1] * 1536)
            await memories.save_entity(memories.create(row))
            found = await memories.search_by_semantic(
                user_id=user_id,
                agent_id=agent_id,
                embedding=[0.1] * 1536,
                max_results=5,
                similarity_threshold=0.5,
            )
            assert [r["memory_id"] for r in found] == [str(row["memory_id"])]
        finally:
            await pool.close()

    async def test_postgres_refuses_a_width_above_two_thousand_before_any_ddl(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        with pytest.raises(MigrationFailedError) as raised:
            await _migrate(url, schema, embedding_dim=2001)
        cause = raised.value.__cause__
        assert isinstance(cause, ValueError)
        assert "at most 2,000" in str(cause)
        assert "16,000" in str(cause)
        conn = await _connect(url, schema)
        try:
            assert await _column_types(conn, schema) == {}, "a table was created before the width was refused"
        finally:
            await conn.close()


class TestAnExistingSchema:
    async def test_a_1024_schema_verified_against_1536_names_the_table(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        await _migrate(url, schema)
        conn = await _connect(url, schema)
        try:
            store = AsyncpgStore(conn)
            await verify_embedding_width(store, 1024)  # type: ignore[arg-type]
            with pytest.raises(EmbeddingWidthMismatch) as raised:
                await verify_embedding_width(store, 1536)  # type: ignore[arg-type]
        finally:
            await conn.close()
        assert raised.value.table == "memories"
        assert (raised.value.stored_width, raised.value.configured_width) == (1024, 1536)
        message = str(raised.value)
        assert "memories.embedding" in message
        assert "resize_embedding_width" in message

    async def test_resize_clears_the_embeddings_and_rebuilds_the_indexes(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        await _migrate(url, schema)
        conn = await _connect(url, schema)
        try:
            held = "[" + ",".join(["0.2"] * 1024) + "]"
            await _insert_memory(conn, held)
            await _insert_memory(conn, held)
            await _insert_memory(conn, None)
            before = await _embedding_index_definitions(conn, schema)
            assert set(before) == _EMBEDDING_INDEXES
            options_before = await _embedding_index_options(conn, schema)
            assert options_before["ix_memories_embedding_hnsw"] == ["m=16", "ef_construction=64"]

            cleared = await resize_embedding_width(AsyncpgStore(conn), 1536)  # type: ignore[arg-type]

            assert cleared == 2
            types = await _column_types(conn, schema)
            assert {table: types.get(table) for table in _EMBEDDING_TABLES} == dict.fromkeys(
                _EMBEDDING_TABLES, "vector(1536)"
            )
            assert await conn.fetchval("SELECT count(*) FROM memories") == 3
            assert await conn.fetchval("SELECT count(*) FROM memories WHERE embedding IS NOT NULL") == 0
            assert await _embedding_index_definitions(conn, schema) == before
            assert await _embedding_index_options(conn, schema) == options_before
            await verify_embedding_width(AsyncpgStore(conn), 1536)  # type: ignore[arg-type]
            assert await resize_embedding_width(AsyncpgStore(conn), 1536) == 0  # type: ignore[arg-type]
        finally:
            await conn.close()
        pool = await _pool(url, schema)
        try:
            memories = _memories(pool, 1536)
            agent_id, user_id = uuid.uuid4(), uuid.uuid4()
            row = _memory_row(agent_id, user_id, [0.3] * 1536)
            await memories.save_entity(memories.create(row))
            found = await memories.search_by_semantic(
                user_id=user_id, agent_id=agent_id, embedding=[0.3] * 1536, max_results=5, similarity_threshold=0.5
            )
            assert [r["memory_id"] for r in found] == [str(row["memory_id"])]
        finally:
            await pool.close()

    async def test_a_resize_stopped_after_dropping_an_index_is_finished_by_the_next(
        self, pg_schema: tuple[str, str]
    ) -> None:
        url, schema = pg_schema
        await _migrate(url, schema)
        conn = await _connect(url, schema)
        try:
            before = await _embedding_index_definitions(conn, schema)
            # where a resize stopped: the column altered, its index dropped and never created again
            await conn.execute(f'DROP INDEX "{schema}".ix_memories_embedding_hnsw')
            await conn.execute(
                "ALTER TABLE memories ALTER COLUMN embedding TYPE public.vector(1536) USING NULL::public.vector(1536)"
            )

            await resize_embedding_width(AsyncpgStore(conn), 1536)  # type: ignore[arg-type]

            assert await _embedding_index_definitions(conn, schema) == before
        finally:
            await conn.close()


class TestAWriteOfAnotherWidth:
    @pytest.mark.parametrize(("schema_width", "written"), [(1536, 1024), (1024, 1536)])
    async def test_is_refused_and_nothing_is_written(
        self, pg_schema: tuple[str, str], schema_width: int, written: int
    ) -> None:
        url, schema = pg_schema
        await _migrate(url, schema, embedding_dim=schema_width)
        pool = await _pool(url, schema)
        try:
            memories = _memories(pool, schema_width)
            row = _memory_row(uuid.uuid4(), uuid.uuid4(), [0.1] * written)
            with pytest.raises(ValueError, match=rf"expected {schema_width}, got {written}"):
                await memories.save_entity(memories.create(row))
            assert await pool.fetchval("SELECT count(*) FROM memories") == 0
            assert await memories.get((row["agent_id"], row["memory_id"])) is None
        finally:
            await pool.close()
