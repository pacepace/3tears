"""integration: agent-intention's embedding width, configured, verified and resized on real PostgreSQL.

- a schema migrated with ``embedding_dim=1536`` has a ``vector(1536)`` column, and a 1536-wide want
  saves and is found by the dedup search;
- a 1024 schema verified against 1536 raises :class:`EmbeddingWidthMismatch` naming ``intentions``;
- ``resize_embedding_width`` clears the stored embeddings and says how many, makes the column the
  new width, and rebuilds ``ix_intentions_embedding_hnsw`` under its own name and definition;
- a write of another width is refused with nothing written;
- PostgreSQL refuses a width above 2,000 before any DDL.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import asyncpg
import pytest

from threetears.agent.intention.collections import IntentionsCollection
from threetears.agent.intention.embedding_width import (
    EmbeddingWidthMismatch,
    resize_embedding_width,
    verify_embedding_width,
)
from threetears.agent.intention.migrations import register as register_intention
from threetears.conversations.migrations import register as register_conversations
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationFailedError, MigrationRunner

from .conftest import AsyncpgStore, make_pool

pytestmark = pytest.mark.integration


async def _migrate(url: str, schema: str, **register_kwargs: Any) -> None:
    runner = MigrationRunner()
    register_conversations(runner)
    register_intention(runner, **register_kwargs)
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


def _intentions(pool: asyncpg.Pool, embedding_dim: int) -> IntentionsCollection:
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return IntentionsCollection(registry=registry, config=config, embedding_dim=embedding_dim)


def _row(agent_id: uuid.UUID, user_id: uuid.UUID, embedding: list[float]) -> dict[str, Any]:
    now = datetime.now(UTC)
    return {
        "intention_id": uuid.uuid4(),
        "agent_id": agent_id,
        "user_id": user_id,
        "status": "open",
        "content": "learn the cello",
        "embedding": embedding,
        "date_created": now,
        "date_updated": now,
    }


async def _column_type(conn: asyncpg.Connection, schema: str) -> str | None:
    value: str | None = await conn.fetchval(
        """
        SELECT format_type(a.atttypid, a.atttypmod)
          FROM pg_catalog.pg_attribute a
          JOIN pg_catalog.pg_class c ON c.oid = a.attrelid
          JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
         WHERE n.nspname = $1 AND c.relname = 'intentions' AND a.attname = 'embedding' AND NOT a.attisdropped
        """,
        schema,
    )
    return value


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


async def _insert(conn: asyncpg.Connection, embedding: str | None) -> None:
    await conn.execute(
        "INSERT INTO intentions (intention_id, agent_id, user_id, content, embedding, date_created, date_updated)"
        " VALUES ($1, $2, $3, 'held', $4::text::public.vector, now(), now())",
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
            assert await _column_type(conn, schema) == "vector(1536)"
            await verify_embedding_width(AsyncpgStore(conn), 1536)  # type: ignore[arg-type]
        finally:
            await conn.close()
        pool = await make_pool(url, schema)
        try:
            intentions = _intentions(pool, 1536)
            agent_id, user_id = uuid.uuid4(), uuid.uuid4()
            row = _row(agent_id, user_id, [0.1] * 1536)
            await intentions.save_entity(intentions.create(row))
            found = await intentions.find_similar_for_dedup(
                user_id=user_id, agent_id=agent_id, embedding=[0.1] * 1536, top_k=5, threshold=0.5
            )
            assert [r["intention_id"] for r in found] == [row["intention_id"]]
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
            assert await _column_type(conn, schema) is None, "the table was created before the width was refused"
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
        assert raised.value.table == "intentions"
        assert "intentions.embedding" in str(raised.value)
        assert "resize_embedding_width" in str(raised.value)

    async def test_resize_clears_the_embeddings_and_rebuilds_the_index(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        await _migrate(url, schema)
        conn = await _connect(url, schema)
        try:
            await _insert(conn, "[" + ",".join(["0.2"] * 1024) + "]")
            await _insert(conn, None)
            before = await _embedding_index_definitions(conn, schema)
            assert set(before) == {"ix_intentions_embedding_hnsw"}
            options_before = await _embedding_index_options(conn, schema)

            cleared = await resize_embedding_width(AsyncpgStore(conn), 1536)  # type: ignore[arg-type]

            assert cleared == 1
            assert await _column_type(conn, schema) == "vector(1536)"
            assert await conn.fetchval("SELECT count(*) FROM intentions") == 2
            assert await conn.fetchval("SELECT count(*) FROM intentions WHERE embedding IS NOT NULL") == 0
            assert await _embedding_index_definitions(conn, schema) == before
            assert await _embedding_index_options(conn, schema) == options_before
        finally:
            await conn.close()


class TestAWriteOfAnotherWidth:
    async def test_is_refused_and_nothing_is_written(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        await _migrate(url, schema, embedding_dim=1536)
        pool = await make_pool(url, schema)
        try:
            intentions = _intentions(pool, 1536)
            with pytest.raises(ValueError, match=r"expected 1536, got 1024"):
                await intentions.save_entity(intentions.create(_row(uuid.uuid4(), uuid.uuid4(), [0.1] * 1024)))
            assert await pool.fetchval("SELECT count(*) FROM intentions") == 0
        finally:
            await pool.close()
