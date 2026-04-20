"""
integration test: apply the full memory migration chain.

memory-task-01. proves the package's migrations (v001-v007) run cleanly
against a fresh pgvector/pg16 database, that every column + table + FTS
artifact the source code references exists after the chain applies, and
that replay is a no-op.

the shard ``memory-task-01-schema-reconciliation.md`` lists the exact
column / table inventory; this test asserts that inventory against the
live schema.
"""

from __future__ import annotations

import asyncpg
import pytest

from threetears.agent.memory.migrations import register as register_memory
from threetears.conversations.migrations import register as register_conversations
from threetears.core.data.migrations import MigrationRunner

from .conftest import AsyncpgStore


pytestmark = pytest.mark.integration


def _build_runner() -> MigrationRunner:
    """
    register conversations + agent-memory migrations on a fresh runner.

    agent-memory declares ``depends_on=("conversations",)`` so the
    conversations package has to be registered too.

    :return: runner with both packages registered
    :rtype: MigrationRunner
    """
    runner = MigrationRunner()
    register_conversations(runner)
    register_memory(runner)
    return runner


async def _columns(
    conn: asyncpg.Connection, schema: str, table: str
) -> dict[str, str]:
    """
    return column_name -> data_type for the given table.

    :param conn: live asyncpg connection
    :ptype conn: asyncpg.Connection
    :param schema: schema containing the table
    :ptype schema: str
    :param table: table to introspect
    :ptype table: str
    :return: mapping of column_name to SQL data_type
    :rtype: dict[str, str]
    """
    rows = await conn.fetch(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_schema = $1 AND table_name = $2",
        schema,
        table,
    )
    result = {r["column_name"]: r["data_type"] for r in rows}
    return result


async def _table_exists(
    conn: asyncpg.Connection, schema: str, table: str
) -> bool:
    """
    return whether ``schema.table`` exists in information_schema.

    :param conn: live asyncpg connection
    :ptype conn: asyncpg.Connection
    :param schema: schema to check
    :ptype schema: str
    :param table: table to check
    :ptype table: str
    :return: True if table exists
    :rtype: bool
    """
    row = await conn.fetchrow(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema = $1 AND table_name = $2",
        schema,
        table,
    )
    result = row is not None
    return result


async def _index_exists(
    conn: asyncpg.Connection, schema: str, index_name: str
) -> bool:
    """
    return whether ``schema.index_name`` exists in pg_indexes.

    :param conn: live asyncpg connection
    :ptype conn: asyncpg.Connection
    :param schema: schema to check
    :ptype schema: str
    :param index_name: index to check
    :ptype index_name: str
    :return: True if index exists
    :rtype: bool
    """
    row = await conn.fetchrow(
        "SELECT 1 FROM pg_indexes WHERE schemaname = $1 AND indexname = $2",
        schema,
        index_name,
    )
    result = row is not None
    return result


class TestFullChainApplies:
    """v001-v007 apply cleanly producing the expected schema."""

    async def test_chain_applies_and_produces_expected_schema(
        self, pg_schema: tuple[str, str]
    ) -> None:
        """
        applying v001-v007 yields every column + table + index the code
        expects. re-applying is a no-op.

        :param pg_schema: (url, schema) tuple from conftest fixture
        :ptype pg_schema: tuple[str, str]
        """
        url, schema = pg_schema
        runner = _build_runner()
        conn = await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            store = AsyncpgStore(conn)
            count = await runner.apply_for_agent_schema(store)  # type: ignore[arg-type]
            assert count > 0

            # memories table reconciled
            memory_cols = await _columns(conn, schema, "memories")
            assert "memory_id" in memory_cols
            assert "type_memory" in memory_cols
            assert "id" not in memory_cols
            assert "memory_type" not in memory_cols
            assert "embedding_model" not in memory_cols
            assert "importance" not in memory_cols
            assert "metadata" not in memory_cols
            assert "date_accessed" not in memory_cols
            assert "conversation_id" in memory_cols
            assert "message_id_source" in memory_cols
            assert "is_deleted" in memory_cols
            assert "media_id" in memory_cols
            assert "date_deleted" in memory_cols
            assert "summary" in memory_cols
            assert "search_vector" in memory_cols

            # conversation_memory_refs table
            assert await _table_exists(conn, schema, "conversation_memory_refs")

            # media + media_content tables
            media_cols = await _columns(conn, schema, "media")
            assert "media_id" in media_cols
            assert "media_category" in media_cols
            assert "metadata_json" in media_cols
            mc_cols = await _columns(conn, schema, "media_content")
            assert "content_id" in mc_cols
            assert "embedding" in mc_cols
            assert "search_vector" in mc_cols

            # memory_chunks table
            chunk_cols = await _columns(conn, schema, "memory_chunks")
            assert "chunk_id" in chunk_cols
            assert "heading_context" in chunk_cols
            assert "page_number" in chunk_cols
            assert "embedding" in chunk_cols
            assert "search_vector" in chunk_cols

            # FTS indexes present
            assert await _index_exists(conn, schema, "idx_mem_search_vector")
            assert await _index_exists(conn, schema, "idx_mc_search_vector")
            assert await _index_exists(conn, schema, "idx_chunks_search_vector")

            # re-apply is a no-op
            count2 = await runner.apply_for_agent_schema(store)  # type: ignore[arg-type]
            assert count2 == 0
        finally:
            await conn.close()


class TestFtsTriggerPopulatesVector:
    """the memory FTS trigger populates ``search_vector`` on INSERT / UPDATE."""

    async def test_insert_populates_search_vector(
        self, pg_schema: tuple[str, str]
    ) -> None:
        """
        inserting a row into memories fills search_vector automatically.

        :param pg_schema: (url, schema) tuple
        :ptype pg_schema: tuple[str, str]
        """
        url, schema = pg_schema
        runner = _build_runner()
        conn = await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            store = AsyncpgStore(conn)
            await runner.apply_for_agent_schema(store)  # type: ignore[arg-type]

            import uuid
            from datetime import UTC, datetime

            now = datetime.now(UTC).replace(tzinfo=None)
            mem_id = uuid.uuid4()
            await conn.execute(
                "INSERT INTO memories ("
                "memory_id, agent_id, customer_id, user_id, "
                "conversation_id, message_id_source, type_memory, content, "
                "summary, embedding, is_deleted, date_created, date_updated"
                ") VALUES ($1, $2, $3, $4, $5, $6, 'fact', $7, $8, "
                "$9::vector, FALSE, $10, $10)",
                mem_id,
                uuid.uuid4(),
                uuid.uuid4(),
                uuid.uuid4(),
                uuid.uuid4(),
                uuid.uuid4(),
                "The quick brown fox jumps over lazy dogs",
                "quick fox summary",
                "[" + ",".join(["0.1"] * 1024) + "]",
                now,
            )
            row = await conn.fetchrow(
                "SELECT search_vector FROM memories WHERE memory_id = $1",
                mem_id,
            )
            assert row is not None
            assert row["search_vector"] is not None

            # FTS query should find it
            match = await conn.fetchval(
                "SELECT count(*) FROM memories "
                "WHERE search_vector @@ websearch_to_tsquery('english', 'fox')"
            )
            assert match == 1
        finally:
            await conn.close()
