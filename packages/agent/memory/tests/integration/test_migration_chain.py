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


async def _columns(conn: asyncpg.Connection, schema: str, table: str) -> dict[str, str]:
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
        "SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = $1 AND table_name = $2",
        schema,
        table,
    )
    result = {r["column_name"]: r["data_type"] for r in rows}
    return result


async def _table_exists(conn: asyncpg.Connection, schema: str, table: str) -> bool:
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
        "SELECT 1 FROM information_schema.tables WHERE table_schema = $1 AND table_name = $2",
        schema,
        table,
    )
    result = row is not None
    return result


async def _index_exists(conn: asyncpg.Connection, schema: str, index_name: str) -> bool:
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


async def _constraint_exists(
    conn: asyncpg.Connection,
    schema: str,
    constraint_name: str,
) -> bool:
    """
    return whether ``schema.constraint_name`` exists in pg_constraint.

    :param conn: live asyncpg connection
    :ptype conn: asyncpg.Connection
    :param schema: schema to check
    :ptype schema: str
    :param constraint_name: constraint name to check
    :ptype constraint_name: str
    :return: True if constraint exists
    :rtype: bool
    """
    row = await conn.fetchrow(
        """
        SELECT 1 FROM pg_constraint c
          JOIN pg_namespace ns ON ns.oid = c.connamespace
         WHERE ns.nspname = $1
           AND c.conname = $2
        """,
        schema,
        constraint_name,
    )
    result = row is not None
    return result


class TestFullChainApplies:
    """v001-v007 apply cleanly producing the expected schema."""

    async def test_chain_applies_and_produces_expected_schema(self, pg_schema: tuple[str, str]) -> None:
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

            # memories table reconciled — transcript-chunks-task-A
            # unified-memory shape after v018 (drop media_id +
            # is_deleted + date_deleted) and v019 (NOT NULL
            # conversation_id).
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
            # v018 dropped the soft-delete + reverse-FK columns; the
            # unified model is hard-delete only and media now parents
            # memory via media.memory_id (NOT the reverse).
            assert "is_deleted" not in memory_cols
            assert "media_id" not in memory_cols
            assert "date_deleted" not in memory_cols
            assert "summary" in memory_cols
            assert "search_vector" in memory_cols

            # conversation_memory_refs table
            assert await _table_exists(conn, schema, "conversation_memory_refs")

            # media + media_content tables — v015 added media.memory_id,
            # v017 made it NOT NULL + CASCADE FK to memories.
            media_cols = await _columns(conn, schema, "media")
            assert "media_id" in media_cols
            assert "memory_id" in media_cols
            assert "media_category" in media_cols
            assert "metadata_json" in media_cols
            mc_cols = await _columns(conn, schema, "media_content")
            assert "content_id" in mc_cols
            assert "embedding" in mc_cols
            assert "search_vector" in mc_cols

            # memory_chunks table — v015 added memory_id +
            # message_id_start + message_id_end, v018 dropped media_id.
            chunk_cols = await _columns(conn, schema, "memory_chunks")
            assert "chunk_id" in chunk_cols
            assert "memory_id" in chunk_cols
            assert "message_id_start" in chunk_cols
            assert "message_id_end" in chunk_cols
            assert "media_id" not in chunk_cols
            assert "heading_context" in chunk_cols
            assert "page_number" in chunk_cols
            assert "embedding" in chunk_cols
            assert "search_vector" in chunk_cols

            # FTS indexes present
            assert await _index_exists(conn, schema, "idx_mem_search_vector")
            assert await _index_exists(conn, schema, "idx_mc_search_vector")
            assert await _index_exists(conn, schema, "idx_chunks_search_vector")

            # unified-memory parent FKs (v017) — chunks -> memories and
            # media -> memories, both CASCADE on memory delete.
            assert await _constraint_exists(
                conn,
                schema,
                "memory_chunks_memory_fk",
            )
            assert await _constraint_exists(
                conn,
                schema,
                "media_memory_fk",
            )

            # v018 dropped the reverse-direction FKs alongside the
            # columns they referenced.
            assert not await _constraint_exists(
                conn,
                schema,
                "memories_media_composite_fk",
            )
            assert not await _constraint_exists(
                conn,
                schema,
                "memory_chunks_media_fk",
            )

            # re-apply is a no-op
            count2 = await runner.apply_for_agent_schema(store)  # type: ignore[arg-type]
            assert count2 == 0
        finally:
            await conn.close()


class TestUnifiedMemoryParentFks:
    """v017 parent FK semantics: memory delete CASCADEs to chunks + media."""

    async def test_chunks_fk_declares_on_delete_cascade(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """``memory_chunks_memory_fk`` declares ``ON DELETE CASCADE``.

        The unified model treats memory as the cognitive anchor;
        deleting it discards every chunk parented to it. CASCADE is
        the correct delete action — anything else would leave dangling
        chunks pointing at a vanished memory.

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

            # confdeltype: 'a' = NO ACTION, 'r' = RESTRICT, 'c' = CASCADE,
            # 'n' = SET NULL, 'd' = SET DEFAULT.
            row = await conn.fetchrow(
                """
                SELECT confdeltype
                  FROM pg_constraint c
                  JOIN pg_namespace ns ON ns.oid = c.connamespace
                 WHERE ns.nspname = $1
                   AND c.conname = 'memory_chunks_memory_fk'
                """,
                schema,
            )
            assert row is not None
            assert row["confdeltype"] == b"c", f"expected CASCADE (b'c'); got {row['confdeltype']!r}"
        finally:
            await conn.close()

    async def test_media_fk_declares_on_delete_cascade(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """``media_memory_fk`` declares ``ON DELETE CASCADE``."""
        url, schema = pg_schema
        runner = _build_runner()
        conn = await asyncpg.connect(url)
        try:
            await conn.execute(f'SET search_path TO "{schema}", public')
            store = AsyncpgStore(conn)
            await runner.apply_for_agent_schema(store)  # type: ignore[arg-type]

            row = await conn.fetchrow(
                """
                SELECT confdeltype
                  FROM pg_constraint c
                  JOIN pg_namespace ns ON ns.oid = c.connamespace
                 WHERE ns.nspname = $1
                   AND c.conname = 'media_memory_fk'
                """,
                schema,
            )
            assert row is not None
            assert row["confdeltype"] == b"c", f"expected CASCADE (b'c'); got {row['confdeltype']!r}"
        finally:
            await conn.close()

    async def test_memory_delete_cascades_to_chunk(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """Deleting a memory cascades to every chunk parented to it.

        Inserts a memory + chunk pair, deletes the memory, observes
        the chunk has vanished.
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

            now = datetime.now(UTC)
            agent_id = uuid.uuid4()
            customer_id = uuid.uuid4()
            user_id = uuid.uuid4()
            memory_id = uuid.uuid4()
            chunk_id = uuid.uuid4()

            await conn.execute(
                "INSERT INTO memories ("
                "memory_id, agent_id, customer_id, user_id, "
                "conversation_id, message_id_source, type_memory, content, "
                "summary, embedding, date_created, date_updated"
                ") VALUES ($1, $2, $3, $4, $5, $6, 'topical_context', $7, $8, "
                "$9::vector, $10, $10)",
                memory_id,
                agent_id,
                customer_id,
                user_id,
                uuid.uuid4(),
                uuid.uuid4(),
                "memory content",
                "memory summary",
                "[" + ",".join(["0.1"] * 1024) + "]",
                now,
            )

            await conn.execute(
                "INSERT INTO memory_chunks ("
                "chunk_id, memory_id, agent_id, customer_id, user_id, "
                "content, summary, embedding, date_created"
                ") VALUES ($1, $2, $3, $4, $5, $6, $7, $8::vector, $9)",
                chunk_id,
                memory_id,
                agent_id,
                customer_id,
                user_id,
                "chunk text",
                "chunk summary",
                "[" + ",".join(["0.1"] * 1024) + "]",
                now,
            )

            await conn.execute(
                "DELETE FROM memories WHERE agent_id = $1 AND memory_id = $2",
                agent_id,
                memory_id,
            )

            row = await conn.fetchrow(
                "SELECT chunk_id FROM memory_chunks WHERE chunk_id = $1",
                chunk_id,
            )
            assert row is None, "memory delete must cascade-delete its chunks"
        finally:
            await conn.close()

    async def test_memory_delete_cascades_through_media_chain(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """Deleting a memory cascades to its media + media_content.

        Chain: memory --(media.memory_id CASCADE)--> media --(media_
        content.media_id CASCADE, v006)--> media_content. Seeds the
        full chain, deletes the root memory, observes both descendant
        rows vanish.
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

            now = datetime.now(UTC)
            agent_id = uuid.uuid4()
            customer_id = uuid.uuid4()
            user_id = uuid.uuid4()
            memory_id = uuid.uuid4()
            media_id = uuid.uuid4()
            content_id = uuid.uuid4()

            await conn.execute(
                "INSERT INTO memories ("
                "memory_id, agent_id, customer_id, user_id, "
                "conversation_id, message_id_source, type_memory, content, "
                "summary, embedding, date_created, date_updated"
                ") VALUES ($1, $2, $3, $4, $5, $6, 'topical_context', $7, $8, "
                "$9::vector, $10, $10)",
                memory_id,
                agent_id,
                customer_id,
                user_id,
                uuid.uuid4(),
                uuid.uuid4(),
                "wraps a media artifact",
                "memory summary",
                "[" + ",".join(["0.1"] * 1024) + "]",
                now,
            )

            await conn.execute(
                "INSERT INTO media ("
                "media_id, memory_id, agent_id, customer_id, user_id, "
                "media_category, metadata_json, date_created, date_updated"
                ") VALUES ($1, $2, $3, $4, $5, 'document', NULL, $6, $6)",
                media_id,
                memory_id,
                agent_id,
                customer_id,
                user_id,
                now,
            )

            await conn.execute(
                "INSERT INTO media_content ("
                "content_id, media_id, agent_id, customer_id, user_id, "
                "content_type, content, summary, embedding, date_created"
                ") VALUES ($1, $2, $3, $4, $5, 'ocr', $6, NULL, $7::vector, $8)",
                content_id,
                media_id,
                agent_id,
                customer_id,
                user_id,
                "extracted text",
                "[" + ",".join(["0.1"] * 1024) + "]",
                now,
            )

            # delete the root memory; the entire chain should vanish
            await conn.execute(
                "DELETE FROM memories WHERE agent_id = $1 AND memory_id = $2",
                agent_id,
                memory_id,
            )

            media_row = await conn.fetchrow("SELECT media_id FROM media WHERE media_id = $1", media_id)
            assert media_row is None, "memory delete must cascade-delete its media"

            content_row = await conn.fetchrow(
                "SELECT content_id FROM media_content WHERE content_id = $1",
                content_id,
            )
            assert content_row is None, "memory delete must cascade through media to media_content"
        finally:
            await conn.close()

    async def test_orphan_memory_id_rejected(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """Inserting a chunk with non-existent parent memory_id fails.

        The CASCADE FK rejects orphan ``memory_id`` references at
        INSERT time.
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

            now = datetime.now(UTC)
            agent_id = uuid.uuid4()
            customer_id = uuid.uuid4()
            user_id = uuid.uuid4()
            chunk_id = uuid.uuid4()
            ghost_memory_id = uuid.uuid4()  # never inserted

            with pytest.raises(asyncpg.exceptions.ForeignKeyViolationError):
                await conn.execute(
                    "INSERT INTO memory_chunks ("
                    "chunk_id, memory_id, agent_id, customer_id, user_id, "
                    "content, summary, embedding, date_created"
                    ") VALUES ($1, $2, $3, $4, $5, $6, $7, $8::vector, $9)",
                    chunk_id,
                    ghost_memory_id,
                    agent_id,
                    customer_id,
                    user_id,
                    "orphan chunk",
                    "orphan summary",
                    "[" + ",".join(["0.1"] * 1024) + "]",
                    now,
                )
        finally:
            await conn.close()


class TestFtsTriggerPopulatesVector:
    """the memory FTS trigger populates ``search_vector`` on INSERT / UPDATE."""

    async def test_insert_populates_search_vector(self, pg_schema: tuple[str, str]) -> None:
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

            now = datetime.now(UTC)
            mem_id = uuid.uuid4()
            await conn.execute(
                "INSERT INTO memories ("
                "memory_id, agent_id, customer_id, user_id, "
                "conversation_id, message_id_source, type_memory, content, "
                "summary, embedding, date_created, date_updated"
                ") VALUES ($1, $2, $3, $4, $5, $6, 'fact', $7, $8, "
                "$9::vector, $10, $10)",
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
                "SELECT count(*) FROM memories WHERE search_vector @@ websearch_to_tsquery('english', 'fox')"
            )
            assert match == 1
        finally:
            await conn.close()
