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

            # v012 composite FK from memories to media on (agent_id, media_id)
            assert await _constraint_exists(
                conn,
                schema,
                "memories_media_composite_fk",
            )

            # re-apply is a no-op
            count2 = await runner.apply_for_agent_schema(store)  # type: ignore[arg-type]
            assert count2 == 0
        finally:
            await conn.close()


class TestV012MemoriesMediaCompositeFK:
    """v012 composite FK semantics: SET NULL on media delete + reject orphans."""

    async def test_v012_fk_declares_on_delete_set_null(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """v012 FK metadata declares ``ON DELETE SET NULL`` semantics.

        memory-references-media is a soft relationship: deleting the
        source media should null the memory's reference, not cascade-
        delete the memory itself. ``ON DELETE CASCADE`` would drop
        extracted facts whenever the source artifact was removed --
        the wrong data semantic. this test verifies the constraint
        in pg_constraint declares the expected delete action, locking
        the choice against accidental migration to CASCADE on a
        future rewrite.

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
            # 'n' = SET NULL, 'd' = SET DEFAULT. asyncpg returns this
            # postgres ``"char"`` column as a single-byte ``bytes`` object.
            row = await conn.fetchrow(
                """
                SELECT confdeltype
                  FROM pg_constraint c
                  JOIN pg_namespace ns ON ns.oid = c.connamespace
                 WHERE ns.nspname = $1
                   AND c.conname = 'memories_media_composite_fk'
                """,
                schema,
            )
            assert row is not None
            assert row["confdeltype"] == b"n", f"expected SET NULL (b'n') ON DELETE; got {row['confdeltype']!r}"
        finally:
            await conn.close()

    async def test_media_delete_nulls_referencing_memory(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """deleting a media row flips referencing ``memories.media_id`` to NULL.

        the v012 FK declares ``ON DELETE SET NULL`` matching the
        memory-references-media semantic: the extracted memory still
        exists after the source media is removed. test inserts a
        media + memory pair, deletes the media, observes the memory
        survives with media_id = NULL.

        the metadata-shape leg of this contract is pinned by
        :meth:`test_v012_fk_declares_on_delete_set_null` -- this
        method exercises the runtime behaviour over an actual
        ``DELETE FROM media``.

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
            agent_id = uuid.uuid4()
            customer_id = uuid.uuid4()
            user_id = uuid.uuid4()
            media_id = uuid.uuid4()
            memory_id = uuid.uuid4()

            await conn.execute(
                "INSERT INTO media ("
                "media_id, agent_id, customer_id, user_id, "
                "media_category, date_created, date_updated"
                ") VALUES ($1, $2, $3, $4, 'document', $5, $5)",
                media_id,
                agent_id,
                customer_id,
                user_id,
                now,
            )

            await conn.execute(
                "INSERT INTO memories ("
                "memory_id, agent_id, customer_id, user_id, "
                "conversation_id, message_id_source, type_memory, content, "
                "summary, embedding, is_deleted, media_id, "
                "date_created, date_updated"
                ") VALUES ($1, $2, $3, $4, $5, $6, 'fact', $7, $8, "
                "$9::vector, FALSE, $10, $11, $11)",
                memory_id,
                agent_id,
                customer_id,
                user_id,
                uuid.uuid4(),
                uuid.uuid4(),
                "fact text content body",
                "fact summary",
                "[" + ",".join(["0.1"] * 1024) + "]",
                media_id,
                now,
            )

            # confirm the link is present before the delete
            pre_row = await conn.fetchrow(
                "SELECT media_id FROM memories WHERE agent_id = $1 AND memory_id = $2",
                agent_id,
                memory_id,
            )
            assert pre_row is not None
            assert pre_row["media_id"] == media_id

            # deleting the media row fires ON DELETE SET NULL on
            # ``media_id`` only (PG 15+ column-list form) -- agent_id
            # stays populated so the partition discipline holds.
            await conn.execute(
                "DELETE FROM media WHERE agent_id = $1 AND media_id = $2",
                agent_id,
                media_id,
            )

            row = await conn.fetchrow(
                "SELECT agent_id, media_id FROM memories WHERE agent_id = $1 AND memory_id = $2",
                agent_id,
                memory_id,
            )
            assert row is not None
            assert row["media_id"] is None
            assert row["agent_id"] == agent_id  # partition not nulled
        finally:
            await conn.close()

    async def test_orphan_media_id_rejected(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """inserting a memory with non-existent ``(agent_id, media_id)`` fails.

        the v012 composite FK rejects orphan ``media_id`` references
        at INSERT time. test attempts to insert a memory pointing at
        a never-existed media row and observes the FK violation.

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
            agent_id = uuid.uuid4()
            ghost_media_id = uuid.uuid4()  # never inserted

            customer_id = uuid.uuid4()
            user_id = uuid.uuid4()
            memory_id = uuid.uuid4()
            with pytest.raises(asyncpg.exceptions.ForeignKeyViolationError):
                await conn.execute(
                    "INSERT INTO memories ("
                    "memory_id, agent_id, customer_id, user_id, "
                    "conversation_id, message_id_source, type_memory, "
                    "content, summary, embedding, is_deleted, media_id, "
                    "date_created, date_updated"
                    ") VALUES ($1, $2, $3, $4, $5, $6, 'fact', $7, $8, "
                    "$9::vector, FALSE, $10, $11, $11)",
                    memory_id,
                    agent_id,
                    customer_id,
                    user_id,
                    uuid.uuid4(),
                    uuid.uuid4(),
                    "fact text",
                    "fact summary",
                    "[" + ",".join(["0.1"] * 1024) + "]",
                    ghost_media_id,
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
                "SELECT count(*) FROM memories WHERE search_vector @@ websearch_to_tsquery('english', 'fox')"
            )
            assert match == 1
        finally:
            await conn.close()
