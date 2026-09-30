"""
integration test: the migrations build what the collections declare.

Builds the package's migrations into fresh schemas of one database and reads
the catalog back. Both sides are derived from code: the built side from
``pg_catalog``, the declared side from every ``SchemaBackedCollection`` in
:mod:`threetears.agent.memory.collections`, so a new index on either side
without the other fails here.

Also pinned:

- v028 moves a foreign key bound to a legacy unique constraint onto the
  declared one, drops a single-column ``media.memory_id`` key, and a second
  run changes nothing.
- two agent schemas in one database each get their own unique constraints
  (v022's guards once found the first schema's constraint and skipped).
- v029/v030: a chunk's heading outranks its summary in search, and rows
  written under v007's trigger are recomputed.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import asyncpg
import pytest

from threetears.agent.memory import collections as memory_collections
from threetears.agent.memory.authorize import MemoryAuthorizerDependencies
from threetears.agent.memory.collections import MemoryChunkCollection
from threetears.agent.memory.migrations import align_indexes_with_declarations
from threetears.agent.memory.migrations import register as register_memory
from threetears.conversations.migrations import register as register_conversations
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import SchemaBackedCollection, TableSchema
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner

from .conftest import AsyncpgStore

pytestmark = pytest.mark.integration


#: the last version whose unique-constraint guards v022 carries; the two-schema
#: test stops there so it reads v022's own work, not v028's repair of it.
_V022 = 22
#: the version before v028, the index and constraint alignment.
_BEFORE_ALIGNMENT = 27
#: the version before v029, the chunk heading weighting.
_BEFORE_HEADING_WEIGHT = 28


def _declared_schemas() -> list[TableSchema]:
    """every ``TableSchema`` a collection in the memory package declares.

    :return: the schemas, one per collection class defined in the module
    :rtype: list[TableSchema]
    """
    result = [
        obj.schema
        for obj in vars(memory_collections).values()
        if isinstance(obj, type)
        and issubclass(obj, SchemaBackedCollection)
        and obj.__module__ == memory_collections.__name__
    ]
    return result


def _runner() -> MigrationRunner:
    """a runner with conversations and agent-memory registered.

    :return: the runner
    :rtype: MigrationRunner
    """
    runner = MigrationRunner()
    register_conversations(runner)
    register_memory(runner)
    return runner


async def _migrate(conn: asyncpg.Connection, schema: str, target: int | None = None) -> None:
    """apply the chain into ``schema``, up to ``target`` when given.

    :param conn: connection to the test database
    :ptype conn: asyncpg.Connection
    :param schema: agent schema to migrate
    :ptype schema: str
    :param target: inclusive version cap, or ``None`` for all
    :ptype target: int | None
    """
    await conn.execute(f'SET search_path TO "{schema}", public')
    await _runner().apply_for_agent_schema(AsyncpgStore(conn), target=target)  # type: ignore[arg-type]


async def _index_names(conn: asyncpg.Connection, schema: str, table: str) -> set[str]:
    """names of every non-primary-key index on ``schema.table``.

    a unique constraint is backed by an index of its own name, so this set
    holds the declared indexes and the declared unique constraints alike.

    :param conn: connection to the test database
    :ptype conn: asyncpg.Connection
    :param schema: schema holding the table
    :ptype schema: str
    :param table: table name
    :ptype table: str
    :return: index names
    :rtype: set[str]
    """
    rows = await conn.fetch(
        """
        SELECT ic.relname
          FROM pg_catalog.pg_index i
          JOIN pg_catalog.pg_class ic ON ic.oid = i.indexrelid
          JOIN pg_catalog.pg_class tc ON tc.oid = i.indrelid
          JOIN pg_catalog.pg_namespace ns ON ns.oid = tc.relnamespace
         WHERE ns.nspname = $1 AND tc.relname = $2 AND NOT i.indisprimary
        """,
        schema,
        table,
    )
    result = {r["relname"] for r in rows}
    return result


async def _unique_constraint_names(conn: asyncpg.Connection, schema: str, table: str) -> set[str]:
    """names of the UNIQUE constraints on ``schema.table``.

    :param conn: connection to the test database
    :ptype conn: asyncpg.Connection
    :param schema: schema holding the table
    :ptype schema: str
    :param table: table name
    :ptype table: str
    :return: constraint names
    :rtype: set[str]
    """
    rows = await conn.fetch(
        """
        SELECT con.conname
          FROM pg_catalog.pg_constraint con
          JOIN pg_catalog.pg_class tc ON tc.oid = con.conrelid
          JOIN pg_catalog.pg_namespace ns ON ns.oid = tc.relnamespace
         WHERE ns.nspname = $1 AND tc.relname = $2 AND con.contype = 'u'
        """,
        schema,
        table,
    )
    result = {r["conname"] for r in rows}
    return result


async def _catalog_snapshot(conn: asyncpg.Connection, schema: str) -> list[tuple[str, str, str]]:
    """every index and constraint definition in ``schema``, for replay comparison.

    :param conn: connection to the test database
    :ptype conn: asyncpg.Connection
    :param schema: schema to read
    :ptype schema: str
    :return: sorted ``(table, name, definition)`` rows
    :rtype: list[tuple[str, str, str]]
    """
    indexes = await conn.fetch(
        "SELECT tablename, indexname, indexdef FROM pg_indexes WHERE schemaname = $1",
        schema,
    )
    constraints = await conn.fetch(
        """
        SELECT tc.relname, con.conname, pg_get_constraintdef(con.oid) AS def
          FROM pg_catalog.pg_constraint con
          JOIN pg_catalog.pg_class tc ON tc.oid = con.conrelid
          JOIN pg_catalog.pg_namespace ns ON ns.oid = tc.relnamespace
         WHERE ns.nspname = $1
        """,
        schema,
    )
    result = sorted(
        [(r["tablename"], r["indexname"], r["indexdef"]) for r in indexes]
        + [(r["relname"], r["conname"], r["def"]) for r in constraints]
    )
    return result


async def _memory_foreign_keys_on_media(conn: asyncpg.Connection, schema: str) -> list[tuple[str, ...]]:
    """the local column lists of every foreign key from ``media`` to ``memories``.

    :param conn: connection to the test database
    :ptype conn: asyncpg.Connection
    :param schema: schema holding the tables
    :ptype schema: str
    :return: one column tuple per key
    :rtype: list[tuple[str, ...]]
    """
    rows = await conn.fetch(
        """
        SELECT ARRAY(
                 SELECT att.attname
                   FROM unnest(con.conkey) WITH ORDINALITY AS k(attnum, ord)
                   JOIN pg_catalog.pg_attribute att
                     ON att.attrelid = con.conrelid AND att.attnum = k.attnum
                  ORDER BY k.ord
               ) AS cols
          FROM pg_catalog.pg_constraint con
          JOIN pg_catalog.pg_class tc ON tc.oid = con.conrelid
          JOIN pg_catalog.pg_class rc ON rc.oid = con.confrelid
          JOIN pg_catalog.pg_namespace ns ON ns.oid = tc.relnamespace
         WHERE ns.nspname = $1 AND tc.relname = 'media' AND rc.relname = 'memories' AND con.contype = 'f'
        """,
        schema,
    )
    result = sorted(tuple(r["cols"]) for r in rows)
    return result


async def _assert_matches_declarations(conn: asyncpg.Connection, schema: str) -> None:
    """assert every declared table's indexes and unique constraints are exactly what ``schema`` holds.

    :param conn: connection to the test database
    :ptype conn: asyncpg.Connection
    :param schema: migrated agent schema
    :ptype schema: str
    """
    declared_schemas = _declared_schemas()
    assert declared_schemas, "found no SchemaBackedCollection in the memory package"
    for declared in declared_schemas:
        declared_unique = {uc.name for uc in declared.unique_constraints}
        declared_indexes = {ix.name for ix in declared.indexes} | declared_unique
        assert await _index_names(conn, schema, declared.name) == declared_indexes, declared.name
        assert await _unique_constraint_names(conn, schema, declared.name) == declared_unique, declared.name
        rows = await conn.fetch(
            "SELECT column_name, is_nullable FROM information_schema.columns WHERE table_schema = $1 AND table_name = $2",
            schema,
            declared.name,
        )
        built_nullable = {r["column_name"]: r["is_nullable"] == "YES" for r in rows}
        declared_nullable = {c.name: c.nullable for c in declared.columns if c.name in built_nullable}
        assert declared_nullable == {k: built_nullable[k] for k in declared_nullable}, declared.name


@pytest.fixture
async def two_schemas(pg_url: str) -> AsyncIterator[tuple[str, str, str]]:
    """two fresh agent schemas in the one test database.

    :param pg_url: testcontainer url
    :ptype pg_url: str
    :return: ``(url, first schema, second schema)``
    :rtype: tuple[str, str, str]
    """
    first = f"mem_decl_a_{uuid.uuid4().hex[:12]}"
    second = f"mem_decl_b_{uuid.uuid4().hex[:12]}"
    conn = await asyncpg.connect(pg_url)
    try:
        await conn.execute('CREATE EXTENSION IF NOT EXISTS "vector"')
        await conn.execute(f'CREATE SCHEMA "{first}"')
        await conn.execute(f'CREATE SCHEMA "{second}"')
    finally:
        await conn.close()
    yield (pg_url, first, second)
    conn = await asyncpg.connect(pg_url)
    try:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{first}" CASCADE')
        await conn.execute(f'DROP SCHEMA IF EXISTS "{second}" CASCADE')
    finally:
        await conn.close()


class TestBuiltSchemaMatchesDeclarations:
    """the full chain builds the indexes and unique constraints the collections declare."""

    async def test_fresh_schema_matches_every_declaration(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _migrate(conn, schema)
            await _assert_matches_declarations(conn, schema)
            assert await _memory_foreign_keys_on_media(conn, schema) == [("agent_id", "memory_id")]
        finally:
            await conn.close()

    async def test_memories_date_updated_is_not_null_on_both_sides(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _migrate(conn, schema)
            nullable = await conn.fetchval(
                """
                SELECT is_nullable FROM information_schema.columns
                 WHERE table_schema = $1 AND table_name = 'memories' AND column_name = 'date_updated'
                """,
                schema,
            )
        finally:
            await conn.close()
        declared = next(c for c in memory_collections.MemoriesCollection.schema.columns if c.name == "date_updated")
        assert nullable == "NO"
        assert declared.nullable is False

    async def test_alignment_replay_changes_nothing(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _migrate(conn, schema)
            before = await _catalog_snapshot(conn, schema)
            await align_indexes_with_declarations(AsyncpgStore(conn))  # type: ignore[arg-type]
            assert await _catalog_snapshot(conn, schema) == before
        finally:
            await conn.close()

    async def test_key_bound_to_a_legacy_unique_moves_to_the_declared_one(self, pg_schema: tuple[str, str]) -> None:
        """a single-column key into ``memories(memory_id)`` survives v028, bound to ``uq_memories_memory_id``.

        and a single-column ``media.memory_id`` key is dropped, leaving the composite.
        """
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _migrate(conn, schema, target=_BEFORE_ALIGNMENT)
            await conn.execute(
                "CREATE TABLE memory_pointer (memory_id UUID REFERENCES memories (memory_id) ON DELETE CASCADE)"
            )
            await conn.execute(
                "ALTER TABLE media ADD CONSTRAINT media_memory_id_single_fk "
                "FOREIGN KEY (memory_id) REFERENCES memories (memory_id)"
            )
            bound_query = """
                SELECT ic.relname
                  FROM pg_catalog.pg_constraint con
                  JOIN pg_catalog.pg_class ic ON ic.oid = con.conindid
                  JOIN pg_catalog.pg_class tc ON tc.oid = con.conrelid
                  JOIN pg_catalog.pg_namespace ns ON ns.oid = tc.relnamespace
                 WHERE ns.nspname = $1 AND tc.relname = 'memory_pointer' AND con.contype = 'f'
            """
            # precondition: the key is bound to the legacy constraint, so the
            # drop below has something to move.
            assert await conn.fetchval(bound_query, schema) == "memories_memory_id_unique"
            assert ("memory_id",) in await _memory_foreign_keys_on_media(conn, schema)

            await _migrate(conn, schema)

            assert await conn.fetchval(bound_query, schema) == "uq_memories_memory_id"
            assert await _memory_foreign_keys_on_media(conn, schema) == [("agent_id", "memory_id")]
            assert "memories_memory_id_unique" not in await _unique_constraint_names(conn, schema, "memories")
        finally:
            await conn.close()


class TestTwoAgentSchemasInOneDatabase:
    """each agent schema gets its own unique constraints."""

    async def test_each_schema_has_its_own_unique_constraints(self, two_schemas: tuple[str, str, str]) -> None:
        url, first, second = two_schemas
        declared_unique = {
            declared.name: {uc.name for uc in declared.unique_constraints}
            for declared in _declared_schemas()
            if declared.unique_constraints
        }
        assert declared_unique, "no declared unique constraints to check"
        conn = await asyncpg.connect(url)
        try:
            # through v022: its guards must look in their own schema only.
            for schema in (first, second):
                await _migrate(conn, schema, target=_V022)
            for schema in (first, second):
                for table, names in declared_unique.items():
                    built = await _unique_constraint_names(conn, schema, table)
                    assert names <= built, f"{schema}.{table} is missing {sorted(names - built)}"
            # and the whole chain leaves each schema exactly as declared.
            for schema in (first, second):
                await _migrate(conn, schema)
            for schema in (first, second):
                await _assert_matches_declarations(conn, schema)
        finally:
            await conn.close()


async def _make_pool(url: str, schema: str) -> asyncpg.Pool:
    """an asyncpg pool pinned to ``schema``.

    :param url: testcontainer url
    :ptype url: str
    :param schema: agent schema
    :ptype schema: str
    :return: the pool
    :rtype: asyncpg.Pool
    """
    from threetears.core.collections import init_connection

    pool: asyncpg.Pool = await asyncpg.create_pool(
        dsn=url,
        min_size=1,
        max_size=4,
        server_settings={"search_path": f"{schema}, public"},
        init=init_connection,
    )
    return pool


async def _insert_memory(conn: asyncpg.Connection, *, agent_id: uuid.UUID, user_id: uuid.UUID) -> uuid.UUID:
    """insert one parent memory row directly and return its id.

    :param conn: connection with the agent schema on its search_path
    :ptype conn: asyncpg.Connection
    :param agent_id: agent partition
    :ptype agent_id: uuid.UUID
    :param user_id: owning user
    :ptype user_id: uuid.UUID
    :return: the memory id
    :rtype: uuid.UUID
    """
    memory_id = uuid.uuid4()
    now = datetime.now(UTC)
    await conn.execute(
        """
        INSERT INTO memories (memory_id, agent_id, customer_id, user_id, conversation_id,
                              type_memory, content, date_created, date_updated)
        VALUES ($1, $2, $3, $4, $5, 'fact', 'parent', $6, $6)
        """,
        memory_id,
        agent_id,
        uuid.uuid4(),
        user_id,
        uuid.uuid4(),
        now,
    )
    return memory_id


class TestChunkHeadingWeight:
    """a chunk's heading is its strongest search signal."""

    async def test_heading_match_outranks_summary_match(
        self,
        pg_schema: tuple[str, str],
        permissive_memory_authorizer: MemoryAuthorizerDependencies,
    ) -> None:
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _migrate(conn, schema)
        finally:
            await conn.close()
        pool = await _make_pool(url, schema)
        try:
            registry = CollectionRegistry()
            registry.configure(l3_pool=pool)
            config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
            memories = memory_collections.MemoriesCollection(
                registry=registry,
                config=config,
                authorizer=permissive_memory_authorizer,
            )
            chunks = MemoryChunkCollection(registry=registry, config=config)
            agent_id, customer_id, user_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
            now = datetime.now(UTC)
            memory_id = uuid.uuid4()
            await memories.save_entity(
                memories.create(
                    {
                        "memory_id": memory_id,
                        "agent_id": agent_id,
                        "customer_id": customer_id,
                        "user_id": user_id,
                        "conversation_id": uuid.uuid4(),
                        "message_id_source": uuid.uuid4(),
                        "type_memory": "topical_context",
                        "content": "parent",
                        "embedding": [0.1] * 1024,
                        "date_created": now,
                        "date_updated": now,
                    }
                )
            )

            async def seed(*, content: str, summary: str | None, heading: str | None, index: int) -> uuid.UUID:
                chunk_id = uuid.uuid4()
                await chunks.save_entity(
                    chunks.create(
                        {
                            "chunk_id": chunk_id,
                            "memory_id": memory_id,
                            "agent_id": agent_id,
                            "customer_id": customer_id,
                            "user_id": user_id,
                            "chunk_index": index,
                            "content": content,
                            "summary": summary,
                            "heading_context": heading,
                            "page_number": None,
                            "token_count": 1,
                            "embedding": [0.1] * 1024,
                            "message_id_start": None,
                            "message_id_end": None,
                            "date_created": now,
                        }
                    )
                )
                return chunk_id

            summary_chunk = await seed(
                content="notes about the budget meeting",
                summary="zebra crossing near the office",
                heading="Budget",
                index=0,
            )
            heading_chunk = await seed(
                content="notes about the quarterly plan",
                summary="plan review",
                heading="Zebra crossing",
                index=1,
            )

            result = await chunks.hybrid_search_within_memory(
                memory_id=memory_id,
                user_id=user_id,
                agent_id=agent_id,
                customer_id=customer_id,
                embedding=[0.1] * 1024,
                user_text="zebra crossing",
                candidate_k=10,
                similarity_threshold=0.0,
                chunk_signal_weights={"semantic": 0.5, "keyword": 0.5},
            )
        finally:
            await pool.close()

        # the search returns chunk ids as text.
        by_id = {str(row["chunk_id"]): row for row in result}
        assert set(by_id) == {str(heading_chunk), str(summary_chunk)}
        assert str(result[0]["chunk_id"]) == str(heading_chunk)
        assert by_id[str(heading_chunk)]["hybrid_score"] > by_id[str(summary_chunk)]["hybrid_score"]

    async def test_rows_written_before_the_weighting_are_recomputed(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _migrate(conn, schema, target=_BEFORE_HEADING_WEIGHT)
            agent_id, user_id = uuid.uuid4(), uuid.uuid4()
            memory_id = await _insert_memory(conn, agent_id=agent_id, user_id=user_id)
            chunk_id = uuid.uuid4()
            await conn.execute(
                """
                INSERT INTO memory_chunks (chunk_id, agent_id, memory_id, user_id, chunk_index,
                                           content, summary, heading_context, token_count, date_created)
                VALUES ($1, $2, $3, $4, 0, 'body text', 'short summary', 'Zebra crossing', 1, now())
                """,
                chunk_id,
                agent_id,
                memory_id,
                user_id,
            )
            match_sql = "SELECT search_vector @@ to_tsquery('english', 'zebra') FROM memory_chunks WHERE chunk_id = $1"
            # precondition: v007's trigger leaves the heading out.
            assert await conn.fetchval(match_sql, chunk_id) is False

            await _migrate(conn, schema)

            assert await conn.fetchval(match_sql, chunk_id) is True
            weighted = await conn.fetchval(
                "SELECT search_vector::text FROM memory_chunks WHERE chunk_id = $1",
                chunk_id,
            )
            # heading at A, content at B, summary at C: 'Zebra crossing' is
            # positions 1-2, 'body text' 3-4, 'short summary' 5-6.
            assert "'zebra':1A" in weighted
            assert "'bodi':3B" in weighted
            assert "'summari':6C" in weighted
        finally:
            await conn.close()


class TestCustomerIdNotNull:
    """v031 tightens ``customer_id`` where it can, and leaves a table holding a NULL as it is."""

    async def test_a_table_holding_a_null_customer_id_is_left_nullable(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _migrate(conn, schema, target=30)
            agent_id, user_id = uuid.uuid4(), uuid.uuid4()
            memory_id = await _insert_memory(conn, agent_id=agent_id, user_id=user_id)
            now = datetime.now(UTC)
            await conn.execute(
                "INSERT INTO media (media_id, memory_id, agent_id, customer_id, user_id, "
                "media_category, metadata_json, date_created, date_updated) "
                "VALUES ($1, $2, $3, NULL, $4, 'document', '{}'::jsonb, $5, $5)",
                uuid.uuid4(),
                memory_id,
                agent_id,
                user_id,
                now,
            )
            await _migrate(conn, schema)
            rows = await conn.fetch(
                "SELECT table_name, is_nullable FROM information_schema.columns "
                "WHERE table_schema = $1 AND column_name = 'customer_id' "
                "AND table_name IN ('media', 'media_content', 'memory_chunks')",
                schema,
            )
        finally:
            await conn.close()
        assert {r["table_name"]: r["is_nullable"] for r in rows} == {
            "media": "YES",
            "media_content": "NO",
            "memory_chunks": "NO",
        }


class TestAnAdoptedSchemaConverges:
    """a schema a consumer built with its own chain, then adopted, reaches the declarations too."""

    async def test_media_gets_the_composite_key_and_date_updated_its_not_null(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        conn = await asyncpg.connect(url)
        try:
            await _migrate(conn, schema, target=27)
            # the consumer's shape: media keyed to memories by memory_id alone, and a nullable
            # memories.date_updated holding a NULL
            await conn.execute("ALTER TABLE media DROP CONSTRAINT media_memory_fk")
            await conn.execute(
                "ALTER TABLE media ADD CONSTRAINT media_memory_id_fkey "
                "FOREIGN KEY (memory_id) REFERENCES memories (memory_id) ON DELETE CASCADE"
            )
            await conn.execute("ALTER TABLE memories ALTER COLUMN date_updated DROP NOT NULL")
            memory_id = await _insert_memory(conn, agent_id=uuid.uuid4(), user_id=uuid.uuid4())
            await conn.execute("UPDATE memories SET date_updated = NULL WHERE memory_id = $1", memory_id)

            await _migrate(conn, schema)

            assert await _memory_foreign_keys_on_media(conn, schema) == [("agent_id", "memory_id")]
            nullable = await conn.fetchval(
                "SELECT is_nullable FROM information_schema.columns "
                "WHERE table_schema = $1 AND table_name = 'memories' AND column_name = 'date_updated'",
                schema,
            )
            assert nullable == "NO"
            filled = await conn.fetchval(
                "SELECT date_updated = date_created FROM memories WHERE memory_id = $1", memory_id
            )
            assert filled is True
        finally:
            await conn.close()
