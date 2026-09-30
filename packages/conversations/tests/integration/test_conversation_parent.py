"""integration: v011 -- a conversation records what started it -- against a real Postgres.

pins four things a fake cannot:

- the migration applies, and applies again, on the package's own schema;
- it applies to a ``conversations`` table shaped like a consumer's (single-column primary key,
  ``user_id_owner`` instead of ``user_id``, no package columns) and adds only the two columns,
  the check and the index;
- the check refuses a row carrying only one half of the pair;
- the pair round-trips through :class:`ConversationsCollection` into the entity accessors.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from uuid import UUID, uuid4

import asyncpg
import pytest

from threetears.conversations import ConversationsCollection
from threetears.conversations.migrations import add_conversation_parent
from threetears.conversations.migrations import register as register_conversations
from threetears.conversations.migrations.v011_conversation_parent import PARENT_CHECK_NAME, PARENT_INDEX_NAME
from threetears.core.backends.sql import SqlL3Backend
from threetears.core.collections import init_connection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner

from .conftest import AsyncpgStore

pytestmark = pytest.mark.integration

#: the consumer-shaped table the brief names: single-column pk, an owner column the package does
#: not have, and none of the package's own columns beyond ``name``.
_CONSUMER_TABLE_SQL = (
    "CREATE TABLE conversations ("
    "conversation_id uuid PRIMARY KEY, "
    "agent_id uuid NOT NULL, "
    "customer_id uuid NOT NULL, "
    "user_id_owner uuid NOT NULL, "
    "name text)"
)


async def _columns(conn: asyncpg.Connection, schema: str) -> dict[str, tuple[str, str]]:
    """return ``{column: (data_type, is_nullable)}`` for ``schema.conversations``.

    :param conn: live asyncpg connection
    :ptype conn: asyncpg.Connection
    :param schema: schema to read
    :ptype schema: str
    :return: column shape map
    :rtype: dict[str, tuple[str, str]]
    """
    rows = await conn.fetch(
        "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
        "WHERE table_schema = $1 AND table_name = 'conversations'",
        schema,
    )
    return {r["column_name"]: (r["data_type"], r["is_nullable"]) for r in rows}


async def _constraints(conn: asyncpg.Connection, schema: str) -> set[str]:
    """return every constraint name on ``schema.conversations``.

    :param conn: live asyncpg connection
    :ptype conn: asyncpg.Connection
    :param schema: schema to read
    :ptype schema: str
    :return: constraint names
    :rtype: set[str]
    """
    rows = await conn.fetch(
        "SELECT c.conname FROM pg_constraint c JOIN pg_class t ON t.oid = c.conrelid "
        "WHERE c.connamespace = $1::regnamespace AND t.relname = 'conversations'",
        schema,
    )
    return {r["conname"] for r in rows}


async def _indexes(conn: asyncpg.Connection, schema: str) -> dict[str, str]:
    """return ``{index_name: indexdef}`` for ``schema.conversations``.

    :param conn: live asyncpg connection
    :ptype conn: asyncpg.Connection
    :param schema: schema to read
    :ptype schema: str
    :return: index definitions
    :rtype: dict[str, str]
    """
    rows = await conn.fetch(
        "SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = $1 AND tablename = 'conversations'",
        schema,
    )
    return {r["indexname"]: r["indexdef"] for r in rows}


async def _connect(url: str, schema: str) -> asyncpg.Connection:
    """open a connection whose search_path selects ``schema``.

    :param url: database url
    :ptype url: str
    :param schema: schema to select
    :ptype schema: str
    :return: live connection
    :rtype: asyncpg.Connection
    """
    conn = await asyncpg.connect(url)
    await conn.execute(f'SET search_path TO "{schema}", public')
    return conn


class TestOnThePackageSchema:
    """v011 on the package's own ``conversations`` table."""

    async def test_applies_then_replays_cleanly(self, pg_schema: tuple[str, str]) -> None:
        """v010 has no parent; v011 adds it; a replay (runner and direct call) changes nothing.

        :param pg_schema: (url, schema) tuple
        :ptype pg_schema: tuple[str, str]
        """
        url, schema = pg_schema
        runner = MigrationRunner()
        register_conversations(runner)
        conn = await _connect(url, schema)
        try:
            store = AsyncpgStore(conn)
            await runner.apply_for_agent_schema(store, target=10)  # type: ignore[arg-type]
            before = await _columns(conn, schema)
            assert "parent_type" not in before and "parent_id" not in before

            assert await runner.apply_for_agent_schema(store) == 1  # type: ignore[arg-type]
            after = await _columns(conn, schema)
            assert after["parent_type"] == ("text", "YES")
            assert after["parent_id"] == ("uuid", "YES")
            assert PARENT_CHECK_NAME in await _constraints(conn, schema)
            assert PARENT_INDEX_NAME in await _indexes(conn, schema)

            constraints_once = await _constraints(conn, schema)
            indexes_once = await _indexes(conn, schema)
            assert await runner.apply_for_agent_schema(store) == 0  # type: ignore[arg-type]
            await add_conversation_parent(store)  # type: ignore[arg-type]
            assert await _columns(conn, schema) == after
            assert await _constraints(conn, schema) == constraints_once
            assert await _indexes(conn, schema) == indexes_once
        finally:
            await conn.close()


class TestOnAConsumerShapedTable:
    """v011 on a ``conversations`` table that is not the package's shape."""

    async def test_adds_only_the_columns_check_and_index(self, pg_schema: tuple[str, str]) -> None:
        """the diff against the table before is exactly two columns, one check, one index.

        :param pg_schema: (url, schema) tuple
        :ptype pg_schema: tuple[str, str]
        """
        url, schema = pg_schema
        conn = await _connect(url, schema)
        try:
            await conn.execute(_CONSUMER_TABLE_SQL)
            columns_before = await _columns(conn, schema)
            constraints_before = await _constraints(conn, schema)
            indexes_before = await _indexes(conn, schema)

            store = AsyncpgStore(conn)
            await add_conversation_parent(store)  # type: ignore[arg-type]
            await add_conversation_parent(store)  # type: ignore[arg-type]  # replay is clean here too

            columns_after = await _columns(conn, schema)
            assert {k: v for k, v in columns_after.items() if k not in columns_before} == {
                "parent_type": ("text", "YES"),
                "parent_id": ("uuid", "YES"),
            }
            assert {k: columns_after[k] for k in columns_before} == columns_before
            assert await _constraints(conn, schema) - constraints_before == {PARENT_CHECK_NAME}
            assert constraints_before <= await _constraints(conn, schema)
            indexes_after = await _indexes(conn, schema)
            assert set(indexes_after) - set(indexes_before) == {PARENT_INDEX_NAME}
            assert "(parent_type, parent_id)" in indexes_after[PARENT_INDEX_NAME]
            assert {k: indexes_after[k] for k in indexes_before} == indexes_before
        finally:
            await conn.close()

    @pytest.mark.parametrize(
        ("parent_type", "parent_id"),
        [("wake", None), (None, "set")],
        ids=["type-without-id", "id-without-type"],
    )
    async def test_check_refuses_half_a_parent(
        self, pg_schema: tuple[str, str], parent_type: str | None, parent_id: str | None
    ) -> None:
        """one half of the pair alone is refused; both and neither are accepted.

        :param pg_schema: (url, schema) tuple
        :ptype pg_schema: tuple[str, str]
        :param parent_type: the type word to write, or ``None``
        :ptype parent_type: str | None
        :param parent_id: ``"set"`` to write an id, or ``None``
        :ptype parent_id: str | None
        """
        url, schema = pg_schema
        conn = await _connect(url, schema)
        insert = (
            "INSERT INTO conversations (conversation_id, agent_id, customer_id, user_id_owner, parent_type, parent_id) "
            "VALUES ($1, $2, $3, $4, $5, $6)"
        )
        try:
            await conn.execute(_CONSUMER_TABLE_SQL)
            await add_conversation_parent(AsyncpgStore(conn))  # type: ignore[arg-type]
            with pytest.raises(asyncpg.CheckViolationError, match=PARENT_CHECK_NAME):
                await conn.execute(
                    insert,
                    uuid4(),
                    uuid4(),
                    uuid4(),
                    uuid4(),
                    parent_type,
                    uuid4() if parent_id else None,
                )
            # the two legal shapes, so the refusal above is the check and not the insert.
            await conn.execute(insert, uuid4(), uuid4(), uuid4(), uuid4(), None, None)
            await conn.execute(insert, uuid4(), uuid4(), uuid4(), uuid4(), "wake", uuid4())
            assert await conn.fetchval("SELECT count(*) FROM conversations") == 2
        finally:
            await conn.close()


@pytest.fixture
async def collection(pg_schema: tuple[str, str]) -> AsyncIterator[ConversationsCollection]:
    """a :class:`ConversationsCollection` over a fully migrated schema.

    :param pg_schema: (url, schema) tuple
    :ptype pg_schema: tuple[str, str]
    :return: live collection
    :rtype: AsyncIterator[ConversationsCollection]
    """
    url, schema = pg_schema
    conn = await _connect(url, schema)
    try:
        runner = MigrationRunner()
        register_conversations(runner)
        await runner.apply_for_agent_schema(AsyncpgStore(conn))  # type: ignore[arg-type]
    finally:
        await conn.close()
    pool = await asyncpg.create_pool(
        url,
        min_size=1,
        max_size=2,
        server_settings={"search_path": f'"{schema}",public'},
        init=init_connection,
    )
    assert pool is not None
    registry = CollectionRegistry()
    registry.configure(l3_pool=SqlL3Backend(pool))
    try:
        yield ConversationsCollection(registry, DefaultCoreConfig(), pool)
    finally:
        await pool.close()


def _new_row(parent_type: str | None, parent_id: UUID | None) -> dict[str, object]:
    """a complete new-conversation row carrying the given parent.

    :param parent_type: parent type word or ``None``
    :ptype parent_type: str | None
    :param parent_id: parent id or ``None``
    :ptype parent_id: UUID | None
    :return: row dict for :meth:`ConversationsCollection.create`
    :rtype: dict[str, object]
    """
    now = datetime.now(UTC)
    return {
        "agent_id": uuid4(),
        "conversation_id": uuid4(),
        "customer_id": uuid4(),
        "user_id": uuid4(),
        "channel_type": "web",
        "status": "active",
        "date_created": now,
        "date_updated": now,
        "message_count": 0,
        "parent_type": parent_type,
        "parent_id": parent_id,
    }


class TestRoundTripThroughTheCollection:
    """the pair survives a save and a read through the real collection."""

    async def test_parent_round_trips(self, collection: ConversationsCollection) -> None:
        """saved with a parent, read back from L3 with the same type and id.

        :param collection: live collection
        :ptype collection: ConversationsCollection
        """
        parent = uuid4()
        row = _new_row("wake", parent)
        await collection.save_entity(collection.create(row))
        key = (row["agent_id"], row["conversation_id"])
        collection.evict_from_cache_sync(key)

        stored = await collection.l3_pool.fetchrow(  # type: ignore[union-attr]
            "SELECT parent_type, parent_id FROM conversations WHERE agent_id = $1 AND conversation_id = $2",
            *key,
        )
        assert (stored["parent_type"], stored["parent_id"]) == ("wake", parent)

        read = await collection.get(key)
        assert read is not None
        assert read.parent_type == "wake"
        assert read.parent_id == parent

    async def test_no_parent_round_trips_as_none(self, collection: ConversationsCollection) -> None:
        """a conversation nothing recorded as starting it reads ``None`` for both.

        :param collection: live collection
        :ptype collection: ConversationsCollection
        """
        row = _new_row(None, None)
        await collection.save_entity(collection.create(row))
        key = (row["agent_id"], row["conversation_id"])
        collection.evict_from_cache_sync(key)

        read = await collection.get(key)
        assert read is not None
        assert read.parent_type is None
        assert read.parent_id is None

    async def test_parent_set_later_round_trips(self, collection: ConversationsCollection) -> None:
        """the pair is mutable: set on an existing conversation, it persists.

        :param collection: live collection
        :ptype collection: ConversationsCollection
        """
        row = _new_row(None, None)
        await collection.save_entity(collection.create(row))
        key = (row["agent_id"], row["conversation_id"])

        entity = await collection.get(key)
        assert entity is not None
        parent = uuid4()
        entity.parent_type = "conversation"
        entity.parent_id = parent
        await collection.save_entity(entity)
        collection.evict_from_cache_sync(key)

        read = await collection.get(key)
        assert read is not None
        assert (read.parent_type, read.parent_id) == ("conversation", parent)
