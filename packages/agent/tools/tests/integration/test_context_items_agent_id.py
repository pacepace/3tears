"""agent-tools v005 and agent-memory v032 on real Postgres: the agent a row was written for.

Proves, against the migrations a consumer runs:

- a row written before the column takes its conversation's own agent (and a
  memory ref its memory's agent), and a replay changes nothing;
- the keyed upsert holds one row per agent per key, and one row per key for a
  caller that names no agent (``NULLS NOT DISTINCT``);
- a reader that names an agent reads only that agent's rows.

requires docker; gated by ``pytest.mark.integration``.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import asyncpg
import pytest

from threetears.agent.memory.collections import MemoryRefsCollection
from threetears.agent.memory.migrations import register as register_memory
from threetears.agent.memory.migrations.v032_memory_refs_agent_id import memory_refs_agent_id
from threetears.agent.tools.collections import ContextItemCollection
from threetears.agent.tools.migrations import register as register_tools
from threetears.agent.tools.migrations.v005_context_items_agent_id import context_items_agent_id
from threetears.conversations.migrations import register as register_conversations
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner

from packages.agent.tools.tests.testing_utils import make_context_metadata, make_nats_mock

pytestmark = pytest.mark.integration


# parity-with: threetears.core.data.store.DataStore
class _AsyncpgStore:
    """DataStore-shape wrapper over one asyncpg connection bound to the schema."""

    def __init__(self, conn: asyncpg.Connection) -> None:
        """capture the connection.

        :param conn: connection with search_path set
        :ptype conn: asyncpg.Connection
        """
        self._conn = conn

    async def execute(self, sql: str, *params: Any) -> str:
        """proxy execute.

        :param sql: SQL statement
        :ptype sql: str
        :param params: positional parameters
        :ptype params: Any
        :return: asyncpg status string
        :rtype: str
        """
        result: str = await self._conn.execute(sql, *params)
        return result

    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        """proxy fetch -> dict rows.

        :param sql: SQL query
        :ptype sql: str
        :param params: positional parameters
        :ptype params: Any
        :return: list of row dicts
        :rtype: list[dict[str, Any]]
        """
        return [dict(r) for r in await self._conn.fetch(sql, *params)]


@pytest.fixture
async def schema_conn(db_container: str) -> AsyncIterator[tuple[asyncpg.Connection, str, str]]:
    """a fresh schema with conversations, agent-tools and agent-memory migrated; yields (conn, url, schema).

    :param db_container: shared Postgres testcontainer DSN
    :ptype db_container: str
    :return: the connection, the URL and the schema
    :rtype: AsyncIterator[tuple[asyncpg.Connection, str, str]]
    """
    schema = f"ctx_agent_{uuid.uuid4().hex[:10]}"
    conn = await asyncpg.connect(db_container)
    await conn.execute(f'CREATE SCHEMA "{schema}"')
    await conn.execute(f'SET search_path TO "{schema}", public')
    runner = MigrationRunner()
    register_conversations(runner)
    register_tools(runner)
    register_memory(runner)
    await runner.apply_for_agent_schema(_AsyncpgStore(conn))  # type: ignore[arg-type]
    try:
        yield conn, db_container, schema
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


async def _conversation(conn: asyncpg.Connection, agent_id: uuid.UUID) -> uuid.UUID:
    conversation_id = uuid.uuid4()
    await conn.execute(
        "INSERT INTO conversations (agent_id, conversation_id, customer_id, user_id, channel_type, status, "
        "date_created, date_updated, message_count) VALUES ($1, $2, $3, $4, 'web', 'active', now(), now(), 0)",
        agent_id,
        conversation_id,
        uuid.uuid4(),
        uuid.uuid4(),
    )
    return conversation_id


async def _context_row(conn: asyncpg.Connection, conversation_id: uuid.UUID, key: str) -> uuid.UUID:
    context_id = uuid.uuid4()
    now = datetime.now(UTC)
    await conn.execute(
        "INSERT INTO context_items (conversation_id, context_id, context_type, key, short_desc, content, "
        "date_accessed, date_created, date_updated) VALUES ($1, $2, 'tool_result', $3, 's', 'c', $4, $4, $4)",
        conversation_id,
        context_id,
        key,
        now,
    )
    return context_id


async def test_v005_backfills_the_conversations_agent_and_replays_clean(
    schema_conn: tuple[asyncpg.Connection, str, str],
) -> None:
    conn, _url, _schema = schema_conn
    owner = uuid.uuid4()
    conversation_id = await _conversation(conn, owner)
    context_id = await _context_row(conn, conversation_id, "web_search:1")
    # a row written before the column: no agent, under the old index shape
    await conn.execute("UPDATE context_items SET agent_id = NULL")
    orphan = await _context_row(conn, uuid.uuid4(), "calc:1")

    await context_items_agent_id(_AsyncpgStore(conn))  # type: ignore[arg-type]
    await context_items_agent_id(_AsyncpgStore(conn))  # type: ignore[arg-type]

    agents = {
        r["context_id"]: r["agent_id"] for r in await conn.fetch("SELECT context_id, agent_id FROM context_items")
    }
    assert agents == {context_id: owner, orphan: None}
    indexes = {
        r["indexname"]
        for r in await conn.fetch(
            "SELECT indexname FROM pg_indexes WHERE tablename = 'context_items' AND schemaname = current_schema()"
        )
    }
    assert {"ix_context_items_agent_var_key", "ix_context_items_agent_tool_result_key"} <= indexes
    assert not {"ix_context_items_var_key", "ix_context_items_tool_result_key"} & indexes


async def test_v032_backfills_a_refs_agent_from_its_memory_then_its_conversation(
    schema_conn: tuple[asyncpg.Connection, str, str],
) -> None:
    conn, _url, _schema = schema_conn
    owner, member = uuid.uuid4(), uuid.uuid4()
    conversation_id = await _conversation(conn, owner)
    memory_id = uuid.uuid4()
    await conn.execute(
        "INSERT INTO memories (agent_id, memory_id, customer_id, user_id, conversation_id, type_memory, content, "
        "date_created, date_updated) VALUES ($1, $2, $3, $4, $5, 'fact', 'm', now(), now())",
        member,
        memory_id,
        uuid.uuid4(),
        uuid.uuid4(),
        conversation_id,
    )
    other_item = uuid.uuid4()
    now = datetime.now(UTC)
    for item_id, item_type in ((memory_id, "memory"), (other_item, "media")):
        await conn.execute(
            "INSERT INTO conversation_memory_refs (conversation_id, item_id, item_type, short_desc, date_created, "
            "date_updated) VALUES ($1, $2, $3, 'd', $4, $4)",
            conversation_id,
            item_id,
            item_type,
            now,
        )

    await memory_refs_agent_id(_AsyncpgStore(conn))  # type: ignore[arg-type]
    await memory_refs_agent_id(_AsyncpgStore(conn))  # type: ignore[arg-type]

    agents = {
        r["item_id"]: r["agent_id"] for r in await conn.fetch("SELECT item_id, agent_id FROM conversation_memory_refs")
    }
    assert agents == {memory_id: member, other_item: owner}


def _collections(pool: asyncpg.Pool) -> tuple[ContextItemCollection, MemoryRefsCollection]:
    l1 = SQLiteBackend(db_name=f"ctx_agent_{uuid.uuid4().hex[:8]}")
    metadata = make_context_metadata()
    MemoryRefsCollection.schema.to_sqlalchemy_table(metadata)
    l1.initialize(metadata)
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, l3_pool=pool, kv_key_scope="test-principal")
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return (
        ContextItemCollection(registry, config, nats_client=make_nats_mock()),
        MemoryRefsCollection(registry=registry, config=config, nats_client=make_nats_mock()),
    )


async def test_the_upsert_keeps_one_row_per_agent_and_one_for_no_agent(
    schema_conn: tuple[asyncpg.Connection, str, str],
) -> None:
    conn, url, schema = schema_conn
    conversation_id = await _conversation(conn, uuid.uuid4())
    pool = await asyncpg.create_pool(
        dsn=url, min_size=1, max_size=2, server_settings={"search_path": f"{schema}, public"}
    )
    try:
        items, refs = _collections(pool)
        agent_a, agent_b = uuid.uuid4(), uuid.uuid4()

        async def upsert(agent_id: uuid.UUID | None, content: str) -> uuid.UUID:
            now = datetime.now(UTC)
            return await items.upsert_tool_result(
                conversation_id,
                {
                    "context_id": uuid.uuid4(),
                    "conversation_id": conversation_id,
                    "agent_id": agent_id,
                    "context_type": "tool_result",
                    "key": "web_search:same",
                    "short_desc": content,
                    "content": content,
                    "metadata": {},
                    "date_accessed": now,
                    "date_created": now,
                    "date_updated": now,
                },
            )

        a1, b1 = await upsert(agent_a, "a"), await upsert(agent_b, "b")
        n1 = await upsert(None, "n")
        assert len({a1, b1, n1}) == 3
        assert await upsert(agent_a, "a again") == a1
        assert await upsert(None, "n again") == n1
        assert await conn.fetchval("SELECT count(*) FROM context_items") == 3

        only_b = await items.find_by_conversation(conversation_id, agent_b)
        assert [e.to_dict()["content"] for e in only_b] == ["b"]
        assert len(await items.find_by_conversation(conversation_id)) == 3

        now = datetime.now(UTC)
        for agent_id in (agent_a, agent_b):
            await refs.save_entity(
                refs.create(
                    {
                        "conversation_id": conversation_id,
                        "agent_id": agent_id,
                        "item_id": uuid.uuid4(),
                        "item_type": "memory",
                        "short_desc": str(agent_id),  # convert at border: test row text
                        "date_created": now,
                        "date_updated": now,
                    }
                )
            )
        assert [r.agent_id for r in await refs.find_by_conversation(conversation_id, agent_a)] == [agent_a]
        assert len(await refs.find_by_conversation(conversation_id)) == 2
    finally:
        await pool.close()
