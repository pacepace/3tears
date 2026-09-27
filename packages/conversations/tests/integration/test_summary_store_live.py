"""Integration: ``ConversationSummaryStore`` on a real conversations table.

What the unit fake cannot prove: that the summary and the cursor survive the real row's column types
(``summary`` TEXT, ``metadata`` JSONB) through ``ConversationsCollection``, and that a save from a
stale expectation is refused against the real row.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import UUID, uuid4

import asyncpg
import pytest

from threetears.conversations import ConversationsCollection, ConversationSummaryStore
from threetears.conversations.migrations import register as register_conversations
from threetears.core.backends.sql import SqlL3Backend
from threetears.core.collections import init_connection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner
from threetears.langgraph import SummaryState

from .conftest import AsyncpgStore

pytestmark = pytest.mark.integration


@pytest.fixture
async def collection(pg_schema: tuple[str, str]) -> AsyncIterator[ConversationsCollection]:
    url, schema = pg_schema
    conn = await asyncpg.connect(url)
    try:
        await conn.execute(f'SET search_path TO "{schema}", public')
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
        init=init_connection,  # the documented pool shape: registers the JSONB codec
    )
    assert pool is not None
    registry = CollectionRegistry()
    registry.configure(l3_pool=SqlL3Backend(pool))
    try:
        yield ConversationsCollection(registry, DefaultCoreConfig(), pool)
    finally:
        await pool.close()


async def _seed(collection: ConversationsCollection) -> tuple[UUID, UUID]:
    agent_id, conversation_id = uuid4(), uuid4()
    await collection.l3_pool.execute(  # type: ignore[union-attr]
        "INSERT INTO conversations (agent_id, conversation_id, customer_id, user_id, channel_type, status, "
        "metadata, date_created, date_updated, message_count) "
        "VALUES ($1, $2, $3, $4, 'web', 'active', '{\"source\": \"live\"}'::jsonb, now(), now(), 0)",
        agent_id,
        conversation_id,
        uuid4(),
        uuid4(),
    )
    return agent_id, conversation_id


async def test_the_summary_and_cursor_round_trip_through_the_real_row(collection: ConversationsCollection) -> None:
    agent_id, conversation_id = await _seed(collection)
    store = ConversationSummaryStore(collection, agent_id=agent_id, conversation_id=conversation_id)
    assert await store.load() is None

    first = SummaryState(text="the story so far", through_id="m9", through_count=10)
    assert await store.save(first, expected=None)
    assert await store.load() == first

    second = SummaryState(text="and then some", through_id="m19", through_count=20)
    assert await store.save(second, expected=None) is False, "a save from a stale expectation is refused"
    assert await store.save(second, expected=first)
    assert await store.load() == second

    row = await collection.get((agent_id, conversation_id))
    assert row is not None and (row.metadata or {}).get("source") == "live", "other metadata survives"
