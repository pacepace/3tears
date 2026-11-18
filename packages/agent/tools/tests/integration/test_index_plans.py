"""integration: each context-items index serves the keyed read its purpose names.

The no-whole-reads rule (:mod:`threetears.enforcement.no_whole_reads`) keys a read by an index only when the index
names that read as its ``purpose``; this is the ``plan_test`` each names. A migrated agent schema is seeded past the
size where the planner would rather scan, and the statement the collection really writes (run over a pool that records
statements) is asked whether its plan found its rows through the index by an index condition
(:func:`threetears.core.testing.plans.plan_uses_index`). Runs on PostgreSQL and YugabyteDB (the session's
``db_container``), a schema of its own.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from threetears.agent.tools.collections import ContextItemCollection
from threetears.agent.tools.migrations import register as register_tools
from threetears.conversations.migrations import register as register_conversations
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner
from threetears.core.testing.captured import CapturedL3
from threetears.core.testing.plans import explain_statement, plan_uses_index

from packages.agent.tools.tests.testing_utils import make_nats_mock

pytestmark = pytest.mark.integration

#: conversations seeded, and each one's context items: one conversation is a small share of the table
_CONVERSATIONS = 200
_PER_CONVERSATION = 50


class _Store:
    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn

    async def execute(self, sql: str, *params: Any) -> str:
        return str(await self._conn.execute(sql, *params))

    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        return [dict(r) for r in await self._conn.fetch(sql, *params)]


@pytest.fixture
async def seeded(db_container: str) -> AsyncIterator[tuple[asyncpg.Connection, uuid.UUID]]:
    """a migrated agent schema whose context_items hold 200 conversations' rows (50 each, the conversation foreign key
    dropped: the conversations are the seeds', not rows), analysed; the connection and one of the conversations."""
    schema = f"ctx_plans_{uuid.uuid4().hex[:12]}"
    connection = await asyncpg.connect(db_container)
    await connection.execute(f'CREATE SCHEMA "{schema}"')
    await connection.execute(f'SET search_path TO "{schema}", public')
    try:
        runner = MigrationRunner()
        register_conversations(runner)
        register_tools(runner)
        await runner.apply_for_agent_schema(_Store(connection))
        for (name,) in await connection.fetch(
            "SELECT conname FROM pg_constraint WHERE conrelid = 'context_items'::regclass AND contype = 'f'"
        ):
            await connection.execute(f'ALTER TABLE context_items DROP CONSTRAINT "{name}"')
        conversations = [uuid.uuid4() for _ in range(_CONVERSATIONS)]
        await connection.execute(
            "INSERT INTO context_items (conversation_id, context_id, context_type, key, short_desc, long_desc, content,"
            " date_accessed, date_created, date_updated)"
            " SELECT c, gen_random_uuid(), 'variable', 'k' || g, 's', 'l', 'c', now(),"
            " now() - g * interval '1 second', now()"
            f" FROM unnest($1::uuid[]) AS c, generate_series(1, {_PER_CONVERSATION}) AS g",
            conversations,
        )
        await connection.execute("ANALYZE context_items")
        yield connection, conversations[0]
    finally:
        await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await connection.close()


async def test_conv_created(seeded: tuple[asyncpg.Connection, uuid.UUID]) -> None:
    """Flips if a conversation's context, oldest first, with or without its agent named, stops reading
    ix_context_items_conv_created: the conversation's rows in creation order, the partition no longer read whole and
    sorted."""
    conn, conversation = seeded
    # the real connection answers which engine it is, so the read is spelled (and hinted) for it; nothing else runs
    captured = CapturedL3(real=conn, passes=lambda _sql: False)
    registry = CollectionRegistry()
    registry.configure(l3_pool=captured)
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    collection = ContextItemCollection(registry, config, nats_client=make_nats_mock())
    collection.l3_pool = captured
    await collection.find_by_conversation(conversation)
    await collection.find_by_conversation(conversation, agent_id=uuid.uuid4())
    reads = [(sql, args) for sql, args in captured.asked if 'FROM "context_items"' in sql]
    assert len(reads) == 2, [sql for sql, _ in captured.asked]
    for sql, args in reads:
        plan = await explain_statement(conn, sql, *args)
        assert plan_uses_index(plan, "ix_context_items_conv_created"), plan.plan
