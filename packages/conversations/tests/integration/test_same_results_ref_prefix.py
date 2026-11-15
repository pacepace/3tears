"""integration: the conversations whose ref starts with a prefix are the ones the statement it replaced found.

``ConversationsCollection.find_by_ref_prefix`` matched ``left(conversation_ref, char_length($3)) = $3`` under the agent
and a channel, newest first; it now reads the prefix as a code-point range (``COLLATE "C"``) through
``idx_conv_agent_ref_c`` and sorts the rows newest first. Compared, through
:func:`~threetears.core.testing.same_results.assert_same_results`, with that statement: refs that start with the prefix
(the prefix itself among them), ones that only begin like it (``story-10:`` for ``story-1:``), a shorter ref, an
upper-case one, non-ASCII ones, a NULL ref, another channel's and another agent's; and an empty prefix (every ref,
no NULL one). Runs on PostgreSQL and YugabyteDB (the session's ``db_container``), a schema of its own.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import pytest
import pytest_asyncio

from threetears.conversations.collection import ConversationsCollection
from threetears.conversations.migrations import register as register_conversations
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner
from threetears.core.testing.same_results import Seeded, assert_same_results

from .conftest import AsyncpgStore

pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="module")]

NS = uuid.UUID("7a2e9c41-3b6d-4f0e-8c15-9d4b2a6e1f03")
AGENTS = (uuid.uuid5(NS, "agent-0"), uuid.uuid5(NS, "agent-1"))
T0 = datetime(2026, 11, 15, 12, 0, tzinfo=UTC)

#: each seeded row's ref, in turn: the prefix's family first, then refs that only resemble it
_REFS = (
    "story-1:object-a",
    "story-1:",
    "story-10:object-a",
    "story-1",
    "STORY-1:object-b",
    "story-1:é-ünïcode",
    "story-1:\U0001f600",
    "story-2:object-a",
    "story-1:object-b",
)

_OLD_SQL = (
    "SELECT * FROM conversations "
    "WHERE agent_id = $1 AND channel_type = $2 AND left(conversation_ref, char_length($3)) = $3 "
    "ORDER BY date_created DESC"
)


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def conn(pg_url: str) -> AsyncIterator[asyncpg.Connection]:
    """a fresh schema holding the conversations table."""
    schema = f"conv_same_{uuid.uuid4().hex[:12]}"
    connection = await asyncpg.connect(pg_url)
    await connection.execute(f'CREATE SCHEMA "{schema}"')
    await connection.execute(f'SET search_path TO "{schema}", public')
    runner = MigrationRunner()
    register_conversations(runner)
    await runner.apply_for_agent_schema(AsyncpgStore(connection))
    try:
        yield connection
    finally:
        await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await connection.close()


def _row(seed: Seeded) -> Mapping[str, Any]:
    """a conversation of its tenant's agent, its ref each of :data:`_REFS` in turn (NULL in the NULLs case), every
    fourth on another channel, created at its tie's minute."""
    stamp = T0 + timedelta(minutes=seed.tie)
    return {
        "agent_id": AGENTS[seed.tenant],
        "conversation_id": uuid.uuid5(NS, f"conversation-{seed.index}"),
        "customer_id": uuid.uuid5(NS, "customer"),
        "user_id": uuid.uuid5(NS, "user"),
        "channel_type": "web" if seed.index % 4 != 3 else "slack",
        "status": "active",
        "date_created": stamp,
        "date_updated": stamp,
        "message_count": 0,
        "conversation_ref": None if seed.null else _REFS[seed.index % len(_REFS)],
    }


@pytest.mark.parametrize("prefix", ["story-1:", "", "story-1:é"])
async def test_the_prefix_s_conversations_are_the_statement_s(
    conn: asyncpg.Connection, monkeypatch: pytest.MonkeyPatch, prefix: str
) -> None:
    """Flips if the conversations a prefix lists are not the ones ``left(...) = $3`` listed, or not newest first: a
    ref that only begins like it, a NULL ref, another channel's or agent's, one missed."""
    agent = AGENTS[0]
    built: list[dict[str, Any]] = []
    coerce = getattr(ConversationsCollection, "_coerce_row")  # noqa: B009 - the row each entity is built from

    def keep(self: ConversationsCollection, row: dict[str, Any]) -> dict[str, Any]:
        coerced = coerce(self, row)
        built.append(dict(row))
        return coerced

    monkeypatch.setattr(ConversationsCollection, "_coerce_row", keep)

    async def helper(c: asyncpg.Connection) -> list[dict[str, Any]]:
        built.clear()
        registry = CollectionRegistry()
        registry.configure(l3_pool=c)
        collection = ConversationsCollection(registry, DefaultCoreConfig(), c)
        await collection.find_by_ref_prefix(agent, "web", prefix)
        return list(built)

    await assert_same_results(
        conn,
        table="conversations",
        row=_row,
        old_sql=_OLD_SQL,
        params=[agent, "web", prefix],
        helper=helper,
        columns=["conversation_id", "conversation_ref", "channel_type", "agent_id", "date_created"],
        order_by=["date_created"],
    )
