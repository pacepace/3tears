"""integration: each conversations index a keyed read names as its purpose serves that read.

The no-whole-reads rule (:mod:`threetears.enforcement.no_whole_reads`) keys a read by an index only when the index
names that read as its ``purpose``; this is the ``plan_test`` it names. It seeds the table past the size where the
planner would rather scan (three agents, three hundred users), takes the statement the code under test really writes
(run over a connection that records it), and asks the plan whether it found its rows through the index.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest
import pytest_asyncio

from threetears.conversations.merge import repoint_user
from threetears.conversations.migrations import register as register_conversations
from threetears.core.data.migrations import MigrationRunner
from threetears.core.testing.plans import explain_write, plan_uses_index

from .conftest import AsyncpgStore

pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="module")]

#: conversations seeded: past the size where the planner would rather scan than use an index
_ROWS = 6000


class _Captured:
    """a connection that runs nothing and records each statement asked of it."""

    def __init__(self) -> None:
        self.asked: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, sql: str, *args: Any) -> list[Any]:
        self.asked.append((sql, args))
        return []


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def world(pg_url: str) -> AsyncIterator[tuple[asyncpg.Connection, uuid.UUID, uuid.UUID]]:
    """a fresh schema with the conversations table built and seeded: 6000 conversations over three agents and three
    hundred users; yields the connection, an agent and a user of it."""
    schema = f"conv_plans_{uuid.uuid4().hex[:12]}"
    conn = await asyncpg.connect(pg_url)
    await conn.execute(f'CREATE SCHEMA "{schema}"')
    await conn.execute(f'SET search_path TO "{schema}", public')
    runner = MigrationRunner()
    register_conversations(runner)
    await runner.apply_for_agent_schema(AsyncpgStore(conn))  # type: ignore[arg-type]
    agents = [uuid.uuid4() for _ in range(3)]
    users = [uuid.uuid4() for _ in range(300)]
    await conn.execute(
        "INSERT INTO conversations (agent_id, conversation_id, customer_id, user_id, channel_type, status,"
        " date_created, date_updated, message_count)"
        " SELECT ($1::uuid[])[1 + g % 3], gen_random_uuid(), gen_random_uuid(), ($2::uuid[])[1 + g % 300], 'web',"
        f" 'active', now() - g * interval '1 minute', now(), 0 FROM generate_series(1, {_ROWS}) g",
        agents,
        users,
    )
    await conn.execute("ANALYZE conversations")
    try:
        # user 1's conversations are all under agent 1 + 1 % 3
        yield conn, agents[1], users[1]
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


async def test_conv_user(world: tuple[asyncpg.Connection, uuid.UUID, uuid.UUID]) -> None:
    """Flips if a user merge's repoint of one user's conversations under the schema's agent stops reading
    idx_conv_user (it would read every conversation of the schema)."""
    conn, agent, user = world
    captured = _Captured()
    await repoint_user(captured, from_user_id=user, to_user_id=uuid.uuid4(), agent_id=agent)
    (sql, args), *rest = captured.asked
    assert not rest, captured.asked
    plan = await explain_write(conn, sql, *args)
    assert plan_uses_index(plan, "idx_conv_user"), plan.plan
