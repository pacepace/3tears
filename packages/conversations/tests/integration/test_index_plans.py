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
from threetears.core.keyed import Sort, keyset_page
from threetears.core.testing.captured import CapturedL3
from threetears.core.testing.plans import (
    StatementPlan,
    explain_statement,
    explain_write,
    on_yugabyte,
    plan_uses_index,
)

from .conftest import AsyncpgStore

pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="module")]

#: conversations seeded: past the size where the planner would rather scan than use an index
_ROWS = 6000


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def world(pg_url: str) -> AsyncIterator[tuple[asyncpg.Connection, uuid.UUID, uuid.UUID]]:
    """a fresh schema with the conversations table built and seeded: 6000 conversations over three agents, three
    hundred users and three statuses; yields the connection, an agent and a user of it."""
    schema = f"conv_plans_{uuid.uuid4().hex[:12]}"
    conn = await asyncpg.connect(pg_url)
    await conn.execute(f'CREATE SCHEMA "{schema}"')
    await conn.execute(f'SET search_path TO "{schema}", public')
    runner = MigrationRunner()
    register_conversations(runner)
    await runner.apply_for_agent_schema(AsyncpgStore(conn))
    agents = [uuid.uuid4() for _ in range(3)]
    users = [uuid.uuid4() for _ in range(300)]
    await conn.execute(
        "INSERT INTO conversations (agent_id, conversation_id, customer_id, user_id, channel_type, status,"
        " date_created, date_updated, message_count)"
        " SELECT ($1::uuid[])[1 + g % 3], gen_random_uuid(), gen_random_uuid(), ($2::uuid[])[1 + g % 300], 'web',"
        " (ARRAY['active', 'closed', 'archived', 'active', 'active'])[1 + g % 5], now() - g * interval '1 minute',"
        f" now() - (g * 7 % {_ROWS}) * interval '1 minute', 0 FROM generate_series(1, {_ROWS}) g",
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
    """Flips if the hub's person erasure, one person's conversations by owner alone, stops reading idx_conv_user (it
    would read every conversation of the schema)."""
    conn, _agent, user = world
    plan = await explain_statement(conn, _OWNED_CONVERSATIONS_SQL, user)
    assert plan_uses_index(plan, "idx_conv_user"), plan.plan


#: the hub's person erasure statement (``aibots.hub.customers.person_erasure.OWNED_CONVERSATIONS_SQL``)
_OWNED_CONVERSATIONS_SQL = "SELECT conversation_id FROM conversations WHERE user_id = $1"


#: the page size: a user holds twenty conversations, so a page of eight leaves a second and a third
_PAGE = 8

#: each sort column's SQL type, which a page past a cursor casts the cursor's text back to
_CASTS = {"date_created": "timestamptz", "date_updated": "timestamptz", "status": "text"}


async def _resumed_page(conn: asyncpg.Connection, by: dict[str, Any], sort: str) -> tuple[str, tuple[Any, ...]]:
    """the statement the hub's admin listing (``_fetch_paginated_rows``, on keyset_page) sends for the second page of
    one pin, newest / last first: the first page read for real for its cursor, the second captured.

    :param conn: the seeded connection
    :ptype conn: asyncpg.Connection
    :param by: the pin (the agent, and a user or a status)
    :ptype by: dict[str, Any]
    :param sort: the sort column
    :ptype sort: str
    :return: the second page's statement and its parameters
    :rtype: tuple[str, tuple[Any, ...]]
    """
    order = (Sort(sort, cast=_CASTS[sort], descending=True), Sort("conversation_id", cast="uuid", descending=True))
    columns = ("conversation_id", sort)
    first = await keyset_page(conn, "conversations", columns=columns, by=by, order=order, limit=_PAGE)
    assert first.next_cursor is not None, first
    captured = CapturedL3(real=conn, passes=lambda sql: 'FROM "conversations"' not in sql)
    await keyset_page(
        captured,
        "conversations",
        columns=columns,
        by=by,
        order=order,
        limit=_PAGE,
        cursor=first.next_cursor,
    )
    # keyset_page asks the server's version once (a YugabyteDB probe) before its page: the page is the read
    (sql, args), *rest = [asked for asked in captured.asked if 'FROM "conversations"' in asked[0]]
    assert not rest, captured.asked
    return sql, args


async def _resumed_plan(conn: asyncpg.Connection, by: dict[str, Any], sort: str) -> StatementPlan:
    """the analysed plan of the resumed page, checked to read about its own rows on YugabyteDB (``limit + 1``, the
    lookahead, and at most as many again).

    :param conn: the seeded connection
    :ptype conn: asyncpg.Connection
    :param by: the pin
    :ptype by: dict[str, Any]
    :param sort: the sort column
    :ptype sort: str
    :return: the plan
    :rtype: StatementPlan
    """
    sql, args = await _resumed_page(conn, by, sort)
    plan = await explain_statement(conn, sql, *args)
    assert plan.rows_returned == _PAGE + 1, plan
    if await on_yugabyte(conn):
        assert plan.rows_scanned <= 2 * (_PAGE + 1), plan
    return plan


async def test_conv_agent_created(world: tuple[asyncpg.Connection, uuid.UUID, uuid.UUID]) -> None:
    """Flips if one agent's conversations by creation date stop reading idx_conv_agent_created."""
    conn, agent, _user = world
    plan = await _resumed_plan(conn, {"agent_id": agent}, "date_created")
    assert plan_uses_index(plan, "idx_conv_agent_created"), plan.plan


async def test_conv_agent_status(world: tuple[asyncpg.Connection, uuid.UUID, uuid.UUID]) -> None:
    """Flips if one agent's conversations by status, or one status's sorted by status, stop reading
    idx_conv_agent_status."""
    conn, agent, _user = world
    plan = await _resumed_plan(conn, {"agent_id": agent}, "status")
    assert plan_uses_index(plan, "idx_conv_agent_status"), plan.plan
    plan = await _resumed_plan(conn, {"agent_id": agent, "status": "active"}, "status")
    assert plan_uses_index(plan, "idx_conv_agent_status"), plan.plan


async def test_conv_agent_user_created(world: tuple[asyncpg.Connection, uuid.UUID, uuid.UUID]) -> None:
    """Flips if one user's conversations by creation date stop reading idx_conv_agent_user_created."""
    conn, agent, user = world
    plan = await _resumed_plan(conn, {"agent_id": agent, "user_id": user}, "date_created")
    assert plan_uses_index(plan, "idx_conv_agent_user_created"), plan.plan
    # a user merge's repoint of the user's conversations under the schema's agent reads an agent-led user index
    captured = CapturedL3()
    await repoint_user(captured, from_user_id=user, to_user_id=uuid.uuid4(), agent_id=agent)
    (sql, args), *rest = captured.asked
    assert not rest, captured.asked
    repoint = await explain_write(conn, sql, *args)
    assert plan_uses_index(repoint, "idx_conv_agent_user_created") or plan_uses_index(
        repoint, "idx_conv_agent_user_status"
    ), repoint.plan


async def test_conv_agent_user_status(world: tuple[asyncpg.Connection, uuid.UUID, uuid.UUID]) -> None:
    """Flips if one user's conversations by status stop reading idx_conv_agent_user_status."""
    conn, agent, user = world
    plan = await _resumed_plan(conn, {"agent_id": agent, "user_id": user}, "status")
    assert plan_uses_index(plan, "idx_conv_agent_user_status"), plan.plan


async def test_conv_agent_status_created(world: tuple[asyncpg.Connection, uuid.UUID, uuid.UUID]) -> None:
    """Flips if one status's conversations by creation date stop reading idx_conv_agent_status_created."""
    conn, agent, _user = world
    plan = await _resumed_plan(conn, {"agent_id": agent, "status": "active"}, "date_created")
    assert plan_uses_index(plan, "idx_conv_agent_status_created"), plan.plan


async def test_conv_agent_updated(world: tuple[asyncpg.Connection, uuid.UUID, uuid.UUID]) -> None:
    """Flips if one agent's conversations by last update stop reading idx_conv_agent_updated."""
    conn, agent, _user = world
    plan = await _resumed_plan(conn, {"agent_id": agent}, "date_updated")
    assert plan_uses_index(plan, "idx_conv_agent_updated"), plan.plan
