"""integration: each agent-wake index serves the keyed reads its purpose names.

The no-whole-reads rule (:mod:`threetears.enforcement.no_whole_reads`) keys a read by an index only when the index
names that read as its ``purpose``; these are the ``plan_test`` each names. Each seeds the tables past the size where
the planner would rather scan, takes the statement the code under test really writes (its collection run over a pool
that records statements, or the module's own SQL constant), and asks the plan whether it found its rows through the
index by an index condition (:func:`threetears.core.testing.plans.plan_uses_index`).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import asyncpg
import pytest

from threetears.agent.skills.migrations import register as register_skills
from threetears.agent.wake import rate_limit
from threetears.agent.wake.collections import (
    WakeFireCollection,
    WakeScheduleCollection,
    WebhookSubscriptionCollection,
)
from threetears.agent.wake.migrations import register as register_wake
from threetears.conversations.migrations import register as register_conversations
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner
from threetears.core.testing.plans import explain_statement, explain_write, on_yugabyte, plan_uses_index

from .conftest import AsyncpgStore

pytestmark = pytest.mark.integration

_NOW = datetime(2026, 11, 9, tzinfo=UTC)
#: rows seeded per table: past the size where the planner would rather scan than use an index
_ROWS = 5000

#: a value for a NOT NULL column with no default, by its type, from the series row ``g``
_BY_TYPE: dict[str, str] = {
    "uuid": "gen_random_uuid()",
    "text": "'v' || g",
    "character varying": "'v' || g",
    "timestamp with time zone": "now() - g * interval '1 second'",
    "timestamp without time zone": "now() - g * interval '1 second'",
    "integer": "g",
    "bigint": "g",
    "smallint": "1",
    "double precision": "g",
    "real": "g",
    "numeric": "g",
    "boolean": "false",
    "jsonb": "'{}'::jsonb",
    "json": "'{}'::json",
    "ARRAY": "'{}'",
}


async def _seed(conn: asyncpg.Connection, table: str, rows: int = _ROWS, **values: str) -> None:
    """seed ``rows`` rows into ``table``, then ANALYZE it, so the planner prices it at size.

    :param conn: the connection
    :ptype conn: asyncpg.Connection
    :param table: the table
    :ptype table: str
    :param rows: how many
    :ptype rows: int
    :param values: a SQL expression of the series row ``g`` per column; every other NOT NULL column with no
        default gets one by its type
    :ptype values: str
    :return: nothing
    :rtype: None
    """
    columns = await conn.fetch(
        "SELECT column_name, data_type, is_nullable, column_default FROM information_schema.columns"
        " WHERE table_schema = current_schema() AND table_name = $1",
        table,
    )
    names: list[str] = []
    exprs: list[str] = []
    for column in columns:
        name = column["column_name"]
        if name in values:
            names.append(name)
            exprs.append(values[name])
        elif column["is_nullable"] == "NO" and column["column_default"] is None:
            names.append(name)
            exprs.append(_BY_TYPE[column["data_type"]])
    await conn.execute(
        f'INSERT INTO "{table}" ({", ".join(names)}) SELECT {", ".join(exprs)} FROM generate_series(1, {rows}) g'
    )
    await conn.execute(f'ANALYZE "{table}"')


class _Captured:
    """an L3 pool that runs nothing and records each statement asked of it, so a test explains the statement the
    code under test really writes."""

    def __init__(self) -> None:
        self.asked: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, sql: str, *args: Any) -> list[Any]:
        self.asked.append((sql, args))
        return []

    async def fetchrow(self, sql: str, *args: Any) -> None:
        self.asked.append((sql, args))

    async def fetchval(self, sql: str, *args: Any) -> int:
        self.asked.append((sql, args))
        return 0

    async def execute(self, sql: str, *args: Any) -> str:
        self.asked.append((sql, args))
        return "UPDATE 0"


@pytest.fixture
async def conn(pg_schema: tuple[str, str]) -> AsyncIterator[asyncpg.Connection]:
    """A connection on a fresh schema with the wake tables built and seeded: 5000 schedules over 500 agents and 250
    users (one in ten active and due), 2000 webhook subscriptions, and 20000 fires, a handful still in flight."""
    url, schema = pg_schema
    connection = await asyncpg.connect(url)
    await connection.execute(f'SET search_path TO "{schema}", public')
    runner = MigrationRunner()
    register_conversations(runner)
    register_skills(runner)
    register_wake(runner)
    await runner.apply_for_agent_schema(AsyncpgStore(connection))  # type: ignore[arg-type]
    agents = [uuid.uuid4() for _ in range(500)]
    users = [uuid.uuid4() for _ in range(250)]
    await connection.execute(
        "INSERT INTO agent_wake_schedules (conversation_id, schedule_id, user_id, agent_id, schedule_type, status,"
        " next_fire_at) SELECT gen_random_uuid(), gen_random_uuid(), ($2::uuid[])[1 + g % 250],"
        " ($1::uuid[])[1 + g % 500], 'cron', CASE WHEN g % 10 = 0 THEN 'active' ELSE 'expired' END,"
        " CASE WHEN g % 10 = 0 THEN now() + g * interval '1 minute' END FROM generate_series(1, 5000) g",
        agents,
        users,
    )
    await connection.execute(
        "INSERT INTO webhook_subscriptions (conversation_id, subscription_id, user_id, agent_id, name,"
        " verification_scheme, secret_ciphertext) SELECT gen_random_uuid(), gen_random_uuid(), ($2::uuid[])[1 + g % 250],"
        " ($1::uuid[])[1 + g % 500], 's' || g, 'generic_hmac_sha256', 'x'::bytea FROM generate_series(1, 2000) g",
        agents,
        users,
    )
    await connection.execute(
        "INSERT INTO wake_fires (conversation_id, fire_id, schedule_id, actual_fired_at, status, fire_source)"
        " SELECT s.conversation_id, gen_random_uuid(), s.schedule_id, now() - n * interval '1 hour',"
        " CASE WHEN n = 1 AND s.status = 'active' THEN 'dispatching' ELSE 'fired' END, 'scheduled_tick'"
        " FROM agent_wake_schedules s, generate_series(1, 4) n"
    )
    for table in ("agent_wake_schedules", "webhook_subscriptions", "wake_fires"):
        await connection.execute(f"ANALYZE {table}")
    try:
        yield connection
    finally:
        await connection.close()


def _collections(pool: _Captured | asyncpg.Connection) -> SimpleNamespace:
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    cfg = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return SimpleNamespace(
        schedules=WakeScheduleCollection(registry=registry, config=cfg),
        fires=WakeFireCollection(registry=registry, config=cfg),
        subscriptions=WebhookSubscriptionCollection(registry=registry, config=cfg),
    )


async def _user_counts(captured: _Captured) -> list[tuple[str, tuple[Any, ...]]]:
    """the statements check_rate_limit writes, the per-user count among them."""
    trigger = SimpleNamespace(
        conversation_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        schedule_id=uuid.uuid4(),
        webhook_subscription_id=None,
        fire_source="scheduled_tick",
        agent_id=uuid.uuid4(),
    )
    config = SimpleNamespace(max_fires_per_conv_per_day=100, max_fires_per_user_per_day=100)
    await rate_limit.check_rate_limit(trigger, captured, config)  # type: ignore[arg-type]
    return [asked for asked in captured.asked if "ws.user_id" in asked[0]]


async def test_wake_schedules_next_fire(conn: asyncpg.Connection) -> None:
    """Flips if the tick's due scan stops reading idx_wake_schedules_next_fire_asc, or if on YugabyteDB a tick past
    every active schedule stops reading only its page (a hashed ``next_fire_at`` reads and sorts every active one)."""
    captured = _Captured()
    await _collections(captured).schedules.list_due_for_tick(_NOW, limit=50)
    (sql, args), *_ = captured.asked
    plan = await explain_statement(conn, sql, *args)
    assert plan_uses_index(plan, "idx_wake_schedules_next_fire_asc"), plan.plan
    # a tick after every one of the 500 active schedules is due: the page of 50 reads about its own rows
    late_tick = datetime.now(UTC) + timedelta(days=30)
    due = await _collections(conn).schedules.list_due_for_tick(late_tick, limit=50)
    assert len(due) == 50, len(due)
    late = _Captured()
    await _collections(late).schedules.list_due_for_tick(late_tick, limit=50)
    (sql, args), *_ = late.asked
    page = await explain_statement(conn, sql, *args)
    # the statement returns the page plus keyset_page's one lookahead row (whether another page follows)
    assert page.rows_returned == 50 + 1, page
    if await on_yugabyte(conn):
        assert page.rows_scanned <= 2 * (50 + 1), page


async def test_wake_schedules_user(conn: asyncpg.Connection) -> None:
    """Flips if the per-user fire count stops reading idx_wake_schedules_user."""
    (sql, args), *_ = await _user_counts(_Captured())
    plan = await explain_statement(conn, sql, *args)
    assert plan_uses_index(plan, "idx_wake_schedules_user"), plan.plan


async def test_webhook_subs_user(conn: asyncpg.Connection) -> None:
    """Flips if the per-user fire count stops reading idx_webhook_subs_user."""
    (sql, args), *_ = await _user_counts(_Captured())
    plan = await explain_statement(conn, sql, *args)
    assert plan_uses_index(plan, "idx_webhook_subs_user"), plan.plan


async def test_wake_schedules_agent_status(conn: asyncpg.Connection) -> None:
    """Flips if any read of an agent's schedules stops reading idx_wake_schedules_agent_status."""
    agent = await conn.fetchval("SELECT agent_id FROM agent_wake_schedules LIMIT 1")
    captured = _Captured()
    collections = _collections(captured)
    await collections.schedules.list_for_agent(agent)
    statements = [
        *captured.asked,
        (rate_limit.COUNT_ACTIVE_SQL, (agent,)),
        (rate_limit.COUNT_ACTIVE_EXCLUDING_SQL, (agent, uuid.uuid4())),
        (rate_limit.COUNT_AGENT_FIRES_SQL, (agent, _NOW - timedelta(days=1), True)),
    ]
    for sql, args in statements:
        plan = await explain_statement(conn, sql, *args)
        assert plan_uses_index(plan, "idx_wake_schedules_agent_status"), (sql, plan.plan)


async def test_webhook_subs_agent(conn: asyncpg.Connection) -> None:
    """Flips if any read of an agent's webhook subscriptions stops reading idx_webhook_subs_agent."""
    agent = await conn.fetchval("SELECT agent_id FROM webhook_subscriptions LIMIT 1")
    captured = _Captured()
    collections = _collections(captured)
    await collections.subscriptions.list_for_agent(agent)
    statements = [*captured.asked, (rate_limit.COUNT_AGENT_FIRES_SQL, (agent, _NOW - timedelta(days=1), True))]
    for sql, args in statements:
        plan = await explain_statement(conn, sql, *args)
        assert plan_uses_index(plan, "idx_webhook_subs_agent"), (sql, plan.plan)


async def test_wake_fires_dispatching(conn: asyncpg.Connection) -> None:
    """Flips if the reaper's sweep of the fires in flight stops reading idx_wake_fires_dispatching."""
    captured = _Captured()
    await _collections(captured).fires.reap_stale_dispatching(_NOW, older_than=timedelta(minutes=10))
    (sql, args), *_ = captured.asked
    plan = await explain_write(conn, sql, *args)
    assert plan_uses_index(plan, "idx_wake_fires_dispatching"), plan.plan


async def _busy_conversation(conn: asyncpg.Connection) -> SimpleNamespace:
    """one conversation's long history, past where its primary key prefix is the cheaper read: 400 schedules (ten
    active), one subscription, and 6000 hourly fires of two schedules and the subscription (2000 each)."""
    conversation, agent, user = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    await conn.execute(
        "INSERT INTO agent_wake_schedules (conversation_id, schedule_id, user_id, agent_id, schedule_type, status,"
        " next_fire_at) SELECT $1, gen_random_uuid(), $2, $3, 'cron', CASE WHEN g <= 10 THEN 'active' ELSE 'expired'"
        " END, CASE WHEN g <= 10 THEN now() + g * interval '1 minute' END FROM generate_series(1, 400) g",
        conversation,
        user,
        agent,
    )
    first, second = [
        row["schedule_id"]
        for row in await conn.fetch(
            "SELECT schedule_id FROM agent_wake_schedules WHERE conversation_id = $1 ORDER BY schedule_id LIMIT 2",
            conversation,
        )
    ]
    subscription = uuid.uuid4()
    await conn.execute(
        "INSERT INTO webhook_subscriptions (conversation_id, subscription_id, user_id, agent_id, name,"
        " verification_scheme, secret_ciphertext) VALUES ($1, $2, $3, $4, 'busy', 'generic_hmac_sha256', 'x'::bytea)",
        conversation,
        subscription,
        user,
        agent,
    )
    await conn.execute(
        "INSERT INTO wake_fires (conversation_id, fire_id, schedule_id, webhook_subscription_id, actual_fired_at,"
        " status, fire_source) SELECT $1, gen_random_uuid(), CASE g % 3 WHEN 0 THEN $2::uuid WHEN 1 THEN $3::uuid END,"
        " CASE WHEN g % 3 = 2 THEN $4::uuid END, now() - g * interval '1 hour', 'fired',"
        " CASE WHEN g % 3 = 2 THEN 'webhook' ELSE 'scheduled_tick' END FROM generate_series(1, 6000) g",
        conversation,
        first,
        second,
        subscription,
    )
    for table in ("agent_wake_schedules", "webhook_subscriptions", "wake_fires"):
        await conn.execute(f"ANALYZE {table}")
    return SimpleNamespace(conversation=conversation, schedule=first, subscription=subscription)


def _of(captured: _Captured, table: str) -> list[tuple[str, tuple[Any, ...]]]:
    """the statements asked of ``table`` (a helper's one question of the server's version left out)."""
    return [asked for asked in captured.asked if table in asked[0]]


async def _wake_count(busy: SimpleNamespace, *, webhook: bool) -> tuple[str, tuple[Any, ...]]:
    """the per-wake count check_fire_limits writes for one of the busy conversation's wakes: its schedule's, or with
    ``webhook`` its subscription's."""
    captured = _Captured()
    trigger = SimpleNamespace(
        conversation_id=busy.conversation,
        schedule_id=None if webhook else busy.schedule,
        webhook_subscription_id=busy.subscription if webhook else None,
        fire_source="webhook" if webhook else "scheduled_tick",
        agent_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        protected=False,
    )
    limits = SimpleNamespace(manual_fires_count=True, per_wake=1, per_agent=1)
    await rate_limit.check_fire_limits(trigger, captured, limits)  # type: ignore[arg-type]
    (asked, *rest) = captured.asked
    # a per-wake count of zero is under the cap: the agent's count follows it, which the agent's index serves
    assert all("ws.agent_id" in sql for sql, _args in rest), captured.asked
    return asked


async def test_wake_schedules_conv_status(conn: asyncpg.Connection) -> None:
    """Flips if a conversation's active schedules, listed or counted for its cap, stop reading
    idx_wake_schedules_conv_status (they would read every schedule the conversation ever had)."""
    busy = await _busy_conversation(conn)
    captured = _Captured()
    schedules = _collections(captured).schedules
    await schedules.list_active_for_conversation(busy.conversation)
    await schedules.count_active_for_conversation(busy.conversation)
    statements = _of(captured, "agent_wake_schedules")
    assert len(statements) >= 2, captured.asked
    for sql, args in statements:
        plan = await explain_statement(conn, sql, *args)
        assert plan_uses_index(plan, "idx_wake_schedules_conv_status"), (sql, plan.plan)


async def test_wake_fires_schedule_time(conn: asyncpg.Connection) -> None:
    """Flips if one schedule's history page or its rate-limit count of the last day stops reading
    idx_wake_fires_schedule_time."""
    busy = await _busy_conversation(conn)
    captured = _Captured()
    await _collections(captured).fires.list_for_schedule(busy.conversation, busy.schedule)
    statements = [
        *_of(captured, "wake_fires"),
        await _wake_count(busy, webhook=False),
    ]
    for sql, args in statements:
        plan = await explain_statement(conn, sql, *args)
        assert plan_uses_index(plan, "idx_wake_fires_schedule_time"), (sql, plan.plan)


async def test_wake_fires_webhook_time(conn: asyncpg.Connection) -> None:
    """Flips if one subscription's history page or its rate-limit count of the last day stops reading
    idx_wake_fires_webhook_time."""
    busy = await _busy_conversation(conn)
    captured = _Captured()
    await _collections(captured).fires.list_for_subscription(busy.conversation, busy.subscription)
    statements = [
        *_of(captured, "wake_fires"),
        await _wake_count(busy, webhook=True),
    ]
    for sql, args in statements:
        plan = await explain_statement(conn, sql, *args)
        assert plan_uses_index(plan, "idx_wake_fires_webhook_time"), (sql, plan.plan)


async def test_wake_fires_conv_time(conn: asyncpg.Connection) -> None:
    """Flips if a conversation's newest fires stop reading idx_wake_fires_conv_time (the page would read and sort
    every fire the conversation ever had)."""
    busy = await _busy_conversation(conn)
    captured = _Captured()
    await _collections(captured).fires.list_for_conversation(busy.conversation)
    (sql, args), *rest = _of(captured, "wake_fires")
    assert not rest, captured.asked
    plan = await explain_statement(conn, sql, *args)
    assert plan_uses_index(plan, "idx_wake_fires_conv_time"), plan.plan


async def test_webhook_refusals_sub_time(conn: asyncpg.Connection) -> None:
    """Flips if a subscription's newest refusals stop reading idx_webhook_refusals_sub_time (the page would read and
    sort every refusal window the subscription ever had)."""
    busy = await _busy_conversation(conn)
    await conn.execute(
        "INSERT INTO webhook_refusals (conversation_id, subscription_id, reason, window_start, status_code,"
        " last_refused_at) SELECT $1, $2, (ARRAY['paused', 'missing_signature', 'bad_signature', 'rate_limited'])"
        "[1 + g % 4], now() - g * interval '1 minute', 401, now() - g * interval '1 minute'"
        " FROM generate_series(1, 4000) g",
        busy.conversation,
        busy.subscription,
    )
    await conn.execute("ANALYZE webhook_refusals")
    captured = _Captured()
    await _collections(captured).subscriptions.list_refusals(busy.subscription)
    (sql, args), *rest = _of(captured, "webhook_refusals")
    assert not rest, captured.asked
    plan = await explain_statement(conn, sql, *args)
    assert plan_uses_index(plan, "idx_webhook_refusals_sub_time"), plan.plan
