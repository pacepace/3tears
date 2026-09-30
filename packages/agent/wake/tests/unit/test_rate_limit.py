"""Unit tests for :mod:`threetears.agent.wake.rate_limit`.

The helper is intentionally pure-async over an asyncpg-compatible pool
(``fetchval`` only). Tests substitute a minimal in-memory stub for the
pool so the boundary contract is exercised without touching Postgres.

Three scenarios drive the per-fire helper:

- both per-conv + per-user counts under cap -> ``True``
- per-conv at cap -> ``False`` (per-user query never runs)
- per-conv under cap, per-user at cap -> ``False``

Plus the per-wake / per-agent fire limits, the per-agent active-schedule
cap helper, and the serialized create.

The fakes are tagged with ``parity-with`` markers per the workspace
fake-parity enforcement rule.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest

from threetears.agent.wake.config import (
    DEFAULT_MAX_FIRES_PER_CONV_PER_DAY,
    DEFAULT_MAX_ACTIVE_SCHEDULES_PER_AGENT,
    DEFAULT_MAX_FIRES_PER_USER_PER_DAY,
)
from threetears.agent.wake.rate_limit import (
    ScheduleCapExceeded,
    check_active_schedule_cap,
    check_fire_limits,
    check_rate_limit,
    create_schedule_serialized,
)
from threetears.agent.wake.types import FireLimits, WakeTrigger


# parity-with: asyncpg.Pool (fetchval-only minimal stand-in for the
# rate-limit helper boundary)
class _StubPool:
    """Minimal asyncpg.Pool stand-in exposing ``fetchval``.

    Drives ``check_rate_limit`` by returning a queued integer for
    each ``fetchval`` call. The per-fire helper makes at most two
    ``fetchval`` calls (per-conv first, per-user second); the cap
    helper makes one.

    :param values: integers returned in FIFO order for each
        ``fetchval`` call
    :ptype values: list[int]
    """

    def __init__(self, values: list[int]) -> None:
        self._values = list(values)
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetchval(self, query: str, *args: Any) -> int:
        """Return the next queued value; record the call for assertion."""
        self.calls.append((query, args))
        if not self._values:
            return 0
        return self._values.pop(0)


# parity-with: threetears.agent.wake.config.WakeConfig (minimal in-memory impl)
class _StubConfig:
    """In-memory :class:`WakeConfig` impl returning the platform defaults.

    The Protocol's other properties are unused by the rate-limit
    helpers so the stubs return reasonable empties; the runtime-checkable
    isinstance is what the helpers care about.
    """

    @property
    def max_fires_per_conv_per_day(self) -> int:
        return DEFAULT_MAX_FIRES_PER_CONV_PER_DAY

    @property
    def max_fires_per_user_per_day(self) -> int:
        return DEFAULT_MAX_FIRES_PER_USER_PER_DAY

    @property
    def max_webhook_fires_per_subscription_per_hour(self) -> int:
        return 60

    @property
    def max_active_schedules_per_agent(self) -> int:
        return DEFAULT_MAX_ACTIVE_SCHEDULES_PER_AGENT

    @property
    def http_allowed_hosts(self) -> tuple[str, ...]:
        return ()

    @property
    def loki_client(self) -> Any | None:
        return None

    @property
    def loki_named_queries(self) -> dict[str, str]:
        return {}

    @property
    def postgres_named_queries(self) -> dict[str, str]:
        return {}


def _trigger(
    *,
    user_id: UUID | None = None,
    conversation_id: UUID | None = None,
) -> WakeTrigger:
    """Build a minimal :class:`WakeTrigger` for the rate-limit boundary."""
    return WakeTrigger(
        schedule_id=uuid4(),
        user_id=user_id or uuid4(),
        agent_id=uuid4(),
        conversation_id=conversation_id or uuid4(),
        fire_source="scheduled_tick",
        execution_mode="inline",
        schedule_type="daily_at",
        fired_at=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_check_rate_limit_returns_none_when_both_counts_under_cap() -> None:
    """Both under cap -> ``None`` (the fire may proceed)."""
    pool = _StubPool([5, 10])  # conv=5, user=10 against (24, 100)
    config = _StubConfig()
    trigger = _trigger()
    assert await check_rate_limit(trigger, pool, config) is None
    assert len(pool.calls) == 2  # both queries ran


@pytest.mark.asyncio
async def test_check_rate_limit_returns_conv_scope_when_per_conv_at_cap() -> None:
    """Per-conv count >= cap -> returns ``'conv'``, per-user query is skipped."""
    pool = _StubPool([DEFAULT_MAX_FIRES_PER_CONV_PER_DAY, 0])
    config = _StubConfig()
    assert await check_rate_limit(_trigger(), pool, config) == "conv"
    assert len(pool.calls) == 1  # per-user query did not run


@pytest.mark.asyncio
async def test_check_rate_limit_returns_user_scope_when_per_user_at_cap() -> None:
    """Per-conv under cap + per-user at cap -> returns ``'user'`` after both queries."""
    pool = _StubPool([0, DEFAULT_MAX_FIRES_PER_USER_PER_DAY])
    config = _StubConfig()
    assert await check_rate_limit(_trigger(), pool, config) == "user"
    assert len(pool.calls) == 2  # both ran


@pytest.mark.asyncio
async def test_check_rate_limit_returns_none_with_none_pool() -> None:
    """``None`` pool -> ``None`` (allows unit tests that omit the DB)."""
    assert await check_rate_limit(_trigger(), None, _StubConfig()) is None


@pytest.mark.asyncio
async def test_check_active_schedule_cap_returns_true_under_cap() -> None:
    """Count strictly under cap -> True (pool path), counted by agent."""
    pool = _StubPool([DEFAULT_MAX_ACTIVE_SCHEDULES_PER_AGENT - 1])
    agent_id = uuid4()
    assert (
        await check_active_schedule_cap(
            agent_id=agent_id,
            cap=DEFAULT_MAX_ACTIVE_SCHEDULES_PER_AGENT,
            pool=pool,
        )
        is True
    )
    sql, args = pool.calls[0]
    assert "agent_id = $1" in sql and "NOT protected" in sql
    assert args == (agent_id,)


@pytest.mark.asyncio
async def test_check_active_schedule_cap_returns_false_at_cap() -> None:
    """Count at the cap boundary -> False (>= rejects; pool path)."""
    pool = _StubPool([DEFAULT_MAX_ACTIVE_SCHEDULES_PER_AGENT])
    assert (
        await check_active_schedule_cap(
            agent_id=uuid4(),
            cap=DEFAULT_MAX_ACTIVE_SCHEDULES_PER_AGENT,
            pool=pool,
        )
        is False
    )


@pytest.mark.asyncio
async def test_check_active_schedule_cap_returns_true_with_none_pool_and_no_count_func() -> None:
    """Neither ``pool`` nor ``count_func`` supplied -> True (short-circuit)."""
    assert (
        await check_active_schedule_cap(
            agent_id=uuid4(),
            cap=DEFAULT_MAX_ACTIVE_SCHEDULES_PER_AGENT,
        )
        is True
    )


@pytest.mark.asyncio
async def test_check_active_schedule_cap_uses_count_func_when_supplied() -> None:
    """When ``count_func`` is supplied, it wins over ``pool``."""
    calls: list[None] = []

    async def count_active() -> int:
        calls.append(None)
        return DEFAULT_MAX_ACTIVE_SCHEDULES_PER_AGENT - 1

    pool = _StubPool([999])  # would say "over cap" if consulted
    assert (
        await check_active_schedule_cap(
            agent_id=uuid4(),
            cap=DEFAULT_MAX_ACTIVE_SCHEDULES_PER_AGENT,
            pool=pool,
            count_func=count_active,
        )
        is True
    )
    # count_func was used; pool was NOT consulted
    assert len(calls) == 1
    assert pool.calls == []


@pytest.mark.asyncio
async def test_check_active_schedule_cap_count_func_at_cap_rejects() -> None:
    """``count_func`` returning >= cap rejects (parity with the pool path)."""

    async def count_active() -> int:
        return DEFAULT_MAX_ACTIVE_SCHEDULES_PER_AGENT

    assert (
        await check_active_schedule_cap(
            agent_id=uuid4(),
            cap=DEFAULT_MAX_ACTIVE_SCHEDULES_PER_AGENT,
            count_func=count_active,
        )
        is False
    )


# ---------------------------------------------------------------------------
# check_fire_limits (the permit's per-wake + per-agent limits)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_fire_limits_under_both_limits_returns_none() -> None:
    """Under both limits -> ``None``; the wake count runs first, then the agent's."""
    pool = _StubPool([3, 40])
    trigger = _trigger()
    assert await check_fire_limits(trigger, pool, FireLimits(per_wake=48, per_agent=500)) is None
    wake_sql, wake_args = pool.calls[0]
    agent_sql, agent_args = pool.calls[1]
    assert "schedule_id = $2" in wake_sql
    assert wake_args[:2] == (trigger.conversation_id, trigger.schedule_id)
    assert "ws.agent_id = $1" in agent_sql
    assert agent_args[0] == trigger.agent_id


@pytest.mark.asyncio
async def test_check_fire_limits_wake_at_limit_returns_wake_and_skips_the_agent_count() -> None:
    pool = _StubPool([48, 0])
    assert await check_fire_limits(_trigger(), pool, FireLimits(per_wake=48, per_agent=500)) == "wake"
    assert len(pool.calls) == 1


@pytest.mark.asyncio
async def test_check_fire_limits_agent_at_limit_returns_agent() -> None:
    pool = _StubPool([1, 500])
    assert await check_fire_limits(_trigger(), pool, FireLimits(per_wake=48, per_agent=500)) == "agent"


@pytest.mark.asyncio
async def test_check_fire_limits_counts_silent_fires() -> None:
    """Silent fires ran, so both counts include them."""
    pool = _StubPool([0, 0])
    await check_fire_limits(_trigger(), pool, FireLimits(per_wake=48, per_agent=500))
    for sql, _args in pool.calls:
        assert "'fired', 'fired_silent'" in sql


@pytest.mark.asyncio
async def test_check_fire_limits_leaves_protected_wakes_out_of_the_agent_count() -> None:
    pool = _StubPool([0, 0])
    await check_fire_limits(_trigger(), pool, FireLimits(per_wake=48, per_agent=500))
    agent_sql, _args = pool.calls[1]
    assert "NOT ws.protected" in agent_sql


@pytest.mark.asyncio
async def test_check_fire_limits_does_not_count_a_protected_wake() -> None:
    """A protected wake is exempt from counting: no query runs, whatever the limits."""
    pool = _StubPool([10_000, 10_000])
    trigger = dataclasses.replace(_trigger(), protected=True)
    assert await check_fire_limits(trigger, pool, FireLimits(per_wake=1, per_agent=1)) is None
    assert pool.calls == []


@pytest.mark.asyncio
async def test_check_fire_limits_counts_a_webhook_fire_by_its_subscription() -> None:
    pool = _StubPool([60, 0])
    subscription_id = uuid4()
    trigger = dataclasses.replace(
        _trigger(), schedule_id=None, fire_source="webhook", webhook_subscription_id=subscription_id
    )
    assert await check_fire_limits(trigger, pool, FireLimits(per_wake=60, per_agent=500)) == "wake"
    sql, args = pool.calls[0]
    assert "webhook_subscription_id = $2" in sql
    assert args[1] == subscription_id


# ---------------------------------------------------------------------------
# create_schedule_serialized (advisory-lock + count + insert)
# ---------------------------------------------------------------------------


# parity-with: asyncpg.Connection (the lock/count seam create_schedule_serialized
# drives: execute(advisory-lock) -> fetchval(count) -> save_entity(conn=self)).
class _SerializedConn:
    """Records the advisory-lock SQL + serves a scripted active count.

    The COUNT value is supplied by the test so the at-cap / under-cap
    branch can be exercised without a DB. ``executed`` captures every
    ``execute`` call so a test can assert the advisory lock was taken.
    """

    def __init__(self, count_value: int) -> None:
        self._count_value = count_value
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self.counted: list[tuple[str, tuple[Any, ...]]] = []
        self.txn_entered = False

    def transaction(self) -> "_SerializedConn":
        return self

    async def __aenter__(self) -> "_SerializedConn":
        self.txn_entered = True
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    async def execute(self, sql: str, *args: Any) -> str:
        self.executed.append((sql, args))
        return "SELECT 1"

    async def fetchval(self, sql: str, *args: Any) -> int:
        self.counted.append((sql, args))
        return self._count_value


# parity-with: asyncpg.Pool (acquire() context manager).
class _SerializedPool:
    """Yields a single :class:`_SerializedConn` from ``acquire()``."""

    def __init__(self, conn: _SerializedConn) -> None:
        self._conn = conn

    def acquire(self) -> _SerializedConn:
        return self._conn


# parity-with: threetears.agent.wake.collections.WakeScheduleCollection
# (only the seams create_schedule_serialized touches: create(data) and
# save_entity(conn=...)).
class _RecordingCollection:
    """Captures the ``create`` + ``save_entity`` calls (or proves they never happened)."""

    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.saved: list[Any] = []
        self.saved_conn: Any = None

    def create(self, data: dict[str, Any]) -> dict[str, Any]:
        self.created.append(data)
        return data

    async def save_entity(self, entity: Any, *, conn: Any = None) -> None:
        self.saved.append(entity)
        self.saved_conn = conn


def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {"schedule_id": uuid4(), "conversation_id": uuid4(), "protected": False}
    row.update(overrides)
    return row


@pytest.mark.asyncio
async def test_create_schedule_serialized_inserts_under_cap() -> None:
    """Under cap -> takes the agent's advisory lock then inserts on the txn conn."""
    conn = _SerializedConn(count_value=DEFAULT_MAX_ACTIVE_SCHEDULES_PER_AGENT - 1)
    pool = _SerializedPool(conn)
    collection = _RecordingCollection()
    agent_id = uuid4()
    row = _row()

    entity = await create_schedule_serialized(
        collection=collection,  # type: ignore[arg-type]
        data=row,
        agent_id=agent_id,
        cap=DEFAULT_MAX_ACTIVE_SCHEDULES_PER_AGENT,
        pool=pool,
    )

    # The advisory lock was acquired inside a transaction, keyed on the agent.
    assert conn.txn_entered is True
    locks = [args for sql, args in conn.executed if "pg_advisory_xact_lock" in sql]
    assert locks == [(str(agent_id),)]
    # The count is the agent's.
    assert conn.counted[0][1] == (agent_id,)
    # The entity was persisted, bound to the locked transaction connection.
    assert collection.saved == [entity]
    assert entity["conversation_id"] == row["conversation_id"]
    assert collection.saved_conn is conn


@pytest.mark.asyncio
async def test_create_schedule_serialized_rejects_at_cap_without_insert() -> None:
    """At cap -> raises ScheduleCapExceeded and never creates or saves."""
    conn = _SerializedConn(count_value=DEFAULT_MAX_ACTIVE_SCHEDULES_PER_AGENT)
    pool = _SerializedPool(conn)
    collection = _RecordingCollection()
    agent_id = uuid4()
    made: list[Any] = []

    async def make(conn_: Any) -> UUID:
        made.append(conn_)
        return uuid4()

    with pytest.raises(ScheduleCapExceeded) as exc_info:
        await create_schedule_serialized(
            collection=collection,  # type: ignore[arg-type]
            data=_row(),
            agent_id=agent_id,
            cap=DEFAULT_MAX_ACTIVE_SCHEDULES_PER_AGENT,
            pool=pool,
            make_wake_conversation=make,
        )

    # The advisory lock was taken (the count ran under it) before rejecting.
    assert any("pg_advisory_xact_lock" in sql for sql, _ in conn.executed)
    # The typed error carries the observed count + cap + agent.
    assert exc_info.value.cap == DEFAULT_MAX_ACTIVE_SCHEDULES_PER_AGENT
    assert exc_info.value.count == DEFAULT_MAX_ACTIVE_SCHEDULES_PER_AGENT
    assert exc_info.value.agent_id == agent_id
    # No wake conversation was made and no insert happened.
    assert made == []
    assert collection.created == []
    assert collection.saved == []


@pytest.mark.asyncio
async def test_create_schedule_serialized_makes_the_wake_conversation_on_the_locked_connection() -> None:
    conn = _SerializedConn(count_value=0)
    collection = _RecordingCollection()
    new_conversation = uuid4()
    made_on: list[Any] = []

    async def make(conn_: Any) -> UUID:
        made_on.append(conn_)
        return new_conversation

    entity = await create_schedule_serialized(
        collection=collection,  # type: ignore[arg-type]
        data=_row(),
        agent_id=uuid4(),
        cap=DEFAULT_MAX_ACTIVE_SCHEDULES_PER_AGENT,
        pool=_SerializedPool(conn),
        make_wake_conversation=make,
    )

    assert made_on == [conn]
    assert entity["conversation_id"] == new_conversation
    assert collection.saved_conn is conn


@pytest.mark.asyncio
async def test_create_schedule_serialized_does_not_count_or_refuse_a_protected_wake() -> None:
    """A protected wake is created at any count: it never crowds out, nor is refused by, the cap."""
    conn = _SerializedConn(count_value=DEFAULT_MAX_ACTIVE_SCHEDULES_PER_AGENT + 7)
    collection = _RecordingCollection()

    await create_schedule_serialized(
        collection=collection,  # type: ignore[arg-type]
        data=_row(protected=True),
        agent_id=uuid4(),
        cap=DEFAULT_MAX_ACTIVE_SCHEDULES_PER_AGENT,
        pool=_SerializedPool(conn),
    )

    assert conn.counted == []
    assert len(collection.saved) == 1
