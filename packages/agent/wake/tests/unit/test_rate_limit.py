"""Unit tests for :mod:`threetears.agent.wake.rate_limit`.

The helper is intentionally pure-async over an asyncpg-compatible pool
(``fetchval`` only). Tests substitute a minimal in-memory stub for the
pool so the boundary contract is exercised without touching Postgres.

Three scenarios drive the per-fire helper:

- both per-conv + per-user counts under cap -> ``True``
- per-conv at cap -> ``False`` (per-user query never runs)
- per-conv under cap, per-user at cap -> ``False``

Plus the active-schedule cap helper:

- count under cap -> ``True``
- count at cap -> ``False``

The fakes are tagged with ``parity-with`` markers per the workspace
fake-parity enforcement rule.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest

from threetears.agent.wake.config import (
    DEFAULT_MAX_FIRES_PER_CONV_PER_DAY,
    DEFAULT_MAX_FIRES_PER_USER_PER_DAY,
    DEFAULT_MAX_SCHEDULES_PER_CONVERSATION,
)
from threetears.agent.wake.rate_limit import (
    ScheduleCapExceeded,
    _check_active_schedule_cap,
    _check_rate_limit,
    create_schedule_serialized,
)
from threetears.agent.wake.types import WakeTrigger


# parity-with: asyncpg.Pool (fetchval-only minimal stand-in for the
# rate-limit helper boundary)
class _StubPool:
    """Minimal asyncpg.Pool stand-in exposing ``fetchval``.

    Drives ``_check_rate_limit`` by returning a queued integer for
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
    def max_email_per_recipient_per_hour(self) -> int:
        return 5

    @property
    def max_webhook_fires_per_subscription_per_hour(self) -> int:
        return 60

    @property
    def max_schedules_per_conversation(self) -> int:
        return DEFAULT_MAX_SCHEDULES_PER_CONVERSATION

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
    assert await _check_rate_limit(trigger, pool, config) is None
    assert len(pool.calls) == 2  # both queries ran


@pytest.mark.asyncio
async def test_check_rate_limit_returns_conv_scope_when_per_conv_at_cap() -> None:
    """Per-conv count >= cap -> returns ``'conv'``, per-user query is skipped."""
    pool = _StubPool([DEFAULT_MAX_FIRES_PER_CONV_PER_DAY, 0])
    config = _StubConfig()
    assert await _check_rate_limit(_trigger(), pool, config) == "conv"
    assert len(pool.calls) == 1  # per-user query did not run


@pytest.mark.asyncio
async def test_check_rate_limit_returns_user_scope_when_per_user_at_cap() -> None:
    """Per-conv under cap + per-user at cap -> returns ``'user'`` after both queries."""
    pool = _StubPool([0, DEFAULT_MAX_FIRES_PER_USER_PER_DAY])
    config = _StubConfig()
    assert await _check_rate_limit(_trigger(), pool, config) == "user"
    assert len(pool.calls) == 2  # both ran


@pytest.mark.asyncio
async def test_check_rate_limit_returns_none_with_none_pool() -> None:
    """``None`` pool -> ``None`` (allows unit tests that omit the DB)."""
    assert await _check_rate_limit(_trigger(), None, _StubConfig()) is None


@pytest.mark.asyncio
async def test_check_active_schedule_cap_returns_true_under_cap() -> None:
    """Count strictly under cap -> True (pool path)."""
    pool = _StubPool([DEFAULT_MAX_SCHEDULES_PER_CONVERSATION - 1])
    assert (
        await _check_active_schedule_cap(
            conversation_id=uuid4(),
            cap=DEFAULT_MAX_SCHEDULES_PER_CONVERSATION,
            pool=pool,
        )
        is True
    )


@pytest.mark.asyncio
async def test_check_active_schedule_cap_returns_false_at_cap() -> None:
    """Count at the cap boundary -> False (>= rejects; pool path)."""
    pool = _StubPool([DEFAULT_MAX_SCHEDULES_PER_CONVERSATION])
    assert (
        await _check_active_schedule_cap(
            conversation_id=uuid4(),
            cap=DEFAULT_MAX_SCHEDULES_PER_CONVERSATION,
            pool=pool,
        )
        is False
    )


@pytest.mark.asyncio
async def test_check_active_schedule_cap_returns_true_with_none_pool_and_no_count_func() -> None:
    """Neither ``pool`` nor ``count_func`` supplied -> True (short-circuit)."""
    assert (
        await _check_active_schedule_cap(
            conversation_id=uuid4(),
            cap=DEFAULT_MAX_SCHEDULES_PER_CONVERSATION,
        )
        is True
    )


@pytest.mark.asyncio
async def test_check_active_schedule_cap_uses_count_func_when_supplied() -> None:
    """When ``count_func`` is supplied, it wins over ``pool``.

    Pins the tool-layer integration: ``wake_schedule_create`` passes a
    ``count_func`` closing over the collection's
    ``count_active_for_conversation``. Verifying the helper invokes the
    callable (not the pool's fetchval) keeps the SQL single-sourced.
    """
    calls: list[None] = []

    async def count_active() -> int:
        calls.append(None)
        return DEFAULT_MAX_SCHEDULES_PER_CONVERSATION - 1

    pool = _StubPool([999])  # would say "over cap" if consulted
    assert (
        await _check_active_schedule_cap(
            conversation_id=uuid4(),
            cap=DEFAULT_MAX_SCHEDULES_PER_CONVERSATION,
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
        return DEFAULT_MAX_SCHEDULES_PER_CONVERSATION

    assert (
        await _check_active_schedule_cap(
            conversation_id=uuid4(),
            cap=DEFAULT_MAX_SCHEDULES_PER_CONVERSATION,
            count_func=count_active,
        )
        is False
    )


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
        del sql, args
        return self._count_value


# parity-with: asyncpg.Pool (acquire() context manager).
class _SerializedPool:
    """Yields a single :class:`_SerializedConn` from ``acquire()``."""

    def __init__(self, conn: _SerializedConn) -> None:
        self._conn = conn

    def acquire(self) -> _SerializedConn:
        return self._conn


# parity-with: threetears.agent.wake.collections.WakeScheduleCollection
# (only the seam create_schedule_serialized touches: save_entity(conn=...)).
class _RecordingCollection:
    """Captures the ``save_entity`` call (or proves it never happened)."""

    def __init__(self) -> None:
        self.saved: list[Any] = []
        self.saved_conn: Any = None

    async def save_entity(self, entity: Any, *, conn: Any = None) -> None:
        self.saved.append(entity)
        self.saved_conn = conn


@pytest.mark.asyncio
async def test_create_schedule_serialized_inserts_under_cap() -> None:
    """Under cap -> takes the advisory lock then inserts on the txn conn."""
    conn = _SerializedConn(count_value=DEFAULT_MAX_SCHEDULES_PER_CONVERSATION - 1)
    pool = _SerializedPool(conn)
    collection = _RecordingCollection()
    entity = object()
    conv_id = uuid4()

    await create_schedule_serialized(
        collection=collection,  # type: ignore[arg-type]
        entity=entity,  # type: ignore[arg-type]
        conversation_id=conv_id,
        cap=DEFAULT_MAX_SCHEDULES_PER_CONVERSATION,
        pool=pool,
    )

    # The advisory lock was acquired inside a transaction before the insert.
    assert conn.txn_entered is True
    assert any("pg_advisory_xact_lock" in sql for sql, _ in conn.executed)
    # The entity was persisted, bound to the locked transaction connection.
    assert collection.saved == [entity]
    assert collection.saved_conn is conn


@pytest.mark.asyncio
async def test_create_schedule_serialized_rejects_at_cap_without_insert() -> None:
    """At cap -> raises ScheduleCapExceeded and never calls save_entity."""
    conn = _SerializedConn(count_value=DEFAULT_MAX_SCHEDULES_PER_CONVERSATION)
    pool = _SerializedPool(conn)
    collection = _RecordingCollection()
    conv_id = uuid4()

    with pytest.raises(ScheduleCapExceeded) as exc_info:
        await create_schedule_serialized(
            collection=collection,  # type: ignore[arg-type]
            entity=object(),  # type: ignore[arg-type]
            conversation_id=conv_id,
            cap=DEFAULT_MAX_SCHEDULES_PER_CONVERSATION,
            pool=pool,
        )

    # The advisory lock was taken (the count ran under it) before rejecting.
    assert any("pg_advisory_xact_lock" in sql for sql, _ in conn.executed)
    # The typed error carries the observed count + cap + conversation.
    assert exc_info.value.cap == DEFAULT_MAX_SCHEDULES_PER_CONVERSATION
    assert exc_info.value.count == DEFAULT_MAX_SCHEDULES_PER_CONVERSATION
    assert exc_info.value.conversation_id == conv_id
    # No insert happened.
    assert collection.saved == []
