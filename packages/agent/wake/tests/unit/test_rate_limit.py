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
    _check_active_schedule_cap,
    _check_rate_limit,
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
async def test_check_rate_limit_returns_true_when_both_counts_under_cap() -> None:
    """Both under cap -> the fire may proceed."""
    pool = _StubPool([5, 10])  # conv=5, user=10 against (24, 100)
    config = _StubConfig()
    trigger = _trigger()
    assert await _check_rate_limit(trigger, pool, config) is True
    assert len(pool.calls) == 2  # both queries ran


@pytest.mark.asyncio
async def test_check_rate_limit_returns_false_when_per_conv_at_cap() -> None:
    """Per-conv count >= cap -> reject, per-user query is skipped."""
    pool = _StubPool([DEFAULT_MAX_FIRES_PER_CONV_PER_DAY, 0])
    config = _StubConfig()
    assert await _check_rate_limit(_trigger(), pool, config) is False
    assert len(pool.calls) == 1  # per-user query did not run


@pytest.mark.asyncio
async def test_check_rate_limit_returns_false_when_per_user_at_cap() -> None:
    """Per-conv under cap + per-user at cap -> reject after both queries."""
    pool = _StubPool([0, DEFAULT_MAX_FIRES_PER_USER_PER_DAY])
    config = _StubConfig()
    assert await _check_rate_limit(_trigger(), pool, config) is False
    assert len(pool.calls) == 2  # both ran


@pytest.mark.asyncio
async def test_check_rate_limit_returns_true_with_none_pool() -> None:
    """``None`` pool -> True (allows unit tests that omit the DB)."""
    assert await _check_rate_limit(_trigger(), None, _StubConfig()) is True


@pytest.mark.asyncio
async def test_check_active_schedule_cap_returns_true_under_cap() -> None:
    """Count strictly under cap -> True."""
    pool = _StubPool([DEFAULT_MAX_SCHEDULES_PER_CONVERSATION - 1])
    assert (
        await _check_active_schedule_cap(
            conversation_id=uuid4(),
            pool=pool,
            config=_StubConfig(),
        )
        is True
    )


@pytest.mark.asyncio
async def test_check_active_schedule_cap_returns_false_at_cap() -> None:
    """Count at the cap boundary -> False (>= rejects)."""
    pool = _StubPool([DEFAULT_MAX_SCHEDULES_PER_CONVERSATION])
    assert (
        await _check_active_schedule_cap(
            conversation_id=uuid4(),
            pool=pool,
            config=_StubConfig(),
        )
        is False
    )


@pytest.mark.asyncio
async def test_check_active_schedule_cap_returns_true_with_none_pool() -> None:
    """``None`` pool -> True (parallel to the rate-limit helper)."""
    assert (
        await _check_active_schedule_cap(
            conversation_id=uuid4(),
            pool=None,
            config=_StubConfig(),
        )
        is True
    )
