"""Integration test: end-to-end ``wake_tick_job`` against real Postgres.

Seeds N schedules with different ``next_fire_at`` and statuses, runs
the tick body with a counting / failing dispatch callback, asserts:

- only due schedules fire (paused / future-due rows untouched).
- each fire produces exactly one ``wake_fires`` row with the right
  status.
- ``next_fire_at`` advances per ``missed_fire_policy``.
- per-fire dispatch failures are isolated and recorded as
  ``status='failed'`` rows.
- two concurrent tick bodies against the same row produce exactly
  one ``wake_fires`` insertion.

The lock-held simulation runs without a NATS broker (the
``client=None`` graceful-degradation path) so the optimistic-CAS UPDATE
on the schedule row is the single source of truth for double-fire
prevention -- which is what would actually protect us if NATS were
unreachable in prod.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import UUID

import asyncpg
import pytest
from uuid_utils import uuid7

from threetears.agent.skills.migrations import register as register_skills
from threetears.agent.wake.migrations import register as register_wake
from threetears.agent.wake.tick import wake_tick_job
from threetears.agent.wake.types import WakeDispatchResult, WakeTrigger
from threetears.conversations.migrations import register as register_conversations
from threetears.core.collections.asyncpg_init import init_connection
from threetears.core.data.migrations import MigrationRunner

from .conftest import AsyncpgStore


pytestmark = pytest.mark.integration


def _new_uuid() -> UUID:
    return UUID(str(uuid7()))


async def _apply_schema(url: str, schema: str) -> asyncpg.Pool:
    setup_conn = await asyncpg.connect(url)
    try:
        await setup_conn.execute(f'SET search_path TO "{schema}", public')
        runner = MigrationRunner()
        register_conversations(runner)
        register_skills(runner)
        register_wake(runner)
        store = AsyncpgStore(setup_conn)
        await runner.apply_for_agent_schema(store)  # type: ignore[arg-type]
    finally:
        await setup_conn.close()
    pool = await asyncpg.create_pool(
        url,
        min_size=2,
        max_size=8,
        server_settings={"search_path": f"{schema}, public"},
        init=init_connection,
    )
    assert pool is not None
    return pool


async def _seed_schedule(
    pool: asyncpg.Pool,
    *,
    next_fire_at: datetime | None,
    status: str = "active",
    schedule_type: str = "interval",
    schedule_config: dict[str, int] | None = None,
    missed_fire_policy: str = "coalesce",
    last_fired_at: datetime | None = None,
) -> tuple[UUID, UUID]:
    if schedule_config is None:
        schedule_config = {"seconds": 60}
    conv = _new_uuid()
    sched = _new_uuid()
    await pool.execute(
        "INSERT INTO agent_wake_schedules "
        "(conversation_id, schedule_id, user_id, agent_id, schedule_type, "
        " schedule_config, execution_mode, status, next_fire_at, last_fired_at, "
        " missed_fire_policy, delivery_target, delivery_config) "
        "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)",
        conv,
        sched,
        _new_uuid(),
        _new_uuid(),
        schedule_type,
        schedule_config,
        "inline",
        status,
        next_fire_at,
        last_fired_at,
        missed_fire_policy,
        "conversation",
        {},
    )
    return conv, sched


async def _count_fires(pool: asyncpg.Pool, schedule_id: UUID) -> int:
    return int(
        await pool.fetchval(
            "SELECT COUNT(*) FROM wake_fires WHERE schedule_id = $1",
            schedule_id,
        )
        or 0
    )


async def _read_schedule(pool: asyncpg.Pool, schedule_id: UUID) -> dict[str, object]:
    row = await pool.fetchrow(
        "SELECT next_fire_at, last_fired_at, status FROM agent_wake_schedules WHERE schedule_id = $1",
        schedule_id,
    )
    assert row is not None
    return dict(row)


class TestTickDispatchesOnlyDueSchedules:
    """Tick fires the past-due rows, leaves paused + future-due rows alone."""

    async def test_three_schedules_one_due(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            now = datetime.now(UTC)
            _, due = await _seed_schedule(pool, next_fire_at=now - timedelta(seconds=10))
            _, future = await _seed_schedule(pool, next_fire_at=now + timedelta(hours=1))
            _, paused = await _seed_schedule(
                pool,
                next_fire_at=now - timedelta(seconds=10),
                status="paused",
            )

            seen: list[UUID] = []

            async def dispatch(trigger: WakeTrigger, fire_id: UUID, _pool: object) -> WakeDispatchResult:
                del fire_id, _pool
                seen.append(trigger.schedule_id)
                return WakeDispatchResult(status="fired", output_text="ok", latency_ms=12)

            await wake_tick_job(pool, None, dispatch)
            assert seen == [due]
            assert await _count_fires(pool, due) == 1
            assert await _count_fires(pool, future) == 0
            assert await _count_fires(pool, paused) == 0

            # the due schedule should have its next_fire_at advanced
            due_row = await _read_schedule(pool, due)
            assert due_row["next_fire_at"] is not None
            assert due_row["last_fired_at"] is not None
            assert due_row["status"] == "active"
        finally:
            await pool.close()


class TestPerFireFailureIsolation:
    """One callback exception does not poison the rest of the tick."""

    async def test_one_failing_one_succeeding(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            now = datetime.now(UTC)
            _, due_a = await _seed_schedule(pool, next_fire_at=now - timedelta(seconds=20))
            _, due_b = await _seed_schedule(pool, next_fire_at=now - timedelta(seconds=10))

            class Boom(RuntimeError):
                pass

            async def dispatch(trigger: WakeTrigger, fire_id: UUID, _pool: object) -> WakeDispatchResult:
                del fire_id, _pool
                if trigger.schedule_id == due_a:
                    raise Boom("dispatcher exploded")
                return WakeDispatchResult(status="fired", output_text="ok")

            await wake_tick_job(pool, None, dispatch)

            # both schedules generated a fire row; due_a finalised as failed
            assert await _count_fires(pool, due_a) == 1
            assert await _count_fires(pool, due_b) == 1
            row_a = await pool.fetchrow(
                "SELECT status, error FROM wake_fires WHERE schedule_id = $1",
                due_a,
            )
            assert row_a is not None
            assert row_a["status"] == "failed"
            assert "dispatcher exploded" in row_a["error"]

            row_b = await pool.fetchrow(
                "SELECT status FROM wake_fires WHERE schedule_id = $1",
                due_b,
            )
            assert row_b is not None
            assert row_b["status"] == "fired"
        finally:
            await pool.close()

    async def test_non_raising_failed_status_records_error(self, pg_schema: tuple[str, str]) -> None:
        """A dispatch that *returns* ``status='failed'`` (without raising)
        still routes its ``error`` string onto the ``wake_fires`` row.

        Pins the Critic-flagged asymmetry: previously, only raised
        exceptions reached ``finalize_failed``; a callback that captured
        a non-exceptional failure (rate-limited downstream, no eligible
        model, etc.) into a ``WakeDispatchResult(status='failed',
        error='...')`` saw its ``error`` silently discarded by
        ``finalize_success``.
        """
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            now = datetime.now(UTC)
            _, sched = await _seed_schedule(pool, next_fire_at=now - timedelta(seconds=10))

            async def dispatch(_trigger: WakeTrigger, _fire_id: UUID, _pool: object) -> WakeDispatchResult:
                return WakeDispatchResult(
                    status="failed",
                    error="downstream rate-limited; backing off",
                    latency_ms=42,
                )

            await wake_tick_job(pool, None, dispatch)

            assert await _count_fires(pool, sched) == 1
            row = await pool.fetchrow(
                "SELECT status, error, latency_ms FROM wake_fires WHERE schedule_id = $1",
                sched,
            )
            assert row is not None
            assert row["status"] == "failed"
            assert row["error"] == "downstream rate-limited; backing off"
            assert row["latency_ms"] == 42
        finally:
            await pool.close()


class TestMissedFirePolicy:
    """``coalesce`` vs ``catch_up`` advance ``next_fire_at`` differently."""

    async def test_coalesce_jumps_forward(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            now = datetime.now(UTC)
            backlog_anchor = now - timedelta(minutes=5)
            _, sched = await _seed_schedule(
                pool,
                next_fire_at=backlog_anchor,
                schedule_type="interval",
                schedule_config={"seconds": 60},
                missed_fire_policy="coalesce",
                last_fired_at=backlog_anchor - timedelta(seconds=60),
            )

            async def dispatch(_trigger: WakeTrigger, _fire_id: UUID, _pool: object) -> WakeDispatchResult:
                return WakeDispatchResult(status="fired")

            await wake_tick_job(pool, None, dispatch)
            row = await _read_schedule(pool, sched)
            # coalesce anchors on now -> next_fire_at = now + 60s, which
            # is well into the future (NOT backlog_anchor + 60s in the past)
            assert row["next_fire_at"] is not None
            next_fire = row["next_fire_at"]
            assert isinstance(next_fire, datetime)
            assert next_fire > now
        finally:
            await pool.close()

    async def test_catch_up_advances_one_step_from_last_fire(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            now = datetime.now(UTC)
            backlog_anchor = now - timedelta(minutes=5)
            last_fire = backlog_anchor - timedelta(seconds=60)
            _, sched = await _seed_schedule(
                pool,
                next_fire_at=backlog_anchor,
                schedule_type="interval",
                schedule_config={"seconds": 60},
                missed_fire_policy="catch_up",
                last_fired_at=last_fire,
            )

            async def dispatch(_trigger: WakeTrigger, _fire_id: UUID, _pool: object) -> WakeDispatchResult:
                return WakeDispatchResult(status="fired")

            await wake_tick_job(pool, None, dispatch)
            row = await _read_schedule(pool, sched)
            next_fire = row["next_fire_at"]
            assert isinstance(next_fire, datetime)
            # catch_up advances exactly one step (60s) from last_fired_at
            assert next_fire == last_fire + timedelta(seconds=60)
        finally:
            await pool.close()


class TestConcurrentTickClaimRace:
    """Two concurrent ``wake_tick_job`` invocations against one due schedule
    produce exactly ONE ``wake_fires`` row."""

    async def test_no_double_fire_under_race(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            now = datetime.now(UTC)
            _, due = await _seed_schedule(pool, next_fire_at=now - timedelta(seconds=10))

            seen: list[UUID] = []

            async def dispatch(trigger: WakeTrigger, fire_id: UUID, _pool: object) -> WakeDispatchResult:
                del fire_id, _pool
                seen.append(trigger.schedule_id)
                # small pause inside the dispatch to widen the race window
                await asyncio.sleep(0.05)
                return WakeDispatchResult(status="fired")

            await asyncio.gather(
                wake_tick_job(pool, None, dispatch),
                wake_tick_job(pool, None, dispatch),
            )
            # the optimistic-CAS UPDATE ensures only one dispatch ran.
            assert seen.count(due) == 1
            assert await _count_fires(pool, due) == 1
        finally:
            await pool.close()


class TestOneShotTerminalTransition:
    """``one_shot_at`` schedules expire after the single fire."""

    async def test_one_shot_marks_expired(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            now = datetime.now(UTC)
            fire_iso = (now - timedelta(seconds=30)).isoformat()
            _, sched = await _seed_schedule(
                pool,
                next_fire_at=now - timedelta(seconds=10),
                schedule_type="one_shot_at",
                schedule_config={"fire_at_iso": fire_iso},
            )

            async def dispatch(_trigger: WakeTrigger, _fire_id: UUID, _pool: object) -> WakeDispatchResult:
                return WakeDispatchResult(status="fired")

            await wake_tick_job(pool, None, dispatch)
            row = await _read_schedule(pool, sched)
            assert row["status"] == "expired"
            assert row["next_fire_at"] is None
            assert await _count_fires(pool, sched) == 1
        finally:
            await pool.close()
