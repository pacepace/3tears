"""Integration: a wake run by hand, and a protected wake's grid, on real Postgres.

- the manual-fire limit is one statement: of two callers at once exactly one
  runs, a second run inside the gap is refused, one after the gap goes through
- a manual fire is refused while a fire of the wake is dispatching
- it is recorded as run by hand with who asked, and moves neither
  ``next_fire_at`` nor ``last_fired_at``
- it spends the fire limits, or does not, as the consumer's limits say
- a protected interval wake keeps its grid across 60 -> 90 -> 60, and setting
  the interval it has writes nothing
- v009 backfills ``fire_source`` on rows written before it
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import asyncpg
import pytest
from uuid_utils import uuid7

from threetears.agent.skills.migrations import register as register_skills
from threetears.agent.wake.collections import WakeFireCollection, WakeScheduleCollection
from threetears.agent.wake.dispatch import dispatch_wake
from threetears.agent.wake.manual import ManualFireRefused, claim_manual_fire, run_manual_fire
from threetears.agent.wake.migrations import register as register_wake
from threetears.agent.wake.migrations.v009_fire_source_and_manual_fires import add_fire_source_and_manual_fires
from threetears.agent.wake.protected import update_protected
from threetears.agent.wake.tick import wake_tick_job
from threetears.agent.wake.types import (
    FireLimits,
    HandlerCallbackResult,
    PreparedWakeContext,
    WakeDispatchResult,
    WakeTrigger,
)
from threetears.conversations.migrations import register as register_conversations
from threetears.core.collections.asyncpg_init import init_connection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner

from .conftest import AsyncpgStore

pytestmark = pytest.mark.integration

_GAP = timedelta(minutes=30)


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
        await runner.apply_for_agent_schema(AsyncpgStore(setup_conn))  # type: ignore[arg-type]
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


def _collections(pool: asyncpg.Pool) -> tuple[WakeScheduleCollection, WakeFireCollection]:
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    cfg = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return WakeScheduleCollection(registry=registry, config=cfg), WakeFireCollection(registry=registry, config=cfg)


async def _seed_schedule(
    pool: asyncpg.Pool,
    *,
    agent_id: UUID,
    next_fire_at: datetime | None = None,
    last_fired_at: datetime | None = None,
    protected: bool = False,
    seconds: int = 3600,
) -> tuple[UUID, UUID]:
    conversation_id = _new_uuid()
    schedule_id = _new_uuid()
    await pool.execute(
        "INSERT INTO agent_wake_schedules "
        "(conversation_id, schedule_id, user_id, agent_id, schedule_type, schedule_config, "
        " status, next_fire_at, last_fired_at, protected, name, task_prompt) "
        "VALUES ($1, $2, $3, $4, 'interval', $5, 'active', $6, $7, $8, 'a wake', 'do the thing')",
        conversation_id,
        schedule_id,
        _new_uuid(),
        agent_id,
        {"seconds": seconds},
        next_fire_at,
        last_fired_at,
        protected,
    )
    return conversation_id, schedule_id


async def _wake(pool: asyncpg.Pool, schedule_id: UUID) -> asyncpg.Record:
    row = await pool.fetchrow(
        "SELECT next_fire_at, last_fired_at, last_manual_fire_at, date_updated, schedule_config "
        "FROM agent_wake_schedules WHERE schedule_id = $1",
        schedule_id,
    )
    assert row is not None
    return row


class _Ran:
    """A dispatch callback that records what it was handed and answers ``fired``."""

    def __init__(self, result: WakeDispatchResult | None = None, raises: Exception | None = None) -> None:
        self.triggers: list[WakeTrigger] = []
        self._result = result or WakeDispatchResult(status="fired", output_text="done", latency_ms=12)
        self._raises = raises

    async def __call__(self, trigger: WakeTrigger, fire_id: UUID, pool: Any) -> WakeDispatchResult:
        self.triggers.append(trigger)
        if self._raises is not None:
            raise self._raises
        return self._result


class TestTheManualFireLimit:
    """The limit is one statement, so it holds however many callers arrive together."""

    async def test_a_manual_fire_is_recorded_as_run_by_hand_and_leaves_the_schedule_where_it_was(
        self, pg_schema: tuple[str, str]
    ) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            schedules, fires = _collections(pool)
            agent = _new_uuid()
            due = datetime.now(UTC) + timedelta(minutes=40)
            last = datetime.now(UTC) - timedelta(minutes=20)
            _conv, sid = await _seed_schedule(pool, agent_id=agent, next_fire_at=due, last_fired_at=last)
            before = await _wake(pool, sid)
            entity = await schedules.find_for_agent(agent, sid)
            assert entity is not None

            claim = await claim_manual_fire(pool, entity, requested_by="user:abc", min_gap=_GAP)
            ran = _Ran()
            result = await run_manual_fire(claim, pool=pool, fires=fires, dispatch_callback=ran)

            assert result.status == "fired"
            row = await pool.fetchrow(
                "SELECT status, fire_source, requested_by, scheduled_fire_at, output_text FROM wake_fires "
                "WHERE fire_id = $1",
                claim.fire_id,
            )
            assert dict(row) == {
                "status": "fired",
                "fire_source": "manual",
                "requested_by": "user:abc",
                "scheduled_fire_at": None,
                "output_text": "done",
            }
            # the callback got the fire as a manual one, with who asked and the wake's own text
            (trigger,) = ran.triggers
            assert (trigger.fire_source, trigger.requested_by, trigger.task_prompt, trigger.fire_id) == (
                "manual",
                "user:abc",
                "do the thing",
                claim.fire_id,
            )
            after = await _wake(pool, sid)
            assert after["next_fire_at"] == due, "a manual fire does not move the next scheduled fire"
            assert after["last_fired_at"] == last, "a manual fire is not the wake's last scheduled fire"
            assert after["date_updated"] == before["date_updated"], "saves of the wake are fenced on date_updated"
            assert after["last_manual_fire_at"] == claim.claimed_at
        finally:
            await pool.close()

    async def test_of_two_callers_at_once_exactly_one_runs(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            schedules, _fires = _collections(pool)
            agent = _new_uuid()
            _conv, sid = await _seed_schedule(pool, agent_id=agent)
            entity = await schedules.find_for_agent(agent, sid)
            assert entity is not None

            async def one(who: str) -> str:
                try:
                    await claim_manual_fire(pool, entity, requested_by=who, min_gap=_GAP)
                except ManualFireRefused as refused:
                    return refused.reason
                return "claimed"

            outcomes = await asyncio.gather(*(one(f"user:{n}") for n in range(8)))
            assert outcomes.count("claimed") == 1, outcomes
            assert set(outcomes) - {"claimed"} <= {"too_soon", "dispatching"}
            assert await pool.fetchval("SELECT count(*) FROM wake_fires WHERE schedule_id = $1", sid) == 1
        finally:
            await pool.close()

    async def test_a_second_run_inside_the_gap_is_refused_and_one_after_it_goes_through(
        self, pg_schema: tuple[str, str]
    ) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            schedules, fires = _collections(pool)
            agent = _new_uuid()
            _conv, sid = await _seed_schedule(pool, agent_id=agent)
            entity = await schedules.find_for_agent(agent, sid)
            assert entity is not None
            start = datetime.now(UTC) - timedelta(hours=2)

            first = await claim_manual_fire(pool, entity, requested_by="user:a", min_gap=_GAP, now=start)
            await run_manual_fire(first, pool=pool, fires=fires, dispatch_callback=_Ran())

            with pytest.raises(ManualFireRefused) as soon:
                await claim_manual_fire(
                    pool, entity, requested_by="user:a", min_gap=_GAP, now=start + timedelta(minutes=29)
                )
            assert soon.value.reason == "too_soon"
            assert soon.value.retry_at == start + _GAP
            assert await pool.fetchval("SELECT count(*) FROM wake_fires WHERE schedule_id = $1", sid) == 1

            # good traffic: the gap has passed, so the next run is taken
            later = await claim_manual_fire(
                pool, entity, requested_by="user:a", min_gap=_GAP, now=start + timedelta(minutes=30)
            )
            assert later.fire_id != first.fire_id
            assert await pool.fetchval("SELECT count(*) FROM wake_fires WHERE schedule_id = $1", sid) == 2
        finally:
            await pool.close()

    async def test_it_is_refused_while_a_fire_of_the_wake_is_dispatching_and_runs_once_that_fire_ends(
        self, pg_schema: tuple[str, str]
    ) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            schedules, fires = _collections(pool)
            agent = _new_uuid()
            conv, sid = await _seed_schedule(pool, agent_id=agent)
            other_conv, other = await _seed_schedule(pool, agent_id=agent)
            entity = await schedules.find_for_agent(agent, sid)
            assert entity is not None
            scheduled_fire = _new_uuid()
            await fires.create_dispatching(
                fire_id=scheduled_fire,
                schedule_id=sid,
                webhook_subscription_id=None,
                conversation_id=conv,
                scheduled_fire_at=datetime.now(UTC),
                actual_fired_at=datetime.now(UTC),
                fire_source="scheduled_tick",
                execution_mode="spawn",
            )
            with pytest.raises(ManualFireRefused) as busy:
                await claim_manual_fire(pool, entity, requested_by="user:a", min_gap=_GAP)
            assert busy.value.reason == "dispatching"
            assert (await _wake(pool, sid))["last_manual_fire_at"] is None, "a refused run takes no limit"

            # good traffic: another wake of the agent is not held up by this one's fire
            other_entity = await schedules.find_for_agent(agent, other)
            assert other_entity is not None
            assert (await claim_manual_fire(pool, other_entity, requested_by="user:a", min_gap=_GAP)).fire_id

            await fires.finalize_success(conv, scheduled_fire, status="fired")
            assert (await claim_manual_fire(pool, entity, requested_by="user:a", min_gap=_GAP)).fire_id
            _ = other_conv
        finally:
            await pool.close()

    async def test_a_callback_that_raises_leaves_a_failed_fire_and_no_fire_in_flight(
        self, pg_schema: tuple[str, str]
    ) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            schedules, fires = _collections(pool)
            agent = _new_uuid()
            _conv, sid = await _seed_schedule(pool, agent_id=agent)
            entity = await schedules.find_for_agent(agent, sid)
            assert entity is not None
            claim = await claim_manual_fire(pool, entity, requested_by="user:a", min_gap=_GAP)
            result = await run_manual_fire(
                claim, pool=pool, fires=fires, dispatch_callback=_Ran(raises=RuntimeError("the model is down"))
            )
            assert (result.status, result.error) == ("failed", "the model is down")
            row = await pool.fetchrow("SELECT status, error FROM wake_fires WHERE fire_id = $1", claim.fire_id)
            assert (row["status"], row["error"]) == ("failed", "the model is down")
        finally:
            await pool.close()

    async def test_a_gap_of_nothing_is_refused_before_anything_is_written(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            schedules, _fires = _collections(pool)
            agent = _new_uuid()
            _conv, sid = await _seed_schedule(pool, agent_id=agent)
            entity = await schedules.find_for_agent(agent, sid)
            assert entity is not None
            with pytest.raises(ValueError, match="min_gap must be positive"):
                await claim_manual_fire(pool, entity, requested_by="user:a", min_gap=timedelta(0))
            assert await pool.fetchval("SELECT count(*) FROM wake_fires WHERE schedule_id = $1", sid) == 0
        finally:
            await pool.close()


class _Handler:
    def __init__(self) -> None:
        self.triggers: list[WakeTrigger] = []

    async def __call__(self, trigger: WakeTrigger, prepared: PreparedWakeContext, pool: Any) -> HandlerCallbackResult:
        self.triggers.append(trigger)
        return HandlerCallbackResult(
            status="fired",
            assistant_message_content="ok",
            target_conversation_id=trigger.conversation_id,
            assistant_message_id=None,
            latency_ms=1,
            error=None,
        )


class TestAManualFireAndTheFireLimits:
    """A manual fire spends the wake's limits by default; the consumer's limits can leave it out."""

    async def _one_manual_fire_run(self, pool: asyncpg.Pool, agent: UUID) -> tuple[UUID, UUID]:
        schedules, fires = _collections(pool)
        conv, sid = await _seed_schedule(pool, agent_id=agent)
        entity = await schedules.find_for_agent(agent, sid)
        assert entity is not None
        claim = await claim_manual_fire(
            pool, entity, requested_by="user:a", min_gap=_GAP, now=datetime.now(UTC) - timedelta(hours=1)
        )
        await run_manual_fire(claim, pool=pool, fires=fires, dispatch_callback=_Ran())
        return conv, sid

    def _scheduled(self, agent: UUID, conv: UUID, sid: UUID, source: str = "scheduled_tick") -> WakeTrigger:
        return WakeTrigger(
            schedule_id=sid,
            user_id=_new_uuid(),
            agent_id=agent,
            conversation_id=conv,
            fire_source=source,  # type: ignore[arg-type]
            execution_mode="spawn",
            schedule_type="interval",
            fired_at=datetime.now(UTC),
            requested_by="user:a" if source == "manual" else None,
        )

    async def test_a_manual_fire_spends_the_wakes_limit(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            agent = _new_uuid()
            conv, sid = await self._one_manual_fire_run(pool, agent)

            async def one_a_day(_trigger: WakeTrigger) -> FireLimits | None:
                return FireLimits(per_wake=1, per_agent=10)

            handler = _Handler()
            capped = await dispatch_wake(
                self._scheduled(agent, conv, sid), _new_uuid(), pool, handler=handler, permit=one_a_day
            )
            assert capped.status == "skipped_rate_limit"
            assert handler.triggers == []
        finally:
            await pool.close()

    async def test_limits_that_leave_manual_fires_out_neither_count_one_nor_refuse_one(
        self, pg_schema: tuple[str, str]
    ) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            agent = _new_uuid()
            conv, sid = await self._one_manual_fire_run(pool, agent)

            async def uncounted(_trigger: WakeTrigger) -> FireLimits | None:
                return FireLimits(per_wake=1, per_agent=1, manual_fires_count=False)

            handler = _Handler()
            scheduled = await dispatch_wake(
                self._scheduled(agent, conv, sid), _new_uuid(), pool, handler=handler, permit=uncounted
            )
            assert scheduled.status == "fired", "the manual fire took none of the scheduled fire's budget"
            # that scheduled fire is now on the record, at the limit; a manual one is still let through
            fire_id = _new_uuid()
            _schedules, fires = _collections(pool)
            await fires.create_dispatching(
                fire_id=fire_id,
                schedule_id=sid,
                webhook_subscription_id=None,
                conversation_id=conv,
                scheduled_fire_at=None,
                actual_fired_at=datetime.now(UTC),
                fire_source="scheduled_tick",
                execution_mode="spawn",
            )
            await fires.finalize_success(conv, fire_id, status="fired")
            by_hand = await dispatch_wake(
                self._scheduled(agent, conv, sid, "manual"), _new_uuid(), pool, handler=handler, permit=uncounted
            )
            assert by_hand.status == "fired"
            again = await dispatch_wake(
                self._scheduled(agent, conv, sid), _new_uuid(), pool, handler=handler, permit=uncounted
            )
            assert again.status == "skipped_rate_limit", "a scheduled fire is still held to the limit"
        finally:
            await pool.close()


class TestTheTickStillRecordsItsSource:
    async def test_a_scheduled_fire_is_recorded_as_scheduled(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            schedules, fires = _collections(pool)
            _conv, sid = await _seed_schedule(
                pool, agent_id=_new_uuid(), next_fire_at=datetime.now(UTC) - timedelta(seconds=5)
            )

            async def dispatch(trigger: WakeTrigger, fire_id: UUID, pool_: Any) -> WakeDispatchResult:
                return WakeDispatchResult(status="fired")

            await wake_tick_job(pool, None, dispatch, schedules=schedules, fires=fires)
            row = await pool.fetchrow("SELECT fire_source, requested_by FROM wake_fires WHERE schedule_id = $1", sid)
            assert (row["fire_source"], row["requested_by"]) == ("scheduled_tick", None)
        finally:
            await pool.close()


class TestTheGridOfAProtectedWake:
    """An interval edit keeps the anchor the wake had, and the same interval writes nothing."""

    async def test_sixty_then_ninety_then_sixty_is_due_where_it_was(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            schedules, _fires = _collections(pool)
            agent = _new_uuid()
            now = datetime(2026, 10, 10, 12, 10, tzinfo=UTC)
            on_the_hour = datetime(2026, 10, 10, 13, 0, tzinfo=UTC)
            _conv, sid = await _seed_schedule(pool, agent_id=agent, protected=True, next_fire_at=on_the_hour)

            to_90 = await update_protected(
                collection=schedules, agent_id=agent, schedule_id=sid, schedule_config={"seconds": 5400}, now=now
            )
            # the anchor is 12:00, the occurrence before the next one: 12:00 + 90 min
            assert to_90.next_fire_after == datetime(2026, 10, 10, 13, 30, tzinfo=UTC)

            back = await update_protected(
                collection=schedules,
                agent_id=agent,
                schedule_id=sid,
                schedule_config={"seconds": 3600},
                now=now + timedelta(minutes=1),
            )
            assert back.changed
            assert back.next_fire_after == on_the_hour, "60 -> 90 -> 60 is back on the grid it left"
            assert (await _wake(pool, sid))["next_fire_at"] == on_the_hour
        finally:
            await pool.close()

    async def test_a_shorter_interval_takes_the_first_slot_of_the_kept_grid_after_now(
        self, pg_schema: tuple[str, str]
    ) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            schedules, _fires = _collections(pool)
            agent = _new_uuid()
            now = datetime(2026, 10, 10, 12, 40, tzinfo=UTC)
            _conv, sid = await _seed_schedule(
                pool, agent_id=agent, protected=True, next_fire_at=datetime(2026, 10, 10, 13, 0, tzinfo=UTC)
            )
            change = await update_protected(
                collection=schedules, agent_id=agent, schedule_id=sid, schedule_config={"seconds": 1800}, now=now
            )
            # anchor 12:00, step 30 min: 12:30 has passed, so 13:00, never a slot in the past
            assert change.next_fire_after == datetime(2026, 10, 10, 13, 0, tzinfo=UTC)
        finally:
            await pool.close()

    async def test_the_interval_it_already_has_writes_nothing(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            schedules, _fires = _collections(pool)
            agent = _new_uuid()
            due = datetime(2026, 10, 10, 13, 0, tzinfo=UTC)
            _conv, sid = await _seed_schedule(pool, agent_id=agent, protected=True, next_fire_at=due)
            before = await _wake(pool, sid)

            same = await update_protected(
                collection=schedules,
                agent_id=agent,
                schedule_id=sid,
                schedule_config={"seconds": 3600},
                now=datetime(2026, 10, 10, 12, 10, tzinfo=UTC),
            )

            assert not same.changed
            assert (same.next_fire_before, same.next_fire_after) == (due, due)
            after = await _wake(pool, sid)
            assert after["next_fire_at"] == due, "the next fire stays where it was"
            assert after["date_updated"] == before["date_updated"], "nothing was written"
        finally:
            await pool.close()


class TestV009Backfill:
    async def test_rows_written_before_it_say_what_started_them_and_a_second_run_changes_nothing(
        self, pg_schema: tuple[str, str]
    ) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            agent = _new_uuid()
            conv, sid = await _seed_schedule(pool, agent_id=agent)
            sub = _new_uuid()
            await pool.execute(
                "INSERT INTO webhook_subscriptions (conversation_id, subscription_id, user_id, agent_id, "
                "secret_ciphertext, task_prompt_template) VALUES ($1, $2, $3, $4, $5, 't')",
                conv,
                sub,
                _new_uuid(),
                agent,
                b"x",
            )
            # rows as v008 left them: the column carries its default on both
            scheduled, delivered, by_hand = _new_uuid(), _new_uuid(), _new_uuid()
            await pool.execute(
                "INSERT INTO wake_fires (conversation_id, fire_id, schedule_id, status) VALUES ($1, $2, $3, 'fired')",
                conv,
                scheduled,
                sid,
            )
            await pool.execute(
                "INSERT INTO wake_fires (conversation_id, fire_id, webhook_subscription_id, status) "
                "VALUES ($1, $2, $3, 'fired')",
                conv,
                delivered,
                sub,
            )
            await pool.execute(
                "INSERT INTO wake_fires (conversation_id, fire_id, schedule_id, status, fire_source, requested_by) "
                "VALUES ($1, $2, $3, 'fired', 'manual', 'user:a')",
                conv,
                by_hand,
                sid,
            )
            async with pool.acquire() as conn:
                for _ in range(2):
                    await add_fire_source_and_manual_fires(AsyncpgStore(conn))  # type: ignore[arg-type]
            rows = {
                r["fire_id"]: r["fire_source"] for r in await pool.fetch("SELECT fire_id, fire_source FROM wake_fires")
            }
            assert rows == {scheduled: "scheduled_tick", delivered: "webhook", by_hand: "manual"}
            with pytest.raises(asyncpg.CheckViolationError):
                await pool.execute(
                    "INSERT INTO wake_fires (conversation_id, fire_id, schedule_id, status, fire_source) "
                    "VALUES ($1, $2, $3, 'fired', 'manual')",
                    conv,
                    _new_uuid(),
                    sid,
                )
        finally:
            await pool.close()
