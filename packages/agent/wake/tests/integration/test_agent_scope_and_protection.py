"""Integration: agent-scoped wakes, protected wakes, fire links and the permit, on real Postgres.

Each class drives the real code path against the migrated schema:

- the protection trigger refuses delete / pause / expire / retype /
  unprotect, and the unprotected config change; the tick still fires a
  protected wake
- :func:`update_protected` and :func:`delete_protected` are the only doors,
  and their gate lasts one transaction
- the agent lookups span conversations and stop at the agent
- ``context_from`` reads another conversation's wake of the same agent
- the permit's "not now" is written as ``'skipped_life_off'``; its limits
  count per wake and per agent, and a protected wake is not counted
- the fire-conversation hook links the fire in the conversation's own
  transaction, before the handler runs
- the reaper hands each reaped fire's started conversation to the hook,
  and a late finish cannot flip a reaped fire
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import asyncpg
import pytest
from uuid_utils import uuid7

from threetears.agent.skills.migrations import register as register_skills
from threetears.agent.wake.collections import (
    WakeFireCollection,
    WakeScheduleCollection,
    WebhookSubscriptionCollection,
)
from threetears.agent.wake.dispatch import dispatch_wake, is_tool_only
from threetears.agent.wake.migrations import register as register_wake
from threetears.agent.wake.protected import ProtectedWakeError, delete_protected, update_protected
from threetears.agent.wake.rate_limit import ScheduleCapExceeded, create_schedule_serialized
from threetears.agent.wake.tick import wake_tick_job
from threetears.agent.wake.types import (
    FireLimits,
    HandlerCallback,
    HandlerCallbackResult,
    PreparedWakeContext,
    ReapedFire,
    WakeDispatchResult,
    WakeTrigger,
)
from threetears.conversations.migrations import register as register_conversations
from threetears.core.collections import CallerTransaction
from threetears.core.collections.asyncpg_init import init_connection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner
from threetears.scheduled_jobs.collections import REAPED_DISPATCH_ERROR

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
        await runner.apply_for_agent_schema(AsyncpgStore(setup_conn))  # type: ignore[arg-type]
        # the consumer's conversations, as far as these tests need them
        await setup_conn.execute("CREATE TABLE fire_conversations (conversation_id UUID PRIMARY KEY, fire_id UUID)")
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


def _tick_collections(pool: asyncpg.Pool) -> dict[str, Any]:
    """the ``schedules=`` / ``fires=`` a tick with no NATS client runs on."""
    schedules, fires = _collections(pool)
    return {"schedules": schedules, "fires": fires}


async def _seed_schedule(
    pool: asyncpg.Pool,
    *,
    agent_id: UUID,
    conversation_id: UUID | None = None,
    next_fire_at: datetime | None = None,
    protected: bool = False,
    schedule_type: str = "interval",
    schedule_config: dict[str, int] | None = None,
    context_from: UUID | None = None,
    status: str = "active",
    name: str | None = None,
) -> tuple[UUID, UUID]:
    conversation_id = conversation_id or _new_uuid()
    schedule_id = _new_uuid()
    await pool.execute(
        "INSERT INTO agent_wake_schedules "
        "(conversation_id, schedule_id, user_id, agent_id, schedule_type, schedule_config, "
        " status, next_fire_at, protected, context_from_schedule_id, name) "
        "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)",
        conversation_id,
        schedule_id,
        _new_uuid(),
        agent_id,
        schedule_type,
        schedule_config or {"seconds": 3600},
        status,
        next_fire_at,
        protected,
        context_from,
        name,
    )
    return conversation_id, schedule_id


async def _fire_row(pool: asyncpg.Pool, fire_id: UUID) -> dict[str, Any]:
    row = await pool.fetchrow("SELECT * FROM wake_fires WHERE fire_id = $1", fire_id)
    assert row is not None
    return dict(row)


class _RecordingHandler(HandlerCallback):
    """Records every fire it is handed and answers ``fired``."""

    def __init__(self) -> None:
        self.triggers: list[WakeTrigger] = []
        self.prepared: list[PreparedWakeContext] = []

    async def __call__(
        self, trigger: WakeTrigger, prepared_context: PreparedWakeContext, pool: Any
    ) -> HandlerCallbackResult:
        del pool
        self.triggers.append(trigger)
        self.prepared.append(prepared_context)
        return HandlerCallbackResult(
            status="fired",
            assistant_message_content="done",
            target_conversation_id=trigger.started_conversation_id or trigger.conversation_id,
        )


class TestProtectedTrigger:
    """The table refuses what a protected wake may never do."""

    async def test_protected_wake_refuses_delete_pause_expire_retype_and_unprotect(
        self, pg_schema: tuple[str, str]
    ) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            schedules, _fires = _collections(pool)
            conv, sid = await _seed_schedule(pool, agent_id=_new_uuid(), protected=True)
            refused = [
                ("DELETE FROM agent_wake_schedules WHERE schedule_id = $1", (sid,)),
                ("UPDATE agent_wake_schedules SET protected = false WHERE schedule_id = $1", (sid,)),
                ("UPDATE agent_wake_schedules SET schedule_type = 'cron' WHERE schedule_id = $1", (sid,)),
                (
                    "UPDATE agent_wake_schedules SET schedule_config = '{\"seconds\": 60}'::jsonb WHERE schedule_id = $1",
                    (sid,),
                ),
            ]
            for sql, args in refused:
                with pytest.raises(asyncpg.CheckViolationError):
                    await pool.execute(sql, *args)
            with pytest.raises(asyncpg.CheckViolationError, match="paused or expired"):
                await schedules.pause(conv, sid)
            with pytest.raises(asyncpg.CheckViolationError, match="paused or expired"):
                await schedules.mark_expired(conv, sid)
            row = await pool.fetchrow(
                "SELECT status, protected, schedule_type, schedule_config FROM agent_wake_schedules WHERE schedule_id = $1",
                sid,
            )
            assert row is not None
            assert (row["status"], row["protected"], row["schedule_type"], row["schedule_config"]) == (
                "active",
                True,
                "interval",
                {"seconds": 3600},
            )
        finally:
            await pool.close()

    async def test_an_unprotected_wake_is_untouched_by_the_trigger(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            schedules, _fires = _collections(pool)
            conv, sid = await _seed_schedule(pool, agent_id=_new_uuid())
            await schedules.pause(conv, sid)
            await pool.execute("DELETE FROM agent_wake_schedules WHERE schedule_id = $1", sid)
            assert await pool.fetchval("SELECT count(*) FROM agent_wake_schedules") == 0
        finally:
            await pool.close()

    async def test_a_protected_wake_cannot_be_a_one_shot(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            with pytest.raises(asyncpg.CheckViolationError, match="protected_type_check"):
                await _seed_schedule(
                    pool,
                    agent_id=_new_uuid(),
                    protected=True,
                    schedule_type="one_shot_at",
                    schedule_config={},
                )
        finally:
            await pool.close()

    async def test_the_tick_still_fires_a_protected_wake(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            _conv, sid = await _seed_schedule(
                pool, agent_id=_new_uuid(), protected=True, next_fire_at=datetime.now(UTC) - timedelta(seconds=5)
            )
            seen: list[WakeTrigger] = []

            async def dispatch(trigger: WakeTrigger, fire_id: UUID, _pool: object) -> WakeDispatchResult:
                seen.append(trigger)
                assert trigger.fire_id == fire_id
                return WakeDispatchResult(status="fired")

            await wake_tick_job(pool, None, dispatch, **_tick_collections(pool))
            assert len(seen) == 1
            assert seen[0].protected is True
            row = await pool.fetchrow(
                "SELECT status, next_fire_at FROM agent_wake_schedules WHERE schedule_id = $1", sid
            )
            assert row is not None
            assert row["status"] == "active"
            assert row["next_fire_at"] > datetime.now(UTC)
        finally:
            await pool.close()


class TestProtectedDoors:
    """``update_protected`` and ``delete_protected`` open the gate for one transaction."""

    async def test_update_protected_changes_the_interval_and_next_fire(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            schedules, _fires = _collections(pool)
            agent = _new_uuid()
            _conv, sid = await _seed_schedule(pool, agent_id=agent, protected=True)
            now = datetime.now(UTC)
            updated = await update_protected(
                collection=schedules, agent_id=agent, schedule_id=sid, schedule_config={"seconds": 7200}, now=now
            )
            assert updated.schedule_config == {"seconds": 7200}
            assert updated.status == "active"
            assert updated.next_fire_at == now + timedelta(seconds=7200)
            # the gate closed with the transaction: a plain change is refused again
            with pytest.raises(asyncpg.CheckViolationError):
                await pool.execute(
                    "UPDATE agent_wake_schedules SET schedule_config = '{\"seconds\": 60}'::jsonb WHERE schedule_id = $1",
                    sid,
                )
        finally:
            await pool.close()

    async def test_update_protected_refuses_a_bad_config_and_changes_nothing(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            schedules, _fires = _collections(pool)
            agent = _new_uuid()
            _conv, sid = await _seed_schedule(pool, agent_id=agent, protected=True)
            with pytest.raises(ProtectedWakeError, match="positive int"):
                await update_protected(
                    collection=schedules, agent_id=agent, schedule_id=sid, schedule_config={"seconds": 0}
                )
            assert await pool.fetchval(
                "SELECT schedule_config FROM agent_wake_schedules WHERE schedule_id = $1", sid
            ) == {"seconds": 3600}
        finally:
            await pool.close()

    async def test_update_protected_refuses_another_agent_and_an_unprotected_wake(
        self, pg_schema: tuple[str, str]
    ) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            schedules, _fires = _collections(pool)
            agent = _new_uuid()
            _conv, protected_id = await _seed_schedule(pool, agent_id=agent, protected=True)
            _conv2, plain_id = await _seed_schedule(pool, agent_id=agent)
            with pytest.raises(ProtectedWakeError, match="has no wake"):
                await update_protected(
                    collection=schedules,
                    agent_id=_new_uuid(),
                    schedule_id=protected_id,
                    schedule_config={"seconds": 60},
                )
            with pytest.raises(ProtectedWakeError, match="not protected"):
                await update_protected(
                    collection=schedules, agent_id=agent, schedule_id=plain_id, schedule_config={"seconds": 60}
                )
        finally:
            await pool.close()

    async def test_delete_protected_deletes_inside_the_callers_transaction_only(
        self, pg_schema: tuple[str, str]
    ) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            schedules, _fires = _collections(pool)
            agent = _new_uuid()
            _conv, doomed = await _seed_schedule(pool, agent_id=agent, protected=True)
            _conv2, survivor = await _seed_schedule(pool, agent_id=_new_uuid(), protected=True)
            async with pool.acquire() as conn, CallerTransaction(conn):
                await delete_protected(collection=schedules, agent_id=agent, schedule_id=doomed, conn=conn)
            assert await pool.fetchval("SELECT count(*) FROM agent_wake_schedules WHERE schedule_id = $1", doomed) == 0
            # the gate did not outlive that transaction
            with pytest.raises(asyncpg.CheckViolationError):
                await pool.execute("DELETE FROM agent_wake_schedules WHERE schedule_id = $1", survivor)
            with pytest.raises(ProtectedWakeError):
                await delete_protected(collection=schedules, agent_id=agent, schedule_id=survivor)
        finally:
            await pool.close()


class TestAgentScope:
    """Lookups span the agent's conversations and stop at the agent."""

    async def test_find_and_list_for_agent(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            schedules, _fires = _collections(pool)
            agent, stranger = _new_uuid(), _new_uuid()
            conv_a, sid_a = await _seed_schedule(pool, agent_id=agent)
            conv_b, sid_b = await _seed_schedule(pool, agent_id=agent, status="paused")
            _conv_c, sid_c = await _seed_schedule(pool, agent_id=stranger)
            found = await schedules.find_for_agent(agent, sid_b)
            assert found is not None and found.conversation_id == conv_b
            assert await schedules.find_for_agent(agent, sid_c) is None
            listed = {(s.conversation_id, s.schedule_id) for s in await schedules.list_for_agent(agent)}
            assert listed == {(conv_a, sid_a), (conv_b, sid_b)}
            active = [s.schedule_id for s in await schedules.list_for_agent(agent, include_paused=False)]
            assert active == [sid_a]
        finally:
            await pool.close()

    async def test_the_active_cap_counts_across_conversations_and_skips_protected(
        self, pg_schema: tuple[str, str]
    ) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            schedules, _fires = _collections(pool)
            agent = _new_uuid()
            await _seed_schedule(pool, agent_id=agent)
            await _seed_schedule(pool, agent_id=agent)
            await _seed_schedule(pool, agent_id=agent, protected=True)
            now = datetime.now(UTC)

            def row(**extra: Any) -> dict[str, Any]:
                base: dict[str, Any] = {
                    "schedule_id": _new_uuid(),
                    "conversation_id": _new_uuid(),
                    "user_id": _new_uuid(),
                    "agent_id": agent,
                    "schedule_type": "interval",
                    "schedule_config": {"seconds": 60},
                    "status": "active",
                    "next_fire_at": now,
                    "date_created": now,
                    "date_updated": now,
                }
                base.update(extra)
                return base

            await create_schedule_serialized(collection=schedules, data=row(), agent_id=agent, cap=3, pool=pool)
            with pytest.raises(ScheduleCapExceeded):
                await create_schedule_serialized(collection=schedules, data=row(), agent_id=agent, cap=3, pool=pool)
            # a protected wake is created whatever the count
            await create_schedule_serialized(
                collection=schedules, data=row(protected=True), agent_id=agent, cap=3, pool=pool
            )
            assert await pool.fetchval("SELECT count(*) FROM agent_wake_schedules WHERE agent_id = $1", agent) == 5
        finally:
            await pool.close()

    async def test_webhook_subscriptions_find_and_list_for_agent(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            registry = CollectionRegistry()
            registry.configure(l3_pool=pool)
            subs = WebhookSubscriptionCollection(
                registry=registry, config=DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
            )
            agent = _new_uuid()
            mine = _new_uuid()
            theirs = _new_uuid()
            for sid, owner in ((mine, agent), (theirs, _new_uuid())):
                await pool.execute(
                    "INSERT INTO webhook_subscriptions (conversation_id, subscription_id, user_id, agent_id, secret_ciphertext) "
                    "VALUES ($1, $2, $3, $4, $5)",
                    _new_uuid(),
                    sid,
                    _new_uuid(),
                    owner,
                    b"x",
                )
            found = await subs.find_for_agent(agent, mine)
            assert found is not None and found.execution_mode == "spawn"
            assert await subs.find_for_agent(agent, theirs) is None
            assert [s.subscription_id for s in await subs.list_for_agent(agent)] == [mine]
        finally:
            await pool.close()


class TestContextFromAcrossConversations:
    async def test_downstream_reads_the_upstream_wake_in_another_conversation(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            _schedules, fires = _collections(pool)
            agent = _new_uuid()
            up_conv, upstream = await _seed_schedule(pool, agent_id=agent, name="mailbox")
            upstream_fire = _new_uuid()
            await fires.create_dispatching(
                fire_id=upstream_fire,
                schedule_id=upstream,
                webhook_subscription_id=None,
                conversation_id=up_conv,
                scheduled_fire_at=None,
                actual_fired_at=datetime.now(UTC) - timedelta(minutes=5),
                fire_source="scheduled_tick",
                execution_mode="spawn",
            )
            await fires.finalize_success(up_conv, upstream_fire, status="fired", output_text="three new letters")
            down_conv, downstream = await _seed_schedule(pool, agent_id=agent, context_from=upstream)
            assert down_conv != up_conv

            handler = _RecordingHandler()
            trigger = WakeTrigger(
                schedule_id=downstream,
                user_id=_new_uuid(),
                agent_id=agent,
                conversation_id=down_conv,
                fire_source="scheduled_tick",
                execution_mode="spawn",
                schedule_type="interval",
                fired_at=datetime.now(UTC),
                context_from_schedule_id=upstream,
            )
            await dispatch_wake(trigger, _new_uuid(), pool, handler=handler)
            (block,) = handler.prepared[0].context_blocks
            assert '"mailbox"' in block and "three new letters" in block

            # the same upstream id, read as another agent, gives nothing
            stranger = dataclasses.replace(trigger, agent_id=_new_uuid())
            await dispatch_wake(stranger, _new_uuid(), pool, handler=handler)
            assert handler.prepared[1].context_blocks == ()
        finally:
            await pool.close()


class TestPermitAndLimits:
    async def test_not_now_writes_skipped_life_off_and_runs_nothing(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            _conv, sid = await _seed_schedule(
                pool, agent_id=_new_uuid(), protected=True, next_fire_at=datetime.now(UTC) - timedelta(seconds=5)
            )
            handler = _RecordingHandler()
            started: list[UUID] = []

            async def not_now(_trigger: WakeTrigger) -> FireLimits | None:
                return None

            async def start(trigger: WakeTrigger, conn: Any) -> UUID:
                started.append(trigger.fire_id or _new_uuid())
                return _new_uuid()

            async def dispatch(trigger: WakeTrigger, fire_id: UUID, pool_: object) -> WakeDispatchResult:
                return await dispatch_wake(
                    trigger, fire_id, pool_, handler=handler, permit=not_now, start_conversation=start
                )

            await wake_tick_job(pool, None, dispatch, **_tick_collections(pool))
            status, started_conversation = await pool.fetchrow(
                "SELECT status, started_conversation_id FROM wake_fires WHERE schedule_id = $1", sid
            )
            assert status == "skipped_life_off"
            assert started_conversation is None
            assert handler.triggers == []
            assert started == []
        finally:
            await pool.close()

    async def test_the_per_wake_limit_counts_silent_fires_and_skips_protected(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            _schedules, fires = _collections(pool)
            agent = _new_uuid()
            plain_conv, plain = await _seed_schedule(pool, agent_id=agent)
            life_conv, life = await _seed_schedule(pool, agent_id=agent, protected=True)
            for conv, sid, status in ((plain_conv, plain, "fired_silent"), (life_conv, life, "fired")):
                fire_id = _new_uuid()
                await fires.create_dispatching(
                    fire_id=fire_id,
                    schedule_id=sid,
                    webhook_subscription_id=None,
                    conversation_id=conv,
                    scheduled_fire_at=None,
                    actual_fired_at=datetime.now(UTC) - timedelta(minutes=1),
                    fire_source="scheduled_tick",
                    execution_mode="spawn",
                )
                await fires.finalize_success(conv, fire_id, status=status)

            async def one_a_day(_trigger: WakeTrigger) -> FireLimits | None:
                return FireLimits(per_wake=1, per_agent=1)

            def trigger(conv: UUID, sid: UUID, protected: bool) -> WakeTrigger:
                return WakeTrigger(
                    schedule_id=sid,
                    user_id=_new_uuid(),
                    agent_id=agent,
                    conversation_id=conv,
                    fire_source="scheduled_tick",
                    execution_mode="spawn",
                    schedule_type="interval",
                    fired_at=datetime.now(UTC),
                    protected=protected,
                )

            handler = _RecordingHandler()
            capped = await dispatch_wake(
                trigger(plain_conv, plain, False), _new_uuid(), pool, handler=handler, permit=one_a_day
            )
            assert capped.status == "skipped_rate_limit"
            assert "(wake)" in (capped.error or "")
            exempt = await dispatch_wake(
                trigger(life_conv, life, True), _new_uuid(), pool, handler=handler, permit=one_a_day
            )
            assert exempt.status == "fired"
            assert len(handler.triggers) == 1
        finally:
            await pool.close()


class TestAQuietCheck:
    """A consumer's check that found nothing: recorded as ``checked_quiet``, never counted as a fire."""

    async def test_a_quiet_check_from_the_handler_is_written_as_checked_quiet(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            _conv, sid = await _seed_schedule(
                pool, agent_id=_new_uuid(), next_fire_at=datetime.now(UTC) - timedelta(seconds=5)
            )

            class _Quiet(HandlerCallback):
                async def __call__(
                    self, trigger: WakeTrigger, prepared_context: PreparedWakeContext, pool: Any
                ) -> HandlerCallbackResult:
                    del prepared_context, pool
                    return HandlerCallbackResult(
                        status="checked_quiet",
                        assistant_message_content="Mail check: nothing new",
                        target_conversation_id=trigger.conversation_id,
                    )

            async def dispatch(trigger: WakeTrigger, fire_id: UUID, pool_: object) -> WakeDispatchResult:
                return await dispatch_wake(trigger, fire_id, pool_, handler=_Quiet())

            await wake_tick_job(pool, None, dispatch, **_tick_collections(pool))
            row = await pool.fetchrow(
                "SELECT status, display_suppressed, error, started_conversation_id FROM wake_fires WHERE schedule_id = $1",
                sid,
            )
            assert dict(row) == {
                "status": "checked_quiet",
                "display_suppressed": False,
                "error": None,
                "started_conversation_id": None,
            }
        finally:
            await pool.close()

    async def test_quiet_checks_are_not_counted_by_the_fire_limits(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            _schedules, fires = _collections(pool)
            agent = _new_uuid()
            conv, sid = await _seed_schedule(pool, agent_id=agent)
            for _ in range(3):
                fire_id = _new_uuid()
                await fires.create_dispatching(
                    fire_id=fire_id,
                    schedule_id=sid,
                    webhook_subscription_id=None,
                    conversation_id=conv,
                    scheduled_fire_at=None,
                    actual_fired_at=datetime.now(UTC) - timedelta(minutes=1),
                    fire_source="scheduled_tick",
                    execution_mode="spawn",
                )
                await fires.finalize_success(conv, fire_id, status="checked_quiet")
            written = await pool.fetch("SELECT status FROM wake_fires WHERE schedule_id = $1", sid)
            assert [r["status"] for r in written] == ["checked_quiet"] * 3, "the quiet rows are really there"

            async def one_a_day(_trigger: WakeTrigger) -> FireLimits | None:
                return FireLimits(per_wake=1, per_agent=1)

            handler = _RecordingHandler()
            result = await dispatch_wake(
                WakeTrigger(
                    schedule_id=sid,
                    user_id=_new_uuid(),
                    agent_id=agent,
                    conversation_id=conv,
                    fire_source="scheduled_tick",
                    execution_mode="spawn",
                    schedule_type="interval",
                    fired_at=datetime.now(UTC),
                ),
                _new_uuid(),
                pool,
                handler=handler,
                permit=one_a_day,
            )
            assert result.status == "fired", "three quiet checks spent nothing of a one-a-day limit"
        finally:
            await pool.close()


class TestAQuietCheckBeforeTheFire:
    """A consumer whose check must start nothing runs it in the dispatch callback, before ``dispatch_wake``."""

    async def test_the_callback_returns_checked_quiet_and_no_conversation_is_started(
        self, pg_schema: tuple[str, str]
    ) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            _conv, sid = await _seed_schedule(
                pool, agent_id=_new_uuid(), next_fire_at=datetime.now(UTC) - timedelta(seconds=5)
            )

            async def dispatch(trigger: WakeTrigger, fire_id: UUID, pool_: object) -> WakeDispatchResult:
                # the check found nothing: the fire ends here, before dispatch_wake could start a conversation
                return WakeDispatchResult(status="checked_quiet", output_text="nothing new", latency_ms=1)

            await wake_tick_job(pool, None, dispatch, **_tick_collections(pool))
            row = await pool.fetchrow(
                "SELECT status, started_conversation_id, output_text FROM wake_fires WHERE schedule_id = $1", sid
            )
            assert dict(row) == {
                "status": "checked_quiet",
                "started_conversation_id": None,
                "output_text": "nothing new",
            }
        finally:
            await pool.close()


class TestContextFromSkipsQuietChecks:
    async def test_a_downstream_wake_reads_the_last_real_fire_past_later_quiet_checks(
        self, pg_schema: tuple[str, str]
    ) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            _schedules, fires = _collections(pool)
            agent = _new_uuid()
            up_conv, upstream = await _seed_schedule(pool, agent_id=agent)
            down_conv, downstream = await _seed_schedule(pool, agent_id=agent)
            for minutes_ago, status, text in (
                (10, "fired", "found three new messages"),
                (5, "checked_quiet", "nothing new"),
            ):
                fire_id = _new_uuid()
                await fires.create_dispatching(
                    fire_id=fire_id,
                    schedule_id=upstream,
                    webhook_subscription_id=None,
                    conversation_id=up_conv,
                    scheduled_fire_at=None,
                    actual_fired_at=datetime.now(UTC) - timedelta(minutes=minutes_ago),
                    fire_source="scheduled_tick",
                    execution_mode="spawn",
                )
                await fires.finalize_success(up_conv, fire_id, status=status, output_text=text)

            handler = _RecordingHandler()
            await dispatch_wake(
                WakeTrigger(
                    schedule_id=downstream,
                    user_id=_new_uuid(),
                    agent_id=agent,
                    conversation_id=down_conv,
                    fire_source="scheduled_tick",
                    execution_mode="spawn",
                    schedule_type="interval",
                    fired_at=datetime.now(UTC),
                    context_from_schedule_id=upstream,
                ),
                _new_uuid(),
                pool,
                handler=handler,
            )
            blocks = handler.prepared[0].context_blocks
            assert len(blocks) == 1 and "found three new messages" in str(blocks[0])
        finally:
            await pool.close()


class TestFireConversationLink:
    async def test_the_hook_links_the_fire_before_the_handler_runs(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            _conv, sid = await _seed_schedule(
                pool, agent_id=_new_uuid(), next_fire_at=datetime.now(UTC) - timedelta(seconds=5)
            )
            linked_when_handled: list[UUID | None] = []

            class _Handler(_RecordingHandler):
                async def __call__(
                    self, trigger: WakeTrigger, prepared_context: PreparedWakeContext, pool_: Any
                ) -> HandlerCallbackResult:
                    linked_when_handled.append(
                        await pool_.fetchval(
                            "SELECT started_conversation_id FROM wake_fires WHERE fire_id = $1", trigger.fire_id
                        )
                    )
                    return await super().__call__(trigger, prepared_context, pool_)

            async def start(trigger: WakeTrigger, conn: Any) -> UUID:
                new_id = _new_uuid()
                await conn.execute(
                    "INSERT INTO fire_conversations (conversation_id, fire_id) VALUES ($1, $2)", new_id, trigger.fire_id
                )
                return new_id

            handler = _Handler()

            async def dispatch(trigger: WakeTrigger, fire_id: UUID, pool_: object) -> WakeDispatchResult:
                return await dispatch_wake(trigger, fire_id, pool_, handler=handler, start_conversation=start)

            await wake_tick_job(pool, None, dispatch, **_tick_collections(pool))
            fire = await pool.fetchrow(
                "SELECT fire_id, status, started_conversation_id FROM wake_fires WHERE schedule_id = $1", sid
            )
            assert fire is not None and fire["status"] == "fired"
            made = await pool.fetchrow("SELECT conversation_id, fire_id FROM fire_conversations")
            assert made is not None
            assert fire["started_conversation_id"] == made["conversation_id"]
            assert made["fire_id"] == fire["fire_id"]
            assert handler.triggers[0].started_conversation_id == made["conversation_id"]
            assert linked_when_handled == [made["conversation_id"]]
        finally:
            await pool.close()

    @pytest.mark.parametrize(("skill", "converses"), [("tool", False), ("prose", True)])
    async def test_a_tool_only_skill_fires_with_no_conversation_and_a_prose_one_with_one(
        self, pg_schema: tuple[str, str], skill: str, converses: bool
    ) -> None:
        """A skill that only calls a tool has no model to converse with, so its fire starts no conversation."""
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            agent_id = _new_uuid()
            skill_id = _new_uuid()
            if skill == "tool":
                await pool.execute(
                    "INSERT INTO agent_skills (agent_id, skill_id, user_id, name, summary, tool, arguments) "
                    "VALUES ($1, $2, $3, 'clock', 'the time', 'threetears.current_time', '{}'::jsonb)",
                    agent_id,
                    skill_id,
                    _new_uuid(),
                )
            else:
                await pool.execute(
                    "INSERT INTO agent_skills (agent_id, skill_id, user_id, name, summary, body) "
                    "VALUES ($1, $2, $3, 'muse', 'think', 'Think about the sea.')",
                    agent_id,
                    skill_id,
                    _new_uuid(),
                )
            _conv, sid = await _seed_schedule(
                pool, agent_id=agent_id, next_fire_at=datetime.now(UTC) - timedelta(seconds=5)
            )
            await pool.execute("UPDATE agent_wake_schedules SET skill_id = $2 WHERE schedule_id = $1", sid, skill_id)
            handler = _RecordingHandler()
            started: list[UUID] = []

            async def start(trigger: WakeTrigger, conn: Any) -> UUID:
                new_id = _new_uuid()
                started.append(new_id)
                await conn.execute(
                    "INSERT INTO fire_conversations (conversation_id, fire_id) VALUES ($1, $2)", new_id, trigger.fire_id
                )
                return new_id

            async def dispatch(trigger: WakeTrigger, fire_id: UUID, pool_: object) -> WakeDispatchResult:
                return await dispatch_wake(trigger, fire_id, pool_, handler=handler, start_conversation=start)

            await wake_tick_job(pool, None, dispatch, **_tick_collections(pool))

            [prepared] = handler.prepared
            assert prepared.attached_skill is not None and prepared.attached_skill.skill_id == skill_id
            assert is_tool_only(prepared.attached_skill) is not converses
            assert bool(started) is converses
            assert (handler.triggers[0].started_conversation_id is not None) is converses
        finally:
            await pool.close()

    async def test_a_fire_that_fails_after_starting_its_conversation_stays_linked(
        self, pg_schema: tuple[str, str]
    ) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            _conv, sid = await _seed_schedule(
                pool, agent_id=_new_uuid(), next_fire_at=datetime.now(UTC) - timedelta(seconds=5)
            )

            class _Boom(_RecordingHandler):
                async def __call__(
                    self, trigger: WakeTrigger, prepared_context: PreparedWakeContext, pool_: Any
                ) -> HandlerCallbackResult:
                    raise RuntimeError("model unavailable")

            async def start(trigger: WakeTrigger, conn: Any) -> UUID:
                new_id = _new_uuid()
                await conn.execute(
                    "INSERT INTO fire_conversations (conversation_id, fire_id) VALUES ($1, $2)", new_id, trigger.fire_id
                )
                return new_id

            async def dispatch(trigger: WakeTrigger, fire_id: UUID, pool_: object) -> WakeDispatchResult:
                return await dispatch_wake(trigger, fire_id, pool_, handler=_Boom(), start_conversation=start)

            await wake_tick_job(pool, None, dispatch, **_tick_collections(pool))
            fire = await pool.fetchrow(
                "SELECT status, started_conversation_id FROM wake_fires WHERE schedule_id = $1", sid
            )
            assert fire is not None
            assert fire["status"] == "failed"
            assert fire["started_conversation_id"] == await pool.fetchval(
                "SELECT conversation_id FROM fire_conversations"
            )
        finally:
            await pool.close()

    async def test_a_hook_that_raises_leaves_no_link_and_no_conversation(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            _conv, sid = await _seed_schedule(
                pool, agent_id=_new_uuid(), next_fire_at=datetime.now(UTC) - timedelta(seconds=5)
            )

            async def start(trigger: WakeTrigger, conn: Any) -> UUID:
                await conn.execute(
                    "INSERT INTO fire_conversations (conversation_id, fire_id) VALUES ($1, $2)",
                    _new_uuid(),
                    trigger.fire_id,
                )
                raise RuntimeError("could not finish making the conversation")

            handler = _RecordingHandler()

            async def dispatch(trigger: WakeTrigger, fire_id: UUID, pool_: object) -> WakeDispatchResult:
                return await dispatch_wake(trigger, fire_id, pool_, handler=handler, start_conversation=start)

            await wake_tick_job(pool, None, dispatch, **_tick_collections(pool))
            fire = await pool.fetchrow(
                "SELECT status, started_conversation_id FROM wake_fires WHERE schedule_id = $1", sid
            )
            assert fire is not None
            assert fire["status"] == "failed"
            assert fire["started_conversation_id"] is None
            assert await pool.fetchval("SELECT count(*) FROM fire_conversations") == 0
            assert handler.triggers == []
        finally:
            await pool.close()


class TestReaper:
    async def _dispatching_fire(self, pool: asyncpg.Pool, *, started: UUID | None, age: timedelta) -> tuple[UUID, UUID]:
        _schedules, fires = _collections(pool)
        conv, sid = await _seed_schedule(pool, agent_id=_new_uuid())
        fire_id = _new_uuid()
        await fires.create_dispatching(
            fire_id=fire_id,
            schedule_id=sid,
            webhook_subscription_id=None,
            conversation_id=conv,
            scheduled_fire_at=None,
            actual_fired_at=datetime.now(UTC) - age,
            fire_source="scheduled_tick",
            execution_mode="spawn",
        )
        if started is not None:
            async with pool.acquire() as conn, conn.transaction():
                assert await fires.link_started_conversation(conv, fire_id, started, conn=conn)
        return conv, fire_id

    async def test_the_reaper_hands_each_fires_started_conversation_to_the_hook(
        self, pg_schema: tuple[str, str]
    ) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            started = _new_uuid()
            conv, lost = await self._dispatching_fire(pool, started=started, age=timedelta(hours=2))
            _conv2, lost_early = await self._dispatching_fire(pool, started=None, age=timedelta(hours=2))
            _conv3, live = await self._dispatching_fire(pool, started=_new_uuid(), age=timedelta(seconds=1))
            heard: list[ReapedFire] = []

            async def on_reaped(reaped: Any) -> None:
                heard.extend(reaped)

            async def dispatch(_trigger: WakeTrigger, _fire_id: UUID, _pool: object) -> WakeDispatchResult:
                raise AssertionError("nothing is due")

            await wake_tick_job(pool, None, dispatch, on_reaped=on_reaped, **_tick_collections(pool))
            by_fire = {r.fire_id: r for r in heard}
            assert set(by_fire) == {lost, lost_early}
            assert by_fire[lost] == ReapedFire(conversation_id=conv, fire_id=lost, started_conversation_id=started)
            assert by_fire[lost_early].started_conversation_id is None
            assert (await _fire_row(pool, live))["status"] == "dispatching"
        finally:
            await pool.close()

    async def test_a_hook_that_raises_does_not_undo_the_reap(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            _conv, lost = await self._dispatching_fire(pool, started=_new_uuid(), age=timedelta(hours=2))

            async def on_reaped(_reaped: Any) -> None:
                raise RuntimeError("consumer down")

            async def dispatch(_trigger: WakeTrigger, _fire_id: UUID, _pool: object) -> WakeDispatchResult:
                raise AssertionError("nothing is due")

            await wake_tick_job(pool, None, dispatch, on_reaped=on_reaped, **_tick_collections(pool))
            assert (await _fire_row(pool, lost))["status"] == "failed"
        finally:
            await pool.close()

    async def test_a_late_finish_cannot_flip_a_reaped_fire(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            _schedules, fires = _collections(pool)
            conv, lost = await self._dispatching_fire(pool, started=_new_uuid(), age=timedelta(hours=2))
            reaped = await fires.reap_stale_dispatching(datetime.now(UTC), older_than=timedelta(minutes=15))
            assert [r.fire_id for r in reaped] == [lost]
            await fires.finalize_success(conv, lost, status="fired", output_text="finished late")
            await fires.finalize_failed(conv, lost, error="also late")
            row = await _fire_row(pool, lost)
            assert row["status"] == "failed"
            assert row["error"] == REAPED_DISPATCH_ERROR
            assert row["output_text"] is None
            # and a reaped fire cannot be linked afterwards either
            async with pool.acquire() as conn, conn.transaction():
                assert not await fires.link_started_conversation(conv, lost, _new_uuid(), conn=conn)
        finally:
            await pool.close()
