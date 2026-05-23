"""End-to-end integration tests for :func:`dispatch_wake`.

Exercises the DB-touching paths against a real Postgres testcontainer:

- ``context_from`` chain resolution -- upstream successful fire output
  is materialised as a labeled block in ``PreparedWakeContext.context_blocks``.
- ``context_from`` with no successful upstream fire -- no block; no
  raise; warning logged.
- attached-skill resolution -- enabled skill returned;
  disabled / missing skill resolves to ``None``.
- end-to-end: seed a schedule + skill, call ``dispatch_wake`` with
  a stub handler, assert the handler receives the expected prepared
  context.

Mirrors the existing wake integration patterns
(``test_wake_tick_loop.py``): canonical ``db_container`` fixture,
per-test ``pg_schema``, ``AsyncpgStore`` wrapper around the migration
runner.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import asyncpg
import pytest
from uuid_utils import uuid7

from threetears.agent.skills.migrations import register as register_skills
from threetears.agent.wake.dispatch import dispatch_wake
from threetears.agent.wake.migrations import register as register_wake
from threetears.agent.wake.types import (
    HandlerCallback,
    HandlerCallbackResult,
    PreparedWakeContext,
    WakeTrigger,
)
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


async def _seed_skill(
    pool: asyncpg.Pool,
    *,
    agent_id: UUID,
    user_id: UUID,
    name: str = "diagnostic-watchdog",
    enabled: bool = True,
) -> UUID:
    skill_id = _new_uuid()
    await pool.execute(
        "INSERT INTO agent_skills "
        "(agent_id, skill_id, user_id, name, summary, body, prompt_mode, "
        " tool_additions, tool_restrictions, trigger_keywords, tags, source, enabled) "
        "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)",
        agent_id,
        skill_id,
        user_id,
        name,
        "summary",
        "do the thing",
        "additive",
        [],
        [],
        "",
        [],
        "manual",
        enabled,
    )
    return skill_id


async def _seed_schedule(
    pool: asyncpg.Pool,
    *,
    conversation_id: UUID,
    agent_id: UUID,
    user_id: UUID,
    schedule_id: UUID | None = None,
    name: str | None = None,
    skill_id: UUID | None = None,
    context_from_schedule_id: UUID | None = None,
    delivery_target: str = "conversation",
) -> UUID:
    sched_id = schedule_id or _new_uuid()
    await pool.execute(
        "INSERT INTO agent_wake_schedules "
        "(conversation_id, schedule_id, user_id, agent_id, skill_id, schedule_type, "
        " schedule_config, execution_mode, status, next_fire_at, missed_fire_policy, "
        " name, context_from_schedule_id, delivery_target, delivery_config) "
        "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)",
        conversation_id,
        sched_id,
        user_id,
        agent_id,
        skill_id,
        "interval",
        {"seconds": 60},
        "inline",
        "active",
        datetime.now(UTC) + timedelta(minutes=5),
        "coalesce",
        name,
        context_from_schedule_id,
        delivery_target,
        {},
    )
    return sched_id


async def _seed_fire(
    pool: asyncpg.Pool,
    *,
    conversation_id: UUID,
    schedule_id: UUID,
    status: str,
    output_text: str | None,
    fired_at: datetime,
) -> UUID:
    fire_id = _new_uuid()
    await pool.execute(
        "INSERT INTO wake_fires "
        "(conversation_id, fire_id, schedule_id, scheduled_fire_at, actual_fired_at, "
        " status, display_suppressed, output_text) "
        "VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
        conversation_id,
        fire_id,
        schedule_id,
        fired_at,
        fired_at,
        status,
        False,
        output_text,
    )
    return fire_id


def _make_trigger(
    *,
    conversation_id: UUID,
    agent_id: UUID,
    user_id: UUID,
    schedule_id: UUID,
    skill_id: UUID | None = None,
    context_from_schedule_id: UUID | None = None,
    schedule_name: str | None = None,
    delivery_target: str = "conversation",
) -> WakeTrigger:
    return WakeTrigger(
        schedule_id=schedule_id,
        user_id=user_id,
        agent_id=agent_id,
        conversation_id=conversation_id,
        fire_source="scheduled_tick",
        execution_mode="inline",
        schedule_type="interval",
        fired_at=datetime.now(UTC),
        schedule_name=schedule_name,
        skill_id=skill_id,
        context_from_schedule_id=context_from_schedule_id,
        delivery_target=delivery_target,
    )


# parity-with: threetears.agent.wake.types.HandlerCallback
class _CapturingHandler(HandlerCallback):
    """Captures the ``PreparedWakeContext`` for assertions; returns a default result."""

    def __init__(self) -> None:
        self.received: PreparedWakeContext | None = None

    async def __call__(
        self,
        trigger: WakeTrigger,
        prepared_context: PreparedWakeContext,
        pool: Any,
    ) -> HandlerCallbackResult:
        del pool
        self.received = prepared_context
        return HandlerCallbackResult(
            status="fired",
            assistant_message_content="ok",
            target_conversation_id=trigger.conversation_id,
        )


class TestContextFromResolutionIntegration:
    """``context_from`` reads the upstream schedule's most recent successful fire."""

    async def test_upstream_success_produces_labeled_block(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            conv = _new_uuid()
            agent = _new_uuid()
            user = _new_uuid()
            upstream_sched = await _seed_schedule(
                pool,
                conversation_id=conv,
                agent_id=agent,
                user_id=user,
                name="upstream-check",
            )
            now = datetime.now(UTC)
            await _seed_fire(
                pool,
                conversation_id=conv,
                schedule_id=upstream_sched,
                status="fired",
                output_text="3 anomalies observed at 09:00 UTC",
                fired_at=now - timedelta(minutes=10),
            )
            downstream_sched = await _seed_schedule(
                pool,
                conversation_id=conv,
                agent_id=agent,
                user_id=user,
                context_from_schedule_id=upstream_sched,
                name="downstream-followup",
            )
            handler = _CapturingHandler()
            trigger = _make_trigger(
                conversation_id=conv,
                agent_id=agent,
                user_id=user,
                schedule_id=downstream_sched,
                context_from_schedule_id=upstream_sched,
            )
            await dispatch_wake(
                trigger,
                _new_uuid(),
                pool=pool,
                handler=handler,
            )
            assert handler.received is not None
            blocks = handler.received.context_blocks
            assert len(blocks) == 1
            block = blocks[0]
            assert "upstream-check" in block
            assert "3 anomalies observed at 09:00 UTC" in block
            assert block.endswith("---")
        finally:
            await pool.close()

    async def test_no_upstream_fire_yields_empty_blocks(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            conv = _new_uuid()
            agent = _new_uuid()
            user = _new_uuid()
            upstream_sched = await _seed_schedule(
                pool,
                conversation_id=conv,
                agent_id=agent,
                user_id=user,
                name="upstream-no-fires-yet",
            )
            downstream_sched = await _seed_schedule(
                pool,
                conversation_id=conv,
                agent_id=agent,
                user_id=user,
                context_from_schedule_id=upstream_sched,
            )
            handler = _CapturingHandler()
            trigger = _make_trigger(
                conversation_id=conv,
                agent_id=agent,
                user_id=user,
                schedule_id=downstream_sched,
                context_from_schedule_id=upstream_sched,
            )
            await dispatch_wake(
                trigger,
                _new_uuid(),
                pool=pool,
                handler=handler,
            )
            assert handler.received is not None
            assert handler.received.context_blocks == ()
        finally:
            await pool.close()

    async def test_upstream_failed_status_not_used(self, pg_schema: tuple[str, str]) -> None:
        """A ``status='failed'`` upstream fire MUST NOT feed the chain."""
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            conv = _new_uuid()
            agent = _new_uuid()
            user = _new_uuid()
            upstream_sched = await _seed_schedule(
                pool,
                conversation_id=conv,
                agent_id=agent,
                user_id=user,
            )
            now = datetime.now(UTC)
            await _seed_fire(
                pool,
                conversation_id=conv,
                schedule_id=upstream_sched,
                status="failed",
                output_text="ignore this -- the fire failed",
                fired_at=now - timedelta(minutes=10),
            )
            downstream_sched = await _seed_schedule(
                pool,
                conversation_id=conv,
                agent_id=agent,
                user_id=user,
                context_from_schedule_id=upstream_sched,
            )
            handler = _CapturingHandler()
            trigger = _make_trigger(
                conversation_id=conv,
                agent_id=agent,
                user_id=user,
                schedule_id=downstream_sched,
                context_from_schedule_id=upstream_sched,
            )
            await dispatch_wake(
                trigger,
                _new_uuid(),
                pool=pool,
                handler=handler,
            )
            assert handler.received is not None
            assert handler.received.context_blocks == ()
        finally:
            await pool.close()


class TestAttachedSkillResolutionIntegration:
    """Skill resolution returns the row when enabled; ``None`` when missing / disabled."""

    async def test_enabled_skill_returned(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            conv = _new_uuid()
            agent = _new_uuid()
            user = _new_uuid()
            skill_id = await _seed_skill(
                pool,
                agent_id=agent,
                user_id=user,
                name="prod-investigation",
                enabled=True,
            )
            sched = await _seed_schedule(
                pool,
                conversation_id=conv,
                agent_id=agent,
                user_id=user,
                skill_id=skill_id,
            )
            handler = _CapturingHandler()
            trigger = _make_trigger(
                conversation_id=conv,
                agent_id=agent,
                user_id=user,
                schedule_id=sched,
                skill_id=skill_id,
            )
            await dispatch_wake(
                trigger,
                _new_uuid(),
                pool=pool,
                handler=handler,
            )
            assert handler.received is not None
            attached = handler.received.attached_skill
            assert attached is not None
            assert attached.skill_id == skill_id
            assert attached.name == "prod-investigation"
            assert attached.enabled is True
        finally:
            await pool.close()

    async def test_disabled_skill_resolves_to_none(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            conv = _new_uuid()
            agent = _new_uuid()
            user = _new_uuid()
            skill_id = await _seed_skill(
                pool,
                agent_id=agent,
                user_id=user,
                enabled=False,
            )
            sched = await _seed_schedule(
                pool,
                conversation_id=conv,
                agent_id=agent,
                user_id=user,
                skill_id=skill_id,
            )
            handler = _CapturingHandler()
            trigger = _make_trigger(
                conversation_id=conv,
                agent_id=agent,
                user_id=user,
                schedule_id=sched,
                skill_id=skill_id,
            )
            await dispatch_wake(
                trigger,
                _new_uuid(),
                pool=pool,
                handler=handler,
            )
            assert handler.received is not None
            assert handler.received.attached_skill is None
        finally:
            await pool.close()

    async def test_missing_skill_id_resolves_to_none(self, pg_schema: tuple[str, str]) -> None:
        """``skill_id`` referencing a non-existent row resolves to ``None`` + warns."""
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            conv = _new_uuid()
            agent = _new_uuid()
            user = _new_uuid()
            ghost_skill = _new_uuid()
            sched = await _seed_schedule(
                pool,
                conversation_id=conv,
                agent_id=agent,
                user_id=user,
                # NOT writing the FK target -- defensive case
                skill_id=None,
            )
            handler = _CapturingHandler()
            trigger = _make_trigger(
                conversation_id=conv,
                agent_id=agent,
                user_id=user,
                schedule_id=sched,
                skill_id=ghost_skill,
            )
            await dispatch_wake(
                trigger,
                _new_uuid(),
                pool=pool,
                handler=handler,
            )
            assert handler.received is not None
            assert handler.received.attached_skill is None
        finally:
            await pool.close()


class TestEndToEndHappyPath:
    """A full schedule + skill + handler round-trip lands the right context."""

    async def test_handler_receives_skill_and_context(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            conv = _new_uuid()
            agent = _new_uuid()
            user = _new_uuid()
            skill_id = await _seed_skill(
                pool,
                agent_id=agent,
                user_id=user,
                name="daily-summary",
            )
            upstream_sched = await _seed_schedule(
                pool,
                conversation_id=conv,
                agent_id=agent,
                user_id=user,
                name="loki-canary",
            )
            now = datetime.now(UTC)
            await _seed_fire(
                pool,
                conversation_id=conv,
                schedule_id=upstream_sched,
                status="fired",
                output_text="canary green; no 500s in last hour",
                fired_at=now - timedelta(minutes=2),
            )
            downstream_sched = await _seed_schedule(
                pool,
                conversation_id=conv,
                agent_id=agent,
                user_id=user,
                skill_id=skill_id,
                context_from_schedule_id=upstream_sched,
                name="hourly-summary",
            )
            handler = _CapturingHandler()
            trigger = _make_trigger(
                conversation_id=conv,
                agent_id=agent,
                user_id=user,
                schedule_id=downstream_sched,
                skill_id=skill_id,
                context_from_schedule_id=upstream_sched,
                schedule_name="hourly-summary",
            )
            result = await dispatch_wake(
                trigger,
                _new_uuid(),
                pool=pool,
                handler=handler,
            )
            assert result.status == "fired"
            assert handler.received is not None
            prepared = handler.received
            assert prepared.attached_skill is not None
            assert prepared.attached_skill.skill_id == skill_id
            assert prepared.attached_skill.name == "daily-summary"
            assert len(prepared.context_blocks) == 1
            assert "loki-canary" in prepared.context_blocks[0]
            assert "canary green" in prepared.context_blocks[0]
        finally:
            await pool.close()
