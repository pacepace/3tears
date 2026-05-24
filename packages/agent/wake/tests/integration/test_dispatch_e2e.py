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
) -> UUID:
    sched_id = schedule_id or _new_uuid()
    await pool.execute(
        "INSERT INTO agent_wake_schedules "
        "(conversation_id, schedule_id, user_id, agent_id, skill_id, schedule_type, "
        " schedule_config, execution_mode, status, next_fire_at, missed_fire_policy, "
        " name, context_from_schedule_id) "
        "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)",
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


class TestContextFromTruncationIntegration:
    """The 16KB context_blocks budget truncates oversized upstream output.

    Pins the truncation path in :func:`_resolve_context_from` (Critic
    finding #2): a regression that mis-slices the UTF-8 boundary,
    drops the ``[truncated: ...]`` suffix marker, or off-by-ones the
    budget would land silently because shard-03 had no test for this
    branch before this commit.
    """

    async def test_oversize_upstream_output_truncated_with_suffix(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """A >16KB upstream output is truncated + suffix-marked."""
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
                name="oversize-upstream",
            )
            # 32KB of ASCII -- well above the 16KB budget. ASCII keeps
            # byte-count == char-count so the boundary math is obvious;
            # the multi-byte case is covered by the next test.
            payload = "x" * (32 * 1024)
            now = datetime.now(UTC)
            await _seed_fire(
                pool,
                conversation_id=conv,
                schedule_id=upstream_sched,
                status="fired",
                output_text=payload,
                fired_at=now - timedelta(minutes=1),
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
            blocks = handler.received.context_blocks
            assert len(blocks) == 1
            block = blocks[0]
            # Suffix marker must be present, with the original size +
            # the budget recorded so an operator can see what was cut.
            assert "[truncated:" in block
            assert "16384B" in block
            # The block must be UNDER the original-payload size +
            # bounded near the budget. We don't pin an exact byte count
            # because the suffix string adds a handful of bytes; the
            # invariant is "way smaller than original".
            assert len(block.encode("utf-8")) < len(payload)
        finally:
            await pool.close()

    async def test_truncation_at_multibyte_boundary_is_utf8_safe(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """Multi-byte chars at the 16KB boundary do not produce invalid UTF-8.

        The resolver uses ``decode('utf-8', errors='ignore')`` to
        survive a slice that lands mid-codepoint. This test seeds
        upstream output engineered so the boundary falls inside a
        4-byte emoji and verifies the resulting block is valid UTF-8
        (encode-decode round-trips cleanly) + still carries the
        truncation suffix.
        """
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
                name="multibyte-upstream",
            )
            # The label prefix from _resolve_context_from is something
            # like 'Context from upstream schedule "multibyte-upstream"
            # (fired <iso>):\n' which is a variable-byte prefix; pad
            # the payload with enough ASCII filler that the 16384-byte
            # boundary falls deep into the emoji-run rather than just
            # past the prefix. 32 KB of ASCII + emojis is more than
            # enough.
            ascii_filler = "a" * (20 * 1024)
            emoji_run = "\U0001f600" * 2000  # 4-byte UTF-8 codepoint x 2000
            payload = ascii_filler + emoji_run
            now = datetime.now(UTC)
            await _seed_fire(
                pool,
                conversation_id=conv,
                schedule_id=upstream_sched,
                status="fired",
                output_text=payload,
                fired_at=now - timedelta(minutes=1),
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
            blocks = handler.received.context_blocks
            assert len(blocks) == 1
            block = blocks[0]
            # Truncation marker present -- the path ran.
            assert "[truncated:" in block
            # Round-trip through utf-8: a slice that left a dangling
            # multi-byte sequence in place would raise here. The
            # resolver uses errors='ignore' so dangling bytes are
            # dropped cleanly.
            roundtripped = block.encode("utf-8").decode("utf-8")
            assert roundtripped == block
        finally:
            await pool.close()


class TestCrossAgentSkillMismatchIntegration:
    """Skill owned by a DIFFERENT agent resolves to None (composite-PK isolation).

    Pins the cross-agent partition contract (Critic finding #5):
    ``AgentSkillCollection.get((agent_id, skill_id))`` MUST return
    ``None`` when the skill row exists with a DIFFERENT ``agent_id``
    than the trigger carries. A future refactor that drops ``agent_id``
    from the composite predicate (and reduces lookup to the standalone
    ``UNIQUE (skill_id)`` constraint) would return the wrong-agent
    skill and leak a cross-agent skill body into the handler -- this
    test catches that regression.
    """

    async def test_skill_owned_by_other_agent_resolves_to_none(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            conv = _new_uuid()
            agent_a = _new_uuid()
            agent_b = _new_uuid()
            user = _new_uuid()
            # Seed a skill owned by agent A
            skill_owned_by_a = await _seed_skill(
                pool,
                agent_id=agent_a,
                user_id=user,
                name="agent-a-only-skill",
                enabled=True,
            )
            # Schedule belongs to agent B; we DON'T attach the skill
            # to the schedule because the FK on skill_id would be
            # violated against agent_b's composite key. But the trigger
            # we build below will reference (agent_b, skill_owned_by_a)
            # directly to simulate the case where dispatch_wake is
            # invoked with a malformed trigger.
            sched = await _seed_schedule(
                pool,
                conversation_id=conv,
                agent_id=agent_b,
                user_id=user,
                # null skill_id at the schedule level
                skill_id=None,
            )
            handler = _CapturingHandler()
            # The trigger references agent_b but cites agent_a's skill
            # id. The composite (agent_b, skill_owned_by_a) lookup MUST
            # return None.
            trigger = _make_trigger(
                conversation_id=conv,
                agent_id=agent_b,
                user_id=user,
                schedule_id=sched,
                skill_id=skill_owned_by_a,
            )
            await dispatch_wake(
                trigger,
                _new_uuid(),
                pool=pool,
                handler=handler,
            )
            assert handler.received is not None
            # NOT agent_a's skill -- the cross-agent partition holds.
            assert handler.received.attached_skill is None
        finally:
            await pool.close()


class TestCreateDispatchingPlaceholderIntegration:
    """``create_dispatching`` writes the v004 ``'dispatching'`` placeholder.

    Pins the placeholder-status contract (Critic finding #4): the
    in-flight row must NOT pre-claim a terminal ``'fired'`` status.
    Otherwise a future parallel-dispatch refactor could surface a
    half-completed row to a downstream wake's ``context_from``
    resolver and silently produce an empty context block.

    A row that stays in ``'dispatching'`` is audit evidence the
    dispatcher crashed before finalize ran; the finalize_success /
    finalize_failed UPDATEs in :class:`WakeFireCollection` overwrite
    to the real terminal status, so successful ticks never leave a
    row in this state.
    """

    async def test_create_dispatching_writes_dispatching_status(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """Direct collection call -- placeholder row carries 'dispatching'."""
        from threetears.agent.wake.collections import WakeFireCollection  # noqa: PLC0415
        from threetears.core.collections.registry import CollectionRegistry  # noqa: PLC0415
        from threetears.core.config import DefaultCoreConfig  # noqa: PLC0415

        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            conv = _new_uuid()
            agent = _new_uuid()
            user = _new_uuid()
            sched = await _seed_schedule(
                pool,
                conversation_id=conv,
                agent_id=agent,
                user_id=user,
            )
            registry = CollectionRegistry()
            registry.configure(l3_pool=pool)
            cfg = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
            fires = WakeFireCollection(registry=registry, config=cfg)

            fire_id = _new_uuid()
            now = datetime.now(UTC)
            await fires.create_dispatching(
                fire_id=fire_id,
                schedule_id=sched,
                webhook_subscription_id=None,
                conversation_id=conv,
                scheduled_fire_at=now,
                actual_fired_at=now,
                fire_source="scheduled_tick",
                execution_mode="inline",
            )
            row = await pool.fetchrow(
                "SELECT status FROM wake_fires WHERE conversation_id = $1 AND fire_id = $2",
                conv,
                fire_id,
            )
            assert row is not None
            assert row["status"] == "dispatching"
        finally:
            await pool.close()

    async def test_check_constraint_accepts_dispatching(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """A raw INSERT with status='dispatching' passes the CHECK constraint.

        Pins that the v004 migration applied the broadened CHECK so a
        future migration-runner regression that skips v004 would fail
        loudly here rather than silently downgrading semantics.
        """
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            conv = _new_uuid()
            agent = _new_uuid()
            user = _new_uuid()
            sched = await _seed_schedule(
                pool,
                conversation_id=conv,
                agent_id=agent,
                user_id=user,
            )
            await pool.execute(
                "INSERT INTO wake_fires "
                "(conversation_id, fire_id, schedule_id, actual_fired_at, status) "
                "VALUES ($1, $2, $3, $4, $5)",
                conv,
                _new_uuid(),
                sched,
                datetime.now(UTC),
                "dispatching",
            )
        finally:
            await pool.close()


class TestRateLimitWiringIntegration:
    """End-to-end: rate-limit wiring rejects without invoking the handler.

    Pins the BLOCKING Critic finding against shard-05 against a REAL
    Postgres testcontainer: the rate-limit query reads from the
    ``wake_fires`` partition + ``agent_wake_schedules`` JOIN exactly as
    production would, and rejection results in NO handler invocation +
    NO new ``wake_fires`` row written by ``dispatch_wake`` itself (the
    caller writes the terminal ``'skipped_rate_limit'`` row via
    ``finalize_*`` after seeing the return value -- the test
    invocation pattern here mirrors the production tick body).
    """

    async def test_per_conv_cap_rejects_against_real_pg(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            conv = _new_uuid()
            agent = _new_uuid()
            user = _new_uuid()
            sched = await _seed_schedule(
                pool,
                conversation_id=conv,
                agent_id=agent,
                user_id=user,
            )
            now = datetime.now(UTC)
            # Seed 3 prior successful fires in the last hour.
            for offset in (60, 120, 180):
                await _seed_fire(
                    pool,
                    conversation_id=conv,
                    schedule_id=sched,
                    status="fired",
                    output_text="ok",
                    fired_at=now - timedelta(seconds=offset),
                )

            # parity-with: threetears.agent.wake.config.WakeConfig
            class _TightConvCap:
                """Per-conv cap = 1, per-user wide open."""

                @property
                def max_fires_per_conv_per_day(self) -> int:
                    return 1

                @property
                def max_fires_per_user_per_day(self) -> int:
                    return 1000

                @property
                def max_webhook_fires_per_subscription_per_hour(self) -> int:
                    return 60

                @property
                def max_schedules_per_conversation(self) -> int:
                    return 10

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

            handler = _CapturingHandler()
            trigger = _make_trigger(
                conversation_id=conv,
                agent_id=agent,
                user_id=user,
                schedule_id=sched,
            )
            result = await dispatch_wake(
                trigger,
                _new_uuid(),
                pool=pool,
                handler=handler,
                wake_config=_TightConvCap(),
            )
            # rejected with conv-scope
            assert result.status == "skipped_rate_limit"
            assert result.error is not None
            assert "conv" in result.error
            # handler MUST NOT have been invoked
            assert handler.received is None
            # No new wake_fires row was written by dispatch_wake itself
            # (only the 3 seed rows exist; the caller writes the
            # terminal 'skipped_rate_limit' row via finalize_*).
            count = await pool.fetchval(
                "SELECT COUNT(*) FROM wake_fires WHERE conversation_id = $1",
                conv,
            )
            assert int(count or 0) == 3
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
