"""Integration test: ``WakeScheduleCollection.claim_and_reschedule`` is
atomic across concurrent ticks.

Verifies:

- a successful claim returns True and advances ``next_fire_at`` +
  ``last_fired_at`` + ``status``.
- a stale ``expected_next_fire`` (set by a concurrent claimant)
  returns False; the row is untouched by the loser.
- two concurrent claims against the same schedule have exactly one
  winner -- no double-fire.
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
from threetears.agent.wake.collections import WakeScheduleCollection
from threetears.agent.wake.migrations import register as register_wake
from threetears.conversations.migrations import register as register_conversations
from threetears.core.collections.asyncpg_init import init_connection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
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


def _build_schedules(pool: asyncpg.Pool) -> WakeScheduleCollection:
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    cfg = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return WakeScheduleCollection(registry=registry, config=cfg)


async def _seed_schedule(
    pool: asyncpg.Pool,
    *,
    next_fire_at: datetime,
    status: str = "active",
) -> tuple[UUID, UUID]:
    """Seed a single active schedule due at ``next_fire_at``.

    Returns ``(conversation_id, schedule_id)``.
    """
    conv = _new_uuid()
    sched = _new_uuid()
    await pool.execute(
        "INSERT INTO agent_wake_schedules "
        "(conversation_id, schedule_id, user_id, agent_id, schedule_type, "
        " schedule_config, execution_mode, status, next_fire_at, "
        " missed_fire_policy) "
        "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)",
        conv,
        sched,
        _new_uuid(),
        _new_uuid(),
        "interval",
        {"seconds": 60},
        "inline",
        status,
        next_fire_at,
        "coalesce",
    )
    return conv, sched


class TestClaimAndReschedule:
    """Single-tick atomicity contract on ``claim_and_reschedule``."""

    async def test_successful_claim_updates_row(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            schedules = _build_schedules(pool)
            now = datetime.now(UTC)
            original_fire = now - timedelta(seconds=10)
            new_fire = now + timedelta(minutes=1)
            conv, sched = await _seed_schedule(pool, next_fire_at=original_fire)

            claimed = await schedules.claim_and_reschedule(
                conversation_id=conv,
                schedule_id=sched,
                expected_next_fire=original_fire,
                computed_next_fire=new_fire,
                new_status="active",
                now=now,
            )
            assert claimed is True
            row = await pool.fetchrow(
                "SELECT next_fire_at, last_fired_at, status FROM agent_wake_schedules WHERE schedule_id = $1",
                sched,
            )
            assert row is not None
            assert row["next_fire_at"] == new_fire
            assert row["last_fired_at"] == now
            assert row["status"] == "active"
        finally:
            await pool.close()

    async def test_stale_expected_returns_false(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            schedules = _build_schedules(pool)
            now = datetime.now(UTC)
            original_fire = now - timedelta(seconds=10)
            conv, sched = await _seed_schedule(pool, next_fire_at=original_fire)

            # First claim wins
            advanced_fire = now + timedelta(minutes=1)
            won = await schedules.claim_and_reschedule(
                conversation_id=conv,
                schedule_id=sched,
                expected_next_fire=original_fire,
                computed_next_fire=advanced_fire,
                new_status="active",
                now=now,
            )
            assert won is True

            # Second claim with stale expected_next_fire loses
            lost = await schedules.claim_and_reschedule(
                conversation_id=conv,
                schedule_id=sched,
                expected_next_fire=original_fire,
                computed_next_fire=now + timedelta(minutes=5),
                new_status="active",
                now=now,
            )
            assert lost is False

            # Row reflects only the first claim
            row = await pool.fetchrow(
                "SELECT next_fire_at FROM agent_wake_schedules WHERE schedule_id = $1",
                sched,
            )
            assert row is not None
            assert row["next_fire_at"] == advanced_fire
        finally:
            await pool.close()

    async def test_concurrent_claims_exactly_one_wins(self, pg_schema: tuple[str, str]) -> None:
        """Two ``claim_and_reschedule`` calls racing on the same expected_next_fire:
        exactly one returns True."""
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            schedules = _build_schedules(pool)
            now = datetime.now(UTC)
            original_fire = now - timedelta(seconds=10)
            conv, sched = await _seed_schedule(pool, next_fire_at=original_fire)

            async def attempt() -> bool:
                return await schedules.claim_and_reschedule(
                    conversation_id=conv,
                    schedule_id=sched,
                    expected_next_fire=original_fire,
                    computed_next_fire=now + timedelta(minutes=1),
                    new_status="active",
                    now=now,
                )

            results = await asyncio.gather(attempt(), attempt())
            assert sum(1 for r in results if r) == 1
            assert sum(1 for r in results if not r) == 1
        finally:
            await pool.close()

    async def test_terminal_one_shot_expires(self, pg_schema: tuple[str, str]) -> None:
        """``computed_next_fire=None`` + ``new_status='expired'`` parks the schedule."""
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            schedules = _build_schedules(pool)
            now = datetime.now(UTC)
            original_fire = now - timedelta(seconds=10)
            conv, sched = await _seed_schedule(pool, next_fire_at=original_fire)

            claimed = await schedules.claim_and_reschedule(
                conversation_id=conv,
                schedule_id=sched,
                expected_next_fire=original_fire,
                computed_next_fire=None,
                new_status="expired",
                now=now,
            )
            assert claimed is True
            row = await pool.fetchrow(
                "SELECT next_fire_at, status FROM agent_wake_schedules WHERE schedule_id = $1",
                sched,
            )
            assert row is not None
            assert row["next_fire_at"] is None
            assert row["status"] == "expired"
        finally:
            await pool.close()
