"""Integration test: ``ScheduledJobCollection.claim_and_reschedule`` is
atomic across concurrent ticks against REAL Postgres.

The double-fire guarantee rests on an optimistic-CAS UPDATE keyed on
``next_fire_at = expected_next_fire``. The unit suite only proves the
SQL is *shaped* right (recording-pool double); this exercises the
actual race against a real engine -- the only level that can prove the
CAS is truly atomic (a recording pool can't double-fire). Mirrors
agent-wake's ``test_claim_and_reschedule.py`` racing-claim test, adapted
to the generic store's PLATFORM-scope migration + ``partition_key`` /
``job_id`` signature.

Verifies:

- a successful claim returns True and advances ``next_fire_at`` +
  ``last_fired_at`` + ``status``.
- a stale ``expected_next_fire`` (set by a concurrent claimant) returns
  False; the row reflects only the winner.
- two concurrent claims against the SAME ``expected_next_fire`` have
  exactly one winner -- no double-fire.
- a terminal one-shot (``computed_next_fire=None``) parks the job.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import UUID

import asyncpg
import pytest
from uuid_utils import uuid7

from threetears.core.collections.asyncpg_init import init_connection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner
from threetears.scheduled_jobs.collections import ScheduledJobCollection
from threetears.scheduled_jobs.migrations import register as register_scheduled_jobs

from .conftest import AsyncpgStore


pytestmark = pytest.mark.integration


def _new_uuid() -> UUID:
    return UUID(str(uuid7()))


async def _apply_schema(url: str, schema: str) -> asyncpg.Pool:
    """Apply the scheduled-jobs PLATFORM migration into ``schema`` and
    return a pool bound to it.

    Unlike agent-wake (AGENT-scope, depends on conversations + skills),
    the generic store is PLATFORM-scope with no upstream tables, so the
    runner applies a single package via ``apply_for_platform_schema``.
    """
    setup_conn = await asyncpg.connect(url)
    try:
        await setup_conn.execute(f'SET search_path TO "{schema}", public')
        runner = MigrationRunner()
        register_scheduled_jobs(runner)
        store = AsyncpgStore(setup_conn)
        await runner.apply_for_platform_schema(store)  # type: ignore[arg-type]
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


def _build_jobs(pool: asyncpg.Pool) -> ScheduledJobCollection:
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    cfg = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return ScheduledJobCollection(registry=registry, config=cfg, nats_client=None)


async def _seed_job(
    pool: asyncpg.Pool,
    *,
    next_fire_at: datetime,
    status: str = "active",
) -> tuple[UUID, UUID]:
    """Seed a single active job due at ``next_fire_at``.

    Returns ``(partition_key, job_id)``.
    """
    partition = _new_uuid()
    job = _new_uuid()
    await pool.execute(
        "INSERT INTO scheduled_jobs "
        "(partition_key, job_id, kind, payload, schedule_type, "
        " schedule_config, status, next_fire_at, missed_fire_policy) "
        "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
        partition,
        job,
        "safety_commit",
        {"repo": "abc"},
        "interval",
        {"seconds": 60},
        status,
        next_fire_at,
        "coalesce",
    )
    return partition, job


class TestClaimAndReschedule:
    """Single-tick atomicity contract on ``claim_and_reschedule``."""

    async def test_successful_claim_updates_row(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            jobs = _build_jobs(pool)
            now = datetime.now(UTC)
            original_fire = now - timedelta(seconds=10)
            new_fire = now + timedelta(minutes=1)
            partition, job = await _seed_job(pool, next_fire_at=original_fire)

            claimed = await jobs.claim_and_reschedule(
                partition_key=partition,
                job_id=job,
                expected_next_fire=original_fire,
                computed_next_fire=new_fire,
                new_status="active",
                now=now,
            )
            assert claimed is True
            row = await pool.fetchrow(
                "SELECT next_fire_at, last_fired_at, status FROM scheduled_jobs WHERE job_id = $1",
                job,
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
            jobs = _build_jobs(pool)
            now = datetime.now(UTC)
            original_fire = now - timedelta(seconds=10)
            partition, job = await _seed_job(pool, next_fire_at=original_fire)

            # First claim wins.
            advanced_fire = now + timedelta(minutes=1)
            won = await jobs.claim_and_reschedule(
                partition_key=partition,
                job_id=job,
                expected_next_fire=original_fire,
                computed_next_fire=advanced_fire,
                new_status="active",
                now=now,
            )
            assert won is True

            # Second claim with stale expected_next_fire loses.
            lost = await jobs.claim_and_reschedule(
                partition_key=partition,
                job_id=job,
                expected_next_fire=original_fire,
                computed_next_fire=now + timedelta(minutes=5),
                new_status="active",
                now=now,
            )
            assert lost is False

            # Row reflects only the first claim.
            row = await pool.fetchrow(
                "SELECT next_fire_at FROM scheduled_jobs WHERE job_id = $1",
                job,
            )
            assert row is not None
            assert row["next_fire_at"] == advanced_fire
        finally:
            await pool.close()

    async def test_concurrent_claims_exactly_one_wins(self, pg_schema: tuple[str, str]) -> None:
        """Two ``claim_and_reschedule`` calls racing on the SAME
        ``expected_next_fire``: exactly one returns True, the other
        False -- the real-engine proof there is no double-fire."""
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            jobs = _build_jobs(pool)
            now = datetime.now(UTC)
            original_fire = now - timedelta(seconds=10)
            advanced_fire = now + timedelta(minutes=1)
            partition, job = await _seed_job(pool, next_fire_at=original_fire)

            async def attempt() -> bool:
                return await jobs.claim_and_reschedule(
                    partition_key=partition,
                    job_id=job,
                    expected_next_fire=original_fire,
                    computed_next_fire=advanced_fire,
                    new_status="active",
                    now=now,
                )

            results = await asyncio.gather(attempt(), attempt())
            assert sum(1 for r in results if r) == 1
            assert sum(1 for r in results if not r) == 1

            # The single winner advanced next_fire_at.
            row = await pool.fetchrow(
                "SELECT next_fire_at FROM scheduled_jobs WHERE job_id = $1",
                job,
            )
            assert row is not None
            assert row["next_fire_at"] == advanced_fire
        finally:
            await pool.close()

    async def test_terminal_one_shot_expires(self, pg_schema: tuple[str, str]) -> None:
        """``computed_next_fire=None`` + ``new_status='expired'`` parks the job."""
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            jobs = _build_jobs(pool)
            now = datetime.now(UTC)
            original_fire = now - timedelta(seconds=10)
            partition, job = await _seed_job(pool, next_fire_at=original_fire)

            claimed = await jobs.claim_and_reschedule(
                partition_key=partition,
                job_id=job,
                expected_next_fire=original_fire,
                computed_next_fire=None,
                new_status="expired",
                now=now,
            )
            assert claimed is True
            row = await pool.fetchrow(
                "SELECT next_fire_at, status FROM scheduled_jobs WHERE job_id = $1",
                job,
            )
            assert row is not None
            assert row["next_fire_at"] is None
            assert row["status"] == "expired"
        finally:
            await pool.close()
