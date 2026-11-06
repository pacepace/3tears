"""Integration: the scheduled-jobs tables are declared once, and the declaration is the migrated table.

Each collection's ``TableSchema`` is what its L1 table, L2 codec and upsert
column lists are read off; nothing else names its columns. Holds:

- each schema to the table the v001 migration leaves
  (:func:`live_tier_parity_findings`): every column, type, nullability,
  default and the primary key, and asyncpg's read of a written row;
- a job saved through the generated upsert reads back the same from L3, and
  an update keeps ``date_created`` and rewrites the rest;
- a fire's update writes only its outcome, keeping its job and instants.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

import asyncpg
import pytest
from uuid_utils import uuid7

from threetears.core.testing.tier_parity import live_tier_parity_findings

from threetears.scheduled_jobs.collections import JobFireCollection, ScheduledJobCollection

from .conftest import apply_schema, build_stores

pytestmark = pytest.mark.integration


def _new_uuid() -> UUID:
    return UUID(str(uuid7()))


async def test_each_schema_is_the_migrated_table(pg_schema: tuple[str, str]) -> None:
    # declared, so the gate below checks them rather than skipping a class with no schema
    assert ScheduledJobCollection.schema.name == "scheduled_jobs"
    assert JobFireCollection.schema.name == "job_fires"
    url, schema = pg_schema
    pool = await apply_schema(url, schema)
    try:
        async with pool.acquire() as conn:
            assert await live_tier_parity_findings(conn, [ScheduledJobCollection, JobFireCollection]) == []
    finally:
        await pool.close()


async def test_a_job_and_a_fire_round_trip_through_the_generated_upsert(pg_schema: tuple[str, str]) -> None:
    url, schema = pg_schema
    pool: asyncpg.Pool = await apply_schema(url, schema)
    try:
        jobs, fires = build_stores(pool)
        partition, job_id = _new_uuid(), _new_uuid()
        now = datetime.now(UTC).replace(microsecond=0)
        await jobs.save_entity(
            jobs.create(
                {
                    "partition_key": partition,
                    "job_id": job_id,
                    "kind": "safety_commit",
                    "payload": {"repo": "abc", "n": 3},
                    "schedule_type": "interval",
                    "schedule_config": {"seconds": 300},
                    "next_fire_at": now,
                    "name": "auto-commit",
                }
            )
        )
        stored = await jobs.fetch_from_store((partition, job_id))
        assert stored is not None
        assert stored["payload"] == {"repo": "abc", "n": 3}
        assert stored["status"] == "active"  # the column's default, not a value the write invented
        assert stored["missed_fire_policy"] == "coalesce"
        assert stored["next_fire_at"] == now

        job = await jobs.get((partition, job_id))
        assert job is not None
        job.payload = {"repo": "xyz"}
        await jobs.save_entity(job)
        updated = await jobs.fetch_from_store((partition, job_id))
        assert updated is not None
        assert updated["payload"] == {"repo": "xyz"}
        assert updated["date_created"] == stored["date_created"]

        fire_id = _new_uuid()
        await fires.create_dispatching(
            fire_id=fire_id,
            job_id=job_id,
            partition_key=partition,
            scheduled_fire_at=now,
            actual_fired_at=now,
        )
        fire = await fires.get((partition, fire_id))
        assert fire is not None
        before = await fires.fetch_from_store((partition, fire_id))
        assert before is not None
        sent = dict(before)
        sent.update({"status": "succeeded", "latency_ms": 12, "scheduled_fire_at": now + timedelta(hours=1)})
        await fires.save_to_store(sent)
        after = await fires.fetch_from_store((partition, fire_id))
        assert after is not None
        assert after["status"] == "succeeded" and after["latency_ms"] == 12
        # an update writes the outcome only: the fire's schedule is kept
        assert after["scheduled_fire_at"] == before["scheduled_fire_at"]
    finally:
        await pool.close()
