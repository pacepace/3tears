"""every write the job and fire stores make past ``save_entity`` evicts the rows it changed.

The defect: ``set_status``, ``update_schedule``, ``request_immediate_fire``,
``claim_and_reschedule`` and the fire store's finalize and reap writes ran a targeted UPDATE and
settled no cache tier, on the reasoning that nothing read a row by primary key. A host that gives
the registry an L1 (the hub turned it on for every collection) reads ``get`` through it, so a paused
job read back from L1 as active, and a saved-back copy wrote the old status over L3. Each write now
runs in ``bypassing_write``; one that changed nothing (a lost claim, an unmatched guard) evicts
nothing.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import MetaData

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig

from threetears.scheduled_jobs.collections import JobFireCollection, ScheduledJobCollection
from threetears.scheduled_jobs.tables import job_fires_table, scheduled_jobs_table


def _now() -> datetime:
    return datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


class _ScriptedPool:
    """asyncpg-pool-shaped double answering every read with one row and every write as told.

    The raw asyncpg pool has no production protocol (``l3_pool`` is typed ``Any`` by design), so
    there is nothing to declare parity against; this is the shape ``test_store_roundtrip`` uses.
    """

    def __init__(self, row: dict[str, Any], *, write_result: Any) -> None:
        self.row = row
        self.write_result = write_result
        #: what a delete's read of the rows its foreign-key actions reach finds
        self.cascaded: list[dict[str, Any]] = []

    async def execute(self, sql: str, *args: Any) -> str:
        return "UPDATE 1"

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        if sql.lstrip().startswith("UPDATE"):
            return [self.row]
        return list(self.cascaded) if "FROM job_fires WHERE job_id" in sql else []

    async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any] | None:
        return dict(self.row)

    async def fetchval(self, sql: str, *args: Any) -> Any:
        return self.write_result


def _registry(pool: _ScriptedPool) -> CollectionRegistry:
    metadata = MetaData()
    scheduled_jobs_table(metadata)
    job_fires_table(metadata)
    l1 = SQLiteBackend(db_name=f"scheduled_jobs_raw_writes_{uuid.uuid4().hex[:8]}")
    l1.initialize(metadata)
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, l3_pool=pool)
    return registry


def _config() -> DefaultCoreConfig:
    return DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")


def _job_row(partition_key: UUID, job_id: UUID) -> dict[str, Any]:
    now = _now()
    return {
        "partition_key": partition_key,
        "job_id": job_id,
        "kind": "safety_commit",
        "payload": {"repo": "abc"},
        "schedule_type": "interval",
        "schedule_config": {"seconds": 300},
        "status": "active",
        "next_fire_at": now,
        "last_fired_at": None,
        "missed_fire_policy": "coalesce",
        "name": "auto-commit",
        "date_created": now,
        "date_updated": now,
    }


def _fire_row(partition_key: UUID, fire_id: UUID) -> dict[str, Any]:
    now = _now()
    return {
        "partition_key": partition_key,
        "fire_id": fire_id,
        "job_id": uuid4(),
        "scheduled_fire_at": now,
        "actual_fired_at": now,
        "status": "dispatching",
        "output": None,
        "latency_ms": None,
        "error": None,
        "date_created": now,
    }


async def _cached_job(*, write_result: Any) -> tuple[ScheduledJobCollection, UUID, UUID]:
    partition_key, job_id = uuid4(), uuid4()
    pool = _ScriptedPool(_job_row(partition_key, job_id), write_result=write_result)
    jobs = ScheduledJobCollection(registry=_registry(pool), config=_config(), nats_client=None)
    assert await jobs.get((partition_key, job_id)) is not None
    assert jobs.get_row_sync((partition_key, job_id)) is not None
    return jobs, partition_key, job_id


async def _cached_fire() -> tuple[JobFireCollection, UUID, UUID]:
    partition_key, fire_id = uuid4(), uuid4()
    pool = _ScriptedPool(_fire_row(partition_key, fire_id), write_result=None)
    fires = JobFireCollection(registry=_registry(pool), config=_config(), nats_client=None)
    assert await fires.get((partition_key, fire_id)) is not None
    assert fires.get_row_sync((partition_key, fire_id)) is not None
    return fires, partition_key, fire_id


class TestJobWritesEvictTheRow:
    """a job write that changed the row leaves no tier holding the row it replaced."""

    async def test_set_status(self) -> None:
        jobs, partition_key, job_id = await _cached_job(write_result=uuid4())
        assert await jobs.set_status(partition_key=partition_key, job_id=job_id, status="paused", now=_now())
        assert jobs.get_row_sync((partition_key, job_id)) is None

    async def test_update_schedule(self) -> None:
        jobs, partition_key, job_id = await _cached_job(write_result=uuid4())
        assert await jobs.update_schedule(
            partition_key=partition_key, job_id=job_id, schedule_config={"seconds": 60}, now=_now()
        )
        assert jobs.get_row_sync((partition_key, job_id)) is None

    async def test_request_immediate_fire(self) -> None:
        jobs, partition_key, job_id = await _cached_job(write_result=uuid4())
        assert await jobs.request_immediate_fire(partition_key=partition_key, job_id=job_id, now=_now())
        assert jobs.get_row_sync((partition_key, job_id)) is None

    async def test_claim_and_reschedule(self) -> None:
        jobs, partition_key, job_id = await _cached_job(write_result=uuid4())
        assert await jobs.claim_and_reschedule(
            partition_key=partition_key,
            job_id=job_id,
            expected_next_fire=_now(),
            computed_next_fire=None,
            new_status="expired",
            now=_now(),
        )
        assert jobs.get_row_sync((partition_key, job_id)) is None

    async def test_a_lost_claim_changed_nothing_and_keeps_the_row(self) -> None:
        jobs, partition_key, job_id = await _cached_job(write_result=None)
        assert not await jobs.claim_and_reschedule(
            partition_key=partition_key,
            job_id=job_id,
            expected_next_fire=_now(),
            computed_next_fire=None,
            new_status="expired",
            now=_now(),
        )
        assert jobs.get_row_sync((partition_key, job_id)) is not None


class TestFireWritesEvictTheRow:
    """a fire write leaves no tier holding the row it replaced."""

    async def test_finalize_success(self) -> None:
        fires, partition_key, fire_id = await _cached_fire()
        await fires.finalize_success(partition_key, fire_id, output={"ok": True}, latency_ms=5)
        assert fires.get_row_sync((partition_key, fire_id)) is None

    async def test_finalize_failed(self) -> None:
        fires, partition_key, fire_id = await _cached_fire()
        await fires.finalize_failed(partition_key, fire_id, error="boom")
        assert fires.get_row_sync((partition_key, fire_id)) is None

    async def test_reap_fires_evicts_each_row_it_reaped(self) -> None:
        fires, partition_key, fire_id = await _cached_fire()
        assert await fires.reap_fires([fire_id]) == [fire_id]
        assert fires.get_row_sync((partition_key, fire_id)) is None


class TestJobDeleteEvictsItsFires:
    """a job's delete announces the fires ``ON DELETE CASCADE`` removes with it."""

    async def test_a_deleted_jobs_cached_fire_is_evicted(self) -> None:
        partition_key, job_id, fire_id = uuid4(), uuid4(), uuid4()
        fire = {**_fire_row(partition_key, fire_id), "job_id": job_id}
        pool = _ScriptedPool(fire, write_result=None)
        registry = _registry(pool)
        jobs = ScheduledJobCollection(registry=registry, config=_config(), nats_client=None)
        fires = JobFireCollection(registry=registry, config=_config(), nats_client=None)
        registry.register(jobs)
        registry.register(fires)
        assert await fires.get((partition_key, fire_id)) is not None
        pool.row = _job_row(partition_key, job_id)
        pool.cascaded = [fire]
        assert await jobs.delete((partition_key, job_id))
        assert fires.get_row_sync((partition_key, fire_id)) is None
