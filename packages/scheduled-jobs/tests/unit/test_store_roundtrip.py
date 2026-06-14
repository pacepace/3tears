"""Unit tests for the default store collections (the L3 store seam).

The collections are wired to a real Postgres pool in integration tests
(out of scope for this unit suite); here we exercise them over a fake
asyncpg-shaped pool that records SQL + params and returns canned rows.
This asserts:

- the collections satisfy the store Protocols structurally (the tick
  engine's contract).
- ``claim_and_reschedule`` issues the optimistic-CAS UPDATE keyed on the
  expected ``next_fire_at`` and returns the right boolean.
- ``create_dispatching`` / ``finalize_success`` / ``finalize_failed``
  issue the expected writes.
- ``list_due_for_tick`` reads back rows as entities that satisfy the
  ``DueSchedule`` Protocol with the opaque ``kind`` + ``payload`` intact.
- ``save_to_store`` -> ``fetch_from_store`` round-trips a job row through
  the entity accessors.
- serialize / deserialize round-trips a row dict through the L2 codec.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig

from threetears.scheduled_jobs.collections import (
    JobFireCollection,
    ScheduledJobCollection,
)
from threetears.scheduled_jobs.entities import JobFireEntity, ScheduledJobEntity
from threetears.scheduled_jobs.protocols import DueSchedule, FireStore, ScheduleStore


def _now() -> datetime:
    return datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


class _RecordingPool:
    """asyncpg-pool-shaped recorder.

    Records every ``execute`` / ``fetch`` / ``fetchrow`` / ``fetchval``
    call as ``(sql, args)`` and returns scripted results. A minimal
    asyncpg surface stand-in -- the same role agent-wake's unit tests
    fill with inline pool doubles; the raw asyncpg pool has no production
    protocol (``l3_pool`` is typed ``Any`` by design), so there is
    nothing to declare parity against.
    """

    def __init__(
        self,
        *,
        fetch_rows: list[dict[str, Any]] | None = None,
        fetchrow_row: dict[str, Any] | None = None,
        fetchval_result: Any = None,
    ) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self._fetch_rows = fetch_rows or []
        self._fetchrow_row = fetchrow_row
        self._fetchval_result = fetchval_result

    async def execute(self, sql: str, *args: Any) -> str:
        self.calls.append((sql, args))
        return "UPDATE 1"

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        self.calls.append((sql, args))
        return self._fetch_rows

    async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any] | None:
        self.calls.append((sql, args))
        return self._fetchrow_row

    async def fetchval(self, sql: str, *args: Any) -> Any:
        self.calls.append((sql, args))
        return self._fetchval_result


def _make_schedule_collection(pool: _RecordingPool) -> ScheduledJobCollection:
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    cfg = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return ScheduledJobCollection(registry=registry, config=cfg, nats_client=None)


def _make_fire_collection(pool: _RecordingPool) -> JobFireCollection:
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    cfg = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return JobFireCollection(registry=registry, config=cfg, nats_client=None)


def _job_row(**overrides: Any) -> dict[str, Any]:
    now = _now()
    row = {
        "partition_key": uuid4(),
        "job_id": uuid4(),
        "kind": "safety_commit",
        "payload": {"repo": "abc", "n": 3},
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
    row.update(overrides)
    return row


class TestProtocolConformance:
    """The collections satisfy the store Protocols structurally."""

    def test_schedule_collection_is_schedule_store(self) -> None:
        pool = _RecordingPool()
        coll = _make_schedule_collection(pool)
        assert isinstance(coll, ScheduleStore)

    def test_fire_collection_is_fire_store(self) -> None:
        pool = _RecordingPool()
        coll = _make_fire_collection(pool)
        assert isinstance(coll, FireStore)

    def test_partition_columns(self) -> None:
        assert ScheduledJobCollection.partition_column == "partition_key"
        assert JobFireCollection.partition_column == "partition_key"

    def test_composite_pks(self) -> None:
        assert ScheduledJobCollection.primary_key_column == ("partition_key", "job_id")
        assert JobFireCollection.primary_key_column == ("partition_key", "fire_id")


class TestClaimAndReschedule:
    """The optimistic-CAS UPDATE binds the expected next-fire predicate."""

    async def test_claim_hit_returns_true(self) -> None:
        job_id = uuid4()
        pool = _RecordingPool(fetchval_result=job_id)
        coll = _make_schedule_collection(pool)
        partition = uuid4()
        expected = _now()
        computed = datetime(2026, 6, 1, 12, 5, tzinfo=UTC)
        ok = await coll.claim_and_reschedule(
            partition_key=partition,
            job_id=job_id,
            expected_next_fire=expected,
            computed_next_fire=computed,
            new_status="active",
            now=_now(),
        )
        assert ok is True
        sql, args = pool.calls[-1]
        assert "UPDATE scheduled_jobs" in sql
        assert "next_fire_at = $6" in sql  # the CAS predicate column
        # bound params: computed, now, status, partition, job, expected
        assert args[0] == computed
        assert args[2] == "active"
        assert args[3] == partition
        assert args[4] == job_id
        assert args[5] == expected

    async def test_claim_miss_returns_false(self) -> None:
        pool = _RecordingPool(fetchval_result=None)
        coll = _make_schedule_collection(pool)
        ok = await coll.claim_and_reschedule(
            partition_key=uuid4(),
            job_id=uuid4(),
            expected_next_fire=_now(),
            computed_next_fire=None,
            new_status="expired",
            now=_now(),
        )
        assert ok is False


class TestFireStoreWrites:
    """create_dispatching / finalize_success / finalize_failed issue writes."""

    async def test_create_dispatching_inserts_dispatching_status(self) -> None:
        pool = _RecordingPool()
        coll = _make_fire_collection(pool)
        fire_id, job_id, partition = uuid4(), uuid4(), uuid4()
        await coll.create_dispatching(
            fire_id=fire_id,
            job_id=job_id,
            partition_key=partition,
            scheduled_fire_at=_now(),
            actual_fired_at=_now(),
        )
        sql, args = pool.calls[-1]
        assert "INSERT INTO job_fires" in sql
        assert "dispatching" in args

    async def test_finalize_success_updates_terminal_columns(self) -> None:
        pool = _RecordingPool()
        coll = _make_fire_collection(pool)
        fire_id, partition = uuid4(), uuid4()
        await coll.finalize_success(partition, fire_id, status="succeeded", output={"ok": 1}, latency_ms=9)
        sql, args = pool.calls[-1]
        assert "UPDATE job_fires SET status = $1" in sql
        assert args[0] == "succeeded"
        assert args[1] == {"ok": 1}
        assert args[2] == 9
        assert args[3] == partition
        assert args[4] == fire_id

    async def test_finalize_failed_records_error(self) -> None:
        pool = _RecordingPool()
        coll = _make_fire_collection(pool)
        fire_id, partition = uuid4(), uuid4()
        await coll.finalize_failed(partition, fire_id, error="boom", latency_ms=None)
        sql, args = pool.calls[-1]
        assert "status = 'failed'" in sql
        assert args[0] == "boom"


class TestListDueForTick:
    """list_due_for_tick reads rows back as DueSchedule-shaped entities."""

    async def test_due_rows_become_due_schedules(self) -> None:
        row = _job_row()
        pool = _RecordingPool(fetch_rows=[row])
        coll = _make_schedule_collection(pool)
        due = await coll.list_due_for_tick(now=_now())
        assert len(due) == 1
        sched = due[0]
        assert isinstance(sched, DueSchedule)
        assert isinstance(sched, ScheduledJobEntity)
        # the opaque kind + payload survive intact
        assert sched.kind == "safety_commit"
        assert sched.payload == {"repo": "abc", "n": 3}
        assert sched.schedule_type == "interval"
        assert sched.missed_fire_policy == "coalesce"
        assert sched.next_fire_at == row["next_fire_at"]
        # the due query filters active + non-null next_fire_at
        sql, _args = pool.calls[-1]
        assert "status = 'active'" in sql
        assert "next_fire_at <= $1" in sql


class TestSaveFetchRoundTrip:
    """save_to_store -> fetch_from_store round-trips a job through the entity."""

    async def test_round_trip(self) -> None:
        row = _job_row()
        # the pool returns the same row on fetchrow so the round-trip is real
        pool = _RecordingPool(fetchrow_row=row)
        coll = _make_schedule_collection(pool)

        # save: the entity's data projects to the upsert params + executes
        n = await coll.save_to_store(row)
        assert n == 1
        save_sql, save_args = pool.calls[-1]
        assert "INSERT INTO scheduled_jobs" in save_sql
        assert "ON CONFLICT (partition_key, job_id)" in save_sql

        # fetch: the row comes back and the entity exposes it
        fetched = await coll.fetch_from_store((row["partition_key"], row["job_id"]))
        assert fetched is not None
        entity = ScheduledJobEntity(fetched, is_new=False, collection=coll)
        assert entity.partition_key == row["partition_key"]
        assert entity.job_id == row["job_id"]
        assert entity.kind == "safety_commit"
        assert entity.payload == {"repo": "abc", "n": 3}


class TestSerializeRoundTrip:
    """serialize -> deserialize round-trips a row through the L2 codec."""

    def test_job_row_serialize_round_trip(self) -> None:
        pool = _RecordingPool()
        coll = _make_schedule_collection(pool)
        row = _job_row()
        blob = coll.serialize(row)
        assert isinstance(blob, bytes)
        restored = coll.deserialize(blob)
        # the type hints rehydrate UUID / datetime / dict back to native
        assert isinstance(restored["partition_key"], UUID)
        assert isinstance(restored["job_id"], UUID)
        assert restored["kind"] == "safety_commit"
        assert restored["payload"] == {"repo": "abc", "n": 3}
        assert isinstance(restored["next_fire_at"], datetime)

    def test_fire_row_serialize_round_trip(self) -> None:
        pool = _RecordingPool()
        coll = _make_fire_collection(pool)
        now = _now()
        row = {
            "partition_key": uuid4(),
            "fire_id": uuid4(),
            "job_id": uuid4(),
            "scheduled_fire_at": now,
            "actual_fired_at": now,
            "status": "succeeded",
            "output": {"result": "ok"},
            "latency_ms": 11,
            "error": None,
            "date_created": now,
        }
        restored = coll.deserialize(coll.serialize(row))
        assert isinstance(restored["fire_id"], UUID)
        assert restored["status"] == "succeeded"
        assert restored["output"] == {"result": "ok"}


class TestFireEntityAccessors:
    """The fire entity exposes the row columns through typed accessors."""

    def test_fire_entity_reads_columns(self) -> None:
        now = _now()
        partition, fire_id, job_id = uuid4(), uuid4(), uuid4()
        entity = JobFireEntity(
            {
                "partition_key": partition,
                "fire_id": fire_id,
                "job_id": job_id,
                "scheduled_fire_at": now,
                "actual_fired_at": now,
                "status": "succeeded",
                "output": {"a": 1},
                "latency_ms": 5,
                "error": None,
                "date_created": now,
            },
            is_new=False,
        )
        assert entity.partition_key == partition
        assert entity.fire_id == fire_id
        assert entity.job_id == job_id
        assert entity.status == "succeeded"
        assert entity.output == {"a": 1}
        assert entity.latency_ms == 5
        assert entity.error is None
