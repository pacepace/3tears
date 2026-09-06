"""Integration test: the operator-facing surface on the scheduled-job stores.

A tick pump is a thing an operator has to be able to see and steer: list what
is scheduled, read why the last run failed, stop one that is misbehaving, retune
its cadence, and make one run now. Before these methods the only way to do any
of it was hand-written SQL against ``scheduled_jobs``, which is what every
consumer's runbook told its operators to do.

Every method here is SQL, so the honest level is a real engine. The unit suite
cannot distinguish a predicate that matches the right row from one that matches
every row.

Verifies:

- listing spans partitions (each job carries its own ``partition_key``, so a
  partition-scoped read cannot enumerate a deployment) and filters by kind and
  status in SQL rather than in Python;
- ``set_status`` moves only the status, leaving ``next_fire_at`` alone -- the
  property that makes pause/resume safe;
- ``update_schedule`` recomputes the next fire through the same function the
  tick engine uses, so a retune cannot drift from the engine's own maths;
- ``request_immediate_fire`` refuses a paused job rather than silently firing
  one an operator believes is stopped;
- ``latest_for_jobs`` returns the newest fire per job in ONE query.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

import asyncpg
import pytest
from uuid_utils import uuid7

from threetears.core.collections.asyncpg_init import init_connection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner
from threetears.scheduled_jobs.collections import JobFireCollection, ScheduledJobCollection
from threetears.scheduled_jobs.migrations import register as register_scheduled_jobs

from .conftest import AsyncpgStore


pytestmark = pytest.mark.integration


def _new_uuid() -> UUID:
    return UUID(str(uuid7()))


async def _apply_schema(url: str, schema: str) -> asyncpg.Pool:
    """Apply the scheduled-jobs migration into ``schema`` and return a pool."""
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


def _build_stores(pool: asyncpg.Pool) -> tuple[ScheduledJobCollection, JobFireCollection]:
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    cfg = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return (
        ScheduledJobCollection(registry=registry, config=cfg, nats_client=None),
        JobFireCollection(registry=registry, config=cfg, nats_client=None),
    )


async def _seed_job(
    pool: asyncpg.Pool,
    *,
    kind: str = "safety_commit",
    name: str | None = None,
    status: str = "active",
    schedule_type: str = "interval",
    schedule_config: dict | None = None,
    next_fire_at: datetime | None = None,
) -> tuple[UUID, UUID]:
    """Seed one job in its OWN partition, as every real seeder does.

    Returns ``(partition_key, job_id)``.
    """
    partition = _new_uuid()
    job = _new_uuid()
    await pool.execute(
        "INSERT INTO scheduled_jobs "
        "(partition_key, job_id, kind, payload, schedule_type, "
        " schedule_config, status, next_fire_at, missed_fire_policy, name) "
        "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)",
        partition,
        job,
        kind,
        {},
        schedule_type,
        schedule_config if schedule_config is not None else {"seconds": 60},
        status,
        next_fire_at if next_fire_at is not None else datetime.now(UTC) + timedelta(minutes=5),
        "coalesce",
        name,
    )
    return partition, job


async def _seed_fire(
    pool: asyncpg.Pool,
    partition: UUID,
    job: UUID,
    *,
    status: str,
    fired_at: datetime,
    error: str | None = None,
) -> UUID:
    fire_id = _new_uuid()
    await pool.execute(
        "INSERT INTO job_fires (partition_key, fire_id, job_id, scheduled_fire_at, "
        "actual_fired_at, status, output, latency_ms, error) "
        "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
        partition,
        fire_id,
        job,
        fired_at,
        fired_at,
        status,
        None,
        100,
        error,
    )
    return fire_id


class TestListJobs:
    """Cross-partition enumeration, filtered in SQL."""

    async def test_it_lists_jobs_from_different_partitions(self, pg_schema: tuple[str, str]) -> None:
        """Each job has its own partition, so listing must span them.

        This is the whole reason a partition-scoped read cannot serve an admin
        view: it would return exactly one job.
        """
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            jobs, _ = _build_stores(pool)
            await _seed_job(pool, kind="alpha")
            await _seed_job(pool, kind="beta")
            await _seed_job(pool, kind="gamma")

            listed = await jobs.list_jobs()

            assert {j.kind for j in listed} == {"alpha", "beta", "gamma"}
            assert len({j.partition_key for j in listed}) == 3
        finally:
            await pool.close()

    async def test_it_includes_paused_and_expired_jobs(self, pg_schema: tuple[str, str]) -> None:
        """Status-blind by default: a paused job is the one you came to find."""
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            jobs, _ = _build_stores(pool)
            await _seed_job(pool, kind="active_one", status="active")
            await _seed_job(pool, kind="paused_one", status="paused")
            await _seed_job(pool, kind="expired_one", status="expired")

            listed = await jobs.list_jobs()

            assert {j.kind for j in listed} == {"active_one", "paused_one", "expired_one"}
        finally:
            await pool.close()

    async def test_filters_are_sql_predicates_not_python(self, pg_schema: tuple[str, str]) -> None:
        """A filter applied after the fetch would still page over everything."""
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            jobs, _ = _build_stores(pool)
            await _seed_job(pool, kind="alpha", status="active")
            await _seed_job(pool, kind="alpha", status="paused")
            await _seed_job(pool, kind="beta", status="active")

            by_kind = await jobs.list_jobs(kinds=["alpha"])
            by_status = await jobs.list_jobs(statuses=["paused"])
            both = await jobs.list_jobs(kinds=["alpha"], statuses=["active"])

            assert len(by_kind) == 2
            assert len(by_status) == 1
            assert len(both) == 1
            assert both[0].kind == "alpha"
        finally:
            await pool.close()

    async def test_an_explicitly_empty_filter_matches_nothing(self, pg_schema: tuple[str, str]) -> None:
        """Empty means empty; only ``None`` means unfiltered.

        Widening an empty list to "everything" is how a caller that computed
        zero kinds ends up operating on the whole deployment.
        """
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            jobs, _ = _build_stores(pool)
            await _seed_job(pool, kind="alpha")

            assert await jobs.list_jobs(kinds=[]) == []
            assert await jobs.list_jobs(statuses=[]) == []
            assert len(await jobs.list_jobs()) == 1
        finally:
            await pool.close()

    async def test_it_pages_deterministically(self, pg_schema: tuple[str, str]) -> None:
        """Ordering is total, so paging cannot repeat or skip a row."""
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            jobs, _ = _build_stores(pool)
            for index in range(5):
                await _seed_job(pool, kind=f"kind_{index}")

            first = await jobs.list_jobs(limit=2, offset=0)
            second = await jobs.list_jobs(limit=2, offset=2)
            third = await jobs.list_jobs(limit=2, offset=4)

            seen = [j.job_id for j in (*first, *second, *third)]
            assert len(seen) == 5
            assert len(set(seen)) == 5
        finally:
            await pool.close()

    async def test_count_matches_the_same_filters(self, pg_schema: tuple[str, str]) -> None:
        """A count that disagreed with the listing would mis-page the caller."""
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            jobs, _ = _build_stores(pool)
            await _seed_job(pool, kind="alpha", status="active")
            await _seed_job(pool, kind="alpha", status="paused")
            await _seed_job(pool, kind="beta", status="active")

            assert await jobs.count_jobs() == 3
            assert await jobs.count_jobs(kinds=["alpha"]) == 2
            assert await jobs.count_jobs(statuses=["active"]) == 2
            assert await jobs.count_jobs(kinds=[]) == 0
        finally:
            await pool.close()


class TestSetStatus:
    """Pause and resume, without disturbing the schedule."""

    async def test_pausing_leaves_next_fire_at_untouched(self, pg_schema: tuple[str, str]) -> None:
        """The property the whole pause/resume story rests on.

        A pause that cleared or advanced ``next_fire_at`` would silently change
        WHEN the job resumes, which is a different operation from stopping it.
        """
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            jobs, _ = _build_stores(pool)
            fire_at = datetime.now(UTC) + timedelta(hours=3)
            partition, job = await _seed_job(pool, next_fire_at=fire_at)

            changed = await jobs.set_status(
                partition_key=partition,
                job_id=job,
                status="paused",
                now=datetime.now(UTC),
            )

            assert changed is True
            row = await pool.fetchrow(
                "SELECT status, next_fire_at FROM scheduled_jobs WHERE job_id = $1",
                job,
            )
            assert row["status"] == "paused"
            assert row["next_fire_at"] == fire_at
        finally:
            await pool.close()

    async def test_resume_restores_the_job_to_the_tick(self, pg_schema: tuple[str, str]) -> None:
        """After a resume the tick engine's own due-scan sees the job again."""
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            jobs, _ = _build_stores(pool)
            due = datetime.now(UTC) - timedelta(minutes=1)
            partition, job = await _seed_job(pool, status="paused", next_fire_at=due)

            assert await jobs.list_due_for_tick(datetime.now(UTC), kinds=["safety_commit"]) == []

            await jobs.set_status(
                partition_key=partition,
                job_id=job,
                status="active",
                now=datetime.now(UTC),
            )

            due_now = await jobs.list_due_for_tick(datetime.now(UTC), kinds=["safety_commit"])
            assert [j.job_id for j in due_now] == [job]
        finally:
            await pool.close()

    async def test_a_missing_job_reports_no_change(self, pg_schema: tuple[str, str]) -> None:
        """The caller needs to tell "paused it" from "there was nothing there"."""
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            jobs, _ = _build_stores(pool)
            changed = await jobs.set_status(
                partition_key=_new_uuid(),
                job_id=_new_uuid(),
                status="paused",
                now=datetime.now(UTC),
            )
            assert changed is False
        finally:
            await pool.close()

    async def test_an_unknown_status_is_refused_before_it_reaches_the_database(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """The table's CHECK would reject it; a clear error beats a raw one."""
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            jobs, _ = _build_stores(pool)
            partition, job = await _seed_job(pool)
            with pytest.raises(ValueError, match="status"):
                await jobs.set_status(
                    partition_key=partition,
                    job_id=job,
                    status="halted",
                    now=datetime.now(UTC),
                )
        finally:
            await pool.close()

    async def test_it_touches_only_the_named_job(self, pg_schema: tuple[str, str]) -> None:
        """A predicate missing the partition would still match by job_id alone.

        Worth pinning: ``job_id`` carries its own UNIQUE constraint, so an
        under-specified predicate passes every single-row test and is only
        wrong in ways a multi-row fixture can show.
        """
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            jobs, _ = _build_stores(pool)
            partition, job = await _seed_job(pool, kind="target")
            _, other = await _seed_job(pool, kind="bystander")

            await jobs.set_status(
                partition_key=partition,
                job_id=job,
                status="paused",
                now=datetime.now(UTC),
            )

            others = await pool.fetchval("SELECT status FROM scheduled_jobs WHERE job_id = $1", other)
            assert others == "active"
        finally:
            await pool.close()


class TestUpdateSchedule:
    """Retuning cadence, using the engine's own next-fire maths."""

    async def test_it_writes_the_config_and_recomputes_the_next_fire(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """A cadence change that left ``next_fire_at`` alone would not take effect
        until the old cadence fired once more."""
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            jobs, _ = _build_stores(pool)
            far_future = datetime.now(UTC) + timedelta(days=30)
            partition, job = await _seed_job(
                pool,
                schedule_config={"seconds": 86400},
                next_fire_at=far_future,
            )

            now = datetime.now(UTC)
            updated = await jobs.update_schedule(
                partition_key=partition,
                job_id=job,
                schedule_config={"seconds": 60},
                now=now,
            )

            assert updated is True
            row = await pool.fetchrow(
                "SELECT schedule_config, next_fire_at FROM scheduled_jobs WHERE job_id = $1",
                job,
            )
            assert row["schedule_config"] == {"seconds": 60}
            # recomputed through compute_next_fire_at for an interval schedule,
            # which anchors on now under the coalesce policy.
            assert row["next_fire_at"] < far_future
            assert abs((row["next_fire_at"] - (now + timedelta(seconds=60))).total_seconds()) < 5
        finally:
            await pool.close()

    async def test_it_honours_the_rows_own_schedule_type(self, pg_schema: tuple[str, str]) -> None:
        """The recompute reads schedule_type from the ROW, not from a caller hint.

        A caller-supplied type could disagree with the stored one and write a
        next fire the engine would never have produced.
        """
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            jobs, _ = _build_stores(pool)
            partition, job = await _seed_job(
                pool,
                schedule_type="every_n_hours",
                schedule_config={"n": 12},
            )

            now = datetime.now(UTC)
            await jobs.update_schedule(
                partition_key=partition,
                job_id=job,
                schedule_config={"n": 2},
                now=now,
            )

            next_fire = await pool.fetchval("SELECT next_fire_at FROM scheduled_jobs WHERE job_id = $1", job)
            assert abs((next_fire - (now + timedelta(hours=2))).total_seconds()) < 5
        finally:
            await pool.close()

    async def test_a_config_the_schedule_type_rejects_raises(self, pg_schema: tuple[str, str]) -> None:
        """Validation comes from the engine's own maths, not a second copy of it."""
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            jobs, _ = _build_stores(pool)
            partition, job = await _seed_job(pool, schedule_type="interval")

            with pytest.raises(ValueError):
                await jobs.update_schedule(
                    partition_key=partition,
                    job_id=job,
                    schedule_config={"seconds": 0},
                    now=datetime.now(UTC),
                )
        finally:
            await pool.close()

    async def test_a_missing_job_reports_no_change(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            jobs, _ = _build_stores(pool)
            updated = await jobs.update_schedule(
                partition_key=_new_uuid(),
                job_id=_new_uuid(),
                schedule_config={"seconds": 60},
                now=datetime.now(UTC),
            )
            assert updated is False
        finally:
            await pool.close()


class TestRequestImmediateFire:
    """Run-now, without becoming a way to fire a stopped job."""

    async def test_it_brings_an_active_job_forward(self, pg_schema: tuple[str, str]) -> None:
        """After the request the engine's due-scan picks the job up."""
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            jobs, _ = _build_stores(pool)
            partition, job = await _seed_job(pool, next_fire_at=datetime.now(UTC) + timedelta(days=1))

            now = datetime.now(UTC)
            requested = await jobs.request_immediate_fire(
                partition_key=partition,
                job_id=job,
                now=now,
            )

            assert requested is True
            due = await jobs.list_due_for_tick(now, kinds=["safety_commit"])
            assert [j.job_id for j in due] == [job]
        finally:
            await pool.close()

    async def test_it_refuses_a_paused_job(self, pg_schema: tuple[str, str]) -> None:
        """Firing a schedule an operator believes is stopped is the wrong default.

        Refusing makes the caller resume first, which is a decision someone
        takes deliberately rather than a side effect of asking for one run.
        """
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            jobs, _ = _build_stores(pool)
            fire_at = datetime.now(UTC) + timedelta(days=1)
            partition, job = await _seed_job(pool, status="paused", next_fire_at=fire_at)

            requested = await jobs.request_immediate_fire(
                partition_key=partition,
                job_id=job,
                now=datetime.now(UTC),
            )

            assert requested is False
            unchanged = await pool.fetchval("SELECT next_fire_at FROM scheduled_jobs WHERE job_id = $1", job)
            assert unchanged == fire_at
        finally:
            await pool.close()

    async def test_a_missing_job_reports_no_change(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            jobs, _ = _build_stores(pool)
            requested = await jobs.request_immediate_fire(
                partition_key=_new_uuid(),
                job_id=_new_uuid(),
                now=datetime.now(UTC),
            )
            assert requested is False
        finally:
            await pool.close()


class TestLatestForJobs:
    """The newest fire per job, in one query."""

    async def test_it_returns_the_newest_fire_for_each_job(self, pg_schema: tuple[str, str]) -> None:
        """An older success must not mask a newer failure."""
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            jobs, fires = _build_stores(pool)
            now = datetime.now(UTC)
            partition_a, job_a = await _seed_job(pool, kind="alpha")
            partition_b, job_b = await _seed_job(pool, kind="beta")
            await _seed_fire(pool, partition_a, job_a, status="succeeded", fired_at=now - timedelta(days=2))
            await _seed_fire(
                pool,
                partition_a,
                job_a,
                status="failed",
                fired_at=now - timedelta(hours=1),
                error="drift beyond tolerance",
            )
            await _seed_fire(pool, partition_b, job_b, status="succeeded", fired_at=now - timedelta(hours=3))

            latest = await fires.latest_for_jobs([job_a, job_b])

            assert latest[job_a].status == "failed"
            assert latest[job_a].error == "drift beyond tolerance"
            assert latest[job_b].status == "succeeded"
            assert len(await jobs.list_jobs()) == 2
        finally:
            await pool.close()

    async def test_a_job_that_never_fired_is_absent_rather_than_null(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """Absence is the caller's signal; a null entry would need the same check
        plus a way to distinguish it from a fire with no fields."""
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            _, fires = _build_stores(pool)
            _, job = await _seed_job(pool)

            latest = await fires.latest_for_jobs([job])

            assert latest == {}
        finally:
            await pool.close()

    async def test_an_empty_request_makes_no_query(self, pg_schema: tuple[str, str]) -> None:
        url, schema = pg_schema
        pool = await _apply_schema(url, schema)
        try:
            _, fires = _build_stores(pool)
            assert await fires.latest_for_jobs([]) == {}
        finally:
            await pool.close()
