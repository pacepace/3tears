"""integration: each scheduled-jobs index serves the keyed read its purpose names.

The no-whole-reads rule (:mod:`threetears.enforcement.no_whole_reads`) keys a read by an index only when the index
names that read as its ``purpose``; these are the ``plan_test`` each names. The tables are built by the package's own
migrations and seeded past the size where the planner would rather scan; the statement the store really writes (run
over a pool that records statements) is asked whether its plan found its rows through the index by an index condition
(:func:`threetears.core.testing.plans.plan_uses_index`).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import asyncpg
import pytest
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.testing.captured import CapturedL3
from threetears.core.testing.plans import explain_statement, explain_write, on_yugabyte, plan_uses_index
from threetears.scheduled_jobs.collections import JobFireCollection, ScheduledJobCollection

from .conftest import apply_schema

pytestmark = pytest.mark.integration

_NOW = datetime(2026, 11, 9, tzinfo=UTC)


@pytest.fixture
async def conn(pg_schema: tuple[str, str]) -> AsyncIterator[asyncpg.Connection]:
    """A connection on a fresh schema with the package's tables, seeded: 5000 jobs (one in ten active and due) and
    four fires each, the newest of an active job still in flight."""
    url, schema = pg_schema
    pool = await apply_schema(url, schema)
    await pool.close()
    connection = await asyncpg.connect(url)
    await connection.execute(f'SET search_path TO "{schema}", public')
    await connection.execute(
        "INSERT INTO scheduled_jobs (partition_key, job_id, kind, schedule_type, status, next_fire_at)"
        " SELECT gen_random_uuid(), gen_random_uuid(), 'k' || g % 5, 'interval',"
        " CASE WHEN g % 10 = 0 THEN 'active' ELSE 'expired' END,"
        " CASE WHEN g % 10 = 0 THEN now() + g * interval '1 minute' END FROM generate_series(1, 5000) g"
    )
    await connection.execute(
        "INSERT INTO job_fires (partition_key, fire_id, job_id, scheduled_fire_at, actual_fired_at, status)"
        " SELECT j.partition_key, gen_random_uuid(), j.job_id, now(), now() - n * interval '1 hour',"
        " CASE WHEN n = 1 AND j.status = 'active' THEN 'dispatching' ELSE 'succeeded' END"
        " FROM scheduled_jobs j, generate_series(1, 4) n"
    )
    for table in ("scheduled_jobs", "job_fires"):
        await connection.execute(f"ANALYZE {table}")
    try:
        yield connection
    finally:
        await connection.close()


def _stores(captured: CapturedL3) -> tuple[ScheduledJobCollection, JobFireCollection]:
    registry = CollectionRegistry()
    registry.configure(l3_pool=captured)
    cfg = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return (
        ScheduledJobCollection(registry=registry, config=cfg, nats_client=None),
        JobFireCollection(registry=registry, config=cfg, nats_client=None),
    )


async def test_scheduled_jobs_next_fire(conn: asyncpg.Connection) -> None:
    """Flips if the tick's due scan stops reading idx_scheduled_jobs_next_fire_asc, or if on YugabyteDB a tick past
    every active job stops reading only its page (a hashed ``next_fire_at`` reads and sorts every active one)."""
    captured = CapturedL3()
    jobs, _ = _stores(captured)
    await jobs.list_due_for_tick(_NOW, kinds=["k1"], limit=50)
    (sql, args), *_ = captured.asked
    plan = await explain_statement(conn, sql, *args)
    assert plan_uses_index(plan, "idx_scheduled_jobs_next_fire_asc"), plan.plan
    # a tick after every one of the 500 active jobs (all of kind k0) is due: the page of 50 reads about its own rows
    late = CapturedL3()
    jobs, _ = _stores(late)
    await jobs.list_due_for_tick(datetime.now(UTC) + timedelta(days=30), kinds=["k0"], limit=50)
    (sql, args), *_ = late.asked
    page = await explain_statement(conn, sql, *args)
    assert page.rows_returned == 50, page
    if await on_yugabyte(conn):
        assert page.rows_scanned <= 2 * (50 + 1), page


async def test_job_fires_job_time(conn: asyncpg.Connection) -> None:
    """Flips if the latest fire of each listed job stops reading idx_job_fires_job_time."""
    jobs = [row["job_id"] for row in await conn.fetch("SELECT job_id FROM scheduled_jobs LIMIT 20")]
    captured = CapturedL3()
    _, fires = _stores(captured)
    await fires.latest_for_jobs(jobs)
    (sql, args), *_ = captured.asked
    plan = await explain_statement(conn, sql, *args)
    assert plan_uses_index(plan, "idx_job_fires_job_time"), plan.plan


async def test_job_fires_dispatching(conn: asyncpg.Connection) -> None:
    """Flips if the reaper's sweep of the fires in flight stops reading idx_job_fires_dispatching."""
    captured = CapturedL3()
    _, fires = _stores(captured)
    await fires.reap_stale_dispatching(_NOW, older_than=timedelta(minutes=10), kinds=["k1"])
    (sql, args), *_ = captured.asked
    plan = await explain_write(conn, sql, *args)
    assert plan_uses_index(plan, "idx_job_fires_dispatching"), plan.plan
