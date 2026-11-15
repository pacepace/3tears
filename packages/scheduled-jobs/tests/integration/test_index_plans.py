"""integration: each scheduled-jobs index serves the keyed read its purpose names.

The no-whole-reads rule (:mod:`threetears.enforcement.no_whole_reads`) keys a read by an index only when the index
names that read as its ``purpose``; these are the ``plan_test`` each names. The tables are built by the package's own
migrations and seeded past the size where the planner would rather scan; the statement the store really writes (run
over a pool that records statements) is asked whether its plan found its rows through the index by an index condition
(:func:`threetears.core.testing.plans.plan_uses_index`).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import pytest
from threetears.core.collections.delete_cascade import read_delete_cascade
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


def _node_types(node: Mapping[str, Any]) -> Iterator[str]:
    yield str(node.get("Node Type", ""))
    for child in node.get("Plans", ()):
        yield from _node_types(child)


def _sorted_rows(node: Mapping[str, Any]) -> Iterator[int]:
    """the rows each sort node under ``node`` put in order, a loop's worth each."""
    if node.get("Node Type") in {"Sort", "Incremental Sort"}:
        yield int(node.get("Actual Rows", 0))
    for child in node.get("Plans", ()):
        yield from _sorted_rows(child)


def _stores(captured: CapturedL3 | asyncpg.Connection) -> tuple[ScheduledJobCollection, JobFireCollection]:
    registry = CollectionRegistry()
    registry.configure(l3_pool=captured)
    cfg = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return (
        ScheduledJobCollection(registry=registry, config=cfg, nats_client=None),
        JobFireCollection(registry=registry, config=cfg, nats_client=None),
    )


async def test_scheduled_jobs_next_fire(conn: asyncpg.Connection) -> None:
    """Flips if the tick's due scan stops reading idx_scheduled_jobs_next_fire_key in its order (``next_fire_at``
    completed by ``job_id``) and sorts instead, or if on YugabyteDB a tick past every active job stops reading only
    its page (a hashed ``next_fire_at``, or an index without the key, reads and sorts every active one)."""
    captured = CapturedL3()
    jobs, _ = _stores(captured)
    await jobs.list_due_for_tick(_NOW, kinds=["k1"], limit=50)
    (sql, args), *_ = captured.asked
    plan = await explain_statement(conn, sql, *args)
    assert plan_uses_index(plan, "idx_scheduled_jobs_next_fire_key"), plan.plan
    # a tick after every one of the 500 active jobs (all of kind k0) is due: the page of 50 reads about its own rows
    late = CapturedL3()
    jobs, _ = _stores(late)
    await jobs.list_due_for_tick(datetime.now(UTC) + timedelta(days=30), kinds=["k0"], limit=50)
    (sql, args), *_ = late.asked
    page = await explain_statement(conn, sql, *args)
    # the page of 50 is read with keyset_page's one row past it (whether a next page exists)
    assert page.rows_returned == 50 + 1, page
    assert not set(_node_types(page.plan)) & {"Sort", "Incremental Sort"}, page.plan
    assert page.rows_scanned <= 2 * (50 + 1), page
    # the answer is the page itself, the row past it left out
    served, _ = _stores(conn)
    due = await served.list_due_for_tick(datetime.now(UTC) + timedelta(days=30), kinds=["k0"], limit=50)
    assert len(due) == 50
    if await on_yugabyte(conn):
        assert page.rows_scanned <= 2 * (50 + 1), page


async def test_scheduled_jobs_kind_name(conn: asyncpg.Connection) -> None:
    """Flips if the operator's listing (list_jobs) or its bounded count (count_jobs) stops walking
    idx_scheduled_jobs_kind_name_key in its order (``kind, name NULLS LAST, job_id``) and reads and sorts every
    schedule instead, or a page read from a cursor reads the rows before it (v007; the LIMIT/OFFSET listing it
    replaced read every schedule for every page)."""
    # names on most jobs, NULL on the rest, so the order's nullable column is exercised; a run of names tied
    await conn.execute(
        "UPDATE scheduled_jobs SET name = 'job ' || lpad((abs(hashtext(job_id::text)) % 300)::text, 3, '0')"
        " WHERE abs(hashtext(job_id::text)) % 7 <> 0"
    )
    await conn.execute("ANALYZE scheduled_jobs")
    sorts = {"Sort", "Incremental Sort"}

    async def _planned(read: Any) -> Any:
        captured = CapturedL3(real=conn)
        jobs, _ = _stores(captured)
        answered = await read(jobs)
        (sql, args), *_ = captured.asked
        return answered, await explain_statement(conn, sql, *args)

    # the first page, unfiltered and by kind: a page of 50 reads about its own rows (and the one past it)
    for kinds in (None, ["k1", "k3"]):
        page, plan = await _planned(lambda jobs, kinds=kinds: jobs.list_jobs(kinds=kinds, limit=50))
        assert len(page.jobs) == 50 and page.next_cursor is not None
        assert plan_uses_index(plan, "idx_scheduled_jobs_kind_name_key"), plan.plan
        assert not set(_node_types(plan.plan)) & sorts, plan.plan
        assert plan.rows_scanned <= 2 * (50 + 1), plan
        # the page after it, from its cursor: the rows before the cursor are not read again
        later, after = await _planned(
            lambda jobs, kinds=kinds, cursor=page.next_cursor: jobs.list_jobs(kinds=kinds, limit=50, cursor=cursor)
        )
        assert len(later.jobs) == 50
        assert not {job.job_id for job in later.jobs} & {job.job_id for job in page.jobs}
        assert plan_uses_index(after, "idx_scheduled_jobs_kind_name_key"), after.plan
        # on YugabyteDB the page past a cursor is legs, each a range of the index in its order, and the legs' rows
        # (the page and the one past it, at most) are put in order at the end; nothing sorts more than that
        assert all(rows <= 50 + 1 for rows in _sorted_rows(after.plan)), after.plan
        assert after.rows_scanned <= 2 * (50 + 1), after
    # a status filter is checked on the index entry, in the index's order, never by a sort: the page reads the
    # entries until it holds 50 of the status (one job in ten is active here, so about ten entries a row kept)
    page, plan = await _planned(lambda jobs: jobs.list_jobs(statuses=["active"], limit=50))
    assert len(page.jobs) == 50 and {job.status for job in page.jobs} == {"active"}
    assert plan_uses_index(plan, "idx_scheduled_jobs_kind_name_key"), plan.plan
    assert not set(_node_types(plan.plan)) & sorts, plan.plan
    assert plan.rows_scanned <= 2 * 10 * (50 + 1), plan
    # the bounded count walks the same index and stops one past its cap, never counting all 5,000
    counted, plan = await _planned(lambda jobs: jobs.count_jobs(at_most=100))
    assert (counted.count, counted.more) == (100, True)
    assert plan_uses_index(plan, "idx_scheduled_jobs_kind_name_key"), plan.plan
    assert plan.rows_scanned <= 2 * (100 + 1), plan


async def test_job_fires_job_time(conn: asyncpg.Connection) -> None:
    """Flips if the latest fire of each listed job, one job's history page (newest first, completed by ``fire_id``),
    or the fires a job's delete is about to cascade (read_delete_cascade, DeleteCascade on ScheduledJobCollection)
    stop reading idx_job_fires_job_time_key, or the page reads and sorts the job's fires rather than walking the
    index in its order (v006: the index holds ``fire_id DESC`` after ``actual_fired_at DESC``)."""
    jobs = [row["job_id"] for row in await conn.fetch("SELECT job_id FROM scheduled_jobs LIMIT 20")]
    captured = CapturedL3()
    _, fires = _stores(captured)
    await fires.latest_for_jobs(jobs)
    # one keyed page of one row a job
    assert len(captured.asked) == len(jobs), captured.asked
    for sql, args in captured.asked:
        plan = await explain_statement(conn, sql, *args)
        assert plan_uses_index(plan, "idx_job_fires_job_time_key"), plan.plan
    # one job's history at a long-lived job's size (2,000 fires; at the seed's 4 PostgreSQL rightly ANDs two bitmaps
    # and sorts the 4), a page of 2
    job = await conn.fetchrow("SELECT partition_key, job_id FROM scheduled_jobs LIMIT 1")
    await conn.execute(
        "INSERT INTO job_fires (partition_key, fire_id, job_id, scheduled_fire_at, actual_fired_at, status)"
        " SELECT $1, gen_random_uuid(), $2, now(), now() - n * interval '1 minute', 'succeeded'"
        " FROM generate_series(1, 2000) n",
        job["partition_key"],
        job["job_id"],
    )
    await conn.execute("ANALYZE job_fires")
    history = CapturedL3(real=conn)
    _, fires = _stores(history)
    await fires.list_for_job(job["partition_key"], job["job_id"], limit=2)
    (sql, args), *_ = history.asked
    page = await explain_statement(conn, sql, *args)
    assert plan_uses_index(page, "idx_job_fires_job_time_key"), page.plan
    assert not set(_node_types(page.plan)) & {"Sort", "Incremental Sort"}, page.plan
    # the page of 2 with keyset_page's one row past it
    assert page.rows_scanned <= 2 + 1, page
    # the long-lived job's newest fire reads that fire and the one after it, never its 2,004 (the DISTINCT ON it
    # replaced read every fire of every listed job to keep one each)
    newest = CapturedL3(real=conn)
    _, fires = _stores(newest)
    latest = await fires.latest_for_jobs([job["job_id"]])
    assert list(latest) == [job["job_id"]], latest
    ((sql, args),) = [asked for asked in newest.asked if "job_fires" in asked[0]]
    page = await explain_statement(conn, sql, *args)
    assert plan_uses_index(page, "idx_job_fires_job_time_key"), page.plan
    assert not set(_node_types(page.plan)) & {"Sort", "Incremental Sort"}, page.plan
    assert page.rows_scanned <= 1 + 1, page
    # a job's delete reads the fires its ON DELETE CASCADE takes, by the job: the long-lived job's 2,004
    cascading = CapturedL3(real=conn)
    held, _ = _stores(cascading)
    found = await read_delete_cascade(held, [(job["partition_key"], job["job_id"])])
    assert [len(rows.keys) for rows in found] == [2004], found
    (sql, args), *_ = cascading.asked
    cascade = await explain_statement(conn, sql, *args)
    assert plan_uses_index(cascade, "idx_job_fires_job_time_key"), cascade.plan
    assert cascade.rows_scanned <= 2004, cascade


async def test_job_fires_dispatching(conn: asyncpg.Connection) -> None:
    """Flips if the reaper's sweep of the fires in flight stops reading idx_job_fires_dispatching."""
    captured = CapturedL3()
    _, fires = _stores(captured)
    await fires.reap_stale_dispatching(_NOW, older_than=timedelta(minutes=10), kinds=["k1"])
    (sql, args), *_ = captured.asked
    plan = await explain_write(conn, sql, *args)
    assert plan_uses_index(plan, "idx_job_fires_dispatching"), plan.plan
