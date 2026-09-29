"""Integration test: :class:`BackgroundDispatch` against REAL Postgres fire rows and a REAL NATS lock.

The unit suite drives ``BackgroundDispatch`` over a recording fire store and a monkeypatched lock.
Two of its promises only mean something against the real things:

- the tick returns while a slow fire is still running -- its ``job_fires`` row is still
  ``'dispatching'`` -- and that fire later finalizes its own row with its real outcome, while a
  fast kind in the same tick has already finished (real ``scheduled_tick_job``, real
  ``JobFireCollection``, real Postgres);
- the same kind handed to two pods at once runs on one of them: the other's fire is recorded as
  skipped, in flight on another pod (two real NATS connections contending for the real
  ``nats_distributed_lock`` key);
- two kinds in one exclusion group, due in the same real tick, run one after the other: the second
  row stays ``'dispatching'`` while the first body runs, then finalizes with its own result, and a
  kind outside the group finishes meanwhile (real tick, real rows, real NATS in-flight locks).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import asyncpg
import pytest
from uuid_utils import uuid7

from threetears.core.collections.asyncpg_init import init_connection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner
from threetears.nats import NatsClient, set_default_namespace
from threetears.scheduled_jobs import (
    DEFAULT_JOB_CONFIG,
    IN_FLIGHT_SKIP_OUTPUT_KEY,
    BackgroundDispatch,
    JobFireCollection,
    JobFireResult,
    JobTrigger,
    ScheduledJobCollection,
    scheduled_tick_job,
)
from threetears.scheduled_jobs.migrations import register as register_scheduled_jobs

from .conftest import AsyncpgStore

pytestmark = pytest.mark.integration


def _new_uuid() -> UUID:
    return UUID(str(uuid7()))


async def _apply_schema(url: str, schema: str) -> asyncpg.Pool:
    setup_conn = await asyncpg.connect(url)
    try:
        await setup_conn.execute(f'SET search_path TO "{schema}", public')
        runner = MigrationRunner()
        register_scheduled_jobs(runner)
        await runner.apply_for_platform_schema(AsyncpgStore(setup_conn))  # type: ignore[arg-type]
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


def _stores(pool: asyncpg.Pool) -> tuple[ScheduledJobCollection, JobFireCollection]:
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    cfg = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return (
        ScheduledJobCollection(registry=registry, config=cfg, nats_client=None),
        JobFireCollection(registry=registry, config=cfg, nats_client=None),
    )


async def _seed_job(pool: asyncpg.Pool, kind: str, *, next_fire_at: datetime) -> tuple[UUID, UUID]:
    partition, job = _new_uuid(), _new_uuid()
    await pool.execute(
        "INSERT INTO scheduled_jobs "
        "(partition_key, job_id, kind, payload, schedule_type, "
        " schedule_config, status, next_fire_at, missed_fire_policy) "
        "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
        partition,
        job,
        kind,
        {},
        "interval",
        {"seconds": 60},
        "active",
        next_fire_at,
        "coalesce",
    )
    return partition, job


async def _fire_row(pool: asyncpg.Pool, job_id: UUID) -> dict[str, Any]:
    row = await pool.fetchrow(
        "SELECT status, output, latency_ms, error FROM job_fires WHERE job_id = $1 ORDER BY actual_fired_at DESC LIMIT 1",
        job_id,
    )
    assert row is not None, "the tick staged no fire row"
    return dict(row)


@pytest.fixture
async def pool(pg_schema: tuple[str, str]) -> AsyncIterator[asyncpg.Pool]:
    url, schema = pg_schema
    created = await _apply_schema(url, schema)
    try:
        yield created
    finally:
        await created.close()


async def test_a_slow_fire_does_not_hold_up_the_tick_and_finalizes_its_own_row(pool: asyncpg.Pool) -> None:
    schedule_store, fire_store = _stores(pool)
    due = datetime.now(UTC) - timedelta(seconds=5)
    _, slow_job = await _seed_job(pool, "slow", next_fire_at=due)
    _, fast_job = await _seed_job(pool, "fast", next_fire_at=due)

    release = asyncio.Event()

    async def _slow(_t: JobTrigger, _f: UUID) -> JobFireResult:
        await release.wait()
        return JobFireResult(output={"body": "slow"})

    async def _fast(_t: JobTrigger, _f: UUID) -> JobFireResult:
        return JobFireResult(output={"body": "fast"})

    background = BackgroundDispatch(fire_store, config=DEFAULT_JOB_CONFIG)
    routes = {"slow": background.wrap(_slow), "fast": background.wrap(_fast)}
    try:
        await asyncio.wait_for(scheduled_tick_job(schedule_store, fire_store, routes), timeout=10)

        # The tick is back while the slow body is still waiting on the event.
        assert (await _fire_row(pool, slow_job))["status"] == "dispatching"
        for _ in range(100):
            if (await _fire_row(pool, fast_job))["status"] != "dispatching":
                break
            await asyncio.sleep(0.05)
        fast = await _fire_row(pool, fast_job)
        assert fast["status"] == "succeeded"
        assert fast["output"] == {"body": "fast"}

        release.set()
        await asyncio.wait_for(background.join(), timeout=10)
        slow = await _fire_row(pool, slow_job)
        assert slow["status"] == "succeeded"
        assert slow["output"] == {"body": "slow"}
        assert slow["latency_ms"] is not None
    finally:
        await background.aclose()


@pytest.fixture
async def two_pods(nats_container: str) -> AsyncIterator[tuple[NatsClient, NatsClient]]:
    set_default_namespace("3tears")
    first = await NatsClient.connect(
        nats_url=nats_container, nats_subject_namespace="3tears", client_name="background-pod-a"
    )
    second = await NatsClient.connect(
        nats_url=nats_container, nats_subject_namespace="3tears", client_name="background-pod-b"
    )
    try:
        yield (first, second)
    finally:
        await first.shutdown()
        await second.shutdown()


async def test_the_same_kind_handed_to_two_pods_runs_on_one(
    pool: asyncpg.Pool, two_pods: tuple[NatsClient, NatsClient]
) -> None:
    _, fire_store = _stores(pool)
    now = datetime.now(UTC)
    partition, job = await _seed_job(pool, "shared", next_fire_at=now + timedelta(hours=1))
    first_fire, second_fire = _new_uuid(), _new_uuid()
    for fire_id in (first_fire, second_fire):
        await fire_store.create_dispatching(
            fire_id=fire_id, job_id=job, partition_key=partition, scheduled_fire_at=now, actual_fired_at=now
        )
    trigger = JobTrigger(
        job_id=job,
        partition_key=partition,
        kind="shared",
        schedule_type="interval",
        fired_at=now,
        scheduled_fire_at=now,
    )

    release = asyncio.Event()
    runs: list[str] = []

    async def _body(_t: JobTrigger, fire_id: UUID) -> JobFireResult:
        runs.append(str(fire_id))  # convert at border: a test-local record of which fire ran
        await release.wait()
        return JobFireResult(output={"ran": True})

    pod_a = BackgroundDispatch(fire_store, config=DEFAULT_JOB_CONFIG, nats_client=two_pods[0])
    pod_b = BackgroundDispatch(fire_store, config=DEFAULT_JOB_CONFIG, nats_client=two_pods[1])
    try:
        assert (await pod_a.wrap(_body)(trigger, first_fire)).handed_off is True
        for _ in range(100):  # let pod A take the in-flight lock before pod B tries
            if runs:
                break
            await asyncio.sleep(0.05)
        assert (await pod_b.wrap(_body)(trigger, second_fire)).handed_off is True
        await asyncio.wait_for(pod_b.join(), timeout=10)

        rows = {
            r["fire_id"]: dict(r)
            for r in await pool.fetch("SELECT fire_id, status, output FROM job_fires WHERE job_id = $1", job)
        }
        assert rows[second_fire]["status"] == "succeeded"
        assert rows[second_fire]["output"][IN_FLIGHT_SKIP_OUTPUT_KEY] == "fire in flight on another pod"

        release.set()
        await asyncio.wait_for(pod_a.join(), timeout=10)
        first = await pool.fetchrow("SELECT status, output FROM job_fires WHERE fire_id = $1", first_fire)
        assert first is not None
        assert first["status"] == "succeeded"
        assert first["output"] == {"ran": True}
        assert runs == [str(first_fire)]  # convert at border: compared against the test-local record
    finally:
        await pod_a.aclose()
        await pod_b.aclose()


async def test_kinds_in_one_exclusion_group_due_in_one_tick_take_turns(
    pool: asyncpg.Pool, two_pods: tuple[NatsClient, NatsClient]
) -> None:
    schedule_store, fire_store = _stores(pool)
    due = datetime.now(UTC) - timedelta(seconds=5)
    jobs = {kind: (await _seed_job(pool, kind, next_fire_at=due))[1] for kind in ("poll:g", "backfill:g", "poll:free")}

    started = {kind: asyncio.Event() for kind in jobs}
    release = {kind: asyncio.Event() for kind in jobs}
    order: list[str] = []

    async def _body(trigger: JobTrigger, _f: UUID) -> JobFireResult:
        order.append(f"start {trigger.kind}")
        started[trigger.kind].set()
        await release[trigger.kind].wait()
        order.append(f"end {trigger.kind}")
        return JobFireResult(output={"body": trigger.kind})

    background = BackgroundDispatch(
        fire_store,
        config=DEFAULT_JOB_CONFIG,
        nats_client=two_pods[0],
        exclusion_groups={"poll:g": "g", "backfill:g": "g"},
    )
    routes = {kind: background.wrap(_body) for kind in jobs}
    release["poll:free"].set()
    try:
        await asyncio.wait_for(
            scheduled_tick_job(schedule_store, fire_store, routes, nats_client=two_pods[0]), timeout=10
        )
        await asyncio.wait_for(started["poll:free"].wait(), timeout=10)
        first = await _first_started(started, ("poll:g", "backfill:g"))
        second = "backfill:g" if first == "poll:g" else "poll:g"
        for _ in range(100):
            if (await _fire_row(pool, jobs["poll:free"]))["status"] != "dispatching":
                break
            await asyncio.sleep(0.05)
        assert (await _fire_row(pool, jobs["poll:free"]))["status"] == "succeeded"
        await asyncio.sleep(0.2)
        assert not started[second].is_set(), "both kinds of the group ran at once"
        assert (await _fire_row(pool, jobs[second]))["status"] == "dispatching"

        release[first].set()
        await asyncio.wait_for(started[second].wait(), timeout=10)
        release[second].set()
        await asyncio.wait_for(background.join(), timeout=10)

        assert order.index(f"end {first}") < order.index(f"start {second}")
        for kind, job in jobs.items():
            row = await _fire_row(pool, job)
            assert (row["status"], row["output"]) == ("succeeded", {"body": kind})
    finally:
        await background.aclose()


async def _first_started(started: dict[str, asyncio.Event], kinds: tuple[str, ...]) -> str:
    """Wait until one of *kinds* has started and return it."""
    waits = {asyncio.ensure_future(started[kind].wait()): kind for kind in kinds}
    done, pending = await asyncio.wait(waits, timeout=10, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    assert done, f"none of {kinds} started"
    return waits[next(iter(done))]
