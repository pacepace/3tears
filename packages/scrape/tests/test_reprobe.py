"""Tests for booking a blocked target's re-probe as a scheduled job.

The adapter is thin on purpose -- all the hard parts (cross-pod tick locking, missed-fire
policy, fire history) belong to ``3tears-scheduled-jobs`` and are tested there. What is
worth pinning here is the part that is this package's own judgement: that the job it writes
is a one-shot relative delay rather than a repeating schedule, and that re-booking a target
replaces its outstanding probe instead of queuing another one.
"""

from __future__ import annotations

from typing import Any

import pytest
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.scheduled_jobs.collections import ScheduledJobCollection
from threetears.scheduled_jobs.entities import ScheduledJobEntity

from threetears.scrape.reprobe import REPROBE_JOB_KIND, ScheduledJobsReprobeScheduler, reprobe_job_id


class _FakeJobCollection(ScheduledJobCollection):
    """The real collection with its one write intercepted, keyed by the real composite pk.

    A subclass rather than a stand-in because the entity constructor calls back into its
    collection, so anything less than the real class is testing a different object than
    production hands this adapter.
    """

    def __init__(self) -> None:
        super().__init__(CollectionRegistry(), DefaultCoreConfig(collection_flush="ALWAYS"), nats_client=None)
        self.saved: dict[tuple[Any, Any], dict[str, Any]] = {}

    async def save_entity(self, entity: Any, **kwargs: Any) -> Any:
        row = dict(entity.to_dict())
        self.saved[(row["partition_key"], row["job_id"])] = row
        return entity


@pytest.mark.asyncio
async def test_the_booked_job_is_a_one_shot_relative_delay() -> None:
    """A repeating schedule would keep probing a wall on a fixed cadence forever.

    The backoff's whole shape is that each probe buys a LONGER silence than the last, and
    the next delay is only known after that probe's outcome. A one-shot is the only schedule
    that lets the outcome decide.
    """
    jobs = _FakeJobCollection()
    await ScheduledJobsReprobeScheduler(jobs).schedule_reprobe(target_id="warn_oh", delay_seconds=3600.0)

    (row,) = jobs.saved.values()
    assert row["schedule_type"] == "relative_delay"
    assert row["schedule_config"] == {"delay": "3600s"}
    assert row["kind"] == REPROBE_JOB_KIND
    assert row["payload"] == {"target_id": "warn_oh"}
    assert row["status"] == "active"
    assert row["missed_fire_policy"] == "coalesce"


@pytest.mark.asyncio
async def test_re_booking_a_target_replaces_its_outstanding_probe() -> None:
    """A random job id would turn the longest backoff into the biggest burst.

    A walled target is re-booked on every failed probe. With a fresh id each time, every
    superseded booking survives and eventually fires, so a target walled for a week arrives
    at the end of its longest silence with a week's worth of queued probes.
    """
    jobs = _FakeJobCollection()
    scheduler = ScheduledJobsReprobeScheduler(jobs)
    await scheduler.schedule_reprobe(target_id="warn_oh", delay_seconds=900.0)
    await scheduler.schedule_reprobe(target_id="warn_oh", delay_seconds=1800.0)

    assert len(jobs.saved) == 1
    (row,) = jobs.saved.values()
    assert row["job_id"] == reprobe_job_id("warn_oh")
    assert row["schedule_config"] == {"delay": "1800s"}


@pytest.mark.asyncio
async def test_two_targets_get_two_jobs() -> None:
    """The determinism is per target, not global."""
    jobs = _FakeJobCollection()
    scheduler = ScheduledJobsReprobeScheduler(jobs)
    await scheduler.schedule_reprobe(target_id="warn_oh", delay_seconds=900.0)
    await scheduler.schedule_reprobe(target_id="warn_ny", delay_seconds=900.0)
    assert len(jobs.saved) == 2


@pytest.mark.asyncio
async def test_a_sub_second_delay_is_floored_rather_than_rounded_to_zero() -> None:
    """``relative_delay`` config is an integer with a unit, so ``"0s"`` fires immediately.

    A backoff that fires immediately is not a backoff. Rounding is the only way to reach
    zero here, so the floor is the guard.
    """
    jobs = _FakeJobCollection()
    await ScheduledJobsReprobeScheduler(jobs).schedule_reprobe(target_id="warn_oh", delay_seconds=0.2)
    (row,) = jobs.saved.values()
    assert row["schedule_config"] == {"delay": "1s"}


@pytest.mark.asyncio
async def test_an_explicit_partition_key_is_honoured() -> None:
    """A deployment that already owns a partition should be able to keep its jobs in it."""
    from uuid import uuid4

    partition = uuid4()
    jobs = _FakeJobCollection()
    await ScheduledJobsReprobeScheduler(jobs, partition_key=partition).schedule_reprobe(
        target_id="warn_oh", delay_seconds=60.0
    )
    (row,) = jobs.saved.values()
    assert row["partition_key"] == partition
