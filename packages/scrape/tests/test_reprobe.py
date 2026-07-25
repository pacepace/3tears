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
from threetears.scheduled_jobs.collections import (
    _JOB_INSERT_COLUMNS,
    ScheduledJobCollection,
    _job_insert_params,
)
from threetears.scheduled_jobs.entities import ScheduledJobEntity

from threetears.scrape.reprobe import REPROBE_JOB_KIND, ScheduledJobsReprobeScheduler, reprobe_job_id


class _FakeJobCollection(ScheduledJobCollection):
    """The real collection with only its L3 write intercepted, keyed by the real composite pk.

    A subclass rather than a stand-in because the entity constructor calls back into its
    collection, so anything less than the real class is testing a different object than
    production hands this adapter.

    Intercepts ``save_to_store`` rather than ``save_entity`` deliberately. ``save_entity`` is
    where the framework stamps ``date_created`` and ``date_updated``, and stubbing it out
    skips that -- which is how a row missing a ``NOT NULL`` column passed every test in this
    file while failing at the real border. Cutting in one layer lower means the captured row
    is the one that would actually have been bound.
    """

    def __init__(self) -> None:
        super().__init__(CollectionRegistry(), DefaultCoreConfig(collection_flush="ALWAYS"), nats_client=None)
        self.saved: dict[tuple[Any, Any], dict[str, Any]] = {}

    async def save_to_store(self, data: dict[str, Any], original_timestamp: Any = None, *, conn: Any = None) -> int:
        del original_timestamp, conn
        row = dict(data)
        self.saved[(row["partition_key"], row["job_id"])] = row
        return 1

    async def delete_from_store(self, entity_id: Any) -> None:
        self.saved.pop(self.normalize_pk(entity_id), None)


#: Every ``scheduled_jobs`` column declared ``NOT NULL`` in v001. The upsert binds all of
#: them positionally, so a server-side DEFAULT never applies -- a key this adapter forgets
#: to set is bound as an explicit NULL and the constraint fires at the border.
_NOT_NULL_JOB_COLUMNS = frozenset(
    {
        "partition_key",
        "job_id",
        "kind",
        "payload",
        "schedule_type",
        "schedule_config",
        "status",
        "missed_fire_policy",
        "date_created",
        "date_updated",
    }
)


@pytest.mark.asyncio
async def test_the_booked_row_binds_a_value_for_every_not_null_column() -> None:
    """The bind path, which every other test in this file goes around.

    Nothing else in this file reaches the projection that turns a row dict into the upsert's
    positional params, and that is exactly where a hand-built row gets judged. It is also why
    ``_FakeJobCollection`` cuts in at ``save_to_store`` rather than ``save_entity``: the
    latter is where the stamping below happens, so intercepting it skips the very step the
    row has to survive. ``save_entity`` stamps ``date_created`` for a
    new entity but stamps ``date_updated`` only when the key is already present or the entity
    is not new, so this adapter must supply it. The column is ``NOT NULL DEFAULT now()`` and
    the default cannot rescue it: the upsert writes every column unconditionally, so a
    missing key binds an explicit NULL.

    The failure this pins is silent, not loud. ``TargetCircuit._book_reprobe`` catches and
    logs, so a constraint violation here does not fail a fetch -- it just means an
    event-driven deployment books no re-probes at all, which is the entire purpose of the
    ``[reprobe]`` extra, and the health row's ``blocked_until`` goes on looking correct.
    """
    jobs = _FakeJobCollection()
    await ScheduledJobsReprobeScheduler(jobs).schedule_reprobe(target_id="warn_oh", delay_seconds=900.0)
    (row,) = jobs.saved.values()

    bound = dict(zip(_JOB_INSERT_COLUMNS, _job_insert_params(row), strict=True))
    nulls = sorted(col for col in _NOT_NULL_JOB_COLUMNS if bound.get(col) is None)
    assert not nulls, f"NOT NULL column(s) bound as NULL, so every booking raises at the border: {nulls}"


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


@pytest.mark.asyncio
async def test_cancelling_deletes_the_booking_rather_than_leaving_it_expired() -> None:
    """A recovered target should leave nothing behind, and the job id makes that possible.

    The booking's id is derived from the target, so a cancel addresses whichever probe is
    outstanding without the caller having kept a handle on it. Deleting rather than expiring,
    because once the target has recovered the row carries no information the health row does
    not already hold -- and expiring in place would leave one row per target that ever
    tripped, which is the retention story this adapter would otherwise have none of.
    """
    jobs = _FakeJobCollection()
    scheduler = ScheduledJobsReprobeScheduler(jobs)
    await scheduler.schedule_reprobe(target_id="warn_oh", delay_seconds=900.0)
    assert len(jobs.saved) == 1

    await scheduler.cancel_reprobe(target_id="warn_oh")

    assert jobs.saved == {}, "the recovered target kept a booking that will fire on it"


@pytest.mark.asyncio
async def test_cancelling_a_booking_that_is_not_there_is_silent() -> None:
    """The caller closing a circuit does not know whether a booking was ever made.

    Asking first would be a round trip to answer a question the delete already answers, and
    raising would turn ordinary cleanup into a failure on the path where a target just came
    back healthy.
    """
    jobs = _FakeJobCollection()
    await ScheduledJobsReprobeScheduler(jobs).cancel_reprobe(target_id="never_booked")
    assert jobs.saved == {}
