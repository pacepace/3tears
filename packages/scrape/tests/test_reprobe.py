"""Tests for booking a blocked target's re-probe as a scheduled job.

The adapter is thin on purpose -- all the hard parts (cross-pod tick locking, missed-fire
policy, fire history) belong to ``3tears-scheduled-jobs`` and are tested there. What is
worth pinning here is the part that is this package's own judgement: that the job it writes
is a one-shot relative delay rather than a repeating schedule, and that re-booking a target
replaces its outstanding probe instead of queuing another one.
"""

from __future__ import annotations

import re
from typing import Any

import pytest
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.scheduled_jobs.collections import (
    _JOB_INSERT_COLUMNS,
    ScheduledJobCollection,
    _job_insert_params,
)
from threetears.scheduled_jobs.migrations.v001_create_scheduled_jobs import _CREATE_SCHEDULED_JOBS_SQL
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


def _not_null_job_columns() -> frozenset[str]:
    """Every ``scheduled_jobs`` column the migration declares ``NOT NULL``, read from the DDL.

    Derived rather than transcribed. A hand-copied list is a second copy of a schema that
    lives in another package, and the failure mode of a stale copy is the quiet one: a column
    added there with ``NOT NULL`` would simply not be checked here, which is precisely the
    class of bug this test exists to catch.

    The upsert binds every column positionally, so a server-side DEFAULT never applies -- a
    key this adapter forgets to set is bound as an explicit NULL and the constraint fires.

    Anchored, because a parser feeding ``assert not nulls`` fails open: an empty or shortened
    derivation is indistinguishable from a pass, which would move the quiet drift this exists
    to prevent from the copy into the parser. A multi-word type (``TIMESTAMP WITH TIME ZONE``,
    ``NUMERIC(10, 2)``), a wrapped ``NOT NULL``, or a DDL built by concatenation would each
    shrink the set silently. The anchors are three columns whose absence means the regex, not
    the schema, has changed.

    Reads three private names across a package boundary (this, plus ``_JOB_INSERT_COLUMNS``
    and ``_job_insert_params``). Accepted deliberately: the alternative is a second copy of
    another package's schema, and a stale copy fails silently where a renamed private symbol
    fails at import.
    """
    body = _CREATE_SCHEDULED_JOBS_SQL[_CREATE_SCHEDULED_JOBS_SQL.index("(") :]
    derived = frozenset(re.findall(r"^\s*([a-z_]+)\s+[A-Z]", body, re.MULTILINE)) & frozenset(
        re.findall(r"^\s*([a-z_]+)\s+\S+\s+NOT NULL", body, re.MULTILINE)
    )
    anchors = {"partition_key", "job_id", "date_updated"}
    assert anchors <= derived, (
        f"the NOT NULL derivation lost known columns {sorted(anchors - derived)}, so it would "
        f"pass vacuously -- the DDL's shape changed under the regex"
    )
    return derived


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
    nulls = sorted(col for col in _not_null_job_columns() if bound.get(col) is None)
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
async def test_cancelling_a_booking_that_is_not_there_neither_raises_nor_announces_one(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The caller closing a circuit does not know whether a booking was ever made.

    Asking first would be a round trip to answer what the delete already handles, and raising
    would turn ordinary cleanup into a failure on the path where a target just came back
    healthy. So the delete is issued blind.

    The consequence that needs pinning is the log. ``Collection.delete`` is idempotent and
    documents itself as returning ``True`` unconditionally, so the return value cannot tell a
    real cancellation from a no-op -- and ``record_reachable`` calls this on EVERY close. An
    INFO line here would therefore announce a cancelled re-probe for the many targets that
    never tripped at all, which is a log that lies at whatever volume the fleet polls.
    """
    jobs = _FakeJobCollection()
    with caplog.at_level("INFO", logger="threetears.scrape.reprobe"):
        await ScheduledJobsReprobeScheduler(jobs).cancel_reprobe(target_id="never_booked")

    assert jobs.saved == {}
    assert caplog.records == [], (
        "cancelling a booking that never existed announced a cancellation at INFO, which "
        "every close would then do for every target that never tripped"
    )
