"""Generic scheduled-jobs tick engine.

One pure-async :func:`scheduled_tick_job` invocation drives one pass of
the scheduler: acquire the cross-pod lock, enumerate due schedules via
the injected :class:`~threetears.scheduled_jobs.protocols.ScheduleStore`,
claim each via optimistic-CAS, insert the initial ``job_fires`` row via
the injected :class:`~threetears.scheduled_jobs.protocols.FireStore`, and
invoke the consumer-supplied dispatch callback. Consumers register this
body as a periodic job (e.g. an APScheduler ``IntervalTrigger``, cadence
~60s); the platform itself does NOT depend on APScheduler.

Generalized from :func:`threetears.agent.wake.tick.wake_tick_job`. The
agent/skill/webhook/conversation-specific machinery is stripped: the
engine takes the store(s) + the dispatch callback + the NATS client as
parameters and has zero domain knowledge. It builds a
:class:`~threetears.scheduled_jobs.types.JobTrigger` from the opaque
``kind`` + ``payload`` off each due row and forwards it verbatim.

design notes
------------

- **Pure-async, one tick per call.** No internal polling. The consumer's
  scheduler drives cadence.
- **Cross-pod lock at a caller-supplied key** (default
  :data:`~threetears.scheduled_jobs.config.DEFAULT_TICK_LOCK_KEY`). A held
  lock means another pod is already running this tick -- we return
  silently (debug log). ``KvError`` (lock infra gone) degrades open: the
  per-row optimistic-CAS in
  :meth:`ScheduleStore.claim_and_reschedule` is the real guard, so the
  tick body runs anyway rather than silencing the scheduler until a
  process restart.
- **Sequential per-schedule dispatch inside the lock.** Parallel
  ``asyncio.gather`` across schedules introduces write-contention on the
  fire store for no meaningful latency win (one tick handles tens of
  fires).
- **Per-schedule failure isolation.** Every dispatch is wrapped in
  ``try/except Exception`` -- one bad row never poisons the tick.
- **Dispatch callback is injected.** The engine only types the callable +
  invokes it; the consumer owns what a fire *does*.
- **Missed-fire policy + drift recording.** The tick records the planned
  fire instant (``scheduled_fire_at``) alongside the actual fire instant
  on every row; per-schedule ``missed_fire_policy`` controls whether a
  backlog coalesces into one fire or fires once per missed tick (via
  :func:`~threetears.scheduled_jobs.reschedule.compute_next_fire_at`).
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any, Final
from uuid import UUID

from threetears.observe import get_logger

from threetears.scheduled_jobs.config import DEFAULT_JOB_CONFIG, JobConfig
from threetears.scheduled_jobs.events import (
    EVENT_FIRE_DISPATCHED,
    EVENT_FIRE_DRIFT,
    EVENT_FIRE_FAILED,
    EVENT_FIRE_REAPED,
    EVENT_FIRE_SKIPPED_BUSY,
    EVENT_TICK_COMPLETED,
    EVENT_TICK_STARTED,
)
from threetears.scheduled_jobs.metrics import get_scheduled_jobs_emitter
from threetears.scheduled_jobs.protocols import DueSchedule, FireStore, ScheduleStore
from threetears.scheduled_jobs.reschedule import compute_next_fire_at
from threetears.scheduled_jobs.types import JobFireResult, JobTrigger

__all__ = ["DispatchCallback", "scheduled_tick_job"]


# Drift threshold for the dedicated drift-event log ("actual - scheduled
# > 60s"). The histogram in :mod:`threetears.scheduled_jobs.metrics`
# observes EVERY fire's drift; this constant only gates whether we ALSO
# emit a structured drift event.
_DRIFT_LOG_THRESHOLD_SECONDS: Final[float] = 60.0


# Terminal one-shot schedule types: this fire IS the single fire. The
# engine forces ``next_fire_at=None`` + ``status='expired'`` for these
# without consulting the reschedule helper (whose ``last_fired_at`` arg
# carries the PREVIOUS fire for catch_up semantics and so would not yet
# recognise this fire as terminal).
_ONE_SHOT_SCHEDULE_TYPES: Final[frozenset[str]] = frozenset({"one_shot_at", "relative_delay"})


log = get_logger(__name__)


# Type alias for the dispatch callback the tick body invokes per fire.
# Takes the trigger envelope + the freshly-minted ``fire_id``. Returns a
# ``JobFireResult`` so the tick can finalize the fire row. The consumer
# owns the body.
DispatchCallback = Callable[
    ["JobTrigger", UUID],
    Awaitable["JobFireResult"],
]


async def scheduled_tick_job(
    schedule_store: ScheduleStore,
    fire_store: FireStore,
    dispatch_callback: DispatchCallback,
    *,
    nats_client: Any = None,
    config: JobConfig = DEFAULT_JOB_CONFIG,
) -> None:
    """Run one tick pass of the scheduled-jobs scheduler.

    Acquires the cross-pod lock at ``config.tick_lock_key``; on hold,
    returns silently. Within the lock, enumerates due schedules, claims
    each via optimistic-CAS, writes the initial in-flight fire row, and
    invokes the dispatch callback. The callback is awaited inline;
    long-running callback bodies are expected to ``asyncio.create_task``
    internally so the tick returns as soon as the row is staged.

    ``nats_client`` is typed ``Any`` to keep the NATS client an optional
    runtime dep on the engine's interface; ``None`` skips lock acquisition
    for single-pod dev environments (the per-row CAS still guards).

    :param schedule_store: the schedule-side store (due-enumeration +
        claim-and-reschedule)
    :ptype schedule_store: ScheduleStore
    :param fire_store: the fire-side store (in-flight insert + finalize)
    :ptype fire_store: FireStore
    :param dispatch_callback: per-fire dispatcher; raised exceptions are
        isolated to a single schedule and recorded as failed fires
    :ptype dispatch_callback: DispatchCallback
    :param nats_client: :class:`threetears.nats.NatsClient` or ``None``
        for single-pod dev mode (no cross-pod lock)
    :ptype nats_client: Any
    :param config: operational config (lock key + due-scan cap); defaults
        to the platform baseline
    :ptype config: JobConfig
    :return: nothing
    :rtype: None
    """
    # local import to keep the lock module out of the engine's always-paid
    # import cost (consumers without NATS skip this).
    from threetears.nats import LockHeld, nats_distributed_lock  # noqa: PLC0415
    from threetears.nats.errors import KvError  # noqa: PLC0415

    try:
        async with nats_distributed_lock(nats_client, config.tick_lock_key):
            await _run_tick_body(schedule_store, fire_store, dispatch_callback, config)
    except LockHeld:
        log.debug(
            "scheduled_tick: lock held by another pod, skipping",
            extra={"extra_data": {"lock_key": config.tick_lock_key}},
        )
    except KvError as exc:
        # Lock INFRASTRUCTURE failed (bucket/stream gone, NATS unreachable)
        # -- distinct from LockHeld. The cross-pod lock is only a
        # redundant-work optimization: per-schedule mutual exclusion is the
        # optimistic-CAS in ScheduleStore.claim_and_reschedule, so the tick
        # body needs zero NATS to be correct. Degrade open rather than
        # silently dropping every tick until a process restart. Worst case:
        # every pod runs the due-scan and contends on the CAS, which is the
        # handled SKIPPED_BUSY path -- no double-fires, no data loss.
        log.warning(
            "scheduled_tick: cross-pod lock unavailable; proceeding without it (CAS still guards fires)",
            extra={"extra_data": {"error_type": type(exc).__name__, "error": str(exc)}},
        )
        await _run_tick_body(schedule_store, fire_store, dispatch_callback, config)


async def _run_tick_body(
    schedule_store: ScheduleStore,
    fire_store: FireStore,
    dispatch_callback: DispatchCallback,
    config: JobConfig,
) -> None:
    """Pump one tick's worth of fires through the dispatch callback."""
    emitter = get_scheduled_jobs_emitter()
    tick_started = time.monotonic()
    now = datetime.now(UTC)
    await _reap_stale_dispatching(fire_store, config, now, emitter)
    due = await schedule_store.list_due_for_tick(now=now, limit=config.tick_due_limit)
    log.info(
        EVENT_TICK_STARTED,
        extra={"extra_data": {"due_count": len(due), "tick_at": now.isoformat()}},
    )
    for schedule in due:
        try:
            await _dispatch_one(schedule, schedule_store, fire_store, dispatch_callback, now, emitter)
        except Exception:  # noqa: BLE001 - boundary: isolate per-schedule failures
            # _dispatch_one already records a failed job_fires row; this
            # outer except is defense in depth in case the row write itself
            # raises.
            log.exception(
                "scheduled_tick: schedule dispatch raised outside per-fire isolation",
                extra={"extra_data": {"job_id": str(schedule.job_id)}},
            )
            emitter.inc_failure(reason="handler_exception")
    duration = time.monotonic() - tick_started
    emitter.observe_tick_duration(duration)
    log.info(
        EVENT_TICK_COMPLETED,
        extra={"extra_data": {"processed": len(due), "duration_seconds": duration}},
    )


async def _reap_stale_dispatching(
    fire_store: FireStore,
    config: JobConfig,
    now: datetime,
    emitter: Any,
) -> None:
    """Reap fire rows abandoned mid-dispatch, once per tick.

    Delegates to :meth:`FireStore.reap_stale_dispatching` with the
    configured age threshold. A pod that died between the in-flight
    insert and a finalize leaves a permanent ``'dispatching'`` zombie;
    this surfaces the loss as a ``'failed'`` fire + a failure-metric
    increment. Wrapped in boundary isolation so a reaper failure (e.g. a
    transient DB hiccup) never blocks the tick's dispatch work.

    :param fire_store: the fire-side store
    :ptype fire_store: FireStore
    :param config: operational config (carries the reap-age threshold)
    :ptype config: JobConfig
    :param now: tick instant
    :ptype now: datetime
    :param emitter: metrics emitter
    :ptype emitter: Any
    :return: nothing
    :rtype: None
    """
    older_than = timedelta(seconds=config.dispatch_reap_after_seconds)
    try:
        reaped = await fire_store.reap_stale_dispatching(now, older_than=older_than)
    except Exception:  # noqa: BLE001 - boundary: a reaper failure must not block the tick
        log.exception(
            "scheduled_tick: reaper sweep raised; continuing with dispatch",
            extra={"extra_data": {"reap_after_seconds": config.dispatch_reap_after_seconds}},
        )
        return
    if reaped > 0:
        log.info(
            EVENT_FIRE_REAPED,
            extra={"extra_data": {"reaped_count": reaped, "reap_after_seconds": config.dispatch_reap_after_seconds}},
        )
        for _ in range(reaped):
            emitter.inc_failure(reason="reaped")


async def _dispatch_one(
    schedule: DueSchedule,
    schedule_store: ScheduleStore,
    fire_store: FireStore,
    dispatch_callback: DispatchCallback,
    tick_at: datetime,
    emitter: Any,
) -> None:
    """Claim one due schedule and dispatch its fire.

    On CAS miss (another tick already claimed) returns silently. On
    dispatch callback success, writes the terminal fire status. On
    dispatch exception (or a returned ``status='failed'``), writes a
    failed fire row + logs.
    """
    expected_next_fire = schedule.next_fire_at
    if expected_next_fire is None:
        # defense in depth: list_due_for_tick already filters NULL
        # next_fire_at, but if we ever loosen that the CAS UPDATE has no
        # anchor.
        log.warning(
            "scheduled_tick: schedule has NULL next_fire_at but appeared due; skipping",
            extra={"extra_data": {"job_id": str(schedule.job_id)}},
        )
        return

    if schedule.schedule_type in _ONE_SHOT_SCHEDULE_TYPES:
        computed_next_fire: datetime | None = None
    else:
        computed_next_fire = compute_next_fire_at(
            schedule_type=schedule.schedule_type,
            schedule_config=schedule.schedule_config,
            missed_fire_policy=schedule.missed_fire_policy,
            last_fired_at=schedule.last_fired_at,
            now=tick_at,
            # catch_up anchors on the occurrence being fired, NOT
            # last_fired_at (the store stamps that to ``now`` on claim, so
            # anchoring there collapses catch_up into coalesce).
            current_fire_at=expected_next_fire,
        )
    new_status = "expired" if computed_next_fire is None else "active"

    claimed = await schedule_store.claim_and_reschedule(
        partition_key=schedule.partition_key,
        job_id=schedule.job_id,
        expected_next_fire=expected_next_fire,
        computed_next_fire=computed_next_fire,
        new_status=new_status,
        now=tick_at,
    )
    if not claimed:
        # Per-schedule "another tick already grabbed this row" path -- the
        # CAS-miss IS the per-fire busy signal. The whole-tick cross-pod
        # LockHeld path is intentionally NOT a skipped-fire event: it skips
        # the entire tick body, logged at debug level by
        # :func:`scheduled_tick_job`. inc_failure(reason='claim_lost')
        # keeps a counter so operators can alert on persistent CAS-miss
        # bursts.
        log.info(
            EVENT_FIRE_SKIPPED_BUSY,
            extra={
                "extra_data": {
                    "job_id": str(schedule.job_id),
                    "partition_key": str(schedule.partition_key),
                    "schedule_type": schedule.schedule_type,
                    "reason": "claim_lost",
                }
            },
        )
        emitter.inc_failure(reason="claim_lost")
        return

    fire_id = _generate_fire_id()
    trigger = JobTrigger(
        job_id=schedule.job_id,
        partition_key=schedule.partition_key,
        kind=schedule.kind,
        schedule_type=schedule.schedule_type,
        fired_at=tick_at,
        scheduled_fire_at=expected_next_fire,
        payload=schedule.payload,
        name=schedule.name,
    )
    await fire_store.create_dispatching(
        fire_id=fire_id,
        job_id=schedule.job_id,
        partition_key=schedule.partition_key,
        scheduled_fire_at=expected_next_fire,
        actual_fired_at=tick_at,
    )

    # Drift observation: every fire's drift is recorded into the
    # histogram; the drift event is emitted only above the threshold so
    # the log doesn't fill with zero-drift noise.
    drift_seconds = max(0.0, (tick_at - expected_next_fire).total_seconds())
    emitter.observe_drift(drift_seconds)
    if drift_seconds > _DRIFT_LOG_THRESHOLD_SECONDS:
        log.info(
            EVENT_FIRE_DRIFT,
            extra={
                "extra_data": {
                    "job_id": str(schedule.job_id),
                    "partition_key": str(schedule.partition_key),
                    "fire_id": str(fire_id),  # convert at border: fire-drift log extra_data field
                    "scheduled_fire_at": expected_next_fire.isoformat(),
                    "actual_fired_at": tick_at.isoformat(),
                    "drift_seconds": drift_seconds,
                    "schedule_type": schedule.schedule_type,
                }
            },
        )

    log.info(
        EVENT_FIRE_DISPATCHED,
        extra={
            "extra_data": {
                "job_id": str(schedule.job_id),
                "fire_id": str(fire_id),  # convert at border: fire-dispatched log extra_data field
                "partition_key": str(schedule.partition_key),
                "kind": schedule.kind,
                "schedule_type": schedule.schedule_type,
                "missed_fire_policy": schedule.missed_fire_policy,
            }
        },
    )

    try:
        result = await dispatch_callback(trigger, fire_id)
    except Exception as exc:  # noqa: BLE001 - boundary: per-fire isolation
        log.exception(
            EVENT_FIRE_FAILED,
            extra={
                "extra_data": {
                    "job_id": str(schedule.job_id),
                    "fire_id": str(fire_id),  # convert at border: fire-failed log extra_data field
                    "partition_key": str(schedule.partition_key),
                    "schedule_type": schedule.schedule_type,
                    "error_type": type(exc).__name__,
                }
            },
        )
        await fire_store.finalize_failed(
            schedule.partition_key,
            fire_id,
            error=str(exc),
            latency_ms=None,
        )
        emitter.inc_fire(status="failed", schedule_type=schedule.schedule_type)
        emitter.inc_failure(reason="handler_exception")
        return

    # A dispatch callback may return ``JobFireResult(status='failed',
    # error='...')`` without raising -- e.g. a handler recording a
    # non-exceptional failure with its own error string. Route that to
    # finalize_failed so the ``error`` field is not dropped on the floor.
    if result.status == "failed":
        await fire_store.finalize_failed(
            schedule.partition_key,
            fire_id,
            error=result.error or "dispatch returned status='failed' without an error string",
            latency_ms=result.latency_ms,
        )
        emitter.inc_fire(status="failed", schedule_type=schedule.schedule_type)
        emitter.inc_failure(reason="handler_exception")
        return

    await fire_store.finalize_success(
        schedule.partition_key,
        fire_id,
        status=result.status,
        output=result.output,
        latency_ms=result.latency_ms,
    )
    emitter.inc_fire(status=result.status, schedule_type=schedule.schedule_type)


def _generate_fire_id() -> UUID:
    """Mint a fresh ``fire_id``.

    UUIDv7 keeps lexicographic ordering for "list fires since" cursor
    queries (the canonical fire-history read pattern).
    """
    from uuid_utils import uuid7  # noqa: PLC0415

    return UUID(str(uuid7()))
