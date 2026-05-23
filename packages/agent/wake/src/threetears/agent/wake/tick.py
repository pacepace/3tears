"""Wake tick engine.

One pure-async ``wake_tick_job`` invocation drives one pass of the
wake scheduler: acquire the cross-pod lock, enumerate due schedules,
claim each via optimistic-CAS, insert the initial ``wake_fires`` row,
and invoke the consumer-supplied dispatch callback. Consumers
register this body as an APScheduler ``IntervalTrigger`` job (cadence
~ 60s in metallm); the platform itself does NOT depend on
APScheduler.

design notes
------------

- **Pure-async, one tick per call.** No internal polling. The
  consumer's APScheduler IntervalTrigger drives cadence.
- **Cross-pod lock at ``"agent_wake_tick"``.** A held lock means
  another pod is already running this tick -- we return silently
  (debug log) rather than queuing or contending.
- **Sequential per-schedule dispatch inside the lock.** See
  PLACEMENT §1.3 anti-pattern: parallel ``asyncio.gather`` across
  schedules introduces write-contention on ``wake_fires`` for no
  meaningful latency win (one tick handles tens of fires).
- **Per-schedule failure isolation.** Every dispatch is wrapped in
  ``try/except Exception`` -- one bad row never poisons the tick.
- **Dispatch callback is injected.** Shard 02 only types the callable
  + invokes it. The real ``dispatch_wake`` lives in shard 03; this
  shard ships with a thin stub-shaped contract so the tick body can
  be exercised end-to-end with a test callback.
- **Missed-fire policy + drift recording** per PLACEMENT §1.7 / §1.8:
  the tick records the originally-scheduled fire instant
  (``scheduled_fire_at``) alongside the actual fire instant
  (``actual_fired_at``) on every row. Per-schedule
  ``missed_fire_policy`` controls whether a backlog coalesces into
  one fire or fires once per missed tick.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, Final
from uuid import UUID

from threetears.observe import get_logger

from threetears.agent.wake.collections import (
    WakeFireCollection,
    WakeScheduleCollection,
)
from threetears.agent.wake.entities import WakeScheduleEntity
from threetears.agent.wake.events import (
    EVENT_FIRE_DISPATCHED,
    EVENT_FIRE_DRIFT,
    EVENT_FIRE_FAILED,
    EVENT_TICK_COMPLETED,
    EVENT_TICK_STARTED,
)
from threetears.agent.wake.metrics import get_wake_emitter
from threetears.agent.wake.reschedule import _compute_next_fire_at
from threetears.agent.wake.types import WakeDispatchResult, WakeTrigger

__all__ = ["DispatchCallback", "wake_tick_job"]


# Drift threshold for the dedicated drift-event log (PLACEMENT §1.8 +
# spec body's "actual - scheduled > 60s" rule). The histogram in
# :mod:`threetears.agent.wake.metrics` observes EVERY fire's drift;
# this constant only gates whether we ALSO emit a Loki event.
_DRIFT_LOG_THRESHOLD_SECONDS: Final[float] = 60.0


log = get_logger(__name__)


_LOCK_KEY: Final[str] = "agent_wake_tick"


# Type alias for the dispatch callback the tick body invokes per fire.
# Takes the trigger envelope + the freshly-minted ``fire_id`` + the
# asyncpg pool the dispatcher will use to read/write related rows.
# Returns a ``WakeDispatchResult`` so the tick can finalize the
# ``wake_fires`` row. The real producer lives in shard 03; shard 02
# only types the callable.
DispatchCallback = Callable[
    ["WakeTrigger", UUID, Any],
    Awaitable["WakeDispatchResult"],
]


async def wake_tick_job(
    pool: Any,
    nats_client: Any,
    dispatch_callback: DispatchCallback,
) -> None:
    """Run one tick pass of the agent-wake scheduler.

    Acquires the cross-pod ``"agent_wake_tick"`` lock; on hold,
    returns silently. Within the lock, enumerates due schedules,
    claims each via optimistic-CAS, writes the initial ``wake_fires``
    row, and invokes the dispatch callback. The callback is awaited
    inline; long-running callback bodies (e.g. LLM round-trips) are
    expected to ``asyncio.create_task`` internally so the tick
    returns as soon as the row is staged -- not when the LLM
    response is complete.

    ``pool`` is typed ``Any`` to keep ``asyncpg`` an optional runtime
    dep on the wake package's interface (consumers' pool objects
    quack-type cleanly enough; the integration tests pin the real
    asyncpg shape). Same for ``nats_client`` (``threetears.nats.
    NatsClient | None``) -- a ``None`` skips lock acquisition for
    single-pod dev environments.

    :param pool: asyncpg-compatible connection pool (or proxy)
    :ptype pool: Any
    :param nats_client: :class:`threetears.nats.NatsClient` or
        ``None`` for single-pod dev mode (no cross-pod lock)
    :ptype nats_client: Any
    :param dispatch_callback: per-fire dispatcher; raised exceptions
        are isolated to a single schedule and recorded as failed
        fires
    :ptype dispatch_callback: DispatchCallback
    :return: nothing
    :rtype: None
    """
    # local import to keep the lock module out of the wake package's
    # always-paid import cost (consumers without NATS skip this).
    from threetears.nats import LockHeld, nats_distributed_lock  # noqa: PLC0415

    try:
        async with nats_distributed_lock(nats_client, _LOCK_KEY):
            await _run_tick_body(pool, dispatch_callback)
    except LockHeld:
        log.debug("wake_tick: lock held by another pod, skipping")


async def _run_tick_body(
    pool: Any,
    dispatch_callback: DispatchCallback,
) -> None:
    """Pump one tick's worth of fires through the dispatch callback."""
    # local imports avoid a hard dep on the registry module from the
    # types module (which the public re-export __init__ imports first).
    from threetears.core.collections.registry import CollectionRegistry  # noqa: PLC0415
    from threetears.core.config import DefaultCoreConfig  # noqa: PLC0415

    emitter = get_wake_emitter()
    tick_started = time.monotonic()
    now = datetime.now(UTC)
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    cfg = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    schedules = WakeScheduleCollection(registry=registry, config=cfg)
    fires = WakeFireCollection(registry=registry, config=cfg)
    due = await schedules.list_due_for_tick(now=now)
    log.info(
        EVENT_TICK_STARTED,
        extra={"extra_data": {"due_count": len(due), "tick_at": now.isoformat()}},
    )
    for schedule in due:
        try:
            await _dispatch_one(pool, schedule, schedules, fires, dispatch_callback, now, emitter)
        except Exception:  # noqa: BLE001 - boundary: isolate per-schedule failures
            # _dispatch_one already records a failed wake_fires row;
            # this outer except is defense in depth in case the row
            # write itself raises.
            log.exception(
                "wake_tick: schedule dispatch raised outside per-fire isolation",
                extra={"extra_data": {"schedule_id": str(schedule.schedule_id)}},
            )
            emitter.inc_failure(reason="handler_exception")
    duration = time.monotonic() - tick_started
    emitter.observe_tick_duration(duration)
    log.info(
        EVENT_TICK_COMPLETED,
        extra={"extra_data": {"processed": len(due), "duration_seconds": duration}},
    )


async def _dispatch_one(
    pool: Any,
    schedule: WakeScheduleEntity,
    schedules: WakeScheduleCollection,
    fires: WakeFireCollection,
    dispatch_callback: DispatchCallback,
    tick_at: datetime,
    emitter: Any,
) -> None:
    """Claim one due schedule and dispatch its fire.

    On CAS miss (another tick already claimed) returns silently. On
    dispatch callback success, writes the terminal fire status. On
    dispatch exception, writes a failed fire row + logs.
    """
    expected_next_fire = schedule.next_fire_at
    if expected_next_fire is None:
        # defense in depth: the list_due_for_tick query already
        # filters NULL next_fire_at, but if we ever loosen that the
        # CAS UPDATE has no anchor.
        log.warning(
            "wake_tick: schedule has NULL next_fire_at but appeared due; skipping",
            extra={"extra_data": {"schedule_id": str(schedule.schedule_id)}},
        )
        return

    # Terminal one-shot schedule types: this fire IS the single fire.
    # Force ``next_fire_at=None`` + ``status='expired'`` without
    # consulting the reschedule helper (whose ``last_fired_at`` arg
    # carries the PREVIOUS fire for catch_up semantics and so would
    # not yet recognise this fire as terminal).
    if schedule.schedule_type in {"one_shot_at", "relative_delay"}:
        computed_next_fire: datetime | None = None
    else:
        computed_next_fire = _compute_next_fire_at(
            schedule_type=schedule.schedule_type,
            schedule_config=schedule.schedule_config,
            missed_fire_policy=schedule.missed_fire_policy,
            last_fired_at=schedule.last_fired_at,
            now=tick_at,
        )
    new_status = "expired" if computed_next_fire is None else "active"

    claimed = await schedules.claim_and_reschedule(
        conversation_id=schedule.conversation_id,
        schedule_id=schedule.schedule_id,
        expected_next_fire=expected_next_fire,
        computed_next_fire=computed_next_fire,
        new_status=new_status,
        now=tick_at,
    )
    if not claimed:
        log.debug(
            "wake_tick: claim lost on schedule",
            extra={"extra_data": {"schedule_id": str(schedule.schedule_id)}},
        )
        return

    fire_id = _generate_fire_id()
    trigger = WakeTrigger(
        schedule_id=schedule.schedule_id,
        user_id=schedule.user_id,
        agent_id=schedule.agent_id,
        conversation_id=schedule.conversation_id,
        fire_source="scheduled_tick",
        execution_mode=schedule.execution_mode,
        schedule_type=schedule.schedule_type,
        fired_at=tick_at,
        schedule_name=schedule.name,
        task_prompt=schedule.task_prompt,
        context_from_schedule_id=schedule.context_from_schedule_id,
        delivery_target=schedule.delivery_target,
        delivery_config=schedule.delivery_config,
        skill_id=schedule.skill_id,
    )
    await fires.create_dispatching(
        fire_id=fire_id,
        schedule_id=schedule.schedule_id,
        webhook_subscription_id=None,
        conversation_id=schedule.conversation_id,
        scheduled_fire_at=expected_next_fire,
        actual_fired_at=tick_at,
        fire_source="scheduled_tick",
        execution_mode=schedule.execution_mode,
        delivery_target_resolved=schedule.delivery_target,
    )

    # Drift observation: every fire's drift is recorded into the
    # histogram; the drift event is emitted only above the threshold
    # so Loki doesn't fill with zero-drift noise.
    drift_seconds = max(0.0, (tick_at - expected_next_fire).total_seconds())
    emitter.observe_drift(drift_seconds)
    if drift_seconds > _DRIFT_LOG_THRESHOLD_SECONDS:
        log.info(
            EVENT_FIRE_DRIFT,
            extra={
                "extra_data": {
                    "schedule_id": str(schedule.schedule_id),
                    "conversation_id": str(schedule.conversation_id),
                    "fire_id": str(fire_id),
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
                "schedule_id": str(schedule.schedule_id),
                "fire_id": str(fire_id),
                "conversation_id": str(schedule.conversation_id),
                "schedule_type": schedule.schedule_type,
                "execution_mode": schedule.execution_mode,
                "missed_fire_policy": schedule.missed_fire_policy,
                "fire_source": "scheduled_tick",
            }
        },
    )

    try:
        result = await dispatch_callback(trigger, fire_id, pool)
    except Exception as exc:  # noqa: BLE001 - boundary: per-fire isolation
        log.exception(
            EVENT_FIRE_FAILED,
            extra={
                "extra_data": {
                    "schedule_id": str(schedule.schedule_id),
                    "fire_id": str(fire_id),
                    "conversation_id": str(schedule.conversation_id),
                    "schedule_type": schedule.schedule_type,
                    "execution_mode": schedule.execution_mode,
                    "error_type": type(exc).__name__,
                }
            },
        )
        await fires.finalize_failed(
            schedule.conversation_id,
            fire_id,
            error=str(exc),
            latency_ms=None,
        )
        emitter.inc_fire(
            status="failed",
            schedule_type=schedule.schedule_type,
            execution_mode=schedule.execution_mode,
        )
        emitter.inc_failure(reason="handler_exception")
        return

    # A dispatch callback may either raise (handled above) OR return
    # ``WakeDispatchResult(status='failed', error='...')`` without
    # raising -- e.g. a handler that wants to record a non-exceptional
    # failure (rate-limited downstream, no eligible model, etc.) with
    # its own error string. Route that case to ``finalize_failed`` so
    # the ``error`` field is not silently dropped on the floor.
    if result.status == "failed":
        await fires.finalize_failed(
            schedule.conversation_id,
            fire_id,
            error=result.error or "dispatch returned status='failed' without an error string",
            latency_ms=result.latency_ms,
        )
        emitter.inc_fire(
            status="failed",
            schedule_type=schedule.schedule_type,
            execution_mode=schedule.execution_mode,
        )
        emitter.inc_failure(reason="handler_exception")
        return

    await fires.finalize_success(
        schedule.conversation_id,
        fire_id,
        status=result.status,
        output_text=result.output_text,
        latency_ms=result.latency_ms,
        display_suppressed=result.display_suppressed,
    )
    emitter.inc_fire(
        status=result.status,
        schedule_type=schedule.schedule_type,
        execution_mode=schedule.execution_mode,
    )
    # Yielded fires also get the wake-to-yield duration histogram per
    # PLACEMENT §8.5.1. ``latency_ms`` carries the wall-clock duration
    # from the dispatch callback (handler-side measurement).
    if result.status == "yielded" and result.latency_ms is not None:
        emitter.observe_yield_duration(result.latency_ms / 1000.0)


def _generate_fire_id() -> UUID:
    """Mint a fresh ``fire_id``.

    UUIDv7 keeps lexicographic ordering for "list fires since" cursor
    queries (the canonical wake_fires read pattern). Pinned by the
    workspace-wide ``test_uuidv7_persisted_ids.py`` enforcement.
    """
    from uuid_utils import uuid7  # noqa: PLC0415

    return UUID(str(uuid7()))
