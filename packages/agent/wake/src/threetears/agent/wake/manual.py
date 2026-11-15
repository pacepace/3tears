"""A wake run by hand: one fire, now, outside the schedule.

A manual fire is a fire like any other -- a ``wake_fires`` row, the consumer's
dispatch callback, a terminal status -- with three differences:

- **It does not move the schedule.** ``next_fire_at`` and ``last_fired_at``
  are not touched, so the next scheduled fire is where it was and the
  reschedule engine never reads a manual fire as the wake's last one.
- **It is recorded as run by hand, with who asked.** The row's
  ``fire_source`` is ``'manual'`` and ``requested_by`` names the asker.
- **It has a limit of its own.** A wake is run by hand at most once per
  ``min_gap``, and never while a fire of it is still dispatching.

The limit is taken by **one transaction** (:func:`_claim`): a keyed ``UPDATE`` of
``agent_wake_schedules.last_manual_fire_at`` whose guards are the limit, then,
only when it wrote the row, the insert of the fire row. Two callers at once
contend on the wake's row lock, held to the end of the first's transaction;
the second re-reads the row the first committed, finds the stamp inside the
gap, updates nothing and inserts nothing. A count of fire rows followed by an
insert would let both through.

Two steps, so a caller can answer at once and run the fire in the background:

1. :func:`claim_manual_fire` takes the limit and inserts the ``'dispatching'``
   row, or raises :class:`ManualFireRefused` saying why not.
2. :func:`run_manual_fire` runs the claimed fire through the consumer's
   dispatch callback and finalizes the row.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from uuid import UUID

from threetears.observe import contain, get_logger
from uuid_utils import uuid7

from threetears.agent.wake.collections import WakeFireCollection
from threetears.agent.wake.entities import WakeScheduleEntity
from threetears.agent.wake.types import WakeDispatchResult, WakeTrigger
from threetears.core.keyed import Fragment, read_row, update_keyed

__all__ = [
    "MANUAL_FIRE_SOURCE",
    "ManualFireClaim",
    "ManualFireRefused",
    "ManualRefusal",
    "claim_manual_fire",
    "run_manual_fire",
]

log = get_logger(__name__)

#: ``wake_fires.fire_source`` of a fire run by hand.
MANUAL_FIRE_SOURCE = "manual"

#: Why a manual fire was not claimed.
ManualRefusal = Literal["too_soon", "dispatching", "gone"]

# A fire of the wake in flight: the limit's second guard, and the refusal's wording. Read through the small partial
# index of the fires in flight (idx_wake_fires_dispatching, WHERE status = 'dispatching'), the wake as a filter: a
# bounded read of the fires dispatching now, never a walk of the wake's whole fire history (coordinator ruling,
# 2026-11-15). ``status`` is written as the index's own literal so a generic plan can prove the predicate. Proven at a
# busy wake's size (its 2,000 fires against the 500 in flight) by test_index_plans.py::test_wake_fires_dispatching.
_IN_FLIGHT = (
    "EXISTS (SELECT 1 FROM wake_fires f"
    " WHERE f.status = 'dispatching' AND f.conversation_id = $1 AND f.schedule_id = $2)"
)


async def _claim(
    conn: Any,
    conversation_id: UUID,
    schedule_id: UUID,
    *,
    at: datetime,
    min_gap: timedelta,
    fire_id: UUID,
    requested_by: str,
) -> bool:
    """Take the limit and insert the in-flight fire row, in the caller's transaction on ``conn``.

    The ``UPDATE``'s guards are the whole limit: the stamp is outside the gap, and no fire of this wake is in flight.
    It takes the wake's row lock, held to the end of the transaction, so the insert that follows is the only one any
    caller makes inside the gap: a second caller waits on the lock, re-reads the row the first committed and writes
    nothing. Was one statement (an ``UPDATE ... RETURNING`` feeding the ``INSERT``); a keyed write and a keyed insert
    in one transaction are as atomic, and the concurrency test proves it (test_manual_fire_and_grid). ``date_updated``
    is left alone on purpose: saves of the wake are fenced on it, and nothing a cached row holds changes.

    :param conn: the transaction's connection
    :ptype conn: Any
    :param conversation_id: the wake's conversation
    :ptype conversation_id: UUID
    :param schedule_id: the wake
    :ptype schedule_id: UUID
    :param at: the instant of the claim, the stamp and the fire's ``actual_fired_at``
    :ptype at: datetime
    :param min_gap: the shortest time between two manual fires of the wake
    :ptype min_gap: timedelta
    :param fire_id: the new fire's id
    :ptype fire_id: UUID
    :param requested_by: who asked
    :ptype requested_by: str
    :return: whether the limit was taken and the row inserted
    :rtype: bool
    """
    stamped = await update_keyed(
        conn,
        "agent_wake_schedules",
        values={"last_manual_fire_at": at},
        by={"conversation_id": conversation_id, "schedule_id": schedule_id},
        where=[
            Fragment('("last_manual_fire_at" IS NULL OR "last_manual_fire_at" <= $1)', (at - min_gap,)),
            Fragment(f"NOT {_IN_FLIGHT}", (conversation_id, schedule_id)),
        ],
    )
    if stamped.count == 0:
        return False
    await conn.execute(
        "INSERT INTO wake_fires ("
        "conversation_id, fire_id, schedule_id, webhook_subscription_id, "
        "scheduled_fire_at, actual_fired_at, status, display_suppressed, fire_source, requested_by"
        ") VALUES ($1, $2, $3, NULL, NULL, $4, 'dispatching', false, 'manual', $5)",
        conversation_id,
        fire_id,
        schedule_id,
        at,
        requested_by,
    )
    return True


async def _why_refused(pool: Any, conversation_id: UUID, schedule_id: UUID, min_gap: timedelta) -> ManualFireRefused:
    """Word a refusal: read only to say why, after the claim decided.

    :param pool: asyncpg-compatible pool
    :ptype pool: Any
    :param conversation_id: the wake's conversation
    :ptype conversation_id: UUID
    :param schedule_id: the wake
    :ptype schedule_id: UUID
    :param min_gap: the gap, for ``retry_at``
    :ptype min_gap: timedelta
    :return: the refusal to raise
    :rtype: ManualFireRefused
    """
    # cache-bypass: a read of committed state, to word the refusal; the stamp is a column no cached row carries.
    row = await read_row(
        pool,
        "agent_wake_schedules",
        by={"conversation_id": conversation_id, "schedule_id": schedule_id},
        columns=("last_manual_fire_at",),
    )
    if row is None:
        return ManualFireRefused("gone")
    last = row["last_manual_fire_at"]
    in_flight = await read_row(
        pool,
        "wake_fires",
        by={"status": "dispatching", "conversation_id": conversation_id, "schedule_id": schedule_id},
        columns=("fire_id",),
    )
    if in_flight is not None:
        return ManualFireRefused("dispatching", last_manual_fire_at=last)
    return ManualFireRefused(
        "too_soon",
        retry_at=(last + min_gap) if last is not None else None,
        last_manual_fire_at=last,
    )


#: The consumer's per-fire dispatcher, as the tick calls it.
ManualDispatch = Callable[[WakeTrigger, UUID, Any], Awaitable[WakeDispatchResult]]


class ManualFireRefused(Exception):
    """A manual fire was not claimed, and nothing was written.

    :ivar reason: ``too_soon`` (run by hand inside the gap), ``dispatching`` (a fire of
        the wake is in flight) or ``gone`` (the wake no longer exists)
    :ivar retry_at: when the gap ends, for ``too_soon``
    :ivar last_manual_fire_at: when the wake was last run by hand, when it has been
    """

    def __init__(
        self,
        reason: ManualRefusal,
        *,
        retry_at: datetime | None = None,
        last_manual_fire_at: datetime | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason: ManualRefusal = reason
        self.retry_at = retry_at
        self.last_manual_fire_at = last_manual_fire_at


@dataclass(frozen=True)
class ManualFireClaim:
    """A manual fire that holds the limit and has its ``'dispatching'`` row.

    :ivar fire_id: the ``wake_fires`` row
    :ivar trigger: the fire's envelope, ``fire_source='manual'``, carrying ``requested_by``
    :ivar claimed_at: the instant the limit was taken, and the fire's ``actual_fired_at``
    """

    fire_id: UUID
    trigger: WakeTrigger
    claimed_at: datetime


async def claim_manual_fire(
    pool: Any,
    schedule: WakeScheduleEntity,
    *,
    requested_by: str,
    min_gap: timedelta,
    task_prompt: str | None = None,
    now: datetime | None = None,
) -> ManualFireClaim:
    """Take the manual-fire limit for a wake and insert its in-flight fire row.

    One transaction decides: of any number of callers at once, exactly one gets
    a claim. The wake's ``next_fire_at`` and ``last_fired_at`` are not touched.

    :param pool: asyncpg-compatible pool
    :ptype pool: Any
    :param schedule: the wake to run
    :ptype schedule: WakeScheduleEntity
    :param requested_by: who asked, as the consumer names them; kept on the fire row
    :ptype requested_by: str
    :param min_gap: the shortest time between two manual fires of this wake; must be positive
    :ptype min_gap: timedelta
    :param task_prompt: the text the fire runs with, when it is not the wake row's own
    :ptype task_prompt: str | None
    :param now: the instant of the claim (defaults to now)
    :ptype now: datetime | None
    :return: the claim, to hand to :func:`run_manual_fire`
    :rtype: ManualFireClaim
    :raises ManualFireRefused: when the wake was run by hand inside ``min_gap``, a fire of
        it is dispatching, or it is gone
    :raises ValueError: when ``min_gap`` is not positive or ``requested_by`` is empty
    """
    if min_gap <= timedelta(0):
        # With no gap the stamp cannot tell two callers apart, and both would run.
        raise ValueError("min_gap must be positive: the gap is what makes two callers one fire")
    if not requested_by.strip():
        raise ValueError("requested_by must name who asked for the fire")
    at = now or datetime.now(UTC)
    fire_id = UUID(str(uuid7()))
    conversation_id = schedule.conversation_id
    schedule_id = schedule.schedule_id
    # cache-bypass: the limit must be decided by the row as committed, on its row lock; no cached
    # row can answer it, and the stamp is a column no cached row carries.
    async with pool.acquire() as conn, conn.transaction():
        claimed = await _claim(
            conn,
            conversation_id,
            schedule_id,
            at=at,
            min_gap=min_gap,
            fire_id=fire_id,
            requested_by=requested_by,
        )
    if not claimed:
        raise await _why_refused(pool, conversation_id, schedule_id, min_gap)
    trigger = WakeTrigger(
        schedule_id=schedule_id,
        user_id=schedule.user_id,
        agent_id=schedule.agent_id,
        conversation_id=conversation_id,
        fire_source="manual",
        execution_mode=schedule.execution_mode,
        schedule_type=schedule.schedule_type,
        fired_at=at,
        schedule_name=schedule.name,
        task_prompt=task_prompt if task_prompt is not None else schedule.task_prompt,
        context_from_schedule_id=schedule.context_from_schedule_id,
        skill_id=schedule.skill_id,
        include_conversation_history=bool(schedule.include_conversation_history),
        fire_id=fire_id,
        protected=bool(schedule.protected),
        requested_by=requested_by,
    )
    log.info(
        "manual wake fire claimed",
        extra={
            "extra_data": {
                "fire_id": str(fire_id),  # convert at border: manual-fire-claimed log extra_data field
                "schedule_id": str(schedule_id),
                "agent_id": str(schedule.agent_id),
                "requested_by": requested_by,
            }
        },
    )
    return ManualFireClaim(fire_id=fire_id, trigger=trigger, claimed_at=at)


async def run_manual_fire(
    claim: ManualFireClaim,
    *,
    pool: Any,
    fires: WakeFireCollection,
    dispatch_callback: ManualDispatch,
) -> WakeDispatchResult:
    """Run a claimed manual fire and finalize its row.

    The callback is the one the tick hands its fires to, so a manual fire is
    permitted, limited, given a conversation and handled exactly as a scheduled
    one is. A callback that raises is recorded as a failed fire and its error
    is the result; nothing is raised, since the caller has usually answered
    already and runs this in the background.

    :param claim: the claim :func:`claim_manual_fire` returned
    :ptype claim: ManualFireClaim
    :param pool: asyncpg-compatible pool, forwarded to the callback
    :ptype pool: Any
    :param fires: the host process's fire collection
    :ptype fires: WakeFireCollection
    :param dispatch_callback: the consumer's per-fire dispatcher
    :ptype dispatch_callback: ManualDispatch
    :return: how the fire ended
    :rtype: WakeDispatchResult
    """
    started = time.monotonic()
    trigger = claim.trigger
    conversation_id = trigger.conversation_id
    try:
        result = await dispatch_callback(trigger, claim.fire_id, pool)
    except Exception as exc:  # noqa: BLE001 - boundary: the fire row must reach a terminal status whatever stopped it
        latency_ms = int((time.monotonic() - started) * 1000)
        contain(
            exc,
            log=log,
            what="manual wake fire failed",
            extra={"fire_id": str(claim.fire_id), "schedule_id": str(trigger.schedule_id)},
            level="error",
        )
        await fires.finalize_failed(conversation_id, claim.fire_id, error=str(exc), latency_ms=latency_ms)
        return WakeDispatchResult(status="failed", error=str(exc), latency_ms=latency_ms)
    if result.status == "failed":
        await fires.finalize_failed(
            conversation_id,
            claim.fire_id,
            error=result.error or "the fire failed",
            latency_ms=result.latency_ms,
        )
    else:
        await fires.finalize_success(
            conversation_id,
            claim.fire_id,
            status=result.status,
            output_text=result.output_text,
            latency_ms=result.latency_ms,
            display_suppressed=result.display_suppressed,
        )
    log.info(
        "manual wake fire finished",
        extra={
            "extra_data": {
                "fire_id": str(claim.fire_id),  # convert at border: manual-fire-finished log extra_data field
                "schedule_id": str(trigger.schedule_id),
                "status": result.status,
            }
        },
    )
    return result
