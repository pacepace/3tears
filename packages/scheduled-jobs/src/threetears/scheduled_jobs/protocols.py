"""Store protocols the tick engine depends on.

The tick engine (:func:`threetears.scheduled_jobs.tick.scheduled_tick_job`)
depends ONLY on these Protocols -- never on a concrete collection. The
default store (:mod:`threetears.scheduled_jobs.collections`) implements
them, and so can a typed consumer collection (e.g. agent-wake's, once it
is refactored onto this core).

The method signatures are derived from agent-wake's
``WakeScheduleCollection`` / ``WakeFireCollection`` surface, generalized:

- ``conversation_id`` -> ``partition_key`` (the generic partition column).
- ``schedule_id`` -> ``job_id``; ``fire_id`` stays ``fire_id``.
- the typed agent columns (``user_id`` / ``agent_id`` / ``skill_id`` /
  ``execution_mode`` / ``include_conversation_history`` / ...) collapse to
  an opaque ``kind`` (TEXT) + ``payload`` (JSON) on the due-row protocol.
- ``create_dispatching`` loses the ``webhook_subscription_id`` /
  ``fire_source`` / ``execution_mode`` parameters (webhook + agent
  concepts); it keeps the two-write audit pattern (write an in-flight
  row, then finalize) so a crashed dispatcher leaves audit evidence.

``runtime_checkable`` is set on each so a consumer can ``isinstance``-check
structural conformance without a hard ABC dependency, mirroring the
agent-wake ``EncryptionService`` / ``WakeConfig`` precedent.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

__all__ = [
    "DueSchedule",
    "FireStore",
    "ScheduleStore",
]


@runtime_checkable
class DueSchedule(Protocol):
    """The schedule-row shape the tick engine reads off a due row.

    :meth:`ScheduleStore.list_due_for_tick` returns a sequence of these.
    The engine reads exactly these attributes to construct a
    :class:`~threetears.scheduled_jobs.types.JobTrigger`, compute the
    next fire instant, and run the optimistic-CAS claim. A consumer's
    schedule entity satisfies this Protocol by exposing the same
    attributes (the default :class:`ScheduledJobEntity` does).

    :ivar partition_key: the row's partition column value (the generic
        equivalent of agent-wake's ``conversation_id``)
    :ivar job_id: the row's bare id
    :ivar kind: opaque routing discriminator (the consumer's job type)
    :ivar payload: opaque per-job JSON payload (never inspected by the
        engine; forwarded verbatim on the trigger)
    :ivar schedule_type: one of
        :data:`~threetears.scheduled_jobs.types.ScheduleType`
    :ivar schedule_config: per-schedule-type config dict consumed by
        :func:`~threetears.scheduled_jobs.reschedule.compute_next_fire_at`
    :ivar missed_fire_policy: one of
        :data:`~threetears.scheduled_jobs.types.MissedFirePolicy`
    :ivar next_fire_at: the planned fire instant; the optimistic-CAS
        anchor (``None`` only as defense-in-depth -- the due query
        filters it out)
    :ivar last_fired_at: timestamp of the most recent fire, or ``None``
    :ivar name: optional human-readable schedule name
    """

    @property
    def partition_key(self) -> UUID: ...

    @property
    def job_id(self) -> UUID: ...

    @property
    def kind(self) -> str: ...

    @property
    def payload(self) -> dict[str, Any]: ...

    @property
    def schedule_type(self) -> str: ...

    @property
    def schedule_config(self) -> dict[str, Any]: ...

    @property
    def missed_fire_policy(self) -> str: ...

    @property
    def next_fire_at(self) -> datetime | None: ...

    @property
    def last_fired_at(self) -> datetime | None: ...

    @property
    def name(self) -> str | None: ...


@runtime_checkable
class ScheduleStore(Protocol):
    """The schedule-side surface the tick engine calls.

    Two methods: enumerate due rows, and atomically claim-and-reschedule
    one. Generalized from ``WakeScheduleCollection``.
    """

    async def list_due_for_tick(
        self,
        now: datetime,
        *,
        limit: int = 200,
    ) -> list[DueSchedule]:
        """Return active schedules whose ``next_fire_at <= now``.

        Cross-partition scan (the engine enumerates ready jobs across
        every partition). Ordered by ``next_fire_at`` ASC, capped at
        ``limit``.

        :param now: tick instant; rows with ``next_fire_at <= now`` are
            returned
        :ptype now: datetime
        :param limit: per-tick cap on rows returned
        :ptype limit: int
        :return: list of due schedule rows ordered by ``next_fire_at`` ASC
        :rtype: list[DueSchedule]
        """
        ...

    async def claim_and_reschedule(
        self,
        *,
        partition_key: UUID,
        job_id: UUID,
        expected_next_fire: datetime,
        computed_next_fire: datetime | None,
        new_status: str,
        now: datetime,
    ) -> bool:
        """Atomically claim a due schedule and advance its ``next_fire_at``.

        Optimistic-CAS UPDATE: the predicate ``next_fire_at =
        expected_next_fire`` ensures exactly one tick wins when two pods
        (or two overlapping tick bodies) briefly disagree. Returns
        ``True`` on a successful claim, ``False`` when another tick has
        already advanced the row. The UPDATE also stamps ``last_fired_at
        = now`` and rewrites ``status``.

        :param partition_key: partition column value
        :ptype partition_key: UUID
        :param job_id: target schedule id
        :ptype job_id: UUID
        :param expected_next_fire: the ``next_fire_at`` value observed at
            claim time; the CAS predicate
        :ptype expected_next_fire: datetime
        :param computed_next_fire: the new ``next_fire_at`` (``None`` for
            terminal one-shots)
        :ptype computed_next_fire: datetime | None
        :param new_status: the new ``status`` value (``'active'`` or
            ``'expired'``)
        :ptype new_status: str
        :param now: tick instant; written as ``last_fired_at`` /
            ``date_updated``
        :ptype now: datetime
        :return: ``True`` on successful claim, ``False`` on CAS miss
        :rtype: bool
        """
        ...


@runtime_checkable
class FireStore(Protocol):
    """The fire-side surface the tick engine calls.

    Three methods: insert the in-flight row, finalize success, finalize
    failure. Generalized from ``WakeFireCollection``; the two-write audit
    pattern (insert ``'dispatching'``, then overwrite to the terminal
    status) is preserved so a dispatcher that crashes mid-fire leaves a
    queryable in-flight row.
    """

    async def create_dispatching(
        self,
        *,
        fire_id: UUID,
        job_id: UUID,
        partition_key: UUID,
        scheduled_fire_at: datetime,
        actual_fired_at: datetime,
    ) -> None:
        """Insert an initial in-flight ``job_fires`` row (``status='dispatching'``).

        Written immediately after a claim succeeds, before the dispatch
        callback runs. The callback's outcome finalizes via
        :meth:`finalize_success` / :meth:`finalize_failed`.

        :param fire_id: pre-generated fire id
        :ptype fire_id: UUID
        :param job_id: source schedule id
        :ptype job_id: UUID
        :param partition_key: partition column value
        :ptype partition_key: UUID
        :param scheduled_fire_at: the planned fire instant (the claimed
            row's ``next_fire_at``)
        :ptype scheduled_fire_at: datetime
        :param actual_fired_at: the actual tick instant
        :ptype actual_fired_at: datetime
        :return: nothing
        :rtype: None
        """
        ...

    async def finalize_success(
        self,
        partition_key: UUID,
        fire_id: UUID,
        *,
        status: str = "succeeded",
        output: dict[str, Any] | None = None,
        latency_ms: int | None = None,
    ) -> None:
        """Stamp a successful dispatch result onto the fire row.

        :param partition_key: partition column value
        :ptype partition_key: UUID
        :param fire_id: target fire row
        :ptype fire_id: UUID
        :param status: terminal status (defaults to ``'succeeded'``)
        :ptype status: str
        :param output: optional captured output payload
        :ptype output: dict[str, Any] | None
        :param latency_ms: optional end-to-end latency
        :ptype latency_ms: int | None
        :return: nothing
        :rtype: None
        """
        ...

    async def finalize_failed(
        self,
        partition_key: UUID,
        fire_id: UUID,
        *,
        error: str,
        latency_ms: int | None = None,
    ) -> None:
        """Stamp a failed-dispatch result onto the fire row.

        :param partition_key: partition column value
        :ptype partition_key: UUID
        :param fire_id: target fire row
        :ptype fire_id: UUID
        :param error: captured error message
        :ptype error: str
        :param latency_ms: optional latency up to the failure
        :ptype latency_ms: int | None
        :return: nothing
        :rtype: None
        """
        ...

    async def reap_stale_dispatching(
        self,
        now: datetime,
        *,
        older_than: timedelta,
    ) -> int:
        """Finalize ``'dispatching'`` fire rows abandoned mid-dispatch.

        A pod that dies after :meth:`create_dispatching` but before a
        finalize leaves the fire row stuck in ``'dispatching'`` forever.
        The occurrence's schedule already advanced (the claim ran first),
        so it never re-fires -- without a sweep the row is a permanent
        zombie and the loss is silent. This cross-partition reaper stamps
        every ``'dispatching'`` row whose ``actual_fired_at`` is older
        than ``older_than`` before ``now`` to ``'failed'`` with a reaper
        marker, so the loss surfaces in fire history + failure metrics.
        Invoked once per tick under the tick's cross-pod lock.

        :param now: sweep instant (TZ-aware); the age cutoff is
            ``now - older_than``
        :ptype now: datetime
        :param older_than: minimum in-flight age before a row is reaped
        :ptype older_than: timedelta
        :return: number of rows reaped
        :rtype: int
        """
        ...
