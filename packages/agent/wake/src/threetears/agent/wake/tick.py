"""Wake tick engine -- a thin adapter over the generic scheduled-jobs core.

Since S-2 the cross-pod tick pump (acquire the lock, enumerate due rows,
optimistic-CAS claim, in-flight fire row, per-fire failure isolation, drift
recording) lives ONCE in
:func:`threetears.scheduled_jobs.scheduled_tick_job`. agent-wake keeps its OWN
conversation-native tables, its richer ``FireStatus`` vocabulary, and its
webhook / ``[SILENT]`` handling; this module is the thin adapter layer that
lets the generic engine drive wake's stores without either side learning the
other's vocabulary.

design notes
------------

- **``wake_tick_job`` signature is preserved.** Consumers and the
  integration tests still call ``wake_tick_job(pool, nats_client,
  dispatch_callback)`` with the wake-shaped
  ``DispatchCallback = (WakeTrigger, fire_id, pool) -> WakeDispatchResult``.
  The generic engine's ``(JobTrigger, fire_id) -> JobFireResult`` shape is
  bridged internally by :func:`_adapt` below, so the delegation is invisible
  to the consumer.
- **Wrap, don't mutate.** :class:`_WakeScheduleStore` / :class:`_WakeFireStore`
  implement the core ``ScheduleStore`` / ``FireStore`` protocols by wrapping the
  UNCHANGED :class:`~threetears.agent.wake.collections.WakeScheduleCollection` /
  :class:`~threetears.agent.wake.collections.WakeFireCollection`. The webhook
  receiver and agent-tools layers call those collections directly with the
  conversation-native signatures, so the collections stay as-is and all the
  impedance-matching (``partition_key`` <-> ``conversation_id``, ``job_id`` <->
  ``schedule_id``, the opaque ``output`` dict <-> wake's typed
  ``output_text`` / ``display_suppressed`` columns, the agent fields <-> the
  opaque ``payload``) is isolated here.
- **The cross-pod lock key stays ``"agent_wake_tick"``** (a wake
  :class:`~threetears.scheduled_jobs.config.JobConfig`, NOT the core default
  ``"scheduled_jobs_tick"``). A rolling deploy where some pods still run the
  pre-S-2 wake tick must contend on the SAME lock, or two home pods could
  briefly co-fire one schedule before the optimistic-CAS catches it.
- **Metrics.** The generic engine owns the tick's fire / drift / tick-duration
  counters now (on the scheduled-jobs emitter). The genuinely wake-specific
  yield-duration histogram has no generic equivalent, so :func:`_adapt`
  re-emits it. The webhook / rate-limit / cap counters are emitted elsewhere
  and are untouched. (The fire counters moving from ``WAKE_FIRES_TOTAL`` to
  ``SCHEDULED_JOBS_FIRES_TOTAL`` is a documented BREAKING observability change
  -- see ``docs/migrating-agent-wake-to-scheduled-jobs.md``.)
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from typing import Any, Final
from uuid import UUID

from threetears.observe import get_logger
from threetears.scheduled_jobs import (
    DEFAULT_DISPATCH_REAP_AFTER_SECONDS,
    DEFAULT_TICK_DUE_LIMIT,
    DueSchedule,
    FireStore,
    JobConfig,
    JobFireResult,
    JobTrigger,
    ScheduleStore,
    scheduled_tick_job,
)

from threetears.agent.wake.collections import (
    WakeFireCollection,
    WakeScheduleCollection,
)
from threetears.agent.wake.entities import WakeScheduleEntity
from threetears.agent.wake.metrics import get_wake_emitter
from threetears.agent.wake.types import FireSource, WakeDispatchResult, WakeTrigger

__all__ = ["DispatchCallback", "wake_tick_job"]


log = get_logger(__name__)


# Cross-pod lock key, preserved across the S-2 delegation so a rolling deploy
# can't split-brain the home pod (see module docstring).
_WAKE_TICK_LOCK_KEY: Final[str] = "agent_wake_tick"

# The opaque ``kind`` discriminator the engine stamps on every wake job's
# trigger. A consumer that registered multiple kinds against one tick pump
# would route on this; wake runs a single dedicated pump.
_WAKE_KIND: Final[str] = "agent_wake"

# Constant ``fire_source`` for every scheduled (non-webhook) wake fire.
_WAKE_FIRE_SOURCE: Final[FireSource] = "scheduled_tick"


# --- payload keys: the agent-specific fields packed into the generic opaque
# ``payload`` by _WakeDueSchedule and read back by _rebuild_wake_trigger. The
# payload rides in-process on the JobTrigger (the engine never persists it), so
# native UUID / bool values round-trip without string coercion.
_P_USER_ID: Final[str] = "user_id"
_P_AGENT_ID: Final[str] = "agent_id"
_P_SKILL_ID: Final[str] = "skill_id"
_P_EXECUTION_MODE: Final[str] = "execution_mode"
_P_TASK_PROMPT: Final[str] = "task_prompt"
_P_CONTEXT_FROM: Final[str] = "context_from_schedule_id"
_P_INCLUDE_HISTORY: Final[str] = "include_conversation_history"


# --- output-dict keys: bridge wake's typed result columns through the generic
# opaque ``JobFireResult.output`` dict (_to_job_fire_result <-> _WakeFireStore).
_O_OUTPUT_TEXT: Final[str] = "output_text"
_O_DISPLAY_SUPPRESSED: Final[str] = "display_suppressed"


# Public dispatch-callback type -- UNCHANGED across S-2 so the consumer's call site
# (and the integration tests) do not move. The consumer's callback takes the
# wake-shaped ``(WakeTrigger, fire_id, pool)`` and returns a
# ``WakeDispatchResult``; :func:`_adapt` bridges it to the generic engine's
# ``(JobTrigger, fire_id) -> JobFireResult`` callback shape.
DispatchCallback = Callable[
    ["WakeTrigger", UUID, Any],
    Awaitable["WakeDispatchResult"],
]


class _WakeJobConfig:
    """:class:`JobConfig` pinning wake's lock key; defaults for the rest.

    Structurally satisfies the core ``JobConfig`` Protocol (pure-read). Only
    the lock key differs from the platform baseline -- the per-tick due-scan
    cap stays at the platform default.
    """

    @property
    def tick_lock_key(self) -> str:
        """Return wake's preserved cross-pod lock key."""
        return _WAKE_TICK_LOCK_KEY

    @property
    def tick_due_limit(self) -> int:
        """Return the platform-default per-tick due-row scan cap."""
        return DEFAULT_TICK_DUE_LIMIT

    @property
    def dispatch_reap_after_seconds(self) -> int:
        """Return the platform-default stale-``'dispatching'`` reap age."""
        return DEFAULT_DISPATCH_REAP_AFTER_SECONDS


_WAKE_JOB_CONFIG: JobConfig = _WakeJobConfig()


class _WakeDueSchedule:
    """Adapts a :class:`WakeScheduleEntity` to the core ``DueSchedule`` surface.

    Wraps one due schedule row so the generic tick engine can read the fields
    it needs (partition / id / scheduling) without learning wake's column
    names. The agent-specific fields (``user_id`` / ``agent_id`` / ``skill_id``
    / ``execution_mode`` / ``task_prompt`` / ``context_from_schedule_id`` /
    ``include_conversation_history``) are packed into the opaque ``payload``;
    :func:`_rebuild_wake_trigger` reads them back to reconstruct the
    ``WakeTrigger`` for the consumer's callback.

    Implements :class:`~threetears.scheduled_jobs.protocols.DueSchedule`
    structurally (conformance is enforced where
    :meth:`_WakeScheduleStore.list_due_for_tick` returns
    ``list[DueSchedule]``).
    """

    def __init__(self, entity: WakeScheduleEntity) -> None:
        """Wrap a wake schedule entity.

        :param entity: the schedule row read off ``list_due_for_tick``
        :ptype entity: WakeScheduleEntity
        :return: nothing
        :rtype: None
        """
        self._entity = entity

    @property
    def partition_key(self) -> UUID:
        """Return the generic partition key (wake's ``conversation_id``)."""
        return self._entity.conversation_id

    @property
    def job_id(self) -> UUID:
        """Return the generic job id (wake's ``schedule_id``)."""
        return self._entity.schedule_id

    @property
    def kind(self) -> str:
        """Return the wake routing discriminator."""
        return _WAKE_KIND

    @property
    def payload(self) -> dict[str, Any]:
        """Pack the agent-specific fields into the opaque payload."""
        return {
            _P_USER_ID: self._entity.user_id,
            _P_AGENT_ID: self._entity.agent_id,
            _P_SKILL_ID: self._entity.skill_id,
            _P_EXECUTION_MODE: self._entity.execution_mode,
            _P_TASK_PROMPT: self._entity.task_prompt,
            _P_CONTEXT_FROM: self._entity.context_from_schedule_id,
            _P_INCLUDE_HISTORY: self._entity.include_conversation_history,
        }

    @property
    def schedule_type(self) -> str:
        """Return the schedule type discriminator."""
        return self._entity.schedule_type

    @property
    def schedule_config(self) -> dict[str, Any]:
        """Return the per-schedule-type config dict."""
        return self._entity.schedule_config

    @property
    def missed_fire_policy(self) -> str:
        """Return the missed-fire policy."""
        return self._entity.missed_fire_policy

    @property
    def next_fire_at(self) -> datetime | None:
        """Return the planned fire instant."""
        return self._entity.next_fire_at

    @property
    def last_fired_at(self) -> datetime | None:
        """Return the most-recent fire instant (or ``None``)."""
        return self._entity.last_fired_at

    @property
    def name(self) -> str | None:
        """Return the optional human-readable schedule name."""
        return self._entity.name


class _WakeScheduleStore:
    """Adapts :class:`WakeScheduleCollection` to the core ``ScheduleStore``.

    Implements :class:`~threetears.scheduled_jobs.protocols.ScheduleStore` by
    delegating to the unchanged collection, translating the generic
    ``partition_key`` / ``job_id`` parameter names to wake's
    ``conversation_id`` / ``schedule_id``.
    """

    def __init__(self, collection: WakeScheduleCollection) -> None:
        """Wrap a wake schedule collection.

        :param collection: the unchanged wake schedule collection
        :ptype collection: WakeScheduleCollection
        :return: nothing
        :rtype: None
        """
        self._collection = collection

    async def list_due_for_tick(
        self,
        now: datetime,
        *,
        limit: int = 200,
    ) -> list[DueSchedule]:
        """Return due schedules wrapped as ``DueSchedule`` rows."""
        entities = await self._collection.list_due_for_tick(now, limit=limit)
        rows: list[DueSchedule] = [_WakeDueSchedule(entity) for entity in entities]
        return rows

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
        """Delegate the optimistic-CAS claim to the wake collection."""
        return await self._collection.claim_and_reschedule(
            conversation_id=partition_key,
            schedule_id=job_id,
            expected_next_fire=expected_next_fire,
            computed_next_fire=computed_next_fire,
            new_status=new_status,
            now=now,
        )


class _WakeFireStore:
    """Adapts :class:`WakeFireCollection` to the core ``FireStore``.

    Implements :class:`~threetears.scheduled_jobs.protocols.FireStore` by
    delegating to the unchanged collection. The generic ``output`` dict is
    unpacked into wake's typed ``output_text`` / ``display_suppressed`` columns
    on finalize; the wake-only ``webhook_subscription_id`` / ``fire_source`` /
    ``execution_mode`` create-args are supplied as the scheduled-fire constants
    (``fire_source`` / ``execution_mode`` are accepted by the collection for
    callsite symmetry and discarded -- they are not v1 ``wake_fires`` columns).
    """

    def __init__(self, collection: WakeFireCollection) -> None:
        """Wrap a wake fire collection.

        :param collection: the unchanged wake fire collection
        :ptype collection: WakeFireCollection
        :return: nothing
        :rtype: None
        """
        self._collection = collection

    async def create_dispatching(
        self,
        *,
        fire_id: UUID,
        job_id: UUID,
        partition_key: UUID,
        scheduled_fire_at: datetime,
        actual_fired_at: datetime,
    ) -> None:
        """Insert the in-flight ``wake_fires`` row for a scheduled fire."""
        await self._collection.create_dispatching(
            fire_id=fire_id,
            schedule_id=job_id,
            webhook_subscription_id=None,
            conversation_id=partition_key,
            scheduled_fire_at=scheduled_fire_at,
            actual_fired_at=actual_fired_at,
            fire_source=_WAKE_FIRE_SOURCE,
            # accepted for callsite symmetry + discarded by the collection (not
            # a v1 wake_fires column); the real execution_mode lives on the
            # schedule row and rides the trigger payload, not the fire row.
            execution_mode="inline",
        )

    async def finalize_success(
        self,
        partition_key: UUID,
        fire_id: UUID,
        *,
        status: str = "succeeded",
        output: dict[str, Any] | None = None,
        latency_ms: int | None = None,
    ) -> None:
        """Unpack the opaque output dict into wake's typed columns."""
        data = output or {}
        await self._collection.finalize_success(
            partition_key,
            fire_id,
            status=status,
            output_text=data.get(_O_OUTPUT_TEXT),
            latency_ms=latency_ms,
            display_suppressed=bool(data.get(_O_DISPLAY_SUPPRESSED, False)),
        )

    async def finalize_failed(
        self,
        partition_key: UUID,
        fire_id: UUID,
        *,
        error: str,
        latency_ms: int | None = None,
    ) -> None:
        """Delegate the failed-fire finalize to the wake collection."""
        await self._collection.finalize_failed(
            partition_key,
            fire_id,
            error=error,
            latency_ms=latency_ms,
        )

    async def reap_stale_dispatching(
        self,
        now: datetime,
        *,
        older_than: timedelta,
    ) -> int:
        """Delegate the abandoned-``'dispatching'`` sweep to the wake collection."""
        return await self._collection.reap_stale_dispatching(now, older_than=older_than)


def _rebuild_wake_trigger(job_trigger: JobTrigger) -> WakeTrigger:
    """Reconstruct the wake-shaped trigger from the generic envelope.

    The scheduling fields come straight off the :class:`JobTrigger`; the
    agent-specific fields are read back out of the opaque ``payload`` packed by
    :meth:`_WakeDueSchedule.payload`.

    :param job_trigger: the generic fire envelope the engine built
    :ptype job_trigger: JobTrigger
    :return: the wake-shaped trigger the consumer callback expects
    :rtype: WakeTrigger
    """
    payload = job_trigger.payload
    return WakeTrigger(
        schedule_id=job_trigger.job_id,
        user_id=payload[_P_USER_ID],
        agent_id=payload[_P_AGENT_ID],
        conversation_id=job_trigger.partition_key,
        fire_source=_WAKE_FIRE_SOURCE,
        execution_mode=payload[_P_EXECUTION_MODE],
        schedule_type=job_trigger.schedule_type,
        fired_at=job_trigger.fired_at,
        schedule_name=job_trigger.name,
        task_prompt=payload.get(_P_TASK_PROMPT),
        context_from_schedule_id=payload.get(_P_CONTEXT_FROM),
        skill_id=payload.get(_P_SKILL_ID),
        include_conversation_history=bool(payload.get(_P_INCLUDE_HISTORY, True)),
    )


def _to_job_fire_result(result: WakeDispatchResult) -> JobFireResult:
    """Pack a wake dispatch result into the generic fire result.

    The wake-specific ``output_text`` / ``display_suppressed`` ride the opaque
    ``output`` dict; :meth:`_WakeFireStore.finalize_success` unpacks them.
    ``status`` flows verbatim (the engine is status-agnostic: it persists any
    terminal status the callback reports and only special-cases ``'failed'``).

    :param result: the wake dispatch result the consumer callback returned
    :ptype result: WakeDispatchResult
    :return: the generic fire result the engine finalizes
    :rtype: JobFireResult
    """
    return JobFireResult(
        status=result.status,
        output={
            _O_OUTPUT_TEXT: result.output_text,
            _O_DISPLAY_SUPPRESSED: result.display_suppressed,
        },
        latency_ms=result.latency_ms,
        error=result.error,
    )


async def wake_tick_job(
    pool: Any,
    nats_client: Any,
    dispatch_callback: DispatchCallback,
) -> None:
    """Run one tick pass of the agent-wake scheduler.

    Builds the adapter stores over ``pool`` + an adapter dispatch callback that
    bridges the consumer's wake-shaped callback to the generic engine, and
    delegates to :func:`threetears.scheduled_jobs.scheduled_tick_job` under the
    preserved ``"agent_wake_tick"`` cross-pod lock. The callback is awaited
    inline; long-running callback bodies (e.g. LLM round-trips) are expected to
    ``asyncio.create_task`` internally so the tick returns as soon as the row
    is staged -- not when the LLM response is complete.

    ``pool`` is typed ``Any`` to keep ``asyncpg`` an optional runtime dep on the
    wake package's interface (consumers' pool objects quack-type cleanly; the
    integration tests pin the real asyncpg shape). Same for ``nats_client``
    (:class:`threetears.nats.NatsClient` or ``None``) -- a ``None`` skips lock
    acquisition for single-pod dev environments (the per-schedule optimistic-CAS
    still guards against double fires).

    :param pool: asyncpg-compatible connection pool (or proxy)
    :ptype pool: Any
    :param nats_client: :class:`threetears.nats.NatsClient` or ``None`` for
        single-pod dev mode (no cross-pod lock)
    :ptype nats_client: Any
    :param dispatch_callback: per-fire dispatcher; raised exceptions are
        isolated to a single schedule and recorded as failed fires by the
        engine
    :ptype dispatch_callback: DispatchCallback
    :return: nothing
    :rtype: None
    """
    # local imports keep the registry / config plumbing out of the wake
    # package's always-paid import cost (mirrors the pre-S-2 tick body).
    from threetears.core.collections.registry import CollectionRegistry  # noqa: PLC0415
    from threetears.core.config import DefaultCoreConfig  # noqa: PLC0415

    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    cfg = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    schedule_store = _WakeScheduleStore(WakeScheduleCollection(registry=registry, config=cfg))
    fire_store = _WakeFireStore(WakeFireCollection(registry=registry, config=cfg))
    emitter = get_wake_emitter()

    async def _adapt(job_trigger: JobTrigger, fire_id: UUID) -> JobFireResult:
        """Bridge one generic fire to wake's dispatch callback + result shape."""
        wake_trigger = _rebuild_wake_trigger(job_trigger)
        result = await dispatch_callback(wake_trigger, fire_id, pool)
        # Re-emit the genuinely wake-specific yield-duration histogram (no
        # generic equivalent); the generic engine owns the fire / drift
        # counters. ``latency_ms`` carries the handler-side wall-clock duration.
        if result.status == "yielded" and result.latency_ms is not None:
            emitter.observe_yield_duration(result.latency_ms / 1000.0)
        return _to_job_fire_result(result)

    await scheduled_tick_job(
        schedule_store,
        fire_store,
        _adapt,
        nats_client=nats_client,
        config=_WAKE_JOB_CONFIG,
    )
