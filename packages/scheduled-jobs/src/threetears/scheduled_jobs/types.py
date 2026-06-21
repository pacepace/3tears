"""Literal types + envelope/result dataclasses for scheduled-jobs.

Generalized from :mod:`threetears.agent.wake.types` with every
agent/skill/webhook/conversation-specific field removed. What remains
is the payload-agnostic scheduling vocabulary: the schedule-type and
missed-fire-policy Literals (which mirror the CHECK constraints on the
default store's tables), the fire-status Literal, and the two frozen
dataclasses the tick engine passes to / receives from the injected
dispatch callback.

Why ``Literal`` and not ``Enum``: callers pass these values through
JSON boundaries where strings round-trip cleanly. ``Literal`` keeps the
runtime payload a plain ``str`` so JSON encoding needs no custom
serializer and mypy still pins valid value sets at every call site.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

__all__ = [
    "JobFireResult",
    "JobTrigger",
    "MissedFirePolicy",
    "ScheduleFireStatus",
    "ScheduleType",
]


# ``schedule_type`` discriminator. Each value maps to a specific
# ``schedule_config`` JSON shape consumed by
# :func:`threetears.scheduled_jobs.reschedule.compute_next_fire_at`:
#
# - ``daily_at``: ``{"hour": 14, "minute": 0, "tz": "America/Los_Angeles"}``
# - ``every_n_hours``: ``{"n": 3}``
# - ``random_within_window``: ``{"start_hour": 9, "end_hour": 21,
#   "tz": "America/Los_Angeles"}``
# - ``one_shot_at``: ``{"fire_at_iso": "2026-05-25T14:00:00+00:00"}``
# - ``cron``: ``{"expr": "0 */3 * * *"}``
# - ``relative_delay``: ``{"delay": "30m"}``
# - ``interval``: ``{"seconds": 1800}``
#
# ``one_shot_at`` and ``relative_delay`` are terminal: their single
# fire flips the schedule to ``status='expired'`` (``next_fire_at``
# becomes ``None``).
ScheduleType = Literal[
    "daily_at",
    "every_n_hours",
    "random_within_window",
    "one_shot_at",
    "cron",
    "relative_delay",
    "interval",
]


# ``missed_fire_policy`` discriminator. ``'coalesce'`` (default) fires
# ONCE for a backlog of missed ticks, recomputing ``next_fire_at``
# forward; ``'catch_up'`` advances ``next_fire_at`` by exactly one
# increment so subsequent ticks fire once per missed interval until
# caught up.
MissedFirePolicy = Literal["coalesce", "catch_up"]


# ``status`` column on the default store's ``job_fires`` table, and the
# status a dispatch callback may report. Generalized from agent-wake's
# ``FireStatus`` -- the agent-specific ``'fired_silent'`` / ``'yielded'``
# / ``'skipped_*'`` values were dropped; what remains is the
# payload-agnostic lifecycle.
#
# - ``'dispatching'`` -- in-flight placeholder written before the
#   dispatch callback runs; overwritten on finalize. A row that stays
#   here means the dispatcher crashed mid-fire (audit evidence).
# - ``'succeeded'`` -- the dispatch callback completed cleanly.
# - ``'failed'`` -- the dispatch callback raised, or returned
#   ``status='failed'`` with an error string.
ScheduleFireStatus = Literal[
    "dispatching",
    "succeeded",
    "failed",
]


@dataclass(frozen=True)
class JobTrigger:
    """Immutable envelope describing a single due-job fire opportunity.

    The tick engine constructs one of these after it claims a due
    schedule row (optimistic-CAS) and hands it to the injected dispatch
    callback. Frozen so the callback cannot mutate fields the engine
    owns.

    Payload-agnostic: ``kind`` is an opaque routing discriminator and
    ``payload`` an opaque JSON blob -- the engine never inspects either.
    The store protocol surfaces them off the schedule row; the dispatch
    callback interprets them. ``partition_key`` is the generic
    partition column (agent-wake's ``conversation_id`` equivalent).

    :ivar job_id: the schedule row's bare id
    :ivar partition_key: the schedule's partition column value
    :ivar kind: opaque routing discriminator (the consumer's job type)
    :ivar payload: opaque per-job JSON payload (never inspected by the
        engine)
    :ivar schedule_type: the schedule-type that produced this fire
    :ivar fired_at: the tick instant (the actual fire time)
    :ivar scheduled_fire_at: the schedule's ``next_fire_at`` at claim
        time (the planned fire instant; drift = ``fired_at`` minus this)
    :ivar name: optional human-readable schedule name
    """

    job_id: UUID
    partition_key: UUID
    kind: str
    schedule_type: str
    fired_at: datetime
    scheduled_fire_at: datetime
    payload: dict[str, Any] = field(default_factory=dict)
    name: str | None = None


@dataclass(frozen=True)
class JobFireResult:
    """Return value from a dispatch callback to the tick engine.

    The engine uses :attr:`status` to write the terminal ``job_fires``
    row; :attr:`output` / :attr:`latency_ms` / :attr:`error` are
    optional fixups the dispatcher may capture. A callback may either
    raise (the engine isolates the failure + records a failed fire) OR
    return ``JobFireResult(status='failed', error=...)`` to record a
    non-exceptional failure with its own error string.

    :ivar status: terminal fire status — an **open string**, NOT the closed
        :data:`ScheduleFireStatus`. The engine is status-agnostic: it persists
        whatever the callback reports via :meth:`FireStore.finalize_success`, so a
        consumer with a richer terminal vocabulary (e.g. agent-wake's
        ``'yielded'`` / ``'fired_silent'`` / ``'skipped_busy'``) expresses it here
        and validates it in its OWN store's CHECK. The default store narrows the
        column to :data:`ScheduleFireStatus`; a different store may widen it.
        ``'failed'`` is the one value the engine itself interprets (→
        :meth:`FireStore.finalize_failed`).
    :ivar output: optional captured output payload (opaque JSON)
    :ivar latency_ms: optional end-to-end fire latency in milliseconds
    :ivar error: optional error string (set when ``status='failed'``)
    """

    status: str = "succeeded"
    output: dict[str, Any] | None = None
    latency_ms: int | None = None
    error: str | None = None
