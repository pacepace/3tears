"""Structured-log event-name constants for the scheduled-jobs lifecycle.

Generalized from :mod:`threetears.agent.wake.events` -- the
webhook/silent/yielded/rate-limit/cap events were agent-specific and are
dropped; what remains is the payload-agnostic tick + fire + drift
lifecycle. Centralising these as string constants prevents typo drift
across emission sites + log-query dashboards. The package emits them
verbatim via ``log.info(EVENT_NAME, extra={"extra_data": {...}})``.

PII discipline: NEVER include opaque ``payload`` content in event
payloads -- the payload is the consumer's domain data and may carry PII.
The tick engine logs only ids, the ``kind`` discriminator, the
``schedule_type``, and timing fields.
"""

from __future__ import annotations

__all__ = [
    "EVENT_FIRE_DISPATCHED",
    "EVENT_FIRE_DRIFT",
    "EVENT_FIRE_FAILED",
    "EVENT_FIRE_REAPED",
    "EVENT_FIRE_SKIPPED_BUSY",
    "EVENT_TICK_COMPLETED",
    "EVENT_TICK_STARTED",
]


# Tick lifecycle -- emitted by :mod:`threetears.scheduled_jobs.tick`.
EVENT_TICK_STARTED: str = "3tears.scheduled_jobs.tick.started"
EVENT_TICK_COMPLETED: str = "3tears.scheduled_jobs.tick.completed"

# Per-fire lifecycle -- emitted by :mod:`threetears.scheduled_jobs.tick`.
# ``EVENT_FIRE_DISPATCHED`` is the happy-path event; ``EVENT_FIRE_FAILED``
# covers a raised / returned-failed dispatch; ``EVENT_FIRE_SKIPPED_BUSY``
# is the per-row optimistic-CAS miss (another tick already claimed it).
EVENT_FIRE_DISPATCHED: str = "3tears.scheduled_jobs.fire.dispatched"
EVENT_FIRE_FAILED: str = "3tears.scheduled_jobs.fire.failed"
EVENT_FIRE_SKIPPED_BUSY: str = "3tears.scheduled_jobs.fire.skipped_busy"

# Drift -- emitted when the actual fire instant differs from the
# scheduled instant by more than a tolerance threshold.
EVENT_FIRE_DRIFT: str = "3tears.scheduled_jobs.fire.drift"

# Reaper -- emitted (with a count) when the tick sweep reclaims fire rows
# stuck in ``'dispatching'`` (a pod died mid-dispatch) to ``'failed'``.
EVENT_FIRE_REAPED: str = "3tears.scheduled_jobs.fire.reaped"
