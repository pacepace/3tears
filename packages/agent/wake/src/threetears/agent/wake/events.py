"""Loki structured-log event-name constants for the wake lifecycle.

Centralising these as string constants prevents typo drift across
emission sites + LogQL dashboards. The shape mirrors
:mod:`threetears.agent.skills.metric_names` -- bare canonical names
the consumer (and the wake package itself) emit verbatim via
``log.info(EVENT_NAME, extra={"extra_data": {...}})``.

Spec ref: ``docs/agent-wake/shard-05-observability-and-models.md``
OBS-11 / OBS-12; PLACEMENT §1.4 (silent), §1.7 (drift), §8.5.1
(yielded).

PII discipline: NEVER include ``task_prompt`` content in event
payloads. Conversation messages are the canonical source of truth for
what was said. Tested by
:mod:`threetears.agent.wake.tests.unit.test_metrics_cardinality`.
"""

from __future__ import annotations

__all__ = [
    "EVENT_FIRE_DISPATCHED",
    "EVENT_FIRE_DRIFT",
    "EVENT_FIRE_FAILED",
    "EVENT_FIRE_RATE_LIMITED",
    "EVENT_FIRE_SILENT",
    "EVENT_FIRE_SKIPPED_BUSY",
    "EVENT_FIRE_YIELDED",
    "EVENT_SCHEDULE_CAP_REJECT",
    "EVENT_TICK_COMPLETED",
    "EVENT_TICK_STARTED",
    "EVENT_WEBHOOK_AUTH_FAILED",
    "EVENT_WEBHOOK_RATE_LIMITED",
    "EVENT_WEBHOOK_RECEIVED",
    "EVENT_WEBHOOK_REJECTED",
]


# Tick lifecycle -- emitted by :mod:`threetears.agent.wake.tick`.
EVENT_TICK_STARTED: str = "3tears.agent_wake.tick.started"
EVENT_TICK_COMPLETED: str = "3tears.agent_wake.tick.completed"

# Per-fire lifecycle -- emitted by :mod:`threetears.agent.wake.dispatch`
# and :mod:`threetears.agent.wake.tick`. ``EVENT_FIRE_DISPATCHED`` is
# the terminal happy-path event for visible fires; the four
# ``EVENT_FIRE_*`` companions cover skips / silent / yielded / failed
# without overloading the dispatched event.
EVENT_FIRE_DISPATCHED: str = "3tears.agent_wake.fire.dispatched"
EVENT_FIRE_SILENT: str = "3tears.agent_wake.fire.silent"
EVENT_FIRE_YIELDED: str = "3tears.agent_wake.fire.yielded"
EVENT_FIRE_SKIPPED_BUSY: str = "3tears.agent_wake.fire.skipped_busy"
EVENT_FIRE_RATE_LIMITED: str = "3tears.agent_wake.fire.rate_limited"
EVENT_FIRE_FAILED: str = "3tears.agent_wake.fire.failed"

# Drift -- emitted when the actual fire instant differs from the
# scheduled instant by more than a tolerance threshold. PLACEMENT §1.8
# (drift recorded but not actioned in v1).
EVENT_FIRE_DRIFT: str = "3tears.agent_wake.fire.drift"

# Per-conv cap rejection -- emitted by the agent-tool layer when a
# ``wake_schedule_create`` call would push the conversation past
# :data:`threetears.agent.wake.config.DEFAULT_MAX_SCHEDULES_PER_CONVERSATION`.
EVENT_SCHEDULE_CAP_REJECT: str = "3tears.agent_wake.schedule_cap.reject"

# Webhook receiver lifecycle -- emitted by
# :mod:`threetears.agent.wake.webhook_adapter`.
EVENT_WEBHOOK_RECEIVED: str = "3tears.agent_wake.webhook.received"
EVENT_WEBHOOK_AUTH_FAILED: str = "3tears.agent_wake.webhook.auth_failed"
EVENT_WEBHOOK_RATE_LIMITED: str = "3tears.agent_wake.webhook.rate_limited"
EVENT_WEBHOOK_REJECTED: str = "3tears.agent_wake.webhook.rejected"
