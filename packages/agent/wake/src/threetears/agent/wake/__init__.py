"""Agent-wake package -- long-running-agent foundation.

Shard 01 lands the schema + Collection layer for the three platform
tables (per the 2026-05-19 PLACEMENT revision that collapsed the
six-table draft into three):

- ``agent_wake_schedules`` -- one row per active wake schedule for a
  conversation. Nullable ``skill_id`` FK to
  ``agent_skills.skill_id``.
- ``wake_fires`` -- one row per wake fire (history).
- ``webhook_subscriptions`` -- one row per inbound HTTP webhook
  subscription. Nullable ``default_skill_id`` FK to
  ``agent_skills.skill_id``.

Subsequent shards add:

- shard 02 -- tick engine + ``_compute_next_fire_at``.
- shard 03 -- ``WakeTrigger`` + ``dispatch_wake`` convergence point.
- shard 04 -- agent tools (CRUD) + webhook adapter glue.
- shard 05 -- observability + rate-limit + Pydantic API models.
- shard 06 -- ``3tears-channels.WebhookReceiver`` framework.

The public surface here re-exports the entity / collection / type /
migration registration triad. Tools, Pydantic models, observability
helpers, and the dispatch surface arrive in later shards.

Version is sourced from the installed package metadata so a future
release that bumps ``pyproject.toml`` without touching this file
cannot drift the runtime ``__version__`` reporting.
"""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("3tears-agent-wake")
except _PackageNotFoundError:  # pragma: no cover - dev fallback
    __version__ = "unknown"

from threetears.agent.wake.collections import (
    WakeFireCollection,
    WakeScheduleCollection,
    WebhookSubscriptionCollection,
)
from threetears.agent.wake.entities import (
    EncryptionService,
    WakeFireEntity,
    WakeScheduleEntity,
    WebhookSubscriptionEntity,
)
from threetears.agent.wake.migrations import register
from threetears.agent.wake.tick import DispatchCallback, wake_tick_job
from threetears.agent.wake.types import (
    DeliveryTarget,
    ExecutionMode,
    FireSource,
    FireStatus,
    MissedFirePolicy,
    ScheduleStatus,
    ScheduleType,
    VerificationScheme,
    WakeDispatchResult,
    WakeTrigger,
    WebhookSubscriptionStatus,
)

__all__ = [
    "DeliveryTarget",
    "DispatchCallback",
    "EncryptionService",
    "ExecutionMode",
    "FireSource",
    "FireStatus",
    "MissedFirePolicy",
    "ScheduleStatus",
    "ScheduleType",
    "VerificationScheme",
    "WakeDispatchResult",
    "WakeFireCollection",
    "WakeFireEntity",
    "WakeScheduleCollection",
    "WakeScheduleEntity",
    "WakeTrigger",
    "WebhookSubscriptionCollection",
    "WebhookSubscriptionEntity",
    "WebhookSubscriptionStatus",
    "register",
    "wake_tick_job",
]
