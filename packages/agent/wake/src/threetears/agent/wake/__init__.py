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
from threetears.agent.wake.dispatch import detect_silent_prefix, dispatch_wake
from threetears.agent.wake.entities import (
    EncryptionService,
    WakeFireEntity,
    WakeScheduleEntity,
    WebhookSubscriptionEntity,
)
from threetears.agent.wake.migrations import register
from threetears.agent.wake.tick import DispatchCallback, wake_tick_job
from threetears.agent.wake.tools import (
    DEFAULT_MAX_SCHEDULES_PER_CONVERSATION,
    WakeRegistryClient,
    load_wake_schedule_create_tool,
    load_wake_schedule_delete_tool,
    load_wake_schedule_list_tool,
    load_wake_schedule_pause_tool,
    load_wake_schedule_resume_tool,
    load_wake_schedule_update_tool,
    load_wake_yield_tool,
    load_webhook_subscription_create_tool,
    load_webhook_subscription_delete_tool,
    load_webhook_subscription_list_tool,
    load_webhook_subscription_pause_tool,
    load_webhook_subscription_resume_tool,
    load_webhook_subscription_rotate_secret_tool,
    load_webhook_subscription_update_tool,
    validate_context_from_chain,
    validate_schedule_config,
)
from threetears.agent.wake.types import (
    DeliveryAdapter,
    DeliveryTarget,
    ExecutionMode,
    FireSource,
    FireStatus,
    HandlerCallback,
    HandlerCallbackResult,
    MissedFirePolicy,
    PreparedWakeContext,
    ScheduleStatus,
    ScheduleType,
    VerificationScheme,
    WakeDispatchResult,
    WakeTrigger,
    WebhookSubscriptionStatus,
)
from threetears.agent.wake.webhook_adapter import (
    WebhookReceiveResult,
    webhook_receive,
)

__all__ = [
    "DEFAULT_MAX_SCHEDULES_PER_CONVERSATION",
    "DeliveryAdapter",
    "DeliveryTarget",
    "DispatchCallback",
    "EncryptionService",
    "ExecutionMode",
    "FireSource",
    "FireStatus",
    "HandlerCallback",
    "HandlerCallbackResult",
    "MissedFirePolicy",
    "PreparedWakeContext",
    "ScheduleStatus",
    "ScheduleType",
    "VerificationScheme",
    "WakeDispatchResult",
    "WakeFireCollection",
    "WakeFireEntity",
    "WakeRegistryClient",
    "WakeScheduleCollection",
    "WakeScheduleEntity",
    "WakeTrigger",
    "WebhookReceiveResult",
    "WebhookSubscriptionCollection",
    "WebhookSubscriptionEntity",
    "WebhookSubscriptionStatus",
    "detect_silent_prefix",
    "dispatch_wake",
    "load_wake_schedule_create_tool",
    "load_wake_schedule_delete_tool",
    "load_wake_schedule_list_tool",
    "load_wake_schedule_pause_tool",
    "load_wake_schedule_resume_tool",
    "load_wake_schedule_update_tool",
    "load_wake_yield_tool",
    "load_webhook_subscription_create_tool",
    "load_webhook_subscription_delete_tool",
    "load_webhook_subscription_list_tool",
    "load_webhook_subscription_pause_tool",
    "load_webhook_subscription_resume_tool",
    "load_webhook_subscription_rotate_secret_tool",
    "load_webhook_subscription_update_tool",
    "register",
    "validate_context_from_chain",
    "validate_schedule_config",
    "wake_tick_job",
    "webhook_receive",
]
