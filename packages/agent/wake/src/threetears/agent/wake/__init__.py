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

from threetears.agent.wake.api_models import (
    CreateWakeScheduleRequest,
    CreateWebhookSubscriptionRequest,
    CreateWebhookSubscriptionResponse,
    UpdateWakeScheduleRequest,
    UpdateWebhookSubscriptionRequest,
    WakeFireListResponse,
    WakeFireResponse,
    WakeScheduleListResponse,
    WakeScheduleResponse,
    WebhookSubscriptionListResponse,
    WebhookSubscriptionResponse,
)
from threetears.agent.wake.collections import (
    WakeFireCollection,
    WakeScheduleCollection,
    WebhookSubscriptionCollection,
)
from threetears.agent.wake.config import (
    DEFAULT_HTTP_ALLOWED_HOSTS,
    DEFAULT_LOKI_NAMED_QUERIES,
    DEFAULT_MAX_EMAIL_PER_RECIPIENT_PER_HOUR,
    DEFAULT_MAX_FIRES_PER_CONV_PER_DAY,
    DEFAULT_MAX_FIRES_PER_USER_PER_DAY,
    DEFAULT_MAX_WEBHOOK_FIRES_PER_SUBSCRIPTION_PER_HOUR,
    DEFAULT_POSTGRES_NAMED_QUERIES,
    DEFAULT_WAKE_CONFIG,
    WakeConfig,
)
from threetears.agent.wake.dispatch import detect_silent_prefix, dispatch_wake
from threetears.agent.wake.entities import (
    EncryptionService,
    WakeFireEntity,
    WakeScheduleEntity,
    WebhookSubscriptionEntity,
)
from threetears.agent.wake.events import (
    EVENT_DELIVERY_ATTEMPT,
    EVENT_DELIVERY_FAILED,
    EVENT_DELIVERY_SKIPPED_SILENT,
    EVENT_DELIVERY_SUCCESS,
    EVENT_FIRE_DISPATCHED,
    EVENT_FIRE_DRIFT,
    EVENT_FIRE_FAILED,
    EVENT_FIRE_RATE_LIMITED,
    EVENT_FIRE_SILENT,
    EVENT_FIRE_SKIPPED_BUSY,
    EVENT_FIRE_YIELDED,
    EVENT_SCHEDULE_CAP_REJECT,
    EVENT_TICK_COMPLETED,
    EVENT_TICK_STARTED,
    EVENT_WEBHOOK_AUTH_FAILED,
    EVENT_WEBHOOK_RATE_LIMITED,
    EVENT_WEBHOOK_RECEIVED,
    EVENT_WEBHOOK_REJECTED,
)
from threetears.agent.wake.metrics import (
    FORBIDDEN_LABEL_NAMES,
    WAKE_DELIVERY_TOTAL,
    WAKE_DRIFT_SECONDS,
    WAKE_FAILURES_TOTAL,
    WAKE_FIRES_TOTAL,
    WAKE_LABEL_SETS,
    WAKE_PROMETHEUS_NAMES,
    WAKE_RATE_LIMIT_REJECTIONS_TOTAL,
    WAKE_SCHEDULE_CAP_REJECTIONS_TOTAL,
    WAKE_TICK_DURATION_SECONDS,
    WAKE_WEBHOOK_RECEIVED_TOTAL,
    WAKE_YIELD_DURATION_SECONDS,
    WakeMetricsEmitter,
    get_wake_emitter,
    reset_wake_emitter_for_testing,
)
from threetears.agent.wake.migrations import register
from threetears.agent.wake.rate_limit import (
    RATE_LIMIT_WINDOW_HOURS,
    RateLimitScope,
    _check_active_schedule_cap,
    _check_rate_limit,
)
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
    "DEFAULT_HTTP_ALLOWED_HOSTS",
    "DEFAULT_LOKI_NAMED_QUERIES",
    "DEFAULT_MAX_EMAIL_PER_RECIPIENT_PER_HOUR",
    "DEFAULT_MAX_FIRES_PER_CONV_PER_DAY",
    "DEFAULT_MAX_FIRES_PER_USER_PER_DAY",
    "DEFAULT_MAX_SCHEDULES_PER_CONVERSATION",
    "DEFAULT_MAX_WEBHOOK_FIRES_PER_SUBSCRIPTION_PER_HOUR",
    "DEFAULT_POSTGRES_NAMED_QUERIES",
    "DEFAULT_WAKE_CONFIG",
    "EVENT_DELIVERY_ATTEMPT",
    "EVENT_DELIVERY_FAILED",
    "EVENT_DELIVERY_SKIPPED_SILENT",
    "EVENT_DELIVERY_SUCCESS",
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
    "FORBIDDEN_LABEL_NAMES",
    "RATE_LIMIT_WINDOW_HOURS",
    "RateLimitScope",
    "WAKE_DELIVERY_TOTAL",
    "WAKE_DRIFT_SECONDS",
    "WAKE_FAILURES_TOTAL",
    "WAKE_FIRES_TOTAL",
    "WAKE_LABEL_SETS",
    "WAKE_PROMETHEUS_NAMES",
    "WAKE_RATE_LIMIT_REJECTIONS_TOTAL",
    "WAKE_SCHEDULE_CAP_REJECTIONS_TOTAL",
    "WAKE_TICK_DURATION_SECONDS",
    "WAKE_WEBHOOK_RECEIVED_TOTAL",
    "WAKE_YIELD_DURATION_SECONDS",
    "CreateWakeScheduleRequest",
    "CreateWebhookSubscriptionRequest",
    "CreateWebhookSubscriptionResponse",
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
    "UpdateWakeScheduleRequest",
    "UpdateWebhookSubscriptionRequest",
    "VerificationScheme",
    "WakeConfig",
    "WakeDispatchResult",
    "WakeFireCollection",
    "WakeFireEntity",
    "WakeFireListResponse",
    "WakeFireResponse",
    "WakeMetricsEmitter",
    "WakeRegistryClient",
    "WakeScheduleCollection",
    "WakeScheduleEntity",
    "WakeScheduleListResponse",
    "WakeScheduleResponse",
    "WakeTrigger",
    "WebhookReceiveResult",
    "WebhookSubscriptionCollection",
    "WebhookSubscriptionEntity",
    "WebhookSubscriptionListResponse",
    "WebhookSubscriptionResponse",
    "WebhookSubscriptionStatus",
    "_check_active_schedule_cap",
    "_check_rate_limit",
    "detect_silent_prefix",
    "dispatch_wake",
    "get_wake_emitter",
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
    "reset_wake_emitter_for_testing",
    "validate_context_from_chain",
    "validate_schedule_config",
    "wake_tick_job",
    "webhook_receive",
]
