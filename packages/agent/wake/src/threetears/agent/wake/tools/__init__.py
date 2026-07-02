"""Agent-facing tool factories for wake schedules + webhook subscriptions.

Shard 04 surface: 13 schedule / webhook CRUD tools plus the
``wake_yield`` cooperative-interrupt tool (14 total) plus the
``webhook_receive`` adapter exposed via
:mod:`threetears.agent.wake.webhook_adapter`.

Tools mirror the :mod:`threetears.agent.skills.tools` precedent --
factory functions return LangChain ``BaseTool`` instances bound to a
``(conversation_id, user_id, agent_id)`` actor triple plus the wake
Collections + a consumer-supplied :class:`WakeRegistryClient` Protocol
implementation. The LLM never sees identity fields in the input schema.

Per-type config validators (``_validate_schedule_config``,
``_validate_context_from_chain``) are importable module-level symbols
so the product's REST surface can reuse the exact same validation the
tool layer enforces (Requirement TOOL-11 in
``docs/agent-wake/shard-04-agent-tools-and-webhook-adapter.md``).

Spec ref: ``docs/agent-wake/shard-04-agent-tools-and-webhook-adapter.md``
+ ``docs/long_running/PLACEMENT.md`` §1.1 / §1.3 / §1.5 / §1.6 /
§1.9 / §8.5.1.
"""

from __future__ import annotations

from threetears.agent.wake.tools.resolve import (
    parse_schedule_id,
    parse_subscription_id,
)
from threetears.agent.wake.tools.schedule_tools import (
    DEFAULT_MAX_SCHEDULES_PER_CONVERSATION,
    NAME_MAX_LEN,
    TASK_PROMPT_MAX_LEN,
    ScheduleCreateInput,
    ScheduleDeleteInput,
    ScheduleIdInput,
    ScheduleListInput,
    ScheduleUpdateInput,
    WakeRegistryClient,
    WakeYieldProbe,
    WakeYieldSetter,
    load_wake_schedule_create_tool,
    load_wake_schedule_delete_tool,
    load_wake_schedule_list_tool,
    load_wake_schedule_pause_tool,
    load_wake_schedule_resume_tool,
    load_wake_schedule_update_tool,
    load_wake_yield_tool,
)
from threetears.agent.wake.tools.validators import (
    CONTEXT_FROM_MAX_DEPTH,
    SUPPORTED_SCHEDULE_TYPES,
    validate_context_from_chain,
    validate_schedule_config,
)
from threetears.agent.wake.tools.webhook_tools import (
    PAYLOAD_TEMPLATE_MAX_BYTES,
    SECRET_BYTE_LEN,
    WebhookSubscriptionCreateInput,
    WebhookSubscriptionIdInput,
    WebhookSubscriptionUpdateInput,
    load_webhook_subscription_create_tool,
    load_webhook_subscription_delete_tool,
    load_webhook_subscription_list_tool,
    load_webhook_subscription_pause_tool,
    load_webhook_subscription_resume_tool,
    load_webhook_subscription_rotate_secret_tool,
    load_webhook_subscription_update_tool,
)

__all__ = [
    "CONTEXT_FROM_MAX_DEPTH",
    "DEFAULT_MAX_SCHEDULES_PER_CONVERSATION",
    "NAME_MAX_LEN",
    "PAYLOAD_TEMPLATE_MAX_BYTES",
    "SECRET_BYTE_LEN",
    "SUPPORTED_SCHEDULE_TYPES",
    "TASK_PROMPT_MAX_LEN",
    "ScheduleCreateInput",
    "ScheduleDeleteInput",
    "ScheduleIdInput",
    "ScheduleListInput",
    "ScheduleUpdateInput",
    "WakeRegistryClient",
    "WakeYieldProbe",
    "WakeYieldSetter",
    "WebhookSubscriptionCreateInput",
    "WebhookSubscriptionIdInput",
    "WebhookSubscriptionUpdateInput",
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
    "parse_schedule_id",
    "parse_subscription_id",
    "validate_context_from_chain",
    "validate_schedule_config",
]
