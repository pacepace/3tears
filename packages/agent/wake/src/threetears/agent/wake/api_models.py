"""Pydantic request/response models for the wake REST API.

Per PLACEMENT §3.2 (2026-05-19 locked) the API contract lives platform-
side; each consumer (metallm in v1) imports these models into its
FastAPI router. One source of truth for the shape across products.

The models are import-safe -- they do NOT pull in ``asyncpg`` or
collection plumbing, so a consumer router can ``from
threetears.agent.wake.api_models import WakeScheduleResponse`` without
dragging in the platform's L3 layer. Tested by
:mod:`threetears.agent.wake.tests.unit.test_api_models`.

Per the 2026-05-19 wake-yield revision the response shapes drop the
pre-check / no_agent fields and gain:

- ``CreateWakeScheduleRequest`` / ``WakeScheduleResponse`` -- ``skill_id``,
  ``missed_fire_policy``.
- ``UpdateWakeScheduleRequest`` / ``UpdateWebhookSubscriptionRequest``
  -- new models for the PATCH endpoints.
- ``WakeFireResponse`` -- ``scheduled_fire_at`` + ``actual_fired_at``
  (alongside the existing ``fired_at`` alias for backwards-compat
  during the consumer migration).
- ``CreateWebhookSubscriptionRequest`` / ``WebhookSubscriptionResponse``
  -- ``default_skill_id``.

Spec ref: ``docs/agent-wake/shard-05-observability-and-models.md``
OBS-17 / OBS-18 + revision deltas at the top of that file.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "CreateWakeScheduleRequest",
    "CreateWebhookSubscriptionRequest",
    "CreateWebhookSubscriptionResponse",
    "UpdateWakeScheduleRequest",
    "UpdateWebhookSubscriptionRequest",
    "WakeFireListResponse",
    "WakeFireResponse",
    "WakeScheduleListResponse",
    "WakeScheduleResponse",
    "WebhookSubscriptionListResponse",
    "WebhookSubscriptionResponse",
]


# Reused literal aliases mirroring the L3 CHECK constraints + the
# :mod:`threetears.agent.wake.types` Literals. Duplicated here (rather
# than imported) so this module stays free of any cross-module
# coupling beyond pydantic / stdlib.
_ScheduleType = Literal[
    "daily_at",
    "every_n_hours",
    "random_within_window",
    "one_shot_at",
    "cron",
    "relative_delay",
    "interval",
]
_ExecutionMode = Literal["inline", "spawn"]
_MissedFirePolicy = Literal["coalesce", "catch_up"]
_ScheduleStatus = Literal["active", "paused", "expired"]
_WebhookStatus = Literal["active", "paused"]
_FireStatus = Literal[
    "dispatching",
    "fired",
    "fired_silent",
    "yielded",
    "skipped_busy",
    "skipped_rate_limit",
    "skipped_cap",
    "skipped_no_handler",
    "failed",
]
_FireSource = Literal["scheduled_tick", "webhook"]

# Format guard for ``verification_scheme`` (mirrors v005's
# ``^[a-z0-9_]+$`` length 1-64 CHECK). Vendor schemes (``'github'``,
# ``'stripe'``, ``'slack_signing'``, ...) register at runtime via
# :meth:`~threetears.channels.webhook.WebhookReceiver.register_verifier`,
# so the Pydantic surface accepts any slug-shaped value rather than
# pinning to a Literal that would defeat the registry. The
# receiver returns 400 for unregistered slugs at handle time.
_VERIFICATION_SCHEME_PATTERN = r"^[a-z0-9_]{1,64}$"


# extra='forbid' fails closed on unexpected fields so a client typo
# is caught at validation time rather than silently dropped. The
# consumer's router can relax via its own subclass when needed.
_STRICT = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Wake schedules
# ---------------------------------------------------------------------------


class CreateWakeScheduleRequest(BaseModel):
    """POST body for ``/conversations/{id}/wake-schedules``.

    Maps onto the agent-tool layer's ``wake_schedule_create`` (shard
    04). Fields are validated by the consumer's router against the
    same ``validate_schedule_config`` helper the tool layer uses.
    """

    model_config = _STRICT

    schedule_type: _ScheduleType
    schedule_config: dict[str, Any]
    execution_mode: _ExecutionMode = "inline"
    missed_fire_policy: _MissedFirePolicy = "coalesce"
    task_prompt: str | None = None
    name: str | None = None
    skill_id: UUID | None = None
    context_from_schedule_id: UUID | None = None


class UpdateWakeScheduleRequest(BaseModel):
    """PATCH body for ``/conversations/{id}/wake-schedules/{schedule_id}``.

    Every field optional; the consumer's router applies only the
    fields the client passes. ``status='expired'`` is server-set only
    (the one-shot transition) -- the router rejects an explicit
    ``'expired'`` from clients.
    """

    model_config = _STRICT

    status: Literal["active", "paused"] | None = None
    name: str | None = None
    task_prompt: str | None = None
    schedule_type: _ScheduleType | None = None
    schedule_config: dict[str, Any] | None = None
    execution_mode: _ExecutionMode | None = None
    missed_fire_policy: _MissedFirePolicy | None = None
    skill_id: UUID | None = None
    detach_skill: bool = False
    context_from_schedule_id: UUID | None = None
    detach_context_from: bool = False


class WakeScheduleResponse(BaseModel):
    """GET / POST / PATCH response shape for one wake schedule."""

    model_config = _STRICT

    schedule_id: UUID
    conversation_id: UUID
    user_id: UUID
    agent_id: UUID
    schedule_type: str
    schedule_config: dict[str, Any]
    task_prompt: str | None
    execution_mode: str
    status: _ScheduleStatus
    next_fire_at: datetime | None
    last_fired_at: datetime | None
    name: str | None
    missed_fire_policy: _MissedFirePolicy
    skill_id: UUID | None
    context_from_schedule_id: UUID | None
    date_created: datetime
    date_updated: datetime


class WakeScheduleListResponse(BaseModel):
    """GET response shape for the schedule list endpoint."""

    model_config = _STRICT

    schedules: list[WakeScheduleResponse]
    total_count: int


# ---------------------------------------------------------------------------
# Wake fires (history)
# ---------------------------------------------------------------------------


class WakeFireResponse(BaseModel):
    """One row from the wake fire history.

    ``scheduled_fire_at`` is ``None`` for webhook fires (no schedule
    instant); ``actual_fired_at`` is always set. ``schedule_id`` is
    ``None`` for webhook fires; ``webhook_subscription_id`` is
    ``None`` for scheduled fires (CHECK-constrained XOR at the L3
    layer).
    """

    model_config = _STRICT

    fire_id: UUID
    schedule_id: UUID | None
    webhook_subscription_id: UUID | None
    conversation_id: UUID
    scheduled_fire_at: datetime | None
    actual_fired_at: datetime
    fire_source: _FireSource
    status: _FireStatus
    output_text: str | None
    latency_ms: int | None
    error: str | None
    display_suppressed: bool
    date_created: datetime


class WakeFireListResponse(BaseModel):
    """GET response shape for the fire history endpoint."""

    model_config = _STRICT

    fires: list[WakeFireResponse]
    total_count: int


# ---------------------------------------------------------------------------
# Webhook subscriptions
# ---------------------------------------------------------------------------


class CreateWebhookSubscriptionRequest(BaseModel):
    """POST body for ``/conversations/{id}/webhook-subscriptions``.

    The plaintext HMAC secret is generated server-side and returned
    ONCE in :class:`CreateWebhookSubscriptionResponse`. Subsequent
    GET / list calls return only :class:`WebhookSubscriptionResponse`
    (no secret).
    """

    model_config = _STRICT

    name: str | None = None
    task_prompt_template: str
    execution_mode: _ExecutionMode = "inline"
    default_skill_id: UUID | None = None
    allowed_source_pattern: str | None = None
    rate_limit_per_minute: int | None = None


class UpdateWebhookSubscriptionRequest(BaseModel):
    """PATCH body for one webhook subscription."""

    model_config = _STRICT

    status: _WebhookStatus | None = None
    name: str | None = None
    task_prompt_template: str | None = None
    execution_mode: _ExecutionMode | None = None
    default_skill_id: UUID | None = None
    detach_default_skill: bool = False
    allowed_source_pattern: str | None = None
    rate_limit_per_minute: int | None = None


class WebhookSubscriptionResponse(BaseModel):
    """GET / PATCH response shape for one webhook subscription.

    Does NOT carry the plaintext secret -- that's only on the
    :class:`CreateWebhookSubscriptionResponse` (and on a future
    rotate-secret response).
    """

    model_config = _STRICT

    subscription_id: UUID
    conversation_id: UUID
    user_id: UUID
    agent_id: UUID
    name: str | None
    execution_mode: str
    status: _WebhookStatus
    task_prompt_template: str | None
    verification_scheme: str = Field(
        pattern=_VERIFICATION_SCHEME_PATTERN,
    )
    default_skill_id: UUID | None
    allowed_source_pattern: str | None
    rate_limit_per_minute: int | None
    last_fired_at: datetime | None
    date_created: datetime
    date_updated: datetime


class CreateWebhookSubscriptionResponse(WebhookSubscriptionResponse):
    """POST response -- carries the display-once plaintext secret."""

    model_config = _STRICT

    secret_plaintext: str


class WebhookSubscriptionListResponse(BaseModel):
    """GET response shape for the subscription list endpoint.

    Listings never carry ``secret_plaintext`` -- only the create /
    rotate paths expose the plaintext.
    """

    model_config = _STRICT

    subscriptions: list[WebhookSubscriptionResponse]
    total_count: int
