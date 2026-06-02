"""Unit tests for the Pydantic request/response models.

Coverage:

- Every model round-trips ``model_dump`` -> ``model_validate``
  losslessly with realistic shapes.
- ``extra='forbid'`` is enforced (unknown fields raise).
- Defaults match the 2026-05-19 revision deltas (no ``no_agent`` /
  ``pre_check_*`` fields; ``skill_id`` / ``missed_fire_policy`` /
  ``scheduled_fire_at`` present).
- The list responses serialise nested rows correctly.

The tests are pure ``pydantic.BaseModel`` exercises -- no DB, no
collection plumbing -- so they execute in sub-second time on a clean
checkout.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

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


# ---------------------------------------------------------------------------
# Wake schedule
# ---------------------------------------------------------------------------


def test_create_wake_schedule_request_round_trip_with_skill() -> None:
    """``skill_id`` + ``missed_fire_policy`` survive a dump+validate cycle."""
    skill_id = uuid4()
    payload = CreateWakeScheduleRequest(
        schedule_type="daily_at",
        schedule_config={"time": "09:00", "tz": "UTC"},
        execution_mode="inline",
        missed_fire_policy="coalesce",
        task_prompt="Check the dashboard",
        name="Morning check",
        skill_id=skill_id,
    )
    dumped = payload.model_dump()
    again = CreateWakeScheduleRequest.model_validate(dumped)
    assert again.skill_id == skill_id
    assert again.missed_fire_policy == "coalesce"


def test_create_wake_schedule_request_rejects_pre_check_fields() -> None:
    """``no_agent`` / ``pre_check_type`` are dropped (extra='forbid')."""
    with pytest.raises(ValidationError):
        CreateWakeScheduleRequest.model_validate(
            {
                "schedule_type": "daily_at",
                "schedule_config": {"time": "09:00"},
                "no_agent": True,
            }
        )
    with pytest.raises(ValidationError):
        CreateWakeScheduleRequest.model_validate(
            {
                "schedule_type": "daily_at",
                "schedule_config": {"time": "09:00"},
                "pre_check_type": "http_get",
            }
        )


def test_update_wake_schedule_request_all_fields_optional() -> None:
    """An empty body is valid (the consumer's router can no-op)."""
    instance = UpdateWakeScheduleRequest.model_validate({})
    assert instance.status is None
    assert instance.skill_id is None
    assert instance.detach_skill is False


def test_update_wake_schedule_request_supports_detach_flags() -> None:
    """Explicit ``detach_*`` flags survive the round trip."""
    instance = UpdateWakeScheduleRequest(
        detach_skill=True,
        detach_context_from=True,
    )
    again = UpdateWakeScheduleRequest.model_validate(instance.model_dump())
    assert again.detach_skill is True
    assert again.detach_context_from is True


def test_wake_schedule_response_round_trip() -> None:
    """The response shape carries every column the spec requires."""
    schedule_id = uuid4()
    response = WakeScheduleResponse(
        schedule_id=schedule_id,
        conversation_id=uuid4(),
        user_id=uuid4(),
        agent_id=uuid4(),
        schedule_type="cron",
        schedule_config={"cron": "0 9 * * *"},
        task_prompt=None,
        execution_mode="inline",
        status="active",
        next_fire_at=datetime(2026, 6, 1, 9, 0, tzinfo=UTC),
        last_fired_at=None,
        name="cron schedule",
        missed_fire_policy="coalesce",
        skill_id=None,
        context_from_schedule_id=None,
        include_conversation_history=True,
        date_created=datetime.now(UTC),
        date_updated=datetime.now(UTC),
    )
    dumped = response.model_dump()
    again = WakeScheduleResponse.model_validate(dumped)
    assert again.schedule_id == schedule_id
    assert again.missed_fire_policy == "coalesce"
    assert again.status == "active"
    assert again.include_conversation_history is True


def test_wake_schedule_list_response_round_trip() -> None:
    """The list shape serialises nested schedule rows."""
    row = WakeScheduleResponse(
        schedule_id=uuid4(),
        conversation_id=uuid4(),
        user_id=uuid4(),
        agent_id=uuid4(),
        schedule_type="daily_at",
        schedule_config={"time": "09:00"},
        task_prompt=None,
        execution_mode="inline",
        status="paused",
        next_fire_at=None,
        last_fired_at=None,
        name=None,
        missed_fire_policy="coalesce",
        skill_id=None,
        context_from_schedule_id=None,
        include_conversation_history=False,
        date_created=datetime.now(UTC),
        date_updated=datetime.now(UTC),
    )
    wrapper = WakeScheduleListResponse(schedules=[row], total_count=1)
    again = WakeScheduleListResponse.model_validate(wrapper.model_dump())
    assert again.total_count == 1
    assert len(again.schedules) == 1
    assert again.schedules[0].include_conversation_history is False


# ---------------------------------------------------------------------------
# Wake fire (history)
# ---------------------------------------------------------------------------


def test_wake_fire_response_round_trip_with_drift_fields() -> None:
    """``scheduled_fire_at`` + ``actual_fired_at`` are first-class fields."""
    response = WakeFireResponse(
        fire_id=uuid4(),
        schedule_id=uuid4(),
        webhook_subscription_id=None,
        conversation_id=uuid4(),
        scheduled_fire_at=datetime(2026, 6, 1, 9, 0, tzinfo=UTC),
        actual_fired_at=datetime(2026, 6, 1, 9, 0, 5, tzinfo=UTC),
        fire_source="scheduled_tick",
        status="fired",
        output_text="ok",
        latency_ms=120,
        error=None,
        display_suppressed=False,
        date_created=datetime.now(UTC),
    )
    again = WakeFireResponse.model_validate(response.model_dump())
    assert again.scheduled_fire_at is not None
    assert again.actual_fired_at >= again.scheduled_fire_at
    assert again.status == "fired"


def test_wake_fire_response_rejects_pre_check_output() -> None:
    """``pre_check_output`` is gone post-revision."""
    with pytest.raises(ValidationError):
        WakeFireResponse.model_validate(
            {
                "fire_id": str(uuid4()),
                "schedule_id": str(uuid4()),
                "webhook_subscription_id": None,
                "conversation_id": str(uuid4()),
                "scheduled_fire_at": None,
                "actual_fired_at": datetime.now(UTC).isoformat(),
                "fire_source": "scheduled_tick",
                "status": "fired",
                "output_text": None,
                "latency_ms": None,
                "error": None,
                "display_suppressed": False,
                "date_created": datetime.now(UTC).isoformat(),
                "pre_check_output": "should be rejected",
            }
        )


def test_wake_fire_list_response_round_trip() -> None:
    """List wrapper carries nested fires."""
    row = WakeFireResponse(
        fire_id=uuid4(),
        schedule_id=None,
        webhook_subscription_id=uuid4(),
        conversation_id=uuid4(),
        scheduled_fire_at=None,
        actual_fired_at=datetime.now(UTC),
        fire_source="webhook",
        status="fired_silent",
        output_text="[SILENT] nothing to report",
        latency_ms=80,
        error=None,
        display_suppressed=True,
        date_created=datetime.now(UTC),
    )
    wrapper = WakeFireListResponse(fires=[row], total_count=1)
    again = WakeFireListResponse.model_validate(wrapper.model_dump())
    assert again.fires[0].fire_source == "webhook"
    assert again.fires[0].status == "fired_silent"


# ---------------------------------------------------------------------------
# Webhook subscription
# ---------------------------------------------------------------------------


def test_create_webhook_subscription_request_round_trip_with_default_skill() -> None:
    """``default_skill_id`` is the post-revision field."""
    skill_id = uuid4()
    payload = CreateWebhookSubscriptionRequest(
        name="github-push",
        task_prompt_template="Push to {{event.repository.full_name}}",
        execution_mode="inline",
        default_skill_id=skill_id,
        allowed_source_pattern=r"^140\.82\.\d+\.\d+$",
        rate_limit_per_minute=30,
    )
    again = CreateWebhookSubscriptionRequest.model_validate(payload.model_dump())
    assert again.default_skill_id == skill_id
    assert again.rate_limit_per_minute == 30


def test_create_webhook_subscription_request_rejects_pre_check_fields() -> None:
    """Pre-check fields are gone post-revision."""
    with pytest.raises(ValidationError):
        CreateWebhookSubscriptionRequest.model_validate(
            {
                "task_prompt_template": "x",
                "pre_check_type": "http_get",
            }
        )


def test_update_webhook_subscription_request_empty_body_is_valid() -> None:
    """Patch with no fields is valid (router decides what to do)."""
    instance = UpdateWebhookSubscriptionRequest.model_validate({})
    assert instance.status is None
    assert instance.default_skill_id is None
    assert instance.detach_default_skill is False


def test_webhook_subscription_response_round_trip() -> None:
    """Response shape mirrors the entity (minus secret_ciphertext)."""
    response = WebhookSubscriptionResponse(
        subscription_id=uuid4(),
        conversation_id=uuid4(),
        user_id=uuid4(),
        agent_id=uuid4(),
        name="github push",
        execution_mode="inline",
        status="active",
        task_prompt_template="x",
        verification_scheme="generic_hmac_sha256",
        default_skill_id=None,
        allowed_source_pattern=None,
        rate_limit_per_minute=None,
        last_fired_at=None,
        date_created=datetime.now(UTC),
        date_updated=datetime.now(UTC),
    )
    again = WebhookSubscriptionResponse.model_validate(response.model_dump())
    assert again.verification_scheme == "generic_hmac_sha256"
    assert "secret_plaintext" not in again.model_dump()


def test_webhook_subscription_response_accepts_vendor_verification_scheme() -> None:
    """Vendor scheme slugs (registered at runtime via the receiver's
    :meth:`~threetears.channels.webhook.WebhookReceiver.register_verifier`)
    pass the Pydantic format guard. The model accepts any slug-shaped
    value (post-v005 the schema does too); the receiver returns 400
    at handle time when the scheme is not registered.
    """
    response = WebhookSubscriptionResponse(
        subscription_id=uuid4(),
        conversation_id=uuid4(),
        user_id=uuid4(),
        agent_id=uuid4(),
        name="github push",
        execution_mode="inline",
        status="active",
        task_prompt_template="x",
        verification_scheme="github",
        default_skill_id=None,
        allowed_source_pattern=None,
        rate_limit_per_minute=None,
        last_fired_at=None,
        date_created=datetime.now(UTC),
        date_updated=datetime.now(UTC),
    )
    again = WebhookSubscriptionResponse.model_validate(response.model_dump())
    assert again.verification_scheme == "github"


def test_webhook_subscription_response_rejects_invalid_scheme_shape() -> None:
    """The slug format guard (``^[a-z0-9_]{1,64}$``) rejects punctuation
    and uppercase. Mirrors v005's CHECK constraint so the API + DB
    enforce the same shape.
    """
    base_kwargs: dict[str, Any] = {
        "subscription_id": uuid4(),
        "conversation_id": uuid4(),
        "user_id": uuid4(),
        "agent_id": uuid4(),
        "name": None,
        "execution_mode": "inline",
        "status": "active",
        "task_prompt_template": None,
        "default_skill_id": None,
        "allowed_source_pattern": None,
        "rate_limit_per_minute": None,
        "last_fired_at": None,
        "date_created": datetime.now(UTC),
        "date_updated": datetime.now(UTC),
    }
    for bad in ("GitHub", "with-hyphen", "with space", "", "a" * 65):
        with pytest.raises(ValidationError, match="String should match pattern"):
            WebhookSubscriptionResponse(verification_scheme=bad, **base_kwargs)


def test_create_webhook_subscription_response_carries_plaintext() -> None:
    """The create response surface is the only one that exposes the secret."""
    response = CreateWebhookSubscriptionResponse(
        subscription_id=uuid4(),
        conversation_id=uuid4(),
        user_id=uuid4(),
        agent_id=uuid4(),
        name=None,
        execution_mode="inline",
        status="active",
        task_prompt_template="x",
        verification_scheme="generic_hmac_sha256",
        default_skill_id=None,
        allowed_source_pattern=None,
        rate_limit_per_minute=None,
        last_fired_at=None,
        date_created=datetime.now(UTC),
        date_updated=datetime.now(UTC),
        secret_plaintext="abc123",
    )
    assert response.secret_plaintext == "abc123"
    again = CreateWebhookSubscriptionResponse.model_validate(response.model_dump())
    assert again.secret_plaintext == "abc123"


def test_webhook_subscription_list_response_round_trip() -> None:
    """List wrapper does NOT carry secrets even when sourced from creates."""
    row = WebhookSubscriptionResponse(
        subscription_id=uuid4(),
        conversation_id=uuid4(),
        user_id=uuid4(),
        agent_id=uuid4(),
        name="x",
        execution_mode="inline",
        status="active",
        task_prompt_template="x",
        verification_scheme="generic_hmac_sha256",
        default_skill_id=None,
        allowed_source_pattern=None,
        rate_limit_per_minute=None,
        last_fired_at=None,
        date_created=datetime.now(UTC),
        date_updated=datetime.now(UTC),
    )
    wrapper = WebhookSubscriptionListResponse(subscriptions=[row], total_count=1)
    again = WebhookSubscriptionListResponse.model_validate(wrapper.model_dump())
    assert again.total_count == 1
    for sub in again.subscriptions:
        assert "secret_plaintext" not in sub.model_dump()
