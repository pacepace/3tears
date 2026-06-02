"""Unit tests for the three agent-wake entity classes.

Covers the entity contract:

- ``primary_key_field`` returns the bare-id column name.
- Composite ``_id`` tuple is set when ``conversation_id`` + the bare
  id are both present in the constructor data.
- Field accessors round-trip values through ``BaseEntity.__setattr__``.
- The JSONB ``schedule_config`` column returns a fresh dict on each
  getter call so callers cannot mutate the cached row state via the
  accessor.
- ``secret_ciphertext`` accepts ``bytes`` / ``memoryview`` / other
  byte-like inputs uniformly.
- ``decrypt_secret`` round-trips a plaintext string via a fake
  encryption service.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest
from uuid_utils import uuid7

from threetears.agent.wake.entities import (
    WakeFireEntity,
    WakeScheduleEntity,
    WebhookSubscriptionEntity,
    _as_uuid,
)


class TestAsUuidDefensive:
    """``_as_uuid`` must fail clearly on a None / empty non-UUID value.

    ``_as_uuid`` is only called for NON-nullable UUID columns (nullable
    ones short-circuit on ``None`` before reaching it). If such a field
    ever reads ``None`` -- e.g. a cache-coherence miss -- the stdlib
    ``UUID(str(None))`` raises the misleading "badly formed hexadecimal
    UUID string", which masks the real problem (a missing field) as a
    UUID-format problem. The helper must instead raise a clear,
    diagnosable error.
    """

    def test_as_uuid_passthrough_uuid(self) -> None:
        u = UUID(str(uuid7()))
        assert _as_uuid(u) is u

    def test_as_uuid_coerces_string(self) -> None:
        u = UUID(str(uuid7()))
        assert _as_uuid(str(u)) == u

    def test_as_uuid_none_raises_clear_error(self) -> None:
        with pytest.raises((ValueError, TypeError)) as exc_info:
            _as_uuid(None)
        msg = str(exc_info.value).lower()
        # must NOT surface the misleading stdlib message
        assert "badly formed hexadecimal" not in msg
        # must name the real problem
        assert "none" in msg


def _new_uuid() -> UUID:
    """Return a fresh UUIDv7 cast to stdlib ``UUID``."""
    return UUID(str(uuid7()))


def _now() -> datetime:
    """Return a fresh aware-UTC ``datetime``."""
    return datetime.now(UTC)


class TestWakeScheduleEntity:
    """Contract tests for :class:`WakeScheduleEntity`."""

    def test_primary_key_field_is_schedule_id(self) -> None:
        """``primary_key_field`` returns the bare-id column name."""
        assert WakeScheduleEntity.primary_key_field == "schedule_id"

    def test_composite_id_set_when_both_columns_present(self) -> None:
        """``id`` returns the ``(conversation_id, schedule_id)`` composite tuple."""
        conversation_id = _new_uuid()
        schedule_id = _new_uuid()
        entity = WakeScheduleEntity(
            {
                "conversation_id": conversation_id,
                "schedule_id": schedule_id,
                "user_id": _new_uuid(),
                "agent_id": _new_uuid(),
                "schedule_type": "daily_at",
                "schedule_config": {"hour": 9, "minute": 0},
                "execution_mode": "inline",
                "status": "active",
                "missed_fire_policy": "coalesce",
                "date_created": _now(),
                "date_updated": _now(),
            },
        )
        assert entity.id == (conversation_id, schedule_id)

    def test_field_accessors_round_trip(self) -> None:
        """Property getters return the values stored at construction."""
        conversation_id = _new_uuid()
        schedule_id = _new_uuid()
        user_id = _new_uuid()
        agent_id = _new_uuid()
        skill_id = _new_uuid()
        context_from = _new_uuid()
        now = _now()
        entity = WakeScheduleEntity(
            {
                "conversation_id": conversation_id,
                "schedule_id": schedule_id,
                "user_id": user_id,
                "agent_id": agent_id,
                "skill_id": skill_id,
                "schedule_type": "every_n_hours",
                "schedule_config": {"n": 3},
                "task_prompt": "Check inbox",
                "execution_mode": "spawn",
                "status": "active",
                "next_fire_at": now,
                "last_fired_at": now,
                "name": "inbox-poller",
                "missed_fire_policy": "catch_up",
                "context_from_schedule_id": context_from,
                "include_conversation_history": False,
                "date_created": now,
                "date_updated": now,
            },
        )
        assert entity.conversation_id == conversation_id
        assert entity.schedule_id == schedule_id
        assert entity.user_id == user_id
        assert entity.agent_id == agent_id
        assert entity.skill_id == skill_id
        assert entity.schedule_type == "every_n_hours"
        assert entity.schedule_config == {"n": 3}
        assert entity.task_prompt == "Check inbox"
        assert entity.execution_mode == "spawn"
        assert entity.status == "active"
        assert entity.next_fire_at == now
        assert entity.last_fired_at == now
        assert entity.name == "inbox-poller"
        assert entity.missed_fire_policy == "catch_up"
        assert entity.context_from_schedule_id == context_from
        assert entity.include_conversation_history is False
        assert entity.date_created == now
        assert entity.date_updated == now

    def test_skill_id_is_optional(self) -> None:
        """A skill-less wake has ``skill_id=None``."""
        entity = WakeScheduleEntity(
            {
                "conversation_id": _new_uuid(),
                "schedule_id": _new_uuid(),
                "user_id": _new_uuid(),
                "agent_id": _new_uuid(),
                "skill_id": None,
                "schedule_type": "daily_at",
                "schedule_config": {"hour": 9, "minute": 0},
                "execution_mode": "inline",
                "status": "active",
                "missed_fire_policy": "coalesce",
                "date_created": _now(),
                "date_updated": _now(),
            },
        )
        assert entity.skill_id is None

    def test_schedule_config_getter_returns_fresh_dict(self) -> None:
        """Mutating the returned dict does not affect the entity."""
        entity = WakeScheduleEntity(
            {
                "conversation_id": _new_uuid(),
                "schedule_id": _new_uuid(),
                "user_id": _new_uuid(),
                "agent_id": _new_uuid(),
                "schedule_type": "every_n_hours",
                "schedule_config": {"n": 3},
                "execution_mode": "inline",
                "status": "active",
                "missed_fire_policy": "coalesce",
                "date_created": _now(),
                "date_updated": _now(),
            },
        )
        snap = entity.schedule_config
        snap["injected"] = True
        assert "injected" not in entity.schedule_config

    def test_schedule_config_defaults_to_empty_dict_on_null(self) -> None:
        """A null JSONB column surfaces as an empty dict, not None."""
        entity = WakeScheduleEntity(
            {
                "conversation_id": _new_uuid(),
                "schedule_id": _new_uuid(),
                "user_id": _new_uuid(),
                "agent_id": _new_uuid(),
                "schedule_type": "cron",
                "schedule_config": None,
                "execution_mode": "inline",
                "status": "active",
                "missed_fire_policy": "coalesce",
                "date_created": _now(),
                "date_updated": _now(),
            },
        )
        assert entity.schedule_config == {}


class TestWakeFireEntity:
    """Contract tests for :class:`WakeFireEntity`."""

    def test_primary_key_field_is_fire_id(self) -> None:
        """``primary_key_field`` returns ``fire_id``."""
        assert WakeFireEntity.primary_key_field == "fire_id"

    def test_composite_id_set_when_both_columns_present(self) -> None:
        """``id`` returns ``(conversation_id, fire_id)`` composite tuple."""
        conversation_id = _new_uuid()
        fire_id = _new_uuid()
        entity = WakeFireEntity(
            {
                "conversation_id": conversation_id,
                "fire_id": fire_id,
                "schedule_id": _new_uuid(),
                "actual_fired_at": _now(),
                "status": "fired",
                "display_suppressed": False,
            },
        )
        assert entity.id == (conversation_id, fire_id)

    def test_field_accessors_round_trip(self) -> None:
        """Property getters return the construction values."""
        conv = _new_uuid()
        fire_id = _new_uuid()
        schedule_id = _new_uuid()
        now = _now()
        entity = WakeFireEntity(
            {
                "conversation_id": conv,
                "fire_id": fire_id,
                "schedule_id": schedule_id,
                "webhook_subscription_id": None,
                "scheduled_fire_at": now,
                "actual_fired_at": now,
                "status": "fired",
                "display_suppressed": True,
                "output_text": "ack",
                "latency_ms": 123,
                "error": None,
                "date_created": now,
            },
        )
        assert entity.fire_id == fire_id
        assert entity.conversation_id == conv
        assert entity.schedule_id == schedule_id
        assert entity.webhook_subscription_id is None
        assert entity.scheduled_fire_at == now
        assert entity.actual_fired_at == now
        assert entity.status == "fired"
        assert entity.display_suppressed is True
        assert entity.output_text == "ack"
        assert entity.latency_ms == 123
        assert entity.error is None

    def test_webhook_source_fire(self) -> None:
        """A webhook-source fire has ``schedule_id=None``."""
        sub_id = _new_uuid()
        entity = WakeFireEntity(
            {
                "conversation_id": _new_uuid(),
                "fire_id": _new_uuid(),
                "schedule_id": None,
                "webhook_subscription_id": sub_id,
                "actual_fired_at": _now(),
                "status": "fired",
                "display_suppressed": False,
            },
        )
        assert entity.schedule_id is None
        assert entity.webhook_subscription_id == sub_id

    def test_yielded_status_round_trips(self) -> None:
        """The post-wake-yield ``'yielded'`` status surfaces unchanged."""
        entity = WakeFireEntity(
            {
                "conversation_id": _new_uuid(),
                "fire_id": _new_uuid(),
                "schedule_id": _new_uuid(),
                "actual_fired_at": _now(),
                "status": "yielded",
                "display_suppressed": False,
            },
        )
        assert entity.status == "yielded"


# parity-with: threetears.agent.wake.entities.EncryptionService
class _FakeEncryptionService:
    """Minimal fake EncryptionService for the decrypt_secret test.

    Satisfies the :class:`EncryptionService` Protocol:
    ``encrypt(bytes) -> bytes`` + ``decrypt(bytes) -> str``.
    ``@runtime_checkable`` on the production protocol lets
    ``isinstance(_FakeEncryptionService(...), EncryptionService)`` pass
    without inheritance.
    """

    def __init__(self, secret: str) -> None:
        self._secret = secret

    def encrypt(self, plaintext: bytes) -> bytes:
        """Return the plaintext unchanged (test stand-in, not real crypto)."""
        return bytes(plaintext)

    def decrypt(self, ciphertext: bytes) -> str:
        """Return the canned plaintext regardless of input."""
        del ciphertext
        return self._secret


class TestWebhookSubscriptionEntity:
    """Contract tests for :class:`WebhookSubscriptionEntity`."""

    def test_primary_key_field_is_subscription_id(self) -> None:
        """``primary_key_field`` returns ``subscription_id``."""
        assert WebhookSubscriptionEntity.primary_key_field == "subscription_id"

    def test_composite_id_set_when_both_columns_present(self) -> None:
        """``id`` returns ``(conversation_id, subscription_id)`` tuple."""
        conv = _new_uuid()
        sub = _new_uuid()
        entity = WebhookSubscriptionEntity(
            {
                "conversation_id": conv,
                "subscription_id": sub,
                "user_id": _new_uuid(),
                "agent_id": _new_uuid(),
                "secret_ciphertext": b"\x00\x01\x02",
                "execution_mode": "inline",
                "verification_scheme": "generic_hmac_sha256",
                "status": "active",
                "date_created": _now(),
                "date_updated": _now(),
            },
        )
        assert entity.id == (conv, sub)

    def test_field_accessors_round_trip(self) -> None:
        """Every getter returns the construction value."""
        conv = _new_uuid()
        sub = _new_uuid()
        user_id = _new_uuid()
        agent_id = _new_uuid()
        default_skill = _new_uuid()
        now = _now()
        entity = WebhookSubscriptionEntity(
            {
                "conversation_id": conv,
                "subscription_id": sub,
                "user_id": user_id,
                "agent_id": agent_id,
                "default_skill_id": default_skill,
                "name": "github-events",
                "secret_ciphertext": b"\xff\x00",
                "allowed_source_pattern": "^140\\.82\\.",
                "execution_mode": "spawn",
                "task_prompt_template": "Investigate {{event.action}}",
                "verification_scheme": "generic_hmac_sha256",
                "status": "paused",
                "rate_limit_per_minute": 30,
                "last_fired_at": now,
                "date_created": now,
                "date_updated": now,
            },
        )
        assert entity.subscription_id == sub
        assert entity.conversation_id == conv
        assert entity.user_id == user_id
        assert entity.agent_id == agent_id
        assert entity.default_skill_id == default_skill
        assert entity.name == "github-events"
        assert entity.secret_ciphertext == b"\xff\x00"
        assert entity.allowed_source_pattern == "^140\\.82\\."
        assert entity.execution_mode == "spawn"
        assert entity.task_prompt_template == "Investigate {{event.action}}"
        assert entity.verification_scheme == "generic_hmac_sha256"
        assert entity.status == "paused"
        assert entity.rate_limit_per_minute == 30
        assert entity.last_fired_at == now

    def test_decrypt_secret_round_trips_via_fake_service(self) -> None:
        """``decrypt_secret`` returns the encryption service's plaintext."""
        entity = WebhookSubscriptionEntity(
            {
                "conversation_id": _new_uuid(),
                "subscription_id": _new_uuid(),
                "user_id": _new_uuid(),
                "agent_id": _new_uuid(),
                "secret_ciphertext": b"\xde\xad\xbe\xef",
                "execution_mode": "inline",
                "verification_scheme": "generic_hmac_sha256",
                "status": "active",
                "date_created": _now(),
                "date_updated": _now(),
            },
        )
        svc = _FakeEncryptionService("super-secret-hmac-key")
        assert entity.decrypt_secret(svc) == "super-secret-hmac-key"

    def test_secret_ciphertext_accepts_memoryview(self) -> None:
        """memoryview input is coerced to bytes via the getter."""
        raw = bytes([1, 2, 3, 4])
        entity = WebhookSubscriptionEntity(
            {
                "conversation_id": _new_uuid(),
                "subscription_id": _new_uuid(),
                "user_id": _new_uuid(),
                "agent_id": _new_uuid(),
                "secret_ciphertext": memoryview(raw),
                "execution_mode": "inline",
                "verification_scheme": "generic_hmac_sha256",
                "status": "active",
                "date_created": _now(),
                "date_updated": _now(),
            },
        )
        assert entity.secret_ciphertext == raw
        assert isinstance(entity.secret_ciphertext, bytes)
