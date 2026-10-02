"""Unit tests for :mod:`threetears.agent.wake.webhook_adapter`'s payload handling.

The end-to-end ``webhook_receive`` flow with a real pool is covered in the
integration suite. Here the receiver runs against an in-memory L3 holding one
subscription whose prompt template renders the decoded payload as JSON, and
``dispatch_wake`` is replaced by one that records the trigger it is handed: the
trigger's task prompt is what the payload decoded to. Plus the result envelope
contract.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import pytest
from sqlalchemy import MetaData
from threetears.agent.skills.tables import agent_skills_table
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.testing.kv import FakeNatsClient

from threetears.agent.wake import webhook_adapter
from threetears.agent.wake.collections import WakeFireCollection, WebhookSubscriptionCollection
from threetears.agent.wake.tables import (
    agent_wake_schedules_table,
    wake_fires_table,
    webhook_subscriptions_table,
)
from threetears.agent.wake.types import WakeDispatchResult, WakeTrigger
from threetears.agent.wake.webhook_adapter import WebhookReceiveResult

_SCOPE = "wake-webhook-decode-principal"
_NOW = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)


def _metadata() -> MetaData:
    metadata = MetaData()
    agent_skills_table(metadata)
    agent_wake_schedules_table(metadata)
    wake_fires_table(metadata)
    webhook_subscriptions_table(metadata)
    return metadata


# parity-with: asyncpg.Pool (fetchrow / fetchval / execute -- the seam the receive path drives)
class _Store:
    """an L3 holding one subscription whose template renders the decoded payload as JSON."""

    def __init__(self, conversation_id: UUID, subscription_id: UUID) -> None:
        self.row: dict[str, Any] = {
            "conversation_id": conversation_id,
            "subscription_id": subscription_id,
            "user_id": uuid.uuid4(),
            "agent_id": uuid.uuid4(),
            "default_skill_id": None,
            "name": "decode",
            "secret_ciphertext": b"secret",
            "allowed_source_pattern": None,
            "execution_mode": "inline",
            "task_prompt_template": "{{ event | tojson }}",
            "verification_scheme": "generic_hmac_sha256",
            "status": "active",
            "rate_limit_per_minute": None,
            "last_fired_at": None,
            "date_created": _NOW,
            "date_updated": _NOW,
        }

    async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any] | None:
        assert "FROM webhook_subscriptions" in sql, sql
        return dict(self.row) if self.row["subscription_id"] in args else None

    async def fetchval(self, sql: str, *args: Any) -> Any:
        del args
        assert "COUNT" in sql.upper(), sql
        return 0

    async def execute(self, sql: str, *args: Any) -> str:
        del sql, args
        return "UPDATE 1"


# parity-with: threetears.agent.wake.entities.EncryptionService
class _Encryption:
    def encrypt(self, plaintext: bytes) -> bytes:
        return bytes(plaintext)

    def decrypt(self, ciphertext: bytes) -> str:
        return ciphertext.decode("utf-8")


# parity-with: threetears.agent.wake.types.HandlerCallback
class _Handler:
    async def __call__(self, trigger: Any, prepared_context: Any, pool: Any) -> Any:
        del trigger, prepared_context, pool
        raise AssertionError("dispatch_wake is replaced; the handler is never reached")


async def _receive(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> tuple[WebhookReceiveResult, list[WakeTrigger]]:
    """deliver ``payload`` to a pre-verified subscription and record what is dispatched.

    :param monkeypatch: pytest's monkeypatch fixture
    :ptype monkeypatch: pytest.MonkeyPatch
    :param payload: the raw HTTP body
    :ptype payload: bytes
    :return: the receiver's result and every trigger handed to the dispatcher
    :rtype: tuple[WebhookReceiveResult, list[WakeTrigger]]
    """
    conv, sub = uuid.uuid4(), uuid.uuid4()
    store = _Store(conv, sub)
    l1 = SQLiteBackend(db_name=f"wake_webhook_decode_{uuid.uuid4().hex[:8]}")
    l1.initialize(_metadata())
    nats = FakeNatsClient()
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, l2_client=nats, l3_pool=store, kv_key_scope=_SCOPE)  # type: ignore[arg-type]
    await registry.start_invalidation_listener(nats)  # type: ignore[arg-type]
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    dispatched: list[WakeTrigger] = []

    async def _dispatched(trigger: WakeTrigger, *args: Any, **kwargs: Any) -> WakeDispatchResult:
        del args, kwargs
        dispatched.append(trigger)
        return WakeDispatchResult(status="fired", output_text="ok", latency_ms=5)

    monkeypatch.setattr(webhook_adapter, "dispatch_wake", _dispatched)
    result = await webhook_adapter.webhook_receive(
        subscription_id=sub,
        payload_bytes=payload,
        signature_header=None,
        source_ip=None,
        pool=store,
        subscriptions=WebhookSubscriptionCollection(registry=registry, config=config),
        fires=WakeFireCollection(registry=registry, config=config),
        encryption_service=_Encryption(),
        handler=_Handler(),
        now=_NOW,
        pre_verified=True,
    )
    return result, dispatched


async def _rendered(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> str | None:
    """the task prompt the payload rendered to, as JSON of the decoded ``event``."""
    result, dispatched = await _receive(monkeypatch, payload)
    assert result.status_code == 202, result.message
    [trigger] = dispatched
    return trigger.task_prompt


async def test_decode_payload_json_object(monkeypatch: pytest.MonkeyPatch) -> None:
    assert await _rendered(monkeypatch, b'{"type": "push"}') == '{"type": "push"}'


async def test_decode_payload_json_array(monkeypatch: pytest.MonkeyPatch) -> None:
    assert await _rendered(monkeypatch, b"[1, 2, 3]") == "[1, 2, 3]"


async def test_decode_payload_plain_text_falls_back_to_string(monkeypatch: pytest.MonkeyPatch) -> None:
    assert await _rendered(monkeypatch, b"hello world") == '"hello world"'


async def test_decode_payload_empty_returns_empty_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    assert await _rendered(monkeypatch, b"") == "{}"


async def test_decode_payload_invalid_utf8_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    result, dispatched = await _receive(monkeypatch, b"\xff\xfe\xfd")
    assert result.status_code == 400
    assert "payload decode error" in result.message
    assert "UTF-8" in result.message
    assert dispatched == []


def test_webhook_receive_result_is_frozen() -> None:
    res = WebhookReceiveResult(status_code=202, fire_id=None, message="ok")
    with pytest.raises(Exception):  # noqa: PT011 - frozen dataclass raises FrozenInstanceError
        res.status_code = 500  # type: ignore[misc]
