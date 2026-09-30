"""a webhook fire's ``last_fired_at`` stamp reaches every replica that cached the subscription.

The receiver stamps ``last_fired_at`` with a targeted UPDATE that evicts the row from every
cache tier and broadcasts the eviction. It used to build its own registry per request with no
NATS client, so the eviction was local to a registry that died with the request: every other
replica kept serving the pre-fire row from its cache, and the webhook update tool there saved
that row back -- ``last_fired_at`` and all -- over the fire's stamp.

The receiver now runs on the collections its host process built once, on the registry that
carries the NATS client and runs the invalidation listener. The test asserts on the PUBLISH, and
on a second replica reading L3 again: a re-read on the receiving replica alone would pass against
a fix that evicted nothing beyond its own process.
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
from threetears.core.collections.registry import CacheInvalidationMessage, CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.testing.kv import FakeNatsClient

from threetears.agent.wake import webhook_adapter
from threetears.agent.wake.collections import WakeFireCollection, WebhookSubscriptionCollection
from threetears.agent.wake.tables import (
    agent_wake_schedules_table,
    wake_fires_table,
    webhook_subscriptions_table,
)
from threetears.agent.wake.types import WakeDispatchResult

_SCOPE = "wake-webhook-principal"
_READ_AT = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)
_FIRED_AT = datetime(2026, 9, 30, 12, 5, tzinfo=UTC)


def _metadata() -> MetaData:
    metadata = MetaData()
    agent_skills_table(metadata)
    agent_wake_schedules_table(metadata)
    wake_fires_table(metadata)
    webhook_subscriptions_table(metadata)
    return metadata


# parity-with: asyncpg.Pool (fetchrow / fetchval / execute -- the seam the receive path drives)
class _Store:
    """an L3 holding one subscription; the fire's ``last_fired_at`` UPDATE applies to it."""

    def __init__(self, row: dict[str, Any]) -> None:
        self.row = row
        self.reads = 0

    async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any] | None:
        assert "FROM webhook_subscriptions" in sql, sql
        self.reads += 1
        return dict(self.row) if self.row["subscription_id"] in args else None

    async def fetchval(self, sql: str, *args: Any) -> Any:
        del args
        assert "COUNT" in sql.upper(), sql
        return 0

    async def execute(self, sql: str, *args: Any) -> str:
        if sql.startswith("UPDATE webhook_subscriptions SET last_fired_at"):
            self.row["last_fired_at"] = args[2]
            self.row["date_updated"] = args[2]
        return "UPDATE 1"


async def _replica(nats: FakeNatsClient, store: _Store) -> CollectionRegistry:
    l1 = SQLiteBackend(db_name=f"wake_webhook_{uuid.uuid4().hex[:8]}")
    l1.initialize(_metadata())
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, l2_client=nats, l3_pool=store, kv_key_scope=_SCOPE)  # type: ignore[arg-type]
    await registry.start_invalidation_listener(nats)  # type: ignore[arg-type]
    return registry


def _config() -> DefaultCoreConfig:
    return DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")


def _subscription_row(conv: UUID, sub: UUID) -> dict[str, Any]:
    return {
        "conversation_id": conv,
        "subscription_id": sub,
        "user_id": uuid.uuid4(),
        "agent_id": uuid.uuid4(),
        "default_skill_id": None,
        "name": "deploys",
        "secret_ciphertext": b"secret",
        "allowed_source_pattern": None,
        "execution_mode": "inline",
        "task_prompt_template": "deploy: {{event.ref}}",
        "verification_scheme": "generic_hmac_sha256",
        "status": "active",
        "rate_limit_per_minute": None,
        "last_fired_at": None,
        "date_created": _READ_AT,
        "date_updated": _READ_AT,
    }


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


@pytest.mark.asyncio
async def test_a_fire_evicts_the_subscription_on_every_replica(monkeypatch: pytest.MonkeyPatch) -> None:
    nats = FakeNatsClient()
    conv, sub = uuid.uuid4(), uuid.uuid4()
    store = _Store(_subscription_row(conv, sub))

    reader = WebhookSubscriptionCollection(registry=await _replica(nats, store), config=_config())
    cached = await reader.get((conv, sub))
    assert cached is not None and cached.last_fired_at is None

    receiving = await _replica(nats, store)
    subscriptions = WebhookSubscriptionCollection(registry=receiving, config=_config())
    fires = WakeFireCollection(registry=receiving, config=_config())

    async def _dispatched(*args: Any, **kwargs: Any) -> WakeDispatchResult:
        del args, kwargs
        return WakeDispatchResult(status="fired", output_text="ok", latency_ms=5)

    monkeypatch.setattr(webhook_adapter, "dispatch_wake", _dispatched)

    result = await webhook_adapter.webhook_receive(
        subscription_id=sub,
        payload_bytes=b'{"ref": "main"}',
        signature_header=None,
        source_ip=None,
        pool=store,
        subscriptions=subscriptions,
        fires=fires,
        encryption_service=_Encryption(),
        handler=_Handler(),
        now=_FIRED_AT,
        pre_verified=True,
    )

    assert result.status_code == 202, result.message
    assert any(
        isinstance(message, CacheInvalidationMessage)
        and message.table == "webhook_subscriptions"
        and message.ids == [str(conv), str(sub)]
        for message in nats.published
    ), "the fire's last_fired_at stamp broadcast no invalidation"
    reads_before = store.reads
    fresh = await reader.get((conv, sub))
    assert store.reads == reads_before + 1, "the other replica answered from a cache L3 no longer agrees with"
    assert fresh is not None and fresh.last_fired_at == _FIRED_AT


@pytest.mark.asyncio
async def test_a_looked_up_subscription_outlives_an_eviction_of_its_row() -> None:
    """the receiver reads the subscription it looked up after its own fire evicted the row.

    A scan-built entity that proxied its fields through L1 read every one of them as absent once
    the key was evicted -- by the receiver's own ``record_fire``, or by any peer's broadcast while
    the handler ran.
    """
    conv, sub = uuid.uuid4(), uuid.uuid4()
    store = _Store(_subscription_row(conv, sub))
    subscriptions = WebhookSubscriptionCollection(
        registry=await _replica(FakeNatsClient(), store),
        config=_config(),
    )

    found = await subscriptions.find_by_id(sub)
    assert found is not None
    await subscriptions.invalidate_cache((conv, sub))

    assert found.conversation_id == conv
    assert found.name == "deploys"


@pytest.mark.asyncio
async def test_a_lookup_caches_nothing() -> None:
    """a scan reads L3 outside the per-key fence, so the row it read must not reach L1."""
    conv, sub = uuid.uuid4(), uuid.uuid4()
    store = _Store(_subscription_row(conv, sub))
    subscriptions = WebhookSubscriptionCollection(
        registry=await _replica(FakeNatsClient(), store),
        config=_config(),
    )

    assert await subscriptions.find_by_id(sub) is not None

    assert subscriptions.get_row_sync((conv, sub)) is None
