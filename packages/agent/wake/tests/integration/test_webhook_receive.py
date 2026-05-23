"""Integration tests for :func:`webhook_receive`.

Exercises HMAC verification, source-IP allow-list, rate-limit
rejection, template render errors, and the dispatch -> finalize
write path. Uses a stub :class:`HandlerCallback` that returns a
canned ``HandlerCallbackResult`` so the dispatcher's plumbing is
verified end-to-end without an LLM.
"""

from __future__ import annotations

import hmac
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any
from uuid import UUID

import asyncpg
import pytest
from uuid_utils import uuid7

from threetears.agent.skills.migrations import register as register_skills
from threetears.agent.wake.collections import WebhookSubscriptionCollection
from threetears.agent.wake.migrations import register as register_wake
from threetears.agent.wake.tools import (
    load_webhook_subscription_create_tool,
)
from threetears.agent.wake.types import (
    HandlerCallback,
    HandlerCallbackResult,
    PreparedWakeContext,
    WakeTrigger,
)
from threetears.agent.wake.webhook_adapter import webhook_receive
from threetears.conversations.migrations import register as register_conversations
from threetears.core.collections.asyncpg_init import init_connection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner

from .conftest import AsyncpgStore


pytestmark = pytest.mark.integration


def _new_uuid() -> UUID:
    return UUID(str(uuid7()))


# parity-with: threetears.agent.wake.entities.EncryptionService
class _IdentityEncryption:
    """Identity encryption: the 'ciphertext' is just the plaintext bytes."""

    def encrypt(self, plaintext: bytes) -> bytes:
        return bytes(plaintext)

    def decrypt(self, ciphertext: bytes) -> str:
        return ciphertext.decode("utf-8")


# parity-with: threetears.agent.wake.tools.schedule_tools.WakeRegistryClient
class _PermissiveRegistry:
    """Allow every skill, no name lookups (subscriptions don't need one here)."""

    async def acl_permits_skill(
        self,
        *,
        user_id: UUID,
        agent_id: UUID,
        skill_id: UUID,
    ) -> bool:
        del user_id, agent_id, skill_id
        return True

    async def skill_name_for_id(
        self,
        *,
        user_id: UUID,
        agent_id: UUID,
        skill_id: UUID,
    ) -> str | None:
        del user_id, agent_id, skill_id
        return None


# parity-with: threetears.agent.wake.types.HandlerCallback
class _RecordingHandler(HandlerCallback):
    """Records the trigger it received + returns a canned 'fired' result."""

    def __init__(self) -> None:
        self.invocations: list[tuple[WakeTrigger, PreparedWakeContext]] = []

    async def __call__(
        self,
        trigger: WakeTrigger,
        prepared_context: PreparedWakeContext,
        pool: Any,
    ) -> HandlerCallbackResult:
        del pool
        self.invocations.append((trigger, prepared_context))
        return HandlerCallbackResult(
            status="fired",
            assistant_message_content="ok",
            target_conversation_id=trigger.conversation_id,
            assistant_message_id=None,
            latency_ms=12,
            error=None,
        )


async def _apply_schema(url: str, schema: str) -> asyncpg.Pool:
    setup_conn = await asyncpg.connect(url)
    try:
        await setup_conn.execute(f'SET search_path TO "{schema}", public')
        runner = MigrationRunner()
        register_conversations(runner)
        register_skills(runner)
        register_wake(runner)
        store = AsyncpgStore(setup_conn)
        await runner.apply_for_agent_schema(store)  # type: ignore[arg-type]
    finally:
        await setup_conn.close()
    pool = await asyncpg.create_pool(
        url,
        min_size=2,
        max_size=8,
        server_settings={"search_path": f"{schema}, public"},
        init=init_connection,
    )
    assert pool is not None
    return pool


async def _seed_subscription(
    pool: asyncpg.Pool,
    *,
    conversation_id: UUID,
    template: str = "event: {{event.type}}",
    rate_limit_per_minute: int | None = None,
    allowed_source_pattern: str | None = None,
) -> tuple[UUID, str]:
    """Use the production create tool so the row + secret + ciphertext are real."""
    registry = CollectionRegistry()
    registry.configure(l3_pool=pool)
    cfg = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    subs = WebhookSubscriptionCollection(registry=registry, config=cfg)
    user_id = _new_uuid()
    agent_id = _new_uuid()
    enc = _IdentityEncryption()

    create_tool = load_webhook_subscription_create_tool(
        conversation_id=conversation_id,
        user_id=user_id,
        agent_id=agent_id,
        subscriptions_collection=subs,
        encryption_service=enc,
        registry=_PermissiveRegistry(),
    )[0]
    result = await create_tool.ainvoke(
        {
            "task_prompt_template": template,
            "name": "test-sub",
            "allowed_source_pattern": allowed_source_pattern,
            "rate_limit_per_minute": rate_limit_per_minute,
        },
    )
    assert "[webhook:" in result, result
    sub_id_str = result.split("[webhook:")[1].split("]")[0]
    secret_line = [line for line in result.splitlines() if "secret (copy now" in line][0]
    secret = secret_line.split(":")[-1].strip()
    return UUID(sub_id_str), secret


def _hmac_header(secret: str, payload: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode("utf-8"), payload, sha256).hexdigest()


@pytest.mark.asyncio
async def test_webhook_receive_valid_signature_dispatches(
    pg_schema: tuple[str, str],
) -> None:
    url, schema = pg_schema
    pool = await _apply_schema(url, schema)
    try:
        conv_id = _new_uuid()
        sub_id, secret = await _seed_subscription(pool, conversation_id=conv_id)
        handler = _RecordingHandler()
        payload = b'{"type": "push", "repo": "foo"}'
        result = await webhook_receive(
            subscription_id=sub_id,
            payload_bytes=payload,
            signature_header=_hmac_header(secret, payload),
            source_ip=None,
            pool=pool,
            encryption_service=_IdentityEncryption(),
            handler=handler,
        )
        assert result.status_code == 202, result
        assert result.fire_id is not None
        assert len(handler.invocations) == 1
        trigger, _ = handler.invocations[0]
        assert "push" in (trigger.task_prompt or "")

        # wake_fires row should have been inserted + finalized.
        row = await pool.fetchrow(
            "SELECT status FROM wake_fires WHERE fire_id = $1",
            result.fire_id,
        )
        assert row is not None
        assert row["status"] == "fired"
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_webhook_receive_invalid_signature_rejected(
    pg_schema: tuple[str, str],
) -> None:
    url, schema = pg_schema
    pool = await _apply_schema(url, schema)
    try:
        conv_id = _new_uuid()
        sub_id, _ = await _seed_subscription(pool, conversation_id=conv_id)
        result = await webhook_receive(
            subscription_id=sub_id,
            payload_bytes=b"{}",
            signature_header="sha256=bogus",
            source_ip=None,
            pool=pool,
            encryption_service=_IdentityEncryption(),
            handler=_RecordingHandler(),
        )
        assert result.status_code == 401
        assert "invalid signature" in result.message
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_webhook_receive_missing_signature_rejected(
    pg_schema: tuple[str, str],
) -> None:
    url, schema = pg_schema
    pool = await _apply_schema(url, schema)
    try:
        conv_id = _new_uuid()
        sub_id, _ = await _seed_subscription(pool, conversation_id=conv_id)
        result = await webhook_receive(
            subscription_id=sub_id,
            payload_bytes=b"{}",
            signature_header=None,
            source_ip=None,
            pool=pool,
            encryption_service=_IdentityEncryption(),
            handler=_RecordingHandler(),
        )
        assert result.status_code == 401
        assert "missing" in result.message
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_webhook_receive_unknown_subscription_404(
    pg_schema: tuple[str, str],
) -> None:
    url, schema = pg_schema
    pool = await _apply_schema(url, schema)
    try:
        result = await webhook_receive(
            subscription_id=_new_uuid(),
            payload_bytes=b"{}",
            signature_header="sha256=anything",
            source_ip=None,
            pool=pool,
            encryption_service=_IdentityEncryption(),
            handler=_RecordingHandler(),
        )
        assert result.status_code == 404
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_webhook_receive_source_ip_allow_list_403(
    pg_schema: tuple[str, str],
) -> None:
    url, schema = pg_schema
    pool = await _apply_schema(url, schema)
    try:
        conv_id = _new_uuid()
        sub_id, secret = await _seed_subscription(
            pool,
            conversation_id=conv_id,
            allowed_source_pattern=r"^10\.0\.",
        )
        payload = b'{"type":"x"}'
        result = await webhook_receive(
            subscription_id=sub_id,
            payload_bytes=payload,
            signature_header=_hmac_header(secret, payload),
            source_ip="192.168.1.1",
            pool=pool,
            encryption_service=_IdentityEncryption(),
            handler=_RecordingHandler(),
        )
        assert result.status_code == 403
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_webhook_receive_rate_limit_429(
    pg_schema: tuple[str, str],
) -> None:
    url, schema = pg_schema
    pool = await _apply_schema(url, schema)
    try:
        conv_id = _new_uuid()
        sub_id, secret = await _seed_subscription(
            pool,
            conversation_id=conv_id,
            rate_limit_per_minute=2,
        )
        payload = b'{"type":"x"}'
        header = _hmac_header(secret, payload)
        handler = _RecordingHandler()

        for _ in range(2):
            res = await webhook_receive(
                subscription_id=sub_id,
                payload_bytes=payload,
                signature_header=header,
                source_ip=None,
                pool=pool,
                encryption_service=_IdentityEncryption(),
                handler=handler,
            )
            assert res.status_code == 202, res

        rejected = await webhook_receive(
            subscription_id=sub_id,
            payload_bytes=payload,
            signature_header=header,
            source_ip=None,
            pool=pool,
            encryption_service=_IdentityEncryption(),
            handler=handler,
        )
        assert rejected.status_code == 429
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_webhook_receive_template_render_error_400(
    pg_schema: tuple[str, str],
) -> None:
    url, schema = pg_schema
    pool = await _apply_schema(url, schema)
    try:
        conv_id = _new_uuid()
        # Template asks for a field that won't exist; Jinja's default
        # behaviour with attribute access on a string is to raise.
        sub_id, secret = await _seed_subscription(
            pool,
            conversation_id=conv_id,
            template="{{ event.missing_field.subfield }}",
        )
        payload = b'"plain string payload"'
        header = _hmac_header(secret, payload)
        result = await webhook_receive(
            subscription_id=sub_id,
            payload_bytes=payload,
            signature_header=header,
            source_ip=None,
            pool=pool,
            encryption_service=_IdentityEncryption(),
            handler=_RecordingHandler(),
        )
        assert result.status_code == 400
        assert "render" in result.message
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_webhook_receive_paused_subscription_404(
    pg_schema: tuple[str, str],
) -> None:
    url, schema = pg_schema
    pool = await _apply_schema(url, schema)
    try:
        conv_id = _new_uuid()
        sub_id, secret = await _seed_subscription(pool, conversation_id=conv_id)
        # Manually pause via SQL.
        await pool.execute(
            "UPDATE webhook_subscriptions SET status = 'paused' WHERE subscription_id = $1",
            sub_id,
        )
        payload = b"{}"
        result = await webhook_receive(
            subscription_id=sub_id,
            payload_bytes=payload,
            signature_header=_hmac_header(secret, payload),
            source_ip=None,
            pool=pool,
            encryption_service=_IdentityEncryption(),
            handler=_RecordingHandler(),
        )
        assert result.status_code == 404
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_webhook_receive_pre_verified_skips_hmac_compute(
    pg_schema: tuple[str, str],
) -> None:
    """With ``pre_verified=True`` the adapter trusts the caller's verify
    decision and skips the inline HMAC compute (so a deliberately
    wrong / arbitrary signature value still dispatches successfully).

    Proves the new receiver-side wiring contract: the channels-side
    :class:`~threetears.channels.webhook.WebhookReceiver` does the
    verifier-registry dispatch then calls this function with
    ``pre_verified=True``, and the adapter does NOT re-verify.
    """
    url, schema = pg_schema
    pool = await _apply_schema(url, schema)
    try:
        conv_id = _new_uuid()
        sub_id, _real_secret = await _seed_subscription(pool, conversation_id=conv_id)
        handler = _RecordingHandler()
        payload = b'{"type":"pre-verified"}'
        # A signature value that would NOT pass the inline HMAC check
        # -- proves the inline compute is skipped when pre_verified=True.
        result = await webhook_receive(
            subscription_id=sub_id,
            payload_bytes=payload,
            signature_header="vendor=trusted-by-receiver",
            source_ip=None,
            pool=pool,
            encryption_service=_IdentityEncryption(),
            handler=handler,
            pre_verified=True,
        )
        assert result.status_code == 202, result
        assert result.fire_id is not None
        assert len(handler.invocations) == 1
        trigger, _ = handler.invocations[0]
        assert "pre-verified" in (trigger.task_prompt or "")
    finally:
        await pool.close()
