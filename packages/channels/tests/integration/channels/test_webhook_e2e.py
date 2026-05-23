"""End-to-end integration test for the channels webhook receiver.

Mounts the :class:`WebhookReceiver` on a real FastAPI app, seeds a
real subscription row via the production agent-tool surface, then POSTs
a valid HMAC-signed payload and asserts:

- 202 + ``fire_id`` returned in the JSON body
- ``wake_fires`` row persisted with ``status='fired'``
- The handler callback was invoked with the rendered task prompt

Also covers invalid-signature (401) and oversized-body (413) paths
end-to-end against the real adapter (the unit tests stub
``webhook_receive`` for the routing-only coverage; this file is the
real integration test required by Requirement WEBHOOK-10).

The test uses ``httpx.AsyncClient`` + ``ASGITransport`` so the FastAPI
app is exercised through the same path a real HTTP request would
take, including starlette's request body read + header dict
construction + response serialisation.
"""

from __future__ import annotations

import hmac
from hashlib import sha256
from typing import Any
from uuid import UUID

import asyncpg
import httpx
import pytest
from fastapi import FastAPI
from uuid_utils import uuid7

from threetears.agent.skills.migrations import register as register_skills
from threetears.agent.wake.collections import WebhookSubscriptionCollection
from threetears.agent.wake.config import DEFAULT_WAKE_CONFIG
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
from threetears.channels.webhook import DEFAULT_SIGNATURE_HEADER, WebhookReceiver
from threetears.conversations.migrations import register as register_conversations
from threetears.core.collections.asyncpg_init import init_connection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.data.migrations import MigrationRunner


pytestmark = pytest.mark.integration


class _AsyncpgStore:
    """``DataStore``-shape wrapper over an asyncpg connection.

    Inline shim for the migration runner (the runner's protocol is
    just ``execute`` + ``query``). Mirrors the helper in the
    ``conftest.py`` next to this file -- inlined here so the test file
    does not need a relative import (relative imports require a
    package ``__init__.py``, which would collide with other packages'
    ``tests/integration/conftest.py`` modules under pytest's importlib
    mode).
    """

    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn

    async def execute(self, sql: str, *params: Any) -> str:
        result: str = await self._conn.execute(sql, *params)
        return result

    async def query(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        rows = await self._conn.fetch(sql, *params)
        return [dict(r) for r in rows]


def _new_uuid() -> UUID:
    return UUID(str(uuid7()))


# parity-with: threetears.agent.wake.entities.EncryptionService
class _IdentityEncryption:
    """Identity encryption: the 'ciphertext' is just the plaintext bytes.

    Mirrors the helper in the wake package's webhook integration
    test; copied rather than cross-imported to keep package test
    trees independent.
    """

    def encrypt(self, plaintext: bytes) -> bytes:
        return bytes(plaintext)

    def decrypt(self, ciphertext: bytes) -> str:
        return ciphertext.decode("utf-8")


# parity-with: threetears.agent.wake.tools.schedule_tools.WakeRegistryClient
class _PermissiveRegistry:
    """Allow every skill, no name lookups."""

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
    """Apply all three migration packs to a fresh schema; return a pool."""
    setup_conn = await asyncpg.connect(url)
    try:
        await setup_conn.execute(f'SET search_path TO "{schema}", public')
        runner = MigrationRunner()
        register_conversations(runner)
        register_skills(runner)
        register_wake(runner)
        store = _AsyncpgStore(setup_conn)
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
        },
    )
    assert "[webhook:" in result, result
    sub_id_str = result.split("[webhook:")[1].split("]")[0]
    secret_line = [line for line in result.splitlines() if "secret (copy now" in line][0]
    secret = secret_line.split(":")[-1].strip()
    return UUID(sub_id_str), secret


def _hmac_header(secret: str, payload: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode("utf-8"), payload, sha256).hexdigest()


def _build_app(pool: asyncpg.Pool, handler: HandlerCallback) -> FastAPI:
    """Construct a FastAPI app with the receiver mounted at /webhooks."""
    receiver = WebhookReceiver(
        pool=pool,
        encryption_service=_IdentityEncryption(),
        handler=handler,
        wake_config=DEFAULT_WAKE_CONFIG,
        delivery_adapters=None,
    )
    app = FastAPI()
    receiver.register(app)
    return app


@pytest.mark.asyncio
async def test_webhook_receiver_valid_post_returns_202_and_dispatches(
    pg_schema: tuple[str, str],
) -> None:
    url, schema = pg_schema
    pool = await _apply_schema(url, schema)
    try:
        conv_id = _new_uuid()
        sub_id, secret = await _seed_subscription(pool, conversation_id=conv_id)
        handler = _RecordingHandler()
        app = _build_app(pool, handler)

        payload = b'{"type": "push", "repo": "foo"}'
        sig = _hmac_header(secret, payload)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                f"/webhooks/{sub_id}",
                content=payload,
                headers={
                    DEFAULT_SIGNATURE_HEADER: sig,
                    "Content-Type": "application/json",
                },
            )

        assert response.status_code == 202, response.text
        body = response.json()
        assert body["fire_id"] is not None
        assert body["message"] == "dispatched"

        # The handler was invoked with a trigger carrying the rendered
        # task prompt.
        assert len(handler.invocations) == 1
        trigger, _ = handler.invocations[0]
        assert "push" in (trigger.task_prompt or "")

        # The wake_fires row was persisted + finalised.
        row = await pool.fetchrow(
            "SELECT status FROM wake_fires WHERE fire_id = $1",
            UUID(body["fire_id"]),
        )
        assert row is not None
        assert row["status"] == "fired"
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_webhook_receiver_invalid_signature_returns_401(
    pg_schema: tuple[str, str],
) -> None:
    url, schema = pg_schema
    pool = await _apply_schema(url, schema)
    try:
        conv_id = _new_uuid()
        sub_id, _secret = await _seed_subscription(pool, conversation_id=conv_id)
        handler = _RecordingHandler()
        app = _build_app(pool, handler)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                f"/webhooks/{sub_id}",
                content=b"{}",
                headers={DEFAULT_SIGNATURE_HEADER: "sha256=bogus"},
            )

        assert response.status_code == 401, response.text
        body = response.json()
        assert body["fire_id"] is None
        assert "invalid signature" in body["message"]
        # Handler must NOT have been invoked.
        assert handler.invocations == []
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_webhook_receiver_oversized_body_returns_413(
    pg_schema: tuple[str, str],
) -> None:
    url, schema = pg_schema
    pool = await _apply_schema(url, schema)
    try:
        conv_id = _new_uuid()
        sub_id, secret = await _seed_subscription(pool, conversation_id=conv_id)
        handler = _RecordingHandler()
        # Receiver capped at 32 bytes; we POST 256 bytes so the
        # size-cap short-circuit fires before any HMAC compute or
        # adapter invocation.
        receiver = WebhookReceiver(
            pool=pool,
            encryption_service=_IdentityEncryption(),
            handler=handler,
            wake_config=DEFAULT_WAKE_CONFIG,
            delivery_adapters=None,
            max_payload_bytes=32,
        )
        app = FastAPI()
        receiver.register(app)

        payload = b"x" * 256
        sig = _hmac_header(secret, payload)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                f"/webhooks/{sub_id}",
                content=payload,
                headers={DEFAULT_SIGNATURE_HEADER: sig},
            )

        assert response.status_code == 413, response.text
        body = response.json()
        assert body == {"fire_id": None, "message": "payload too large"}
        # No fire row should exist.
        count = await pool.fetchval("SELECT COUNT(*) FROM wake_fires")
        assert count == 0
        assert handler.invocations == []
    finally:
        await pool.close()
