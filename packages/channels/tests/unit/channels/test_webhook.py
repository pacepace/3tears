"""Unit tests for the channels webhook receiver framework.

Covers:

- :func:`verify_generic_hmac_sha256` correctness (golden vector +
  missing-header + wrong-prefix paths, constant-time compare via
  :func:`hmac.compare_digest`).
- :meth:`WebhookReceiver.register_verifier` registry behaviour.
- HTTP status mapping for every adapter outcome (202 / 400 / 401 /
  403 / 404 / 429 / 500), exercised against a stubbed
  ``webhook_receive`` so the receiver's routing-and-response plumbing
  is verified independently of the wake-side adapter.
- 413 short-circuit for oversized bodies (no adapter invocation).
- 429 carries ``Retry-After: 60``.
- ``X-Forwarded-For`` first-hop source-IP resolution.
- Constructor wiring (signature header override + custom max payload
  bytes).

Wake-side ``webhook_receive`` integration coverage lives in
``packages/agent/wake/tests/integration/test_webhook_receive.py``; the
end-to-end shard-06 integration test (real testcontainer Postgres +
real :func:`webhook_receive` mounted on FastAPI) lives next to this
file in ``tests/integration/channels/test_webhook_e2e.py``.
"""

from __future__ import annotations

import hmac
from hashlib import sha256
from typing import Any
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from threetears.agent.wake.config import DEFAULT_WAKE_CONFIG
from threetears.agent.wake.webhook_adapter import WebhookReceiveResult
from threetears.channels.webhook import (
    DEFAULT_MAX_PAYLOAD_BYTES,
    DEFAULT_SIGNATURE_HEADER,
    WebhookReceiver,
    verify_generic_hmac_sha256,
)


# parity-with: threetears.agent.wake.entities.EncryptionService
class _IdentityEncryption:
    """Identity encryption stand-in -- the receiver never invokes it on the
    routing-and-response tests (those stub ``webhook_receive``), so a
    no-op shape is sufficient.
    """

    def encrypt(self, plaintext: bytes) -> bytes:
        return bytes(plaintext)

    def decrypt(self, ciphertext: bytes) -> str:
        return ciphertext.decode("utf-8")


# parity-with: threetears.agent.wake.types.HandlerCallback
class _NullHandler:
    """No-op :class:`HandlerCallback`; the routing tests stub
    ``webhook_receive`` so the handler is never invoked. Carries the
    Protocol surface to satisfy mypy without a real LLM in scope.
    """

    async def __call__(
        self,
        trigger: Any,
        prepared_context: Any,
        pool: Any,
    ) -> Any:
        del trigger, prepared_context, pool
        msg = "_NullHandler should never be invoked when webhook_receive is stubbed"
        raise AssertionError(msg)


def _build_receiver(
    *,
    signature_header: str | None = None,
    max_payload_bytes: int | None = None,
) -> WebhookReceiver:
    """Construct a receiver with stub dependencies suitable for routing tests."""
    kwargs: dict[str, Any] = {
        "pool": object(),
        "encryption_service": _IdentityEncryption(),
        "handler": _NullHandler(),
        "wake_config": DEFAULT_WAKE_CONFIG,
        "delivery_adapters": None,
    }
    if signature_header is not None:
        kwargs["signature_header"] = signature_header
    if max_payload_bytes is not None:
        kwargs["max_payload_bytes"] = max_payload_bytes
    return WebhookReceiver(**kwargs)


def _make_app(receiver: WebhookReceiver, *, mount_path: str = "/webhooks") -> FastAPI:
    """Mount a receiver on a fresh FastAPI app and return it."""
    app = FastAPI()
    receiver.register(app, mount_path=mount_path)
    return app


# ============================================================
# verify_generic_hmac_sha256 -- the platform-default verifier
# ============================================================


class TestVerifyGenericHmacSha256:
    """Default ``generic_hmac_sha256`` verifier semantics."""

    def test_valid_signature_returns_true(self) -> None:
        secret = b"s3cret"
        payload = b'{"hello": "world"}'
        sig = "sha256=" + hmac.new(secret, payload, sha256).hexdigest()
        headers = {DEFAULT_SIGNATURE_HEADER.lower(): sig}
        assert verify_generic_hmac_sha256(secret, payload, headers) is True

    def test_invalid_signature_returns_false(self) -> None:
        headers = {DEFAULT_SIGNATURE_HEADER.lower(): "sha256=" + "0" * 64}
        assert verify_generic_hmac_sha256(b"s3cret", b"payload", headers) is False

    def test_missing_header_returns_false(self) -> None:
        assert verify_generic_hmac_sha256(b"s3cret", b"payload", {}) is False

    def test_wrong_prefix_returns_false(self) -> None:
        headers = {DEFAULT_SIGNATURE_HEADER.lower(): "md5=" + "0" * 32}
        assert verify_generic_hmac_sha256(b"s3cret", b"payload", headers) is False

    def test_empty_signature_returns_false(self) -> None:
        headers = {DEFAULT_SIGNATURE_HEADER.lower(): ""}
        assert verify_generic_hmac_sha256(b"s3cret", b"payload", headers) is False

    def test_uses_constant_time_compare(self) -> None:
        """``verify_generic_hmac_sha256`` must use :func:`hmac.compare_digest`.

        Source-level assertion: the function body must reference
        ``hmac.compare_digest`` (or ``compare_digest``) rather than
        the variable-time ``==`` operator. A timing-attack-resilient
        compare is mandatory for HMAC verification per the spec
        anti-pattern list.
        """
        import inspect

        source = inspect.getsource(verify_generic_hmac_sha256)
        assert "compare_digest" in source, "must use hmac.compare_digest"
        # Belt-and-braces: the source must NOT do raw ``==`` between
        # the expected and provided signatures.
        assert "expected == sig_header" not in source
        assert "sig_header == expected" not in source


# ============================================================
# WebhookReceiver.register_verifier -- pluggable scheme registry
# ============================================================


class TestRegisterVerifier:
    """Pluggable :data:`Verifier` registry behaviour."""

    def test_default_scheme_is_preregistered(self) -> None:
        receiver = _build_receiver()
        # The default ``generic_hmac_sha256`` scheme is wired at
        # construction time so subscriptions land with a working
        # verifier without consumer ceremony.
        assert receiver._verifiers["generic_hmac_sha256"] is verify_generic_hmac_sha256  # noqa: SLF001

    def test_register_custom_scheme(self) -> None:
        receiver = _build_receiver()

        def _github_stub(secret: bytes, payload: bytes, headers: dict[str, str]) -> bool:
            del secret, payload, headers
            return True

        receiver.register_verifier("github", _github_stub)
        assert receiver._verifiers["github"] is _github_stub  # noqa: SLF001

    def test_register_overrides_existing(self) -> None:
        receiver = _build_receiver()

        def _replacement(secret: bytes, payload: bytes, headers: dict[str, str]) -> bool:
            del secret, payload, headers
            return False

        receiver.register_verifier("generic_hmac_sha256", _replacement)
        assert receiver._verifiers["generic_hmac_sha256"] is _replacement  # noqa: SLF001


# ============================================================
# HTTP status mapping -- routing-and-response plumbing
# ============================================================


@pytest.fixture
def fixed_subscription_id() -> UUID:
    return uuid4()


def _patch_receive(
    *,
    status_code: int,
    fire_id: UUID | None = None,
    message: str = "stubbed",
) -> Any:
    """Patch ``webhook_receive`` to return a canned result."""

    async def _stub(**_kwargs: Any) -> WebhookReceiveResult:
        return WebhookReceiveResult(
            status_code=status_code,
            fire_id=fire_id,
            message=message,
        )

    return patch("threetears.agent.wake.webhook_adapter.webhook_receive", _stub)


class TestHttpStatusMapping:
    """Adapter outcome -> HTTP status code mapping per WEBHOOK-03."""

    def test_202_accepted_includes_fire_id(self, fixed_subscription_id: UUID) -> None:
        receiver = _build_receiver()
        app = _make_app(receiver)
        fire_id = uuid4()
        with _patch_receive(status_code=202, fire_id=fire_id, message="dispatched"):
            client = TestClient(app)
            r = client.post(
                f"/webhooks/{fixed_subscription_id}",
                content=b'{"x":1}',
                headers={DEFAULT_SIGNATURE_HEADER: "sha256=stub"},
            )
        assert r.status_code == 202
        body = r.json()
        assert body == {"fire_id": str(fire_id), "message": "dispatched"}

    def test_400_malformed_payload(self, fixed_subscription_id: UUID) -> None:
        receiver = _build_receiver()
        app = _make_app(receiver)
        with _patch_receive(status_code=400, message="template render error"):
            client = TestClient(app)
            r = client.post(
                f"/webhooks/{fixed_subscription_id}",
                content=b"bad",
                headers={DEFAULT_SIGNATURE_HEADER: "sha256=stub"},
            )
        assert r.status_code == 400
        assert r.json() == {"fire_id": None, "message": "template render error"}

    def test_401_missing_signature(self, fixed_subscription_id: UUID) -> None:
        receiver = _build_receiver()
        app = _make_app(receiver)
        with _patch_receive(status_code=401, message="missing signature header"):
            client = TestClient(app)
            r = client.post(
                f"/webhooks/{fixed_subscription_id}",
                content=b"{}",
            )
        assert r.status_code == 401
        assert r.json() == {"fire_id": None, "message": "missing signature header"}

    def test_403_source_ip_rejected(self, fixed_subscription_id: UUID) -> None:
        receiver = _build_receiver()
        app = _make_app(receiver)
        with _patch_receive(status_code=403, message="source IP not allowed"):
            client = TestClient(app)
            r = client.post(
                f"/webhooks/{fixed_subscription_id}",
                content=b"{}",
                headers={DEFAULT_SIGNATURE_HEADER: "sha256=stub"},
            )
        assert r.status_code == 403
        assert r.json() == {"fire_id": None, "message": "source IP not allowed"}

    def test_404_unknown_subscription(self, fixed_subscription_id: UUID) -> None:
        receiver = _build_receiver()
        app = _make_app(receiver)
        with _patch_receive(status_code=404, message="subscription not found or paused"):
            client = TestClient(app)
            r = client.post(
                f"/webhooks/{fixed_subscription_id}",
                content=b"{}",
                headers={DEFAULT_SIGNATURE_HEADER: "sha256=stub"},
            )
        assert r.status_code == 404
        assert r.json() == {
            "fire_id": None,
            "message": "subscription not found or paused",
        }

    def test_429_rate_limited_includes_retry_after(self, fixed_subscription_id: UUID) -> None:
        receiver = _build_receiver()
        app = _make_app(receiver)
        with _patch_receive(status_code=429, message="rate limit exceeded"):
            client = TestClient(app)
            r = client.post(
                f"/webhooks/{fixed_subscription_id}",
                content=b"{}",
                headers={DEFAULT_SIGNATURE_HEADER: "sha256=stub"},
            )
        assert r.status_code == 429
        # The static 60s Retry-After matches the spec simplification:
        # the wake-side per-subscription window is 60s, so 60s is the
        # worst-case wait for the oldest fire to age out.
        assert r.headers.get("Retry-After") == "60"
        body = r.json()
        assert body["fire_id"] is None
        assert "rate limit" in body["message"]

    def test_500_dispatch_failed(self, fixed_subscription_id: UUID) -> None:
        receiver = _build_receiver()
        app = _make_app(receiver)
        fire_id = uuid4()
        with _patch_receive(status_code=500, fire_id=fire_id, message="dispatch failed: oops"):
            client = TestClient(app)
            r = client.post(
                f"/webhooks/{fixed_subscription_id}",
                content=b"{}",
                headers={DEFAULT_SIGNATURE_HEADER: "sha256=stub"},
            )
        assert r.status_code == 500
        assert r.json() == {"fire_id": str(fire_id), "message": "dispatch failed: oops"}


# ============================================================
# 413 short-circuit + Retry-After absence on non-429 responses
# ============================================================


class TestPayloadSizeCap:
    """Body-size cap short-circuits at 413 without invoking the adapter."""

    def test_oversized_body_returns_413_without_adapter_invocation(
        self,
        fixed_subscription_id: UUID,
    ) -> None:
        receiver = _build_receiver(max_payload_bytes=16)
        app = _make_app(receiver)

        called = {"flag": False}

        async def _stub(**_kwargs: Any) -> WebhookReceiveResult:
            called["flag"] = True
            return WebhookReceiveResult(status_code=202, fire_id=None, message="should not run")

        with patch("threetears.agent.wake.webhook_adapter.webhook_receive", _stub):
            client = TestClient(app)
            r = client.post(
                f"/webhooks/{fixed_subscription_id}",
                content=b"x" * 128,
                headers={DEFAULT_SIGNATURE_HEADER: "sha256=stub"},
            )

        assert r.status_code == 413
        assert r.json() == {"fire_id": None, "message": "payload too large"}
        assert called["flag"] is False

    def test_default_max_payload_bytes_is_1_mib(self) -> None:
        """``DEFAULT_MAX_PAYLOAD_BYTES`` must match the spec default of 1 MiB."""
        assert DEFAULT_MAX_PAYLOAD_BYTES == 1 << 20

    def test_body_at_cap_size_is_accepted(self, fixed_subscription_id: UUID) -> None:
        """Body exactly at the cap should pass through (the check is ``>`` not ``>=``)."""
        receiver = _build_receiver(max_payload_bytes=16)
        app = _make_app(receiver)
        with _patch_receive(status_code=202, fire_id=uuid4(), message="ok"):
            client = TestClient(app)
            r = client.post(
                f"/webhooks/{fixed_subscription_id}",
                content=b"x" * 16,
                headers={DEFAULT_SIGNATURE_HEADER: "sha256=stub"},
            )
        assert r.status_code == 202


# ============================================================
# Signature header configurability
# ============================================================


class TestSignatureHeaderOverride:
    """Backwards-compat path: consumers can override the signature header name."""

    def test_custom_header_is_forwarded_to_adapter(
        self,
        fixed_subscription_id: UUID,
    ) -> None:
        receiver = _build_receiver(signature_header="X-MetaLLM-Signature")
        app = _make_app(receiver)

        captured: dict[str, Any] = {}

        async def _stub(**kwargs: Any) -> WebhookReceiveResult:
            captured.update(kwargs)
            return WebhookReceiveResult(status_code=202, fire_id=uuid4(), message="ok")

        with patch("threetears.agent.wake.webhook_adapter.webhook_receive", _stub):
            client = TestClient(app)
            r = client.post(
                f"/webhooks/{fixed_subscription_id}",
                content=b"{}",
                headers={"X-MetaLLM-Signature": "sha256=overridden"},
            )

        assert r.status_code == 202
        assert captured["signature_header"] == "sha256=overridden"

    def test_default_header_when_consumer_uses_default(
        self,
        fixed_subscription_id: UUID,
    ) -> None:
        receiver = _build_receiver()
        app = _make_app(receiver)

        captured: dict[str, Any] = {}

        async def _stub(**kwargs: Any) -> WebhookReceiveResult:
            captured.update(kwargs)
            return WebhookReceiveResult(status_code=202, fire_id=uuid4(), message="ok")

        with patch("threetears.agent.wake.webhook_adapter.webhook_receive", _stub):
            client = TestClient(app)
            r = client.post(
                f"/webhooks/{fixed_subscription_id}",
                content=b"{}",
                headers={DEFAULT_SIGNATURE_HEADER: "sha256=via-default"},
            )

        assert r.status_code == 202
        assert captured["signature_header"] == "sha256=via-default"


# ============================================================
# X-Forwarded-For source IP resolution
# ============================================================


class TestSourceIpResolution:
    """``X-Forwarded-For`` first-hop convention per WEBHOOK-07."""

    def test_x_forwarded_for_first_hop_is_used(
        self,
        fixed_subscription_id: UUID,
    ) -> None:
        receiver = _build_receiver()
        app = _make_app(receiver)

        captured: dict[str, Any] = {}

        async def _stub(**kwargs: Any) -> WebhookReceiveResult:
            captured.update(kwargs)
            return WebhookReceiveResult(status_code=202, fire_id=uuid4(), message="ok")

        with patch("threetears.agent.wake.webhook_adapter.webhook_receive", _stub):
            client = TestClient(app)
            r = client.post(
                f"/webhooks/{fixed_subscription_id}",
                content=b"{}",
                headers={
                    DEFAULT_SIGNATURE_HEADER: "sha256=stub",
                    "X-Forwarded-For": "203.0.113.7, 10.0.0.1, 10.0.0.2",
                },
            )

        assert r.status_code == 202
        assert captured["source_ip"] == "203.0.113.7"

    def test_socket_address_fallback(
        self,
        fixed_subscription_id: UUID,
    ) -> None:
        """Without ``X-Forwarded-For``, the socket address is used.

        TestClient sets ``request.client.host`` to ``'testclient'``.
        """
        receiver = _build_receiver()
        app = _make_app(receiver)

        captured: dict[str, Any] = {}

        async def _stub(**kwargs: Any) -> WebhookReceiveResult:
            captured.update(kwargs)
            return WebhookReceiveResult(status_code=202, fire_id=uuid4(), message="ok")

        with patch("threetears.agent.wake.webhook_adapter.webhook_receive", _stub):
            client = TestClient(app)
            r = client.post(
                f"/webhooks/{fixed_subscription_id}",
                content=b"{}",
                headers={DEFAULT_SIGNATURE_HEADER: "sha256=stub"},
            )

        assert r.status_code == 202
        assert captured["source_ip"] == "testclient"


# ============================================================
# Route shape -- mount path + path param typing
# ============================================================


class TestRouteShape:
    """``register(app, mount_path=...)`` mounts the right URL pattern."""

    def test_default_mount_path(self, fixed_subscription_id: UUID) -> None:
        receiver = _build_receiver()
        app = _make_app(receiver)
        # Routes the FastAPI app added should include the
        # ``/webhooks/{subscription_id}`` template.
        paths = [getattr(route, "path", "") for route in app.routes]
        assert "/webhooks/{subscription_id}" in paths

    def test_custom_mount_path(self) -> None:
        receiver = _build_receiver()
        app = _make_app(receiver, mount_path="/api/v1/inbound")
        paths = [getattr(route, "path", "") for route in app.routes]
        assert "/api/v1/inbound/{subscription_id}" in paths

    def test_invalid_uuid_returns_422(self) -> None:
        """FastAPI rejects malformed UUIDs in the path with 422 before
        ever invoking the receiver. Verifies the ``UUID`` type annotation
        on ``_handle`` is actually being respected by the framework.
        """
        receiver = _build_receiver()
        app = _make_app(receiver)
        client = TestClient(app)
        r = client.post(
            "/webhooks/not-a-uuid",
            content=b"{}",
            headers={DEFAULT_SIGNATURE_HEADER: "sha256=stub"},
        )
        assert r.status_code == 422
