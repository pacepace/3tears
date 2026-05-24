"""Unit tests for the channels webhook receiver framework.

Covers:

- :func:`verify_generic_hmac_sha256` correctness (golden vector +
  missing-header + wrong-prefix paths, constant-time compare via
  :func:`hmac.compare_digest`).
- :meth:`WebhookReceiver.register_verifier` registry behaviour and
  end-to-end wire-up: registered verifier is invoked at handle time;
  unknown scheme returns 400; verifier-returns-False returns 403;
  verifier-returns-True forwards to the wake adapter with
  ``pre_verified=True``.
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
from threetears.agent.wake.entities import WebhookSubscriptionEntity
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


def _build_subscription(
    *,
    status: str = "active",
    verification_scheme: str = "generic_hmac_sha256",
    secret_plaintext: str = "test-secret",
) -> WebhookSubscriptionEntity:
    """Construct a real :class:`WebhookSubscriptionEntity` for stub lookups.

    Using the production entity avoids fake-parity violations and
    keeps the receiver test exercising the same ``decrypt_secret`` /
    ``verification_scheme`` surfaces production code reads. The
    :class:`_IdentityEncryption` stand-in returns the plaintext bytes
    decoded as UTF-8, so ``secret_ciphertext = secret_plaintext.encode()``
    round-trips through :meth:`decrypt_secret` cleanly.
    """
    return WebhookSubscriptionEntity(
        {
            "conversation_id": uuid4(),
            "subscription_id": uuid4(),
            "user_id": uuid4(),
            "agent_id": uuid4(),
            "default_skill_id": None,
            "name": "stub",
            "secret_ciphertext": secret_plaintext.encode("utf-8"),
            "allowed_source_pattern": None,
            "execution_mode": "inline",
            "task_prompt_template": None,
            "verification_scheme": verification_scheme,
            "status": status,
            "rate_limit_per_minute": None,
            "last_fired_at": None,
            "date_created": None,
            "date_updated": None,
        },
        is_new=False,
        collection=None,
    )


def _patch_subscription_lookup(entity: WebhookSubscriptionEntity | None) -> Any:
    """Patch :class:`WebhookSubscriptionCollection.find_by_id` on the receiver.

    Returns ``entity`` regardless of the subscription id passed. Pass
    ``None`` to simulate a missing subscription (the receiver should
    forward to the wake adapter for the 404 mapping).
    """

    async def _find_by_id(self: Any, subscription_id: Any) -> Any:
        del self, subscription_id
        return entity

    return patch(
        "threetears.agent.wake.collections.WebhookSubscriptionCollection.find_by_id",
        _find_by_id,
    )


# ============================================================
# verify_generic_hmac_sha256 -- the platform-default verifier
# ============================================================


class TestVerifyGenericHmacSha256:
    """Default ``generic_hmac_sha256`` verifier semantics."""

    def test_valid_signature_returns_true(self) -> None:
        secret = b"s3cret"
        payload = b'{"hello": "world"}'
        sig = "sha256=" + hmac.new(secret, payload, sha256).hexdigest()
        assert verify_generic_hmac_sha256(secret, payload, sig) is True

    def test_invalid_signature_returns_false(self) -> None:
        assert verify_generic_hmac_sha256(b"s3cret", b"payload", "sha256=" + "0" * 64) is False

    def test_empty_signature_value_returns_false(self) -> None:
        assert verify_generic_hmac_sha256(b"s3cret", b"payload", "") is False

    def test_wrong_prefix_returns_false(self) -> None:
        assert verify_generic_hmac_sha256(b"s3cret", b"payload", "md5=" + "0" * 32) is False

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
        assert "expected == signature_value" not in source
        assert "signature_value == expected" not in source


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

        def _github_stub(secret: bytes, payload: bytes, signature_value: str) -> bool:
            del secret, payload, signature_value
            return True

        receiver.register_verifier("github", _github_stub)
        assert receiver._verifiers["github"] is _github_stub  # noqa: SLF001

    def test_register_overrides_existing(self) -> None:
        receiver = _build_receiver()

        def _replacement(secret: bytes, payload: bytes, signature_value: str) -> bool:
            del secret, payload, signature_value
            return False

        receiver.register_verifier("generic_hmac_sha256", _replacement)
        assert receiver._verifiers["generic_hmac_sha256"] is _replacement  # noqa: SLF001


# ============================================================
# Registry wiring -- the receiver actually dispatches through the registry
# ============================================================


@pytest.fixture
def fixed_subscription_id() -> UUID:
    return uuid4()


class TestRegistryDispatch:
    """End-to-end wiring: registered verifiers actually run at handle time."""

    def test_custom_verifier_is_invoked_for_its_scheme(
        self,
        fixed_subscription_id: UUID,
    ) -> None:
        receiver = _build_receiver()
        captured: dict[str, Any] = {}

        def _github_verifier(secret: bytes, payload: bytes, signature_value: str) -> bool:
            captured["secret"] = secret
            captured["payload"] = payload
            captured["signature_value"] = signature_value
            return True

        receiver.register_verifier("github", _github_verifier)

        # Subscription declares the 'github' scheme; default verifier
        # must NOT run for this row.
        sub = _build_subscription(
            verification_scheme="github",
            secret_plaintext="gh-secret",
        )

        async def _adapter_stub(**kwargs: Any) -> WebhookReceiveResult:
            # The adapter must be invoked with pre_verified=True after
            # the registry-dispatched verifier returns True.
            captured["adapter_pre_verified"] = kwargs.get("pre_verified")
            return WebhookReceiveResult(status_code=202, fire_id=uuid4(), message="ok")

        app = _make_app(receiver)
        with (
            _patch_subscription_lookup(sub),
            patch(
                "threetears.agent.wake.webhook_adapter.webhook_receive",
                _adapter_stub,
            ),
        ):
            client = TestClient(app)
            r = client.post(
                f"/webhooks/{fixed_subscription_id}",
                content=b"github-payload",
                headers={DEFAULT_SIGNATURE_HEADER: "vendor=42"},
            )

        assert r.status_code == 202
        # The custom verifier was invoked with exactly the receiver-
        # extracted signature value (not the full headers dict).
        assert captured["secret"] == b"gh-secret"
        assert captured["payload"] == b"github-payload"
        assert captured["signature_value"] == "vendor=42"
        # The adapter received pre_verified=True.
        assert captured["adapter_pre_verified"] is True

    def test_unknown_scheme_returns_400(self, fixed_subscription_id: UUID) -> None:
        receiver = _build_receiver()
        # Subscription declares a scheme nobody registered.
        sub = _build_subscription(verification_scheme="exotic_vendor")

        async def _adapter_stub(**_kwargs: Any) -> WebhookReceiveResult:
            msg = "adapter must not run for unknown-scheme rows"
            raise AssertionError(msg)

        app = _make_app(receiver)
        with (
            _patch_subscription_lookup(sub),
            patch(
                "threetears.agent.wake.webhook_adapter.webhook_receive",
                _adapter_stub,
            ),
        ):
            client = TestClient(app)
            r = client.post(
                f"/webhooks/{fixed_subscription_id}",
                content=b"{}",
                headers={DEFAULT_SIGNATURE_HEADER: "sha256=whatever"},
            )

        assert r.status_code == 400
        body = r.json()
        assert body["fire_id"] is None
        assert "exotic_vendor" in body["message"]

    def test_verifier_returns_false_yields_403(self, fixed_subscription_id: UUID) -> None:
        receiver = _build_receiver()

        def _always_reject(secret: bytes, payload: bytes, signature_value: str) -> bool:
            del secret, payload, signature_value
            return False

        receiver.register_verifier("always_reject", _always_reject)
        sub = _build_subscription(verification_scheme="always_reject")

        async def _adapter_stub(**_kwargs: Any) -> WebhookReceiveResult:
            msg = "adapter must not run when verifier rejects"
            raise AssertionError(msg)

        app = _make_app(receiver)
        with (
            _patch_subscription_lookup(sub),
            patch(
                "threetears.agent.wake.webhook_adapter.webhook_receive",
                _adapter_stub,
            ),
        ):
            client = TestClient(app)
            r = client.post(
                f"/webhooks/{fixed_subscription_id}",
                content=b"{}",
                headers={DEFAULT_SIGNATURE_HEADER: "sha256=bogus"},
            )

        assert r.status_code == 403
        body = r.json()
        assert body["fire_id"] is None
        assert "invalid signature" in body["message"]

    def test_verifier_returns_true_calls_adapter_with_pre_verified(
        self,
        fixed_subscription_id: UUID,
    ) -> None:
        """When the verifier succeeds the adapter MUST be invoked with
        ``pre_verified=True`` so it skips its inline HMAC compute.
        """
        receiver = _build_receiver()

        def _always_accept(secret: bytes, payload: bytes, signature_value: str) -> bool:
            del secret, payload, signature_value
            return True

        receiver.register_verifier("always_accept", _always_accept)
        sub = _build_subscription(verification_scheme="always_accept")
        captured: dict[str, Any] = {}

        async def _adapter_stub(**kwargs: Any) -> WebhookReceiveResult:
            captured.update(kwargs)
            return WebhookReceiveResult(
                status_code=202,
                fire_id=uuid4(),
                message="dispatched",
            )

        app = _make_app(receiver)
        with (
            _patch_subscription_lookup(sub),
            patch(
                "threetears.agent.wake.webhook_adapter.webhook_receive",
                _adapter_stub,
            ),
        ):
            client = TestClient(app)
            r = client.post(
                f"/webhooks/{fixed_subscription_id}",
                content=b"{}",
                headers={DEFAULT_SIGNATURE_HEADER: "sha256=stub"},
            )

        assert r.status_code == 202
        assert captured["pre_verified"] is True
        # The receiver forwards the raw signature header to the
        # adapter (so the adapter still records it for auditing /
        # logging downstream).
        assert captured["signature_header"] == "sha256=stub"

    def test_missing_signature_header_returns_401(
        self,
        fixed_subscription_id: UUID,
    ) -> None:
        """An active subscription with no signature header maps to 401
        before any verifier runs.
        """
        receiver = _build_receiver()
        sub = _build_subscription(verification_scheme="generic_hmac_sha256")

        async def _adapter_stub(**_kwargs: Any) -> WebhookReceiveResult:
            msg = "adapter must not run when the signature header is missing"
            raise AssertionError(msg)

        app = _make_app(receiver)
        with (
            _patch_subscription_lookup(sub),
            patch(
                "threetears.agent.wake.webhook_adapter.webhook_receive",
                _adapter_stub,
            ),
        ):
            client = TestClient(app)
            r = client.post(
                f"/webhooks/{fixed_subscription_id}",
                content=b"{}",
                # NO signature header
            )

        assert r.status_code == 401
        body = r.json()
        assert body["fire_id"] is None
        assert "missing signature header" in body["message"]


# ============================================================
# HTTP status mapping -- routing-and-response plumbing
# ============================================================


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
    """Adapter outcome -> HTTP status code mapping per WEBHOOK-03.

    These tests target the response-mapping plumbing. The subscription
    lookup is stubbed to return ``None`` so the receiver forwards
    straight to the adapter (which produces the canned result).
    """

    def test_202_accepted_includes_fire_id(self, fixed_subscription_id: UUID) -> None:
        receiver = _build_receiver()
        app = _make_app(receiver)
        fire_id = uuid4()
        # Subscription lookup returns None -> receiver forwards to
        # adapter for the canned response; the adapter would normally
        # produce a 404 itself, but the stub overrides that.
        with (
            _patch_subscription_lookup(None),
            _patch_receive(
                status_code=202,
                fire_id=fire_id,
                message="dispatched",
            ),
        ):
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
        """A 400 from the adapter (template render error) propagates verbatim.

        Drives the receiver via an active, registered-scheme subscription
        whose verifier always passes, so the adapter is reached + its
        canned 400 is returned.
        """
        receiver = _build_receiver()
        receiver.register_verifier(
            "always_accept",
            lambda _s, _p, _v: True,
        )
        sub = _build_subscription(verification_scheme="always_accept")
        app = _make_app(receiver)
        with (
            _patch_subscription_lookup(sub),
            _patch_receive(
                status_code=400,
                message="template render error",
            ),
        ):
            client = TestClient(app)
            r = client.post(
                f"/webhooks/{fixed_subscription_id}",
                content=b"bad",
                headers={DEFAULT_SIGNATURE_HEADER: "sha256=stub"},
            )
        assert r.status_code == 400
        assert r.json() == {"fire_id": None, "message": "template render error"}

    def test_401_missing_signature_pass_through_via_adapter(
        self,
        fixed_subscription_id: UUID,
    ) -> None:
        """When the subscription is missing the adapter owns the 401 mapping."""
        receiver = _build_receiver()
        app = _make_app(receiver)
        with (
            _patch_subscription_lookup(None),
            _patch_receive(
                status_code=401,
                message="missing signature header",
            ),
        ):
            client = TestClient(app)
            r = client.post(
                f"/webhooks/{fixed_subscription_id}",
                content=b"{}",
            )
        assert r.status_code == 401
        assert r.json() == {"fire_id": None, "message": "missing signature header"}

    def test_403_source_ip_rejected(self, fixed_subscription_id: UUID) -> None:
        receiver = _build_receiver()
        receiver.register_verifier(
            "always_accept",
            lambda _s, _p, _v: True,
        )
        sub = _build_subscription(verification_scheme="always_accept")
        app = _make_app(receiver)
        with (
            _patch_subscription_lookup(sub),
            _patch_receive(
                status_code=403,
                message="source IP not allowed",
            ),
        ):
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
        with (
            _patch_subscription_lookup(None),
            _patch_receive(
                status_code=404,
                message="subscription not found or paused",
            ),
        ):
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
        receiver.register_verifier(
            "always_accept",
            lambda _s, _p, _v: True,
        )
        sub = _build_subscription(verification_scheme="always_accept")
        app = _make_app(receiver)
        with (
            _patch_subscription_lookup(sub),
            _patch_receive(
                status_code=429,
                message="rate limit exceeded",
            ),
        ):
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
        receiver.register_verifier(
            "always_accept",
            lambda _s, _p, _v: True,
        )
        sub = _build_subscription(verification_scheme="always_accept")
        app = _make_app(receiver)
        fire_id = uuid4()
        with (
            _patch_subscription_lookup(sub),
            _patch_receive(
                status_code=500,
                fire_id=fire_id,
                message="dispatch failed: oops",
            ),
        ):
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
        receiver.register_verifier(
            "always_accept",
            lambda _s, _p, _v: True,
        )
        sub = _build_subscription(verification_scheme="always_accept")
        app = _make_app(receiver)
        with (
            _patch_subscription_lookup(sub),
            _patch_receive(
                status_code=202,
                fire_id=uuid4(),
                message="ok",
            ),
        ):
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
        receiver.register_verifier(
            "always_accept",
            lambda _s, _p, _v: True,
        )
        sub = _build_subscription(verification_scheme="always_accept")
        app = _make_app(receiver)

        captured: dict[str, Any] = {}

        async def _stub(**kwargs: Any) -> WebhookReceiveResult:
            captured.update(kwargs)
            return WebhookReceiveResult(status_code=202, fire_id=uuid4(), message="ok")

        with (
            _patch_subscription_lookup(sub),
            patch(
                "threetears.agent.wake.webhook_adapter.webhook_receive",
                _stub,
            ),
        ):
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
        receiver.register_verifier(
            "always_accept",
            lambda _s, _p, _v: True,
        )
        sub = _build_subscription(verification_scheme="always_accept")
        app = _make_app(receiver)

        captured: dict[str, Any] = {}

        async def _stub(**kwargs: Any) -> WebhookReceiveResult:
            captured.update(kwargs)
            return WebhookReceiveResult(status_code=202, fire_id=uuid4(), message="ok")

        with (
            _patch_subscription_lookup(sub),
            patch(
                "threetears.agent.wake.webhook_adapter.webhook_receive",
                _stub,
            ),
        ):
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
        receiver.register_verifier(
            "always_accept",
            lambda _s, _p, _v: True,
        )
        sub = _build_subscription(verification_scheme="always_accept")
        app = _make_app(receiver)

        captured: dict[str, Any] = {}

        async def _stub(**kwargs: Any) -> WebhookReceiveResult:
            captured.update(kwargs)
            return WebhookReceiveResult(status_code=202, fire_id=uuid4(), message="ok")

        with (
            _patch_subscription_lookup(sub),
            patch(
                "threetears.agent.wake.webhook_adapter.webhook_receive",
                _stub,
            ),
        ):
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
        receiver.register_verifier(
            "always_accept",
            lambda _s, _p, _v: True,
        )
        sub = _build_subscription(verification_scheme="always_accept")
        app = _make_app(receiver)

        captured: dict[str, Any] = {}

        async def _stub(**kwargs: Any) -> WebhookReceiveResult:
            captured.update(kwargs)
            return WebhookReceiveResult(status_code=202, fire_id=uuid4(), message="ok")

        with (
            _patch_subscription_lookup(sub),
            patch(
                "threetears.agent.wake.webhook_adapter.webhook_receive",
                _stub,
            ),
        ):
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
