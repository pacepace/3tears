"""Learning that a message did not arrive.

An SMTP send is fire-and-forget: the relay accepts the message and the conversation
ends, so nothing on the outbound path can ever report a bounce. The provider tells us
later, over HTTP, and the only thing standing between that callback and a product's
suppression list is a signature check.

The receiver deliberately does NOT go through `WebhookReceiver`. That path resolves a
`webhook_subscriptions` row, decrypts a per-subscription secret and dispatches a
`WakeTrigger` into a conversation -- a bounce has no conversation and no agent, and is
the same thread-shaped mismatch `ChannelDeliveryMessage` has for an email recipient.
What it DOES reuse is that path's verification: the same `Verifier` callable shape, the
same canonical `verify_generic_hmac_sha256`, the same signature header and size cap.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from pydantic import SecretStr
from threetears.agent.wake.hmac_util import compute_generic_hmac_sha256_signature

from threetears.channels.mail import bounce as bounce_module
from threetears.channels.mail.bounce import (
    BounceReceiver,
    DeliveryEvent,
    DeliveryEventType,
)

_SECRET = "provider-shared-secret"
_SECRET_REF = "env://MAIL_BOUNCE_SECRET"


@pytest.fixture(autouse=True)
def _resolved_secret(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Answer every secret-reference resolution with the shared test secret.

    :param monkeypatch: the active monkeypatch fixture
    :ptype monkeypatch: pytest.MonkeyPatch
    :return: the list each requested reference is appended to
    :rtype: list[str]
    """
    seen: list[str] = []

    def _resolve(ref: str) -> SecretStr:
        seen.append(ref)
        return SecretStr(_SECRET)

    monkeypatch.setattr(bounce_module, "resolve_secret", _resolve)
    return seen


# parity-with: threetears.channels.mail.bounce.DeliveryEventParser
class _JsonParser:
    """A minimal stand-in for a provider adapter: one JSON object per event."""

    def parse(self, payload: bytes) -> list[DeliveryEvent]:
        raw = json.loads(payload)
        return [
            DeliveryEvent(
                event_type=DeliveryEventType(item["type"]),
                recipient=item["recipient"],
                date_occurred=datetime.now(UTC),
                diagnostic_code=item.get("diagnostic"),
            )
            for item in raw
        ]


class _RecordingHandler:
    def __init__(self, raises: Exception | None = None) -> None:
        self.events: list[DeliveryEvent] = []
        self._raises = raises

    async def __call__(self, event: DeliveryEvent) -> None:
        self.events.append(event)
        if self._raises is not None:
            raise self._raises


def _receiver(handler: _RecordingHandler, *, max_payload_bytes: int | None = None) -> BounceReceiver:
    if max_payload_bytes is None:
        return BounceReceiver(secret_ref=_SECRET_REF, parser=_JsonParser(), handler=handler)
    return BounceReceiver(
        secret_ref=_SECRET_REF,
        parser=_JsonParser(),
        handler=handler,
        max_payload_bytes=max_payload_bytes,
    )


def _payload(*events: dict[str, str]) -> bytes:
    return json.dumps(list(events)).encode("utf-8")


def _signed(payload: bytes) -> str:
    return compute_generic_hmac_sha256_signature(_SECRET.encode("utf-8"), payload)


class TestAVerifiedCallbackBecomesDeliveryEvents:
    async def test_a_hard_bounce_reaches_the_handler(self) -> None:
        handler = _RecordingHandler()
        payload = _payload({"type": "bounced_hard", "recipient": "ada@acme.example"})

        result = await _receiver(handler).handle_payload(payload, _signed(payload))

        assert result.status_code == 202
        assert result.accepted == 1
        assert handler.events[0].event_type is DeliveryEventType.BOUNCED_HARD
        assert handler.events[0].recipient == "ada@acme.example"

    async def test_every_event_in_one_callback_is_dispatched(self) -> None:
        """Providers batch. A receiver that handled only the first would silently drop
        the rest of a delivery report."""
        handler = _RecordingHandler()
        payload = _payload(
            {"type": "bounced_hard", "recipient": "a@acme.example"},
            {"type": "bounced_soft", "recipient": "b@acme.example"},
            {"type": "complained", "recipient": "c@acme.example"},
        )

        result = await _receiver(handler).handle_payload(payload, _signed(payload))

        assert result.accepted == 3
        assert [event.event_type for event in handler.events] == [
            DeliveryEventType.BOUNCED_HARD,
            DeliveryEventType.BOUNCED_SOFT,
            DeliveryEventType.COMPLAINED,
        ]

    def test_hard_and_soft_bounces_are_distinct_values(self) -> None:
        """A consumer suppresses on a hard bounce immediately and retries a soft one.
        Collapsing them into one `bounced` value makes that policy unexpressible."""
        assert DeliveryEventType.BOUNCED_HARD is not DeliveryEventType.BOUNCED_SOFT
        assert {member.value for member in DeliveryEventType} >= {
            "delivered",
            "bounced_hard",
            "bounced_soft",
            "complained",
            "deferred",
            "unsubscribed",
        }


class TestWhatItRefuses:
    async def test_an_invalid_signature_is_refused_and_nothing_is_dispatched(self) -> None:
        handler = _RecordingHandler()
        payload = _payload({"type": "bounced_hard", "recipient": "ada@acme.example"})

        result = await _receiver(handler).handle_payload(payload, "sha256=" + "0" * 64)

        assert result.status_code == 403
        assert handler.events == []

    async def test_a_signature_over_a_different_body_is_refused(self) -> None:
        """The whole point of signing the raw bytes: a valid signature for some other
        payload must not admit this one."""
        handler = _RecordingHandler()
        signature = _signed(_payload({"type": "delivered", "recipient": "ada@acme.example"}))
        payload = _payload({"type": "bounced_hard", "recipient": "mallory@evil.example"})

        result = await _receiver(handler).handle_payload(payload, signature)

        assert result.status_code == 403
        assert handler.events == []

    async def test_a_missing_signature_is_refused(self) -> None:
        handler = _RecordingHandler()
        payload = _payload({"type": "bounced_hard", "recipient": "ada@acme.example"})

        result = await _receiver(handler).handle_payload(payload, None)

        assert result.status_code == 401
        assert handler.events == []

    async def test_an_oversized_payload_is_refused_before_the_hmac(self) -> None:
        """The whole body has to sit in memory for a constant-time compare, so the cap
        comes first -- otherwise the size limit is enforced by the memory allocator."""
        handler = _RecordingHandler()
        payload = b"x" * 100

        receiver = _receiver(handler, max_payload_bytes=10)

        result = await receiver.handle_payload(payload, _signed(payload))

        assert result.status_code == 413
        assert handler.events == []

    async def test_a_payload_the_parser_cannot_read_is_a_bad_request(self) -> None:
        """Signed but unreadable is the provider's problem, not ours. A 500 would make
        it retry the same broken body forever."""
        handler = _RecordingHandler()
        payload = b"{not json"

        result = await _receiver(handler).handle_payload(payload, _signed(payload))

        assert result.status_code == 400
        assert handler.events == []

    async def test_an_unknown_event_type_is_a_bad_request_not_a_crash(self) -> None:
        handler = _RecordingHandler()
        payload = _payload({"type": "teleported", "recipient": "ada@acme.example"})

        result = await _receiver(handler).handle_payload(payload, _signed(payload))

        assert result.status_code == 400


class TestTheSecretIsAReferenceResolvedPerCallback:
    async def test_the_reference_is_resolved_on_every_callback(self, _resolved_secret: list[str]) -> None:
        """Same discipline as the send path. A rotated shared secret takes effect on the
        next callback, not the next deploy."""
        handler = _RecordingHandler()
        payload = _payload({"type": "delivered", "recipient": "ada@acme.example"})
        receiver = _receiver(handler)

        await receiver.handle_payload(payload, _signed(payload))
        await receiver.handle_payload(payload, _signed(payload))

        assert _resolved_secret == [_SECRET_REF, _SECRET_REF]

    async def test_an_unresolvable_reference_refuses_rather_than_admits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fail CLOSED. A receiver that could not verify must not accept, because
        "accepted" is what tells the provider to stop resending."""

        def _explode(ref: str) -> SecretStr:
            raise RuntimeError("secret backend unavailable")

        monkeypatch.setattr(bounce_module, "resolve_secret", _explode)
        handler = _RecordingHandler()
        payload = _payload({"type": "delivered", "recipient": "ada@acme.example"})

        result = await _receiver(handler).handle_payload(payload, _signed(payload))

        assert result.status_code == 500
        assert handler.events == []


class TestHandlerFailure:
    async def test_a_failing_handler_asks_the_provider_to_resend(self) -> None:
        """A 2xx tells the provider the event is durably ours. Returning one after the
        write failed loses the bounce, and nothing else will ever report it."""
        handler = _RecordingHandler(raises=RuntimeError("suppression write failed"))
        payload = _payload({"type": "bounced_hard", "recipient": "ada@acme.example"})

        result = await _receiver(handler).handle_payload(payload, _signed(payload))

        assert result.status_code == 500

    async def test_the_recipient_never_appears_in_the_result_message(self) -> None:
        """The message reaches a log line, and an address is PII."""
        handler = _RecordingHandler(raises=RuntimeError("could not write ada@acme.example"))
        payload = _payload({"type": "bounced_hard", "recipient": "ada@acme.example"})

        result = await _receiver(handler).handle_payload(payload, _signed(payload))

        assert "ada@acme.example" not in result.message


class TestTheHttpMount:
    async def test_it_mounts_on_a_fastapi_app_and_returns_the_status(self) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        handler = _RecordingHandler()
        app = FastAPI()
        _receiver(handler).register(app, path="/webhooks/mail/bounce")
        payload = _payload({"type": "bounced_hard", "recipient": "ada@acme.example"})

        with TestClient(app) as client:
            response = client.post(
                "/webhooks/mail/bounce",
                content=payload,
                headers={bounce_module.DEFAULT_SIGNATURE_HEADER: _signed(payload)},
            )

        assert response.status_code == 202
        assert handler.events[0].recipient == "ada@acme.example"

    async def test_an_unsigned_post_is_refused_at_the_http_boundary(self) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        handler = _RecordingHandler()
        app = FastAPI()
        _receiver(handler).register(app, path="/webhooks/mail/bounce")

        with TestClient(app) as client:
            response = client.post("/webhooks/mail/bounce", content=b"[]")

        assert response.status_code == 401
        assert handler.events == []
