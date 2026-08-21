"""unit tests for the oversized-publish refusal (structured-result-tiers task-03).

``nats-py`` refuses a publish larger than the broker's advertised ``max_payload``
**client-side**, before any bytes leave the process. That refusal used to reach
callers as a generic :class:`~threetears.nats.errors.PublishError` carrying a
stringified cause, which left the two failures a caller most needs to tell apart
-- "the frame we built is too big" and "the broker went away" -- distinguishable
only by matching on message text. One is a bug in what we built and must never
be retried; the other is an outage and often should be.

These tests assert:

* the refusal arrives as :class:`~threetears.nats.errors.PayloadTooLargeError`,
  carrying both numbers as attributes rather than only as prose;
* every public publish entry point reaches that type, including the two reply
  methods that do NOT share the ``_publish_bytes`` funnel;
* every OTHER publish failure still raises a plain ``PublishError`` -- the
  narrowing must not swallow the general case, which is the regression this
  shape invites;
* :attr:`~threetears.nats.NatsClient.max_payload` answers ``None`` until a
  server has said otherwise, and never substitutes ``nats-py``'s own 1 MB
  pre-connect default.

**No integration test at the real limit is owed here, and that is deliberate.**
Driving it would need a broker with a known ``max_payload`` and a megabyte of
traffic to assert what these fakes already pin -- the classification, not the
enforcement, which is nats-py's and is already tested there.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from nats.errors import MaxPayloadError as _NatsMaxPayloadError
from nats.errors import OutboundBufferLimitError as _NatsOutboundBufferLimitError
from pydantic import BaseModel

from threetears.nats import NatsClient, PayloadTooLargeError, PublishError, Subject, Subjects, set_default_namespace

_BROKER_MAX_PAYLOAD = 1_048_576


class _Hello(BaseModel):
    """sample message type, so the typed publish path carries a real encode."""

    greeting: str


class _RefusingRaw:
    """fake nats-py client whose every publish is refused as oversized.

    models the connected-and-healthy case: the socket is fine, the broker is
    fine, and this one frame is bigger than the server said it would accept.
    """

    def __init__(self, *, max_payload: int | None = _BROKER_MAX_PAYLOAD, is_connected: bool = True) -> None:
        self.is_connected = is_connected
        self.is_closed = False
        if max_payload is not None:
            self.max_payload = max_payload

    async def publish(self, *args: Any, **kwargs: Any) -> None:
        raise _NatsMaxPayloadError


class _BrokenRaw:
    """fake nats-py client whose publish fails for some ORDINARY reason."""

    def __init__(self) -> None:
        self.is_connected = True
        self.is_closed = False
        self.max_payload = _BROKER_MAX_PAYLOAD

    async def publish(self, *args: Any, **kwargs: Any) -> None:
        raise OSError("connection reset by peer")


class _OverflowRaw:
    """fake nats-py client whose publish overflows the outbound buffer.

    the wedge state ``test_outbound_bounds.py`` covers; repeated here only to
    pin that the new narrowing did not capture it.
    """

    def __init__(self) -> None:
        self.is_connected = True
        self.is_closed = False
        self.max_payload = _BROKER_MAX_PAYLOAD

    async def publish(self, *args: Any, **kwargs: Any) -> None:
        raise _NatsOutboundBufferLimitError


def _client(raw: Any) -> NatsClient:
    """build a bare wrapper over a fake raw client (no connect())."""
    set_default_namespace("3tears")
    return NatsClient(raw=raw, namespace="3tears", client_name="test")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# the refusal gets a type, and it names both numbers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_oversized_publish_raises_the_size_specific_type() -> None:
    """the refusal is its own type, not a PublishError carrying a string."""
    client = _client(_RefusingRaw())
    payload = b"x" * 4096

    with pytest.raises(PayloadTooLargeError) as caught:
        await client.publish_raw(subject=Subjects.tools_call(), payload=payload)

    error = caught.value
    assert error.subject == Subjects.tools_call().path
    assert error.size_bytes == len(payload)
    assert error.max_payload == _BROKER_MAX_PAYLOAD
    assert isinstance(error.__cause__, _NatsMaxPayloadError)


@pytest.mark.asyncio
async def test_the_numbers_are_attributes_not_only_message_text() -> None:
    """a caller must be able to act on both numbers without parsing prose."""
    client = _client(_RefusingRaw())

    with pytest.raises(PayloadTooLargeError) as caught:
        await client.publish_raw(subject=Subjects.tools_call(), payload=b"y" * 10)

    error = caught.value
    # the message says it too -- that is for the log line, not for the caller
    assert "10 bytes" in str(error)
    assert str(_BROKER_MAX_PAYLOAD) in str(error)
    assert (error.size_bytes, error.max_payload) == (10, _BROKER_MAX_PAYLOAD)


@pytest.mark.asyncio
async def test_it_stays_catchable_as_a_publish_error() -> None:
    """a refinement of the hierarchy, not a breaking rename: existing catches hold."""
    client = _client(_RefusingRaw())

    with pytest.raises(PublishError):
        await client.publish_raw(subject=Subjects.tools_call(), payload=b"z")
    assert issubclass(PayloadTooLargeError, PublishError)


@pytest.mark.asyncio
async def test_the_limit_is_reported_as_unknown_rather_than_guessed() -> None:
    """a client that cannot say what the broker advertised must not invent 1 MB."""
    client = _client(_RefusingRaw(max_payload=None))

    with pytest.raises(PayloadTooLargeError) as caught:
        await client.publish_raw(subject=Subjects.tools_call(), payload=b"q")

    assert caught.value.max_payload is None
    assert "unknown" in str(caught.value)


# ---------------------------------------------------------------------------
# every entry point, because they do not share one call into nats-py
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_typed_publish_path_reaches_the_type() -> None:
    """``publish(subject=, message=)`` -- the canonical form."""
    client = _client(_RefusingRaw())
    with pytest.raises(PayloadTooLargeError):
        await client.publish(subject=Subjects.tools_call(), message=_Hello(greeting="hi"))


@pytest.mark.asyncio
async def test_the_positional_shorthand_reaches_the_type() -> None:
    """``publish(subject_str, payload_bytes)`` -- the raw shorthand fanouts use."""
    client = _client(_RefusingRaw())
    with pytest.raises(PayloadTooLargeError):
        await client.publish("3tears.tools.call", b"payload")


@pytest.mark.asyncio
async def test_publish_reply_reaches_the_type() -> None:
    """the typed reply path publishes DIRECTLY, bypassing ``_publish_bytes``."""
    client = _client(_RefusingRaw())
    with pytest.raises(PayloadTooLargeError) as caught:
        await client.publish_reply(reply_subject="_INBOX.abc", message=_Hello(greeting="hi"))
    assert caught.value.subject == "_INBOX.abc"


@pytest.mark.asyncio
async def test_publish_raw_reply_reaches_the_type() -> None:
    """so does the raw reply path -- the transparent-proxy escape hatch."""
    client = _client(_RefusingRaw())
    with pytest.raises(PayloadTooLargeError) as caught:
        await client.publish_raw_reply(reply_subject="_INBOX.def", payload=b"body")
    assert caught.value.size_bytes == 4


@pytest.mark.asyncio
async def test_the_jetstream_path_reaches_the_type() -> None:
    """the refusal fires before the ack wait, so the bounded publish sees it too.

    a caller branching on the type must not have to know which publish path
    built the frame.
    """
    raw = _RefusingRaw()
    js = MagicMock()
    js.publish = AsyncMock(side_effect=_NatsMaxPayloadError)
    raw.jetstream = MagicMock(return_value=js)  # type: ignore[attr-defined]
    client = _client(raw)

    with pytest.raises(PayloadTooLargeError) as caught:
        await client.jetstream_publish(
            subject=Subject.raw("3tears.oplog.x"),
            payload=b"a" * 32,
            timeout=timedelta(seconds=1),
        )
    assert caught.value.size_bytes == 32
    assert caught.value.max_payload == _BROKER_MAX_PAYLOAD


# ---------------------------------------------------------------------------
# the narrowing must not swallow the general case
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_ordinary_publish_failure_stays_a_plain_publish_error() -> None:
    """ "the broker went away" must not start reading as "the frame was too big"."""
    client = _client(_BrokenRaw())

    with pytest.raises(PublishError) as caught:
        await client.publish_raw(subject=Subjects.tools_call(), payload=b"x")

    assert not isinstance(caught.value, PayloadTooLargeError)
    assert "connection reset" in str(caught.value)


@pytest.mark.asyncio
async def test_an_outbound_overflow_is_still_counted_and_still_generic() -> None:
    """the overflow -> health signal path is untouched by the new branch."""
    raw = _OverflowRaw()
    client = _client(raw)

    with pytest.raises(PublishError) as caught:
        await client.publish_raw(subject=Subjects.tools_call(), payload=b"x")

    assert not isinstance(caught.value, PayloadTooLargeError)
    assert client._health_state["overflow_events"] == 1  # noqa: SLF001


# ---------------------------------------------------------------------------
# max_payload: an answer, or nothing -- never a guess
# ---------------------------------------------------------------------------


def test_max_payload_reports_what_the_broker_advertised() -> None:
    """the value comes from the connected server, which is its only source of truth."""
    assert _client(_RefusingRaw()).max_payload == _BROKER_MAX_PAYLOAD


def test_max_payload_is_none_while_not_connected() -> None:
    """nats-py pre-fills its own attribute with 1 MB; reading that early is a guess.

    ``None`` is the honest answer, and it is what makes this property safe to
    build a frame against.
    """
    raw = _RefusingRaw(is_connected=False)
    assert raw.max_payload == _BROKER_MAX_PAYLOAD  # nats-py's own default is sitting right there
    assert _client(raw).max_payload is None


def test_max_payload_is_none_when_the_client_cannot_say() -> None:
    """a raw client that does not carry the attribute answers nothing, not zero."""
    assert _client(_RefusingRaw(max_payload=None)).max_payload is None
