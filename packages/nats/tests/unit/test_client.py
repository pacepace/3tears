"""unit tests for :class:`threetears.nats.NatsClient`.

these tests substitute fakes for the underlying nats-py client so the
wrapper logic (typed publish, kw-only subscribe, deadletter, error
mapping) can be exercised without a live broker. integration tests
against a real NATS testcontainer live in tests/integration/.
"""

from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from threetears.nats import (
    DEFAULT_NAMESPACE,
    IncomingMessage,
    NatsClient,
    PublishError,
    RequestError,
    Subject,
    Subjects,
    SubscribeError,
    Subscription,
    set_default_namespace,
)


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class _FakeSubscription:
    """fake nats-py subscription with an asyncio.Queue-backed message stream."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue[Any] = asyncio.Queue()
        self.unsubscribed = False

    async def unsubscribe(self) -> None:
        self.unsubscribed = True

    @property
    def messages(self) -> Any:
        async def _gen() -> Any:
            while True:
                msg = await self.queue.get()
                if msg is None:
                    return
                yield msg

        return _gen()


class _FakeMsg:
    """tiny stand-in for nats.aio.msg.Msg with .data, .reply, and .subject."""

    def __init__(self, *, data: bytes, reply: str = "", subject: str = "aibots.tools.call") -> None:
        self.data = data
        self.reply = reply
        self.subject = subject


class _FakeNatsPyClient:
    """minimal fake of nats.aio.client.Client used by tests."""

    def __init__(self) -> None:
        self.is_closed = False
        self.is_connected = True
        self.published: list[tuple[str, bytes, str | None]] = []
        self.subscribed: list[tuple[str, str | None, _FakeSubscription]] = []
        self.request_response: bytes = b"{}"
        self.request_calls: list[tuple[str, bytes, float]] = []

    async def publish(self, subject: str, payload: bytes, reply: str | None = None) -> None:
        self.published.append((subject, payload, reply))

    async def subscribe(self, subject: str, queue: str = "") -> _FakeSubscription:
        sub = _FakeSubscription()
        self.subscribed.append((subject, queue or None, sub))
        return sub

    async def request(self, subject: str, payload: bytes, timeout: float) -> _FakeMsg:
        self.request_calls.append((subject, payload, timeout))
        return _FakeMsg(data=self.request_response)

    async def drain(self) -> None:
        self.is_closed = True

    async def close(self) -> None:
        self.is_closed = True

    async def flush(self, timeout: float = 2.0) -> None:
        self.flush_calls: list[float]
        if not hasattr(self, "flush_calls"):
            self.flush_calls = []
        self.flush_calls.append(timeout)
        flush_error = getattr(self, "flush_error", None)
        if flush_error is not None:
            raise flush_error


class _Hello(BaseModel):
    """sample message type for typed publish/subscribe tests."""

    greeting: str
    count: int


class _Reply(BaseModel):
    """sample response type for request/reply tests."""

    ok: bool


@pytest.fixture(autouse=True)
def _bind_namespace() -> None:
    """ensure each test starts from a known namespace."""
    set_default_namespace(DEFAULT_NAMESPACE)


def _make_client() -> tuple[NatsClient, _FakeNatsPyClient]:
    """construct a NatsClient backed by a fake nats-py client."""
    fake = _FakeNatsPyClient()
    client = NatsClient(raw=fake, namespace="aibots", client_name="test")  # type: ignore[arg-type]
    return client, fake


@pytest.mark.asyncio
async def test_publish_serializes_pydantic() -> None:
    """publish encodes BaseModel via model_dump_json."""
    client, fake = _make_client()
    subject = Subjects.tools_call()
    msg = _Hello(greeting="hi", count=3)
    await client.publish(subject=subject, message=msg)
    assert len(fake.published) == 1
    sub_path, payload, reply_to = fake.published[0]
    assert sub_path == "aibots.tools.call"
    assert json.loads(payload) == {"greeting": "hi", "count": 3}
    assert reply_to is None


@pytest.mark.asyncio
async def test_publish_with_reply_to() -> None:
    """publish forwards reply_to to underlying client."""
    client, fake = _make_client()
    await client.publish(
        subject=Subjects.tools_call(),
        message=_Hello(greeting="x", count=1),
        reply_to=Subject.raw("aibots.reply.inbox"),
    )
    assert fake.published[0][2] == "aibots.reply.inbox"


@pytest.mark.asyncio
async def test_publish_raw_passes_bytes_unchanged() -> None:
    """publish_raw does not re-serialize payload."""
    client, fake = _make_client()
    await client.publish_raw(subject=Subjects.l3_query(), payload=b"raw-bytes")
    assert fake.published[0][1] == b"raw-bytes"


@pytest.mark.asyncio
async def test_publish_reply_validates_subject() -> None:
    """publish_reply rejects empty reply_subject."""
    client, _ = _make_client()
    with pytest.raises(PublishError):
        await client.publish_reply(reply_subject="", message=_Hello(greeting="x", count=1))


@pytest.mark.asyncio
async def test_publish_wraps_underlying_error() -> None:
    """publish errors raise PublishError, not the underlying exception."""

    class _Boom(_FakeNatsPyClient):
        async def publish(self, subject: str, payload: bytes, reply: str | None = None) -> None:
            raise RuntimeError("transport down")

    fake = _Boom()
    client = NatsClient(raw=fake, namespace="aibots", client_name="test")  # type: ignore[arg-type]
    with pytest.raises(PublishError):
        await client.publish_raw(subject=Subjects.l3_query(), payload=b"x")


@pytest.mark.asyncio
async def test_subscribe_is_kw_only() -> None:
    """positional subscribe args raise TypeError — the cb-as-positional footgun is impossible."""
    client, _ = _make_client()

    async def handler(_msg: IncomingMessage) -> None:
        pass

    with pytest.raises(TypeError):
        await client.subscribe(Subjects.tools_call(), handler)  # type: ignore[misc]


@pytest.mark.asyncio
async def test_subscribe_typed_decodes_pydantic() -> None:
    """subscribe_typed parses incoming bytes into the declared BaseModel."""
    client, fake = _make_client()
    received: list[_Hello] = []

    async def on_msg(msg: _Hello) -> None:
        received.append(msg)

    sub = await client.subscribe_typed(
        subject=Subjects.tools_call(),
        cb=on_msg,
        message_type=_Hello,
    )
    assert isinstance(sub, Subscription)
    fake_sub = fake.subscribed[0][2]
    payload = _Hello(greeting="hello", count=7).model_dump_json().encode("utf-8")
    await fake_sub.queue.put(_FakeMsg(data=payload))
    # let the dispatch task run
    await asyncio.sleep(0.05)
    assert received == [_Hello(greeting="hello", count=7)]
    await client.unsubscribe(sub)


@pytest.mark.asyncio
async def test_subscribe_typed_validation_failure_deadletters() -> None:
    """validation failure publishes to deadletter when deadletter_on_error=True (default)."""
    client, fake = _make_client()

    async def on_msg(_msg: _Hello) -> None:
        pytest.fail("callback should not run on invalid payload")

    sub = await client.subscribe_typed(
        subject=Subjects.tools_call(),
        cb=on_msg,
        message_type=_Hello,
    )
    fake_sub = fake.subscribed[0][2]
    bad = b'{"not_the_right_field": true}'
    await fake_sub.queue.put(_FakeMsg(data=bad))
    await asyncio.sleep(0.05)
    # validate that the bad message landed on deadletter subject
    dl_publishes = [p for p in fake.published if p[0].startswith("aibots.deadletter.")]
    assert len(dl_publishes) == 1
    assert dl_publishes[0][0] == "aibots.deadletter.aibots.tools.call"
    envelope = json.loads(dl_publishes[0][1])
    assert envelope["original_subject"] == "aibots.tools.call"
    assert envelope["error_type"] == "ValidationError"
    await client.unsubscribe(sub)


@pytest.mark.asyncio
async def test_subscribe_callback_exception_deadletters() -> None:
    """callback exceptions deadletter when deadletter_on_error=True."""
    client, fake = _make_client()

    async def on_msg(_msg: _Hello) -> None:
        raise RuntimeError("kaboom")

    sub = await client.subscribe_typed(
        subject=Subjects.tools_call(),
        cb=on_msg,
        message_type=_Hello,
    )
    fake_sub = fake.subscribed[0][2]
    payload = _Hello(greeting="x", count=1).model_dump_json().encode("utf-8")
    await fake_sub.queue.put(_FakeMsg(data=payload))
    await asyncio.sleep(0.05)
    dl = [p for p in fake.published if p[0].startswith("aibots.deadletter.")]
    assert len(dl) == 1
    envelope = json.loads(dl[0][1])
    assert envelope["error_type"] == "RuntimeError"
    assert envelope["error_message"] == "kaboom"
    await client.unsubscribe(sub)


@pytest.mark.asyncio
async def test_subscribe_callback_exception_no_deadletter_when_disabled() -> None:
    """deadletter_on_error=False — callback exceptions log only, no republish."""
    client, fake = _make_client()

    async def on_msg(_msg: _Hello) -> None:
        raise RuntimeError("kaboom")

    sub = await client.subscribe_typed(
        subject=Subjects.tools_call(),
        cb=on_msg,
        message_type=_Hello,
        deadletter_on_error=False,
    )
    fake_sub = fake.subscribed[0][2]
    payload = _Hello(greeting="x", count=1).model_dump_json().encode("utf-8")
    await fake_sub.queue.put(_FakeMsg(data=payload))
    await asyncio.sleep(0.05)
    dl = [p for p in fake.published if p[0].startswith("aibots.deadletter.")]
    assert len(dl) == 0
    await client.unsubscribe(sub)


@pytest.mark.asyncio
async def test_subscribe_with_queue_group() -> None:
    """queue group is forwarded to underlying subscribe."""
    client, fake = _make_client()

    async def handler(_msg: IncomingMessage) -> None:
        pass

    await client.subscribe(
        subject=Subjects.audit_wildcard(),
        cb=handler,
        queue="audit-consumer",
    )
    assert fake.subscribed[0][1] == "audit-consumer"


@pytest.mark.asyncio
async def test_subscribe_max_in_flight_invalid() -> None:
    """max_in_flight=0 is rejected."""
    client, _ = _make_client()

    async def handler(_msg: IncomingMessage) -> None:
        pass

    with pytest.raises(SubscribeError):
        await client.subscribe(
            subject=Subjects.tools_call(),
            cb=handler,
            max_in_flight=0,
        )


@pytest.mark.asyncio
async def test_request_typed_round_trip() -> None:
    """request encodes message, decodes response into declared type."""
    client, fake = _make_client()
    fake.request_response = _Reply(ok=True).model_dump_json().encode("utf-8")
    response = await client.request(
        subject=Subjects.tools_call(),
        message=_Hello(greeting="ping", count=1),
        response_type=_Reply,
        timeout=timedelta(seconds=2),
    )
    assert response == _Reply(ok=True)
    assert fake.request_calls[0][2] == 2.0  # timeout converted to seconds


@pytest.mark.asyncio
async def test_request_decode_failure_raises_request_error() -> None:
    """response that fails to decode raises RequestError."""
    client, fake = _make_client()
    fake.request_response = b'{"not_a_valid_reply": 1}'
    with pytest.raises(RequestError):
        await client.request(
            subject=Subjects.tools_call(),
            message=_Hello(greeting="x", count=1),
            response_type=_Reply,
        )


@pytest.mark.asyncio
async def test_request_timeout_maps_to_request_error() -> None:
    """nats-py TimeoutError maps to RequestError."""
    from nats.errors import TimeoutError as NatsTimeout

    class _TimeoutClient(_FakeNatsPyClient):
        async def request(self, subject: str, payload: bytes, timeout: float) -> _FakeMsg:
            raise NatsTimeout()

    fake = _TimeoutClient()
    client = NatsClient(raw=fake, namespace="aibots", client_name="t")  # type: ignore[arg-type]
    with pytest.raises(RequestError) as exc_info:
        await client.request_raw(
            subject=Subjects.tools_call(),
            payload=b"x",
            timeout=timedelta(seconds=1),
        )
    assert "timed out" in str(exc_info.value)


@pytest.mark.asyncio
async def test_unsubscribe_marks_closed_and_drops_underlying() -> None:
    """unsubscribe is idempotent and unsubscribes underlying."""
    client, fake = _make_client()

    async def handler(_msg: IncomingMessage) -> None:
        pass

    sub = await client.subscribe(subject=Subjects.tools_call(), cb=handler)
    assert sub.is_closed is False
    await client.unsubscribe(sub)
    assert sub.is_closed is True
    fake_sub = fake.subscribed[0][2]
    assert fake_sub.unsubscribed is True
    # second call is a no-op
    await client.unsubscribe(sub)


@pytest.mark.asyncio
async def test_shutdown_drains_subscriptions() -> None:
    """shutdown unsubscribes everything and drains underlying client."""
    client, fake = _make_client()

    async def handler(_msg: IncomingMessage) -> None:
        pass

    sub = await client.subscribe(subject=Subjects.tools_call(), cb=handler)
    await client.shutdown(drain_timeout=timedelta(seconds=1))
    assert sub.is_closed is True
    assert fake.is_closed is True


@pytest.mark.asyncio
async def test_subscribe_callback_receives_incoming_message_envelope() -> None:
    """raw subscribe callback receives :class:`IncomingMessage` carrying data + reply_subject."""
    client, fake = _make_client()
    received: list[IncomingMessage] = []

    async def handler(msg: IncomingMessage) -> None:
        received.append(msg)

    sub = await client.subscribe(subject=Subjects.tools_call(), cb=handler)
    fake_sub = fake.subscribed[0][2]
    await fake_sub.queue.put(_FakeMsg(data=b"payload-bytes", reply="aibots._INBOX.42"))
    await asyncio.sleep(0.05)
    assert len(received) == 1
    assert received[0].data == b"payload-bytes"
    assert received[0].reply_subject == "aibots._INBOX.42"
    await client.unsubscribe(sub)


@pytest.mark.asyncio
async def test_subscribe_callback_reply_subject_none_for_pubsub() -> None:
    """raw subscribe callback receives ``reply_subject=None`` when producer skipped reply-to."""
    client, fake = _make_client()
    received: list[IncomingMessage] = []

    async def handler(msg: IncomingMessage) -> None:
        received.append(msg)

    sub = await client.subscribe(subject=Subjects.tools_call(), cb=handler)
    fake_sub = fake.subscribed[0][2]
    # _FakeMsg.reply defaults to "" — the wrapper coerces empty to None
    await fake_sub.queue.put(_FakeMsg(data=b"pubsub"))
    await asyncio.sleep(0.05)
    assert received == [
        IncomingMessage(data=b"pubsub", reply_subject=None, subject="aibots.tools.call"),
    ]
    await client.unsubscribe(sub)


@pytest.mark.asyncio
async def test_connect_validates_url() -> None:
    """connect rejects empty nats_url."""
    from threetears.nats import NatsClientError

    with pytest.raises(NatsClientError):
        await NatsClient.connect(nats_url="", client_name="x")


@pytest.mark.asyncio
async def test_connect_validates_client_name() -> None:
    """connect rejects empty client_name."""
    from threetears.nats import NatsClientError

    with pytest.raises(NatsClientError):
        await NatsClient.connect(nats_url="nats://x", client_name="")


@pytest.mark.asyncio
async def test_connect_validates_namespace() -> None:
    """connect rejects empty subject namespace."""
    from threetears.nats import NatsClientError

    with pytest.raises(NatsClientError):
        await NatsClient.connect(
            nats_url="nats://x",
            client_name="x",
            nats_subject_namespace="",
        )


@pytest.mark.asyncio
async def test_ping_returns_true_when_flush_succeeds() -> None:
    """ping forwards to nats-py flush and returns True on success."""
    client, fake = _make_client()
    fake.is_connected = True
    result = await client.ping(timeout=1.5)
    assert result is True
    assert fake.flush_calls == [1.5]


@pytest.mark.asyncio
async def test_ping_returns_false_when_disconnected() -> None:
    """ping short-circuits when the local socket reports disconnected."""
    client, fake = _make_client()
    fake.is_connected = False
    result = await client.ping()
    assert result is False
    assert not hasattr(fake, "flush_calls") or fake.flush_calls == []


@pytest.mark.asyncio
async def test_ping_returns_false_when_flush_raises() -> None:
    """ping converts a flush exception (timeout / broker error) into False."""
    client, fake = _make_client()
    fake.is_connected = True
    fake.flush_error = TimeoutError("server slow")
    result = await client.ping(timeout=0.1)
    assert result is False


# ---------------------------------------------------------------------------
# JetStream durable-delivery helpers
# ---------------------------------------------------------------------------


def _client_with_js() -> tuple[NatsClient, Any]:
    """build a client whose JetStream context is a mock JS recorder.

    ``jetstream_context()`` returns ``self._raw.jetstream()``, so the seam is
    stubbed on the fake raw client (the wrapper method is read-only).
    """
    client, fake = _make_client()
    js = MagicMock()
    js.add_stream = AsyncMock()
    js.update_stream = AsyncMock()
    js.publish = AsyncMock()
    js.subscribe = AsyncMock(return_value="sub-handle")
    fake.jetstream = MagicMock(return_value=js)  # type: ignore[attr-defined]
    return client, js


@pytest.mark.asyncio
async def test_ensure_jetstream_stream_creates_and_returns_full_name() -> None:
    """ensure_jetstream_stream adds the stream and returns the prefixed name."""
    from nats.js.api import StorageType

    client, js = _client_with_js()
    full = await client.ensure_jetstream_stream(
        name="channels-deliver",
        subjects=["aibots.channels.deliver.*"],
        storage="file",
    )

    assert full == "aibots-channels-deliver"
    js.add_stream.assert_awaited_once()
    js.update_stream.assert_not_awaited()
    config = js.add_stream.await_args.args[0]
    assert config.name == "aibots-channels-deliver"
    assert config.subjects == ["aibots.channels.deliver.*"]
    assert config.storage == StorageType.FILE


@pytest.mark.asyncio
async def test_ensure_jetstream_stream_memory_storage() -> None:
    """storage='memory' selects the MEMORY storage type."""
    from nats.js.api import StorageType

    client, js = _client_with_js()
    await client.ensure_jetstream_stream(
        name="ephemeral",
        subjects=["aibots.x.*"],
        storage="memory",
    )
    config = js.add_stream.await_args.args[0]
    assert config.storage == StorageType.MEMORY


@pytest.mark.asyncio
async def test_ensure_jetstream_stream_reconciles_existing() -> None:
    """when the stream already exists, fall back to update_stream (reconcile)."""
    client, js = _client_with_js()
    js.add_stream.side_effect = RuntimeError("stream name already in use")

    full = await client.ensure_jetstream_stream(
        name="channels-deliver",
        subjects=["aibots.channels.deliver.*", "aibots.channels.deliver.slack"],
        storage="file",
    )

    assert full == "aibots-channels-deliver"
    js.update_stream.assert_awaited_once()
    config = js.update_stream.await_args.args[0]
    assert config.subjects == [
        "aibots.channels.deliver.*",
        "aibots.channels.deliver.slack",
    ]


@pytest.mark.asyncio
async def test_ensure_jetstream_stream_propagates_when_both_fail() -> None:
    """a malformed config that fails create AND update surfaces, not silently."""
    client, js = _client_with_js()
    js.add_stream.side_effect = RuntimeError("create failed")
    js.update_stream.side_effect = ValueError("malformed config")

    with pytest.raises(ValueError, match="malformed config"):
        await client.ensure_jetstream_stream(
            name="bad",
            subjects=["aibots.x.*"],
            storage="file",
        )


@pytest.mark.asyncio
async def test_jetstream_publish_awaits_puback() -> None:
    """jetstream_publish persists the payload via the JS context."""
    client, js = _client_with_js()
    subject = Subjects.channels_deliver("slack")
    await client.jetstream_publish(subject=subject, payload=b"answer-bytes")

    js.publish.assert_awaited_once_with(subject.path, b"answer-bytes")


def _fake_js_msg(*, data: bytes = b"answer", num_delivered: int = 1) -> Any:
    """build a fake nats-py JetStream message for the bounded-cb tests."""
    msg = MagicMock()
    msg.data = data
    msg.metadata = MagicMock()
    msg.metadata.num_delivered = num_delivered
    msg.ack = AsyncMock()
    msg.nak = AsyncMock()
    return msg


@pytest.mark.asyncio
async def test_jetstream_subscribe_durable_binds_manual_ack_and_max_deliver() -> None:
    """the durable consumer binds manual-ack with a bounded max_deliver config."""
    client, js = _client_with_js()
    subject = Subjects.channels_deliver("slack")
    cb = AsyncMock()

    handle = await client.jetstream_subscribe_durable(
        subject=subject,
        durable="slack-delivery",
        cb=cb,
        max_deliver=5,
        dead_letter_subject=Subjects.channels_deliver("deadletter"),
        stream="aibots-channels-deliver",
    )

    assert handle == "sub-handle"
    kwargs = js.subscribe.await_args.kwargs
    assert js.subscribe.await_args.args[0] == subject.path
    assert kwargs["durable"] == "slack-delivery"
    assert kwargs["manual_ack"] is True
    assert kwargs["stream"] == "aibots-channels-deliver"
    # the consumer config carries the bounded redelivery policy.
    assert kwargs["config"].max_deliver == 5
    assert kwargs["config"].durable_name == "slack-delivery"
    # the cb is WRAPPED (not the raw cb) so the factory owns redelivery.
    assert kwargs["cb"] is not cb


@pytest.mark.asyncio
async def test_durable_requires_positive_max_deliver() -> None:
    """max_deliver < 1 is rejected: a durable consumer cannot redeliver forever."""
    client, _js = _client_with_js()
    with pytest.raises(ValueError, match="max_deliver"):
        await client.jetstream_subscribe_durable(
            subject=Subjects.channels_deliver("slack"),
            durable="d",
            cb=AsyncMock(),
            max_deliver=0,
        )


@pytest.mark.asyncio
async def test_durable_cb_success_defers_ack_to_handler() -> None:
    """on success the wrapper just runs the handler (which owns the ack)."""
    client, js = _client_with_js()
    cb = AsyncMock()
    await client.jetstream_subscribe_durable(
        subject=Subjects.channels_deliver("slack"),
        durable="d",
        cb=cb,
        max_deliver=5,
    )
    wrapped = js.subscribe.await_args.kwargs["cb"]
    msg = _fake_js_msg()
    await wrapped(msg)
    cb.assert_awaited_once_with(msg)
    msg.nak.assert_not_awaited()
    msg.ack.assert_not_awaited()  # the handler acks, not the wrapper


@pytest.mark.asyncio
async def test_durable_cb_naks_with_backoff_before_budget() -> None:
    """a retryable raise on an early attempt naks with a backoff delay."""
    client, js = _client_with_js()
    cb = AsyncMock(side_effect=RuntimeError("slack 429"))
    await client.jetstream_subscribe_durable(
        subject=Subjects.channels_deliver("slack"),
        durable="d",
        cb=cb,
        max_deliver=5,
        dead_letter_subject=Subjects.channels_deliver("deadletter"),
    )
    wrapped = js.subscribe.await_args.kwargs["cb"]
    msg = _fake_js_msg(num_delivered=1)
    await wrapped(msg)
    msg.nak.assert_awaited_once()
    assert msg.nak.await_args.kwargs.get("delay", 0) > 0
    msg.ack.assert_not_awaited()
    js.publish.assert_not_awaited()  # not yet at the budget -> no dead-letter


@pytest.mark.asyncio
async def test_durable_cb_dead_letters_at_budget() -> None:
    """a retryable raise on the LAST attempt parks the payload + acks (one alert)."""
    client, js = _client_with_js()
    cb = AsyncMock(side_effect=RuntimeError("channel deleted"))
    dlq = Subjects.channels_deliver("deadletter")
    await client.jetstream_subscribe_durable(
        subject=Subjects.channels_deliver("slack"),
        durable="d",
        cb=cb,
        max_deliver=5,
        dead_letter_subject=dlq,
    )
    wrapped = js.subscribe.await_args.kwargs["cb"]
    msg = _fake_js_msg(data=b"poison", num_delivered=5)
    await wrapped(msg)
    # parked on the dead-letter subject, then acked so it leaves the live consumer.
    js.publish.assert_awaited_once_with(dlq.path, b"poison")
    msg.ack.assert_awaited_once()
    msg.nak.assert_not_awaited()
