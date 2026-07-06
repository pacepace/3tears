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

    def __init__(self, *, data: bytes, reply: str = "", subject: str = "3tears.tools.call") -> None:
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
        # records errors passed to the nats-py op-error reconnect entry point that
        # :meth:`NatsClient.reconnect` drives.
        self.op_err_calls: list[Exception] = []

    async def _process_op_err(self, e: Exception) -> None:
        self.op_err_calls.append(e)

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
    set_default_namespace("3tears")


def _make_client() -> tuple[NatsClient, _FakeNatsPyClient]:
    """construct a NatsClient backed by a fake nats-py client."""
    fake = _FakeNatsPyClient()
    client = NatsClient(raw=fake, namespace="3tears", client_name="test")  # type: ignore[arg-type]
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
    assert sub_path == "3tears.tools.call"
    assert json.loads(payload) == {"greeting": "hi", "count": 3}
    assert reply_to is None


@pytest.mark.asyncio
async def test_publish_with_reply_to() -> None:
    """publish forwards reply_to to underlying client."""
    client, fake = _make_client()
    await client.publish(
        subject=Subjects.tools_call(),
        message=_Hello(greeting="x", count=1),
        reply_to=Subject.raw("3tears.reply.inbox"),
    )
    assert fake.published[0][2] == "3tears.reply.inbox"


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
    client = NatsClient(raw=fake, namespace="3tears", client_name="test")  # type: ignore[arg-type]
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
    """validation failure publishes to deadletter when deadletter_on_failure=True (default)."""
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
    dl_publishes = [p for p in fake.published if p[0].startswith("3tears.deadletter.")]
    assert len(dl_publishes) == 1
    assert dl_publishes[0][0] == "3tears.deadletter.3tears.tools.call"
    envelope = json.loads(dl_publishes[0][1])
    assert envelope["original_subject"] == "3tears.tools.call"
    assert envelope["error_type"] == "ValidationError"
    await client.unsubscribe(sub)


@pytest.mark.asyncio
async def test_subscribe_callback_exception_deadletters() -> None:
    """callback exceptions deadletter when deadletter_on_failure=True."""
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
    dl = [p for p in fake.published if p[0].startswith("3tears.deadletter.")]
    assert len(dl) == 1
    envelope = json.loads(dl[0][1])
    assert envelope["error_type"] == "RuntimeError"
    assert envelope["error_message"] == "kaboom"
    await client.unsubscribe(sub)


@pytest.mark.asyncio
async def test_subscribe_callback_exception_no_deadletter_when_disabled() -> None:
    """deadletter_on_failure=False — callback exceptions log only, no republish."""
    client, fake = _make_client()

    async def on_msg(_msg: _Hello) -> None:
        raise RuntimeError("kaboom")

    sub = await client.subscribe_typed(
        subject=Subjects.tools_call(),
        cb=on_msg,
        message_type=_Hello,
        deadletter_on_failure=False,
    )
    fake_sub = fake.subscribed[0][2]
    payload = _Hello(greeting="x", count=1).model_dump_json().encode("utf-8")
    await fake_sub.queue.put(_FakeMsg(data=payload))
    await asyncio.sleep(0.05)
    dl = [p for p in fake.published if p[0].startswith("3tears.deadletter.")]
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
    client = NatsClient(raw=fake, namespace="3tears", client_name="t")  # type: ignore[arg-type]
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
    await fake_sub.queue.put(_FakeMsg(data=b"payload-bytes", reply="3tears._INBOX.42"))
    await asyncio.sleep(0.05)
    assert len(received) == 1
    assert received[0].data == b"payload-bytes"
    assert received[0].reply_subject == "3tears._INBOX.42"
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
        IncomingMessage(data=b"pubsub", reply_subject=None, subject="3tears.tools.call"),
    ]
    await client.unsubscribe(sub)


@pytest.mark.asyncio
async def test_connect_validates_url() -> None:
    """connect rejects empty nats_url."""
    from threetears.nats import NatsClientError

    with pytest.raises(NatsClientError):
        await NatsClient.connect(nats_url="", client_name="x", nats_subject_namespace="3tears")


@pytest.mark.asyncio
async def test_connect_validates_client_name() -> None:
    """connect rejects empty client_name."""
    from threetears.nats import NatsClientError

    with pytest.raises(NatsClientError):
        await NatsClient.connect(nats_url="nats://x", client_name="", nats_subject_namespace="3tears")


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
async def test_connect_passes_credentials_and_scoped_inbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """the auth_token provider, user_credentials, and a scoped inbox_prefix reach the connect options."""
    import threetears.nats.client as client_module

    captured: dict[str, Any] = {}

    async def _fake_establish(servers: list[str], options: dict[str, Any], nats_url: str) -> Any:
        captured["options"] = options
        return MagicMock()

    monkeypatch.setattr(client_module, "_establish_connection", _fake_establish)

    await NatsClient.connect(
        nats_url="nats://localhost:4222",
        nats_subject_namespace="3tears",
        client_name="agent-x",
        auth_token=lambda: "bootstrap-tok",
        user_credentials="/run/secrets/agent.creds",
        inbox_prefix="_INBOX_agent_pod_pod-1",
        verify_jetstream=False,
    )

    options = captured["options"]
    assert options["token"]() == "bootstrap-tok"  # a provider callable; nats-py invokes it per connect
    assert options["user_credentials"] == "/run/secrets/agent.creds"
    assert options["inbox_prefix"] == b"_INBOX_agent_pod_pod-1"  # bytes, scoped per-principal


@pytest.mark.asyncio
async def test_connect_auth_token_provider_is_reinvoked_per_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """the auth_token provider is passed through UNWRAPPED so nats-py re-invokes it on every
    (re)connect — each reconnect presents a freshly-minted token, never a cached (expired) snapshot."""
    import threetears.nats.client as client_module

    captured: dict[str, Any] = {}

    async def _fake_establish(servers: list[str], options: dict[str, Any], nats_url: str) -> Any:
        captured["options"] = options
        return MagicMock()

    monkeypatch.setattr(client_module, "_establish_connection", _fake_establish)

    minted: list[str] = []

    def _mint() -> str:
        token = f"tok-{len(minted)}"
        minted.append(token)
        return token

    await NatsClient.connect(
        nats_url="nats://localhost:4222",
        nats_subject_namespace="3tears",
        client_name="agent-x",
        auth_token=_mint,
        verify_jetstream=False,
    )

    provider = captured["options"]["token"]
    # stored as the callable itself (not a snapshotted string): successive calls yield successive
    # fresh tokens, which is exactly what nats-py's per-(re)connect invocation relies on.
    assert callable(provider)
    assert provider() == "tok-0"
    assert provider() == "tok-1"


@pytest.mark.asyncio
async def test_connect_passes_user_and_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """user + password reach the nats-py connect options (config-mode static auth_users)."""
    import threetears.nats.client as client_module

    captured: dict[str, Any] = {}

    async def _fake_establish(servers: list[str], options: dict[str, Any], nats_url: str) -> Any:
        captured["options"] = options
        return MagicMock()

    monkeypatch.setattr(client_module, "_establish_connection", _fake_establish)

    await NatsClient.connect(
        nats_url="nats://localhost:4222",
        nats_subject_namespace="3tears",
        client_name="gateway-svc",
        user="gateway",
        password="s3cret",
        verify_jetstream=False,
    )

    options = captured["options"]
    assert options["user"] == "gateway"
    assert options["password"] == "s3cret"


@pytest.mark.asyncio
async def test_connect_omits_credential_options_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """anonymous connect (the legacy shared bus) sets none of the credential/inbox options."""
    import threetears.nats.client as client_module

    captured: dict[str, Any] = {}

    async def _fake_establish(servers: list[str], options: dict[str, Any], nats_url: str) -> Any:
        captured["options"] = options
        return MagicMock()

    monkeypatch.setattr(client_module, "_establish_connection", _fake_establish)

    await NatsClient.connect(
        nats_url="nats://localhost:4222",
        client_name="agent-x",
        verify_jetstream=False,
    )

    options = captured["options"]
    assert "token" not in options
    assert "user_credentials" not in options
    assert "user" not in options
    assert "password" not in options
    assert "inbox_prefix" not in options


@pytest.mark.asyncio
async def test_connect_uses_forever_reconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """runtime reconnect is unbounded: the options reaching nats-py carry
    ``max_reconnect_attempts=-1`` (forever) paced by a bounded per-attempt wait.

    this is the k8s-resilience contract -- a NATS outage of ANY duration must be
    ridden out and auto-recovered with no human action, never give up after a
    finite ceiling. the previous value (100, ~200s of budget) closed the client
    for good on any longer outage, bricking every service with no self-heal.
    """
    import threetears.nats.client as client_module

    # the contract the constant encodes is forever, not a finite ceiling.
    assert client_module.RUNTIME_MAX_RECONNECT_ATTEMPTS == -1

    captured: dict[str, Any] = {}

    async def _fake_establish(servers: list[str], options: dict[str, Any], nats_url: str) -> Any:
        captured["options"] = options
        return MagicMock()

    monkeypatch.setattr(client_module, "_establish_connection", _fake_establish)

    await NatsClient.connect(
        nats_url="nats://localhost:4222",
        client_name="agent-x",
        verify_jetstream=False,
    )

    options = captured["options"]
    # forever-reconnect: the value that actually reaches nats-py (``< 0`` ==
    # unbounded; nats-py never empties the server pool, so it never raises
    # NoServersError and never permanently closes the client on a reconnect).
    assert options["max_reconnect_attempts"] == -1
    assert options["allow_reconnect"] is True
    # each attempt is paced so forever-retry is a bounded loop, not a hot spin.
    assert options["reconnect_time_wait"] == 2


@pytest.mark.asyncio
async def test_dispatch_loop_survives_message_gap_across_reconnect() -> None:
    """the dispatch loop keeps delivering after a quiet gap (a reconnect window).

    the audit's failure mode: after a long outage the old finite-ceiling client
    CLOSED, which ended ``raw_sub.messages`` so the dispatch loop exited silently
    -- messages that resumed post-recovery were never delivered. with
    forever-reconnect the client never closes and nats-py replays the
    subscription on the SAME object, so the loop reads the same queue across the
    gap. here the fake's message stream goes quiet then resumes (no
    close/sentinel); the loop must still deliver the post-gap message and stay
    alive.
    """
    client, fake = _make_client()
    received: list[_Hello] = []

    async def on_msg(msg: _Hello) -> None:
        received.append(msg)

    sub = await client.subscribe_typed(
        subject=Subjects.tools_call(),
        cb=on_msg,
        message_type=_Hello,
    )
    fake_sub = fake.subscribed[0][2]

    # pre-outage message delivered.
    before = _Hello(greeting="before", count=1)
    await fake_sub.queue.put(_FakeMsg(data=before.model_dump_json().encode("utf-8")))
    await asyncio.sleep(0.05)
    assert received == [before]

    # a quiet gap stands in for the disconnect/reconnect window: no message and
    # NO close/sentinel. the loop must NOT end.
    await asyncio.sleep(0.05)
    assert sub.is_closed is False
    assert sub.dispatch_task.done() is False

    # post-recovery message: nats-py replayed the SUB onto the same queue, so the
    # SAME dispatch loop delivers it.
    after = _Hello(greeting="after", count=2)
    await fake_sub.queue.put(_FakeMsg(data=after.model_dump_json().encode("utf-8")))
    await asyncio.sleep(0.05)
    assert received == [before, after]
    assert sub.dispatch_task.done() is False

    await client.unsubscribe(sub)


@pytest.mark.asyncio
async def test_reconnect_callbacks_fan_out_on_reconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """callbacks registered via ``add_reconnect_callback`` fire in order on reconnect.

    forever-reconnect rides out an outage and replays subscriptions, but the BROKER
    may have dropped state backing a short-lived credential meanwhile; a consumer
    hook re-establishes it once the connection returns. this pins the shared-list
    bridge: nats-py has a single ``reconnected_cb`` slot wired at connect time, and
    the instance adopts the SAME list the dispatcher closes over, so post-connect
    registrations are visible to the already-installed slot.
    """
    import threetears.nats.client as client_module

    captured: dict[str, Any] = {}

    async def _fake_establish(servers: list[str], options: dict[str, Any], nats_url: str) -> Any:
        captured["options"] = options
        return MagicMock()

    monkeypatch.setattr(client_module, "_establish_connection", _fake_establish)

    client = await NatsClient.connect(
        nats_url="nats://localhost:4222",
        client_name="agent-x",
        verify_jetstream=False,
    )

    order: list[str] = []

    async def _first() -> None:
        order.append("first")

    async def _second() -> None:
        order.append("second")

    client.add_reconnect_callback(_first)
    client.add_reconnect_callback(_second)

    # invoke the exact slot nats-py would call on a reconnect.
    await captured["options"]["reconnected_cb"]()

    assert order == ["first", "second"]


@pytest.mark.asyncio
async def test_reconnect_callback_failure_is_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """one reconnect hook raising is logged and does not abort the others.

    a re-mint hook that fails must not break the callback chain or the nats-py
    reconnect path -- otherwise a single bad consumer would re-introduce the wedge
    forever-reconnect exists to prevent.
    """
    import threetears.nats.client as client_module

    captured: dict[str, Any] = {}

    async def _fake_establish(servers: list[str], options: dict[str, Any], nats_url: str) -> Any:
        captured["options"] = options
        return MagicMock()

    monkeypatch.setattr(client_module, "_establish_connection", _fake_establish)

    client = await NatsClient.connect(
        nats_url="nats://localhost:4222",
        client_name="agent-x",
        verify_jetstream=False,
    )

    ran: list[str] = []

    async def _boom() -> None:
        ran.append("boom")
        raise RuntimeError("re-mint failed")

    async def _after() -> None:
        ran.append("after")

    client.add_reconnect_callback(_boom)
    client.add_reconnect_callback(_after)

    # must NOT raise: the dispatcher logs the failure and continues to the next hook.
    await captured["options"]["reconnected_cb"]()

    assert ran == ["boom", "after"]


# ---------------------------------------------------------------------------
# reconnect() — proactive force-reconnect primitive (NATS-JWT re-auth)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconnect_triggers_nats_py_reconnect_when_connected() -> None:
    """on a CONNECTED client, reconnect() drives nats-py's own op-error reconnect path.

    this is the proactive NATS-JWT re-auth primitive: forcing a reconnect while the current
    short-lived user JWT is still valid makes nats-py re-run the auth-callout and mint a FRESH
    JWT, so the connection never reaches the auth-expiry close (which nats-py routes straight to
    a terminal ``_close``, bypassing forever-reconnect). the synthesized error is a
    StaleConnectionError -- the connection is about to go stale as the JWT nears expiry.
    """
    from nats.errors import StaleConnectionError

    client, fake = _make_client()
    fake.is_connected = True
    fake.is_closed = False

    await client.reconnect()

    assert len(fake.op_err_calls) == 1
    assert isinstance(fake.op_err_calls[0], StaleConnectionError)


@pytest.mark.asyncio
async def test_reconnect_raises_on_closed_client() -> None:
    """reconnect() on a terminally-closed client raises -- it cannot revive a closed ``_raw``.

    once nats-py has run ``_close(CLOSED)`` (the auth-expiry terminal path), ``_attempt_reconnect``
    early-returns on ``is_closed``; a proactive trigger cannot help. the caller's reactive
    self-heal path (a fresh re-establish) owns that case. failing loud keeps the two paths distinct
    and never silently lets ``_process_op_err`` fall into its connection-closing else-branch.
    """
    from threetears.nats import NatsClientError

    client, fake = _make_client()
    fake.is_closed = True

    with pytest.raises(NatsClientError):
        await client.reconnect()
    assert fake.op_err_calls == []


@pytest.mark.asyncio
async def test_reconnect_noops_when_not_currently_connected() -> None:
    """reconnect() is a no-op when not connected (but not closed) -- a reconnect is already in flight.

    forever-reconnect drives an in-flight reconnect after a network drop. forcing another through
    ``_process_op_err`` while disconnected would hit its else-branch and ``_close`` the client; the
    ``is_connected`` guard keeps reconnect() out of that branch entirely.
    """
    client, fake = _make_client()
    fake.is_connected = False
    fake.is_closed = False

    await client.reconnect()

    assert fake.op_err_calls == []


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
        subjects=["3tears.channels.deliver.*"],
        storage="file",
    )

    assert full == "3tears-channels-deliver"
    js.add_stream.assert_awaited_once()
    js.update_stream.assert_not_awaited()
    config = js.add_stream.await_args.args[0]
    assert config.name == "3tears-channels-deliver"
    assert config.subjects == ["3tears.channels.deliver.*"]
    assert config.storage == StorageType.FILE


@pytest.mark.asyncio
async def test_ensure_jetstream_stream_memory_storage() -> None:
    """storage='memory' selects the MEMORY storage type."""
    from nats.js.api import StorageType

    client, js = _client_with_js()
    await client.ensure_jetstream_stream(
        name="ephemeral",
        subjects=["3tears.x.*"],
        storage="memory",
    )
    config = js.add_stream.await_args.args[0]
    assert config.storage == StorageType.MEMORY


@pytest.mark.asyncio
async def test_ensure_jetstream_stream_defaults_to_memory() -> None:
    """NATS is the L2 tier in 3tears: with no storage arg the stream is MEMORY, never FILE."""
    from nats.js.api import StorageType

    client, js = _client_with_js()
    await client.ensure_jetstream_stream(name="default", subjects=["3tears.x.*"])
    config = js.add_stream.await_args.args[0]
    assert config.storage == StorageType.MEMORY


@pytest.mark.asyncio
async def test_ensure_jetstream_stream_reconciles_existing() -> None:
    """when the stream already exists, fall back to update_stream (reconcile)."""
    client, js = _client_with_js()
    js.add_stream.side_effect = RuntimeError("stream name already in use")

    full = await client.ensure_jetstream_stream(
        name="channels-deliver",
        subjects=["3tears.channels.deliver.*", "3tears.channels.deliver.slack"],
        storage="file",
    )

    assert full == "3tears-channels-deliver"
    js.update_stream.assert_awaited_once()
    config = js.update_stream.await_args.args[0]
    assert config.subjects == [
        "3tears.channels.deliver.*",
        "3tears.channels.deliver.slack",
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
            subjects=["3tears.x.*"],
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
        stream="3tears-channels-deliver",
    )

    assert handle == "sub-handle"
    kwargs = js.subscribe.await_args.kwargs
    assert js.subscribe.await_args.args[0] == subject.path
    assert kwargs["durable"] == "slack-delivery"
    assert kwargs["manual_ack"] is True
    assert kwargs["stream"] == "3tears-channels-deliver"
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


# ---------------------------------------------------------------------------
# reconnect callback fan-out (consumer-registered post-reconnect hooks)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_reconnect_callback_runs_on_reconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """a callback registered via ``add_reconnect_callback`` fires when nats-py reconnects.

    nats-py exposes a single ``reconnected_cb`` slot; the wrapper installs a dispatcher there at
    connect time and fans each reconnect out to consumer-registered callbacks. this is the seam the
    agent SDK uses to re-handshake (re-mint its short-lived identity token + re-establish the Hub
    session) after an outage, so tool calls resume immediately instead of failing until the next
    scheduled refresh.
    """
    import threetears.nats.client as client_module

    captured: dict[str, Any] = {}

    async def _fake_establish(servers: list[str], options: dict[str, Any], nats_url: str) -> Any:
        captured["options"] = options
        return MagicMock()

    monkeypatch.setattr(client_module, "_establish_connection", _fake_establish)

    client = await NatsClient.connect(
        nats_url="nats://localhost:4222",
        client_name="agent-x",
        verify_jetstream=False,
    )

    hook = AsyncMock()
    client.add_reconnect_callback(hook)

    # the dispatcher nats-py would invoke on reconnect is whatever reached the connect options.
    dispatcher = captured["options"]["reconnected_cb"]
    await dispatcher()

    hook.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_reconnect_dispatcher_isolates_failing_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """a callback that raises is logged but does not abort the others or the reconnect path."""
    import threetears.nats.client as client_module

    captured: dict[str, Any] = {}

    async def _fake_establish(servers: list[str], options: dict[str, Any], nats_url: str) -> Any:
        captured["options"] = options
        return MagicMock()

    monkeypatch.setattr(client_module, "_establish_connection", _fake_establish)

    client = await NatsClient.connect(
        nats_url="nats://localhost:4222",
        client_name="agent-x",
        verify_jetstream=False,
    )

    boom = AsyncMock(side_effect=RuntimeError("hook blew up"))
    after = AsyncMock()
    client.add_reconnect_callback(boom)
    client.add_reconnect_callback(after)

    dispatcher = captured["options"]["reconnected_cb"]
    # must not raise even though the first hook does.
    await dispatcher()

    boom.assert_awaited_once_with()
    after.assert_awaited_once_with()


def test_is_authorization_violation_detects_the_server_rejection() -> None:
    """the wedged-auth signal is matched on the server's -ERR text, robust across nats-py versions."""
    from threetears.nats.client import _is_authorization_violation

    assert _is_authorization_violation(Exception("nats: 'Authorization Violation'"))
    assert _is_authorization_violation(Exception("AUTHORIZATION VIOLATION"))
    assert not _is_authorization_violation(OSError("connection reset by peer"))


def test_is_authorization_violation_detects_typed_authorization_error() -> None:
    """nats-py's typed AuthorizationError (str: 'nats: authorization failed') must also trip the signal.

    Its message shares no substring with 'authorization violation', so the reconnect-loop -ERR match
    alone would miss it — detecting it by type (and its own text) keeps the wedge signal alive if a
    future nats-py routes the typed error to error_cb instead of the generic -ERR string."""
    from nats.errors import AuthorizationError

    from threetears.nats.client import _is_authorization_violation

    assert _is_authorization_violation(AuthorizationError())
    assert _is_authorization_violation(Exception("nats: authorization failed"))


@pytest.mark.asyncio
async def test_is_healthy_trips_on_persistent_auth_violation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """a persistent Authorization-Violation reconnect loop flips is_healthy False (so a /healthz keyed
    on it trips and k8s restarts the pod); a successful reconnect clears it; a network drop never does."""
    import threetears.nats.client as client_module

    captured: dict[str, Any] = {}

    async def _fake_establish(servers: list[str], options: dict[str, Any], nats_url: str) -> Any:
        captured["options"] = options
        return MagicMock()

    monkeypatch.setattr(client_module, "_establish_connection", _fake_establish)

    client = await NatsClient.connect(
        nats_url="nats://localhost:4222",
        nats_subject_namespace="3tears",
        client_name="agent-x",
        verify_jetstream=False,
    )
    error_cb = captured["options"]["error_cb"]
    reconnected_cb = captured["options"]["reconnected_cb"]

    assert client.is_healthy  # a fresh connection is healthy

    # a network drop (non-auth error) never trips health, no matter how many land.
    for _ in range(5):
        await error_cb(OSError("connection reset"))
    assert client.is_healthy

    auth_violation = Exception("nats: 'Authorization Violation'")
    # below the threshold (3): still healthy.
    await error_cb(auth_violation)
    await error_cb(auth_violation)
    assert client.is_healthy

    # crossing the threshold: unhealthy -- the connection is stuck being auth-rejected.
    await error_cb(auth_violation)
    assert not client.is_healthy

    # a successful reconnect clears the wedged-auth signal.
    await reconnected_cb()
    assert client.is_healthy
