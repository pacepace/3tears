"""unit tests for :meth:`threetears.nats.NatsClient.renew_connection`, the make-before-break handover.

A credential renewal opens a successor connection, moves every subscription onto it, makes it
current, and keeps the replaced connection open for the work it carries. These tests pin the
handover's order and its failure paths against scripted stand-ins for nats-py connections; the
live proof against a real nats-server and auth-callout is
``tests/integration/test_credential_renewal_live.py``.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

import pytest

import threetears.nats.client as client_module
from threetears.nats import (
    IncomingMessage,
    NatsClient,
    NatsClientError,
    Subject,
    Subscription,
    set_default_namespace,
)
from threetears.nats.credential_refusal import CredentialRefusal, CredentialRefusalReason

pytestmark = pytest.mark.asyncio


# parity-exempt: stands in for nats-py's Msg on a subscription; the wrapper reads data, reply and subject only
class _Msg:
    def __init__(self, data: bytes, *, reply: str = "", subject: str = "events.x") -> None:
        self.data = data
        self.reply = reply
        self.subject = subject


# parity-exempt: stands in for a nats-py core Subscription; the wrapper iterates messages, drains and unsubscribes
class _Sub:
    def __init__(self, subject: str, queue: str, sid: int, calls: list[str]) -> None:
        self.subject = subject
        self.queue = queue
        self._sid = sid
        self._calls = calls
        self.queue_in: asyncio.Queue[_Msg | None] = asyncio.Queue()
        self.drained = False
        self.unsubscribed = False

    @property
    def _id(self) -> int:
        # nats-py's name for the subscription id, which the handover's UNSUB addresses
        return self._sid

    @property
    def messages(self) -> Any:
        async def _gen() -> Any:
            while True:
                msg = await self.queue_in.get()
                if msg is None:
                    return
                yield msg

        return _gen()

    async def drain(self) -> None:
        # nats-py hands over what is queued, then ends the iterator
        self._calls.append(f"drain {self._id}")
        self.drained = True
        await self.queue_in.put(None)

    async def unsubscribe(self) -> None:
        self.unsubscribed = True
        await self.queue_in.put(None)


# parity-exempt: stands in for nats-py's transport; a round trip hands it the pending buffer and a PING
class _Wire:
    def __init__(self, conn: _Conn) -> None:
        self._conn = conn

    def writelines(self, payload: list[bytes]) -> None:
        self._conn.calls.extend(chunk.decode().strip() for chunk in payload)

    def write(self, payload: bytes) -> None:
        # the round trip's PING, answered at once -- or once the test opens the pong gate
        self._conn.calls.append("round-trip")
        future = self._conn.pongs.pop(0)
        if self._conn.pong_gate is None:
            future.set_result(True)
            return

        async def _pong_later(gate: asyncio.Event) -> None:
            await gate.wait()
            future.set_result(True)

        self._conn.late_pongs.append(asyncio.create_task(_pong_later(self._conn.pong_gate)))


# parity-exempt: stands in for a nats-py connection; the handover calls subscribe, flush, publish, drain and close
class _Conn:
    def __init__(self, name: str, *, subscribe_fails: bool = False) -> None:
        self.name = name
        self.is_closed = False
        self.is_connected = True
        self.subs: list[_Sub] = []
        self.published: list[tuple[str, bytes]] = []
        self.drained = False
        self._subscribe_fails = subscribe_fails
        # the protocol the handover writes, in order
        self.calls: list[str] = []
        # set by a test to hold every PONG until it opens the gate
        self.pong_gate: asyncio.Event | None = None
        self.late_pongs: list[asyncio.Task[None]] = []
        # set by a test to hold a subscribe until it opens the gate
        self.subscribe_gate: asyncio.Event | None = None
        # the fields of nats-py's own a round trip reads and writes
        self._pongs: list[asyncio.Future[bool]] = []
        self._pending: list[bytes] = []
        self._pending_data_size = 0
        self._transport = _Wire(self)
        self._flush_queue: asyncio.Queue[asyncio.Future[None]] = asyncio.Queue()

    @property
    def pongs(self) -> list[asyncio.Future[bool]]:
        return self._pongs

    async def subscribe(self, subject: str, queue: str = "") -> _Sub:
        if self.subscribe_gate is not None:
            await self.subscribe_gate.wait()
        if self._subscribe_fails:
            raise RuntimeError(f"{self.name}: subscribe refused")
        sub = _Sub(subject, queue, len(self.subs) + 1, self.calls)
        self.subs.append(sub)
        return sub

    async def _send_unsubscribe(self, sid: int, limit: int = 0) -> None:
        self.calls.append(f"unsub {sid}")

    async def publish(self, subject: str, payload: bytes, reply: str = "", headers: Any = None) -> None:
        self.published.append((subject, payload))

    async def drain(self) -> None:
        self.drained = True
        self.is_closed = True

    async def close(self) -> None:
        self.is_closed = True

    def jetstream(self) -> Any:
        return object()


@pytest.fixture(autouse=True)
def _namespace() -> None:
    set_default_namespace("3tears")


async def _connected(monkeypatch: pytest.MonkeyPatch, *connections: _Conn | BaseException) -> NatsClient:
    """a client opened through :meth:`NatsClient.connect` whose connections come from ``connections``.

    :param monkeypatch: pytest's patcher
    :ptype monkeypatch: pytest.MonkeyPatch
    :param connections: each connection the client opens, in order; an exception is raised instead
    :ptype connections: _Conn | BaseException
    :return: the connected client
    :rtype: NatsClient
    """
    queue = list(connections)

    async def _establish(servers: list[str], options: dict[str, Any], url: str) -> Any:
        step = queue.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step

    monkeypatch.setattr(client_module, "_establish_connection", _establish)
    return await NatsClient.connect(
        nats_url="nats://localhost:4222",
        nats_subject_namespace="3tears",
        client_name="handover-test",
        verify_jetstream=False,
    )


async def _settle() -> None:
    """let the handover's background tasks run.

    :return: nothing
    :rtype: None
    """
    for _ in range(5):
        await asyncio.sleep(0)


async def test_a_subscription_without_a_group_holds_one_of_its_own() -> None:
    """never plain on the wire: a group of which it is the only member, distinct per subscription."""
    conn = _Conn("only")
    client = NatsClient(raw=conn, namespace="3tears", client_name="t")  # type: ignore[arg-type]

    async def _cb(msg: IncomingMessage) -> None:
        return None

    first = await client.subscribe(Subject.raw("events.>"), cb=_cb)
    second = await client.subscribe(Subject.raw("events.>"), cb=_cb)
    named = await client.subscribe(Subject.raw("work"), cb=_cb, queue="workers")

    assert first.queue.startswith("_solo.") and second.queue.startswith("_solo.")
    assert first.queue != second.queue  # two subscriptions each still receive every message
    assert named.queue == "workers"
    assert [s.queue for s in conn.subs] == [first.queue, second.queue, "workers"]
    await client.unsubscribe(first)
    await client.unsubscribe(second)
    await client.unsubscribe(named)


async def test_a_client_that_did_not_open_its_connection_cannot_renew_it() -> None:
    client = NatsClient(raw=_Conn("given"), namespace="3tears", client_name="t")  # type: ignore[arg-type]

    with pytest.raises(NatsClientError, match="did not open"):
        await client.renew_connection(retire_after=timedelta(seconds=1))


async def test_a_successor_that_cannot_open_changes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    current = _Conn("current")
    client = await _connected(monkeypatch, current, RuntimeError("callout denied"))

    with pytest.raises(RuntimeError, match="callout denied"):
        await client.renew_connection(retire_after=timedelta(seconds=1))

    assert client.raw is current
    assert not current.is_closed


async def test_a_successor_that_cannot_take_the_subscriptions_is_closed_and_changes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current, successor = _Conn("current"), _Conn("successor", subscribe_fails=True)
    client = await _connected(monkeypatch, current, successor)
    received: list[bytes] = []

    async def _cb(msg: IncomingMessage) -> None:
        received.append(bytes(msg.data))

    sub = await client.subscribe(Subject.raw("events.>"), cb=_cb)

    with pytest.raises(RuntimeError, match="subscribe refused"):
        await client.renew_connection(retire_after=timedelta(seconds=1))

    assert client.raw is current
    assert successor.is_closed
    assert not current.subs[0].drained
    await current.subs[0].queue_in.put(_Msg(b"still here"))
    await _settle()
    assert received == [b"still here"]
    await client.unsubscribe(sub)


async def test_the_handover_moves_every_subscription_before_the_old_half_is_released(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """subscribe the successor in the same group, make it current, THEN drain the old half.

    a message the server routed to the old connection before its UNSUB still reaches the callback,
    and so does one routed to the successor: both connections feed the one subscription.
    """
    current, successor = _Conn("current"), _Conn("successor")
    client = await _connected(monkeypatch, current, successor)
    received: list[bytes] = []

    async def _cb(msg: IncomingMessage) -> None:
        received.append(bytes(msg.data))

    sub = await client.subscribe(Subject.raw("events.>"), cb=_cb)
    # subscribe's own round trip: the server has the SUB before subscribe returns
    assert current.calls == ["round-trip"]
    current.calls.clear()
    old_half = current.subs[0]
    await old_half.queue_in.put(_Msg(b"routed to the old connection"))

    await client.renew_connection(retire_after=timedelta(seconds=30))
    await _settle()

    assert client.raw is successor
    assert [(s.subject, s.queue) for s in successor.subs] == [("events.>", sub.queue)]
    assert old_half.drained
    # a round trip settles what was published on the old connection before anything is published
    # on the successor; then the UNSUB on its own, then a round trip ordered after it, and only then
    # nats-py's drain, which forgets the subscription.
    assert current.calls == ["round-trip", "unsub 1", "round-trip", "drain 1"]
    await successor.subs[0].queue_in.put(_Msg(b"routed to the successor"))
    await _settle()
    assert received == [b"routed to the old connection", b"routed to the successor"]
    # the old connection is held for the work it carries, not closed with the handover
    assert not current.is_closed
    await client.shutdown()


async def test_the_replaced_connection_is_drained_after_the_hold(monkeypatch: pytest.MonkeyPatch) -> None:
    current, successor = _Conn("current"), _Conn("successor")
    client = await _connected(monkeypatch, current, successor)

    await client.renew_connection(retire_after=timedelta(seconds=0.05))
    for _ in range(100):
        if current.drained:
            break
        await asyncio.sleep(0.01)

    assert current.drained
    assert not successor.is_closed
    await client.shutdown()


async def test_shutdown_closes_a_connection_still_being_held(monkeypatch: pytest.MonkeyPatch) -> None:
    current, successor = _Conn("current"), _Conn("successor")
    client = await _connected(monkeypatch, current, successor)
    await client.renew_connection(retire_after=timedelta(seconds=3600))

    await client.shutdown()

    assert current.is_closed
    assert successor.is_closed


async def test_a_reply_owed_across_a_renewal_leaves_on_the_connection_that_received_the_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NATS lets only the receiving connection answer; the successor's publish would be refused."""
    monkeypatch.setattr(client_module, "seconds_until_reauth", lambda _ttl, **_kw: 3600.0)
    current, successor = _Conn("current"), _Conn("successor")
    client = await _connected(monkeypatch, current, successor)
    client.renew_credential(ttl_seconds=lambda: 300, longest_request_seconds=30.0)
    owed: list[IncomingMessage] = []

    async def _cb(msg: IncomingMessage) -> None:
        owed.append(msg)

    await client.subscribe(Subject.raw("calls"), cb=_cb)
    await current.subs[0].queue_in.put(_Msg(b"call-1", reply="_INBOX.requester.1", subject="calls"))
    await _settle()

    await client.renew_connection(retire_after=timedelta(seconds=30))
    await _settle()
    await successor.subs[0].queue_in.put(_Msg(b"call-2", reply="_INBOX.requester.2", subject="calls"))
    await _settle()

    for msg in owed:
        assert msg.reply_subject is not None
        await client.publish_raw_reply(reply_subject=msg.reply_subject, payload=b"done")

    assert current.published == [("_INBOX.requester.1", b"done")]
    assert successor.published == [("_INBOX.requester.2", b"done")]
    await client.shutdown()


async def test_without_a_renewal_a_reply_simply_uses_the_current_connection() -> None:
    """a client whose credential never expires records nothing: it only ever has one connection."""
    conn = _Conn("only")
    client = NatsClient(raw=conn, namespace="3tears", client_name="t")  # type: ignore[arg-type]
    owed: list[IncomingMessage] = []

    async def _cb(msg: IncomingMessage) -> None:
        owed.append(msg)

    sub = await client.subscribe(Subject.raw("calls"), cb=_cb)
    await conn.subs[0].queue_in.put(_Msg(b"call", reply="_INBOX.requester.1", subject="calls"))
    await _settle()
    assert owed[0].reply_subject is not None
    await client.publish_raw_reply(reply_subject=owed[0].reply_subject, payload=b"done")

    assert conn.published == [("_INBOX.requester.1", b"done")]
    await client.unsubscribe(sub)


async def test_a_subscription_dropped_during_the_handover_is_not_revived(monkeypatch: pytest.MonkeyPatch) -> None:
    current, successor = _Conn("current"), _Conn("successor")
    client = await _connected(monkeypatch, current, successor)

    async def _cb(msg: IncomingMessage) -> None:
        return None

    sub = await client.subscribe(Subject.raw("events.>"), cb=_cb)
    original_subscribe_on = Subscription.subscribe_on

    async def _dropped_meanwhile(self: Subscription, connection: Any) -> Any:
        raw_sub = await original_subscribe_on(self, connection)
        await client.unsubscribe(self)
        return raw_sub

    monkeypatch.setattr(Subscription, "subscribe_on", _dropped_meanwhile)
    await client.renew_connection(retire_after=timedelta(seconds=30))
    await _settle()

    assert successor.subs[0].unsubscribed
    assert sub.raw_subscription is current.subs[0]
    await client.shutdown()


# parity-exempt: stands in for a nats-py JetStream pull subscription; the consumer calls fetch and unsubscribe
class _PullSub:
    def __init__(self, name: str) -> None:
        self.name = name
        self.fetches = 0
        self.unsubscribed = False

    async def fetch(self, batch: int, timeout: float) -> list[Any]:
        self.fetches += 1
        return []

    async def unsubscribe(self) -> None:
        self.unsubscribed = True


async def test_a_pull_consumer_follows_the_renewal_before_its_next_fetch() -> None:
    """a fetch is a request on the connection it was bound on, which the renewal retires."""
    from threetears.nats.client import JetStreamPullConsumer

    old_psub, new_psub = _PullSub("old"), _PullSub("new")
    connections = {"current": "A"}

    async def _resubscribe() -> Any:
        return new_psub

    async def _cb(msg: Any) -> None:
        return None

    async def _redeliver(msg: Any, exc: BaseException) -> None:
        return None

    consumer = JetStreamPullConsumer(
        psub=old_psub,
        cb=_cb,
        redeliver=_redeliver,
        durable="d",
        subject=Subject.raw("jobs"),
        batch=1,
        fetch_timeout_seconds=0.01,
        bound_to="A",
        current_connection=lambda: connections["current"],
        resubscribe=_resubscribe,
    )
    await consumer.fetch_and_process()
    connections["current"] = "B"  # the renewal
    await consumer.fetch_and_process()

    assert old_psub.fetches == 1
    assert old_psub.unsubscribed
    assert new_psub.fetches == 1


# parity-exempt: stands in for a nats-py JetStream push subscription; the move drains it and stop unsubscribes it
class _PushSub:
    def __init__(self) -> None:
        self.drained = False
        self.unsubscribed = False

    @property
    def _id(self) -> int:
        # nats-py's name for the subscription id, which the handover's UNSUB addresses
        return 7

    async def drain(self) -> None:
        self.drained = True

    async def unsubscribe(self) -> None:
        self.unsubscribed = True


async def test_a_push_consumer_is_released_then_bound_again_on_the_successor() -> None:
    """released FIRST: a push durable with no deliver group admits one bound subscription at a time."""
    from threetears.nats.client import JetStreamPushConsumer

    old, new = _PushSub(), _PushSub()
    order: list[str] = []

    async def _bind(js: Any) -> Any:
        order.append(f"bind:{js}:{old.drained}")
        return new

    old_connection = _Conn("current")
    consumer = JetStreamPushConsumer(
        raw_subscription=old,
        subject=Subject.raw("jobs"),
        durable="d",
        resubscribe=_bind,
        connection=old_connection,
    )
    await consumer.move_to("successor-js", _Conn("successor"))

    assert order == ["bind:successor-js:True"]
    assert consumer.raw_subscription is new
    assert old_connection.calls == ["unsub 7", "round-trip"]


async def test_a_push_consumer_stopped_during_the_move_is_not_bound_again() -> None:
    from threetears.nats.client import JetStreamPushConsumer

    old, new = _PushSub(), _PushSub()
    consumer: JetStreamPushConsumer

    async def _bind(js: Any) -> Any:
        await consumer.stop()
        return new

    consumer = JetStreamPushConsumer(
        raw_subscription=old,
        subject=Subject.raw("jobs"),
        durable="d",
        resubscribe=_bind,
        connection=_Conn("current"),
    )
    await consumer.move_to("successor-js", _Conn("successor"))

    assert new.unsubscribed
    assert consumer.is_closed


async def test_nothing_is_published_on_the_successor_before_the_old_connection_settles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """a publisher's A on the old connection must reach the server before its B on the successor.

    they travel on two sockets, so without the settle B can be routed first -- a streamed answer's
    tokens arriving out of order.
    """
    current, successor = _Conn("current"), _Conn("successor")
    client = await _connected(monkeypatch, current, successor)
    settle = asyncio.Event()
    current.pong_gate = settle
    await client.publish_raw(subject=Subject.raw("tokens"), payload=b"A")
    renewal = asyncio.create_task(client.renew_connection(retire_after=timedelta(seconds=30)))
    await _settle()
    publish_b = asyncio.create_task(client.publish_raw(subject=Subject.raw("tokens"), payload=b"B"))
    await _settle()

    assert successor.published == []  # B waits while the old connection is being settled
    settle.set()
    await renewal
    await publish_b

    assert current.published == [("tokens", b"A")]
    assert successor.published == [("tokens", b"B")]
    await client.shutdown()


async def test_refused_renewals_do_not_count_against_the_connection_still_in_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """a refused renewal leaves the current connection valid, and it is kept until it expires (Q16).

    only a DELIBERATE refusal the auth-callout names this runner in stops it
    (:meth:`NatsClient.abandon_on_refusal`); a refusal that says nothing -- a callout that was down,
    or slow -- must not count the pod unhealthy and have its supervisor kill a working connection.
    """
    current, successor = _Conn("current"), _Conn("successor")
    # the first open is the client's own connect; then three refused renewals and one admitted
    outcomes = iter([current, None, None, None, successor])

    async def _establish(servers: list[str], options: dict[str, Any], url: str) -> Any:
        opened = next(outcomes)
        if opened is None:
            await options["error_cb"](RuntimeError("nats: 'Authorization Violation'"))
            raise RuntimeError("renewal refused")
        return opened

    monkeypatch.setattr(client_module, "_establish_connection", _establish)
    client = await NatsClient.connect(
        nats_url="nats://localhost:4222",
        nats_subject_namespace="3tears",
        client_name="handover-test",
        verify_jetstream=False,
    )
    for _ in range(3):
        with pytest.raises(RuntimeError, match="refused"):
            await client.renew_connection(retire_after=timedelta(seconds=30))

    assert client.is_healthy is True
    assert client.raw is current
    await client.renew_connection(retire_after=timedelta(seconds=30))
    assert client.raw is successor
    await client.shutdown()


async def test_an_abandoned_client_closes_every_connection_and_never_renews(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current, successor = _Conn("current"), _Conn("successor")
    client = await _connected(monkeypatch, current, successor)
    await client.renew_connection(retire_after=timedelta(seconds=3600))

    await client.abandon(reason="credential refused: superseded")

    assert current.is_closed and successor.is_closed
    assert client.is_closed
    with pytest.raises(NatsClientError):
        await client.renew_connection(retire_after=timedelta(seconds=30))


async def test_a_renewal_cancelled_while_it_waits_for_the_handover_lock_closes_its_successor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """the successor is owned from the moment it opens, so no cancellation can orphan it.

    a subscribe holds the handover lock across the server's SUB; a renewal whose successor is open
    waits for that lock. cancelled there, the successor used to escape every sweep -- abandon,
    shutdown and retirement all walk the client's registry -- and forever-reconnect on its own.
    """
    current, successor = _Conn("current"), _Conn("successor")
    client = await _connected(monkeypatch, current, successor)

    async def _cb(msg: IncomingMessage) -> None:
        return None

    current.subscribe_gate = asyncio.Event()
    subscribing = asyncio.create_task(client.subscribe(Subject.raw("late"), cb=_cb))
    await _settle()
    renewal = asyncio.create_task(client.renew_connection(retire_after=timedelta(seconds=30)))
    await _settle()
    assert not successor.is_closed  # opened, and waiting for the lock

    renewal.cancel()
    with pytest.raises(asyncio.CancelledError):
        await renewal

    assert successor.is_closed
    assert client.raw is current
    current.subscribe_gate.set()
    await client.unsubscribe(await subscribing)
    await client.shutdown()


async def test_shutdown_during_a_renewal_waiting_for_the_handover_lock_closes_its_successor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """shutdown cancels the renewal loop; the successor that loop had opened is closed with it."""
    monkeypatch.setattr(client_module, "seconds_until_reauth", lambda _ttl, **_kw: 0.0)
    monkeypatch.setattr(client_module, "REAUTH_MIN_SLEEP_SECONDS", 0.0)
    current, successor = _Conn("current"), _Conn("successor")
    client = await _connected(monkeypatch, current, successor)

    async def _cb(msg: IncomingMessage) -> None:
        return None

    current.subscribe_gate = asyncio.Event()
    subscribing = asyncio.create_task(client.subscribe(Subject.raw("late"), cb=_cb))
    await _settle()
    client.renew_credential(ttl_seconds=lambda: 300, longest_request_seconds=30.0)
    await _settle()
    assert not successor.is_closed  # the loop opened it, and waits for the lock

    await client.shutdown()

    assert successor.is_closed
    assert current.is_closed
    current.subscribe_gate.set()
    await subscribing


async def test_a_client_abandoned_during_a_direct_renewal_stays_abandoned(monkeypatch: pytest.MonkeyPatch) -> None:
    """abandon lands while a renewal it did not start is subscribing its successor.

    the successor was not yet registered, so abandon's sweep missed it, and the handover then made
    it current: an abandoned client with an open connection that reports itself open, which no
    supervisor restarts -- the zombie a deliberate refusal exists to stop.
    """
    current, successor = _Conn("current"), _Conn("successor")
    client = await _connected(monkeypatch, current, successor)
    successor.pong_gate = asyncio.Event()
    renewal = asyncio.create_task(client.renew_connection(retire_after=timedelta(seconds=30)))
    await _settle()  # the successor's round trip waits for its PONG

    await client.abandon(reason="credential refused: superseded")
    successor.pong_gate.set()

    with pytest.raises(NatsClientError, match="abandoned"):
        await renewal
    assert successor.is_closed
    assert current.is_closed
    assert client.is_closed


async def test_a_refusal_abandons_only_the_runner_it_names_and_only_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """a principal's inbox is shared by every runner of it; only a refusal naming THIS one counts.

    a foreign generation is another runner's refusal and changes nothing. this runner's own refusal
    closes every connection at once, and a duplicate delivery of it starts no second abandonment.
    """
    current, successor = _Conn("current"), _Conn("successor")
    client = await _connected(monkeypatch, current, successor)
    await client.renew_connection(retire_after=timedelta(seconds=3600))  # two connections held
    abandonments: list[str] = []
    original_abandon = NatsClient.abandon

    async def _counting_abandon(self: NatsClient, *, reason: str) -> None:
        abandonments.append(reason)
        await original_abandon(self, reason=reason)

    monkeypatch.setattr(NatsClient, "abandon", _counting_abandon)

    def _is_mine(refusal: CredentialRefusal) -> bool:
        return refusal.pod_id == "pod-7" and refusal.identity_generation == "g-3"

    await client.abandon_on_refusal(inbox_prefix="_INBOX_agent_pod_a1", is_mine=_is_mine)
    refusals = successor.subs[-1]
    assert refusals.subject == "_INBOX_agent_pod_a1.credential-refused"

    def _refusal(generation: str) -> _Msg:
        body = CredentialRefusal(
            reason=CredentialRefusalReason.SUPERSEDED, pod_id="pod-7", identity_generation=generation
        )
        return _Msg(body.model_dump_json().encode(), subject=refusals.subject)

    await refusals.queue_in.put(_refusal("g-2"))
    await _settle()
    assert abandonments == []
    assert not client.is_closed and not current.is_closed and not successor.is_closed

    await refusals.queue_in.put(_refusal("g-3"))
    await refusals.queue_in.put(_refusal("g-3"))
    for _ in range(100):
        if client.is_closed:
            break
        await asyncio.sleep(0.01)
    await _settle()

    assert abandonments == ["credential refused: superseded"]
    assert current.is_closed and successor.is_closed and client.is_closed


async def test_one_renewal_at_a_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """a second renewal while one is handing over is refused and opens nothing."""
    current, successor = _Conn("current"), _Conn("successor")
    client = await _connected(monkeypatch, current, successor)
    successor.pong_gate = asyncio.Event()
    first = asyncio.create_task(client.renew_connection(retire_after=timedelta(seconds=30)))
    await _settle()

    with pytest.raises(NatsClientError, match="already in progress"):
        await client.renew_connection(retire_after=timedelta(seconds=30))

    successor.pong_gate.set()
    await first
    assert client.raw is successor
    await client.shutdown()


async def test_an_abandoned_or_shut_down_client_arms_no_renewal(monkeypatch: pytest.MonkeyPatch) -> None:
    abandoned = await _connected(monkeypatch, _Conn("abandoned"))
    await abandoned.abandon(reason="credential refused: superseded")
    shut_down = await _connected(monkeypatch, _Conn("shut-down"))
    await shut_down.shutdown()

    for client in (abandoned, shut_down):
        with pytest.raises(NatsClientError, match="cannot renew"):
            client.renew_credential(ttl_seconds=lambda: 300, longest_request_seconds=30.0)


async def test_every_reply_path_leaves_on_the_receiving_connection_and_forgets_its_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """a reply sent with the positional shorthand or publish_raw is still a reply.

    it used to leave on the CURRENT connection -- the one NATS refuses under allow_responses for a
    request received before the handover -- and its route lingered until retirement.
    """
    monkeypatch.setattr(client_module, "seconds_until_reauth", lambda _ttl, **_kw: 3600.0)
    current, successor = _Conn("current"), _Conn("successor")
    client = await _connected(monkeypatch, current, successor)
    client.renew_credential(ttl_seconds=lambda: 300, longest_request_seconds=30.0)
    owed: list[IncomingMessage] = []

    async def _cb(msg: IncomingMessage) -> None:
        owed.append(msg)

    await client.subscribe(Subject.raw("calls"), cb=_cb)
    await current.subs[0].queue_in.put(_Msg(b"call-1", reply="_INBOX.requester.1", subject="calls"))
    await current.subs[0].queue_in.put(_Msg(b"call-2", reply="_INBOX.requester.2", subject="calls"))
    await _settle()
    await client.renew_connection(retire_after=timedelta(seconds=30))
    await _settle()

    first, second = (msg.reply_subject for msg in owed)
    assert first is not None and second is not None
    await client.publish(first, b"done-1")
    await client.publish_raw(subject=Subject.raw(second), payload=b"done-2")
    # the routes are gone: the same subjects published again are ordinary publishes on the current one
    await client.publish(first, b"again")

    assert current.published == [("_INBOX.requester.1", b"done-1"), ("_INBOX.requester.2", b"done-2")]
    assert successor.published == [("_INBOX.requester.1", b"again")]
    await client.shutdown()


async def _connected_capturing(
    monkeypatch: pytest.MonkeyPatch, *connections: _Conn | BaseException
) -> tuple[NatsClient, dict[_Conn, dict[str, Any]]]:
    """like :func:`_connected`, also returning the nats-py options each connection was opened with.

    :param monkeypatch: pytest's patcher
    :ptype monkeypatch: pytest.MonkeyPatch
    :param connections: each connection the client opens, in order; an exception is raised instead
    :ptype connections: _Conn | BaseException
    :return: the connected client, and each opened connection's options
    :rtype: tuple[NatsClient, dict[_Conn, dict[str, Any]]]
    """
    queue = list(connections)
    options_of: dict[_Conn, dict[str, Any]] = {}

    async def _establish(servers: list[str], options: dict[str, Any], url: str) -> Any:
        step = queue.pop(0)
        if isinstance(step, BaseException):
            raise step
        options_of[step] = options
        return step

    monkeypatch.setattr(client_module, "_establish_connection", _establish)
    client = await NatsClient.connect(
        nats_url="nats://localhost:4222",
        nats_subject_namespace="3tears",
        client_name="lame-duck-test",
        verify_jetstream=False,
    )
    return client, options_of


async def _until(condition: Any, *, seconds: float = 2.0) -> None:
    """wait for ``condition()`` to hold, polling.

    :param condition: a zero-argument predicate
    :ptype condition: Any
    :param seconds: how long to wait
    :ptype seconds: float
    :return: nothing
    :rtype: None
    """
    for _ in range(int(seconds / 0.01)):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("the condition never held")


async def test_a_server_in_lame_duck_mode_moves_the_client_to_a_successor(monkeypatch: pytest.MonkeyPatch) -> None:
    """a rolling restart is a handover, not a reconnect: the old connection keeps its work."""
    current, successor = _Conn("current"), _Conn("successor")
    client, options_of = await _connected_capturing(monkeypatch, current, successor)
    received: list[bytes] = []

    async def _cb(msg: IncomingMessage) -> None:
        received.append(bytes(msg.data))

    await client.subscribe(Subject.raw("events.>"), cb=_cb)

    await options_of[current]["lame_duck_mode_cb"]()
    await _until(lambda: client.raw is successor)
    await _settle()

    assert [s.subject for s in successor.subs] == ["events.>"]
    assert not current.is_closed, "the connection leaving is held for its work, not dropped"
    await successor.subs[0].queue_in.put(_Msg(b"after the move"))
    await _settle()
    assert received == [b"after the move"]
    await client.shutdown()


async def test_a_move_that_cannot_open_a_successor_is_retried_until_it_lands(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client_module, "REAUTH_RETRY_SECONDS", 0.01)
    current, successor = _Conn("current"), _Conn("successor")
    client, options_of = await _connected_capturing(monkeypatch, current, RuntimeError("no server free"), successor)

    await options_of[current]["lame_duck_mode_cb"]()
    await _until(lambda: client.raw is successor)

    assert not current.is_closed
    await client.shutdown()


async def test_a_move_ends_once_the_server_closed_the_connection_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """once nats-py's own reconnect owns the connection there is nothing left to move."""
    monkeypatch.setattr(client_module, "REAUTH_RETRY_SECONDS", 0.01)
    current = _Conn("current")
    client, options_of = await _connected_capturing(monkeypatch, current, RuntimeError("no server free"))

    await options_of[current]["lame_duck_mode_cb"]()
    current.is_closed = True
    await asyncio.sleep(0.1)

    assert client.raw is current
    await client.shutdown()


async def test_lame_duck_under_a_connection_already_replaced_moves_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    current, successor = _Conn("current"), _Conn("successor")
    client, options_of = await _connected_capturing(monkeypatch, current, successor)
    await client.renew_connection(retire_after=timedelta(seconds=30))

    await options_of[current]["lame_duck_mode_cb"]()
    await _settle()

    assert client.raw is successor
    await client.shutdown()


async def test_shutdown_stops_a_move_under_way(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client_module, "REAUTH_RETRY_SECONDS", 3600.0)
    current = _Conn("current")
    client, options_of = await _connected_capturing(monkeypatch, current, RuntimeError("no server free"))

    await options_of[current]["lame_duck_mode_cb"]()
    await _settle()
    await asyncio.wait_for(client.shutdown(), timeout=2.0)

    assert current.is_closed


async def test_a_pinned_run_stays_on_the_connection_it_started_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """two connections are two publishers: a run split across them can arrive out of order."""
    current, successor = _Conn("current"), _Conn("successor")
    client = await _connected(monkeypatch, current, successor)
    pin = client.publish_pin()
    stream = Subject.raw("hub.stream.agent.corr")

    await client.publish_raw(subject=stream, payload=b"token-1", pin=pin)
    await client.renew_connection(retire_after=timedelta(seconds=30))
    await client.publish_raw(subject=stream, payload=b"token-2", pin=pin)
    await client.publish_raw(subject=Subject.raw("unpinned"), payload=b"other")

    assert current.published == [("hub.stream.agent.corr", b"token-1"), ("hub.stream.agent.corr", b"token-2")]
    assert successor.published == [("unpinned", b"other")]
    await client.shutdown()


async def test_a_run_pinned_after_the_handover_starts_on_the_successor(monkeypatch: pytest.MonkeyPatch) -> None:
    current, successor = _Conn("current"), _Conn("successor")
    client = await _connected(monkeypatch, current, successor)
    await client.renew_connection(retire_after=timedelta(seconds=30))
    pin = client.publish_pin()

    await client.publish_raw(subject=Subject.raw("s"), payload=b"t", pin=pin)

    assert successor.published == [("s", b"t")]
    assert current.published == []
    await client.shutdown()


async def test_a_run_that_outlives_its_connection_continues_on_the_current_one(monkeypatch: pytest.MonkeyPatch) -> None:
    current, successor = _Conn("current"), _Conn("successor")
    client = await _connected(monkeypatch, current, successor)
    pin = client.publish_pin()
    await client.publish_raw(subject=Subject.raw("s"), payload=b"t1", pin=pin)

    await client.renew_connection(retire_after=timedelta(seconds=0.01))
    await _until(lambda: current.drained)
    await client.publish_raw(subject=Subject.raw("s"), payload=b"t2", pin=pin)

    assert current.published == [("s", b"t1")]
    assert successor.published == [("s", b"t2")]
    await client.shutdown()


async def test_a_renewal_request_for_this_runner_moves_it_to_a_successor(monkeypatch: pytest.MonkeyPatch) -> None:
    """a changed grant reaches a live connection by a lossless renewal, not a day later."""
    from threetears.nats import CredentialRenewalReason, CredentialRenewalRequest, Subjects

    current, successor = _Conn("current"), _Conn("successor")
    client = await _connected(monkeypatch, current, successor)
    await client.renew_on_request(inbox_prefix="_INBOX_pod", is_mine=lambda request: request.pod_id in (None, "pod-a"))
    [notice_sub] = current.subs
    assert notice_sub.subject == Subjects.credential_renewal_request("_INBOX_pod").path

    request = CredentialRenewalRequest(reason=CredentialRenewalReason.GRANTS_CHANGED, pod_id="pod-a")
    await notice_sub.queue_in.put(_Msg(request.model_dump_json().encode(), subject=notice_sub.subject))
    await _until(lambda: client.raw is successor)

    assert not current.is_closed, "the replaced connection is held for its work"
    await client.shutdown()


async def test_a_renewal_request_for_another_runner_moves_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    from threetears.nats import CredentialRenewalReason, CredentialRenewalRequest

    current, successor = _Conn("current"), _Conn("successor")
    client = await _connected(monkeypatch, current, successor)
    await client.renew_on_request(inbox_prefix="_INBOX_pod", is_mine=lambda request: request.pod_id in (None, "pod-a"))
    [notice_sub] = current.subs

    request = CredentialRenewalRequest(reason=CredentialRenewalReason.GRANTS_CHANGED, pod_id="pod-b")
    await notice_sub.queue_in.put(_Msg(request.model_dump_json().encode(), subject=notice_sub.subject))
    await asyncio.sleep(0.1)

    assert client.raw is current
    await client.shutdown()
