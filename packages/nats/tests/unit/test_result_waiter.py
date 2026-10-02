"""An answer must survive a reconnect on EITHER end of the wire -- and reach only its caller.

The production failure was a responder losing the right to publish: ``allow_responses`` belongs to the
connection that received the request, and the credential refresh that keeps a pod authenticated is a
reconnect, so a 92-second scan finished with exit 0 and 68KB of results it could never deliver.

Moving the answer onto a subject the responder holds a standing grant on fixes that half. These tests
cover the other half -- the CALLER. If the caller's consumer is lost while it waits, an answer that is
sitting in the stream must still be collected. Otherwise the loss has been relocated rather than ended.

And the caller collects it the one way a pod's grant admits: a NAMED consumer whose filter rides in
its create subject, PUSHED to the caller's own inbox. A pull consumer needs ``CONSUMER.MSG.NEXT``,
which reaches any consumer on the stream by name -- the registry's over every pod's results included.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

import pytest
from nats.errors import TimeoutError as NatsTimeoutError

from threetears.nats import RequestTimeoutError, Subject
from threetears.nats.client import JetStreamResultWaiter

pytestmark = pytest.mark.asyncio

_SUBJECT = Subject.raw("3tears.tools.reply.019470a8-b5c3-7def-8123-0000000000aa.call-1")
_STREAM = "3tears-tools-results"


class _Msg:
    """one message pushed to the waiter's inbox; records whether the waiter acked it."""

    def __init__(self, data: bytes, headers: dict[str, str] | None = None) -> None:
        self.data = data
        self.headers = headers
        self.acked = False

    async def ack(self) -> None:
        self.acked = True


def _heartbeat() -> _Msg:
    return _Msg(b"", {"Status": "100", "Description": "Idle Heartbeat"})


class _Sub:
    """a core inbox subscription whose deliveries each test scripts.

    ``script`` is consumed one entry per ``next_msg``: a ``_Msg`` is delivered, an exception is
    raised, and ``None`` means "nothing yet" (which nats-py signals as a TimeoutError).
    """

    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)
        self.polls = 0
        self.unsubscribed = False

    async def next_msg(self, timeout: float) -> Any:
        self.polls += 1
        step = self._script.pop(0) if self._script else None
        if step is None:
            # nothing delivered: nats-py waits out the timeout, then raises
            await asyncio.sleep(timeout)
            raise NatsTimeoutError
        if isinstance(step, BaseException):
            raise step
        return step

    async def unsubscribe(self) -> None:
        self.unsubscribed = True


class _Raw:
    """a nats-py client handing out one scripted subscription per inbox subscribe."""

    def __init__(self, subs: list[_Sub]) -> None:
        self._subs = list(subs)
        self.inboxes: list[str] = []
        self.subscribed: list[str] = []

    def new_inbox(self) -> str:
        inbox = f"_INBOX_agent_pod_x.{len(self.inboxes)}"
        self.inboxes.append(inbox)
        return inbox

    async def subscribe(self, subject: str) -> _Sub:
        self.subscribed.append(subject)
        if not self._subs:
            raise RuntimeError("no scripted subscription left")
        return self._subs.pop(0)


class _RawWithJetStream(_Raw):
    """a scripted nats-py client that also hands out one JetStream context, as a live one does."""

    def __init__(self, subs: list[_Sub], js: _Js) -> None:
        super().__init__(subs)
        self._js = js

    def jetstream(self) -> _Js:
        return self._js


class _Js:
    """a JetStream context recording every consumer create; each may be scripted to fail."""

    def __init__(self, failures: list[BaseException | None] | None = None) -> None:
        self._failures = list(failures or [])
        self.creates: list[tuple[str, Any]] = []
        self.pulls = 0

    async def add_consumer(self, stream: str, config: Any = None) -> Any:
        self.creates.append((stream, config))
        failure = self._failures.pop(0) if self._failures else None
        if failure is not None:
            raise failure
        return object()

    async def pull_subscribe(self, *args: Any, **kwargs: Any) -> Any:
        self.pulls += 1
        raise AssertionError("a pod may not pull: CONSUMER.MSG.NEXT reaches any consumer by name")


def _waiter(
    raw: _Raw, js: _Js, *, poll: float = 0.01, heartbeat: float = 60.0, rebuild_backoff: float = 1.0
) -> JetStreamResultWaiter:
    return JetStreamResultWaiter(
        connection=lambda: raw,
        jetstream=lambda: js,
        subject=_SUBJECT,
        stream=_STREAM,
        inactive_threshold_seconds=600.0,
        poll_seconds=poll,
        heartbeat_seconds=heartbeat,
        rebuild_backoff_seconds=rebuild_backoff,
    )


async def test_the_answer_is_returned_and_acked() -> None:
    """the ordinary case: the tool finishes, the answer is pushed, collected and acked."""
    answer = _Msg(b"68KB of results")
    sub = _Sub([None, _heartbeat(), answer])
    waiter = _waiter(_Raw([sub]), _Js())
    await waiter.open()

    payload = await waiter.wait(timeout=timedelta(seconds=5))

    assert payload == b"68KB of results"
    assert answer.acked


async def test_the_consumer_is_named_filtered_and_pushed_to_the_callers_own_inbox() -> None:
    """the one consumer shape a pod's grant admits.

    A name plus a filter makes nats-py issue ``CONSUMER.CREATE.{stream}.{name}.{filter}``, the form
    nats-server checks against the body; the deliver subject is the inbox this connection subscribed
    before the create, so nothing the consumer pushes can arrive before anything listens.
    """
    raw = _Raw([_Sub([])])
    js = _Js()
    await _waiter(raw, js).open()

    stream, config = js.creates[0]
    assert stream == _STREAM
    assert config.name and config.name.startswith("result-waiter-")
    assert config.filter_subject == _SUBJECT.path
    assert config.deliver_subject == raw.inboxes[0] == raw.subscribed[0]
    assert config.durable_name is None
    assert js.pulls == 0


async def test_every_consumer_gets_a_fresh_name() -> None:
    """two waiters -- or one waiter's rebuild -- never collide on a consumer name."""
    js = _Js()
    await _waiter(_Raw([_Sub([])]), js).open()
    await _waiter(_Raw([_Sub([])]), js).open()
    assert js.creates[0][1].name != js.creates[1][1].name


async def test_the_consumer_is_created_before_the_wait_begins() -> None:
    """opening first is what makes the ordering safe to read, not merely safe.

    the caller opens the waiter before dispatching the call, so there is no window in which the
    answer could be published with nothing yet listening for it.
    """
    js = _Js()
    waiter = _waiter(_Raw([_Sub([])]), js)

    assert js.creates == []
    await waiter.open()
    assert len(js.creates) == 1


async def test_the_consumer_reads_from_the_start_of_the_stream() -> None:
    """DeliverPolicy.ALL on a per-call subject removes the race entirely.

    the subject is minted for this one call, so "everything on this subject" is exactly "this call's
    answer" -- whether it was published before or after the consumer existed. a NEW-only policy would
    silently drop an answer that beat the consumer into being.
    """
    from nats.js.api import AckPolicy, DeliverPolicy

    js = _Js()
    await _waiter(_Raw([_Sub([])]), js).open()

    config = js.creates[0][1]
    assert config.deliver_policy == DeliverPolicy.ALL
    assert config.ack_policy == AckPolicy.EXPLICIT
    assert config.max_ack_pending == 1


async def test_the_consumer_heartbeats_so_its_loss_is_noticed() -> None:
    """a pushed consumer the server lost delivers nothing and says nothing; the heartbeat says so."""
    js = _Js()
    await _waiter(_Raw([_Sub([])]), js, heartbeat=7.0).open()
    assert js.creates[0][1].idle_heartbeat == 7.0


async def test_the_consumer_outlives_the_call_it_is_waiting_for() -> None:
    """the consumer's keepalive must exceed the whole wait budget.

    a threshold shorter than the call means the server reaps the consumer mid-tool and the answer
    arrives with nothing bound to receive it -- the original bug, re-created on the consumer side.

    Driven through :meth:`NatsClient.jetstream_result_waiter`, the method that derives the
    keepalive from the wait budget, so the margin it adds is what is checked.
    """
    from threetears.nats.client import NatsClient

    js = _Js()
    raw = _RawWithJetStream([_Sub([])], js)
    client = NatsClient(raw=raw, namespace="3tears", client_name="t")  # type: ignore[arg-type]

    waiter = await client.jetstream_result_waiter(subject=_SUBJECT, stream=_STREAM, wait_budget=timedelta(seconds=1200))
    await waiter.close()

    assert js.creates[0][1].inactive_threshold > 1200.0


async def test_a_refused_create_leaves_no_subscription_behind() -> None:
    """a create the grant refuses fails the open, and the inbox it subscribed is dropped."""
    sub = _Sub([])
    waiter = _waiter(_Raw([sub]), _Js([RuntimeError("consumer create refused")]))

    with pytest.raises(RuntimeError, match="refused"):
        await waiter.open()
    assert sub.unsubscribed


async def test_a_consumer_that_goes_quiet_is_replaced_and_the_answer_still_collected() -> None:
    """THE OTHER HALF OF THE BUG. A lost consumer must not lose a computed result.

    After a broker restart the consumer may be gone, and a pushed consumer that no longer exists
    neither delivers nor heartbeats. Waiting on it forever would throw away an answer the stream is
    still holding -- the same loss as before, moved to the receiving end. So once it misses its
    heartbeats it is replaced, and the replacement reads the answer from the start of the stream.
    """
    quiet = _Sub([None, None, None, None, None, None])
    revived = _Sub([_Msg(b"delivered after the reconnect")])
    raw = _Raw([quiet, revived])
    js = _Js()
    waiter = _waiter(raw, js, poll=0.01, heartbeat=0.01)
    await waiter.open()

    payload = await waiter.wait(timeout=timedelta(seconds=5))

    assert payload == b"delivered after the reconnect"
    assert len(js.creates) == 2, "the waiter did not replace its quiet consumer"
    assert quiet.unsubscribed


async def test_heartbeats_keep_a_live_consumer() -> None:
    """a consumer that is heartbeating is alive and is not replaced while the tool runs."""
    sub = _Sub([_heartbeat(), _heartbeat(), _heartbeat(), _heartbeat(), _Msg(b"done")])
    js = _Js()
    waiter = _waiter(_Raw([sub]), js, heartbeat=60.0)
    await waiter.open()

    assert await waiter.wait(timeout=timedelta(seconds=5)) == b"done"
    assert len(js.creates) == 1


async def test_a_consumer_the_server_ended_is_replaced_at_once() -> None:
    """``409 Consumer Deleted`` and its kin end the consumer; waiting out its heartbeats is pointless."""
    ended = _Sub([_Msg(b"", {"Status": "409", "Description": "Consumer Deleted"})])
    revived = _Sub([_Msg(b"eventually")])
    js = _Js()
    waiter = _waiter(_Raw([ended, revived]), js)
    await waiter.open()

    assert await waiter.wait(timeout=timedelta(seconds=5)) == b"eventually"
    assert len(js.creates) == 2


async def test_a_failed_delivery_replaces_the_consumer() -> None:
    """a subscription that fails is replaced rather than ending the wait."""
    broken = _Sub([ConnectionResetError("connection closed")])
    revived = _Sub([_Msg(b"delivered after the blip")])
    js = _Js()
    waiter = _waiter(_Raw([broken, revived]), js)
    await waiter.open()

    assert await waiter.wait(timeout=timedelta(seconds=5)) == b"delivered after the blip"
    assert broken.unsubscribed


async def test_a_failed_rebuild_is_retried_rather_than_fatal() -> None:
    """a broker still coming back must not end the wait on the first failed rebuild."""
    js = _Js([None, RuntimeError("broker still down"), None])
    raw = _Raw([_Sub([RuntimeError("consumer gone")]), _Sub([]), _Sub([_Msg(b"eventually")])])
    waiter = _waiter(raw, js, rebuild_backoff=0.0)
    await waiter.open()

    payload = await waiter.wait(timeout=timedelta(seconds=5))

    assert payload == b"eventually"
    assert len(js.creates) == 3


async def test_an_answer_that_never_comes_ends_at_the_deadline() -> None:
    """the wait is bounded: a pod that died mid-tool must not hang its caller forever."""
    waiter = _waiter(_Raw([_Sub([])]), _Js())
    await waiter.open()

    with pytest.raises(RequestTimeoutError, match=_SUBJECT.path):
        await waiter.wait(timeout=timedelta(seconds=0.05))


async def test_wait_before_open_is_a_programming_error() -> None:
    """waiting on a consumer that was never created would silently time out every call."""
    waiter = _waiter(_Raw([]), _Js())
    with pytest.raises(RuntimeError, match="before open"):
        await waiter.wait(timeout=timedelta(seconds=0.05))


async def test_close_is_idempotent_and_never_raises() -> None:
    """close runs in the caller's ``finally`` while it already holds its answer.

    the consumer ages out on its own threshold, so a failing unsubscribe leaks nothing durable and
    must not turn a successful call into an error.
    """

    class _Hostile(_Sub):
        async def unsubscribe(self) -> None:
            raise RuntimeError("broker unreachable")

    waiter = _waiter(_Raw([_Hostile([])]), _Js())
    await waiter.open()

    await waiter.close()
    await waiter.close()


async def test_cancellation_is_not_swallowed_as_a_transport_blip() -> None:
    """shutdown must end the wait, not be mistaken for a delivery failure and retried forever."""

    class _Hangs(_Sub):
        async def next_msg(self, timeout: float) -> Any:
            await asyncio.sleep(3600)
            return None

    waiter = _waiter(_Raw([_Hangs([])]), _Js())
    await waiter.open()
    task = asyncio.create_task(waiter.wait(timeout=timedelta(seconds=60)))
    await asyncio.sleep(0.01)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


async def test_a_credential_renewal_mid_wait_moves_the_consumer_to_the_successor() -> None:
    """a renewal retires the connection the waiter's consumer was made on; the answer is still collected.

    the replacement is made on whichever connection is current when it is made -- the successor --
    and reads the stream from the start, so an answer published while the move happened is there.
    """
    from nats.errors import ConnectionClosedError

    answer = _Msg(b"the answer")
    replaced = _Raw([_Sub([None, ConnectionClosedError()])])
    successor = _Raw([_Sub([answer])])
    current = {"connection": replaced}
    js = _Js()
    waiter = JetStreamResultWaiter(
        connection=lambda: current["connection"],
        jetstream=lambda: js,
        subject=_SUBJECT,
        stream=_STREAM,
        inactive_threshold_seconds=600.0,
        poll_seconds=0.01,
        heartbeat_seconds=60.0,
    )
    await waiter.open()
    current["connection"] = successor  # the renewal: the client's connection is now the successor

    assert await waiter.wait(timeout=timedelta(seconds=2)) == b"the answer"
    assert answer.acked
    assert len(successor.subscribed) == 1
    assert len(js.creates) == 2
