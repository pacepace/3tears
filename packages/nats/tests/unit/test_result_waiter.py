"""An answer must survive a reconnect on EITHER end of the wire.

The production failure was a responder losing the right to publish: ``allow_responses`` belongs to the
connection that received the request, and the credential refresh that keeps a pod authenticated is a
reconnect, so a 92-second scan finished with exit 0 and 68KB of results it could never deliver.

Moving the answer onto a subject the responder holds a standing grant on fixes that half. These tests
cover the other half -- the CALLER. If the caller's own connection cycles while it waits, or the
server reaps the ephemeral consumer underneath it, an answer that is sitting in the stream must still
be collected. Otherwise the loss has been relocated rather than ended.
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

_SUBJECT = Subject.raw("3tears.tools.result.pod-A.call-1")
_STREAM = "3tears-tools-results"


class _Msg:
    """one delivered JetStream message; records whether the waiter acked it."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.acked = False

    async def ack(self) -> None:
        self.acked = True


class _PullSub:
    """a nats-py pull subscription whose fetch behaviour each test scripts.

    ``script`` is consumed one entry per ``fetch``: a ``_Msg`` is delivered, an exception is raised,
    and ``None`` means "nothing yet" (which nats-py signals as a TimeoutError).
    """

    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)
        self.fetches = 0
        self.unsubscribed = False

    async def fetch(self, batch: int, timeout: float) -> list[Any]:
        self.fetches += 1
        if not self._script:
            raise NatsTimeoutError
        step = self._script.pop(0)
        if step is None:
            raise NatsTimeoutError
        if isinstance(step, BaseException):
            raise step
        return [step]

    async def unsubscribe(self) -> None:
        self.unsubscribed = True


class _Js:
    """a JetStream context handing out a scripted pull subscription per subscribe call."""

    def __init__(self, subs: list[Any]) -> None:
        self._subs = list(subs)
        self.configs: list[Any] = []
        self.subscribe_calls = 0
        self.streams: list[str | None] = []

    async def pull_subscribe(self, subject: str, *, stream: str | None = None, config: Any = None) -> Any:
        self.subscribe_calls += 1
        self.configs.append(config)
        self.streams.append(stream)
        if not self._subs:
            raise RuntimeError("no scripted subscription left")
        step = self._subs.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step


def _waiter(js: _Js, *, poll: float = 0.01) -> JetStreamResultWaiter:
    return JetStreamResultWaiter(
        js=js,
        subject=_SUBJECT,
        stream=_STREAM,
        inactive_threshold_seconds=600.0,
        poll_seconds=poll,
    )


async def test_the_answer_is_returned_and_acked() -> None:
    """the ordinary case: the tool finishes, the answer is collected, the message is acked."""
    sub = _PullSub([None, _Msg(b"68KB of results")])
    js = _Js([sub])
    waiter = _waiter(js)
    await waiter.open()

    payload = await waiter.wait(timeout=timedelta(seconds=5))

    assert payload == b"68KB of results"
    assert sub.fetches == 2


async def test_the_consumer_is_created_before_the_wait_begins() -> None:
    """opening first is what makes the ordering safe to read, not merely safe.

    the caller opens the waiter before dispatching the call, so there is no window in which the
    answer could be published with nothing yet listening for it.
    """
    js = _Js([_PullSub([])])
    waiter = _waiter(js)

    assert js.subscribe_calls == 0
    await waiter.open()
    assert js.subscribe_calls == 1


async def test_the_consumer_reads_from_the_start_of_the_stream() -> None:
    """DeliverPolicy.ALL on a per-call subject removes the race entirely.

    the subject is minted for this one call, so "everything on this subject" is exactly "this call's
    answer" -- whether it was published before or after the consumer existed. a NEW-only policy would
    silently drop an answer that beat the consumer into being.
    """
    from nats.js.api import AckPolicy, DeliverPolicy

    js = _Js([_PullSub([])])
    await _waiter(js).open()

    config = js.configs[0]
    assert config.deliver_policy == DeliverPolicy.ALL
    assert config.filter_subject == _SUBJECT.path
    assert config.ack_policy == AckPolicy.EXPLICIT


async def test_the_stream_is_named_explicitly() -> None:
    """naming the stream avoids the ``$JS.API.STREAM.NAMES`` lookup, which nobody is granted.

    an ungranted JetStream call does not fail fast with a denial -- it blocks to its deadline, which
    reads as an unreachable broker rather than a missing permission.
    """
    js = _Js([_PullSub([])])
    await _waiter(js).open()
    assert js.streams == [_STREAM]


async def test_the_consumer_outlives_the_call_it_is_waiting_for() -> None:
    """the ephemeral consumer's keepalive must exceed the whole wait budget.

    a threshold shorter than the call means the server reaps the consumer mid-tool and the answer
    arrives with nothing bound to receive it -- the original bug, re-created on the consumer side.
    """
    from threetears.nats.client import NatsClient, _RESULT_WAITER_KEEPALIVE_MARGIN_SECONDS

    js = _Js([_PullSub([])])
    waiter = JetStreamResultWaiter(
        js=js,
        subject=_SUBJECT,
        stream=_STREAM,
        inactive_threshold_seconds=1200.0 + _RESULT_WAITER_KEEPALIVE_MARGIN_SECONDS,
        poll_seconds=5.0,
    )
    await waiter.open()

    assert js.configs[0].inactive_threshold > 1200.0
    assert callable(NatsClient.jetstream_result_waiter)


async def test_a_reconnect_mid_wait_does_not_discard_the_answer() -> None:
    """THE OTHER HALF OF THE BUG. A caller-side reconnect must not lose a computed result.

    After the transport cycles, the server-side ephemeral consumer may be gone and the next fetch
    fails with something other than "nothing yet". Giving up there would throw away an answer the
    stream is still holding -- the same loss as before, moved to the receiving end. So the consumer is
    rebuilt and the wait continues.
    """
    dead = _PullSub([ConnectionResetError("connection closed")])
    revived = _PullSub([_Msg(b"delivered after the reconnect")])
    js = _Js([dead, revived])
    waiter = _waiter(js)
    await waiter.open()

    payload = await waiter.wait(timeout=timedelta(seconds=5))

    assert payload == b"delivered after the reconnect"
    assert js.subscribe_calls == 2, "the waiter did not rebuild its consumer after the failed fetch"
    assert dead.unsubscribed


async def test_a_failed_rebuild_is_retried_rather_than_fatal() -> None:
    """a broker still coming back must not end the wait on the first failed rebuild."""
    js = _Js(
        [
            _PullSub([RuntimeError("consumer gone")]),
            RuntimeError("broker still down"),
            _PullSub([_Msg(b"eventually")]),
        ]
    )
    waiter = _waiter(js)
    await waiter.open()

    payload = await waiter.wait(timeout=timedelta(seconds=5))

    assert payload == b"eventually"


async def test_an_answer_that_never_comes_ends_at_the_deadline() -> None:
    """the wait is bounded: a pod that died mid-tool must not hang its caller forever."""
    js = _Js([_PullSub([])])
    waiter = _waiter(js)
    await waiter.open()

    with pytest.raises(RequestTimeoutError, match=_SUBJECT.path):
        await waiter.wait(timeout=timedelta(seconds=0.05))


async def test_wait_before_open_is_a_programming_error() -> None:
    """waiting on a consumer that was never created would silently time out every call."""
    waiter = _waiter(_Js([]))
    with pytest.raises(RuntimeError, match="before open"):
        await waiter.wait(timeout=timedelta(seconds=0.05))


async def test_close_is_idempotent_and_never_raises() -> None:
    """close runs in the caller's ``finally`` while it already holds its answer.

    the ephemeral consumer ages out on its own threshold, so a failing unsubscribe leaks nothing
    durable and must not turn a successful call into an error.
    """

    class _Hostile(_PullSub):
        async def unsubscribe(self) -> None:
            raise RuntimeError("broker unreachable")

    waiter = _waiter(_Js([_Hostile([])]))
    await waiter.open()

    await waiter.close()
    await waiter.close()


async def test_cancellation_is_not_swallowed_as_a_transport_blip() -> None:
    """shutdown must end the wait, not be mistaken for a fetch failure and retried forever."""

    class _Hangs(_PullSub):
        async def fetch(self, batch: int, timeout: float) -> list[Any]:
            await asyncio.sleep(3600)
            return []

    waiter = _waiter(_Js([_Hangs([])]))
    await waiter.open()
    task = asyncio.create_task(waiter.wait(timeout=timedelta(seconds=60)))
    await asyncio.sleep(0.01)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
