"""a KV failure blames a missing grant only when a missing grant can be the cause.

**The incident.** During a NATS restart in production, the hub's agent catalog logged::

    agent catalog writes to its bucket have failed 3 times in a row: ... 'stream is offline'.
    FIX: grant this principal the KV bucket 'agent_router_catalog'. Add a JsResource.kv(...) ...

The grant was present. The bucket's stream was offline for the length of the restart and came back
by itself; thirty seconds later the same catalog logged that its writes were landing again. The
remedy sent the reader to the one place that was not broken.

**Why the two are distinguishable.** A request the connection is not granted is refused at the
publish and never answered -- it arrives as a deadline, which is why the grant is a fair suspect
for an unanswered request. A stream that is offline, or a JetStream that is not up yet, is the
server ANSWERING: ``stream is offline`` (``JSStreamOfflineErr``, err_code 10118), its with-reason
variant (10194), ``JetStream system temporarily unavailable`` (``JSClusterNotAvailErr``, 10008),
``JetStream cluster can not handle request`` (``JSClusterNotLeaderErr``, 10009), or a 503 carrying
no code, which is what nats-py raises when nothing serves the JetStream API at all. None of those is
something a grant causes or cures.

**The error shapes are nats-py's own.** Every failure here is produced by a real
``nats.js.JetStreamContext`` parsing the reply a broker sends -- the server's JSON error body, or the
``NoRespondersError`` nats-py's core request raises -- so the exception types and codes the remedy is
chosen from are exactly the ones production sees. Only the transport under the context is scripted:
it answers ``request()``, the single call nats-py sends a KV put and a ``STREAM.*`` request through.

**Every opener a self-heal can take is driven.** A put that fails re-opens the bucket and retries,
and the re-open's failure is what the caller sees. A declaring handle the client refills looks the
stream up first (``STREAM.INFO``); a declaring handle with no refill creates, then binds; a bind-only
handle waits for its declarer. Each reached the grant remedy by its own line, so each is driven.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest
from nats.errors import NoRespondersError, TimeoutError as NatsTimeoutError
from nats.js.client import JetStreamContext
from nats.js.kv import KeyValue

from threetears.nats.errors import KvError
from threetears.nats.kv import KvTimings, NatsKvBucket

_BUCKET = "agent_router_catalog"

#: the words the grant remedy (``threetears.nats.diagnostics.kv_grant_remedy``) opens with, certain
#: or hedged; the outage remedy must contain neither.
_GRANT_ADVICE = ("grant this principal", "JsResource.kv")

#: what a broker mid-restart answers. Each is the error body the server sends, as nats-py receives it.
_OUTAGE_REPLIES: dict[str, dict[str, Any]] = {
    "stream-offline": {"code": 500, "err_code": 10118, "description": "stream is offline"},
    "stream-offline-with-reason": {
        "code": 500,
        "err_code": 10194,
        "description": "stream is offline: catchup in progress",
    },
    "jetstream-temporarily-unavailable": {
        "code": 503,
        "err_code": 10008,
        "description": "JetStream system temporarily unavailable",
    },
    "cluster-cannot-handle-request": {
        "code": 500,
        "err_code": 10009,
        "description": "JetStream cluster can not handle request",
    },
}


@dataclass(frozen=True)
class _Reply:
    """a reply as nats-py reads it: the JetStream API and a publish ack read only its body.

    :ivar data: the reply body
    """

    data: bytes


# parity-exempt: core-NATS connection stand-in answering request() alone, the one call nats-py's JetStreamContext sends a KV put and every STREAM.* request through
class _FakeBrokerConnection:
    """the transport under a real nats-py JetStream context, answering every request one way.

    :param answer: given the request subject, returns the reply's JSON body or raises what nats-py's
        core request raises
    :ptype answer: Callable[[str], dict[str, Any]]
    """

    def __init__(self, answer: Callable[[str], dict[str, Any]]) -> None:
        self._answer = answer
        self.subjects: list[str] = []

    async def request(
        self, subject: str, payload: bytes = b"", timeout: float = 0.5, headers: dict[str, str] | None = None
    ) -> _Reply:
        """answer one request the way the scripted broker does.

        :param subject: the request subject
        :ptype subject: str
        :param payload: the request body, unread
        :ptype payload: bytes
        :param timeout: the caller's deadline, unused
        :ptype timeout: float
        :param headers: the request headers, unread
        :ptype headers: dict[str, str] | None
        :return: the reply
        :rtype: _Reply
        """
        del payload, timeout, headers
        self.subjects.append(subject)
        return _Reply(data=json.dumps(self._answer(subject)).encode())


# parity-exempt: NatsClient stand-in exposing only raw and jetstream_context(), the two members a NatsKvBucket reads off its client
class _FakeClient:
    """the wrapper client a bucket is built on: one real JetStream context over the scripted broker.

    :param connection: the scripted broker connection
    :ptype connection: _FakeBrokerConnection
    """

    def __init__(self, connection: _FakeBrokerConnection) -> None:
        self.raw = connection
        self._js = JetStreamContext(connection, timeout=0.5)  # type: ignore[arg-type]

    def jetstream_context(self) -> JetStreamContext:
        """the client's JetStream context.

        :return: the context
        :rtype: JetStreamContext
        """
        return self._js


def _replying(body: dict[str, Any]) -> Callable[[str], dict[str, Any]]:
    """a broker that answers every request with the error ``body``.

    :param body: the server's error object
    :ptype body: dict[str, Any]
    :return: the answer function
    :rtype: Callable[[str], dict[str, Any]]
    """
    return lambda _subject: {"error": body}


def _raising(exc: Exception) -> Callable[[str], dict[str, Any]]:
    """a broker whose every request fails the way nats-py's core request fails.

    :param exc: what nats-py raises
    :ptype exc: Exception
    :return: the answer function
    :rtype: Callable[[str], dict[str, Any]]
    """

    def _answer(_subject: str) -> dict[str, Any]:
        raise exc

    return _answer


#: the three openers a self-heal re-open runs, as the constructor arguments that select them.
_OPENERS: dict[str, dict[str, Any]] = {
    # the catalog's own handle: declared by the client, which refills it when it comes back empty.
    "declaring-refilled": {"create_if_missing": True, "on_recreated": lambda: None},
    "declaring": {"create_if_missing": True, "on_recreated": None},
    "bind-only": {"create_if_missing": False, "on_recreated": None},
}


async def _put_error(answer: Callable[[str], dict[str, Any]], opener: str) -> str:
    """drive one catalog-shaped put through the scripted broker and return what the caller is told.

    :param answer: how the broker answers every request
    :ptype answer: Callable[[str], dict[str, Any]]
    :param opener: which self-heal opener the bucket re-opens with (a key of :data:`_OPENERS`)
    :ptype opener: str
    :return: the message of the ``KvError`` the put raised
    :rtype: str
    """
    client = _FakeClient(_FakeBrokerConnection(answer))
    js = client.jetstream_context()
    bucket = NatsKvBucket(
        client=client,  # type: ignore[arg-type]
        full_name=_BUCKET,
        kv=KeyValue(name=_BUCKET, stream=f"KV_{_BUCKET}", pre=f"$KV.{_BUCKET}.", js=js, direct=False),
        ttl=None,
        storage="file",
        timings=KvTimings(bind_wait_for_declarer_seconds=0.0),
        **_OPENERS[opener],
    )
    with pytest.raises(KvError) as caught:
        await bucket.put(key="agent-a", value=b"{}")
    return str(caught.value)


@pytest.mark.parametrize("opener", sorted(_OPENERS))
@pytest.mark.parametrize("reply", sorted(_OUTAGE_REPLIES))
async def test_a_stream_the_server_says_is_unavailable_is_not_blamed_on_a_grant(reply: str, opener: str) -> None:
    """the server ANSWERED that the stream cannot serve right now: say that, and send nobody to the grants."""
    message = await _put_error(_replying(_OUTAGE_REPLIES[reply]), opener)

    assert _OUTAGE_REPLIES[reply]["description"] in message, "the server's own words must survive"
    for advice in _GRANT_ADVICE:
        assert advice not in message, f"an answered outage was blamed on a grant: {message}"
    assert "temporarily unavailable" in message, message
    assert "recovers on its own" in message, message


@pytest.mark.parametrize("opener", sorted(_OPENERS))
async def test_a_jetstream_nothing_is_serving_yet_is_not_blamed_on_a_grant(opener: str) -> None:
    """no responders on the JetStream API: nats-py raises a codeless 503 -- JetStream is not up yet."""
    message = await _put_error(_raising(NoRespondersError()), opener)

    for advice in _GRANT_ADVICE:
        assert advice not in message, f"a JetStream still starting was blamed on a grant: {message}"
    assert "temporarily unavailable" in message, message


@pytest.mark.parametrize("opener", sorted(_OPENERS))
async def test_an_unanswered_request_still_names_the_grant(opener: str) -> None:
    """the contrast that keeps the fix honest: a refused request is never answered, so the grant stays named."""
    message = await _put_error(_raising(NatsTimeoutError()), opener)

    assert "grant this principal" in message, message
    assert "temporarily unavailable" not in message, message


async def test_an_answered_error_that_is_no_outage_is_not_blamed_on_a_grant_either() -> None:
    """any answer at all rules the grant out; only the outage words are reserved for an outage."""
    body = {"code": 503, "err_code": 10039, "description": "JetStream not enabled for account"}
    message = await _put_error(_replying(body), "declaring-refilled")

    assert "JetStream not enabled for account" in message
    for advice in _GRANT_ADVICE:
        assert advice not in message, f"an answered refusal was blamed on a grant: {message}"
    assert "temporarily unavailable" not in message, "a configuration answer is not an outage"


async def test_a_create_and_a_bind_that_fail_differently_both_say_why() -> None:
    """a nats-py APIError's repr is empty, so the create's own words must be printed beside it."""
    create_refusal = {"code": 500, "err_code": 10023, "description": "insufficient resources"}
    bind_refusal = _OUTAGE_REPLIES["stream-offline"]

    def _answer(subject: str) -> dict[str, Any]:
        if ".STREAM.CREATE." in subject:
            return {"error": create_refusal}
        return {"error": bind_refusal}

    message = await _put_error(_answer, "declaring")

    assert "insufficient resources" in message, f"the create's reason was dropped: {message}"
    assert "stream is offline" in message, message
