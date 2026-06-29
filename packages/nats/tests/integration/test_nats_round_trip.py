"""integration test: NatsClient round-trip against a NATS testcontainer.

requires docker; mark integration. covers connect / publish_typed /
subscribe_typed / request / kv_bucket round-trips against a real
JetStream-enabled NATS server.

uses the canonical session-scoped ``nats_container`` fixture from
:mod:`threetears.core.testing.fixtures` (registered at the workspace
root conftest). ``check_docker_available`` inside that fixture gates
the suite on the docker daemon -- a fresh checkout without docker
skips cleanly rather than erroring on ``DockerException`` from a raw
``DockerContainer.start()``.

run with:

::

    uv run pytest -m integration packages/nats/tests/integration/
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest
from pydantic import BaseModel

from threetears.nats import IncomingMessage, NatsClient, Subjects, set_default_namespace


pytestmark = pytest.mark.integration


class _Echo(BaseModel):
    """tiny round-trip payload for the integration tests."""

    text: str
    n: int


async def test_publish_subscribe_round_trip(nats_container: str) -> None:
    """typed publish lands on subscribe_typed callback."""
    set_default_namespace("itest")
    async with await NatsClient.connect(
        nats_url=nats_container,
        nats_subject_namespace="itest",
        client_name="round-trip",
    ) as nc:
        received: list[_Echo] = []

        async def on_msg(msg: _Echo) -> None:
            received.append(msg)

        sub = await nc.subscribe_typed(
            subject=Subjects.tools_call(),
            cb=on_msg,
            message_type=_Echo,
        )
        await asyncio.sleep(0.1)  # let subscription register
        await nc.publish(
            subject=Subjects.tools_call(),
            message=_Echo(text="hello", n=1),
        )
        await asyncio.sleep(0.5)  # let dispatch fire
        await nc.unsubscribe(sub)
        assert received == [_Echo(text="hello", n=1)]


async def test_request_response_round_trip(nats_container: str) -> None:
    """request encodes message and decodes typed response."""
    set_default_namespace("itest")
    async with await NatsClient.connect(
        nats_url=nats_container,
        nats_subject_namespace="itest",
        client_name="rr",
    ) as nc:

        async def responder(msg: IncomingMessage) -> None:
            """decode request bytes, build reply, publish via reply_subject."""
            req = _Echo.model_validate_json(msg.data)
            response = _Echo(text=f"echo: {req.text}", n=req.n + 1)
            if msg.reply_subject is not None:
                await nc.publish_reply(reply_subject=msg.reply_subject, message=response)

        sub = await nc.subscribe(subject=Subjects.tools_call(), cb=responder)
        try:
            await asyncio.sleep(0.1)
            response = await nc.request(
                subject=Subjects.tools_call(),
                message=_Echo(text="ping", n=1),
                response_type=_Echo,
                timeout=timedelta(seconds=3),
            )
            assert response == _Echo(text="echo: ping", n=2)
        finally:
            await nc.unsubscribe(sub)


async def test_reconnect_preserves_connection_and_subscription(nats_container: str) -> None:
    """reconnect() drives a real nats-py reconnect: the client stays open and subs survive.

    proves the proactive re-auth primitive against a real broker -- after ``reconnect()`` the
    SAME client is still connected (not closed) and the subscription registered before the
    reconnect still delivers, because nats-py replays it under its original ``sid``. on the live
    enforcing bus the reconnect additionally re-runs the auth-callout to mint a fresh user JWT;
    that end-to-end re-auth is proven against the running stack, while this pins the wrapper-level
    reconnect contract (connection + subscription continuity) on a real server.
    """
    set_default_namespace("itest")
    async with await NatsClient.connect(
        nats_url=nats_container,
        nats_subject_namespace="itest",
        client_name="reconnect-itest",
    ) as nc:
        received: list[_Echo] = []

        async def on_msg(msg: _Echo) -> None:
            received.append(msg)

        reconnected: list[bool] = []

        async def _on_reconnect() -> None:
            reconnected.append(True)

        nc.add_reconnect_callback(_on_reconnect)

        sub = await nc.subscribe_typed(
            subject=Subjects.tools_call(),
            cb=on_msg,
            message_type=_Echo,
        )
        await asyncio.sleep(0.1)  # let subscription register

        # pre-reconnect delivery
        await nc.publish(subject=Subjects.tools_call(), message=_Echo(text="before", n=1))
        await asyncio.sleep(0.3)
        assert received == [_Echo(text="before", n=1)]

        # force a clean reconnect while connected (the proactive re-auth trigger).
        await nc.reconnect()

        # wait for nats-py to complete the reconnect (it spawns _attempt_reconnect in the
        # background); confirm a real server round-trip, not just the cached socket state.
        deadline = 10.0
        waited = 0.0
        while waited < deadline and not await nc.ping(timeout=1.0):
            await asyncio.sleep(0.2)
            waited += 0.2
        assert nc.is_closed is False
        assert await nc.ping(timeout=2.0) is True
        # the reconnected_cb fan-out fired on the real reconnect.
        assert reconnected == [True]

        # post-reconnect delivery: nats-py replayed the SUB under its original sid, so the SAME
        # subscription still delivers without any re-subscribe.
        await nc.publish(subject=Subjects.tools_call(), message=_Echo(text="after", n=2))
        await asyncio.sleep(0.3)
        assert _Echo(text="after", n=2) in received

        await nc.unsubscribe(sub)


async def test_kv_bucket_round_trip(nats_container: str) -> None:
    """KV bucket put / get / delete round-trip."""
    set_default_namespace("itest")
    async with await NatsClient.connect(
        nats_url=nats_container,
        nats_subject_namespace="itest",
        client_name="kv",
    ) as nc:
        bucket = await nc.kv_bucket(name="test_round_trip", ttl=timedelta(minutes=5))
        await bucket.put(key="alpha", value=b"value-1")
        assert await bucket.get(key="alpha") == b"value-1"
        entry = await bucket.get_entry(key="alpha")
        assert entry is not None
        value, rev = entry
        assert value == b"value-1"

        # CAS update with correct revision succeeds
        new_rev = await bucket.update(key="alpha", value=b"value-2", revision=rev)
        assert new_rev is not None and new_rev != rev

        # CAS update with stale revision returns None
        stale = await bucket.update(key="alpha", value=b"value-3", revision=rev)
        assert stale is None

        await bucket.delete(key="alpha")
        assert await bucket.get(key="alpha") is None
