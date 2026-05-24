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
