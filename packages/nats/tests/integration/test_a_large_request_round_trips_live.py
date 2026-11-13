"""Integration test: a large request and a large publish cross a real nats-server whole, in order.

From ``UNJOINED_FROM_BYTES`` :class:`~threetears.nats.NatsClient` no longer hands a payload to nats-py's own
``publish`` / ``request`` (which copy it into one ``bytes`` on the event loop): it queues the ``PUB`` line, the payload
and CRLF as three pending items, and requests through nats-py's response mux itself. Here a real server routes them:
a responder answers each request with the size and sha256 of what it got, and a large publish between two small ones
arrives between them.

Gated on docker: a checkout without docker skips cleanly.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import timedelta

import pytest

from threetears.core.testing.containers import check_docker_available
from threetears.nats import IncomingMessage, NatsClient, Subject, set_default_namespace

from .nats_support import running_nats

pytestmark = pytest.mark.integration

#: the Arrow copy's size; the server's limit is raised past it, as the deployments' is
_LARGE = 12 * 1024 * 1024

_CONFIG = "max_payload: 16MB\nmax_pending: 64MB\n"


async def test_a_large_request_and_publish_cross_a_real_server_whole() -> None:
    """a 12 MB request is answered with its own size and digest; publishes keep their order around a large one."""
    if not check_docker_available():
        pytest.skip("Docker not available")
    set_default_namespace("bigpub")
    payload = (hashlib.sha256(b"large").digest() * (_LARGE // 32 + 1))[:_LARGE]
    with running_nats(_CONFIG) as uri:
        async with (
            await NatsClient.connect(
                nats_url=uri, nats_subject_namespace="bigpub", client_name="responder", verify_jetstream=False
            ) as server,
            await NatsClient.connect(
                nats_url=uri, nats_subject_namespace="bigpub", client_name="requester", verify_jetstream=False
            ) as client,
        ):
            received: list[tuple[int, str]] = []

            async def _answer(msg: IncomingMessage) -> None:
                assert msg.reply_subject is not None
                data = bytes(msg.data)
                await server.publish_raw_reply(
                    reply_subject=msg.reply_subject, payload=f"{len(data)} {hashlib.sha256(data).hexdigest()}".encode()
                )

            async def _record(msg: IncomingMessage) -> None:
                data = bytes(msg.data)
                received.append((len(data), hashlib.sha256(data).hexdigest()))

            await server.subscribe(Subject.raw("bigpub.copy"), cb=_answer)
            await server.subscribe(Subject.raw("bigpub.events"), cb=_record)
            await server.flush()

            for _ in range(2):
                reply = await client.request_raw(
                    subject=Subject.raw("bigpub.copy"), payload=payload, timeout=timedelta(seconds=30)
                )
                assert reply == f"{_LARGE} {hashlib.sha256(payload).hexdigest()}".encode()

            for data in (b"before", payload, b"after"):
                await client.publish_raw(subject=Subject.raw("bigpub.events"), payload=data)
            await client.flush()
            async with asyncio.timeout(10):
                while len(received) < 3:
                    await asyncio.sleep(0.02)
            assert received == [
                (len(data), hashlib.sha256(data).hexdigest()) for data in (b"before", payload, b"after")
            ]
