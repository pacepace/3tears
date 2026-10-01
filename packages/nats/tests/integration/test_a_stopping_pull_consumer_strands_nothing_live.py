"""a message published while a pull consumer stops is handled, never stranded on its dead fetch.

Found through the aibots hub's audit-anonymize test, intermittent under parallel load: an erasure
waits until the shared audit durable has acked everything published before it, and it timed out
with ``0 pending, 1 awaiting ack``. A probe of the stuck state showed the message delivered to the
STOPPED consumer's pull request: ``stop()`` removed the fetch's inbox client-side at once, the
``UNSUB`` had not reached the server yet, and the pod's event -- published on another connection --
was delivered to the request and dropped. It then waits out the durable's ``ack_wait`` (60s in the
hub) before any other fetcher can have it. Measured 6 of 20 without any injection.

Deterministic here: the ``UNSUB`` is held back on its way to the server, so the window the race
needs is always open, and the event is published from a second connection inside it.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import pytest

from threetears.nats import NatsClient, Subjects, set_default_namespace

pytestmark = pytest.mark.integration


async def test_an_event_published_as_the_consumer_stops_is_handled(nats_container: str) -> None:
    namespace = f"stopping{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    hub = await NatsClient.connect(nats_url=nats_container, nats_subject_namespace=namespace, client_name="hub")
    pod = await NatsClient.connect(nats_url=nats_container, nats_subject_namespace=namespace, client_name="pod")
    try:
        stream = await hub.ensure_jetstream_stream(
            name=f"STOPPING{namespace}", subjects=[Subjects.audit_wildcard().path]
        )
        handled: list[bytes] = []

        async def handle(msg: Any) -> None:
            handled.append(msg.data)
            await msg.ack()

        consumer = await hub.jetstream_pull_subscribe(
            subject=Subjects.audit_wildcard(),
            durable="stopping",
            cb=handle,
            max_deliver=5,
            ack_wait_seconds=60.0,
            stream=stream,
            fetch_timeout_seconds=3.0,
        )
        loop = asyncio.create_task(consumer.run())
        await asyncio.sleep(0.3)  # the loop is inside a fetch: its pull request is on the server

        raw = hub.raw
        send_unsubscribe = raw._send_unsubscribe  # noqa: SLF001 -- the injection point: hold the UNSUB back

        async def slow_unsubscribe(*args: Any, **kwargs: Any) -> None:
            await asyncio.sleep(1.0)
            await send_unsubscribe(*args, **kwargs)

        raw._send_unsubscribe = slow_unsubscribe  # noqa: SLF001
        stopping = asyncio.create_task(consumer.stop())
        await asyncio.sleep(0.05)
        await pod.jetstream_publish(subject=Subjects.audit_event("survey.response.submit"), payload=b"during-stop")
        await stopping
        loop.cancel()
        await asyncio.sleep(0.3)

        info = await pod.jetstream_context().consumer_info(stream, "stopping")
        assert handled == [b"during-stop"]
        assert (info.num_pending, info.num_ack_pending) == (0, 0)
    finally:
        await hub.shutdown()
        await pod.shutdown()
