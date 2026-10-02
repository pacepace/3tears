"""a pull handler that outlives ack_wait keeps its message: no other fetcher gets it, nothing is redelivered.

The aibots hub's person erasure waits until the shared audit durable has acked everything published
before it, bounded at 30 seconds. A hub killed mid-handle leaves its message awaiting ack until the
durable's ``ack_wait`` runs out, so the erasure can only survive a dead fetcher when ``ack_wait`` is
shorter than its bound -- and a short ``ack_wait`` is only safe when a LIVE handler that runs past it
is not redelivered under it. Here two fetchers share one durable with a one-second ``ack_wait``, and
the handler the message reaches first takes three.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import pytest

from threetears.nats import NatsClient, Subjects, set_default_namespace

pytestmark = pytest.mark.integration


async def test_a_handler_running_past_ack_wait_is_the_only_one_to_see_its_message(nats_container: str) -> None:
    namespace = f"slowpull{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    first = await NatsClient.connect(nats_url=nats_container, nats_subject_namespace=namespace, client_name="one")
    second = await NatsClient.connect(nats_url=nats_container, nats_subject_namespace=namespace, client_name="two")
    try:
        stream = await first.ensure_jetstream_stream(
            name=f"SLOWPULL{namespace}", subjects=[Subjects.audit_wildcard().path]
        )
        handled: list[tuple[str, int]] = []

        def handler(name: str) -> Any:
            async def handle(msg: Any) -> None:
                handled.append((name, int(msg.metadata.num_delivered)))
                await asyncio.sleep(3.0)
                await msg.ack()

            return handle

        consumers = [
            await client.jetstream_pull_subscribe(
                subject=Subjects.audit_wildcard(),
                durable="slowpull",
                cb=handler(name),
                max_deliver=5,
                ack_wait_seconds=1.0,
                stream=stream,
                fetch_timeout_seconds=0.5,
            )
            for name, client in (("one", first), ("two", second))
        ]
        loops = [asyncio.create_task(consumer.run()) for consumer in consumers]
        await first.jetstream_publish(subject=Subjects.audit_event("survey.response.submit"), payload=b"slow")
        await asyncio.sleep(4.5)
        for consumer in consumers:
            await consumer.stop()
        for loop in loops:
            loop.cancel()

        info = await first.jetstream_context().consumer_info(stream, "slowpull")
        assert len(handled) == 1, f"the message reached more than one handler: {handled}"
        assert handled[0][1] == 1, "the message was a redelivery"
        assert (info.num_pending, info.num_ack_pending, info.num_redelivered) == (0, 0, 0)
    finally:
        await first.shutdown()
        await second.shutdown()
