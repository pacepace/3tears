"""a message published while a pull consumer stops is handled, never stranded on its dead fetch.

Found through the aibots hub's audit-anonymize test, intermittent under parallel load: an erasure
waits until the shared audit durable has acked everything published before it, and it timed out
with ``0 pending, 1 awaiting ack``. A probe of the stuck state showed the message delivered to the
STOPPED consumer's pull request: ``stop()`` removed the fetch's inbox client-side at once, the
``UNSUB`` had not reached the server yet, and the pod's event -- published on another connection --
was delivered to the request and dropped. It then waits out the durable's ``ack_wait`` (60s in the
hub) before any other fetcher can have it. Measured 6 of 20 without any injection.

Deterministic here: the ``UNSUB`` is held back on its way to the server, so the window the race
needs is always open, and the event is published from a second connection inside it. The hold is
injected at the wrapper's own seam -- the function the stop sends its ``UNSUB`` through -- so
nothing here reaches into nats-py.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import pytest
from nats.js.client import JetStreamContext

import threetears.nats.client as client_module
from threetears.nats import NatsClient, Subjects, set_default_namespace

pytestmark = pytest.mark.integration


async def test_an_event_published_as_the_consumer_stops_is_handled(
    nats_container: str, monkeypatch: pytest.MonkeyPatch
) -> None:
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

        send_unsubscribe = client_module.send_unsubscribe

        async def slow_unsubscribe(*args: Any, **kwargs: Any) -> None:
            await asyncio.sleep(1.0)
            await send_unsubscribe(*args, **kwargs)

        # the injection point: hold the stop's UNSUB back on its way to the server
        monkeypatch.setattr(client_module, "send_unsubscribe", slow_unsubscribe)
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


async def test_an_event_delivered_to_a_request_the_client_gave_up_on_is_handled(
    nats_container: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """the request can outlive the fetch: the client's timer starts before the request reaches the server.

    Under load the hub's fetch timed out client-side while its pull request was still live on the
    server, so a stop that waited for the fetch still unsubscribed with a request outstanding, and
    the pod's event went to it. Here the fetch sends its request and gives up at once, as such a
    fetch does, and the event is delivered to that request before the stop: it is queued on the
    still-subscribed inbox, and the stop must handle it rather than discard the queue.
    """
    from nats.errors import TimeoutError as NatsTimeoutError

    namespace = f"gaveup{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    hub = await NatsClient.connect(nats_url=nats_container, nats_subject_namespace=namespace, client_name="hub")
    pod = await NatsClient.connect(nats_url=nats_container, nats_subject_namespace=namespace, client_name="pod")
    try:
        stream = await hub.ensure_jetstream_stream(name=f"GAVEUP{namespace}", subjects=[Subjects.audit_wildcard().path])
        handled: list[bytes] = []

        async def handle(msg: Any) -> None:
            handled.append(msg.data)
            await msg.ack()

        consumer = await hub.jetstream_pull_subscribe(
            subject=Subjects.audit_wildcard(),
            durable="gaveup",
            cb=handle,
            max_deliver=5,
            ack_wait_seconds=60.0,
            stream=stream,
            fetch_timeout_seconds=3.0,
        )
        fetch = JetStreamContext.PullSubscription.fetch

        async def fetch_that_gave_up(
            psub: JetStreamContext.PullSubscription,
            batch: int = 1,
            timeout: float | None = None,
            heartbeat: float | None = None,
        ) -> Any:
            # nats-py's own fetch sends a request the server holds for 5s; the client stops
            # waiting for it almost at once, which is what a fetch whose timer fired first does
            try:
                return await asyncio.wait_for(fetch(psub, batch, timeout=5.0, heartbeat=heartbeat), timeout=0.2)
            except TimeoutError as exc:
                raise NatsTimeoutError from exc

        # the injection point: a fetch whose timer fired first
        monkeypatch.setattr(JetStreamContext.PullSubscription, "fetch", fetch_that_gave_up)
        assert await consumer.fetch_and_process() == 0  # the request is live on the server, the fetch is over

        await pod.jetstream_publish(subject=Subjects.audit_event("survey.response.submit"), payload=b"late")
        await asyncio.sleep(0.3)  # delivered to the live request: on its way into the inbox's queue
        await consumer.stop()
        await asyncio.sleep(0.3)

        info = await pod.jetstream_context().consumer_info(stream, "gaveup")
        assert handled == [b"late"]
        assert (info.num_pending, info.num_ack_pending) == (0, 0)
    finally:
        await hub.shutdown()
        await pod.shutdown()
