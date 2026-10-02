"""a memory stream, and the durables on it, come back after the server lost them -- no restart of ours.

Found live in the devx bring-up: the tool registry declared its memory-backed result stream only
at startup, a single-node NATS restart deleted it, and every tool call then failed with ``stream
not found`` until the registry was restarted by hand.

Against a real broker. The server-side loss is what a memory-storage restart does -- the stream is
deleted, and every consumer on it with it -- done by a second connection, and the client then goes
through nats-py's real reconnect path (:meth:`NatsClient.reconnect`), so the restoration runs from
the same ``reconnected_cb`` a broker restart fires. The shared test broker is not restarted: other
tests hold connections to it.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import timedelta
from typing import Any

import pytest
from nats.js.errors import NotFoundError

from threetears.nats import NatsClient, Subjects, set_default_namespace

pytestmark = pytest.mark.integration

#: how long the restoration may take to put everything back once the client has reconnected
_SETTLE_SECONDS = 15.0


async def _until_present(client: NatsClient, stream: str, durable: str | None = None) -> None:
    """wait until the stream (and the durable, when named) exists on the server.

    :param client: a connected client to look through
    :ptype client: NatsClient
    :param stream: the stream name
    :ptype stream: str
    :param durable: a durable that must exist on it too, or ``None``
    :ptype durable: str | None
    :return: nothing
    :rtype: None
    """
    js = client.jetstream_context()
    deadline = asyncio.get_running_loop().time() + _SETTLE_SECONDS
    present = False
    while not present:
        try:
            await js.stream_info(stream)
            if durable is not None:
                await js.consumer_info(stream, durable)
            present = True
        except NotFoundError:
            if asyncio.get_running_loop().time() > deadline:
                raise
            await asyncio.sleep(0.1)


async def test_a_wiped_result_stream_and_its_durables_come_back_after_a_reconnect(nats_container: str) -> None:
    namespace = f"wiped{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    registry = await NatsClient.connect(
        nats_url=nats_container, nats_subject_namespace=namespace, client_name="registry"
    )
    operator = await NatsClient.connect(
        nats_url=nats_container, nats_subject_namespace=namespace, client_name="operator"
    )
    pull_handled: asyncio.Queue[bytes] = asyncio.Queue()
    push_handled: asyncio.Queue[bytes] = asyncio.Queue()

    async def handle_pull(msg: Any) -> None:
        await pull_handled.put(bytes(msg.data))
        await msg.ack()

    async def handle_push(msg: Any) -> None:
        await push_handled.put(bytes(msg.data))
        await msg.ack()

    pull_loop: asyncio.Task[None] | None = None
    try:
        stream = await registry.ensure_jetstream_stream(
            name="tools-results",
            subjects=[Subjects.tools_result_wildcard().path, Subjects.audit_wildcard().path],
            max_age_seconds=900.0,
            max_msgs_per_subject=1,
        )
        before = (await operator.jetstream_context().stream_info(stream)).config
        pull = await registry.jetstream_pull_subscribe(
            subject=Subjects.audit_event("pulled"),
            durable="pulled",
            cb=handle_pull,
            max_deliver=3,
            stream=stream,
            fetch_timeout_seconds=0.5,
        )
        pull_loop = asyncio.create_task(pull.run())
        push = await registry.jetstream_subscribe_durable(
            subject=Subjects.audit_event("pushed"), durable="pushed", cb=handle_push, max_deliver=3, stream=stream
        )

        # what a memory-storage restart does on the server: the stream and every consumer on it go
        await operator.jetstream_context().delete_stream(stream)
        await registry.reconnect()
        await _until_present(operator, stream, "pulled")
        await _until_present(operator, stream, "pushed")

        after = (await operator.jetstream_context().stream_info(stream)).config
        assert (after.subjects, after.storage, after.max_age, after.max_msgs_per_subject) == (
            before.subjects,
            before.storage,
            before.max_age,
            before.max_msgs_per_subject,
        )

        # the registry's own path: a pod's result published to the stream reaches a waiter on it
        result_subject = Subjects.tools_result(uuid.uuid4(), uuid.uuid4())
        waiter = await registry.jetstream_result_waiter(
            subject=result_subject, stream=stream, wait_budget=timedelta(seconds=10)
        )
        try:
            await operator.jetstream_publish(subject=result_subject, payload=b"tool-answer")
            assert await waiter.wait(timeout=timedelta(seconds=10)) == b"tool-answer"
        finally:
            await waiter.close()

        # and both durables deliver again
        await operator.jetstream_publish(subject=Subjects.audit_event("pulled"), payload=b"to-pull")
        await operator.jetstream_publish(subject=Subjects.audit_event("pushed"), payload=b"to-push")
        assert await asyncio.wait_for(pull_handled.get(), timeout=_SETTLE_SECONDS) == b"to-pull"
        assert await asyncio.wait_for(push_handled.get(), timeout=_SETTLE_SECONDS) == b"to-push"
        await pull.stop()
        await push.stop()
    finally:
        if pull_loop is not None:
            pull_loop.cancel()
        await registry.shutdown()
        await operator.shutdown()
