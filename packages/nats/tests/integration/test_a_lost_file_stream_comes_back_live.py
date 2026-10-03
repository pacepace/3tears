"""a FILE-storage stream, and the durable on it, come back after the server lost them.

Found live in a cold-start validation: a NATS restart that lost its JetStream volume -- what a
restarted NATS pod on Kubernetes can do, whatever the declared storage -- took the hub's
file-storage turn, delivery and audit streams with it, and nothing put them back. The
agent-router's durable turn consumer logged ``stream not found`` on every rebind until the hub was
restarted by hand, and a tool-call turn spanning the restart failed after 122 s.

Against a real broker, as ``test_a_restart_wiped_stream_comes_back_live.py`` does for memory
storage: the server-side loss is a second connection deleting the stream, and the client then goes
through nats-py's real reconnect path (:meth:`NatsClient.reconnect`). The shared test broker is not
restarted: other tests hold connections to it.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import pytest
from nats.js.errors import NotFoundError

from threetears.nats import NatsClient, Subjects, set_default_namespace

pytestmark = pytest.mark.integration

#: how long the restoration may take to put everything back once the client has reconnected
_SETTLE_SECONDS = 15.0


async def _until_present(client: NatsClient, stream: str, durable: str) -> None:
    """wait until the stream and the durable on it exist on the server.

    :param client: a connected client to look through
    :ptype client: NatsClient
    :param stream: the stream name
    :ptype stream: str
    :param durable: the durable that must exist on it
    :ptype durable: str
    :return: nothing
    :rtype: None
    """
    js = client.jetstream_context()
    deadline = asyncio.get_running_loop().time() + _SETTLE_SECONDS
    present = False
    while not present:
        try:
            await js.stream_info(stream)
            await js.consumer_info(stream, durable)
            present = True
        except NotFoundError:
            if asyncio.get_running_loop().time() > deadline:
                raise
            await asyncio.sleep(0.1)


async def test_a_lost_file_stream_and_its_durable_come_back_after_a_reconnect(nats_container: str) -> None:
    namespace = f"filelost{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    hub = await NatsClient.connect(nats_url=nats_container, nats_subject_namespace=namespace, client_name="hub")
    operator = await NatsClient.connect(
        nats_url=nats_container, nats_subject_namespace=namespace, client_name="operator"
    )
    handled: asyncio.Queue[bytes] = asyncio.Queue()

    async def handle(msg: Any) -> None:
        await handled.put(bytes(msg.data))
        await msg.ack()

    pull_loop: asyncio.Task[None] | None = None
    try:
        stream = await hub.ensure_jetstream_stream(
            name="agents-turn", subjects=[Subjects.audit_wildcard().path], storage="file"
        )
        before = (await operator.jetstream_context().stream_info(stream)).config
        pull = await hub.jetstream_pull_subscribe(
            subject=Subjects.audit_event("turn"),
            durable="agent-turn-router",
            cb=handle,
            max_deliver=3,
            stream=stream,
            fetch_timeout_seconds=0.5,
        )
        pull_loop = asyncio.create_task(pull.run())

        # what a restart that lost the JetStream volume does: the stream and its consumers go
        await operator.jetstream_context().delete_stream(stream)
        await hub.reconnect()
        await _until_present(operator, stream, "agent-turn-router")

        after = (await operator.jetstream_context().stream_info(stream)).config
        assert (after.subjects, after.storage) == (before.subjects, before.storage)
        assert getattr(after.storage, "value", after.storage) == "file"

        await operator.jetstream_publish(subject=Subjects.audit_event("turn"), payload=b"a-turn")
        assert await asyncio.wait_for(handled.get(), timeout=_SETTLE_SECONDS) == b"a-turn"
        await pull.stop()
    finally:
        if pull_loop is not None:
            pull_loop.cancel()
        await hub.shutdown()
        await operator.shutdown()


async def test_a_lost_file_kv_bucket_comes_back_after_a_reconnect(nats_container: str) -> None:
    namespace = f"filekv{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    declarer = await NatsClient.connect(
        nats_url=nats_container, nats_subject_namespace=namespace, client_name="declarer"
    )
    operator = await NatsClient.connect(
        nats_url=nats_container, nats_subject_namespace=namespace, client_name="operator"
    )
    try:
        bucket = await declarer.ensure_kv_bucket(name="agent-config", storage="file")
        kv_stream = f"KV_{bucket.name}"
        before = (await operator.jetstream_context().stream_info(kv_stream)).config

        await operator.jetstream_context().delete_stream(kv_stream)
        await declarer.reconnect()
        js = operator.jetstream_context()
        deadline = asyncio.get_running_loop().time() + _SETTLE_SECONDS
        after = None
        while after is None:
            try:
                after = (await js.stream_info(kv_stream)).config
            except NotFoundError:
                if asyncio.get_running_loop().time() > deadline:
                    raise
                await asyncio.sleep(0.1)

        assert (after.storage, after.allow_direct, after.max_msgs_per_subject) == (
            before.storage,
            before.allow_direct,
            before.max_msgs_per_subject,
        )
    finally:
        await declarer.shutdown()
        await operator.shutdown()
