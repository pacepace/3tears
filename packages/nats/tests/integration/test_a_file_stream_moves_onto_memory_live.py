"""a stream live on file storage moves onto memory, and the service bound to it keeps consuming.

NATS is memory only (owner's rule; 2026-11-15 found six platform streams and buckets still on file
storage on cobalt dev and prod). JetStream cannot change a live stream's storage, and a declaration
on memory leaves a file stream as it is, so the move is an operator's: delete, create again on
memory, keep what is still owed. Against a real broker: the consumer bound before the move must
receive a message published after it with no restart, the parked dead letter must survive, and the
declaration on memory must then take its bound in place.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import pytest

from threetears.nats import NatsClient, Subjects, set_default_namespace
from threetears.nats.stream_storage import stream_move, stream_move_plan

pytestmark = pytest.mark.integration

_SETTLE_SECONDS = 15.0


async def test_a_file_stream_moves_onto_memory_and_its_consumer_keeps_consuming(nats_container: str) -> None:
    namespace = f"tomem{uuid.uuid4().hex[:6]}"
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
        subjects = [Subjects.audit_wildcard().path, Subjects.audit_deadletter().path]
        stream = await hub.ensure_jetstream_stream(name="audit", subjects=subjects, storage="file")
        pull = await hub.jetstream_pull_subscribe(
            subject=Subjects.audit_event("turn"),
            durable="unified-audit-consumer",
            cb=handle,
            max_deliver=3,
            stream=stream,
            fetch_timeout_seconds=0.5,
        )
        pull_loop = asyncio.create_task(pull.run())
        await operator.jetstream_publish(subject=Subjects.audit_event("turn"), payload=b"before")
        assert await asyncio.wait_for(handled.get(), timeout=_SETTLE_SECONDS) == b"before"
        await operator.jetstream_publish(subject=Subjects.audit_deadletter(), payload=b"parked")
        js = operator.jetstream_context()
        # the consumer's ack reaches the server a moment after the handler returns
        deadline = asyncio.get_running_loop().time() + _SETTLE_SECONDS
        while (await stream_move_plan(js, stream, replicas=1)).refused is not None:
            assert asyncio.get_running_loop().time() < deadline, "the consumer never settled"
            await asyncio.sleep(0.1)

        moved = await stream_move(js, stream, replicas=1, max_age_seconds=3600.0)

        assert moved.copied == 1 and dict(moved.skipped) == {"consumed": 1}
        after = (await js.stream_info(stream)).config
        assert getattr(after.storage, "value", after.storage) == "memory"
        parked = await js.get_msg(stream, 1, subject=Subjects.audit_deadletter().path, next=True)
        assert bytes(parked.data) == b"parked"

        await operator.jetstream_publish(subject=Subjects.audit_event("turn"), payload=b"after")
        assert await asyncio.wait_for(handled.get(), timeout=_SETTLE_SECONDS) == b"after"

        # the declaration on memory now meets its own storage and applies its bound in place
        await hub.ensure_jetstream_stream(name="audit", subjects=subjects, storage="memory", max_bytes=1 << 20)
        assert (await js.stream_info(stream)).config.max_bytes == 1 << 20
        await pull.stop()
    finally:
        if pull_loop is not None:
            pull_loop.cancel()
        await hub.shutdown()
        await operator.shutdown()


async def test_a_file_kv_bucket_moves_onto_memory_with_its_values(nats_container: str) -> None:
    namespace = f"tomemkv{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    declarer = await NatsClient.connect(
        nats_url=nats_container, nats_subject_namespace=namespace, client_name="declarer"
    )
    try:
        bucket = await declarer.ensure_kv_bucket(name="catalog", storage="file")
        await bucket.put(key="a", value=b"A")
        await bucket.put(key="b", value=b"B")
        await bucket.delete(key="b")
        kv_stream = f"KV_{bucket.name}"
        js = declarer.jetstream_context()

        moved = await stream_move(js, kv_stream, replicas=1)

        assert moved.copied == 1 and dict(moved.skipped) == {"kv marker": 1}
        storage = (await js.stream_info(kv_stream)).config.storage
        assert getattr(storage, "value", storage) == "memory"
        assert await bucket.get(key="a") == b"A"
        assert await bucket.get(key="b") is None
    finally:
        await declarer.shutdown()
