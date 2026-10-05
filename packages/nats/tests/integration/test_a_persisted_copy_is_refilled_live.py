"""a bucket that is the persisted copy of memory is put back AND refilled after the server loses it.

Found on cobalt-dev: after a NATS rolling restart the registry refused every tool registration with
``CATALOG_UNAVAILABLE`` until its pod was deleted by hand. Its catalog bucket was held through a raw
nats-py handle that did not follow the client, and a bucket the client puts back after a restart
comes back empty. :class:`~threetears.nats.PersistedCopyBucket` declares the bucket through the
client with a refill, and this proves both paths that create it again on a real broker:

- the client reconnects (nats-py's real reconnect path), and its restoration finds the stream gone;
- no reconnect at all -- the next write finds the stream gone and the handle's self-heal creates it.

The server-side loss is what a restart that lost its storage does -- the stream is deleted -- done by a
second connection; the shared test broker is not restarted, since other tests hold connections to it.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import pytest
from nats.js.errors import BucketNotFoundError, NoKeysError

from threetears.nats import NatsClient, PersistedCopyBucket, set_default_namespace

pytestmark = pytest.mark.integration

#: how long the refill may take to land once the stream is gone
_SETTLE_SECONDS = 15.0


async def _keys_on_the_server(operator: NatsClient, bucket: str) -> dict[str, bytes]:
    """every live entry of ``bucket`` as a fresh bind reads it, ``{}`` while the bucket is absent.

    :param operator: a second connection, holding no state of the client under test
    :ptype operator: NatsClient
    :param bucket: the bucket's exact name
    :ptype bucket: str
    :return: key -> value
    :rtype: dict[str, bytes]
    """
    entries: dict[str, bytes] = {}
    try:
        kv = await operator.jetstream_context().key_value(bucket)
        for key in await kv.keys():
            entry = await kv.get(key)
            if entry.value is not None:
                entries[key] = entry.value
    except BucketNotFoundError, NoKeysError:
        # NOSILENT: an absent or empty bucket IS the answer this reads
        entries = {}
    return entries


async def _until_the_server_holds(operator: NatsClient, bucket: str, expected: dict[str, bytes]) -> None:
    """wait until the bucket on the server holds exactly ``expected``, failing the test if it never does.

    :param operator: a second connection
    :ptype operator: NatsClient
    :param bucket: the bucket's exact name
    :ptype bucket: str
    :param expected: the entries the bucket must hold
    :ptype expected: dict[str, bytes]
    :return: nothing
    :rtype: None
    """
    deadline = asyncio.get_running_loop().time() + _SETTLE_SECONDS
    held = await _keys_on_the_server(operator, bucket)
    while held != expected:
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError(f"bucket {bucket} holds {held}, expected {expected}")
        await asyncio.sleep(0.1)
        held = await _keys_on_the_server(operator, bucket)


async def test_a_persisted_copy_is_refilled_after_a_reconnect_and_after_a_plain_write(nats_container: str) -> None:
    namespace = f"refill{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    bucket_name = f"catalog_{uuid.uuid4().hex[:8]}"
    registry = await NatsClient.connect(
        nats_url=nats_container, nats_subject_namespace=namespace, client_name="registry"
    )
    operator = await NatsClient.connect(
        nats_url=nats_container, nats_subject_namespace=namespace, client_name="operator"
    )
    memory: dict[str, bytes] = {"tool-a": b"a", "tool-b": b"b"}
    loads: list[int] = []

    async def _load(bucket: Any) -> None:
        loads.append(len(loads))
        for key in await bucket.list_keys():
            value = await bucket.get(key=key)
            if value is not None:
                memory.setdefault(key, value)

    async def _write_back(bucket: Any) -> None:
        for key, value in dict(memory).items():
            await bucket.put(key=key, value=value)

    owner = PersistedCopyBucket(client=registry, bucket=bucket_name, load=_load, write_back=_write_back)
    try:
        await owner.start()
        await _until_the_server_holds(operator, bucket_name, memory)
        stream = f"KV_{bucket_name}"

        # (i) the server loses the bucket and the client reconnects: the restoration puts the stream
        # back, and the declarer's refill writes memory back into it.
        await operator.jetstream_context().delete_stream(stream)
        await registry.reconnect()
        await _until_the_server_holds(operator, bucket_name, memory)

        # (ii) the server loses it again and nothing reconnects: the next write finds the stream gone,
        # the handle's self-heal creates it, and the refill puts back everything else.
        await operator.jetstream_context().delete_stream(stream)
        memory["tool-c"] = b"c"
        handle = owner.bucket
        assert handle is not None
        await handle.put(key="tool-c", value=b"c")
        await _until_the_server_holds(operator, bucket_name, memory)

        assert loads == [0], "the copy is loaded once, never on a refill"
    finally:
        await owner.stop()
        await registry.shutdown()
        await operator.shutdown()
