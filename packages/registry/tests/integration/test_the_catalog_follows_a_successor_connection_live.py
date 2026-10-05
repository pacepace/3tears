"""the registry keeps recording registrations after its NATS connection moves to a successor.

A server going lame-duck makes the client open a successor connection and retire the old one -- not
a reconnect, so no reconnect hook fires. A catalog bucket handle bound to the retired connection
would fail every write with ``nats: connection closed``; the catalog's handle must follow the client.

Against a real broker: the catalog is opened through the owner ``RegistryServer.serve`` starts,
the client moves to a successor through :meth:`NatsClient.renew_connection` (the same handover a
lame-duck move runs), the old connection is retired and closed, and only then is a tool registered.
It must reach the bucket, read back over a second connection.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import timedelta

import pytest
from nats.js.errors import BucketNotFoundError, NoKeysError

from threetears.nats import NatsClient, set_default_namespace
from threetears.registry.catalog import CatalogEntry, ToolCatalog, ToolEndpoint
from threetears.registry.catalog_persistence import catalog_bucket

from ..unit.registry.copy_entries import uniform_entry

pytestmark = pytest.mark.integration

_SETTLE_SECONDS = 15.0

#: how long the replaced connection is kept for the work it carries before it is drained and closed
_RETIRE_AFTER = timedelta(seconds=0.2)


def _entry(name: str) -> CatalogEntry:
    return uniform_entry(
        tool_name=name,
        tool_version="1.0.0",
        full_name=f"{name}@1.0.0",
        description=f"test tool {name}",
        input_schema={"type": "object", "properties": {}},
        endpoints=[ToolEndpoint(pod_id="pod-001", status="available")],
    )


async def _stored_full_names(operator: NatsClient, bucket: str) -> set[str]:
    """every tool the bucket holds as a fresh bind on a second connection reads it; empty while absent.

    :param operator: a second connection, holding no state of the registry's
    :ptype operator: NatsClient
    :param bucket: the bucket's exact name
    :ptype bucket: str
    :return: the full names stored
    :rtype: set[str]
    """
    names: set[str] = set()
    try:
        kv = await operator.jetstream_context().key_value(bucket)
        for key in await kv.keys():
            entry = await kv.get(key)
            if entry.value is not None:
                names.add(json.loads(entry.value)["full_name"])
    except BucketNotFoundError, NoKeysError:
        # NOSILENT: an absent or empty bucket IS the answer this reads
        names = set()
    return names


async def _until_stored(operator: NatsClient, bucket: str, expected: set[str]) -> None:
    deadline = asyncio.get_running_loop().time() + _SETTLE_SECONDS
    held = await _stored_full_names(operator, bucket)
    while held != expected:
        assert asyncio.get_running_loop().time() < deadline, f"bucket {bucket} holds {held}, expected {expected}"
        await asyncio.sleep(0.1)
        held = await _stored_full_names(operator, bucket)


async def test_a_registration_after_a_successor_move_reaches_the_bucket(nats_container: str) -> None:
    namespace = f"succ{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    bucket = f"tool_catalog_{uuid.uuid4().hex[:8]}"
    registry = await NatsClient.connect(
        nats_url=nats_container, nats_subject_namespace=namespace, client_name="registry"
    )
    operator = await NatsClient.connect(
        nats_url=nats_container, nats_subject_namespace=namespace, client_name="operator"
    )
    catalog = ToolCatalog()
    owner = catalog_bucket(catalog=catalog, client=registry, bucket=bucket)
    try:
        await owner.start()
        deadline = asyncio.get_running_loop().time() + _SETTLE_SECONDS
        while owner.bucket is None:
            assert asyncio.get_running_loop().time() < deadline, "the catalog bucket was never declared"
            await asyncio.sleep(0.05)
        await catalog.register(_entry("threetears.calculator"))
        await _until_stored(operator, bucket, {"threetears.calculator@1.0.0"})

        replaced = registry.raw
        await registry.renew_connection(retire_after=_RETIRE_AFTER)
        assert registry.raw is not replaced, "the renewal did not move the client to a successor"
        deadline = asyncio.get_running_loop().time() + _SETTLE_SECONDS
        while not replaced.is_closed:
            assert asyncio.get_running_loop().time() < deadline, "the replaced connection was never retired"
            await asyncio.sleep(0.05)

        await catalog.register(_entry("threetears.clock"))

        await _until_stored(operator, bucket, {"threetears.calculator@1.0.0", "threetears.clock@1.0.0"})
    finally:
        await owner.stop()
        try:
            await operator.jetstream_context().delete_key_value(bucket)
        finally:
            await registry.shutdown()
            await operator.shutdown()
