"""the registry's result stream and catalog bucket come back after a restart that lost them.

Found live in a cold-start validation: after a NATS restart that lost its JetStream volume, every
tool registration answered ``CATALOG_UNAVAILABLE`` -- the catalog write failed with "no response
from stream" -- because the registry created its catalog bucket only at startup, and a restarted
agent could not register its tools until the registry was restarted by hand.

Against a real broker, with the registry's own two declarations: the result stream through
:meth:`NatsClient.ensure_jetstream_stream` exactly as ``RegistryServer`` declares it, and the
catalog through the owner :func:`~threetears.registry.catalog_persistence.catalog_bucket` builds,
which ``RegistryServer.serve`` starts. The loss is a second connection deleting both; the client
then goes through nats-py's real reconnect path, and its restoration puts the bucket back and has
the owner write the catalog into it.
"""

from __future__ import annotations

import asyncio
import json
import uuid

import pytest
from nats.js.errors import BucketNotFoundError, NoKeysError, NotFoundError

from threetears.nats import RESULT_RETENTION_SECONDS, RESULT_STREAM_SUFFIX, NatsClient, Subjects, set_default_namespace
from threetears.registry.catalog import ToolCatalog, ToolEndpoint
from threetears.registry.catalog_persistence import catalog_bucket

from ..unit.registry.copy_entries import uniform_entry

pytestmark = pytest.mark.integration

_SETTLE_SECONDS = 15.0


async def _stored_full_names(operator: NatsClient, bucket: str) -> list[str]:
    """every tool the bucket holds as a fresh bind on a second connection reads it; ``[]`` while absent.

    :param operator: a second connection, holding no state of the registry's
    :ptype operator: NatsClient
    :param bucket: the bucket's exact name
    :ptype bucket: str
    :return: the full names stored
    :rtype: list[str]
    """
    names: list[str] = []
    try:
        kv = await operator.jetstream_context().key_value(bucket)
        for key in await kv.keys():
            entry = await kv.get(key)
            if entry.value is not None:
                names.append(json.loads(entry.value)["full_name"])
    except BucketNotFoundError, NoKeysError:
        # NOSILENT: an absent or empty bucket IS the answer this reads
        names = []
    return names


async def test_the_result_stream_and_the_catalog_come_back_holding_the_catalog(nats_container: str) -> None:
    namespace = f"catalog{uuid.uuid4().hex[:6]}"
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
        stream = await registry.ensure_jetstream_stream(
            name=RESULT_STREAM_SUFFIX,
            subjects=[str(Subjects.tools_result_wildcard()), str(Subjects.tools_reply_wildcard())],
            max_age_seconds=RESULT_RETENTION_SECONDS,
            max_msgs_per_subject=1,
        )
        await owner.start()
        deadline = asyncio.get_running_loop().time() + _SETTLE_SECONDS
        while owner.bucket is None:
            assert asyncio.get_running_loop().time() < deadline, "the catalog bucket was never declared"
            await asyncio.sleep(0.05)
        await catalog.register(
            uniform_entry(
                tool_name="threetears.calculator",
                tool_version="1.0.0",
                full_name="threetears.calculator@1.0.0",
                description="adds",
                input_schema={"type": "object", "properties": {}},
                endpoints=[ToolEndpoint(pod_id="pod-001", status="available")],
            )
        )

        js = operator.jetstream_context()
        await js.delete_stream(stream)
        await js.delete_stream(f"KV_{bucket}")
        await registry.reconnect()

        deadline = asyncio.get_running_loop().time() + _SETTLE_SECONDS
        stored: list[str] = []
        stream_back = False
        while not (stream_back and stored):
            assert asyncio.get_running_loop().time() < deadline, (
                f"after the loss: result stream back={stream_back}, catalog bucket holds {stored}"
            )
            await asyncio.sleep(0.1)
            try:
                await js.stream_info(stream)
                stream_back = True
            except NotFoundError:
                # NOSILENT: the stream not back yet is what this loop waits out
                stream_back = False
            stored = await _stored_full_names(operator, bucket)

        assert stored == ["threetears.calculator@1.0.0"]
        assert catalog.persisting is True
    finally:
        await owner.stop()
        try:
            await operator.jetstream_context().delete_key_value(bucket)
        finally:
            await registry.shutdown()
            await operator.shutdown()
