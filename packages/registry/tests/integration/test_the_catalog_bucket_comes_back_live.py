"""the registry's result stream and catalog bucket come back after a restart that lost them.

Found live in a cold-start validation: after a NATS restart that lost its JetStream volume, every
tool registration answered ``CATALOG_UNAVAILABLE`` -- the catalog write failed with "no response
from stream" -- because the registry created its catalog bucket only at startup, and a restarted
agent could not register its tools until the registry was restarted by hand.

Against a real broker, with the registry's own two declarations: the result stream through
:meth:`NatsClient.ensure_jetstream_stream` exactly as ``RegistryServer`` declares it, and the
catalog through :class:`CatalogPersistence`, which ``RegistryServer.serve`` starts. The loss is a
second connection deleting both; the client then goes through nats-py's real reconnect path.
"""

from __future__ import annotations

import asyncio
import json
import uuid

import pytest

from threetears.nats import RESULT_RETENTION_SECONDS, RESULT_STREAM_SUFFIX, NatsClient, Subjects, set_default_namespace
from threetears.registry.catalog import ToolCatalog, ToolEndpoint
from threetears.registry.catalog_persistence import CatalogPersistence

from ..unit.registry.copy_entries import uniform_entry

pytestmark = pytest.mark.integration

_SETTLE_SECONDS = 15.0


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
    persistence = CatalogPersistence(catalog=catalog, nc=registry, bucket=bucket)
    try:
        stream = await registry.ensure_jetstream_stream(
            name=RESULT_STREAM_SUFFIX,
            subjects=[str(Subjects.tools_result_wildcard()), str(Subjects.tools_reply_wildcard())],
            max_age_seconds=RESULT_RETENTION_SECONDS,
            max_msgs_per_subject=1,
        )
        await persistence.start()
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
        while not stored:
            try:
                await js.stream_info(stream)
                kv = await js.key_value(bucket)
                stored = [json.loads((await kv.get(key)).value)["full_name"] for key in await kv.keys()]
            except Exception:
                if asyncio.get_running_loop().time() > deadline:
                    raise
                await asyncio.sleep(0.1)

        assert stored == ["threetears.calculator@1.0.0"]
    finally:
        await persistence.stop()
        try:
            await operator.jetstream_context().delete_key_value(bucket)
        finally:
            await registry.shutdown()
            await operator.shutdown()
