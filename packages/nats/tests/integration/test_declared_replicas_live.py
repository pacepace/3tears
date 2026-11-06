"""a declaration that asks for no replicas creates on one server, and its every re-declaration keeps that count.

Against a real single-node broker: the local stack's NATS. A replica count is never lowered by a re-declaration
(cobalt prod, 2026-11-02: 56 of 58 streams found on one server after their declarers restarted), and a declaration
that asks for none still creates on one, as the local stack needs: a single node cannot hold three.
"""

from __future__ import annotations

import uuid

import pytest

from threetears.nats import NatsClient, set_default_namespace

pytestmark = pytest.mark.integration


async def test_a_stream_and_buckets_declared_without_replicas_are_created_and_kept_on_one_server(
    nats_container: str,
) -> None:
    namespace = f"repl{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    client = await NatsClient.connect(nats_url=nats_container, nats_subject_namespace=namespace, client_name="repl")
    try:
        js = client.jetstream_context()
        stream = await client.ensure_jetstream_stream(name="audit", subjects=[f"{namespace}.audit.>"])
        assert (await js.stream_info(stream)).config.num_replicas == 1
        # every re-declaration, with no count and with one, updates nothing onto another count
        await client.ensure_jetstream_stream(name="audit", subjects=[f"{namespace}.audit.>"], max_age_seconds=60.0)
        live = (await js.stream_info(stream)).config
        assert (live.num_replicas, live.max_age) == (1, 60.0)
        await client.ensure_jetstream_stream(
            name="audit", subjects=[f"{namespace}.audit.>"], max_age_seconds=60.0, replicas=1
        )
        assert (await js.stream_info(stream)).config.num_replicas == 1

        await client.kv_bucket(name="pointers")
        assert (await js.stream_info(f"KV_{namespace}-pointers")).config.num_replicas == 1
        await client.ensure_object_store(name="objects", max_bytes=1024 * 1024)
        await client.ensure_object_store(name="objects", max_bytes=2 * 1024 * 1024)
        objects = (await js.stream_info(f"OBJ_{namespace}-objects")).config
        assert objects.num_replicas == 1
        assert objects.max_bytes == 2 * 1024 * 1024
    finally:
        await client.shutdown()
