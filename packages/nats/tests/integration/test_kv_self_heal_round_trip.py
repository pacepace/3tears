"""Integration test: NatsKvBucket self-heals after the underlying stream is wiped.

Reproduces the production failure mode end-to-end against a real broker: a single-node
NATS restart on ephemeral JetStream storage deletes every stream/KV bucket, leaving the
cached bucket handle dead ("nats: no response from stream") until the process restarts --
which silenced the wake scheduler. Here we delete the stream out from under a live handle
and assert the next op recreates the bucket and succeeds.

Uses the session-scoped ``nats_container`` fixture; a checkout without docker skips cleanly.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from threetears.nats import NatsClient, set_default_namespace

pytestmark = pytest.mark.integration


async def test_kv_op_self_heals_after_stream_deleted(nats_container: str) -> None:
    set_default_namespace("healtest")
    async with await NatsClient.connect(
        nats_url=nats_container,
        nats_subject_namespace="healtest",
        client_name="kv-heal",
    ) as nc:
        bucket = await nc.kv_bucket(name="heal", ttl=timedelta(seconds=60))
        await bucket.put(key="k", value=b"v1")

        # Simulate the NATS wipe: delete the underlying JetStream stream out from under the
        # cached handle (KV buckets back onto a stream named ``KV_<bucket>``).
        js = nc.jetstream_context()
        await js.delete_stream(f"KV_{bucket.name}")

        # The cached handle is now dead. The next op must re-open (recreate) + retry rather
        # than failing forever. A lock-style create is the canonical wake-scheduler call.
        rev = await bucket.create(key="agent_wake_tick", value=b"1")
        assert rev is not None and rev > 0

        # The bucket is fully usable again on the recreated stream.
        assert await bucket.get(key="agent_wake_tick") == b"1"
