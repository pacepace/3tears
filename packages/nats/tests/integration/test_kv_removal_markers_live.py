"""Integration: a bucket owner's sweep purges the markers plain deletes left, and keeps every live key.

What a double cannot prove: that a stream purge filtered to a key's subject and bounded by the marker's
sequence plus one is accepted by the server, removes the marker, and leaves the bucket's live entries.

Uses the session-scoped ``nats_container`` fixture; a checkout without docker skips cleanly.
"""

from __future__ import annotations

import uuid

import pytest

from threetears.nats import NatsClient, set_default_namespace
from threetears.nats.subject_permissions import kv_stream_name

pytestmark = pytest.mark.integration


async def test_the_sweep_purges_markers_and_keeps_live_keys(nats_container: str) -> None:
    namespace = f"sweep{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    async with await NatsClient.connect(
        nats_url=nats_container, nats_subject_namespace=namespace, client_name="owner"
    ) as nc:
        bucket = await nc.kv_bucket(name="leases", history=1)
        for n in range(20):
            revision = await bucket.put(key=f"enr_answers.answer-{n}", value=b"held")
            assert await bucket.delete(key=f"enr_answers.answer-{n}", revision=revision) is True
        await bucket.put(key="refresh.enr", value=b"live")
        # a key deleted and written again: its latest message is live, so nothing of it is a marker
        revision = await bucket.put(key="build.enr", value=b"first")
        await bucket.delete(key="build.enr", revision=revision)
        await bucket.put(key="build.enr", value=b"again")

        stream = kv_stream_name(f"{namespace}-leases")
        js = nc.jetstream_context()
        assert await bucket.purge_removal_markers(apply=False) == 20
        assert (await js.stream_info(stream)).state.messages == 22

        assert await bucket.purge_removal_markers() == 20

        assert (await js.stream_info(stream)).state.messages == 2
        assert await bucket.get(key="refresh.enr") == b"live"
        assert await bucket.get(key="build.enr") == b"again"
        assert await bucket.purge_removal_markers() == 0
