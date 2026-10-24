"""Integration: a released lease leaves nothing in a real JetStream KV bucket, and no live lease is lost.

What the in-memory bucket cannot prove: that the server accepts a release's lapsing purge marker under the
revision fence, removes it once its lifetime passes, and leaves a lease another pod took over the marker in
the meantime -- the marker's lifetime is its own message's, never the key's.

Uses the session-scoped ``nats_container`` fixture; a checkout without docker skips cleanly.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import timedelta

import pytest

from threetears.core.coordination import KVLease
from threetears.core.coordination import lease as lease_module
from threetears.nats import NatsClient, set_default_namespace
from threetears.nats.subject_permissions import kv_stream_name

pytestmark = pytest.mark.integration

_CYCLES = 50


async def _messages(nc: NatsClient, bucket: str) -> int:
    info = await nc.jetstream_context().stream_info(kv_stream_name(bucket))
    return int(info.state.messages)


async def test_released_unique_keys_leave_no_marker_and_live_leases_stay(
    nats_container: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(lease_module, "_RELEASED_MARKER_TTL", timedelta(seconds=1))
    namespace = f"marks{uuid.uuid4().hex[:8]}"
    set_default_namespace(namespace)
    async with (
        await NatsClient.connect(nats_url=nats_container, nats_subject_namespace=namespace, client_name="pod-a") as a,
        await NatsClient.connect(nats_url=nats_container, nats_subject_namespace=namespace, client_name="pod-b") as b,
    ):
        pod_a, pod_b = KVLease(a, pod_id="pod-a"), KVLease(b, pod_id="pod-b")
        held = await pod_a.acquire("refresh.enr", ttl_seconds=60, max_wait_seconds=0)
        for n in range(_CYCLES):
            handle = await pod_a.acquire(f"enr_answers.answer-{n}", ttl_seconds=60, max_wait_seconds=0)
            await handle.release()
        # pod-b takes a key the instant pod-a's release left its marker there
        released = await pod_a.acquire("build.enr", ttl_seconds=60, max_wait_seconds=0)
        await released.release()
        taken = await pod_b.acquire("build.enr", ttl_seconds=60, max_wait_seconds=0)

        bucket = f"{namespace}-leases"
        deadline = asyncio.get_running_loop().time() + 10
        while (left := await _messages(a, bucket)) > 2:
            assert asyncio.get_running_loop().time() < deadline, (
                f"{left} messages left; only the two live leases belong"
            )
            await asyncio.sleep(0.25)

        assert await pod_a.holder("refresh.enr") == "pod-a"
        assert await pod_b.holder("build.enr") == "pod-b", "the released marker's lapse took the successor's lease"
        await held.refresh()
        await taken.refresh()
        await held.release()
        await taken.release()


async def test_a_stale_release_is_refused_and_leaves_the_entry(nats_container: str) -> None:
    namespace = f"marks{uuid.uuid4().hex[:8]}"
    set_default_namespace(namespace)
    async with await NatsClient.connect(
        nats_url=nats_container, nats_subject_namespace=namespace, client_name="stale"
    ) as nc:
        bucket = await nc.kv_bucket(name="leases", history=1)
        judged = await bucket.put(key="k", value=b"old")
        await bucket.put(key="k", value=b"new")
        assert await bucket.delete(key="k", revision=judged, marker_ttl=timedelta(seconds=1)) is False
        await asyncio.sleep(2)
        assert await bucket.get(key="k") == b"new", "a refused release removed the key"
