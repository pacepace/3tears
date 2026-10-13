"""Integration: ``CoalescedRun`` on two replicas over a real JetStream KV.

What the in-memory bucket cannot prove: that the request key sits beside the lease under a real
owner scope, that its compare-and-swap delete is accepted by JetStream, and that requests made on
two connections at once while one replica runs come to one more run, not one per request.

Uses the session-scoped ``nats_container`` fixture; a checkout without docker skips cleanly.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import timedelta

import pytest

from threetears.core.coordination import KVLease
from threetears.core.coordination.coalesced_run import CoalescedRun
from threetears.nats import NatsClient, set_default_namespace

pytestmark = pytest.mark.integration


async def test_overlapping_requests_on_two_replicas_run_once_and_once_more(nats_container: str) -> None:
    namespace = f"coalesce{uuid.uuid4().hex[:8]}"
    set_default_namespace(namespace)
    runs: list[str] = []
    running = asyncio.Event()
    finish = asyncio.Event()

    def body(replica: str) -> object:
        async def run() -> None:
            runs.append(replica)
            running.set()
            if len(runs) == 1:
                await finish.wait()

        return run

    async with (
        await NatsClient.connect(nats_url=nats_container, nats_subject_namespace=namespace, client_name="pod-a") as a,
        await NatsClient.connect(nats_url=nats_container, nats_subject_namespace=namespace, client_name="pod-b") as b,
    ):
        scope = "enr-pod"
        replica_a = CoalescedRun(
            KVLease(a, pod_id="pod-a", key_scope=scope),
            "enr/refresh",
            body("a"),  # type: ignore[arg-type]
            ttl=timedelta(seconds=10),
            renew_every=timedelta(seconds=2),
        )
        replica_b = CoalescedRun(
            KVLease(b, pod_id="pod-b", key_scope=scope),
            "enr/refresh",
            body("b"),  # type: ignore[arg-type]
            ttl=timedelta(seconds=10),
            renew_every=timedelta(seconds=2),
        )
        await replica_a.request()
        first = asyncio.create_task(replica_a.drain())
        await running.wait()

        # five signals land on both replicas while a's run is going
        await asyncio.gather(
            *(replica.request() for replica in (replica_a, replica_b, replica_b, replica_a, replica_b))
        )
        assert await replica_b.drain() == 0
        finish.set()

        assert await first == 2
        assert runs == ["a", "a"]
        assert not await replica_b.requested()
