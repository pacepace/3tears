"""Integration: ``KVLease.hold`` against a real JetStream KV.

What the in-memory bucket cannot prove: that the key is legal on a real broker, that two pods'
creates contend on it for real, that renewal's compare-and-swap is accepted by JetStream, and that
the second pod gets the key once the first releases it.

Uses the session-scoped ``nats_container`` fixture; a checkout without docker skips cleanly.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import timedelta

import pytest

from threetears.core.coordination import KVLease, LeaseUnavailable
from threetears.nats import NatsClient, set_default_namespace

pytestmark = pytest.mark.integration


async def test_two_pods_contend_for_one_held_lease(nats_container: str) -> None:
    namespace = f"held{uuid.uuid4().hex[:8]}"
    set_default_namespace(namespace)
    key = f"job.{uuid.uuid4().hex[:8]}"
    async with (
        await NatsClient.connect(nats_url=nats_container, nats_subject_namespace=namespace, client_name="pod-a") as a,
        await NatsClient.connect(nats_url=nats_container, nats_subject_namespace=namespace, client_name="pod-b") as b,
    ):
        first = await KVLease(a, pod_id="pod-a").hold(key, ttl=timedelta(seconds=2), renew_every=timedelta(seconds=0.5))
        try:
            await asyncio.sleep(3)  # longer than the TTL: only renewal keeps it
            assert first.held
            with pytest.raises(LeaseUnavailable):
                await KVLease(b, pod_id="pod-b").hold(key, ttl=timedelta(seconds=2), renew_every=timedelta(seconds=0.5))
        finally:
            await first.release()

        second = await KVLease(b, pod_id="pod-b").hold(
            key, ttl=timedelta(seconds=2), renew_every=timedelta(seconds=0.5)
        )
        assert second.held
        await second.release()
