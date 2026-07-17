"""Integration test: ``TokenBucket``/``DistributedCounter`` against a real NATS broker.

Exercises the actual JetStream KV ``create``/``update`` CAS semantics
the in-process unit tests (against ``FakeKvBucket``) cannot validate --
per shard 20a's own Success Criteria
(docs/3tears-migration/20a-3tears-rate-limiter-primitive-proposal.md,
14-eng-ai-survey), this must prove state is genuinely cluster-wide, not
per-pod: every concurrency test below opens SEPARATE
``NatsClient.connect()`` connections (standing in for separate pods)
racing against the identical bucket key, rather than sharing one
connection across all concurrent callers. Uses the canonical
session-scoped ``nats_container`` fixture from
:mod:`threetears.core.testing.fixtures` -- a fresh checkout without
docker skips cleanly via the fixture's ``check_docker_available`` gate.
"""

from __future__ import annotations

import asyncio

import pytest

from threetears.core.coordination import DistributedCounter, TokenBucket
from threetears.nats import NatsClient, set_default_namespace

pytestmark = pytest.mark.integration


class TestTokenBucketRoundTrip:
    """TokenBucket against a real NATS broker, one connection per simulated pod."""

    async def test_claim_and_refill_round_trip(self, nats_container: str) -> None:
        """A claimed token is gone; after refill time elapses, a further claim succeeds."""
        set_default_namespace("ratelimit-itest")
        async with await NatsClient.connect(
            nats_url=nats_container,
            nats_subject_namespace="ratelimit-itest",
            client_name="bucket-rr",
        ) as nc:
            bucket = TokenBucket(nc, bucket_name="rt-tokenbucket", refill_rate=20.0, capacity=1.0)

            first = await bucket.claim("k")
            assert first.claimed is True

            immediately = await bucket.claim("k")
            assert immediately.claimed is False

            await asyncio.sleep(0.1)  # 20 tokens/sec * 0.1s == 2 tokens refilled
            later = await bucket.claim("k")
            assert later.claimed is True

    async def test_cross_pod_concurrent_claims_exactly_capacity_succeed(self, nats_container: str) -> None:
        """N separate pod connections claiming concurrently: exactly `capacity` succeed, cluster-wide.

        each "pod" opens its OWN NatsClient connection against the same
        broker/bucket/key -- proving the token count is shared state in
        JetStream, not per-connection in-memory state that would let
        every pod independently believe it has a full bucket.
        """
        set_default_namespace("ratelimit-itest")
        capacity = 5
        num_pods = 15

        async def claim_from_one_pod(pod_index: int) -> bool:
            async with await NatsClient.connect(
                nats_url=nats_container,
                nats_subject_namespace="ratelimit-itest",
                client_name=f"bucket-pod-{pod_index}",
            ) as nc:
                bucket = TokenBucket(
                    nc, bucket_name="rt-tokenbucket-concurrent", refill_rate=0.0001, capacity=float(capacity)
                )
                outcome = await bucket.claim("shared-key")
                return outcome.claimed

        results = await asyncio.gather(*(claim_from_one_pod(i) for i in range(num_pods)))

        assert sum(1 for claimed in results if claimed) == capacity


class TestDistributedCounterRoundTrip:
    """DistributedCounter against a real NATS broker, one connection per simulated pod."""

    async def test_increment_and_get_round_trip(self, nats_container: str) -> None:
        """An increment from one connection is visible via get() from another."""
        set_default_namespace("ratelimit-itest")
        async with await NatsClient.connect(
            nats_url=nats_container,
            nats_subject_namespace="ratelimit-itest",
            client_name="counter-writer",
        ) as writer_nc:
            writer = DistributedCounter(writer_nc, bucket_name="rt-counter")
            await writer.increment("k", delta=3)

        async with await NatsClient.connect(
            nats_url=nats_container,
            nats_subject_namespace="ratelimit-itest",
            client_name="counter-reader",
        ) as reader_nc:
            reader = DistributedCounter(reader_nc, bucket_name="rt-counter")
            assert await reader.get("k") == 3

    async def test_cross_pod_concurrent_increments_sum_exactly(self, nats_container: str) -> None:
        """N separate pod connections incrementing concurrently sum to exactly N, cluster-wide."""
        set_default_namespace("ratelimit-itest")
        num_pods = 25

        async def increment_from_one_pod(pod_index: int) -> None:
            async with await NatsClient.connect(
                nats_url=nats_container,
                nats_subject_namespace="ratelimit-itest",
                client_name=f"counter-pod-{pod_index}",
            ) as nc:
                counter = DistributedCounter(nc, bucket_name="rt-counter-concurrent")
                await counter.increment("shared-key")

        await asyncio.gather(*(increment_from_one_pod(i) for i in range(num_pods)))

        async with await NatsClient.connect(
            nats_url=nats_container,
            nats_subject_namespace="ratelimit-itest",
            client_name="counter-verifier",
        ) as verifier_nc:
            verifier = DistributedCounter(verifier_nc, bucket_name="rt-counter-concurrent")
            assert await verifier.get("shared-key") == num_pods

    async def test_cross_pod_concurrent_in_flight_pattern_stays_race_free(self, nats_container: str) -> None:
        """the increment-then-compensate-if-over-limit pattern, across real separate pod connections."""
        set_default_namespace("ratelimit-itest")
        limit = 6
        num_pods = 14

        async def try_acquire_slot_from_one_pod(pod_index: int) -> bool:
            async with await NatsClient.connect(
                nats_url=nats_container,
                nats_subject_namespace="ratelimit-itest",
                client_name=f"inflight-pod-{pod_index}",
            ) as nc:
                counter = DistributedCounter(nc, bucket_name="rt-counter-inflight")
                new_value = await counter.increment("inflight")
                if new_value > limit:
                    await counter.decrement("inflight")
                    return False
                return True

        results = await asyncio.gather(*(try_acquire_slot_from_one_pod(i) for i in range(num_pods)))

        assert sum(1 for acquired in results if acquired) == limit

        async with await NatsClient.connect(
            nats_url=nats_container,
            nats_subject_namespace="ratelimit-itest",
            client_name="inflight-verifier",
        ) as verifier_nc:
            verifier = DistributedCounter(verifier_nc, bucket_name="rt-counter-inflight")
            assert await verifier.get("inflight") == limit
