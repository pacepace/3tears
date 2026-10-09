"""two replicas of one tool pod, over a real NATS: an answer is computed once and the other serves it.

The replicas share one key scope, so one L2 key per answer, and contend for one lease per key in a
bucket they bind (the hub declares it on the platform; here the first lease creates it).
"""

from __future__ import annotations

import asyncio
import gzip
from collections.abc import AsyncIterator

import pytest

from threetears.core.collections.derived import LeaseBuildLock
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.versioned_answers import VersionedAnswers
from threetears.core.config import DefaultCoreConfig
from threetears.core.coordination.lease import KVLease
from threetears.nats import NatsClient, set_default_namespace

pytestmark = pytest.mark.integration

_SCOPE = "tool_pod-answersit"


@pytest.fixture
async def replicas(nats_container: str) -> AsyncIterator[tuple[NatsClient, NatsClient]]:
    set_default_namespace("3tears")
    first = await NatsClient.connect(nats_url=nats_container, nats_subject_namespace="3tears", client_name="answers-a")
    second = await NatsClient.connect(nats_url=nats_container, nats_subject_namespace="3tears", client_name="answers-b")
    try:
        yield (first, second)
    finally:
        await first.shutdown()
        await second.shutdown()


def _answers(nats: NatsClient, pod_id: str) -> VersionedAnswers:
    registry = CollectionRegistry()
    registry.configure(l1_backend=None, l2_client=nats, l3_pool=None, kv_key_scope=_SCOPE)
    lease = KVLease(nats, bucket_name="answers-leases-it", pod_id=pod_id, key_scope=_SCOPE)
    return VersionedAnswers(
        registry, DefaultCoreConfig(), nats, table_name="answers_it", build_lock=LeaseBuildLock(lease, ttl_seconds=30)
    )


class TestTwoReplicas:
    async def test_racing_replicas_compute_once(self, replicas: tuple[NatsClient, NatsClient]) -> None:
        calls: list[str] = []

        def computer(name: str):  # noqa: ANN202 - a local factory
            async def compute() -> str:
                calls.append(name)
                await asyncio.sleep(0.4)
                return '{"answer": 42}'

            return compute

        a, b = _answers(replicas[0], "replica-a"), _answers(replicas[1], "replica-b")
        got = await asyncio.gather(
            a.answer("v1", "race", computer("a"), order=1), b.answer("v1", "race", computer("b"), order=1)
        )
        assert len(calls) == 1, calls
        assert {gzip.decompress(g).decode() for g in got} == {'{"answer": 42}'}

    async def test_a_replica_serves_what_the_other_computed(self, replicas: tuple[NatsClient, NatsClient]) -> None:
        a, b = _answers(replicas[0], "replica-a"), _answers(replicas[1], "replica-b")

        async def computed() -> str:
            return "computed by a"

        async def never() -> str:
            raise AssertionError("replica b computed an answer replica a already holds")

        await a.answer("v2", "served", computed, order=2)
        assert gzip.decompress(await b.answer("v2", "served", never, order=2)).decode() == "computed by a"

    async def test_retiring_on_one_replica_retires_for_both(self, replicas: tuple[NatsClient, NatsClient]) -> None:
        a, b = _answers(replicas[0], "replica-a"), _answers(replicas[1], "replica-b")

        async def old() -> str:
            return "old"

        await a.answer("v3", "kept-or-not", old, order=3)
        assert await b.retire_older_than(4) >= 1
        recomputed: list[str] = []

        async def again() -> str:
            recomputed.append("x")
            return "again"

        # v3 is below the floor now: answered, and never stored again
        await a.answer("v3", "kept-or-not", again, order=3)
        await a.answer("v3", "kept-or-not", again, order=3)
        assert recomputed == ["x", "x"]
