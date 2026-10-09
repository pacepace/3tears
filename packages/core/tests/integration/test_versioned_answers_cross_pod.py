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
from threetears.nats.collection_key_requests import purge_scoped_keys

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


async def _subjects(nats: NatsClient, version: str) -> dict[str, int]:
    """every subject of ``version``'s answers and index left in the stream, with its message count."""
    js = nats.jetstream_context()
    info = await js.stream_info("KV_3tears-collections", subjects_filter=f"$KV.3tears-collections.{_SCOPE}.>")
    return {s: n for s, n in (info.state.subjects or {}).items() if f".{version}_" in s or f".{version}." in s}


class TestARetiredVersionLeavesNothing:
    async def test_deleting_alone_would_leave_a_marker_per_key(self, replicas: tuple[NatsClient, NatsClient]) -> None:
        a = _answers(replicas[0], "replica-a")

        async def old() -> str:
            return "old"

        await a.answer("v20", "marked", old, order=20)
        await a.retire_older_than(21)
        assert await _subjects(replicas[0], "v20") != {}  # no purger: the delete markers stay

    async def test_a_purged_version_leaves_zero_messages(self, replicas: tuple[NatsClient, NatsClient]) -> None:
        admin = replicas[1]

        async def purger(keys: list[str]) -> int:
            # the hub's half, as its responder runs it for this scope
            return await purge_scoped_keys(
                admin.jetstream_context(), bucket="3tears-collections", scope=_SCOPE, keys=keys
            )

        registry = CollectionRegistry()
        registry.configure(l1_backend=None, l2_client=replicas[0], l3_pool=None, kv_key_scope=_SCOPE)
        a = VersionedAnswers(registry, DefaultCoreConfig(), replicas[0], table_name="answers_it", purger=purger)

        async def old() -> str:
            return "old"

        for request in ("one", "two", "three"):
            await a.answer("v30", request, old, order=30)
        assert await _subjects(replicas[0], "v30") != {}
        await a.retire_older_than(31)
        assert await _subjects(replicas[0], "v30") == {}
