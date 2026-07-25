"""cross-pod integration test for :class:`DerivedCollection`.

the unit tests pass ``nats_client=None``, which makes
:func:`nats_distributed_lock` a documented no-op -- so they prove the
in-process gate and prove nothing at all about the cross-pod one. that gate
is the entire justification for the class: without it, N pods missing the
same key on a cold cache each run an expensive derivation, which is the
stampede the design exists to prevent.

so this exercises it against a real NATS: two independently constructed
collections, standing in for two pods, sharing one durable store and one
JetStream KV, racing on the same key. exactly one derivation must run.

requires docker; gated by ``pytest.mark.integration``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from threetears.core.collections.derived import DerivedCollection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.nats import NatsClient, set_default_namespace

pytestmark = pytest.mark.integration


class _Entity(BaseEntity):
    primary_key_field = "bucket"


class _SharedStore:
    """durable tier shared by both pods, standing in for the object store.

    also counts derivations, which is what the test actually asserts on.
    """

    def __init__(self) -> None:
        self.rows: dict[tuple[Any, ...], dict[str, Any]] = {}
        self.derivations: list[tuple[Any, ...]] = []


class _PodCollection(DerivedCollection[_Entity]):
    """one pod's view of a derived collection over ``store``."""

    primary_key_column: tuple[str, ...] = ("bucket",)
    # a real derivation is slow -- that is why it is cached. the delay makes
    # the race deterministic instead of relying on scheduler luck.
    derive_delay = 0.4

    def __init__(self, *args: Any, store: _SharedStore, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._store = store

    @property
    def table_name(self) -> str:
        return "derived_buckets"

    @property
    def entity_class(self) -> type[_Entity]:
        return _Entity

    def derive_key(self, request: Any) -> Any:
        return (int(float(request) // 10),)

    async def load_derived(self, entity_id: Any) -> dict[str, Any] | None:
        return self._store.rows.get(self.normalize_pk(entity_id))

    async def compute(self, entity_id: Any) -> dict[str, Any] | None:
        key = self.normalize_pk(entity_id)
        self._store.derivations.append(key)
        await asyncio.sleep(self.derive_delay)
        return {"bucket": key[0], "value": f"derived-{key[0]}"}

    async def save_to_store(self, data: dict[str, Any], original_timestamp: Any = None, *, conn: Any = None) -> int:
        self._store.rows[(data["bucket"],)] = data
        return 1

    async def delete_from_store(self, entity_id: Any) -> None:
        self._store.rows.pop(self.normalize_pk(entity_id), None)

    def serialize(self, data: dict[str, Any]) -> bytes:
        import json

        return json.dumps(data).encode()

    def deserialize(self, data: bytes) -> dict[str, Any]:
        import json

        result: dict[str, Any] = json.loads(data)
        return result


@pytest.fixture
async def nats_clients(nats_container: str) -> AsyncIterator[tuple[NatsClient, NatsClient]]:
    """two independently connected NATS clients: one per simulated pod."""
    set_default_namespace("3tears")
    first = await NatsClient.connect(
        nats_url=nats_container,
        nats_subject_namespace="3tears",
        client_name="derived-pod-a",
    )
    second = await NatsClient.connect(
        nats_url=nats_container,
        nats_subject_namespace="3tears",
        client_name="derived-pod-b",
    )
    try:
        yield (first, second)
    finally:
        await first.shutdown()
        await second.shutdown()


def _make_pod(store: _SharedStore, nats_client: NatsClient) -> _PodCollection:
    registry = CollectionRegistry()
    registry.configure(l1_backend=None, l2_client=None, l3_pool=None)
    return _PodCollection(registry, DefaultCoreConfig(), nats_client, None, store=store)


class TestCrossPodSingleFlight:
    async def test_two_pods_racing_one_key_derive_once(
        self,
        nats_clients: tuple[NatsClient, NatsClient],
    ) -> None:
        """the property the class exists for.

        both pods miss the same cold key simultaneously. one wins the
        JetStream KV lock and derives; the other must observe the winner's
        result rather than duplicating an expensive computation.
        """
        store = _SharedStore()
        pod_a = _make_pod(store, nats_clients[0])
        pod_b = _make_pod(store, nats_clients[1])

        results = await asyncio.gather(
            pod_a.fetch_from_store((7,)),
            pod_b.fetch_from_store((7,)),
        )

        assert store.derivations == [(7,)], f"expected exactly one derivation, got {store.derivations}"
        assert all(row is not None and row["value"] == "derived-7" for row in results)

    async def test_loser_returns_the_winners_value_not_none(
        self,
        nats_clients: tuple[NatsClient, NatsClient],
    ) -> None:
        """a pod that loses the lock must not report the key as absent.

        returning ``None`` here would surface to a caller as "nothing to
        render", which on a map is indistinguishable from empty geography.
        """
        store = _SharedStore()
        pod_a = _make_pod(store, nats_clients[0])
        pod_b = _make_pod(store, nats_clients[1])

        first, second = await asyncio.gather(
            pod_a.fetch_from_store((3,)),
            pod_b.fetch_from_store((3,)),
        )
        assert first == second
        assert first is not None

    async def test_sequential_second_pod_hits_the_durable_tier(
        self,
        nats_clients: tuple[NatsClient, NatsClient],
    ) -> None:
        """once a value is durable, a second pod never derives it at all."""
        store = _SharedStore()
        pod_a = _make_pod(store, nats_clients[0])
        await pod_a.fetch_from_store((11,))
        assert len(store.derivations) == 1

        pod_b = _make_pod(store, nats_clients[1])
        row = await pod_b.fetch_from_store((11,))
        assert row is not None
        assert len(store.derivations) == 1, "second pod re-derived an already-durable value"

    async def test_distinct_keys_are_not_blocked_by_each_other(
        self,
        nats_clients: tuple[NatsClient, NatsClient],
    ) -> None:
        """the lock is per key: two pods on different keys both proceed.

        a lock keyed only by table would serialize every derivation platform
        wide, which trades a stampede for a queue.
        """
        store = _SharedStore()
        pod_a = _make_pod(store, nats_clients[0])
        pod_b = _make_pod(store, nats_clients[1])

        loop = asyncio.get_running_loop()
        started = loop.time()
        await asyncio.gather(pod_a.fetch_from_store((1,)), pod_b.fetch_from_store((2,)))
        elapsed = loop.time() - started

        assert sorted(store.derivations) == [(1,), (2,)]
        # serialized would be ~2x the delay; concurrent is ~1x.
        assert elapsed < _PodCollection.derive_delay * 2
