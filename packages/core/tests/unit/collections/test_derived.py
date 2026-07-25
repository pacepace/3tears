"""unit tests for :class:`DerivedCollection`.

covers the two properties the class exists to provide: quantization of a
continuous request onto a discrete key, and compute-on-miss that runs at most
once per key under concurrency. the durable tier is an in-memory dict, which
is a legitimate L3 per :meth:`BaseCollection.fetch_from_store`'s own contract
("the L3 backend is pluggable ... an in-memory backend reads a dict"), so no
database or NATS is required.

``nats_client=None`` throughout: :func:`nats_distributed_lock` no-ops on a
``None`` client by documented design, which isolates these tests to the
in-process gate. cross-pod exclusion is the lock's own tested behaviour, not
this class's.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from threetears.core.collections.derived import DerivedCollection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity


class _TileEntity(BaseEntity):
    primary_key_field = "z"


class _BucketCollection(DerivedCollection[_TileEntity]):
    """quantizes a float onto a fixed-width bucket grid.

    the simplest faithful instance of the pattern: the request is continuous,
    the key is not, and the value costs something to produce.
    """

    primary_key_column: tuple[str, ...] = ("bucket",)
    bucket_width: float = 10.0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.store: dict[tuple[Any, ...], dict[str, Any]] = {}
        self.compute_calls: list[tuple[Any, ...]] = []
        self.compute_delay: float = 0.0
        self.computable: bool = True

    @property
    def table_name(self) -> str:
        return "buckets"

    @property
    def entity_class(self) -> type[_TileEntity]:
        return _TileEntity

    def derive_key(self, request: Any) -> Any:
        return (int(float(request) // self.bucket_width),)

    async def load_derived(self, entity_id: Any) -> dict[str, Any] | None:
        return self.store.get(self.normalize_pk(entity_id))

    async def compute(self, entity_id: Any) -> dict[str, Any] | None:
        key = self.normalize_pk(entity_id)
        self.compute_calls.append(key)
        if self.compute_delay:
            await asyncio.sleep(self.compute_delay)
        if not self.computable:
            return None
        return {"bucket": key[0], "value": f"derived-{key[0]}"}

    async def save_to_store(self, data: dict[str, Any], original_timestamp: Any = None, *, conn: Any = None) -> int:
        self.store[(data["bucket"],)] = data
        return 1

    async def delete_from_store(self, entity_id: Any) -> None:
        self.store.pop(self.normalize_pk(entity_id), None)

    def serialize(self, data: dict[str, Any]) -> bytes:
        return repr(data).encode()

    def deserialize(self, data: bytes) -> dict[str, Any]:
        return dict(eval(data.decode()))  # noqa: S307 - test stub, input is our own repr


@pytest.fixture
def collection() -> _BucketCollection:
    registry = CollectionRegistry()
    registry.configure(l1_backend=None, l2_client=None, l3_pool=None)
    return _BucketCollection(registry, DefaultCoreConfig(), None, None)


class TestDeriveKey:
    """quantization is the contract that makes the cache shareable."""

    def test_requests_in_the_same_cell_produce_equal_keys(self, collection: _BucketCollection) -> None:
        # the whole reason the class exists: two different continuous requests
        # must collapse onto one key or nothing is ever shared between callers.
        assert collection.derive_key(11.0) == collection.derive_key(19.999)

    def test_requests_in_different_cells_produce_different_keys(self, collection: _BucketCollection) -> None:
        assert collection.derive_key(9.9) != collection.derive_key(10.1)

    def test_get_for_resolves_a_raw_request(self, collection: _BucketCollection) -> None:
        entity = asyncio.run(_get_for(collection, 42.5))
        assert entity is not None
        assert collection.store[(4,)]["value"] == "derived-4"


class TestComputeOnMiss:
    def test_durable_hit_does_not_compute(self, collection: _BucketCollection) -> None:
        collection.store[(4,)] = {"bucket": 4, "value": "preexisting"}
        row = asyncio.run(collection.fetch_from_store((4,)))
        assert row == {"bucket": 4, "value": "preexisting"}
        assert collection.compute_calls == []

    def test_durable_miss_computes_and_persists(self, collection: _BucketCollection) -> None:
        row = asyncio.run(collection.fetch_from_store((7,)))
        assert row is not None
        assert row["value"] == "derived-7"
        # persisted, so the next caller -- on any pod -- takes the cheap path
        assert collection.store[(7,)] == row
        assert collection.compute_calls == [(7,)]

    def test_underivable_key_returns_none_without_persisting(self, collection: _BucketCollection) -> None:
        # an out-of-range cell is a legitimate miss, not an error, and must not
        # be written as a phantom row.
        collection.computable = False
        assert asyncio.run(collection.fetch_from_store((99,))) is None
        assert (99,) not in collection.store


class TestSingleFlight:
    def test_concurrent_callers_compute_once(self, collection: _BucketCollection) -> None:
        """the stampede guard: derivation is the expensive step by definition."""
        collection.compute_delay = 0.05

        async def scenario() -> list[dict[str, Any] | None]:
            return list(await asyncio.gather(*(collection.fetch_from_store((3,)) for _ in range(8))))

        results = asyncio.run(scenario())
        assert collection.compute_calls == [(3,)]
        assert all(r is not None and r["value"] == "derived-3" for r in results)

    def test_distinct_keys_are_not_serialized_against_each_other(self, collection: _BucketCollection) -> None:
        # the gate is per key; two different cells must proceed in parallel.
        collection.compute_delay = 0.05

        async def scenario() -> None:
            await asyncio.gather(*(collection.fetch_from_store((n,)) for n in range(4)))

        started = asyncio.run(_elapsed(scenario))
        assert sorted(collection.compute_calls) == [(0,), (1,), (2,), (3,)]
        # serialized would be ~4x the delay; parallel is ~1x. generous bound so
        # this does not turn into a timing-flaky test on a loaded machine.
        assert started < 0.05 * 4

    def test_gate_bookkeeping_does_not_leak(self, collection: _BucketCollection) -> None:
        async def scenario() -> None:
            await asyncio.gather(*(collection.fetch_from_store((n,)) for n in range(5)))

        asyncio.run(scenario())
        # entries are dropped once the last waiter leaves, so a long-lived pod
        # serving many keys does not accumulate one lock per key ever seen.
        assert collection.inflight_derivations == 0


class TestBuildLockKey:
    def test_namespaced_by_table(self, collection: _BucketCollection) -> None:
        # two collections quantizing onto similar grids share one KV bucket, so
        # an unnamespaced key would let them block each other.
        assert collection.build_lock_key((4,)) == "buckets/4"


async def _get_for(collection: _BucketCollection, request: float) -> Any:
    return await collection.get_for(request)


async def _elapsed(fn: Any) -> float:
    loop = asyncio.get_running_loop()
    start = loop.time()
    await fn()
    return loop.time() - start
