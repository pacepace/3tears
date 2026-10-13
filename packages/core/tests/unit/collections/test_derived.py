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
import hashlib
import sys
import types
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

from threetears.core.collections.derived import BuildLockHeld, DerivedCollection, LeaseBuildLock
from threetears.core.coordination.lease import LeaseUnavailable
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


class TestWithoutTheNatsClient:
    """a single pod (no NATS client) derives without the optional NATS client installed."""

    def test_a_derivation_never_reaches_for_the_lock(
        self, collection: _BucketCollection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _NoClient(types.ModuleType):
            def __getattr__(self, name: str) -> Any:
                raise ImportError(f"{name} requires the NATS client, which is not installed")

        # an install without core's nats extra: every lock name raises on access
        monkeypatch.setitem(sys.modules, "threetears.nats", _NoClient("threetears.nats"))
        row = asyncio.run(collection.fetch_from_store((7,)))
        assert row is not None and row["value"] == "derived-7"
        assert collection.store[(7,)] == row


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


class _HeldLock:
    """a build lock another pod holds: every entry is refused."""

    def __init__(self) -> None:
        self.asked: list[str] = []

    @asynccontextmanager
    async def holding(self, key: str) -> AsyncIterator[None]:
        self.asked.append(key)
        raise BuildLockHeld(key)
        yield  # pragma: no cover - never reached; makes this a generator


class _FreeLock:
    """a build lock nobody else holds: records each hold and its release."""

    def __init__(self) -> None:
        self.events: list[str] = []

    @asynccontextmanager
    async def holding(self, key: str) -> AsyncIterator[None]:
        self.events.append(f"hold {key}")
        try:
            yield
        finally:
            self.events.append(f"release {key}")


def _with_lock(lock: Any) -> _BucketCollection:
    registry = CollectionRegistry()
    registry.configure(l1_backend=None, l2_client=None, l3_pool=None)
    return _BucketCollection(registry, DefaultCoreConfig(), None, None, build_lock=lock)


class TestTheBuildLockIsInjected:
    """a pod's collection takes a lock over a bucket the hub declared, not the bucket it may not declare."""

    def test_a_free_lock_is_held_around_the_derivation_and_released(self) -> None:
        lock = _FreeLock()
        collection = _with_lock(lock)
        assert asyncio.run(collection.fetch_from_store((3,))) == {"bucket": 3, "value": "derived-3"}
        assert lock.events == ["hold buckets/3", "release buckets/3"]

    def test_a_held_lock_waits_for_the_peers_value_and_does_not_derive(self) -> None:
        lock = _HeldLock()
        collection = _with_lock(lock)
        collection.peer_poll_interval = 0.01

        async def peer_lands() -> dict[str, Any] | None:
            fetched = asyncio.create_task(collection.fetch_from_store((4,)))
            await asyncio.sleep(0.03)
            collection.store[(4,)] = {"bucket": 4, "value": "from-peer"}
            return await fetched

        assert asyncio.run(peer_lands()) == {"bucket": 4, "value": "from-peer"}
        assert lock.asked == ["buckets/4"]
        assert collection.compute_calls == []

    def test_no_client_and_no_lock_derives_without_one(self, collection: _BucketCollection) -> None:
        assert asyncio.run(collection.fetch_from_store((5,))) == {"bucket": 5, "value": "derived-5"}


class _Handle:
    def __init__(self, released: list[str], name: str) -> None:
        self._released = released
        self._name = name

    async def __aenter__(self) -> _Handle:
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._released.append(self._name)


class _Lease:
    """KVLease's acquire: a held name is refused fail-fast, as max_wait_seconds=0 asks."""

    def __init__(self, held: set[str] | None = None) -> None:
        self.held = held or set()
        self.acquired: list[tuple[str, int, int]] = []
        self.released: list[str] = []

    async def acquire(self, key: str, ttl_seconds: int = 30, max_wait_seconds: int = 60) -> _Handle:
        if key in self.held:
            raise LeaseUnavailable(key)
        self.acquired.append((key, ttl_seconds, max_wait_seconds))
        return _Handle(self.released, key)


class TestTheLeaseBuildLock:
    def test_it_holds_the_lease_fail_fast_and_releases_it(self) -> None:
        lease = _Lease()

        async def hold() -> None:
            async with LeaseBuildLock(lease, ttl_seconds=45).holding("answers/ID_v1_ab"):  # type: ignore[arg-type]
                assert lease.released == []

        asyncio.run(hold())
        assert lease.acquired == [("answers/ID_v1_ab", 45, 0)]
        assert lease.released == ["answers/ID_v1_ab"]

    def test_a_held_lease_is_a_held_build_lock(self) -> None:
        lease = _Lease(held={"answers/x"})

        async def hold() -> None:
            async with LeaseBuildLock(lease).holding("answers/x"):  # type: ignore[arg-type]
                pytest.fail("entered a lock another pod holds")

        with pytest.raises(BuildLockHeld):
            asyncio.run(hold())

    def test_a_key_outside_the_kv_grammar_is_hashed(self) -> None:
        lease = _Lease()

        async def hold() -> None:
            async with LeaseBuildLock(lease).holding("answers/a b:c"):  # type: ignore[arg-type]
                pass

        asyncio.run(hold())
        assert lease.acquired[0][0] == hashlib.sha256(b"answers/a b:c").hexdigest()
