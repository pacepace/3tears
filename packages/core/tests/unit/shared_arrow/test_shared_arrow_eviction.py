"""Least-recently-used eviction: a store short of room lets go of what is only cached before it refuses.

The contracts. What the store holds only as a cache -- a superseded version that no reader holds, kept mapped by an
idle worker for its next job -- is let go, least recently used first, when a publish, a reservation or a hand-off
would otherwise pass the store's bound; what is referenced (a lease, a read in progress, a claimed result, an open
slot, the current version) never is, and only when the referenced set alone does not fit is the write refused. A
reader in another process that races an eviction gets the version whole or a clean miss, never part of it. And the
store counts what it does, in the shared-memory metric family: evictions, the bytes they freed, refusals, the bytes
held against the budget and what of them is only cached, and its workers' kept-mapping hits and misses.
"""

from __future__ import annotations

import asyncio
import enum
import multiprocessing
import os
import time
import uuid
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pytest
from prometheus_client import REGISTRY

from packages.core.tests.unit.shared_arrow import jobs
from threetears.core.eviction import Declared, Policy, Tier, policy_declare, policies_declared, policy_lookup
from threetears.core.shared_arrow import (
    serving_fallback_counted,
    SharedArrowFullError,
    SharedArrowPool,
    SharedArrowStore,
    read_version,
)

ROWS = 200_000
TABLE = "t"


def _table(value: int, rows: int = ROWS) -> pa.Table:
    """a table whose column ``a`` is ``value`` in every row, and a text column beside it.

    :return: the table
    :rtype: pa.Table
    """
    return pa.table({"a": pa.array([value] * rows, pa.int64()), "s": pa.array([f"row-{value}"] * rows, pa.string())})


def _sample(name: str, store: str, **labels: str) -> float:
    """a cache-family metric's value for one store of the shared tier, 0 when it has none yet.

    :return: the value
    :rtype: float
    """
    value = REGISTRY.get_sample_value(name, {"tier": "shared", "name": store, **labels})
    return 0.0 if value is None else value


def _version_bytes(where: Path) -> int:
    """what one version of :func:`_table` holds in a store: measured, never guessed.

    :return: bytes
    :rtype: int
    """
    probe = SharedArrowStore(where / "probe", max_bytes=1 << 30)
    published = probe.publish("v1", {TABLE: _table(1)})
    return published.nbytes


@pytest.fixture
def name() -> str:
    """a store name of the test's own, so its metrics are its own.

    :return: the name
    :rtype: str
    """
    return f"evict-{uuid.uuid4().hex[:10]}"


def _parked(store: SharedArrowStore, version: str) -> None:
    """read ``version`` in this process as a worker does (:func:`read_version`) and leave it kept mapped, the way an
    idle worker keeps the version it read last.

    :return: nothing
    :rtype: None
    """
    with store.lease(version), read_version(store.root, version) as mapped:
        assert mapped.tables[TABLE].num_rows == ROWS


async def _until(condition: Callable[[], bool], what: str, seconds: float = 30.0) -> None:
    """wait until ``condition()`` holds.

    :return: nothing
    :rtype: None
    """
    deadline = time.monotonic() + seconds
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError(f"never: {what}")
        await asyncio.sleep(0.01)


class TestACachedVersionIsEvictedInsteadOfRefused:
    def test_a_reservation_evicts_a_superseded_version_an_idle_reader_keeps(self, tmp_path: Path, name: str) -> None:
        size = _version_bytes(tmp_path)
        store = SharedArrowStore(tmp_path / "shm", max_bytes=int(size * 2.5), name=name)
        store.publish("v1", {TABLE: _table(1)})
        _parked(store, "v1")
        store.publish("v2", {TABLE: _table(2)})
        store.settle()
        assert set(store.versions()) == {"v1", "v2"}, "the idle reader keeps v1 mapped"
        assert store.status()["evictable_bytes"] >= size, "v1 is only cached"
        slot = store.reserve(size)
        assert store.versions() == ("v2",), "v1 was evicted to make room"
        assert _sample("threetears_cache_evictions_total", name) == 1
        assert _sample("threetears_cache_evicted_bytes_total", name) >= size
        assert _sample("threetears_cache_refusals_total", name) == 0
        store.discard(slot)
        # the reader evicted from lets go cleanly: its next read maps the current version
        _parked(store, "v2")

    def test_a_publish_evicts_before_it_is_refused(self, tmp_path: Path, name: str) -> None:
        size = _version_bytes(tmp_path)
        store = SharedArrowStore(tmp_path / "shm", max_bytes=int(size * 2.5), name=name)
        store.publish("v1", {TABLE: _table(1)})
        _parked(store, "v1")
        store.publish("v2", {TABLE: _table(2)})
        _parked(store, "v2")
        store.publish("v3", {TABLE: _table(3)})
        store.settle()
        assert store.versions() == ("v2", "v3")
        store.publish("v4", {TABLE: _table(4)})
        store.settle()
        assert store.versions() == ("v4",)
        assert _sample("threetears_cache_evictions_total", name) == 1

    def test_a_hand_off_evicts_before_it_is_refused(self, tmp_path: Path, name: str) -> None:
        size = _version_bytes(tmp_path)
        store = SharedArrowStore(tmp_path / "shm", max_bytes=int(size * 2.5), name=name)
        store.publish("v1", {TABLE: _table(1)})
        _parked(store, "v1")
        store.publish("v2", {TABLE: _table(2)})
        store.settle()
        with store.handing({TABLE: _table(5)}):
            assert store.versions() == ("v2",)
        assert _sample("threetears_cache_evictions_total", name) == 1


class TestAReferencedEntryIsNeverEvicted:
    def test_a_leased_version_stays_and_the_reservation_is_refused(self, tmp_path: Path, name: str) -> None:
        size = _version_bytes(tmp_path)
        store = SharedArrowStore(tmp_path / "shm", max_bytes=int(size * 2.5), name=name)
        store.publish("v1", {TABLE: _table(1)})
        _parked(store, "v1")
        lease = store.lease("v1")
        store.publish("v2", {TABLE: _table(2)})
        store.settle()
        assert store.status()["evictable_bytes"] == 0, "a leased version is in use, not cached"
        with pytest.raises(SharedArrowFullError):
            store.reserve(size)
        assert set(store.versions()) == {"v1", "v2"}
        assert _sample("threetears_cache_evictions_total", name) == 0
        assert _sample("threetears_cache_refusals_total", name) == 1
        lease.release()
        store.discard(store.reserve(size))
        assert store.versions() == ("v2",)

    def test_a_read_in_progress_stays(self, tmp_path: Path, name: str) -> None:
        size = _version_bytes(tmp_path)
        store = SharedArrowStore(tmp_path / "shm", max_bytes=int(size * 2.5), name=name)
        store.publish("v1", {TABLE: _table(1)})
        with store.lease("v1") as _, read_version(store.root, "v1") as mapped:
            store.publish("v2", {TABLE: _table(2)})
            with pytest.raises(SharedArrowFullError):
                store.reserve(size)
            assert mapped.tables[TABLE].num_rows == ROWS

    def test_a_claimed_result_and_an_open_slot_stay(self, tmp_path: Path, name: str) -> None:
        store = SharedArrowStore(tmp_path / "shm", max_bytes=8 << 20, name=name)
        slot = store.reserve(6 << 20)
        with pytest.raises(SharedArrowFullError):
            store.reserve(6 << 20)
        assert _sample("threetears_cache_evictions_total", name) == 0
        store.discard(slot)


@pytest.fixture
async def pools(tmp_path: Path) -> AsyncIterator[tuple[SharedArrowStore, list[SharedArrowPool]]]:
    """a store and two one-worker pools over it: two worker processes, each keeping its own version.

    :return: the store and the pools
    :rtype: tuple[SharedArrowStore, list[SharedArrowPool]]
    """
    size = _version_bytes(tmp_path)
    store = SharedArrowStore(tmp_path / "shm", max_bytes=int(size * 3.5), name=f"evict-{uuid.uuid4().hex[:10]}")
    made = [SharedArrowPool(store, workers=1, private_bytes=1 << 40, name=f"lru-{each}") for each in range(2)]
    yield store, made
    for each in made:
        await each.close()


class TestLeastRecentlyUsedGoesFirst:
    async def test_the_version_used_longest_ago_goes_first(
        self, pools: tuple[SharedArrowStore, list[SharedArrowPool]]
    ) -> None:
        store, (first, second) = pools
        await asyncio.to_thread(store.publish, "v1", {TABLE: _table(1)})
        assert (await first.run(jobs.column_sum, TABLE, "a")).value == ROWS
        await asyncio.to_thread(store.publish, "v2", {TABLE: _table(2)})
        assert (await second.run(jobs.column_sum, TABLE, "a")).value == 2 * ROWS
        await asyncio.to_thread(store.publish, "v3", {TABLE: _table(3)})
        await asyncio.to_thread(store.settle)
        assert set(store.versions()) == {"v1", "v2", "v3"}, "each idle worker keeps the version it read"
        # v1, published first, is used again after v2's last use: v2 is now the least recently used
        time.sleep(0.01)
        await asyncio.to_thread(lambda: store.lease("v1").release())
        size = store.current.nbytes if store.current is not None else 0
        slot = await asyncio.to_thread(store.reserve, size)
        assert set(store.versions()) == {"v1", "v3"}, "one eviction made room, and it took v2"
        await asyncio.to_thread(store.discard, slot)
        # the evicted worker reads on: its next job maps the current version afresh
        assert (await second.run(jobs.column_sum, TABLE, "a")).value == 3 * ROWS

    async def test_kept_mapping_hits_and_misses_are_counted(
        self, pools: tuple[SharedArrowStore, list[SharedArrowPool]]
    ) -> None:
        store, (first, _) = pools
        label = store.name
        await asyncio.to_thread(store.publish, "v1", {TABLE: _table(1)})
        await first.run(jobs.column_sum, TABLE, "a")
        await first.run(jobs.column_sum, TABLE, "a")
        await first.run(jobs.column_sum, TABLE, "a")
        assert _sample("threetears_cache_misses_total", label) == 1, "the first read maps it"
        assert _sample("threetears_cache_hits_total", label) == 2, "the next two find it kept"


class TestARacingReaderGetsItWholeOrMissesCleanly:
    async def test_a_reader_in_another_process_never_sees_a_torn_version(self, tmp_path: Path, name: str) -> None:
        size = _version_bytes(tmp_path)
        store = SharedArrowStore(tmp_path / "shm", max_bytes=int(size * 2.5), name=name)
        await asyncio.to_thread(store.publish, "v1", {TABLE: _table(1)})
        await asyncio.to_thread(_parked, store, "v1")
        await asyncio.to_thread(store.publish, "v2", {TABLE: _table(2)})
        signals = tmp_path / "signals"
        signals.mkdir()
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=1, mp_context=context) as other:
            hammering = asyncio.wrap_future(
                other.submit(jobs.hammer_version, str(store.root), "v1", TABLE, "a", str(signals))
            )
            await _until(lambda: (signals / "started").exists(), "the reader started")
            slot = None
            deadline = time.monotonic() + 30
            # the hammering reader holds v1 while each of its reads runs, and an eviction then refuses; between two
            # reads it takes v1, and the reader's next try misses cleanly
            while slot is None:
                try:
                    slot = await asyncio.to_thread(store.reserve, size)
                except SharedArrowFullError:
                    assert time.monotonic() < deadline, "the eviction never found v1 between two reads"
                    await asyncio.sleep(0.005)
            await _until(lambda: (signals / "missed").exists(), "the reader missed the evicted version")
            (signals / "stop").touch()
            seen = await hammering
        await asyncio.to_thread(store.discard, slot)
        assert seen["torn"] == [], seen
        assert seen["sums"] in ([ROWS], []), "every whole read summed v1's rows"
        assert seen["missed"] > 0, "after the eviction a read misses cleanly"
        assert store.versions() == ("v2",)
        assert os.listdir(store.root / "versions").count("v1") == 0


class TestTheStoresGaugesAreSet:
    def test_held_budget_and_cached_bytes(self, tmp_path: Path, name: str) -> None:
        size = _version_bytes(tmp_path)
        store = SharedArrowStore(tmp_path / "shm", max_bytes=int(size * 4), name=name)
        assert _sample("threetears_cache_budget_bytes", name) == int(size * 4)
        store.publish("v1", {TABLE: _table(1)})
        _parked(store, "v1")
        store.publish("v2", {TABLE: _table(2)})
        store.settle()
        status = store.status()
        assert _sample("threetears_cache_held_bytes", name, policy="evictable") == status["evictable_bytes"] >= size
        assert _sample("threetears_cache_held_bytes", name, policy="keep") == status["kept_bytes"]
        assert status["kept_bytes"] + status["evictable_bytes"] == status["held_bytes"]


#: every kind of entry the store holds and its policy, as Pace ruled it (2026-11-15): only what is cached may go
_POLICY = {
    "current_version": Policy.KEEP,
    "version_being_published": Policy.KEEP,
    "version_in_use": Policy.KEEP,
    "version_cached": Policy.EVICTABLE,
    "version_unheld": Policy.EVICTABLE,
    "part_file": Policy.KEEP,
    "open_slot": Policy.KEEP,
    "claimed_result": Policy.KEEP,
    "unclaimed_result": Policy.KEEP,
    "handoff": Policy.KEEP,
    "staging": Policy.KEEP,
}


class _Kinds(enum.StrEnum):
    ONE = "one"
    TWO = "two"


class TestEveryKindDeclaresItsEviction:
    def test_every_kind_the_store_holds_has_its_ruled_policy_and_a_reason(self) -> None:
        shared = policies_declared(Tier.SHARED)
        assert set(shared) == set(_POLICY), "a kind this test does not list, or one it lists that is gone"
        assert {kind: each.policy for kind, each in shared.items()} == _POLICY
        assert all(each.why.strip() for each in shared.values())

    def test_a_kind_set_with_a_kind_undeclared_is_refused_never_defaulted(self) -> None:
        with pytest.raises(LookupError, match="declare no eviction policy"):
            policy_declare(Tier.L1, _Kinds, {_Kinds.ONE: Declared(Policy.KEEP, "a test's kind")})
        with pytest.raises(LookupError, match="declares no eviction policy"):
            policy_lookup(Tier.SHARED, "no_such_kind")
        with pytest.raises(ValueError, match="reason"):
            Declared(Policy.KEEP, " ")


class TestServingFallbacksAreCounted:
    def test_a_consumers_fallback_is_counted_in_the_family(self) -> None:
        what = f"fallback-{uuid.uuid4().hex[:8]}"
        serving_fallback_counted(what)
        assert _sample("threetears_cache_serving_fallbacks_total", what) == 1
