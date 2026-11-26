"""A superseded version goes as soon as nothing reads it: idle workers move onto the current version at once.

The contracts. A version an idle worker keeps mapped for its next job is moved onto each newer version as it is
published -- the unchanged files taken over as they are, only the changed ones mapped -- and the superseded one is
let go by every process and retired, with no write short of room needed (Pace, 2026-11-26; before, it was kept until
the store ran short, e1c5cacb4). A busy worker keeps its job's version until the job ends, then moves on. A write that
arrives while idle workers are still letting a version go waits for that let-go rather than being refused; what is
referenced (a lease, a read in progress, a claimed result, an open slot, the current version) is never let go, and
only when the referenced set alone does not fit is the write refused. A reader in another process that races a
retirement gets the version whole or a clean miss, never part of it. And the store counts what it does, in the
shared-memory metric family: let-gos waited for room, the bytes they freed, refusals, the bytes held against the
budget and what of them is only cached, and its workers' kept-mapping hits and misses.
"""

from __future__ import annotations

import asyncio
import contextlib
import enum
import fcntl
import multiprocessing
import os
import subprocess
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
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
    Parts,
    SharedArrowStore,
    read_version,
    version_files,
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


@contextlib.contextmanager
def _moving(store: SharedArrowStore, version: str) -> Iterator[None]:
    """hold ``version``'s cached lock, as an idle worker does until it has moved its mapping onto the current version:
    a let-go in flight, held for the block. White-box on purpose: a worker's move takes a few milliseconds, too short
    to catch, so the test stands in for one that has not finished.

    :return: nothing
    :rtype: Iterator[None]
    """
    descriptor = os.open(store.root / "versions" / version / ".cached", os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        yield
    finally:
        os.close(descriptor)


def _mapped_paths(pid: int) -> set[str]:
    """the files process ``pid`` maps or holds open: Linux's ``/proc/<pid>/maps`` (a removed file named as it was),
    else ``lsof``.

    :return: the paths
    :rtype: set[str]
    """
    maps = Path(f"/proc/{pid}/maps")
    if maps.exists():
        found: set[str] = set()
        for line in maps.read_text().splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) == 6 and fields[5].startswith("/"):
                found.add(fields[5].removesuffix(" (deleted)"))
        return found
    listed = subprocess.run(["lsof", "-p", str(pid), "-Fn"], capture_output=True, text=True, check=False).stdout
    return {line[1:] for line in listed.splitlines() if line.startswith("n/")}


def _files(store: SharedArrowStore, version: str) -> set[str]:
    """the files ``version`` names, read while it is held: those a newer version does not name are what letting it
    go must unmap everywhere.

    :return: their paths, resolved
    :rtype: set[str]
    """
    return {str(each.path.resolve()) for each in version_files(store.root, version).values()}


class TestASupersededVersionIsLetGoAtOnce:
    def test_an_idle_readers_superseded_version_goes_with_no_write_short_of_room(
        self, tmp_path: Path, name: str
    ) -> None:
        size = _version_bytes(tmp_path)
        store = SharedArrowStore(tmp_path / "shm", max_bytes=int(size * 2.5), name=name)
        store.publish("v1", {TABLE: _table(1)})
        _parked(store, "v1")
        store.publish("v2", {TABLE: _table(2)})
        store.settle()
        assert store.versions() == ("v2",), "the idle reader moved onto v2 and let v1 go as v2 was published"
        assert store.status()["evictable_bytes"] == 0, "nothing superseded is held"
        store.discard(store.reserve(size))
        assert _sample("threetears_cache_evictions_total", name) == 0, "no write waited for room"
        assert _sample("threetears_cache_refusals_total", name) == 0
        # the reader moved on cleanly: its next read is of v2 (a pool's workers count it a hit, below)
        _parked(store, "v2")

    @pytest.mark.parametrize("write", ["reserve", "publish", "hand"])
    def test_a_write_short_of_room_waits_for_a_let_go_in_flight(self, tmp_path: Path, name: str, write: str) -> None:
        size = _version_bytes(tmp_path)
        store = SharedArrowStore(tmp_path / "shm", max_bytes=int(size * 2.5), name=name)
        store.publish("v1", {TABLE: _table(1)})
        with _moving(store, "v1"):
            store.publish("v2", {TABLE: _table(2)})
            store.sweep()
            assert set(store.versions()) == {"v1", "v2"}, "v1's idle worker has not moved on yet"
            assert store.status()["evictable_bytes"] >= size, "v1 is held only cached"
            done = threading.Event()

            def written() -> None:
                if write == "reserve":
                    store.discard(store.reserve(size))
                elif write == "publish":
                    store.publish("v3", {TABLE: _table(3)})
                else:
                    with store.handing({TABLE: _table(5)}):
                        pass
                done.set()

            writer = threading.Thread(target=written, daemon=True)
            writer.start()
            assert not done.wait(0.3), "the write waits for v1's let-go, never refused while it is in flight"
        writer.join(30)
        assert done.is_set()
        assert "v1" not in store.versions()
        assert _sample("threetears_cache_evictions_total", name) == 1, "one superseded version let go for room"
        assert _sample("threetears_cache_evicted_bytes_total", name) >= size
        assert _sample("threetears_cache_refusals_total", name) == 0


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
    made = [SharedArrowPool(store, workers=1, private_bytes=1 << 40, name=f"kept-{each}") for each in range(2)]
    yield store, made
    for each in made:
        await each.close()


#: a version in parts, and the part a change rewrites: the rest are the same files in the next version
PARTS = 8
PART_ROWS = 20_000


def _parts(value: int, changed: str | None = None) -> Parts:
    """the table in :data:`PARTS` parts, each of ``value``; with ``changed``, only that part, for a publish on a base.

    :return: the parts
    :rtype: Parts
    """
    names = [changed] if changed is not None else [f"p{index}" for index in range(PARTS)]
    return Parts(schema=_table(0, rows=0).schema, parts={part: _table(value, rows=PART_ROWS) for part in names})


class TestIdleWorkersMoveOntoTheCurrentVersion:
    async def test_every_process_unmaps_a_superseded_version_once_its_worker_is_idle(
        self, pools: tuple[SharedArrowStore, list[SharedArrowPool]]
    ) -> None:
        store, (first, second) = pools
        await asyncio.to_thread(store.publish, "v1", {TABLE: _parts(1)})
        one = (await first.run(jobs.mapped_identities)).value["pid"]
        v1_files = _files(store, "v1")
        await asyncio.to_thread(store.publish, "v2", {TABLE: _parts(2, changed="p3")}, base="v1")
        two = (await second.run(jobs.mapped_identities)).value["pid"]
        v2_files = _files(store, "v2")
        await asyncio.to_thread(store.settle)
        assert set(store.versions()) == {"v2"}, "the first worker, idle, moved off v1 as v2 was published"
        await asyncio.to_thread(store.publish, "v3", {TABLE: _parts(3, changed="p5")}, base="v2")
        v3_files = _files(store, "v3")
        only_v1, only_v2 = v1_files - v3_files, v2_files - v3_files
        await asyncio.to_thread(store.settle)
        assert store.versions() == ("v3",), "both idle workers moved onto v3 and let v1 and v2 go"
        assert store.status()["evictable_bytes"] == 0
        assert only_v1 and only_v2
        for pid in (one, two, os.getpid()):
            mapped = _mapped_paths(pid)
            assert not mapped & (only_v1 | only_v2), f"process {pid} still maps a superseded version's file"
        hits = _sample("threetears_cache_hits_total", store.name)
        await first.run(jobs.mapped_identities)
        await second.run(jobs.mapped_identities)
        assert _sample("threetears_cache_hits_total", store.name) == hits + 2, "each worker found v3 kept"

    async def test_unchanged_files_are_reused_not_mapped_again(
        self, pools: tuple[SharedArrowStore, list[SharedArrowPool]]
    ) -> None:
        store, (first, _) = pools
        await asyncio.to_thread(store.publish, "v1", {TABLE: _parts(1)})
        before = (await first.run(jobs.mapped_identities)).value["tables"]
        await asyncio.to_thread(store.publish, "v2", {TABLE: _parts(2, changed="p3")}, base="v1")
        await asyncio.to_thread(store.settle)
        assert store.versions() == ("v2",), "the idle worker moved onto v2 as it was published"
        hits = _sample("threetears_cache_hits_total", store.name)
        after = (await first.run(jobs.mapped_identities)).value["tables"]
        assert _sample("threetears_cache_hits_total", store.name) == hits + 1, "v2 was kept, moved on before the job"
        shared = before.keys() & after.keys()
        assert len(shared) == len(after) - 1, "every file but the changed part is the same file in both versions"
        assert all(after[identity] == before[identity] for identity in shared), "each unchanged file taken as it was"
        (changed,) = after.keys() - shared
        assert after[changed] not in before.values(), "only the changed part was mapped"

    async def test_a_busy_worker_moves_on_as_its_job_ends(
        self, pools: tuple[SharedArrowStore, list[SharedArrowPool]], tmp_path: Path
    ) -> None:
        store, (first, _) = pools
        signals = tmp_path / "signals"
        signals.mkdir()
        await asyncio.to_thread(store.publish, "v1", {TABLE: _parts(1)})
        reading = asyncio.ensure_future(first.run(jobs.identities_on_signal, str(signals)))
        await _until(lambda: (signals / "started").exists(), "the job mapped v1")
        await asyncio.to_thread(store.publish, "v2", {TABLE: _parts(2, changed="p3")}, base="v1")
        await asyncio.to_thread(store.settle)
        assert set(store.versions()) == {"v1", "v2"}, "the busy worker keeps v1 until its job ends"
        only_v1 = _files(store, "v1") - _files(store, "v2")
        (signals / "go").touch()
        done = (await reading).value
        await asyncio.to_thread(store.settle)
        assert store.versions() == ("v2",), "v1 went as the job ended"
        assert not _mapped_paths(done["pid"]) & only_v1, "the worker unmapped v1's own files"
        hits = _sample("threetears_cache_hits_total", store.name)
        after = (await first.run(jobs.mapped_identities)).value["tables"]
        assert _sample("threetears_cache_hits_total", store.name) == hits + 1, "it moved onto v2 as its job ended"
        assert len(after.keys() & done["tables"].keys()) == len(after) - 1
        assert all(after[each] == done["tables"][each] for each in after.keys() & done["tables"].keys())

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
        signals = tmp_path / "signals"
        signals.mkdir()
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=1, mp_context=context) as other:
            hammering = asyncio.wrap_future(
                other.submit(jobs.hammer_version, str(store.root), "v1", TABLE, "a", str(signals))
            )
            try:
                await _until(lambda: (signals / "started").exists(), "the reader started")
                await asyncio.to_thread(_parked, store, "v1")
                await asyncio.to_thread(store.publish, "v2", {TABLE: _table(2)})
                deadline = time.monotonic() + 30
                # the hammering reader holds v1 while each of its reads runs, and a sweep then leaves it; between two
                # reads a sweep retires it, and the reader's next try misses cleanly
                while "v1" in store.versions():
                    assert time.monotonic() < deadline, "no sweep found v1 between two reads"
                    await asyncio.to_thread(store.sweep)
                    await asyncio.sleep(0.005)
                await _until(lambda: (signals / "missed").exists(), "the reader missed the retired version")
            finally:
                # stopped whichever way the test went, so a failure never leaves the reader hammering
                (signals / "stop").touch()
            seen = await hammering
        await asyncio.to_thread(store.settle)
        assert seen["torn"] == [], seen
        assert seen["sums"] in ([ROWS], []), "every whole read summed v1's rows"
        assert seen["missed"] > 0, "after the let-go a read misses cleanly"
        assert store.versions() == ("v2",)
        assert os.listdir(store.root / "versions").count("v1") == 0


class TestTheStoresGaugesAreSet:
    def test_held_budget_and_cached_bytes(self, tmp_path: Path, name: str) -> None:
        size = _version_bytes(tmp_path)
        store = SharedArrowStore(tmp_path / "shm", max_bytes=int(size * 4), name=name)
        assert _sample("threetears_cache_budget_bytes", name) == int(size * 4)
        store.publish("v1", {TABLE: _table(1)})
        with _moving(store, "v1"):
            store.publish("v2", {TABLE: _table(2)})
            store.sweep()
            status = store.status()
            assert _sample("threetears_cache_held_bytes", name, policy="evictable") == status["evictable_bytes"] >= size
            assert _sample("threetears_cache_held_bytes", name, policy="keep") == status["kept_bytes"]
            assert status["kept_bytes"] + status["evictable_bytes"] == status["held_bytes"]
        store.settle()
        assert store.status()["evictable_bytes"] == 0


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
