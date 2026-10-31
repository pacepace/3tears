"""Tables handed to a worker process through shared memory: no row pickled, none copied, nothing left behind.

The contracts. A job's input tables are written by the serving process into a hand-off of their own and only its name
crosses to the worker (a 100 MB table pickles in a few hundred bytes); the worker maps them read-only and reads
exactly what was handed, its Arrow pool allocating nothing; the hand-off goes when the worker acknowledges, whether
its job answered or its worker died, and not before a worker still mapping it lets go; it counts against the store's
bound; and the serving process's event loop is never held while a large table is handed over.
"""

from __future__ import annotations

import asyncio
import hashlib
import multiprocessing
import os
import pickle
import time
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from packages.core.tests.unit.shared_arrow import jobs
from threetears.core.shared_arrow import SharedArrowFullError, SharedArrowPool, SharedArrowStore

#: the large table's first column: 100 MiB of int64
BIG_BYTES = 100 << 20

#: the most the event loop may run late while a large table is handed over
LOOP_MAY_BE_LATE = 0.020


def _big() -> tuple[pa.Table, str]:
    """a 100 MiB table of random int64 values, a text column beside it, and the digest of its first column's values.

    :return: the table and the digest
    :rtype: tuple[pa.Table, str]
    """
    values = os.urandom(BIG_BYTES)
    rows = BIG_BYTES // 8
    column = pa.Array.from_buffers(pa.int64(), rows, [None, pa.py_buffer(values)])
    labels = pa.array(["x"] * 1000 + [None] * (rows - 1000), pa.string())
    return pa.table({"v": column, "s": labels}), hashlib.sha256(values).hexdigest()


@pytest.fixture(scope="module")
def big() -> tuple[pa.Table, str]:
    """the large table, made once.

    :return: the table and its digest
    :rtype: tuple[pa.Table, str]
    """
    return _big()


@pytest.fixture
def store(tmp_path: Path) -> SharedArrowStore:
    """a store with room for a couple of large hand-offs.

    :return: the store
    :rtype: SharedArrowStore
    """
    (tmp_path / "signals").mkdir()
    return SharedArrowStore(tmp_path / "shm", max_bytes=512 << 20)


@pytest.fixture
async def single(store: SharedArrowStore) -> AsyncIterator[SharedArrowPool]:
    """one worker over the store, spawned before the test, shut down after it.

    :return: the pool
    :rtype: SharedArrowPool
    """
    made = SharedArrowPool(store, workers=1, private_bytes=1 << 40, name="handoff")
    await made.run(jobs.process_id, read=False)
    yield made
    await made.close()


@pytest.fixture
def executor() -> Iterator[ProcessPoolExecutor]:
    """a plain spawned executor, as a pod that keeps its own pool runs one.

    :return: the executor
    :rtype: ProcessPoolExecutor
    """
    made = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
    made.submit(os.getpid).result()
    yield made
    made.shutdown(wait=True)


def _hidden(store: SharedArrowStore) -> list[str]:
    """every staging or retiring directory among the store's hand-offs.

    :return: their names
    :rtype: list[str]
    """
    return [entry.name for entry in (store.root / "handoffs").iterdir() if entry.name.startswith(".")]


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


class TestALargeTableCrossesWithoutAPickle:
    def test_a_100mb_hand_off_pickles_tiny_and_the_worker_reads_it_unchanged_in_place(
        self, store: SharedArrowStore, executor: ProcessPoolExecutor, big: tuple[pa.Table, str]
    ) -> None:
        table, digest = big
        with store.handing({"t": table}) as handed:
            assert handed.nbytes >= BIG_BYTES
            assert len(pickle.dumps(handed)) < 1024, "only the hand-off's name and size cross"
            seen = executor.submit(jobs.digest_of_handed, handed, "t").result()
        assert seen["rows"] == table.num_rows
        assert seen["sha256"] == digest
        assert seen["allocated"] < 1 << 20, "the worker read the mapped file's bytes, never a copy"
        assert store.handoffs() == () and _hidden(store) == []
        assert store.held_bytes() == 0

    async def test_the_pool_submits_a_few_hundred_bytes_for_a_100mb_table(
        self, store: SharedArrowStore, single: SharedArrowPool, big: tuple[pa.Table, str], monkeypatch: Any
    ) -> None:
        table, digest = big
        pickled: list[int] = []
        submit = ProcessPoolExecutor.submit

        def measured(self: ProcessPoolExecutor, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future[Any]:
            pickled.append(len(pickle.dumps((fn, args, kwargs))))
            return submit(self, fn, *args, **kwargs)

        monkeypatch.setattr(ProcessPoolExecutor, "submit", measured)
        result = await single.run(jobs.handed_digest, "t", read=False, tables={"t": table})
        assert result.value["sha256"] == digest
        assert result.value["allocated"] < 1 << 20
        assert pickled and max(pickled) < 4096, pickled
        await _until(lambda: store.handoffs() == (), "the hand-off released once the worker answered")
        assert store.status()["handoffs_open"] == 0


class TestTheHandOffGoesWhenTheWorkerIsDone:
    async def test_it_is_held_while_the_job_runs_and_released_when_it_answers(
        self, store: SharedArrowStore, single: SharedArrowPool, tmp_path: Path
    ) -> None:
        signals = tmp_path / "signals"
        running = asyncio.ensure_future(
            single.run(jobs.handed_on_signal, str(signals), read=False, tables={"t": pa.table({"a": [1, 2, 3]})})
        )
        await _until(lambda: (signals / "started").exists(), "the worker mapped the hand-off")
        assert len(store.handoffs()) == 1
        assert store.status()["handoffs_open"] == 1
        (signals / "go").touch()
        assert (await running).value == 3
        assert store.handoffs() == ()
        assert store.status()["handoffs_open"] == 0

    async def test_a_worker_that_dies_holding_it_leaves_nothing(
        self, store: SharedArrowStore, single: SharedArrowPool
    ) -> None:
        with pytest.raises(BrokenProcessPool):
            await single.run(jobs.die_holding_handed, read=False, tables={"t": pa.table({"a": [1, 2, 3]})})
        await _until(lambda: store.handoffs() == () and _hidden(store) == [], "the dead worker's hand-off removed")
        assert store.status()["handoffs_open"] == 0
        assert store.held_bytes() == 0
        rows = await single.run(jobs.handed_rows, read=False, tables={"t": pa.table({"a": [1]})})
        assert rows.value == 1, "a fresh pool"

    def test_a_released_hand_off_a_worker_still_maps_stays_until_it_lets_go(
        self, store: SharedArrowStore, executor: ProcessPoolExecutor, tmp_path: Path
    ) -> None:
        signals = tmp_path / "signals"
        handed = store.hand({"t": pa.table({"a": [1, 2]})})
        future = executor.submit(jobs.hold_handed_on_signal, handed, str(signals))
        deadline = time.monotonic() + 30
        while not (signals / "mapped").exists():
            assert time.monotonic() < deadline, "the worker never mapped the hand-off"
            time.sleep(0.01)
        store.release(handed)
        assert store.handoffs() == (handed.handoff_id,), "the worker's mapping holds it"
        (signals / "go").touch()
        assert future.result() == 2
        store.sweep()
        assert store.handoffs() == ()

    def test_a_hand_off_released_before_its_worker_maps_it_is_gone_to_the_worker(
        self, store: SharedArrowStore, executor: ProcessPoolExecutor
    ) -> None:
        handed = store.hand({"t": pa.table({"a": [1]})})
        store.release(handed)
        store.release(handed)  # harmless twice
        with pytest.raises(LookupError, match="gone"):
            executor.submit(jobs.digest_of_handed, handed, "t").result()

    def test_a_new_store_clears_a_hand_off_left_by_a_process_that_died(self, tmp_path: Path) -> None:
        first = SharedArrowStore(tmp_path / "shm", max_bytes=64 << 20)
        first.hand({"t": pa.table({"a": [1]})})
        assert len(first.handoffs()) == 1
        again = SharedArrowStore(tmp_path / "shm", max_bytes=64 << 20)
        assert again.handoffs() == ()


class TestASweepNeverRetiresAHandOffBeingRegistered:
    def test_a_sweep_between_the_rename_and_the_registration_keeps_the_hand_off(
        self, store: SharedArrowStore, executor: ProcessPoolExecutor, monkeypatch: Any
    ) -> None:
        rename = os.rename
        swept: list[list[str]] = []

        def rename_then_sweep(source: Any, target: Any) -> None:
            rename(source, target)
            # once, on the staging directory's rename into its final name (a sweep renames too)
            if Path(source).name.startswith(".staging-") and Path(target).parent.name == "handoffs" and not swept:
                # another thread's publish, release, claim or discard, landing in the window
                swept.append(store.sweep())

        monkeypatch.setattr("threetears.core.shared_arrow.os.rename", rename_then_sweep)
        handed = store.hand({"t": pa.table({"a": [1, 2, 3]})})
        assert swept, "the window was not staged"
        assert store.handoffs() == (handed.handoff_id,)
        assert executor.submit(jobs.digest_of_handed, handed, "t").result()["rows"] == 3
        store.release(handed)


class TestTheHandOffIsBounded:
    def test_a_hand_off_past_the_stores_room_is_refused_and_leaves_nothing(self, tmp_path: Path) -> None:
        small = SharedArrowStore(tmp_path / "shm", max_bytes=1 << 20)
        with pytest.raises(SharedArrowFullError):
            small.hand({"t": pa.table({"a": pa.array(range(200_000), pa.int64())})})
        assert small.handoffs() == () and _hidden(small) == []
        assert small.status()["refused"] == 1

    def test_a_hand_off_counts_against_the_store_until_it_goes(self, tmp_path: Path) -> None:
        small = SharedArrowStore(tmp_path / "shm", max_bytes=4 << 20)
        handed = small.hand({"t": pa.table({"a": pa.array(range(300_000), pa.int64())})})
        assert handed.nbytes >= 300_000 * 8
        with pytest.raises(SharedArrowFullError):
            small.reserve((4 << 20) - handed.nbytes + 1)
        small.release(handed)
        small.reserve((4 << 20) - 1).root  # noqa: B018 -- the room is back once the hand-off is gone

    def test_a_write_past_the_hand_offs_own_bound_is_refused(self, store: SharedArrowStore) -> None:
        with pytest.raises(SharedArrowFullError):
            store.hand({"t": pa.table({"a": pa.array(range(100_000), pa.int64())})}, max_bytes=64 << 10)
        assert store.handoffs() == () and _hidden(store) == []
        assert store.status()["handoffs_open"] == 0

    def test_a_reader_of_unknown_size_needs_its_bound_named(self, store: SharedArrowStore) -> None:
        reader = pa.RecordBatchReader.from_batches(pa.schema([("a", pa.int64())]), iter(()))
        with pytest.raises(ValueError, match="max_bytes"):
            store.hand({"t": reader})
        assert store.hand({"t": reader}, max_bytes=1 << 20).tables == ("t",)


@pytest.mark.xdist_group("loop_timing")
class TestTheLoopRunsWhileALargeTableIsHanded:
    async def test_a_ticker_is_never_late_while_100mb_is_handed_to_a_worker(
        self, store: SharedArrowStore, single: SharedArrowPool, big: tuple[pa.Table, str]
    ) -> None:
        table, digest = big
        late = 0.0
        ticking = True

        async def tick() -> None:
            nonlocal late
            while ticking:
                start = time.perf_counter()
                await asyncio.sleep(0.002)
                late = max(late, time.perf_counter() - start - 0.002)

        ticker = asyncio.create_task(tick())
        await asyncio.sleep(0.05)
        result = await single.run(jobs.handed_digest, "t", read=False, tables={"t": table})
        await asyncio.sleep(0.05)
        ticking = False
        await ticker
        assert result.value["sha256"] == digest
        assert late < LOOP_MAY_BE_LATE, f"the loop ran {late * 1000:.1f} ms late while the table was handed"
