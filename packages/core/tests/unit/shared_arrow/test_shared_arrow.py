"""Shared-memory Arrow: one copy of a version read by many processes, and results handed back with no pickle.

The contracts. Down: several worker processes map ONE copy of a version (on Linux each process is charged a fraction
of the mapped file); a reader mid-read keeps its version across a swap and reads it whole; a version is retired only
after every reader -- the serving process's lease and a worker's mapping -- lets it go. Up: a worker's result tables
are mapped by the serving process with no copy; a result nobody claims (the job failed, the caller stopped waiting,
the worker died mid-write) leaves nothing behind. Both: the directory holds no more than its bound.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import gc
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures.process import BrokenProcessPool, ProcessPoolExecutor
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow as pa
import pytest

from packages.core.tests.unit.shared_arrow import jobs
from threetears.core.shared_arrow import (
    MappedVersion,
    Parts,
    SharedArrowFullError,
    SharedArrowPool,
    SharedArrowStore,
    epochs_version,
    read_version,
    version_files,
)

ROWS = 200_000
TABLE = "t"

#: the threads a DuckDB point query runs on, for the speed ratio's CPU trust check (coordinator ruling)
_POINT_QUERY_THREADS = 2


def _table(value: int, rows: int = ROWS) -> pa.Table:
    """a table whose column ``a`` is ``value`` in every row, and a text column beside it.

    built from buffers, never a Python object a row: two 200k-item lists converted on the loop held it 0.351 s on the
    pool (the loop-stall guard)

    :return: the table
    :rtype: pa.Table
    """
    text = f"row-{value}".encode()
    offsets = np.arange(0, (rows + 1) * len(text), len(text), dtype=np.int32)
    strings = pa.StringArray.from_buffers(rows, pa.py_buffer(offsets), pa.py_buffer(text * rows))
    return pa.table({"a": pa.array(np.full(rows, value, dtype=np.int64)), "s": strings})


@pytest.fixture
def where(tmp_path: Path) -> Path:
    """the test's directory: ``shm`` the store's root, ``signals`` the jobs' signal files.

    :return: the directory
    :rtype: Path
    """
    (tmp_path / "signals").mkdir()
    return tmp_path


@pytest.fixture
def store(where: Path) -> SharedArrowStore:
    """a store with room for a handful of versions.

    :return: the store
    :rtype: SharedArrowStore
    """
    return SharedArrowStore(where / "shm", max_bytes=64 << 20)


@pytest.fixture
async def pool(store: SharedArrowStore) -> AsyncIterator[SharedArrowPool]:
    """three workers over the store, shut down after the test.

    :return: the pool
    :rtype: SharedArrowPool
    """
    made = SharedArrowPool(store, workers=3, private_bytes=1 << 40, name="test")
    yield made
    await made.close()


@pytest.fixture
async def single(store: SharedArrowStore) -> AsyncIterator[SharedArrowPool]:
    """one worker over the store, shut down after the test.

    :return: the pool
    :rtype: SharedArrowPool
    """
    made = SharedArrowPool(store, workers=1, private_bytes=1 << 40, name="single")
    yield made
    await made.close()


async def _until(
    condition: Callable[[], bool], what: str, seconds: float | None = 30.0, *, unless: asyncio.Future[Any] | None = None
) -> None:
    """wait until ``condition()`` holds, or until ``unless`` (a run the condition waits on) ends first: then the wait
    ends with what the run raised, or says it returned.

    :param seconds: the most to wait; None to wait on the condition and the run alone, with no clock: a run that
        hangs is caught by the test's timeout (pyproject's ``timeout``), which prints every thread's stack
    :ptype seconds: float | None
    :param unless: a run whose end means the condition never will hold
    :ptype unless: asyncio.Future[Any] | None
    :return: nothing
    :rtype: None
    """
    deadline = None if seconds is None else time.monotonic() + seconds
    while not condition():
        if unless is not None and unless.done():
            if unless.cancelled():
                raise AssertionError(f"never: {what} (the run was cancelled first)")
            unless.result()  # what the run raised, as it raised it
            raise AssertionError(f"never: {what} (the run returned {unless.result()!r} first)")
        if deadline is not None and time.monotonic() > deadline:
            raise AssertionError(f"never: {what}")
        await asyncio.sleep(0.01)


def _hidden(store: SharedArrowStore) -> list[str]:
    """every staging or retiring directory the store's root holds.

    :return: their names
    :rtype: list[str]
    """
    return [
        entry.name
        for half in ("versions", "segments")
        for entry in (store.root / half).iterdir()
        if entry.name.startswith(".")
    ]


@pytest.fixture
def published(store: SharedArrowStore) -> SharedArrowStore:
    """the store with ``v1`` published: :data:`ROWS` rows of 3, made and written off the loop (a sync fixture).

    :return: the store
    :rtype: SharedArrowStore
    """
    store.publish("v1", {TABLE: _table(3)})
    return store


class TestManyProcessesShareOneCopy:
    async def test_three_workers_read_one_mapping(
        self, published: SharedArrowStore, pool: SharedArrowPool, where: Path
    ) -> None:
        store = published
        path = str(version_files(store.root, "v1")[f"{TABLE}.arrow"].path)
        signals = str(where / "signals")
        answers = [
            result.value
            for result in await asyncio.gather(
                *(pool.run(jobs.shared_read, TABLE, "a", signals, 3, path) for _ in range(3))
            )
        ]
        assert len({answer["pid"] for answer in answers}) == 3, "each read ran in a process of its own"
        assert {answer["sum"] for answer in answers} == {3 * ROWS}
        if sys.platform.startswith("linux"):
            # every process has the column it read resident (int64: 8 bytes a row), and is charged a share of it:
            # one physical copy
            for answer in answers:
                assert answer["rss"] >= ROWS * 8 * 0.9
                assert answer["pss"] <= answer["rss"] / 2

    async def test_duckdb_reads_the_mapped_tables(self, store: SharedArrowStore, pool: SharedArrowPool) -> None:
        store.publish("v1", {TABLE: _table(5), "other": _table(7, rows=10)})
        assert (await pool.run(jobs.duckdb_sum, TABLE, "a")).value == 5 * ROWS
        assert (await pool.run(jobs.duckdb_sum, "other", "a")).value == 70

    def test_a_mapped_table_is_not_copied_into_the_process(self, store: SharedArrowStore) -> None:
        store.publish("v1", {TABLE: _table(1, rows=2_000_000)})
        before = pa.total_allocated_bytes()
        mapped = MappedVersion(store.root, "v1")
        try:
            table = mapped.tables[TABLE]
            assert table.num_rows == 2_000_000
            assert pa.total_allocated_bytes() - before < table.nbytes / 10
        finally:
            del table
            mapped.close()


class TestAVersionSwapsUnderAReader:
    async def test_a_reader_mid_read_finishes_on_its_version(
        self, store: SharedArrowStore, pool: SharedArrowPool, where: Path
    ) -> None:
        signals = where / "signals"
        store.publish("v1", {TABLE: _table(1)})
        reading = asyncio.ensure_future(pool.run(jobs.read_on_signal, TABLE, "a", str(signals)))
        await _until(lambda: (signals / "started").exists(), "the reader mapped v1")
        store.publish("v2", {TABLE: _table(2)})
        assert store.current is not None and store.current.name == "v2"
        assert set(store.versions()) == {"v1", "v2"}, "v1 is kept while it is read"
        (signals / "go").touch()
        assert (await reading).value == 1 * ROWS, "the reader read v1 whole, never v2"
        assert store.versions() == ("v2",), "v1 retired once its reader let it go"
        assert (await pool.run(jobs.column_sum, TABLE, "a")).value == 2 * ROWS

    async def test_a_cancelled_wait_keeps_the_version_until_the_worker_is_done(
        self, store: SharedArrowStore, pool: SharedArrowPool, where: Path
    ) -> None:
        signals = where / "signals"
        store.publish("v1", {TABLE: _table(1)})
        reading = asyncio.ensure_future(pool.run(jobs.read_on_signal, TABLE, "a", str(signals)))
        await _until(lambda: (signals / "started").exists(), "the reader mapped v1")
        reading.cancel()
        store.publish("v2", {TABLE: _table(2)})
        assert "v1" in store.versions(), "the worker still reads v1"
        (signals / "go").touch()
        await _until(lambda: store.versions() == ("v2",), "v1 retired after the worker finished")


class TestRetirementWaitsForEveryReader:
    def test_a_lease_holds_its_version(self, store: SharedArrowStore) -> None:
        store.publish("v1", {TABLE: _table(1)})
        lease = store.lease()
        store.publish("v2", {TABLE: _table(2)})
        store.publish("v3", {TABLE: _table(3)})
        store.settle()
        assert store.versions() == ("v1", "v3"), "v2 went once the publish's sweep ran; v1 is leased"
        lease.release()
        assert store.versions() == ("v3",)
        lease.release()  # harmless twice

    def test_a_named_lease_reads_a_held_version_and_refuses_a_gone_one(self, store: SharedArrowStore) -> None:
        store.publish("v1", {TABLE: _table(1)})
        with store.lease("v1"):
            store.publish("v2", {TABLE: _table(2)})
            with store.lease("v1") as again:
                assert again.name == "v1"
        with pytest.raises(LookupError):
            store.lease("v1")

    def test_nothing_published_is_nothing_to_lease(self, store: SharedArrowStore) -> None:
        with pytest.raises(LookupError):
            store.lease()

    def test_a_worker_side_read_keeps_its_mapping_only_while_it_is_current(self, store: SharedArrowStore) -> None:
        store.publish("v1", {TABLE: _table(1)})
        with store.lease(), read_version(store.root, "v1") as mapped:
            assert mapped.tables[TABLE].num_rows == ROWS
        store.settle()
        assert _held(store, "v1"), "the read kept v1 mapped while it was current"
        store.publish("v2", {TABLE: _table(2)})
        store.settle()
        assert store.versions() == ("v2",), "the kept mapping moved onto v2 as it was published, and let v1 go"
        assert _held(store, "v2"), "kept, now v2"
        with store.lease(), read_version(store.root, "v2") as mapped:
            assert mapped.name == "v2"
        assert store.versions() == ("v2",)

    async def test_an_idle_worker_moves_its_mapping_onto_each_version_published(
        self, store: SharedArrowStore, single: SharedArrowPool
    ) -> None:
        # until 2026-11-26 an idle worker held the version it read last until its next job; now it moves on at once
        store.publish("v1", {TABLE: _table(1)})
        assert (await single.run(jobs.column_sum, TABLE, "a")).value == ROWS
        await asyncio.to_thread(store.settle)
        assert _held(store, "v1"), "the idle worker keeps v1 while it is current"
        await asyncio.to_thread(store.publish, "v2", {TABLE: _table(2)})
        await asyncio.to_thread(store.settle)
        assert store.versions() == ("v2",), "the idle worker moved onto v2 and let v1 go, before any next job"
        assert _held(store, "v2"), "it keeps v2 for its next job"
        assert (await single.run(jobs.column_sum, TABLE, "a")).value == 2 * ROWS
        assert store.versions() == ("v2",)

    async def test_a_worker_that_exits_lets_go(self, store: SharedArrowStore) -> None:
        one = SharedArrowPool(store, workers=1, private_bytes=1 << 40, name="one")
        store.publish("v1", {TABLE: _table(1)})
        await one.run(jobs.column_sum, TABLE, "a")
        await asyncio.to_thread(store.publish, "v2", {TABLE: _table(2)})
        await asyncio.to_thread(store.settle)
        assert store.versions() == ("v2",)
        assert _held(store, "v2"), "the idle worker keeps the version it moved onto"
        await one.close()
        await asyncio.to_thread(store.settle)
        assert not _held(store, "v2"), "its exit let it go"

    def test_a_new_store_clears_what_nobody_holds(self, where: Path) -> None:
        first = SharedArrowStore(where / "shm", max_bytes=64 << 20)
        first.publish("v1", {TABLE: _table(1)})
        first.publish("v2", {TABLE: _table(2)})
        first.settle()
        orphan = first.reserve(1 << 20)
        orphan.write({TABLE: _table(1, rows=10)})
        held = os.open(where / "shm" / "versions" / "v2" / ".lock", os.O_RDONLY)
        fcntl.flock(held, fcntl.LOCK_SH)
        (where / "shm" / "versions" / ".staging-v3-dead").mkdir()
        try:
            second = SharedArrowStore(where / "shm", max_bytes=64 << 20)
            assert second.current is None
            assert second.versions() == ("v2",), "only the version another reader holds is kept"
            assert second.segments() == (), "a result no slot of this store owns goes"
            assert _hidden(second) == []
            assert not (where / "shm" / "versions" / "CURRENT").exists()
        finally:
            os.close(held)
        second.sweep()
        assert second.versions() == ()


class TestResultsComeBackMapped:
    async def test_a_workers_tables_are_mapped_here(self, store: SharedArrowStore, single: SharedArrowPool) -> None:
        result = await single.run(jobs.make_rows, 2_000_000, 4, read=False, result_bytes=48 << 20)
        assert result.value == "written"
        assert result.segment is not None
        before = pa.total_allocated_bytes()
        big = result.segment.tables["big"]
        assert big.num_rows == 2_000_000
        assert big.column("a")[0].as_py() == 4
        assert result.segment.tables["small"].column("n")[0].as_py() == 2_000_000
        assert pa.total_allocated_bytes() - before < big.nbytes / 10, "mapped, not copied"
        assert store.segments() == (result.segment.name,)
        del big
        result.segment.close()
        assert store.segments() == ()
        result.segment.close()  # harmless twice

    async def test_a_job_reads_a_version_and_hands_back_what_it_derived(
        self, store: SharedArrowStore, single: SharedArrowPool
    ) -> None:
        store.publish("v1", {TABLE: pa.table({"a": list(range(1000))})})
        result = await single.run(jobs.filter_version, TABLE, "a", 10, result_bytes=1 << 20)
        assert result.value == 10
        assert result.segment is not None
        with result.segment as segment:
            assert segment.tables["kept"].column("a").to_pylist() == list(range(10))
        assert store.segments() == ()

    async def test_a_job_given_a_slot_may_write_nothing(self, store: SharedArrowStore, single: SharedArrowPool) -> None:
        result = await single.run(jobs.process_id, read=False, result_bytes=1 << 20)
        assert result.segment is None
        assert store.status()["slots_open"] == 0

    async def test_a_result_from_a_job_that_raised_is_discarded(
        self, store: SharedArrowStore, single: SharedArrowPool
    ) -> None:
        with pytest.raises(ValueError, match="after writing"):
            await single.run(jobs.write_then_fail, read=False, result_bytes=1 << 20)
        await _until(lambda: store.segments() == () and store.status()["slots_open"] == 0, "the result discarded")

    async def test_a_failed_jobs_result_is_discarded_off_the_event_loop(
        self, store: SharedArrowStore, single: SharedArrowPool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """a discard sweeps the directory (walks, flocks, rmtree): the loop the pool exists to keep free never runs it."""
        threads: list[threading.Thread] = []
        discard = store.discard

        def watched(slot: Any) -> None:
            threads.append(threading.current_thread())
            discard(slot)

        monkeypatch.setattr(store, "discard", watched)
        with pytest.raises(ValueError, match="after writing"):
            await single.run(jobs.write_then_fail, read=False, result_bytes=1 << 20)
        await _until(lambda: bool(threads), "the result discarded")
        assert threads[0] is not threading.main_thread()

    async def test_a_discard_that_fails_off_the_loop_is_logged(
        self,
        store: SharedArrowStore,
        single: SharedArrowPool,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """the discard runs on a thread nothing awaits: what it raises is logged, not left to a warning at collection."""

        def failing(slot: Any) -> None:
            raise OSError("the slot's directory would not go")

        monkeypatch.setattr(store, "discard", failing)
        caplog.set_level(logging.WARNING)
        with pytest.raises(ValueError, match="after writing"):
            await single.run(jobs.write_then_fail, read=False, result_bytes=1 << 20)
        await _until(lambda: "result could not be discarded" in caplog.text, "the failed discard logged")

    async def test_a_worker_that_dies_mid_write_leaves_nothing(
        self, store: SharedArrowStore, single: SharedArrowPool
    ) -> None:
        with pytest.raises(BrokenProcessPool):
            await single.run(jobs.die_mid_write, read=False, result_bytes=1 << 20)
        await _until(lambda: store.segments() == () and _hidden(store) == [], "the dead writer's staging removed")
        assert store.status()["slots_open"] == 0
        assert (await single.run(jobs.make_rows, 10, 1, read=False, result_bytes=1 << 20)).segment is not None, (
            "a fresh pool"
        )

    async def test_an_abandoned_result_goes_once_the_worker_is_done(
        self, store: SharedArrowStore, single: SharedArrowPool, where: Path
    ) -> None:
        signals = where / "signals"
        running = asyncio.ensure_future(
            single.run(jobs.write_on_signal, str(signals), read=False, result_bytes=1 << 20)
        )
        # waited on the worker's own signal, or the run's end, never a clock: a 30 s wait here once ran out on
        # brooks-5950x (2026-11-15) at load1 12 of 32 cores, 0.18 cores of outside load, so not a starved host (a spawn
        # to this signal takes 0.7-1.3 s idle, 1.4-2.4 s beside 4x as many busy processes as cores). A run that failed
        # ends the wait with its own error; one that hangs meets the test's timeout, with every thread's stack
        await _until(lambda: (signals / "started").exists(), "the worker started", seconds=None, unless=running)
        running.cancel()
        (signals / "go").touch()
        await _until(lambda: store.status()["slots_open"] == 0, "the abandoned slot discarded")
        # the discard closes the slot, then sweeps: its staging is renamed retiring and removed after the lock
        await _until(lambda: store.segments() == () and _hidden(store) == [], "the abandoned slot's staging removed")
        assert store.status()["slots_open"] == 0

    def test_a_slot_is_written_once_and_claimed_once(self, store: SharedArrowStore) -> None:
        slot = store.reserve(1 << 20)
        slot.write({TABLE: _table(1, rows=10)})
        with pytest.raises(FileExistsError):
            slot.write({TABLE: _table(2, rows=10)})
        claimed = store.claim(slot)
        assert claimed is not None
        with pytest.raises(KeyError):
            store.claim(slot)
        claimed.close()
        store.discard(slot)  # harmless after a claim


class TestTheDirectoryIsBounded:
    def test_a_publish_past_the_bound_is_refused_and_leaves_nothing(self, where: Path) -> None:
        one = _table(1)
        store = SharedArrowStore(where / "shm", max_bytes=int(one.nbytes * 2.5))
        store.publish("v1", {TABLE: one})
        lease = store.lease()
        store.publish("v2", {TABLE: _table(2)})
        with pytest.raises(SharedArrowFullError):
            store.publish("v3", {TABLE: _table(3)})
        assert store.versions() == ("v1", "v2")
        assert store.current is not None and store.current.name == "v2"
        assert _hidden(store) == []
        assert store.refused == 1
        lease.release()
        store.publish("v3", {TABLE: _table(3)})
        store.settle()
        assert store.versions() == ("v3",)

    def test_many_versions_hold_only_the_current_one(self, store: SharedArrowStore) -> None:
        for value in range(10):
            last = store.publish(f"v{value}", {TABLE: _table(value)})
        store.settle()
        assert store.versions() == ("v9",)
        assert store.held_bytes() == last.nbytes
        assert store.retired == 9

    def test_an_open_slot_counts_at_its_most(self, where: Path) -> None:
        store = SharedArrowStore(where / "shm", max_bytes=10 << 20)
        first = store.reserve(6 << 20)
        with pytest.raises(SharedArrowFullError):
            store.reserve(6 << 20)
        with pytest.raises(SharedArrowFullError):
            store.publish("v1", {TABLE: _table(1, rows=400_000)})  # about 6.8 MB, beside the open 6 MB
        store.discard(first)
        store.reserve(6 << 20)

    async def test_a_worker_writing_past_its_slot_is_refused(
        self, store: SharedArrowStore, single: SharedArrowPool
    ) -> None:
        with pytest.raises(SharedArrowFullError):
            await single.run(jobs.make_rows, 1_000_000, 1, read=False, result_bytes=1 << 20)
        assert store.segments() == ()
        assert _hidden(store) == []


class TestPublishing:
    def test_a_version_already_present_is_made_current_as_it_is(self, store: SharedArrowStore) -> None:
        first = store.publish("v1", {TABLE: _table(1)})
        with store.lease("v1"):
            store.publish("v2", {TABLE: _table(2)})
            again = store.publish("v1", {TABLE: _table(9)})
        assert again == first
        assert store.current == first
        assert store.versions() == ("v1",), "v2 retired once v1 was current again"
        with MappedVersion(store.root, "v1") as mapped:
            assert mapped.tables[TABLE].column("a")[0].as_py() == 1, "the name already present kept its rows"

    def test_a_duckdb_answer_is_published_a_batch_at_a_time(self, store: SharedArrowStore) -> None:
        source = duckdb.connect(":memory:")
        source.execute("CREATE TABLE live AS SELECT range AS a, 'x' || range AS s FROM range(300000)")
        version = store.publish_query(
            "v1", source, {"live": "SELECT * FROM live", "few": "SELECT 1 AS a"}, batch_rows=10_000
        )
        assert version.tables == ("few", "live")
        with MappedVersion(store.root, "v1") as mapped:
            connection = mapped.connect()
            assert connection.execute("SELECT count(*), sum(a) FROM live").fetchone() == (300000, 299999 * 300000 // 2)
            connection.close()

    def test_a_table_given_as_a_callable_is_opened_only_when_written(self, store: SharedArrowStore) -> None:
        opened: list[str] = []

        def table(name: str, value: int) -> Callable[[], pa.Table]:
            def made() -> pa.Table:
                opened.append(name)
                return _table(value, rows=10)

            return made

        store.publish("v1", {"x": table("x", 1), "y": table("y", 2)})
        assert opened == ["x", "y"]
        with store.lease("v1"):
            store.publish("v2", {"x": _table(3, rows=10)})
            store.publish("v1", {"x": table("again", 9)})
        assert opened == ["x", "y"], "a version present already opens nothing"
        with MappedVersion(store.root, "v1") as mapped:
            assert mapped.tables["y"].column("a")[0].as_py() == 2

    def test_a_sweep_between_the_rename_and_current_keeps_the_new_version(self, where: Path) -> None:
        class SweptMidPublish(SharedArrowStore):
            """a store whose sweep runs right after a version is renamed into place, as a lease released on another
            thread would run it."""

            def _write_version(self, name: str, tables: Any, base: Path | None = None) -> None:
                super()._write_version(name, tables, base)
                self.sweep()

        swept = SweptMidPublish(where / "shm", max_bytes=64 << 20)
        swept.publish("v1", {TABLE: _table(1, rows=10)})
        swept.publish("v2", {TABLE: _table(2, rows=10)})
        swept.settle()
        assert swept.versions() == ("v2",)

    @pytest.mark.parametrize("name", ["", ".hidden", "../up", "a/b", "CURRENT", "x" * 129])
    def test_a_name_is_one_plain_path_segment(self, store: SharedArrowStore, name: str) -> None:
        with pytest.raises(ValueError):
            store.publish(name, {TABLE: _table(1, rows=1)})
        with pytest.raises(ValueError):
            store.publish("v1", {name: _table(1, rows=1)})

    def test_epochs_name_a_version(self) -> None:
        assert epochs_version({"VA": 3, "GA": 7}) == epochs_version({"GA": 7, "VA": 3})
        assert epochs_version({"VA": 3, "GA": 7}) != epochs_version({"VA": 4, "GA": 7})


class TestThePool:
    async def test_a_worker_past_its_private_bound_is_replaced(self, store: SharedArrowStore) -> None:
        bounded = SharedArrowPool(store, workers=1, private_bytes=1, name="bounded")
        try:
            first = (await bounded.run(jobs.process_id, read=False)).value
            second = (await bounded.run(jobs.process_id, read=False)).value
            assert first != second
            assert bounded.replaced == 1
        finally:
            await bounded.close()

    async def test_a_worker_under_its_bound_is_kept(self, store: SharedArrowStore, pool: SharedArrowPool) -> None:
        seen = {(await pool.run(jobs.process_id, read=False)).value for _ in range(9)}
        assert len(seen) <= 3
        assert pool.replaced == 0

    async def test_a_job_that_raises_raises_here_and_releases_its_version(
        self, store: SharedArrowStore, pool: SharedArrowPool
    ) -> None:
        store.publish("v1", {TABLE: _table(1)})
        with pytest.raises(ValueError, match="no good"):
            await pool.run(jobs.fail, "no good")
        store.publish("v2", {TABLE: _table(2)})
        assert (await pool.run(jobs.column_sum, TABLE, "a")).value == 2 * ROWS

    async def test_nothing_published_runs_nothing_that_reads(self, pool: SharedArrowPool) -> None:
        with pytest.raises(LookupError):
            await pool.run(jobs.process_id)


def _parts(values: dict[str, int], rows: int = 1_000) -> Parts:
    """a table in parts, each part's ``a`` its value.

    :return: the parts
    :rtype: Parts
    """
    return Parts(schema=_table(0, rows=0).schema, parts={part: _table(value, rows) for part, value in values.items()})


def _inode(store: SharedArrowStore, version: str, file_name: str) -> int:
    return version_files(store.root, version)[file_name].path.stat().st_ino


class TestPartsAreSharedAcrossVersions:
    def test_a_version_rewrites_only_the_parts_that_changed(self, store: SharedArrowStore) -> None:
        first = store.publish("v1", {TABLE: _parts({"VA": 1, "GA": 2, "TX": 3}), "whole": _table(5, rows=10)})
        with store.lease("v1"):
            second = store.publish(
                "v2",
                {TABLE: Parts(schema=_table(0, rows=0).schema, parts={"GA": _table(20, 1_000), "TX": None})},
                base="v1",
            )
            assert _inode(store, "v1", "t@VA.arrow") == _inode(store, "v2", "t@VA.arrow"), "VA is the same file"
            assert _inode(store, "v1", "t@GA.arrow") != _inode(store, "v2", "t@GA.arrow")
            assert "t@TX.arrow" not in version_files(store.root, "v2"), "TX was dropped"
            assert _inode(store, "v1", "whole.arrow") == _inode(store, "v2", "whole.arrow"), "a table not named is kept"
            # the shared files counted once: v1 whole, plus what v2 holds that v1 does not
            v1_files, v2_files = version_files(store.root, "v1"), version_files(store.root, "v2")
            shared = sum(
                held.nbytes for name, held in v2_files.items() if name in v1_files and v1_files[name].path == held.path
            )
            assert store.held_bytes() == first.nbytes + second.nbytes - shared
        with store.lease("v2"), read_version(store.root, "v2") as mapped:
            assert mapped.tables[TABLE].num_rows == 2_000
            connection = mapped.connect()
            try:
                assert connection.execute(f'SELECT sum(a) FROM "{TABLE}"').fetchone() == ((1 + 20) * 1_000,)
                assert connection.execute('SELECT sum(a) FROM "whole"').fetchone() == (50,)
            finally:
                connection.close()
        assert store.versions() == ("v2",)

    def test_shared_files_are_counted_once_against_the_bound(self, where: Path) -> None:
        # room for one version and a part, never two versions whole
        one = _parts({f"S{index}": index for index in range(8)}, rows=20_000)
        probe = SharedArrowStore(where / "probe", max_bytes=1 << 30).publish("p", {TABLE: one})
        bounded = SharedArrowStore(where / "bounded", max_bytes=int(probe.nbytes * 1.3))
        bounded.publish("v1", {TABLE: one})
        with bounded.lease("v1"):
            changed = Parts(schema=one.schema, parts={"S0": _table(100, 20_000)})
            bounded.publish("v2", {TABLE: changed}, base="v1")
            assert bounded.held_bytes() < probe.nbytes * 1.3
            with pytest.raises(SharedArrowFullError):
                bounded.publish("v3", {TABLE: one})

    async def test_a_worker_reads_a_table_in_parts(self, store: SharedArrowStore, pool: SharedArrowPool) -> None:
        store.publish("v1", {TABLE: _parts({"VA": 1, "GA": 2})})
        assert (await pool.run(jobs.duckdb_sum, TABLE, "a")).value == 3 * 1_000

    def test_a_table_of_no_parts_has_its_columns_and_no_rows(self, store: SharedArrowStore) -> None:
        store.publish("v1", {TABLE: Parts(schema=_table(0, rows=0).schema, parts={})})
        with MappedVersion(store.root, "v1") as mapped:
            assert mapped.tables[TABLE].num_rows == 0
            assert mapped.tables[TABLE].schema.names == ["a", "s"]

    def test_a_part_with_other_columns_is_refused_and_nothing_kept(self, store: SharedArrowStore) -> None:
        wrong = Parts(schema=_table(0, rows=0).schema, parts={"VA": pa.table({"b": [1]})})
        with pytest.raises(ValueError, match="columns"):
            store.publish("v1", {TABLE: wrong})
        assert store.versions() == ()
        assert _hidden(store) == []

    def test_a_base_whose_columns_differ_is_refused(self, store: SharedArrowStore) -> None:
        store.publish("v1", {TABLE: _parts({"VA": 1})})
        other = pa.schema([("a", pa.int64())])
        with pytest.raises(ValueError, match="changed its columns"):
            store.publish("v2", {TABLE: Parts(schema=other, parts={"GA": pa.table({"a": [1]})})}, base="v1")
        assert store.versions() == ("v1",)

    def test_a_base_not_held_is_refused(self, store: SharedArrowStore) -> None:
        with pytest.raises(LookupError):
            store.publish("v1", {TABLE: _parts({"VA": 1})}, base="v0")
        assert store.versions() == ()

    @pytest.mark.parametrize("part", ["", ".x", "a/b", "a@b"])
    def test_a_part_name_is_one_plain_path_segment(self, store: SharedArrowStore, part: str) -> None:
        with pytest.raises(ValueError):
            store.publish("v1", {TABLE: Parts(schema=_table(0, rows=0).schema, parts={part: _table(1, rows=1)})})


@pytest.fixture
def mapped_files(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """the name of each file this process memory-maps during the test."""
    names: list[str] = []
    original = pa.memory_map

    def recording(path: str, *args: Any, **kwargs: Any) -> Any:
        names.append(Path(path).name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pa, "memory_map", recording)
    return names


class TestANewVersionMapsOnlyWhatChanged:
    """a version made from another holds the other's unchanged files as hard links: a worker's kept mapping takes those
    from the version it had mapped and maps only the files that changed."""

    def test_a_workers_kept_version_lends_the_next_its_files(
        self, store: SharedArrowStore, mapped_files: list[str]
    ) -> None:
        store.publish("v1", {TABLE: _parts({f"S{index}": index for index in range(10)})})
        with store.lease("v1"), read_version(store.root, "v1") as mapped:
            assert mapped.tables[TABLE].num_rows == 10_000
        store.publish("v2", {TABLE: Parts(schema=_table(0, rows=0).schema, parts={"S0": _table(7, 1_000)})}, base="v1")
        mapped_files.clear()
        with store.lease("v2"), read_version(store.root, "v2") as mapped:
            column = mapped.tables[TABLE].column("a")
            assert pa.compute.sum(column).as_py() == (sum(range(10)) + 7) * 1_000
        assert sorted(mapped_files) == ["t@S0.arrow"], "only the changed files were mapped"


class TestNothingLeaksWhenTheCallerStopsWaiting:
    async def test_a_cancel_while_the_job_is_submitted_discards_its_slot(
        self, store: SharedArrowStore, single: SharedArrowPool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reserving, go, reserved = threading.Event(), threading.Event(), threading.Event()
        reserve = store.reserve

        def slow_reserve(max_bytes: int) -> Any:
            reserving.set()
            go.wait(30)
            try:
                return reserve(max_bytes)
            finally:
                reserved.set()

        monkeypatch.setattr(store, "reserve", slow_reserve)
        running = asyncio.ensure_future(single.run(jobs.process_id, read=False, result_bytes=1 << 20))
        await _until(reserving.is_set, "the submission is reserving its slot")
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        go.set()
        await _until(reserved.is_set, "the abandoned submission reserved its slot")
        await _until(lambda: store.status()["slots_open"] == 0, "the slot of the abandoned submission discarded", 10)
        assert store.segments() == ()

    async def test_a_cancel_while_the_result_is_claimed_closes_it(
        self, store: SharedArrowStore, single: SharedArrowPool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        claiming, go, claimed = threading.Event(), threading.Event(), threading.Event()
        claim = store.claim

        def slow_claim(slot: Any) -> Any:
            # mapped, the slot the claim's own, before the caller hears of it
            mapped = claim(slot)
            claiming.set()
            go.wait(30)
            claimed.set()
            return mapped

        monkeypatch.setattr(store, "claim", slow_claim)
        running = asyncio.ensure_future(single.run(jobs.make_rows, 10, 1, read=False, result_bytes=1 << 20))
        await _until(claiming.is_set, "the result is being claimed")
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        go.set()
        await _until(claimed.is_set, "the abandoned claim mapped the result")
        await _until(
            lambda: store.status()["segments_claimed"] == 0 and store.segments() == (),
            "the result nobody received closed and gone",
            10,
        )


class TestTheBoundHoldsUnderConcurrentWriters:
    def test_a_slot_reserved_while_a_version_is_written_takes_its_room_from_that_write(self, where: Path) -> None:
        bounded = SharedArrowStore(where / "bounded", max_bytes=10 << 20)
        schema = pa.schema([("a", pa.int64())])
        batch = pa.record_batch([pa.array(range(256 * 1024), pa.int64())], schema=schema)  # 2 MiB
        slots: list[Any] = []

        def batches() -> Iterator[pa.RecordBatch]:
            yield batch
            # a compute reserving its result while the publish writes (another thread, in a pod)
            slots.append(bounded.reserve(6 << 20))
            yield from [batch] * 3

        with pytest.raises(SharedArrowFullError):
            bounded.publish("v1", {TABLE: pa.RecordBatchReader.from_batches(schema, batches())})
        assert len(slots) == 1, "the slot was granted: the publish had written only 2 MiB"
        assert bounded.held_bytes() + slots[0].max_bytes <= bounded.max_bytes
        assert bounded.versions() == ()

    def test_a_slot_claimed_while_a_version_is_written_counts_its_result_against_that_write(self, where: Path) -> None:
        bounded = SharedArrowStore(where / "bounded", max_bytes=10 << 20)
        slot = bounded.reserve(6 << 20)
        slot.write({TABLE: pa.table({"a": pa.array(range(640 * 1024), pa.int64())})})  # 5 MiB
        schema = pa.schema([("a", pa.int64())])
        batch = pa.record_batch([pa.array(range(256 * 1024), pa.int64())], schema=schema)  # 2 MiB
        claimed: list[Any] = []

        def batches() -> Iterator[pa.RecordBatch]:
            yield batch
            # the serving process claims the result while the publish writes: its slot stops counting at 6 MiB,
            # and its 5 MiB of files are still held
            claimed.append(bounded.claim(slot))
            yield from [batch] * 2

        try:
            with pytest.raises(SharedArrowFullError):
                bounded.publish("v1", {TABLE: pa.RecordBatchReader.from_batches(schema, batches())})
            assert claimed and claimed[0] is not None
            assert bounded.held_bytes() <= bounded.max_bytes
            assert bounded.versions() == ()
        finally:
            for segment in claimed:
                if segment is not None:
                    segment.close()

    def test_a_refused_slot_is_logged(self, where: Path, caplog: pytest.LogCaptureFixture) -> None:
        small = SharedArrowStore(where / "small", max_bytes=1 << 20)
        caplog.set_level(logging.WARNING)
        with pytest.raises(SharedArrowFullError):
            small.reserve(2 << 20)
        assert "result slot refused" in caplog.text


class TestTheSweepRemovesOutsideTheLock:
    def test_retired_files_are_removed_with_the_state_lock_free(
        self, store: SharedArrowStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store.publish("v1", {TABLE: _table(1, rows=10)})
        held: list[bool] = []
        rmtree = shutil.rmtree

        def watched(path: Any, *args: Any, **kwargs: Any) -> None:
            # another thread asks the store what it holds, which takes the store's lock: it waits while the sweep
            # removing these files holds it
            asking = threading.Thread(target=store.status)
            asking.start()
            asking.join(0.5)
            held.append(asking.is_alive())
            askers.append(asking)
            rmtree(path, *args, **kwargs)

        askers: list[threading.Thread] = []
        monkeypatch.setattr(shutil, "rmtree", watched)
        store.publish("v2", {TABLE: _table(2, rows=10)})
        store.settle()
        for asking in askers:
            asking.join()
        assert store.versions() == ("v2",)
        assert held, "v1's files were removed"
        assert not any(held), "a sweep removed files holding the store's lock"


class TestAWorkersPartsAreChecked:
    def test_a_slots_part_name_is_one_plain_path_segment(self, store: SharedArrowStore) -> None:
        slot = store.reserve(1 << 20)
        schema = _table(0, rows=0).schema
        with pytest.raises(ValueError):
            slot.write({TABLE: Parts(schema=schema, parts={"../out": _table(1, rows=10)})})
        assert not (store.root / f"{TABLE}@..").exists()
        assert not any(path.name.startswith(f"{TABLE}@") for path in store.root.iterdir())
        store.discard(slot)


class TestAWorkerDeathIsSaid:
    async def test_a_dead_worker_is_logged_once(
        self, store: SharedArrowStore, single: SharedArrowPool, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.WARNING)
        with pytest.raises(BrokenProcessPool):
            await single.run(jobs.die_mid_write, read=False, result_bytes=1 << 20)
        assert caplog.text.count("worker died") == 1

    async def test_closing_while_the_dead_workers_pool_is_reaped_shuts_it_down_once_at_a_time(
        self, single: SharedArrowPool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # the reaper and close() both shut the broken executor down; two at once closed its result pipe twice
        # (OSError: [Errno 9] Bad file descriptor in close(), seen on the pool's runners)
        shutdown = ProcessPoolExecutor.shutdown
        in_flight: set[int] = set()
        overlapped: list[int] = []
        guard = threading.Lock()

        def watched(executor: ProcessPoolExecutor, wait: bool = True, *, cancel_futures: bool = False) -> None:
            if not wait:
                shutdown(executor, wait=wait, cancel_futures=cancel_futures)
                return
            with guard:
                if id(executor) in in_flight:
                    overlapped.append(id(executor))
                in_flight.add(id(executor))
            try:
                time.sleep(0.3)  # widen the window two unserialised shutdowns would share
                shutdown(executor, wait=wait, cancel_futures=cancel_futures)
            finally:
                with guard:
                    in_flight.discard(id(executor))

        monkeypatch.setattr(ProcessPoolExecutor, "shutdown", watched)
        with pytest.raises(BrokenProcessPool):
            await single.run(jobs.die_mid_write, read=False, result_bytes=1 << 20)

        await single.close()
        await _until(
            lambda: not any(t.name == "shared-arrow-reap-single" and t.is_alive() for t in threading.enumerate()),
            "the reaper finishing",
        )

        assert overlapped == []


def _reaped(pid: int) -> bool:
    """whether process ``pid`` is gone, its exit collected: a dead worker the executor has joined.

    :return: whether it is
    :rtype: bool
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def _stopped(pid: int) -> bool:
    """whether process ``pid`` is stopped (``SIGSTOP``): Linux's ``/proc/<pid>/stat``, else ``ps``.

    :return: whether it is
    :rtype: bool
    """
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists():
        # the state follows the command's closing parenthesis (the command may hold spaces or parentheses itself)
        return stat.read_text().rpartition(")")[2].split()[0] == "T"
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, check=False).stdout
    return state.strip().startswith("T")


@contextlib.contextmanager
def _frozen(pid: int) -> Iterator[None]:
    """worker ``pid`` stopped (``SIGSTOP``) for the block, continued after.

    An idle worker moves its kept mapping onto each version the store publishes, a few milliseconds after the publish
    (``_KeptVersions``); stopped, it cannot, so the version it read last stays its own, by construction, while it is
    stopped: the state a worker that died before it moved on leaves. Unstopped, a test that asserts the worker still
    holds the superseded version races that move, and loses it on a loaded host (2026-11-26: 3 ms on the pool).

    :return: nothing
    :rtype: Iterator[None]
    """
    os.kill(pid, signal.SIGSTOP)
    try:
        while not _stopped(pid):
            time.sleep(0.001)
        yield
    finally:
        # a worker killed in the block is gone or not yet joined: continuing it is a no-op either way
        with contextlib.suppress(ProcessLookupError):  # NOSILENT: already joined, which is what was wanted
            os.kill(pid, signal.SIGCONT)


def _held(store: SharedArrowStore, version: str) -> bool:
    """whether a reader holds ``version``: its read lock or its cached lock (a kept mapping) taken by anyone. Ask only
    while no sweep runs (after :meth:`SharedArrowStore.settle`): a sweep beside the exclusive try would take it for a
    reader.

    :return: whether it is held
    :rtype: bool
    """
    directory = store.root / "versions" / version
    for name in (".lock", ".cached"):
        descriptor = os.open(directory / name, os.O_RDONLY)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        finally:
            os.close(descriptor)
    return False


class TestADeadWorkersPoolIsReplaced:
    async def test_a_worker_dead_between_jobs_has_its_pool_replaced_and_what_it_held_swept(
        self, store: SharedArrowStore, single: SharedArrowPool, caplog: pytest.LogCaptureFixture
    ) -> None:
        # no job is pending when the worker dies, so no job's future fails: the executor refuses the next submission.
        # It dies stopped, before it could move off v1 (``_frozen``), so v1 is what it held and what must be swept
        caplog.set_level(logging.WARNING)
        store.publish("v1", {TABLE: _table(1)})
        dead = (await single.run(jobs.process_id)).value
        with _frozen(dead):
            store.publish("v2", {TABLE: _table(2)})
            assert set(store.versions()) == {"v1", "v2"}, "the idle worker still maps v1"
            os.kill(dead, signal.SIGKILL)
        await _until(lambda: _reaped(dead), "the executor joining its dead worker")

        assert (await single.run(jobs.column_sum, TABLE, "a")).value == 2 * ROWS, "the job ran on a fresh pool"
        assert single.replaced == 1
        assert caplog.text.count("worker died") == 1
        await _until(lambda: store.versions() == ("v2",), "the dead worker's v1 swept")

    async def test_a_broken_pool_is_shut_down_by_its_reaper_alone(
        self, single: SharedArrowPool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # the reaper's waiting shutdown, and a replacement's shutdown of the same executor without waiting, could
        # interleave: the second clears the executor's queues and thread out from under the first
        shutdown = ProcessPoolExecutor.shutdown
        waited: list[int] = []
        not_waited: list[int] = []

        def watched(executor: ProcessPoolExecutor, wait: bool = True, *, cancel_futures: bool = False) -> None:
            (waited if wait else not_waited).append(id(executor))
            shutdown(executor, wait=wait, cancel_futures=cancel_futures)

        monkeypatch.setattr(ProcessPoolExecutor, "shutdown", watched)
        with pytest.raises(BrokenProcessPool):
            await single.run(jobs.die_mid_write, read=False, result_bytes=1 << 20)
        assert (await single.run(jobs.make_rows, 10, 1, read=False, result_bytes=1 << 20)).segment is not None, (
            "a fresh pool"
        )
        await single.close()
        await _until(
            lambda: not any(t.name == "shared-arrow-reap-single" and t.is_alive() for t in threading.enumerate()),
            "the reaper finishing",
        )

        assert [executor for executor in not_waited if executor in waited] == []

    async def test_a_pool_stale_for_its_memory_whose_worker_then_died_is_said_reaped_and_swept_once(
        self, store: SharedArrowStore, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # the stale pool is replaced without a submission, so nothing but its retirement hears the idle death
        shutdown = ProcessPoolExecutor.shutdown
        shutdowns: list[tuple[int, bool]] = []

        def watched(executor: ProcessPoolExecutor, wait: bool = True, *, cancel_futures: bool = False) -> None:
            shutdowns.append((id(executor), wait))
            shutdown(executor, wait=wait, cancel_futures=cancel_futures)

        monkeypatch.setattr(ProcessPoolExecutor, "shutdown", watched)
        caplog.set_level(logging.WARNING)
        bounded = SharedArrowPool(store, workers=1, private_bytes=0, name="bounded")
        try:
            store.publish("v1", {TABLE: _table(1)})
            dead = (await bounded.run(jobs.process_id)).value
            # it dies stopped, before it could move off v1 (``_frozen``), so v1 is what it held and what must be swept
            with _frozen(dead):
                store.publish("v2", {TABLE: _table(2)})
                assert set(store.versions()) == {"v1", "v2"}, "the idle worker still maps v1"
                os.kill(dead, signal.SIGKILL)
            await _until(lambda: _reaped(dead), "the executor joining its dead worker")

            assert (await bounded.run(jobs.column_sum, TABLE, "a")).value == 2 * ROWS, "the job ran on a fresh pool"
            assert bounded.replaced == 1
            await _until(lambda: store.versions() == ("v2",), "the dead worker's v1 swept")
        finally:
            await bounded.close()
        await _until(
            lambda: not any(t.name == "shared-arrow-reap-bounded" and t.is_alive() for t in threading.enumerate()),
            "the reaper finishing",
        )

        assert caplog.text.count("worker died") == 1
        executors = [executor for executor, _ in shutdowns]
        assert len(set(executors)) == 2, "the dead worker's pool and its replacement"
        assert len(executors) == 2, "each shut down once"
        assert all(wait for _, wait in shutdowns), "each waited for"

    async def test_a_retired_pool_finishing_a_long_job_holds_up_no_other_retired_pool(
        self, store: SharedArrowStore, where: Path
    ) -> None:
        # every job leaves its pool past its bound, so each next submission retires the pool before it
        signals = where / "signals"
        bounded = SharedArrowPool(store, workers=2, private_bytes=0, name="bounded")
        long = None
        try:
            store.publish("v1", {TABLE: _table(1)})
            await bounded.run(jobs.process_id, read=False)
            # the second pool: one worker on the long job, the other's short job leaves the pool stale
            long = asyncio.ensure_future(
                bounded.run(jobs.write_on_signal, str(signals), read=False, result_bytes=1 << 20)
            )
            await _until(lambda: (signals / "started").exists(), "the long job started")
            await bounded.run(jobs.process_id, read=False)
            # the third pool, retired (by the next job) while the second still runs the long job. Its worker keeps v1
            # mapped, the one reader of it: v1 stays current until the third pool is gone, so the worker never moves
            # off it, and only that pool's shutdown lets it go
            mapping = (await bounded.run(jobs.process_id)).value
            await asyncio.to_thread(store.settle)
            assert _held(store, "v1"), "the third pool's worker still maps v1"
            await bounded.run(jobs.process_id, read=False)

            await _until(lambda: _reaped(mapping), "the third pool's worker shut down")
            await asyncio.to_thread(store.settle)
            assert not _held(store, "v1"), "the third pool's worker let v1 go"
            await asyncio.to_thread(store.publish, "v2", {TABLE: _table(2)})
            await asyncio.to_thread(store.settle)
            assert store.versions() == ("v2",), "nothing holds v1: swept as it was superseded"
            assert not long.done(), "the long job still runs"
        finally:
            (signals / "go").touch()
            if long is not None:
                result = await long
                assert result.segment is not None
                result.segment.close()
            await bounded.close()
        assert bounded.replaced == 3


class TestAWorkersResultBecomesAPart:
    def test_a_claimed_result_is_linked_into_a_version_never_copied(self, store: SharedArrowStore) -> None:
        schema = _table(0, rows=0).schema
        store.publish(
            "v1", {TABLE: Parts(schema=schema, parts={"VA": _table(1, rows=1000), "GA": _table(2, rows=1000)})}
        )
        slot = store.reserve(1 << 20)
        slot.write({"t.GA": _table(20, rows=1000)})
        segment = store.claim(slot)
        assert segment is not None
        source = segment.files["t.GA"]
        store.publish("v2", {TABLE: Parts(schema=schema, parts={"GA": source})}, base="v1")
        assert version_files(store.root, "v2")["t@GA.arrow"].path.stat().st_ino == source.stat().st_ino
        segment.close()
        with MappedVersion(store.root, "v2") as mapped:
            assert mapped.tables[TABLE].column("a").to_pylist() == [20] * 1000 + [1] * 1000
        assert store.segments() == ()

    def test_a_linked_part_with_other_columns_is_refused(self, store: SharedArrowStore) -> None:
        slot = store.reserve(1 << 20)
        slot.write({"other": pa.table({"b": [1]})})
        segment = store.claim(slot)
        assert segment is not None
        with pytest.raises(ValueError, match="columns"):
            store.publish("v1", {TABLE: Parts(schema=_table(0, rows=0).schema, parts={"VA": segment.files["other"]})})
        segment.close()


class TestAResultKeptPastItsCloseStaysCounted:
    """a claimed result's tables are its files' pages, zero-copy: a table (or a slice of one) kept past the segment's
    close keeps those pages in memory, so the store keeps counting its files until the last buffer is freed (cobalt
    prod, 2026-11-01: up to 2.9 GiB of shared memory held by results closed and unlinked, outside ``max_bytes``)."""

    def test_a_table_kept_past_close_is_counted_until_it_is_freed(self, where: Path) -> None:
        bounded = SharedArrowStore(where / "bounded", max_bytes=12 << 20)
        slot = bounded.reserve(8 << 20)
        slot.write({TABLE: pa.table({"a": pa.array(range(640 * 1024), pa.int64())})})  # 5 MiB
        segment = bounded.claim(slot)
        assert segment is not None
        kept = segment.tables[TABLE].slice(10, 100)
        segment.close()
        assert bounded.held_bytes() >= 5 << 20, "a result's pages still mapped are counted"
        assert bounded.segments() != ()
        with pytest.raises(SharedArrowFullError):
            bounded.reserve(8 << 20)
        assert kept.column("a")[0].as_py() == 10
        del kept
        gc.collect()
        bounded.sweep()
        assert bounded.segments() == ()
        assert bounded.held_bytes() == 0
        bounded.discard(bounded.reserve(8 << 20))

    def test_a_result_closed_with_nothing_kept_goes_at_once(self, store: SharedArrowStore) -> None:
        slot = store.reserve(1 << 20)
        slot.write({TABLE: _table(1, rows=1000)})
        segment = store.claim(slot)
        assert segment is not None
        assert segment.tables[TABLE].num_rows == 1000
        segment.close()
        assert store.segments() == ()
        assert store.held_bytes() == 0

    def test_a_kept_result_counts_against_a_publish(self, where: Path) -> None:
        bounded = SharedArrowStore(where / "bounded", max_bytes=8 << 20)
        slot = bounded.reserve(6 << 20)
        slot.write({TABLE: pa.table({"a": pa.array(range(640 * 1024), pa.int64())})})  # 5 MiB
        segment = bounded.claim(slot)
        assert segment is not None
        kept = segment.tables[TABLE]
        segment.close()
        with pytest.raises(SharedArrowFullError):
            bounded.publish("v1", {TABLE: pa.table({"a": pa.array(range(512 * 1024), pa.int64())})})  # 4 MiB
        del kept
        gc.collect()
        bounded.publish("v1", {TABLE: pa.table({"a": pa.array(range(512 * 1024), pa.int64())})})
        assert bounded.segments() == ()


class TestAResultWrittenInSteps:
    """a worker may write its result in steps (``SegmentSlot.writing``), so it holds one step's rows at a time, not the
    whole result's: the reports pod's apply worker writes a change of every state a slice of states at a time."""

    def test_the_steps_are_one_result_claimed_whole(self, store: SharedArrowStore) -> None:
        slot = store.reserve(1 << 20)
        with slot.writing() as write:
            write({"a": _table(1, rows=10)})
            assert store.segments() == (), "nothing is the slot's until the writing ends"
            write({"b": _table(2, rows=20)})
        segment = store.claim(slot)
        assert segment is not None
        assert {name: table.num_rows for name, table in segment.tables.items()} == {"a": 10, "b": 20}
        segment.close()

    def test_the_slots_bound_holds_across_its_steps(self, store: SharedArrowStore) -> None:
        one = _table(1, rows=50_000)
        slot = store.reserve(int(one.nbytes * 1.5))
        with pytest.raises(SharedArrowFullError):
            with slot.writing() as write:
                write({"a": one})
                write({"b": one})
        assert store.claim(slot) is None
        assert _hidden(store) == []

    def test_a_name_is_written_once_across_steps(self, store: SharedArrowStore) -> None:
        slot = store.reserve(1 << 20)
        with pytest.raises(ValueError, match="once"):
            with slot.writing() as write:
                write({"a": _table(1, rows=10)})
                write({"a": _table(2, rows=10)})
        assert store.claim(slot) is None

    def test_steps_that_write_nothing_leave_no_result(self, store: SharedArrowStore) -> None:
        slot = store.reserve(1 << 20)
        with slot.writing() as write:
            write({})
        assert store.claim(slot) is None


class TestASweepNeverTakesAnOpenSlotsStaging:
    """a slot's staging directory is made, given its lock file and locked in three steps, and the serving process
    sweeps on other threads meanwhile (reserve, claim, discard, release, close). A sweep in between retired the
    directory, or locked and renamed it away, and the worker's export failed with FileNotFoundError (the local replay
    of 2026-11-01: 3 times in 40 minutes, 8 states each falling back to the slow L3 republish). An open slot's staging
    is never swept, as an open hand-off's is not."""

    def test_a_sweep_between_the_staging_directory_and_its_lock_file_leaves_the_write_whole(
        self, store: SharedArrowStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        slot = store.reserve(1 << 20)
        touch = Path.touch

        def touch_after_a_sweep(path: Path, *args: Any, **kwargs: Any) -> None:
            if path.name == ".lock" and path.parent.name.startswith(".staging-"):
                store.sweep()
            touch(path, *args, **kwargs)

        monkeypatch.setattr(Path, "touch", touch_after_a_sweep)

        slot.write({TABLE: _table(1, rows=10)})

        segment = store.claim(slot)
        assert segment is not None, "the swept write left no result"
        assert segment.tables[TABLE].num_rows == 10
        segment.close()

    def test_a_sweep_between_the_lock_file_and_its_lock_leaves_the_staging_directory(
        self, store: SharedArrowStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        slot = store.reserve(1 << 20)
        opened = os.open
        swept: list[bool] = []

        def open_after_a_sweep(path: Any, *args: Any, **kwargs: Any) -> int:
            # the writer opens its lock file to lock it: the sweep runs after the file is made, before the lock
            lock = Path(path)
            if not swept and lock.name == ".lock" and lock.parent.name.startswith(".staging-"):
                swept.append(True)
                store.sweep()
                assert lock.parent.is_dir(), "the sweep took an open slot's staging directory"
            return opened(path, *args, **kwargs)

        monkeypatch.setattr(os, "open", open_after_a_sweep)

        slot.write({TABLE: _table(1, rows=10)})

        assert swept, "no sweep ran between the lock file and the lock"
        segment = store.claim(slot)
        assert segment is not None
        assert segment.tables[TABLE].num_rows == 10
        segment.close()

    def test_a_discarded_slots_staging_directory_is_still_removed(self, store: SharedArrowStore) -> None:
        slot = store.reserve(1 << 20)
        segments = store.root / "segments"
        # a writer that died with its lock file made, and one that died before it
        (segments / f".staging-{slot.slot_id}-deadbeef").mkdir()
        (segments / f".staging-{slot.slot_id}-deadbeef" / ".lock").touch()
        (segments / f".staging-{slot.slot_id}-0badf00d").mkdir()
        store.sweep()
        assert len(_hidden(store)) == 2, "an open slot's staging was swept"

        store.discard(slot)

        assert _hidden(store) == []
