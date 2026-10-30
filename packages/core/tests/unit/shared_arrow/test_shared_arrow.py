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
import logging
import os
import sys
import shutil
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any

import duckdb
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
)

ROWS = 200_000
TABLE = "t"


def _table(value: int, rows: int = ROWS) -> pa.Table:
    """a table whose column ``a`` is ``value`` in every row, and a text column beside it.

    :return: the table
    :rtype: pa.Table
    """
    return pa.table({"a": pa.array([value] * rows, pa.int64()), "s": pa.array([f"row-{value}"] * rows, pa.string())})


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


class TestManyProcessesShareOneCopy:
    async def test_three_workers_read_one_mapping(
        self, store: SharedArrowStore, pool: SharedArrowPool, where: Path
    ) -> None:
        version = store.publish("v1", {TABLE: _table(3)})
        path = str(version.path / f"{TABLE}.arrow")
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
        assert store.versions() == ("v1", "v3"), "v2 went at once; v1 is leased"
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
        store.publish("v2", {TABLE: _table(2)})
        assert "v1" in store.versions(), "the read kept v1 mapped while it was current"
        with store.lease(), read_version(store.root, "v2") as mapped:
            assert mapped.name == "v2"
        assert store.versions() == ("v2",)

    async def test_an_idle_worker_holds_its_mapping_until_its_next_job(
        self, store: SharedArrowStore, single: SharedArrowPool
    ) -> None:
        store.publish("v1", {TABLE: _table(1)})
        assert (await single.run(jobs.column_sum, TABLE, "a")).value == ROWS
        store.publish("v2", {TABLE: _table(2)})
        assert set(store.versions()) == {"v1", "v2"}, "the idle worker still maps v1"
        assert (await single.run(jobs.column_sum, TABLE, "a")).value == 2 * ROWS
        assert store.versions() == ("v2",), "its next job let v1 go"

    async def test_a_worker_that_exits_lets_go(self, store: SharedArrowStore) -> None:
        one = SharedArrowPool(store, workers=1, private_bytes=1 << 40, name="one")
        store.publish("v1", {TABLE: _table(1)})
        await one.run(jobs.column_sum, TABLE, "a")
        store.publish("v2", {TABLE: _table(2)})
        assert "v1" in store.versions()
        await one.close()
        assert store.versions() == ("v2",)

    def test_a_new_store_clears_what_nobody_holds(self, where: Path) -> None:
        first = SharedArrowStore(where / "shm", max_bytes=64 << 20)
        first.publish("v1", {TABLE: _table(1)})
        first.publish("v2", {TABLE: _table(2)})
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
        assert store.segments() == ()
        assert store.status()["slots_open"] == 0

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
        await _until(lambda: (signals / "started").exists(), "the worker started")
        running.cancel()
        (signals / "go").touch()
        await _until(lambda: store.status()["slots_open"] == 0, "the abandoned slot discarded")
        assert store.segments() == ()
        assert _hidden(store) == []

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
        assert store.versions() == ("v3",)

    def test_many_versions_hold_only_the_current_one(self, store: SharedArrowStore) -> None:
        for value in range(10):
            last = store.publish(f"v{value}", {TABLE: _table(value)})
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


def _serving_sum(read: Any, table: str = TABLE, column: str = "a") -> int:
    """the sum of one column through a serving read's DuckDB cursor.

    :return: the sum
    :rtype: int
    """
    found = read.cursor.execute(f'SELECT sum("{column}") FROM "{table}"').fetchone()
    assert found is not None
    return int(found[0])


class TestTheServingProcessReadsInPlace:
    async def test_the_serving_process_and_its_workers_read_one_mapping(
        self, store: SharedArrowStore, pool: SharedArrowPool, where: Path
    ) -> None:
        version = store.publish("v1", {TABLE: _table(3)})
        path = os.path.realpath(version.path / f"{TABLE}.arrow")
        signals = str(where / "signals")
        reader = store.reader()
        try:
            before = pa.total_allocated_bytes()
            with reader.read() as read:
                assert read.version == "v1"
                assert _serving_sum(read) == 3 * ROWS
                assert pa.total_allocated_bytes() - before < read.tables[TABLE].nbytes / 10, (
                    "no copy in the serving process"
                )
                answers = [
                    result.value
                    for result in await asyncio.gather(
                        *(pool.run(jobs.shared_read, TABLE, "a", signals, 3, path) for _ in range(3))
                    )
                ]
                assert {answer["sum"] for answer in answers} == {3 * ROWS}
                assert os.getpid() not in {answer["pid"] for answer in answers}
                if sys.platform.startswith("linux"):
                    # four processes map the file; each is charged a share of the pages it has resident
                    serving = jobs.mapping_of(path)
                    assert serving["rss"] >= ROWS * 8 * 0.9
                    assert serving["pss"] <= serving["rss"] / 2
        finally:
            reader.close()

    def test_a_read_keeps_its_version_across_a_swap_and_the_old_goes_when_it_ends(
        self, store: SharedArrowStore
    ) -> None:
        store.publish("v1", {TABLE: _table(1)})
        reader = store.reader(schema="s")
        with contextlib.ExitStack() as reads:
            old = reads.enter_context(reader.read())
            store.publish("v2", {TABLE: _table(2)})
            assert set(store.versions()) == {"v1", "v2"}, "v1 is kept while a serving read is open on it"
            assert old.cursor.execute(f'SELECT sum(a) FROM s."{TABLE}"').fetchone() == (1 * ROWS,)
            with reader.read() as new:
                assert new.version == "v2"
                assert new.cursor.execute(f'SELECT sum(a) FROM s."{TABLE}"').fetchone() == (2 * ROWS,)
            assert set(store.versions()) == {"v1", "v2"}
        assert store.versions() == ("v2",), "v1 retired once its last serving read ended"
        assert reader.status()["version"] == "v2"
        reader.close()

    def test_an_idle_reader_lets_a_superseded_version_go_when_the_next_is_published(
        self, store: SharedArrowStore
    ) -> None:
        store.publish("v1", {TABLE: _table(1)})
        reader = store.reader()
        with reader.read():
            pass
        assert reader.status()["version"] == "v1"
        store.publish("v2", {TABLE: _table(2)})
        assert store.versions() == ("v2",), "the reader let v1 go at the publish, not at its next read"
        assert reader.status()["version"] is None
        with reader.read() as read:
            assert _serving_sum(read) == 2 * ROWS
        reader.close()
        store.publish("v3", {TABLE: _table(3)})
        assert store.versions() == ("v3",)

    async def test_a_worker_and_the_serving_reader_hold_their_versions_apart(
        self, store: SharedArrowStore, pool: SharedArrowPool, where: Path
    ) -> None:
        signals = where / "signals"
        store.publish("v1", {TABLE: _table(1)})
        reader = store.reader()
        reading = asyncio.ensure_future(pool.run(jobs.read_on_signal, TABLE, "a", str(signals)))
        await _until(lambda: (signals / "started").exists(), "the worker mapped v1")
        store.publish("v2", {TABLE: _table(2)})
        with reader.read() as read:
            assert _serving_sum(read) == 2 * ROWS
            (signals / "go").touch()
            assert (await reading).value == 1 * ROWS, "the worker read v1 whole"
            await _until(lambda: store.versions() == ("v2",), "v1 retired once the worker let it go")
        reader.close()
        assert store.versions() == ("v2",), "the current version stays after the reader closes"

    def test_reads_on_many_threads_across_publishes_each_see_one_version(self, store: SharedArrowStore) -> None:
        store.publish("v0", {TABLE: _table(0, rows=10_000)})
        reader = store.reader()
        seen: list[tuple[str, int]] = []
        failures: list[BaseException] = []
        stop = threading.Event()

        def read_until_stopped() -> None:
            try:
                while not stop.is_set():
                    with reader.read() as read:
                        seen.append((read.version, _serving_sum(read)))
            except BaseException as exc:  # noqa: BLE001 - the test reports it
                failures.append(exc)

        threads = [threading.Thread(target=read_until_stopped) for _ in range(4)]
        for thread in threads:
            thread.start()
        for value in range(1, 8):
            store.publish(f"v{value}", {TABLE: _table(value, rows=10_000)})
            time.sleep(0.02)
        stop.set()
        for thread in threads:
            thread.join()
        assert not failures
        assert all(total == int(name[1:]) * 10_000 for name, total in seen), "every read saw one version whole"
        reader.close()
        assert store.versions() == ("v7",), "every superseded version retired"

    def test_each_version_is_prepared_once_and_its_views_serve_every_read(self, store: SharedArrowStore) -> None:
        prepared: list[str] = []

        def views(connection: Any) -> str:
            connection.execute(f'CREATE VIEW doubled AS SELECT a * 2 AS a FROM "{TABLE}"')
            prepared.append("made")
            return f"views-{len(prepared)}"

        store.publish("v1", {TABLE: _table(1, rows=10)})
        reader = store.reader(prepare=views)
        for _ in range(3):
            with reader.read() as read:
                assert read.prepared == "views-1"
                assert read.cursor.execute("SELECT sum(a) FROM doubled").fetchone() == (20,)
        store.publish("v2", {TABLE: _table(2, rows=10)})
        with reader.read() as read:
            assert read.prepared == "views-2"
            assert read.cursor.execute("SELECT sum(a) FROM doubled").fetchone() == (40,)
        assert prepared == ["made", "made"]
        reader.close()

    def test_a_prepare_that_fails_holds_nothing(self, store: SharedArrowStore) -> None:
        def broken(connection: Any) -> None:
            raise RuntimeError("no views")

        store.publish("v1", {TABLE: _table(1, rows=10)})
        reader = store.reader(prepare=broken)
        with pytest.raises(RuntimeError, match="no views"), reader.read():
            pass
        store.publish("v2", {TABLE: _table(2, rows=10)})
        assert store.versions() == ("v2",), "the failed mapping released its lease"
        reader.close()

    def test_nothing_published_is_nothing_to_read_and_a_closed_reader_reads_nothing(
        self, store: SharedArrowStore
    ) -> None:
        reader = store.reader()
        with pytest.raises(LookupError), reader.read():
            pass
        store.publish("v1", {TABLE: _table(1, rows=10)})
        reader.close()
        with pytest.raises(RuntimeError), reader.read():
            pass


def _parts(values: dict[str, int], rows: int = 1_000) -> Parts:
    """a table in parts, each part's ``a`` its value.

    :return: the parts
    :rtype: Parts
    """
    return Parts(schema=_table(0, rows=0).schema, parts={part: _table(value, rows) for part, value in values.items()})


def _inode(store: SharedArrowStore, version: str, file_name: str) -> int:
    return (store.root / "versions" / version / file_name).stat().st_ino


class TestPartsAreSharedAcrossVersions:
    def test_a_version_rewrites_only_the_parts_that_changed(self, store: SharedArrowStore) -> None:
        first = store.publish("v1", {TABLE: _parts({"VA": 1, "GA": 2, "TX": 3}), "whole": _table(5, rows=10)})
        with store.lease("v1"):
            store.publish(
                "v2",
                {TABLE: Parts(schema=_table(0, rows=0).schema, parts={"GA": _table(20, 1_000), "TX": None})},
                base="v1",
            )
            assert _inode(store, "v1", "t@VA.arrow") == _inode(store, "v2", "t@VA.arrow"), "VA is the same file"
            assert _inode(store, "v1", "t@GA.arrow") != _inode(store, "v2", "t@GA.arrow")
            assert not (store.root / "versions" / "v2" / "t@TX.arrow").exists(), "TX was dropped"
            assert _inode(store, "v1", "whole.arrow") == _inode(store, "v2", "whole.arrow"), "a table not named is kept"
            # the shared file counted once: v1 whole, plus v2's own GA part and schema file
            v2_own = sum((store.root / "versions" / "v2" / name).stat().st_size for name in ("t@GA.arrow", "t@.arrow"))
            assert store.held_bytes() == first.nbytes + v2_own
        reader = store.reader()
        with reader.read() as read:
            assert read.tables[TABLE].num_rows == 2_000
            assert read.cursor.execute(f'SELECT sum(a) FROM "{TABLE}"').fetchone() == ((1 + 20) * 1_000,)
            assert read.cursor.execute('SELECT sum(a) FROM "whole"').fetchone() == (50,)
        reader.close()
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
