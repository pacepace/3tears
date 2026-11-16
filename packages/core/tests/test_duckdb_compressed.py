"""A compressed DuckDB L1: the tables held compressed, checkpointed off the write's path, read as any L1's.

The contracts: every way into the copy (a write, a read cursor, a thread's connection) reaches the compressed
tables; a write asks for one checkpoint on the backend's own thread and returns without waiting for it; the
checkpoint makes the tables smaller than the same rows uncompressed; a reset stops the thread.
"""

from __future__ import annotations

import functools
import gc
import queue
import subprocess
import sys
import textwrap
import threading
import time
import weakref
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from sqlalchemy import Column, Integer, MetaData, String, Table
from threetears.core.testing.l1_tables import L1_EVICTION_INFO, TEST_L1_TABLE

duckdb = pytest.importorskip("duckdb")
pa = pytest.importorskip("pyarrow")

from threetears.core.cache.duckdb import (  # noqa: E402
    DuckDBBackend,
    PartitionReplacement,
)

ROWS = 400_000


#: the scope the tests replace: a result's state, declared as a scope key
_STATE_SCOPES = {"results": (("state",),)}


def _metadata() -> MetaData:
    metadata = MetaData()
    Table(
        "results",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("state", String(8)),
        Column("name", String(64)),
        Column("votes", Integer),
        info={L1_EVICTION_INFO: TEST_L1_TABLE},
    )
    return metadata


def _rows(state: str, start: int, rows: int) -> Any:
    """rows of one state: a text column of a few thousand distinct values, as election tables have.

    :return: the rows
    :rtype: pa.Table
    """
    ids = range(start, start + rows)
    return pa.table(
        {
            "id": pa.array(ids, pa.int32()),
            "state": pa.array([state] * rows),
            "name": pa.array([f"candidate-{(i * 7919) % 4093}" for i in ids]),
            "votes": pa.array([(i * 31) % 100_003 for i in ids], pa.int32()),
        }
    )


def _memory(backend: DuckDBBackend) -> int:
    with backend.read_snapshot() as cursor:
        found = cursor.execute("SELECT sum(memory_usage_bytes) FROM duckdb_memory()").fetchone()
    assert found is not None
    return int(found[0])


def _until(condition: Callable[[], bool], what: str, seconds: float = 30.0) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError(f"never: {what}")
        time.sleep(0.01)


def _checkpointers() -> set[threading.Thread]:
    """the threads alive now: a backend's checkpointer is the thread it starts.

    :return: the threads
    :rtype: set[threading.Thread]
    """
    return set(threading.enumerate())


def _checkpointed(backend: DuckDBBackend) -> bool:
    return backend.background_checkpoints >= 1


@pytest.fixture
def compressed() -> Iterator[DuckDBBackend]:
    backend = DuckDBBackend(compressed=True, scope_keys=_STATE_SCOPES)
    backend.initialize(_metadata())
    yield backend
    backend.reset()


def _fill(backend: DuckDBBackend) -> None:
    backend.replace_partitions(
        [
            PartitionReplacement(table="results", column="state", value=state, arrow=_rows(state, index * ROWS, ROWS))
            for index, state in enumerate(("VA", "GA"))
        ]
    )


class TestCompressedCopy:
    def test_every_way_in_reaches_the_compressed_tables(self, compressed: DuckDBBackend) -> None:
        assert compressed.compressed
        compressed.upsert_many("results", [{"id": 1, "state": "VA", "name": "a", "votes": 5}], "id")
        with compressed.read_snapshot() as cursor:
            assert cursor.execute("SELECT current_database()").fetchone() == ("l1",)
            # attached with COMPRESS: an in-memory catalog attached without it holds real tables uncompressed (the
            # reports' extracts: 513 MiB without, 116 MiB with), however often checkpointed
            options = cursor.execute("SELECT options FROM duckdb_databases() WHERE database_name = 'l1'").fetchone()
            assert options == ({"compress": "true"},)
            assert cursor.execute("SELECT votes FROM results").fetchall() == [(5,)]
        assert compressed.get_connection().execute("SELECT count(*) FROM results").fetchone() == (1,)
        assert compressed.execute_query("SELECT name FROM results") == [{"name": "a"}]

    def test_a_write_checkpoints_off_its_path_and_the_copy_shrinks(self, compressed: DuckDBBackend) -> None:
        plain = DuckDBBackend(scope_keys=_STATE_SCOPES)
        plain.initialize(_metadata())
        try:
            _fill(plain)
            _fill(compressed)
            _until(lambda: compressed.background_checkpoints >= 1, "the checkpointer ran")
            _until(lambda: _memory(compressed) < _memory(plain) / 2, "the compressed copy is under half the plain one")
            assert compressed.background_checkpoints_failed == 0
            with compressed.read_snapshot() as cursor:
                assert cursor.execute("SELECT count(*), sum(votes) FROM results").fetchone() == (
                    2 * ROWS,
                    sum((i * 31) % 100_003 for i in range(2 * ROWS)),
                )
        finally:
            plain.reset()

    def test_a_reader_keeps_its_state_across_a_write_and_its_checkpoint(self, compressed: DuckDBBackend) -> None:
        _fill(compressed)
        _until(lambda: compressed.background_checkpoints >= 1, "the first checkpoint ran")
        with compressed.read_snapshot() as cursor:
            compressed.replace_partitions(
                [PartitionReplacement(table="results", column="state", value="VA", arrow=_rows("VA", 10 * ROWS, 10))]
            )
            _until(lambda: compressed.background_checkpoints >= 2, "the second checkpoint ran")
            assert cursor.execute("SELECT count(*) FROM results").fetchone() == (2 * ROWS,)
        with compressed.read_snapshot() as cursor:
            assert cursor.execute("SELECT count(*) FROM results").fetchone() == (ROWS + 10,)

    def test_a_plain_backend_asks_for_no_checkpoint(self) -> None:
        before = _checkpointers()
        plain = DuckDBBackend(scope_keys=_STATE_SCOPES)
        plain.initialize(_metadata())
        try:
            _fill(plain)
            plain.request_checkpoint()
            assert plain.background_checkpoints == 0
            assert _checkpointers() <= before, "no thread was started"
        finally:
            plain.reset()

    def test_a_reset_stops_the_checkpointer(self, compressed: DuckDBBackend) -> None:
        before = _checkpointers()
        _fill(compressed)
        _until(lambda: compressed.background_checkpoints >= 1, "the checkpointer ran")
        started = _checkpointers() - before
        assert len(started) == 1
        compressed.reset()
        assert not any(thread.is_alive() for thread in started)


class TestTheCheckpointerEnds:
    def test_a_reset_while_writes_keep_asking_returns(self) -> None:
        for _ in range(30):
            backend = DuckDBBackend(compressed=True, scope_keys=_STATE_SCOPES)
            backend.initialize(_metadata())
            stop = threading.Event()

            def ask(on: DuckDBBackend = backend, until: threading.Event = stop) -> None:
                while not until.is_set():
                    on.request_checkpoint()

            asker = threading.Thread(target=ask)
            asker.start()
            resetting = threading.Thread(target=backend.reset)
            resetting.start()
            resetting.join(10)
            stop.set()
            asker.join()
            assert not resetting.is_alive(), "reset never returned"

    def test_a_backend_dropped_without_a_reset_is_collected_and_its_thread_ends(self) -> None:
        before = _checkpointers()
        backend = DuckDBBackend(compressed=True, scope_keys=_STATE_SCOPES)
        backend.initialize(_metadata())
        _fill(backend)
        _until(functools.partial(_checkpointed, backend), "the checkpointer ran")
        (thread,) = _checkpointers() - before
        held = weakref.ref(backend)
        del backend
        gc.collect()
        assert held() is None, "the checkpointer kept the backend alive"
        thread.join(10)
        assert not thread.is_alive()

    def test_a_stop_landing_between_the_wake_and_its_taking_is_seen(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """the window a reset can land in: the thread woke, and the stop and its wake arrive before the thread takes
        the wakes queued. The stop must still end the thread, never be taken away with the wake."""
        made: list[StopAtTake] = []
        resets: list[threading.Thread] = []

        class StopAtTake(queue.SimpleQueue[None]):
            """the checkpointer's wakes (made by the first request): its first taking of the queued wakes first lets a
            reset land (stop, then its wake)."""

            def __init__(self) -> None:
                super().__init__()
                self.puts = 0
                self.armed = True
                made.append(self)

            def put(self, item: None, block: bool = True, timeout: float | None = None) -> None:
                self.puts += 1
                super().put(item, block, timeout)

            def get_nowait(self) -> None:
                if self.armed:
                    self.armed = False
                    before = self.puts
                    resetting = threading.Thread(target=backend.reset, daemon=True)
                    resets.append(resetting)
                    resetting.start()
                    _until(lambda: self.puts > before, "the reset's stop and wake landed")
                return super().get_nowait()

        backend = DuckDBBackend(compressed=True, scope_keys=_STATE_SCOPES)
        backend.initialize(_metadata())
        with monkeypatch.context() as patched:
            patched.setattr(queue, "SimpleQueue", StopAtTake)
            backend.request_checkpoint()
        assert made, "the first request made the checkpointer's wakes"
        _until(lambda: bool(resets), "the checkpointer woke")
        resets[0].join(10)
        assert not resets[0].is_alive(), "the stop was taken with the wake: the thread waits forever"


#: a compressed backend dropped in a cycle, collected on its checkpointer's own thread as the thread takes its wake:
#: an Event's ``clear`` or a queue's ``get_nowait``, whichever the checkpointer takes its wakes with, collects there
_COLLECTED_ON_ITS_THREAD = textwrap.dedent(
    """
    import gc, queue, sys, threading
    from sqlalchemy import Column, Integer, MetaData, Table
    from threetears.core.cache.duckdb import _CHECKPOINTER_THREAD, DuckDBBackend
    from threetears.core.testing.l1_tables import L1_EVICTION_INFO, TEST_L1_TABLE

    gc.disable()  # the one collection is the checkpointer's own
    dropped = threading.Event()

    def collect_on_the_checkpointer():
        if threading.current_thread().name == _CHECKPOINTER_THREAD:
            dropped.wait()
            gc.collect()

    class CollectingEvent(threading.Event):
        def clear(self):
            with self._cond:
                collect_on_the_checkpointer()
                self._flag = False

    class CollectingQueue(queue.SimpleQueue):
        def get_nowait(self):
            collect_on_the_checkpointer()
            return super().get_nowait()

    threading.Event = CollectingEvent
    queue.SimpleQueue = CollectingQueue
    metadata = MetaData()
    Table("rows", metadata, Column("id", Integer, primary_key=True), info={L1_EVICTION_INFO: TEST_L1_TABLE})
    backend = DuckDBBackend(compressed=True)
    backend.initialize(metadata)
    backend.itself = backend  # a reference cycle: only a collection frees it
    backend.request_checkpoint()
    (checkpointer,) = [thread for thread in threading.enumerate() if thread.name == _CHECKPOINTER_THREAD]
    del backend
    dropped.set()
    checkpointer.join(20)
    print("ended" if not checkpointer.is_alive() else "hung", flush=True)
    """
)


class TestABackendCollectedOnItsCheckpointer:
    def test_the_finalizers_wake_ends_the_thread(self) -> None:
        # a finalizer runs inside whatever its collection interrupted: the backend's wake, run on the checkpointer's
        # own thread inside its Event's lock (taken by wait and clear), waited on that lock for ever (2026-11-16)
        child = subprocess.run(
            [sys.executable, "-c", _COLLECTED_ON_ITS_THREAD],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert child.returncode == 0, child.stderr
        assert child.stdout.splitlines() == ["ended"], child.stdout + child.stderr
