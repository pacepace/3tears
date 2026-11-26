"""Queries over the copy in query worker processes (:mod:`threetears.core.collections.copy_queries`): a query never
runs in the serving process, answers what the same SQL answers in-process over the same version, reads the version it
was submitted at whatever is published meanwhile, waits in a bounded queue in the order it came, is stopped in its
worker by a timeout or a cancel, fails clearly when its worker dies (the worker replaced), and crosses as Arrow,
never as Python objects a row."""

from __future__ import annotations

import asyncio
import functools
import os
import sys
import threading
import tracemalloc
from collections.abc import AsyncIterator
from pathlib import Path
from types import FrameType
from typing import Any

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from packages.core.tests.unit.collections import copy_query_jobs
from threetears.core.collections.copy_queries import (
    CopyQueries,
    QueryAnswer,
    QueryPoolFull,
    QueryTimeout,
    QueryWorkerDied,
    query_workers,
)
from threetears.core.shared_arrow import SharedArrowStore, read_version

_STATES = pa.array(["AZ", "GA", "NC", "PA", "VA"], pa.string())

#: the prefix a split table's locator is published beside it under
_LOCATED = "__located__"

#: a query that runs for minutes on one thread: 10^10 rows hashed
_ENDLESS = "SELECT sum(hash(a.range * b.range)) AS h FROM range(100000) a, range(100000) b"

_DECLARED = {
    "gated": copy_query_jobs.gated,
    "held": copy_query_jobs.held,
    "touch": copy_query_jobs.touch,
    "dies": copy_query_jobs.dies,
}


def _rows(count: int, seed: int) -> pa.Table:
    rng = np.random.default_rng(seed)
    race = rng.integers(0, max(count // 4, 1), count)
    return pa.table(
        {
            "state": _STATES.take(pa.array(race % len(_STATES))),
            "office_key": pc.cast(pa.array(race), pa.string()),
            "candidate_id": pa.array(np.arange(count), pa.int64()),
            "votes": pa.array(rng.integers(0, 1_000, count), pa.int64()),
        }
    )


async def _publish(store: SharedArrowStore, version: str, rows: pa.Table) -> None:
    located = pa.table({"office_key": rows["office_key"], "state": rows["state"]})
    await asyncio.to_thread(store.publish, version, {"results": rows, f"{_LOCATED}results": located})


def _in_process(store: SharedArrowStore, version: str, sql: str, binds: list[Any], **tables: pa.Table) -> pa.Table:
    """the same SQL run here, over the same version mapped the same way."""
    with read_version(store.root, version) as mapped:
        connection = mapped.connect()
        try:
            for name, table in tables.items():
                connection.register(name, table)
            copy_query_jobs.views(connection)
            answer: pa.Table = connection.execute(sql, binds).to_arrow_table()
            return answer.combine_chunks()
        finally:
            connection.close()


async def _answer(answer: QueryAnswer) -> pa.Table:
    """the answer's rows, copied out, and the answer closed."""
    table = answer.table.combine_chunks()
    await answer.aclose()
    return table


async def _until(check: Any, seconds: float = 30) -> None:
    deadline = asyncio.get_running_loop().time() + seconds
    while not check():
        assert asyncio.get_running_loop().time() < deadline, "never happened"
        await asyncio.sleep(0.01)


@pytest.fixture
async def store(tmp_path: Path) -> SharedArrowStore:
    store = SharedArrowStore(tmp_path / "shm", max_bytes=1 << 30)
    await _publish(store, "c-0", _rows(20_000, 1))
    return store


@pytest.fixture
async def queries(store: SharedArrowStore) -> AsyncIterator[CopyQueries]:
    pool = CopyQueries(store, workers=2, prepare=copy_query_jobs.views, declared=_DECLARED)
    yield pool
    await pool.close()


@pytest.fixture
async def one_worker(store: SharedArrowStore) -> AsyncIterator[CopyQueries]:
    pool = CopyQueries(store, workers=1, max_waiting=2, declared=_DECLARED)
    yield pool
    await pool.close()


async def _worker_pid(queries: CopyQueries) -> int:
    answer = await queries.run("SELECT 1 AS one")
    await answer.aclose()
    return answer.worker


class _DuckDBCalls:
    """every DuckDB connection this process opens, and every call of the query job's own functions, on any thread."""

    _NAMES = frozenset({"query_job", "_connected", "_write_answer"})

    def __init__(self) -> None:
        self.seen: list[tuple[str, int]] = []
        self._lock = threading.Lock()

    def __call__(self, frame: FrameType, event: str, arg: Any) -> None:
        if event == "c_call" and arg is duckdb.connect:
            name = "duckdb.connect"
        elif event == "call" and frame.f_code.co_name in self._NAMES:
            name = frame.f_code.co_name
        else:
            return
        with self._lock:
            self.seen.append((name, threading.get_ident()))

    def __enter__(self) -> _DuckDBCalls:
        sys.setprofile(self)
        threading.setprofile_all_threads(self)
        return self

    def __exit__(self, *exc: object) -> None:
        threading.setprofile_all_threads(None)
        sys.setprofile(None)


class TestAQueryRunsInAWorker:
    async def test_a_query_never_runs_in_the_serving_process(self, queries: CopyQueries) -> None:
        await _worker_pid(queries)
        keys = pa.table({"office_key": pa.array(["1", "2", "3"], pa.string())})
        with _DuckDBCalls() as calls:
            answer = await queries.run(
                "SELECT r.* FROM results r JOIN keys USING (office_key) WHERE votes > ?", [10], tables={"keys": keys}
            )
            totals = await queries.run(f"SELECT * FROM {copy_query_jobs.TOTALS_VIEW}")
        assert calls.seen == []
        assert answer.worker != os.getpid() and totals.worker != os.getpid()
        assert answer.table.num_rows > 0
        await answer.aclose()
        await totals.aclose()

    @pytest.mark.parametrize(
        ("sql", "binds"),
        [
            ("SELECT * FROM results ORDER BY candidate_id", []),
            ("SELECT state, count(*) AS n, sum(votes) AS v FROM results GROUP BY state ORDER BY state", []),
            ("SELECT * FROM results WHERE state = ? AND votes >= ? ORDER BY candidate_id", ["GA", 500]),
            (f"SELECT * FROM {copy_query_jobs.TOTALS_VIEW} ORDER BY state", []),
            ("SELECT r.* FROM results r JOIN keys USING (office_key) ORDER BY candidate_id", []),
            ("SELECT * FROM results WHERE votes < 0", []),
        ],
    )
    async def test_an_answer_is_the_same_sql_run_in_process_over_the_same_version(
        self, queries: CopyQueries, store: SharedArrowStore, sql: str, binds: list[Any]
    ) -> None:
        keys = pa.table({"office_key": pa.array([str(n) for n in range(0, 5_000, 7)], pa.string())})
        answer = await queries.run(sql, binds, tables={"keys": keys})
        assert answer.version == "c-0"
        assert (await _answer(answer)).equals(_in_process(store, "c-0", sql, binds, keys=keys))

    async def test_a_declared_query_answers_by_name_and_the_locators_are_no_views(
        self, queries: CopyQueries, store: SharedArrowStore
    ) -> None:
        sql = "SELECT count(*) AS n FROM results WHERE state = ?"
        answer = await queries.run_named("gated", ["/", sql, "AZ"])
        assert (await _answer(answer)).equals(_in_process(store, "c-0", sql, ["AZ"]))
        with pytest.raises(duckdb.CatalogException):
            await queries.run(f'SELECT * FROM "{_LOCATED}results"')
        with pytest.raises(KeyError):
            await queries.run_named("not declared")

    async def test_rows_go_as_a_table_never_as_binds(self, queries: CopyQueries) -> None:
        with pytest.raises(ValueError, match="hand rows as a table"):
            await queries.run("SELECT ?", [["a", "b"]])
        with pytest.raises(ValueError, match="at most"):
            await queries.run("SELECT 1", list(range(65)))


class TestAQueryReadsItsVersion:
    async def test_a_publish_during_a_query_does_not_change_its_answer(
        self, one_worker: CopyQueries, store: SharedArrowStore, tmp_path: Path
    ) -> None:
        gate = tmp_path / "gate"
        sql = "SELECT sum(votes) AS v, count(*) AS n FROM results"
        before = _in_process(store, "c-0", sql, [])
        # running when the publish lands, and waiting behind it for the one worker: both submitted at c-0
        running = asyncio.create_task(one_worker.run_named("gated", [str(gate), sql]))
        await _until(lambda: one_worker.status().admitted == 1)
        waiting = asyncio.create_task(one_worker.run(sql))
        await _until(lambda: one_worker.status().admitted == 2)
        await _publish(store, "c-1", _rows(30_000, 2))
        gate.touch()
        for task in (running, waiting):
            answer = await task
            assert answer.version == "c-0"
            assert (await _answer(answer)).equals(before)
        # submitted after the publish: the version current then
        after = await one_worker.run(sql)
        assert after.version == "c-1"
        assert (await _answer(after)).equals(_in_process(store, "c-1", sql, []))

    async def test_a_query_names_the_version_it_reads(self, one_worker: CopyQueries, store: SharedArrowStore) -> None:
        lease = await asyncio.to_thread(store.lease, "c-0")
        try:
            await _publish(store, "c-1", _rows(30_000, 2))
            answer = await one_worker.run("SELECT count(*) AS n FROM results", version="c-0")
            assert answer.version == "c-0"
            assert (await _answer(answer))["n"].to_pylist() == [20_000]
        finally:
            await asyncio.to_thread(lease.release)
        with pytest.raises(LookupError):
            await one_worker.run("SELECT 1", version="c-99")


def _said_done(answered: list[int], n: int, _: object) -> None:
    answered.append(n)


class TestQueueing:
    async def test_queueing_is_bounded_and_in_the_order_callers_came(
        self, one_worker: CopyQueries, tmp_path: Path
    ) -> None:
        gate = tmp_path / "gate"
        blocker = asyncio.create_task(one_worker.run_named("gated", [str(gate), "SELECT 0 AS n"]))
        await _until(lambda: one_worker.status().admitted == 1)
        answered: list[int] = []
        tasks = []
        for n in range(1, 7):
            task = asyncio.create_task(one_worker.run("SELECT ? AS n", [n]))
            task.add_done_callback(functools.partial(_said_done, answered, n))
            tasks.append(task)
            await asyncio.sleep(0.02)
        # one running and two waiting are admitted; the other four callers wait for room, holding no lease
        await _until(lambda: one_worker.status().admitted == 3)
        await asyncio.sleep(0.2)
        assert one_worker.status().admitted == 3
        with pytest.raises(QueryPoolFull):
            await one_worker.run("SELECT 1", wait=False)
        assert answered == []
        gate.touch()
        await (await blocker).aclose()
        for n, task in enumerate(tasks, start=1):
            assert (await _answer(await task))["n"].to_pylist() == [n]
        assert answered == [1, 2, 3, 4, 5, 6]
        assert one_worker.status().admitted == 0
        assert one_worker.status().refused == 1


class TestStoppingAQuery:
    async def test_a_timeout_stops_the_workers_query(self, one_worker: CopyQueries) -> None:
        worker = await _worker_pid(one_worker)
        with pytest.raises(QueryTimeout):
            await one_worker.run(_ENDLESS, timeout=1.0)
        # the one worker is free again at once, the same process: its query was interrupted, not abandoned
        assert await asyncio.wait_for(_worker_pid(one_worker), 10) == worker
        assert one_worker.status().timed_out == 1

    async def test_a_cancel_stops_the_workers_query(self, one_worker: CopyQueries) -> None:
        worker = await _worker_pid(one_worker)
        running = asyncio.create_task(one_worker.run(_ENDLESS))
        await asyncio.sleep(1.0)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert await asyncio.wait_for(_worker_pid(one_worker), 10) == worker
        assert one_worker.status().stopped == 1

    async def test_a_query_cancelled_while_it_waits_never_runs(self, one_worker: CopyQueries, tmp_path: Path) -> None:
        gate, ran = tmp_path / "gate", tmp_path / "ran"
        blocker = asyncio.create_task(one_worker.run_named("gated", [str(gate), "SELECT 0 AS n"]))
        await _until(lambda: one_worker.status().admitted == 1)
        waiting = asyncio.create_task(one_worker.run_named("touch", [str(ran)]))
        await _until(lambda: one_worker.status().admitted == 2)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        gate.touch()
        await (await blocker).aclose()
        await _worker_pid(one_worker)
        assert not ran.exists()


class TestADeadWorker:
    async def test_a_dead_worker_fails_its_query_clearly_and_is_replaced(self, one_worker: CopyQueries) -> None:
        worker = await _worker_pid(one_worker)
        with pytest.raises(QueryWorkerDied, match="died"):
            await one_worker.run_named("dies")
        replaced = await _worker_pid(one_worker)
        assert replaced not in (worker, os.getpid())
        assert one_worker.status().died == 1

    async def test_a_dead_worker_fails_its_own_query_alone(self, store: SharedArrowStore, tmp_path: Path) -> None:
        sql = "SELECT state, sum(votes) AS v FROM results WHERE state = ? GROUP BY state"
        gate = tmp_path / "gate"
        queries = CopyQueries(store, workers=3, max_waiting=0, declared=_DECLARED)
        try:
            # two queries running in two workers, then the third worker dies under its own query
            started = [tmp_path / "a", tmp_path / "b"]
            running = [
                asyncio.create_task(queries.run_named("held", [str(mark), str(gate), sql, state]))
                for mark, state in zip(started, ("AZ", "GA"), strict=True)
            ]
            await _until(lambda: all(mark.exists() for mark in started))
            with pytest.raises(QueryWorkerDied):
                await queries.run_named("dies")
            gate.touch()
            for task, state in zip(running, ("AZ", "GA"), strict=True):
                answer = await task
                assert (await _answer(answer)).equals(_in_process(store, "c-0", sql, [state]))
            assert queries.status().died == 1
            assert queries.status().answered == 2
        finally:
            await queries.close()


class TestArrowEndToEnd:
    async def test_an_answer_crosses_as_arrow_never_python_objects_a_row(
        self, store: SharedArrowStore, queries: CopyQueries
    ) -> None:
        await _publish(store, "c-1", _rows(400_000, 3))
        await _worker_pid(queries)
        allocated = pa.total_allocated_bytes()
        tracemalloc.start()
        try:
            answer = await queries.run("SELECT * FROM results")
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        rows = answer.table
        assert rows.num_rows == 400_000
        # the rows are the worker's files mapped here, never pickled (a pickle is Python bytes) nor copied into
        # Arrow's own memory
        assert rows.nbytes > 10 << 20
        assert peak < 1 << 20, f"{peak} bytes of Python objects for {rows.nbytes} bytes of rows"
        assert pa.total_allocated_bytes() - allocated < rows.nbytes // 10
        await answer.aclose()


_KEY = ["state", "office_key", "candidate_id"]


class _RowsNamed:
    """every call this process makes, on any thread, to the lookup's probe (``rows_named``) or to ``duckdb.connect``."""

    def __init__(self) -> None:
        self.probes = 0
        self.connects = 0
        self._lock = threading.Lock()

    def __call__(self, frame: FrameType, event: str, arg: Any) -> None:
        if event == "call" and frame.f_code.co_name == "rows_named":
            with self._lock:
                self.probes += 1
        elif event == "c_call" and arg is duckdb.connect:
            with self._lock:
                self.connects += 1

    def __enter__(self) -> _RowsNamed:
        sys.setprofile(self)
        threading.setprofile_all_threads(self)
        return self

    def __exit__(self, *exc: object) -> None:
        threading.setprofile_all_threads(None)
        sys.setprofile(None)


def _keys(count: int) -> pa.Table:
    """``count`` keys: the held rows' keys, each named as often as it takes, and one key no row holds."""
    held = _rows(20_000, 1).select(_KEY)
    if count == 0:
        return held.slice(0, 0)
    named = held.take(pa.array(np.arange(count - 1) % held.num_rows))
    absent = pa.table({"state": ["AZ"], "office_key": ["none"], "candidate_id": pa.array([-1], pa.int64())})
    return pa.concat_tables([named, absent])


def _held_rows_of(store: SharedArrowStore, keys: pa.Table) -> pa.Table:
    sql = (
        "SELECT r.* FROM results r SEMI JOIN keys k ON r.state = k.state AND r.office_key = k.office_key "
        "AND r.candidate_id = k.candidate_id ORDER BY r.candidate_id"
    )
    return _in_process(store, "c-0", sql, [], keys=keys)


class TestKeyedLookups:
    @pytest.mark.parametrize("count", [1, 2, 1_000])
    async def test_a_lookup_runs_in_a_worker_however_few_its_keys(
        self, queries: CopyQueries, store: SharedArrowStore, count: int
    ) -> None:
        """Flips if a lookup, even of one key, probes the copy in the serving process: inline lost to the worker's round
        trip at every size measured, so there is no inline path."""
        await _worker_pid(queries)
        keys = _keys(count)
        with _RowsNamed() as calls:
            answer = await queries.lookup("results", keys, _KEY)
        assert answer.worker != os.getpid()
        assert calls.probes == 0 and calls.connects == 0
        found = (await _answer(answer)).sort_by("candidate_id")
        assert found.equals(_held_rows_of(store, keys))

    async def test_a_lookup_reads_the_version_it_names(self, queries: CopyQueries, store: SharedArrowStore) -> None:
        keys = _keys(10)
        lease = await asyncio.to_thread(store.lease, "c-0")
        try:
            await _publish(store, "c-1", _rows(5, 9))
            answer = await queries.lookup("results", keys, _KEY, version="c-0")
            assert answer.version == "c-0"
            assert (await _answer(answer)).sort_by("candidate_id").equals(_held_rows_of(store, keys))
        finally:
            await asyncio.to_thread(lease.release)

    @pytest.mark.parametrize("count", [1, 2])
    async def test_a_lookup_of_a_table_the_copy_lacks_is_refused(self, queries: CopyQueries, count: int) -> None:
        with pytest.raises(LookupError, match="no table"):
            await queries.lookup("absent", _keys(count), _KEY)
        with pytest.raises(KeyError):
            await queries.lookup("results", _keys(count).select(["state"]), _KEY)

    async def test_a_lookup_in_a_worker_crosses_as_arrow(self, queries: CopyQueries) -> None:
        await _worker_pid(queries)
        keys = _keys(20_001)
        tracemalloc.start()
        try:
            answer = await queries.lookup("results", keys, _KEY)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert answer.table.num_rows == 20_000
        assert peak < 1 << 20, f"{peak} bytes of Python objects for {keys.num_rows} keys"
        await answer.aclose()


def test_the_workers_are_the_pods_cpus_less_those_held_back() -> None:
    # reports' pods: 6 CPUs, less the serving process, the apply worker and the reload worker
    assert query_workers(6) == 3
    assert query_workers(6, held_back=5) == 1
    assert query_workers(2) == 1
    assert query_workers() >= 1
