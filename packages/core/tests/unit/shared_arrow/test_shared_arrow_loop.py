"""The serving process's reads of the shared copy leave its event loop free.

The contracts. DuckDB scans a mapped table with the interpreter lock released: it takes the lock a few times a query
(to bind and to make the scan), never a batch at a time, so threads querying the copy at cobalt dev's size (4 M long
rows in 200 parts, each read's DuckDB on a serving engine's 2 threads) never hold the event loop past 20 ms beyond
what the host's own idle loop is late at the same moments, while versions are published and mapped beside them. And the
module DuckDB scans Arrow through is imported when shared Arrow is, never by a serving thread's first read.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

import pyarrow as pa
import pytest

from threetears.core.shared_arrow import Parts, SharedArrowStore
from threetears.core.testing.loop_lateness import (
    HostTicker,
    Tick,
    added_lateness,
    hold_the_loop,
    latest,
    ticks_while,
    trusted,
)

#: the most the event loop may run late while threads query the copy, past what the host's own idle loop, ticking in a
#: process of its own, was late at the same moments (:func:`added_lateness`: the runner's own scheduling delay is no
#: property of the reads)
LOOP_MAY_BE_LATE = 0.020

#: what the test loop's ticker asks to sleep
TICK = 0.002

#: cobalt dev's long rows (2026-10-23) and the parts they are held in
DEV_ROWS = 4_000_000
DEV_PARTS = 200

#: the threads querying the copy at once
READERS = 4

#: the threads each read's DuckDB runs on: a serving process's engines run a fixed few, never one per core of the node
#: (reports' serving engines run 2: engines sized to the cores filled the pod's 4-CPU quota and every throttled period
#: froze its loop, prod 2026-10-31)
READER_THREADS = 2

#: the queries each reader runs, as the pod's reads do: an aggregate, a filter, a point read, a join
QUERIES = (
    "SELECT k, sum(v), count(*) FROM long GROUP BY k",
    "SELECT count(*) FROM long WHERE s = 's7' AND v > 0.5",
    "SELECT * FROM long WHERE k = 17 LIMIT 100",
    "SELECT count(*) FROM long l JOIN keys USING (k) WHERE keys.n < 10",
)


def _part(seed: int, rows: int) -> pa.Table:
    """one part of the long table: a key, a value and a text column.

    :return: the part
    :rtype: pa.Table
    """
    numbers = range(seed, seed + rows)
    return pa.table(
        {
            "k": pa.array([n % 1000 for n in numbers], pa.int64()),
            "v": pa.array([(n % 997) / 997 for n in numbers], pa.float64()),
            "s": pa.array([f"s{n % 50}" for n in numbers], pa.string()),
        }
    )


@pytest.fixture(scope="module")
def parts() -> dict[str, pa.Table]:
    """the long table's parts at dev's size, made once (one part's rows in every part: each its own file).

    :return: part name -> part
    :rtype: dict[str, pa.Table]
    """
    part = _part(0, DEV_ROWS // DEV_PARTS)
    return {f"p{index:04d}": part for index in range(DEV_PARTS)}


def _views(connection: object) -> None:
    """each version's views, as a serving process's ``prepare`` makes them.

    :param connection: the version's DuckDB connection
    :ptype connection: duckdb.DuckDBPyConnection
    """
    connection.execute("CREATE VIEW by_key AS SELECT k, sum(v) AS v FROM long GROUP BY k")  # type: ignore[attr-defined]


async def _queried_while_published(
    tmp_path: Path, parts: dict[str, pa.Table], *, planted: Callable[[], None] | None = None
) -> tuple[list[Tick], list[Tick], list[int], int]:
    """the test loop's ticks and the host's idle loop's while 4 threads query dev-sized versions and new ones are
    published and mapped beside them, for 1.5 s.

    :param planted: called on the loop 0.5 s in (a red case plants its blocking here)
    :return: the loop's ticks, the host's ticks, each reader's reads and the versions mapped
    """
    # a store of its own each measurement: a retaken one publishes its versions afresh
    store = SharedArrowStore(Path(tempfile.mkdtemp(dir=tmp_path)) / "shm", max_bytes=4 << 30)
    schema = next(iter(parts.values())).schema
    keys = pa.table({"k": pa.array(range(1000), pa.int64()), "n": pa.array(range(1000), pa.int64())})
    await asyncio.to_thread(store.publish, "v1", {"long": Parts(schema=schema, parts=parts), "keys": keys})
    reader = store.reader(config={"threads": READER_THREADS}, prepare=_views)
    stop = threading.Event()
    reads = [0] * READERS

    def query(index: int) -> None:
        while not stop.is_set():
            with reader.read() as read:
                for sql in QUERIES:
                    read.cursor.execute(sql).fetchall()
            reads[index] += 1

    changes = {version: _part(version, DEV_ROWS // DEV_PARTS) for version in range(2, 5)}

    def publish() -> None:
        # a version a change makes: one part replaced, the rest linked; the readers' next reads map it
        for version, part in changes.items():
            if stop.wait(0.3):
                return
            replaced = Parts(schema=schema, parts={"p0000": part})
            store.publish(f"v{version}", {"long": replaced}, base=f"v{version - 1}")

    async def run() -> None:
        workers = [asyncio.create_task(asyncio.to_thread(query, index)) for index in range(READERS)]
        publishing = asyncio.create_task(asyncio.to_thread(publish))
        if planted is not None:
            asyncio.get_running_loop().call_later(0.5, planted)
        await asyncio.sleep(1.5)
        stop.set()
        await asyncio.gather(*workers, publishing)
        await asyncio.sleep(0.05)

    try:
        with HostTicker() as host:
            ticks = await ticks_while(run, every=TICK)
    finally:
        stop.set()
        reader.close()
    return ticks, host.ticks, reads, reader.mapped


def _window(measured: tuple[list[Tick], list[Tick], list[int], int]) -> tuple[list[Tick], list[Tick]]:
    """a measurement's test loop and host loop ticks."""
    return measured[0], measured[1]


@pytest.mark.loop_timing
class TestTheLoopRunsWhileThreadsQueryTheCopy:
    async def test_a_ticker_is_never_late_while_threads_query_dev_sized_versions(
        self, tmp_path: Path, parts: dict[str, pa.Table]
    ) -> None:
        ticks, host, reads, mapped = await trusted(
            lambda: _queried_while_published(tmp_path, parts),
            _window,
            host_may_be_late=LOOP_MAY_BE_LATE,
            threads=READERS * READER_THREADS,
        )
        assert min(reads) > 0 and mapped > 1, (reads, mapped)
        added = added_lateness(ticks, host, every=TICK)
        late = latest(ticks, every=TICK)
        print(
            f"reads {reads}, versions mapped {mapped}, late {late * 1000:.1f} ms, past the host {added * 1000:.1f} ms"
        )
        assert added < LOOP_MAY_BE_LATE, (
            f"the loop ran {added * 1000:.1f} ms late while threads queried the copy, past the host's own idle loop "
            f"({late * 1000:.1f} ms late in all)"
        )

    async def test_a_block_planted_on_the_loop_is_caught(self, tmp_path: Path, parts: dict[str, pa.Table]) -> None:
        """the judgement's red case: 30 ms of blocking on the loop itself while the threads query fails it; the
        host's idle loop, in its own process, is not late for it, so it credits none of it."""
        ticks, host, _, _ = await trusted(
            lambda: _queried_while_published(tmp_path, parts, planted=lambda: hold_the_loop(0.03)),
            _window,
            host_may_be_late=LOOP_MAY_BE_LATE,
            threads=READERS * READER_THREADS,
        )
        added = added_lateness(ticks, host, every=TICK)
        assert added >= LOOP_MAY_BE_LATE, f"a 30 ms block on the loop was judged {added * 1000:.1f} ms"


class TestTheScanModuleIsImportedWithSharedArrow:
    def test_importing_shared_arrow_imports_what_duckdb_scans_arrow_through(self) -> None:
        """DuckDB imports ``pyarrow.dataset`` at a process's first scan of a registered table (25-70 ms); importing
        shared Arrow does it first, so no serving thread's read pays it."""
        code = "import sys, threetears.core.shared_arrow; print('pyarrow.dataset' in sys.modules)"
        answer = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        assert answer.stdout.strip() == "True"
