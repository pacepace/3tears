"""The serving process's reads of the shared copy leave its event loop free.

The contracts. DuckDB scans a mapped table with the interpreter lock released: it takes the lock a few times a query
(to bind and to make the scan), never a batch at a time, so threads querying the copy at cobalt dev's size (4 M long
rows in 200 parts) never hold the event loop past 20 ms, while versions are published and mapped beside them. And the
module DuckDB scans Arrow through is imported when shared Arrow is, never by a serving thread's first read.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import threading
import time
from pathlib import Path

import pyarrow as pa
import pytest

from threetears.core.shared_arrow import Parts, SharedArrowStore

#: the most the event loop may run late while threads query the copy
LOOP_MAY_BE_LATE = 0.020

#: cobalt dev's long rows (2026-10-23) and the parts they are held in
DEV_ROWS = 4_000_000
DEV_PARTS = 200

#: the threads querying the copy at once
READERS = 4

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


@pytest.mark.loop_timing
class TestTheLoopRunsWhileThreadsQueryTheCopy:
    async def test_a_ticker_is_never_late_while_threads_query_dev_sized_versions(
        self, tmp_path: Path, parts: dict[str, pa.Table]
    ) -> None:
        store = SharedArrowStore(tmp_path / "shm", max_bytes=4 << 30)
        schema = next(iter(parts.values())).schema
        keys = pa.table({"k": pa.array(range(1000), pa.int64()), "n": pa.array(range(1000), pa.int64())})
        await asyncio.to_thread(store.publish, "v1", {"long": Parts(schema=schema, parts=parts), "keys": keys})
        reader = store.reader(prepare=_views)
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
        workers = [asyncio.create_task(asyncio.to_thread(query, index)) for index in range(READERS)]
        publishing = asyncio.create_task(asyncio.to_thread(publish))
        await asyncio.sleep(1.5)
        stop.set()
        await asyncio.gather(*workers, publishing)
        await asyncio.sleep(0.05)
        ticking = False
        await ticker
        reader.close()
        assert min(reads) > 0 and reader.mapped > 1, (reads, reader.mapped)
        print(f"reads {reads}, versions mapped {reader.mapped}, late {late * 1000:.1f} ms")
        assert late < LOOP_MAY_BE_LATE, f"the loop ran {late * 1000:.1f} ms late while threads queried the copy"


class TestTheScanModuleIsImportedWithSharedArrow:
    def test_importing_shared_arrow_imports_what_duckdb_scans_arrow_through(self) -> None:
        """DuckDB imports ``pyarrow.dataset`` at a process's first scan of a registered table (25-70 ms); importing
        shared Arrow does it first, so no serving thread's read pays it."""
        code = "import sys, threetears.core.shared_arrow; print('pyarrow.dataset' in sys.modules)"
        answer = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        assert answer.stdout.strip() == "True"
