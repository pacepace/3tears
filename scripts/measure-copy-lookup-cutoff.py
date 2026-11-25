#!/usr/bin/env python
"""Measure the keys below which a keyed lookup of the copy runs in the serving process (``copy_queries.
INLINE_LOOKUP_KEYS``), by the method of ``copy_key_work.INLINE_KEYS`` (``inventory-copy-worker-backend.md`` section 4).

A table of ``rows`` rows in 51 state parts is published in a shared Arrow store. For 1 to 100k keys (each a held row's
key, of three columns):

``inline``
    the lookup in this process (``copy_queries._inline_lookup``, as the pool runs it on a thread): the version leased,
    the table's part files found in its manifest and mapped, each part probed (``arrow_keys.rows_named``), the rows
    found copied out. Its CPU is this thread's (``time.thread_time``), all of it the serving process's.
``worker``
    the same lookup through a query worker (``CopyQueries.lookup`` with every lookup sent there): the keys handed over,
    the job pickled, its answer claimed and mapped. Its cost to the serving process is this process's CPU across the
    round trip (``time.process_time``, every thread), beside its wall time and the event loop's worst lateness on a 1 ms
    ticker while it runs.

Each figure is the median of its runs. The cut-off is the largest key count whose inline CPU stays below the worker
round trip's serving-process CPU and whose inline wall time stays under 2 ms.

Run::

    uv run python scripts/measure-copy-lookup-cutoff.py [rows ...]
"""

from __future__ import annotations

import asyncio
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Final

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from threetears.core.collections.copy_queries import CopyQueries, _inline_lookup
from threetears.core.shared_arrow import Parts, SharedArrowStore

_STATES: Final = pa.array([f"S{i:02d}" for i in range(51)], pa.string())
_KEY: Final = ("state", "office_key", "candidate_id")
_COUNTS: Final = (1, 10, 100, 1_000, 10_000, 100_000)
_RUNS: Final = 9
_INLINE_WALL_BAR: Final = 0.002


def _rows(count: int) -> pa.Table:
    rng = np.random.default_rng(1)
    race = rng.integers(0, max(count // 8, 1), count)
    return pa.table(
        {
            "state": _STATES.take(pa.array(race % len(_STATES))),
            "office_key": pc.cast(pa.array(race), pa.string()),
            "candidate_id": pa.array(np.arange(count), pa.int64()),
            "votes": pa.array(rng.integers(0, 1_000_000, count), pa.int64()),
            "name": pc.cast(pa.array(rng.integers(0, 1 << 30, count)), pa.string()),
        }
    )


def _published(store: SharedArrowStore, rows: pa.Table) -> None:
    order = pc.sort_indices(rows["state"])
    ordered = rows.take(order)
    parts = {}
    for state in _STATES.to_pylist():
        mask = pc.equal(ordered["state"], state)
        parts[state] = ordered.filter(mask)
    store.publish("c-0", {"results": Parts(rows.schema, parts)})


def _inline(store: SharedArrowStore, keys: pa.Table) -> tuple[float, float]:
    cpus, walls = [], []
    for _ in range(_RUNS):
        cpu, wall = time.thread_time(), time.perf_counter()
        found, _ = _inline_lookup(store, None, "results", keys, _KEY)
        cpus.append(time.thread_time() - cpu)
        walls.append(time.perf_counter() - wall)
        assert found.num_rows == keys.num_rows
    return statistics.median(cpus), statistics.median(walls)


async def _worker(queries: CopyQueries, keys: pa.Table) -> tuple[float, float, float]:
    cpus, walls, lags = [], [], []
    for _ in range(_RUNS):
        late = 0.0
        running = True

        async def tick() -> None:
            nonlocal late
            loop = asyncio.get_running_loop()
            while running:
                due = loop.time() + 0.001
                await asyncio.sleep(0.001)
                late = max(late, loop.time() - due)

        ticker = asyncio.create_task(tick())
        await asyncio.sleep(0.005)
        cpu, wall = time.process_time(), time.perf_counter()
        answer = await queries.lookup("results", keys, _KEY)
        cpus.append(time.process_time() - cpu)
        walls.append(time.perf_counter() - wall)
        running = False
        await ticker
        lags.append(late)
        assert answer.table.num_rows == keys.num_rows
        await answer.aclose()
    return statistics.median(cpus), statistics.median(walls), max(lags)


async def main(count: int) -> None:
    rows = _rows(count)
    rng = np.random.default_rng(2)
    with tempfile.TemporaryDirectory() as root:
        store = SharedArrowStore(Path(root) / "shm", max_bytes=8 << 30)
        await asyncio.to_thread(_published, store, rows)
        queries = CopyQueries(store, workers=1, inline_keys=0)
        try:
            warm = rows.slice(0, 10).select(list(_KEY))
            await (await queries.lookup("results", warm, _KEY)).aclose()
            print(f"{count:,} rows in 51 parts, {rows.nbytes / 2**20:.0f} MiB; medians of {_RUNS} runs (ms)")
            print(f"{'keys':>8} {'inline cpu':>11} {'inline wall':>12} {'worker cpu':>11} {'worker wall':>12} {'loop late':>10}")
            cut = 0
            for keys_count in _COUNTS:
                if keys_count > count:
                    continue
                keys = rows.take(pa.array(rng.choice(count, keys_count, replace=False))).select(list(_KEY))
                inline_cpu, inline_wall = _inline(store, keys)
                worker_cpu, worker_wall, lag = await _worker(queries, keys)
                print(
                    f"{keys_count:>8,} {inline_cpu * 1e3:>11.3f} {inline_wall * 1e3:>12.3f} {worker_cpu * 1e3:>11.3f}"
                    f" {worker_wall * 1e3:>12.3f} {lag * 1e3:>10.3f}"
                )
                if inline_cpu < worker_cpu and inline_wall < _INLINE_WALL_BAR:
                    cut = keys_count
            print(f"largest count inline wins at: {cut:,}")
        finally:
            await queries.close()


if __name__ == "__main__":
    for size in [int(arg) for arg in sys.argv[1:]] or [20_000, 200_000, 2_000_000]:
        asyncio.run(main(size))
