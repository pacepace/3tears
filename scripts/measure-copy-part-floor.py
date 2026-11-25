#!/usr/bin/env python
"""Measure the size a copied table is split into per-scope parts at (``copy_parts.SPLIT_FLOOR_BYTES``).

A split table's change rewrites one part, not the table; but every part is one more Arrow chunk a reader's DuckDB
scans. This measures both sides at sizes from 1k to 1M rows, so the floor sits where the first pays for the second.

``rebuild``
    what the apply does to the part a change lands in: the held rows less those the change names
    (``arrow_keys.rows_named``), plus the change's rows, in key order, written as one Arrow IPC file to memory (the
    copy's store is tmpfs). ``whole`` is the table held as one part; ``part`` is one state's part of 51
    (``copy_parts.split_parts``). The change is 100 rows of one state, half of them replacing held rows.
``scan``
    DuckDB over the registered Arrow table, as a query worker reads a version: one chunk (one part) against 51
    chunks (the parts concatenated). Three queries: one state's group-by, every state's group-by, one race's rows.

Each time is the fastest of its runs, as the machine is shared. DuckDB runs on one thread (a query worker's).

Run::

    uv run python scripts/measure-copy-part-floor.py [rows ...]
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from typing import Any, Final

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc

from threetears.core.arrow_keys import rows_named
from threetears.core.collections.copy_parts import PartRule, split_parts

_STATES: Final = pa.array([f"S{i:02d}" for i in range(51)], pa.string())
_KEY: Final = ("state", "office_key", "candidate_id")
_RULE: Final = PartRule("results", _KEY, scope_column="state")
_SIZES: Final = (1_000, 3_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000)
_QUERIES: Final = {
    "one_state": "SELECT office_key, sum(votes) FROM t WHERE state = 'S17' GROUP BY 1",
    "all_states": "SELECT state, count(*), sum(votes) FROM t GROUP BY 1",
    "one_race": "SELECT * FROM t WHERE office_key = 'race-00000003'",
}


def _rows(count: int, seed: int, first_id: int = 0) -> pa.Table:
    rng = np.random.default_rng(seed)
    ids = pa.array(np.arange(first_id, first_id + count, dtype=np.int64))
    office = pc.cast(pa.array(rng.integers(0, max(count // 8, 1), count) + 100_000_000), pa.string())
    return pa.table(
        {
            "state": _STATES.take(pa.array(np.sort(rng.integers(0, len(_STATES), count)))),
            "office_key": pc.binary_join_element_wise("race-", pc.utf8_slice_codeunits(office, 1), ""),
            "candidate_id": ids,
            "votes": pa.array(rng.integers(0, 1_000_000, count)),
            "pct": pa.array(rng.random(count)),
            "name": pc.binary_join_element_wise("candidate ", pc.cast(ids, pa.string()), ""),
            "updated": pa.array(
                rng.integers(1_700_000_000_000_000, 1_800_000_000_000_000, count), pa.timestamp("us", "UTC")
            ),
        }
    ).combine_chunks()


def _change(held: pa.Table, state: str, count: int = 100) -> pa.Table:
    replaced = held.filter(pc.equal(held["state"], state)).slice(0, count // 2)
    replaced = replaced.set_column(3, "votes", pc.add(replaced["votes"], 1))
    added = _rows(count - replaced.num_rows, seed=99, first_id=10_000_000_000)
    added = added.set_column(0, "state", pa.repeat(pa.scalar(state, pa.string()), added.num_rows))
    return pa.concat_tables([replaced, added])


def _mapped(rows: pa.Table) -> pa.Table:
    sink = pa.BufferOutputStream()
    with ipc.new_file(sink, rows.schema) as writer:
        writer.write_table(rows)
    return ipc.open_file(pa.BufferReader(sink.getvalue())).read_all()


def _rebuild(held: pa.Table, change: pa.Table) -> int:
    kept = np.ones(held.num_rows, dtype=bool)
    kept[rows_named(held, change, _KEY)] = False
    rows = pa.concat_tables([held.filter(pa.array(kept)), change])
    rows = rows.take(pc.sort_indices(rows, sort_keys=[(column, "ascending") for column in _KEY]))
    sink = pa.BufferOutputStream()
    with ipc.new_file(sink, rows.schema) as writer:
        writer.write_table(rows)
    return int(sink.getvalue().size)


def _fastest(run: Callable[[], Any], runs: int) -> float:
    run()
    best = float("inf")
    for _ in range(runs):
        start = time.perf_counter()
        run()
        best = min(best, time.perf_counter() - start)
    return best * 1000


def _scan(rows: pa.Table, sql: str) -> float:
    connection = duckdb.connect()
    try:
        connection.execute("SET threads = 1")
        connection.register("t", rows)
        return _fastest(lambda: connection.execute(sql).to_arrow_table(), 60)
    finally:
        connection.close()


def main() -> None:
    sizes = [int(arg) for arg in sys.argv[1:]] or list(_SIZES)
    print("rows,mb,whole_rebuild_ms,part_rebuild_ms,saved_ms,query,scan_1_ms,scan_51_ms,scan_cost_ms")
    for count in sizes:
        table = _rows(count, seed=7)
        parts = split_parts(_RULE, table)
        one = next(part.rows for part in parts if part.name == "S17")
        change = _change(table, "S17")
        runs = 40 if count <= 200_000 else 12
        held_whole, held_one = _mapped(table), _mapped(one)
        whole_ms = _fastest(lambda: _rebuild(held_whole, change), runs)
        part_ms = _fastest(lambda: _rebuild(held_one, change), runs)
        held_parts = pa.concat_tables([_mapped(part.rows) for part in parts])
        for name, sql in _QUERIES.items():
            scan_one, scan_split = _scan(held_whole, sql), _scan(held_parts, sql)
            print(
                f"{count},{table.nbytes / 1e6:.2f},{whole_ms:.3f},{part_ms:.3f},{whole_ms - part_ms:.3f},{name},"
                f"{scan_one:.3f},{scan_split:.3f},{scan_split - scan_one:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
