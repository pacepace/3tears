"""Readers of the L1 never wait out a writer (SQ1 audit item 5, bench_opts.py, as tests).

The L1 was a memdb database, whose lock covers the whole database: a reader that met a writer on another thread slept
in SQLite's busy handler, 11-36 ms an overlap, and a writer that never paused starved readers for seconds (2,552 ms
p50 on Linux tmpfs, 5,202 ms on macOS; bench_opts_linux.log, bench_opts.log). The L1 is now a WAL file: a reader reads
the last commit while one writer writes, measured at 0.86-3.45 ms max read with the same writers. The bound here is
wall clock, never the reader thread's CPU (a reader waiting on a lock sleeps and uses none: turn_holds cannot see it).
"""

from __future__ import annotations

import sqlite3
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest
from sqlalchemy import Column, MetaData, String, Table

from threetears.core.cache.l1_lru import L1Ledger
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.testing.l1_tables import L1_EVICTION_INFO, TEST_L1_TABLE

_ROWS = 10_000
_PAD = "x" * 400
#: the longest one point read may take while a writer rewrites 10k keys back to back: the WAL file's measured max was
#: 1.59 ms (Linux tmpfs) and 1.64 ms (macOS); memdb's was 2,770 ms and 5,202 ms. 15 ms leaves a loaded runner room
#: and is still two orders under what memdb did.
#:
#: The writer is another process writing the same file (WAL works across processes), so what is measured is SQLite's
#: wait alone. A writer THREAD in the reader's process adds the interpreter lock's handoffs (up to the 5 ms switch
#: interval for each call into SQLite that gives the lock up): up to 31 ms measured on a laptop with a writer doing
#: Python work per row, which no storage choice removes; CPU work belongs in a worker process.
_READ_MAX_SECONDS = 0.015
#: a write may cost this much more on the file than on memdb: measured 0.4-1.3x on tmpfs, 1.2-1.9x on macOS APFS
_WRITE_RATIO_MAX = 2.5


def _backend(directory: Path) -> SQLiteBackend:
    metadata = MetaData()
    Table(
        "wal_rows",
        metadata,
        Column("id", String, primary_key=True),
        Column("member_id", String),
        Column("body", String),
        info={L1_EVICTION_INFO: TEST_L1_TABLE},
    )
    backend = SQLiteBackend(
        db_name=f"wal_{uuid.uuid4().hex[:8]}", directory=directory, budget=L1Ledger(1 << 34, thread=False)
    )
    backend.initialize(metadata)
    backend.upsert_many("wal_rows", [{"id": f"k{i:08d}", "member_id": "p", "body": _PAD} for i in range(_ROWS)])
    return backend


def _rewrite(conn: object, deletes: list[tuple[str]], rows: list[tuple[str, str, str]]) -> None:
    """one writer's transaction: delete and re-insert the keys, as an eviction then a pull-through do; the same
    statements on any connection, so the storage is the only difference."""
    conn.execute("BEGIN IMMEDIATE")  # type: ignore[attr-defined]
    conn.executemany("DELETE FROM wal_rows WHERE id = ?", deletes)  # type: ignore[attr-defined]
    conn.executemany("INSERT INTO wal_rows VALUES (?,?,?)", rows)  # type: ignore[attr-defined]
    conn.execute("COMMIT")  # type: ignore[attr-defined]


_WRITER = """
import sqlite3, sys
conn = sqlite3.connect(sys.argv[1], isolation_level=None)
conn.execute("PRAGMA busy_timeout=5000")
conn.execute("PRAGMA synchronous=OFF")
keys = [f"k{i:08d}" for i in range(int(sys.argv[2]))]
deletes = [(key,) for key in keys]
rows = [(key, "w", "x" * 400) for key in keys]
print("writing", flush=True)
while True:
    conn.execute("BEGIN IMMEDIATE")
    conn.executemany("DELETE FROM wal_rows WHERE id = ?", deletes)
    conn.executemany("INSERT INTO wal_rows VALUES (?,?,?)", rows)
    conn.execute("COMMIT")
    print("w", flush=True)
"""


@pytest.mark.loop_timing
class TestAReaderNeverWaitsOutAWriter:
    @pytest.mark.parametrize("reader", ["statement", "backend"])
    def test_point_reads_stay_fast_while_a_writer_never_pauses(self, tmp_path: Path, reader: str) -> None:
        backend = _backend(tmp_path)
        keys = [f"k{i:08d}" for i in range(_ROWS)]
        writer = subprocess.Popen(
            [sys.executable, "-c", _WRITER, str(backend.file_path), str(_ROWS)], stdout=subprocess.PIPE, text=True
        )
        try:
            assert writer.stdout is not None and writer.stdout.readline().strip() == "writing"
            own = backend.get_connection()
            reads: list[float] = []
            end = time.perf_counter() + 2.0
            index = 0
            while time.perf_counter() < end:
                key = keys[(index * 7919) % _ROWS]
                started = time.perf_counter()
                if reader == "statement":
                    own.execute("SELECT * FROM wal_rows WHERE id = ?", (key,)).fetchall()
                else:
                    backend.select_by_id("wal_rows", key)
                reads.append(time.perf_counter() - started)
                index += 1
                time.sleep(0.0005)
        finally:
            writer.kill()
            out, _ = writer.communicate(timeout=30)
        backend.reset()
        writes = out.count("w")
        assert writes >= 3, f"the writer wrote {writes} times: it did not run beside the reader"
        # starved, memdb gave 4 reads in 4 s; a reader that never waits makes one every 0.5 ms sleep or so
        assert len(reads) > 500, f"only {len(reads)} reads in 2 s: the reader was starved"
        assert max(reads) < _READ_MAX_SECONDS, (
            f"a read waited {max(reads) * 1000:.1f} ms behind the writer (p50 {statistics.median(reads) * 1000:.3f} ms)"
        )

    def test_a_write_costs_about_what_it_did_on_memdb(self, tmp_path: Path) -> None:
        backend = _backend(tmp_path)
        keys = [f"k{i:08d}" for i in range(_ROWS)]
        memdb = sqlite3.connect(f"file:/wal_baseline_{uuid.uuid4().hex[:8]}?vfs=memdb", uri=True, isolation_level=None)
        memdb.execute("CREATE TABLE wal_rows (id TEXT PRIMARY KEY, member_id TEXT, body TEXT)")
        memdb.executemany("INSERT INTO wal_rows VALUES (?,?,?)", [(key, "p", _PAD) for key in keys])

        deletes = [(key,) for key in keys]
        rows = [(key, "w", _PAD) for key in keys]

        def best(work: object, rounds: int = 5) -> float:
            taken = []
            for _ in range(rounds):
                started = time.perf_counter()
                work()  # type: ignore[operator]
                taken.append(time.perf_counter() - started)
            return min(taken)

        on_memdb = best(lambda: _rewrite(memdb, deletes, rows))
        on_file = best(lambda: _rewrite(backend.get_connection(), deletes, rows))
        backend.reset()
        memdb.close()
        assert on_file < on_memdb * _WRITE_RATIO_MAX, (
            f"rewriting {_ROWS} rows took {on_file * 1000:.1f} ms on the WAL file, {on_memdb * 1000:.1f} ms on memdb"
        )
