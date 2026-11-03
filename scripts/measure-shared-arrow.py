#!/usr/bin/env python
"""Measure what shared-memory Arrow saves: one copy of the data for a serving process and N workers.

``docs/shared-arrow.md`` claims the workers of a :class:`~threetears.core.shared_arrow.SharedArrowPool` share ONE
physical copy of a version. This measures it at a real size, on Linux, where ``/proc/<pid>/smaps_rollup`` charges
each process its proportional share (PSS) of every page it maps: the sum of the processes' PSS is the physical memory
they hold together, shared pages counted once.

Four layouts, the same data:

``today``
    one process holding the tables in an in-memory DuckDB (a scoped snapshot's L1): its PSS.
``shared``
    that process, plus the tables published once to the memory-backed store, plus N workers each scanning every
    column of every table through DuckDB over the mapped Arrow, all mapped at once: each process's PSS, and the sum.
``naive``
    each worker loading its own DuckDB copy instead: one worker measured, the total as the serving process plus N of
    it.
``store``
    the serving process reading the store itself (:meth:`SharedArrowStore.reader`), its DuckDB copy kept only to
    apply changes and held compressed (``DuckDBBackend(compressed=True)``), published a scope's part at a time
    (:class:`~threetears.core.collections.shared_snapshot.SharedSnapshot`), plus N workers: each process's PSS and the
    sum, measured in a fresh process. Then what a change costs there against a plain copy: one scope's rows of the
    biggest table replaced (``--lsf-table``: an LSF run) and one scope of every results table (``--ap-tables``: an AP
    load), the checkpoint that compresses it, and the publish of that one scope.

Generated values compress better in DuckDB than real ones do (the reports' real extracts: a quarter of the plain copy;
these: about a tenth), so the ``store`` layout's copy is a floor; the pod's own memory is the proof.

The data is generated to a spec: ``{"table": {"rows": N, "columns": [["name", "DUCKDB TYPE"], ...]}}``, each column
filled by its type (text from a few thousand distinct values, numbers and instants varying by row). Run on Linux,
with the store on ``/dev/shm``::

    uv run python scripts/measure-shared-arrow.py --spec spec.json --workers 2 4 --root /dev/shm/measure

It prints a table and writes the figures as JSON (``--out``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import MappingProxyType
from typing import Any

import duckdb

from threetears.core.collections.scoped_snapshot import SnapshotTable, VersionedRead
from threetears.core.collections.shared_snapshot import SharedSnapshot
from threetears.core.shared_arrow import JobContext, SharedArrowPool, SharedArrowStore
from threetears.observe import get_logger, remove_tree

log = get_logger(__name__)

#: the scopes a generated scope column holds (states, and the national rows)
_SCOPES = 51

#: the smaps_rollup fields reported, in KiB as the kernel gives them
_ROLLUP_FIELDS = ("Rss", "Pss", "Pss_Anon", "Pss_File", "Pss_Shmem", "Shared_Clean", "Private_Clean", "Private_Dirty")


def _memory(pid: int | str = "self") -> dict[str, int]:
    """a process's memory, in MiB: its smaps_rollup fields and its status's RssAnon, RssFile and RssShmem.

    :param pid: the process; ``"self"`` for this one
    :ptype pid: int | str
    :return: field -> MiB
    :rtype: dict[str, int]
    """
    found: dict[str, int] = {}
    with open(f"/proc/{pid}/smaps_rollup") as rollup:
        for line in rollup:
            fields = line.split()
            if fields and fields[0].rstrip(":") in _ROLLUP_FIELDS:
                found[fields[0].rstrip(":")] = int(fields[1]) // 1024
    with open(f"/proc/{pid}/status") as status:
        for line in status:
            fields = line.split()
            if fields and fields[0].rstrip(":") in ("RssAnon", "RssFile", "RssShmem"):
                found[fields[0].rstrip(":")] = int(fields[1]) // 1024
    return found


def _filled(column: str, kind: str, scope: str | None = None, rows: int = 1) -> str:
    """the SQL that fills one column of a generated row ``i`` by its type.

    :param column: the column's name
    :ptype column: str
    :param kind: its DuckDB type
    :ptype kind: str
    :param scope: the scope column, whose values are :data:`_SCOPES` scopes, each a run of rows
    :ptype scope: str | None
    :param rows: the table's rows
    :ptype rows: int
    :return: the expression, aliased to the column
    :rtype: str
    """
    quoted = '"' + column.replace('"', '""') + '"'
    if column == scope:
        # a scope's rows side by side, as a scoped snapshot loads them (a scope at a time) and a change writes them
        value = f"'S' || ((i * {_SCOPES}) // {max(int(rows), 1)})::VARCHAR"
    elif kind == "VARCHAR":
        value = f"'{column[:6]}-' || ((i * 7919 + {len(column)}) % 4093)::VARCHAR"
    elif kind == "BIGINT":
        value = "(i * 31) % 1000003"
    elif kind == "BOOLEAN":
        value = "i % 3 = 0"
    elif kind.startswith("DECIMAL"):
        value = f"((i % 100000) / 1000.0)::{kind}"
    elif kind.startswith("TIMESTAMP"):
        value = "TIMESTAMPTZ '2026-11-04 00:00:00+00' + to_seconds(i % 86400)"
    else:
        value = f"NULL::{kind}"
    return f"{value} AS {quoted}"


def load(connection: Any, spec: dict[str, Any], scope: str | None = None) -> None:
    """create and fill every table of the spec in ``connection``.

    :param connection: a DuckDB connection
    :ptype connection: duckdb.DuckDBPyConnection
    :param spec: table -> rows and columns
    :ptype spec: dict[str, Any]
    :param scope: the scope column, filled with :data:`_SCOPES` values
    :ptype scope: str | None
    :return: nothing
    :rtype: None
    """
    for table, shape in spec.items():
        columns = ", ".join(_filled(name, kind, scope, int(shape["rows"])) for name, kind in shape["columns"])
        connection.execute(f'CREATE TABLE "{table}" AS SELECT {columns} FROM range({int(shape["rows"])}) r(i)')


def scan_all(context: JobContext, signals: str, readers: int) -> dict[str, Any]:
    """scan every column of every table through DuckDB over the mapped version, wait until every reader has, then
    measure this worker.

    :return: the worker's id, seconds scanning, and memory
    :rtype: dict[str, Any]
    """
    assert context.version is not None
    started = time.monotonic()
    connection = context.version.connect(config={"threads": 2})
    for table in context.version.tables:
        connection.execute(f'SELECT count(*), sum(hash(COLUMNS(*))) FROM "{table}"').fetchall()
    seconds = time.monotonic() - started
    where = Path(signals)
    (where / f"read-{os.getpid()}").touch()
    deadline = time.monotonic() + 600
    while len(list(where.glob("read-*"))) < readers and time.monotonic() < deadline:
        time.sleep(0.05)
    memory = _memory()
    # held until the serving process has measured every worker
    while not (where / "measured").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    connection.close()
    return {"pid": os.getpid(), "seconds": round(seconds, 2), "memory": memory}


def own_copy(context: JobContext, spec: dict[str, Any], scope: str | None = None) -> dict[str, Any]:
    """load a DuckDB copy of its own (the naive worker) and measure this worker.

    :return: the worker's memory
    :rtype: dict[str, Any]
    """
    connection = duckdb.connect(":memory:", config={"threads": 2})
    load(connection, spec, scope)
    memory = _memory()
    connection.close()
    return {"pid": os.getpid(), "memory": memory}


async def measure(spec: dict[str, Any], workers: int, root: Path, scope: str | None = None) -> dict[str, Any]:
    """the three layouts with ``workers`` workers.

    :return: the figures
    :rtype: dict[str, Any]
    """
    remove_tree(root, log=log, what="the measurement's scratch store")
    l1 = duckdb.connect(":memory:", config={"threads": 2})
    started = time.monotonic()
    load(l1, spec, scope)
    loaded = time.monotonic() - started
    today = _memory()
    store = SharedArrowStore(root / "store", max_bytes=64 << 30)
    started = time.monotonic()
    version = await asyncio.to_thread(
        store.publish_query, "v1", l1.cursor(), {table: f'SELECT * FROM "{table}"' for table in spec}
    )
    published = time.monotonic() - started
    signals = root / "signals"
    signals.mkdir(parents=True)
    pool = SharedArrowPool(store, workers=workers, private_bytes=64 << 30, name="measure")
    try:
        jobs = [asyncio.ensure_future(pool.run(scan_all, str(signals), workers)) for _ in range(workers)]
        while len(list(signals.glob("read-*"))) < workers:
            await asyncio.sleep(0.05)
        await asyncio.sleep(1.0)
        pids = [int(path.name.split("-")[1]) for path in signals.glob("read-*")]
        serving = _memory()
        held = {pid: _memory(pid) for pid in pids}
        (signals / "measured").touch()
        scans = [job.value for job in await asyncio.gather(*jobs)]
        naive = (await pool.run(own_copy, spec, scope, read=False)).value
    finally:
        await pool.close()
    # the store's pages counted once, whoever maps them (a page no process maps is charged to the pod all the same)
    shared_total = serving["Pss"] + sum(memory["Pss"] - memory["Pss_Shmem"] for memory in held.values())
    shared_total += version.nbytes >> 20
    return {
        "workers": workers,
        "rows": {table: int(shape["rows"]) for table, shape in spec.items()},
        "store_mib": version.nbytes >> 20,
        "seconds": {
            "load_l1": round(loaded, 1),
            "publish": round(published, 1),
            "scans": [scan["seconds"] for scan in scans],
        },
        "today": {"serving": today, "total_pss_mib": today["Pss"]},
        "shared": {"serving": serving, "workers": list(held.values()), "total_pss_mib": shared_total},
        "naive": {"worker": naive["memory"], "total_pss_mib": today["Pss"] + workers * naive["memory"]["Pss"]},
    }


def _compressed_copy(spec: dict[str, Any], scope: str) -> Any:
    """a DuckDB copy held compressed, as ``DuckDBBackend(compressed=True)`` holds it, filled and checkpointed.

    :return: the connection
    :rtype: duckdb.DuckDBPyConnection
    """
    l1 = duckdb.connect(":memory:", config={"threads": 2})
    l1.execute("ATTACH ':memory:' AS l1 (COMPRESS)")
    l1.execute("USE l1")
    load(l1, spec, scope)
    l1.execute("CHECKPOINT")
    return l1


def _cursor(l1: Any) -> Any:
    cursor = l1.cursor()
    cursor.execute("USE l1")
    return cursor


def _replace_scope(l1: Any, tables: list[str], value: str) -> float:
    """replace one scope's rows of ``tables`` by the same rows again, in one transaction: a change's write.

    :return: seconds
    :rtype: float
    """
    started = time.monotonic()
    l1.execute("BEGIN TRANSACTION")
    for table in tables:
        quoted = f'"{table}"'
        l1.execute(f'CREATE TEMP TABLE held AS SELECT * FROM {quoted} WHERE "{_SCOPE_OF[0]}" = ?', [value])
        l1.execute(f'DELETE FROM {quoted} WHERE "{_SCOPE_OF[0]}" = ?', [value])
        l1.execute(f"INSERT INTO {quoted} SELECT * FROM held")
        l1.execute("DROP TABLE held")
    l1.execute("COMMIT")
    return time.monotonic() - started


#: the scope column, for :func:`_replace_scope` (set by :func:`measure_store`)
_SCOPE_OF: list[str] = [""]


def _timed(action: Any) -> float:
    started = time.monotonic()
    action()
    return time.monotonic() - started


async def _measure_store(
    spec: dict[str, Any], workers: int, root: Path, scope: str, lsf_table: str, ap_tables: list[str]
) -> dict[str, Any]:
    """the ``store`` layout, in this (fresh) process: memory, then what one change costs.

    :return: the figures
    :rtype: dict[str, Any]
    """
    remove_tree(root, log=log, what="the measurement's scratch store")
    _SCOPE_OF[0] = scope
    started = time.monotonic()
    l1 = _compressed_copy(spec, scope)
    loaded = time.monotonic() - started
    scoped = [table for table, shape in spec.items() if any(name == scope for name, _ in shape["columns"])]
    whole = [table for table in spec if table not in scoped]
    epochs = {f"S{index}": 1 for index in range(_SCOPES)}
    store = SharedArrowStore(root / "store", max_bytes=64 << 30)
    reads = _cursor(l1)

    def extra(read: VersionedRead) -> dict[str, Any]:
        return {
            table: (lambda t=table: read.cursor.execute(f'SELECT * FROM "{t}"').to_arrow_reader(65536))
            for table in whole
        }

    shared = SharedSnapshot(
        store, [SnapshotTable(name=t, scope_column=scope, key=()) for t in scoped], extra_tables=extra
    )
    started = time.monotonic()
    version = await asyncio.to_thread(shared.publish, VersionedRead(cursor=reads, epochs=MappingProxyType(epochs)))
    published = time.monotonic() - started
    signals = root / "signals"
    signals.mkdir(parents=True)
    reader = store.reader(config={"threads": 2})
    pool = SharedArrowPool(store, workers=workers, private_bytes=64 << 30, name="measure")
    try:
        with reader.read() as read:
            started = time.monotonic()
            for table in read.tables:
                read.cursor.execute(f'SELECT count(*), sum(hash(COLUMNS(*))) FROM "{table}"').fetchall()
            serving_scan = time.monotonic() - started
            jobs = [asyncio.ensure_future(pool.run(scan_all, str(signals), workers)) for _ in range(workers)]
            while len(list(signals.glob("read-*"))) < workers:
                await asyncio.sleep(0.05)
            await asyncio.sleep(1.0)
            pids = [int(path.name.split("-")[1]) for path in signals.glob("read-*")]
            serving = _memory()
            held = {pid: _memory(pid) for pid in pids}
            (signals / "measured").touch()
            scans = [job.value for job in await asyncio.gather(*jobs)]
    finally:
        await pool.close()
    held_bytes = store.held_bytes()
    total = serving["Pss"] - serving.get("Pss_Shmem", 0)
    total += sum(memory["Pss"] - memory.get("Pss_Shmem", 0) for memory in held.values())
    total += held_bytes >> 20

    # one change, against a plain copy of the same tables
    plain = duckdb.connect(":memory:", config={"threads": 2})
    load(plain, {table: spec[table] for table in [lsf_table, *ap_tables]}, scope)
    timings: dict[str, Any] = {}
    for label, tables in (("lsf_update", [lsf_table]), ("ap_load", ap_tables)):
        plain_seconds = _replace_scope(plain, tables, "S7")
        compressed_seconds = _replace_scope(l1, tables, "S7")
        checkpoint = _timed(lambda: l1.execute("CHECKPOINT"))
        epochs = {**epochs, "S7": epochs["S7"] + 1}
        publish_one = _timed(
            lambda: shared.publish(VersionedRead(cursor=_cursor(l1), epochs=MappingProxyType(dict(epochs))))
        )
        timings[label] = {
            "tables": tables,
            "write_plain_s": round(plain_seconds, 3),
            "write_compressed_s": round(compressed_seconds, 3),
            "checkpoint_s": round(checkpoint, 3),
            "publish_one_scope_s": round(publish_one, 3),
        }
    reader.close()
    copy_mib = int(l1.execute("SELECT sum(memory_usage_bytes) FROM duckdb_memory()").fetchone()[0]) >> 20
    return {
        "workers": workers,
        "store_mib": held_bytes >> 20,
        "copy_compressed_mib": copy_mib,
        "seconds": {
            "load_l1_and_checkpoint": round(loaded, 1),
            "publish_every_scope": round(published, 1),
            "serving_scan": round(serving_scan, 2),
            "scans": [scan["seconds"] for scan in scans],
        },
        "store": {"serving": serving, "workers": list(held.values()), "total_pss_mib": total},
        "change": timings,
        "version_bytes": version.nbytes,
    }


def _store_in_a_fresh_process(*arguments: Any) -> dict[str, Any]:
    return asyncio.run(_measure_store(*arguments))


def measure_store(
    spec: dict[str, Any], workers: int, root: Path, scope: str, lsf_table: str, ap_tables: list[str]
) -> dict[str, Any]:
    """the ``store`` layout, in a process of its own (no memory left over from the other layouts).

    :return: the figures
    :rtype: dict[str, Any]
    """
    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as one:
        return one.submit(_store_in_a_fresh_process, spec, workers, root, scope, lsf_table, ap_tables).result()


def main() -> None:
    """parse the arguments, measure each worker count, print and write the figures.

    :return: nothing
    :rtype: None
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--workers", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--root", type=Path, default=Path("/dev/shm/measure-shared-arrow"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--scope", default="join_state_code", help="the scope column the tables are partitioned by")
    parser.add_argument("--lsf-table", default="lsf_projections", help="the table an LSF run changes")
    parser.add_argument(
        "--ap-tables",
        nargs="+",
        default=["latest_results_candidates_by_id", "latest_results_geos", "result_snapshots_by_id"],
        help="the tables an AP load changes",
    )
    arguments = parser.parse_args()
    spec = json.loads(arguments.spec.read_text())
    figures = [asyncio.run(measure(spec, workers, arguments.root, arguments.scope)) for workers in arguments.workers]
    for each in figures:
        each["store_layout"] = measure_store(
            spec, each["workers"], arguments.root, arguments.scope, arguments.lsf_table, arguments.ap_tables
        )
    for each in figures:
        print(f"\n== {each['workers']} workers; {sum(each['rows'].values()):,} rows; store {each['store_mib']} MiB")
        print(f"   seconds: {each['seconds']}")
        print(f"   today  : serving PSS {each['today']['total_pss_mib']} MiB")
        serving = each["shared"]["serving"]
        print(
            f"   shared : serving PSS {serving['Pss']} MiB (RSS {serving['Rss']}, anon {serving['RssAnon']}, shmem {serving['RssShmem']})"
        )
        for worker in each["shared"]["workers"]:
            print(
                f"            worker  PSS {worker['Pss']} MiB (RSS {worker['Rss']}, anon {worker['RssAnon']}, shmem {worker['RssShmem']})"
            )
        print(f"            total PSS {each['shared']['total_pss_mib']} MiB")
        print(
            f"   naive  : worker PSS {each['naive']['worker']['Pss']} MiB each; total PSS {each['naive']['total_pss_mib']} MiB"
        )
        layout = each["store_layout"]
        serving = layout["store"]["serving"]
        print(
            f"   store  : serving PSS {serving['Pss']} MiB (anon {serving['RssAnon']}, shmem {serving['RssShmem']}); "
            f"compressed copy {layout['copy_compressed_mib']} MiB; store {layout['store_mib']} MiB"
        )
        for worker in layout["store"]["workers"]:
            print(f"            worker  PSS {worker['Pss']} MiB (anon {worker['RssAnon']})")
        print(f"            total PSS {layout['store']['total_pss_mib']} MiB; seconds {layout['seconds']}")
        for label, change in layout["change"].items():
            print(f"   change {label}: {change}")
    if arguments.out is not None:
        arguments.out.write_text(json.dumps(figures, indent=1))
    remove_tree(arguments.root, log=log, what="the measurement's scratch store")


if __name__ == "__main__":
    main()
