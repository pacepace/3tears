#!/usr/bin/env python
"""Measure what shared-memory Arrow saves: one copy of the data for a serving process and N workers.

``docs/shared-arrow.md`` claims the workers of a :class:`~threetears.core.shared_arrow.SharedArrowPool` share ONE
physical copy of a version. This measures it at a real size, on Linux, where ``/proc/<pid>/smaps_rollup`` charges
each process its proportional share (PSS) of every page it maps: the sum of the processes' PSS is the physical memory
they hold together, shared pages counted once.

Three layouts, the same data:

``today``
    one process holding the tables in an in-memory DuckDB (a scoped snapshot's L1): its PSS.
``shared``
    that process, plus the tables published once to the memory-backed store, plus N workers each scanning every
    column of every table through DuckDB over the mapped Arrow, all mapped at once: each process's PSS, and the sum.
``naive``
    each worker loading its own DuckDB copy instead: one worker measured, the total as the serving process plus N of
    it.

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
import os
import shutil
import time
from pathlib import Path
from typing import Any

import duckdb

from threetears.core.shared_arrow import JobContext, SharedArrowPool, SharedArrowStore

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


def _filled(column: str, kind: str) -> str:
    """the SQL that fills one column of a generated row ``i`` by its type.

    :param column: the column's name
    :ptype column: str
    :param kind: its DuckDB type
    :ptype kind: str
    :return: the expression, aliased to the column
    :rtype: str
    """
    quoted = '"' + column.replace('"', '""') + '"'
    if kind == "VARCHAR":
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


def load(connection: Any, spec: dict[str, Any]) -> None:
    """create and fill every table of the spec in ``connection``.

    :param connection: a DuckDB connection
    :ptype connection: duckdb.DuckDBPyConnection
    :param spec: table -> rows and columns
    :ptype spec: dict[str, Any]
    :return: nothing
    :rtype: None
    """
    for table, shape in spec.items():
        columns = ", ".join(_filled(name, kind) for name, kind in shape["columns"])
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


def own_copy(context: JobContext, spec: dict[str, Any]) -> dict[str, Any]:
    """load a DuckDB copy of its own (the naive worker) and measure this worker.

    :return: the worker's memory
    :rtype: dict[str, Any]
    """
    connection = duckdb.connect(":memory:", config={"threads": 2})
    load(connection, spec)
    memory = _memory()
    connection.close()
    return {"pid": os.getpid(), "memory": memory}


async def measure(spec: dict[str, Any], workers: int, root: Path) -> dict[str, Any]:
    """the three layouts with ``workers`` workers.

    :return: the figures
    :rtype: dict[str, Any]
    """
    shutil.rmtree(root, ignore_errors=True)
    l1 = duckdb.connect(":memory:", config={"threads": 2})
    started = time.monotonic()
    load(l1, spec)
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
        naive = (await pool.run(own_copy, spec, read=False)).value
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
    arguments = parser.parse_args()
    spec = json.loads(arguments.spec.read_text())
    figures = [asyncio.run(measure(spec, workers, arguments.root)) for workers in arguments.workers]
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
    if arguments.out is not None:
        arguments.out.write_text(json.dumps(figures, indent=1))
    shutil.rmtree(arguments.root, ignore_errors=True)


if __name__ == "__main__":
    main()
