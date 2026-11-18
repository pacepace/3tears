"""A claimed result costs what mapping its files costs, never a flush of the disk per file.

The contracts. A claim maps no file through Python's ``mmap`` (counted: its ``mmap.__new__`` audit event), on every
host. Timed, :meth:`SharedArrowStore.claim` of a worker's result of many files takes within twice what mapping the
same number of files of the same size takes bare (``pa.memory_map`` and an IPC read of each), turn about with it
(:func:`best_seconds_interleaved`), at a reports full start's file size, while another process writes beside it as a
pod's workers write the next result while the serving process claims one.

Why: a claim maps every file pinned (:func:`threetears.core.shared_arrow._map_pinned`), and it mapped them through
CPython's ``mmap.mmap``, which on macOS calls ``fcntl(F_FULLFSYNC)`` on every file it maps (bpo-11277): a flush of
the drive's cache per file. A reports full start's apply claimed one result of 2240 files (1.98 GB) in 8.45-11.2 s,
7.9 s of it the flushes, where Arrow maps the same files in 0.49-0.51 s (2026-11-17, the follower 2.1x the bare start
for it). A flush costs that only while something else writes: on a quiet disk ``mmap.mmap`` maps 200 files of 0.9 MB in
8-19 ms, with one process writing 8 MB files beside it in 1.0-1.3 s, where ``pa.memory_map`` takes 14-45 ms either
way (markpace-mbp, 2026-11-17). So the test writes beside its claim: there it read 81.6x and 148.2x mapping bare on
the old ``mmap.mmap`` path (1166 and 2370 ms against 14.3 and 16.0 ms) and 1.11-1.21x on Arrow's (12.9-13.8 ms against
10.9-12.1 ms), the budget twice. Linux (``/dev/shm`` in prod) flushes nothing on a
map, so there the bound holds either way.
"""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from threetears.core.shared_arrow import MappedSegment, SegmentSlot, SharedArrowStore
from threetears.core.testing.calibrate import best_seconds_interleaved, budget_problem
from threetears.core.testing.loop_lateness import MEASURE_ATTEMPTS

#: the files of one result: enough that a cost paid per file is most of a claim
FILES = 200

#: one file's rows of int64: about 0.9 MB of Arrow, a reports full start's files' mean (1.98 GB over 2240)
FILE_ROWS = 115_000

#: runs of each side a measurement takes (:func:`best_seconds_interleaved`)
RUNS = 3

#: the results written before the timing starts: one a claim, for every run of every measurement it may take
RESULTS = RUNS * MEASURE_ATTEMPTS


#: what the writer beside the timed claims runs: 8 MB files rewritten in turn over 16 names, never synced, until killed
_WRITER = textwrap.dedent(
    """
    import os, sys
    block = os.urandom(1 << 20)
    turn = 0
    while True:
        with open(os.path.join(sys.argv[1], f"w{turn % 16}"), "wb") as out:
            for _ in range(8):
                out.write(block)
        turn += 1
    """
)

#: the Python ``mmap`` maps made while a count is open (:func:`_counting_maps`), and whether the hook is in
_MAPS: list[list[str]] = []
_HOOKED: list[bool] = []


def _audited(event: str, args: tuple[Any, ...]) -> None:
    """note a Python ``mmap`` made while a count is open.

    :return: nothing
    :rtype: None
    """
    if _MAPS and event == "mmap.__new__":
        _MAPS[-1].append(repr(args))


def _file(index: int) -> pa.Table:
    """one file's rows.

    :param index: which file
    :ptype index: int
    :return: the table
    :rtype: pa.Table
    """
    return pa.table({"a": pa.array(range(index, index + FILE_ROWS), pa.int64())})


def _bare_files(directory: Path) -> list[str]:
    """the same number of files, of the same size, written bare: what the baseline maps.

    :param directory: where
    :ptype directory: Path
    :return: their paths
    :rtype: list[str]
    """
    directory.mkdir()
    paths = []
    for index in range(FILES):
        path = str(directory / f"f{index}.arrow")
        table = _file(index)
        with ipc.new_file(path, table.schema) as writer:
            writer.write_table(table)
        paths.append(path)
    return paths


def _result(store: SharedArrowStore) -> SegmentSlot:
    """a result of ``FILES`` files, written as a worker writes one.

    :param store: the store
    :ptype store: SharedArrowStore
    :return: its slot, to claim
    :rtype: SegmentSlot
    """
    tables = {f"f{index}": _file(index) for index in range(FILES)}
    slot = store.reserve(sum(table.nbytes for table in tables.values()) * 2)
    slot.write(tables)
    return slot


def test_a_claim_maps_no_file_through_python_mmap(tmp_path: Path) -> None:
    """Flips if a claim maps a file through ``mmap.mmap``, which flushes the drive's cache per file on macOS."""
    if not _HOOKED:
        # an audit hook stays for the process; it notes only while a count is open
        sys.addaudithook(_audited)
        _HOOKED.append(True)
    store = SharedArrowStore(tmp_path / "shm", max_bytes=64 << 20)
    slot = store.reserve(16 << 20)
    slot.write({f"f{index}": pa.table({"a": pa.array(range(index, index + 1000), pa.int64())}) for index in range(5)})
    maps: list[str] = []
    _MAPS.append(maps)
    try:
        segment = store.claim(slot)
    finally:
        _MAPS.pop()
    assert segment is not None
    try:
        assert sum(table.num_rows for table in segment.tables.values()) == 5000
    finally:
        segment.close()
    assert maps == [], f"a claim of 5 files made {len(maps)} Python mmap maps: {maps[:2]}"


@contextlib.contextmanager
def _writing_beside(directory: Path) -> Iterator[None]:
    """another process writing files into ``directory`` while the block runs, as a pod's workers write beside a claim.

    :param directory: where it writes
    :ptype directory: Path
    :return: a context manager
    :rtype: Iterator[None]
    """
    directory.mkdir()
    writer = subprocess.Popen([sys.executable, "-c", _WRITER, str(directory)])
    try:
        yield
    finally:
        writer.kill()
        writer.wait()


# times against the host's quiet (best_seconds_interleaved): run apart, never beside other work (tests/timing_suite.py)
@pytest.mark.loop_timing
def test_a_claim_of_many_files_is_within_twice_mapping_them_bare(tmp_path: Path) -> None:
    """Flips if a claim pays a cost per file past mapping it (a drive flush per file on macOS before the fix)."""
    store = SharedArrowStore(tmp_path / "shm", max_bytes=RESULTS * FILES * FILE_ROWS * 8 * 3)
    results = [_result(store) for _ in range(RESULTS)]
    bare_paths = _bare_files(tmp_path / "bare")
    rows = FILES * FILE_ROWS

    async def bare() -> None:
        tables = [ipc.open_file(pa.memory_map(path, "r")).read_all() for path in bare_paths]
        assert sum(table.num_rows for table in tables) == rows

    # closed after the timing: a close removes the result's files, a cost of its own that mapping them bare never pays
    segments: list[MappedSegment] = []

    async def claimed() -> None:
        segment = store.claim(results.pop())
        assert segment is not None
        segments.append(segment)
        assert sum(table.num_rows for table in segment.tables.values()) == rows

    try:
        with _writing_beside(tmp_path / "beside"):
            baseline, took = asyncio.run(best_seconds_interleaved(bare, claimed, runs=RUNS))
    finally:
        for segment in segments:
            segment.close()
    print(f"a claim of {FILES} files of ~0.9 MB: {took * 1000:.1f} ms, mapping them bare {baseline * 1000:.1f} ms")
    problem = budget_problem(f"a claim of {FILES} files", took, baseline)
    assert problem is None, problem
