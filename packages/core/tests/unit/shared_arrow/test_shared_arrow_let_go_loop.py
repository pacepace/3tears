"""A reader lets go of what a publish superseded without holding the serving process's event loop.

The contracts. A publish tells the store's readers (:meth:`SharedArrowStore.publish`), on the store's settling thread.
A reader holding the version it replaced lets that version go: its DuckDB connection closed, its mapped tables freed,
its files unmapped and its lease released. An idle reader, holding no version, keeps the files it mapped that the
current version still names, to lend its next read, and lets go of those the current version no longer names. At the
reports pod's size (2240 files a version, each a table part of about 200 columns: candidates 195, geographies 201,
snapshots 197) neither holds the event loop more than 20 ms past what the host's own idle loop is late at the same
moments, while a publish replaces every part.

What a let-go costs the loop is the arrays it frees, a file's columns over every file, not the rows: each part here
holds few rows, so the test writes little, and frees as many arrays as the pod's version does.
"""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path

import pyarrow as pa
import pytest

from threetears.core.shared_arrow import Parts, SharedArrowReader, SharedArrowStore
from threetears.core.testing.loop_lateness import HostTicker, Tick, added_lateness, latest, ticks_while, trusted

#: the most the event loop may run late while a reader lets go, past what the host's own idle loop, ticking in a
#: process of its own, was late at the same moments (the loop watchdog's bound is 50 ms)
LOOP_MAY_BE_LATE = 0.020

#: what the test loop's ticker asks to sleep
TICK = 0.002

#: the reports pod's files a version (a full start's result, 1.98 GB over 2240 files, 2026-11-17) and its tables' width
PARTS = 2240
COLUMNS = 200

#: each part's rows: few (the arrays freed are as many as at the pod's rows)
ROWS = 8

#: the parts a change replaces, in the version an idle reader sees before the full one
CHANGED = 4


def _part(seed: int) -> pa.Table:
    """one part: ``COLUMNS`` integer columns of ``ROWS`` rows.

    :param seed: the first value
    :ptype seed: int
    :return: the part
    :rtype: pa.Table
    """
    return pa.table({f"c{index:03d}": pa.array(range(seed, seed + ROWS), pa.int64()) for index in range(COLUMNS)})


#: the parts' names, in order
_NAMES = [f"p{index:05d}" for index in range(PARTS)]


@pytest.fixture(scope="module")
def parts() -> tuple[pa.Table, pa.Table, pa.Table]:
    """three parts of one table at the pod's width, each of other values.

    :return: the parts
    :rtype: tuple[pa.Table, pa.Table, pa.Table]
    """
    return _part(0), _part(1), _part(2)


def _whole(part: pa.Table) -> Parts:
    """a version of every part, each ``part``.

    :param part: the part
    :ptype part: pa.Table
    :return: the version's table
    :rtype: Parts
    """
    return Parts(schema=part.schema, parts=dict.fromkeys(_NAMES, part))


async def _measured(
    tmp_path: Path, before: Callable[[SharedArrowStore, SharedArrowReader], None], replace: Parts
) -> tuple[list[Tick], list[Tick]]:
    """the test loop's ticks and the host's idle loop's while a publish replaces every part and the store's settling
    thread tells the reader, after ``before`` brought it to the state measured.

    :param before: what is done first, off the loop: publishes and reads
    :ptype before: Callable[[SharedArrowStore, SharedArrowReader], None]
    :param replace: the version published while measured, every part other than before
    :ptype replace: Parts
    :return: the loop's ticks and the host's
    :rtype: tuple[list[Tick], list[Tick]]
    """
    # a store of its own each measurement: a retaken one publishes its versions afresh
    store = SharedArrowStore(Path(tempfile.mkdtemp(dir=tmp_path)) / "shm", max_bytes=8 << 30)
    reader = store.reader(config={"threads": 2})
    try:
        await asyncio.to_thread(before, store, reader)

        def publish() -> None:
            store.publish("last", {"long": replace})
            # the reader told, and what it held let go
            store.settle()

        async def run() -> None:
            await asyncio.to_thread(publish)
            await asyncio.sleep(0.05)

        with HostTicker() as host:
            ticks = await ticks_while(run, every=TICK)
        assert reader.status()["version"] is None, "the reader holds no version until its next read"
    finally:
        reader.close()
    return ticks, host.ticks


def _read(reader: SharedArrowReader) -> None:
    """one read of the current version, which maps it.

    :param reader: the reader
    :ptype reader: SharedArrowReader
    """
    with reader.read() as held:
        assert held.cursor.execute("SELECT count(*) FROM long").fetchone() == (PARTS * ROWS,)


async def _judged(measure: Callable[[], Awaitable[tuple[list[Tick], list[Tick]]]]) -> tuple[float, float]:
    """a trusted measurement's lateness past the host and in all.

    :param measure: takes one measurement
    :ptype measure: Callable[[], Awaitable[tuple[list[Tick], list[Tick]]]]
    :return: (past the host, in all), in seconds
    :rtype: tuple[float, float]
    """
    ticks, host = await trusted(measure, lambda measured: measured, host_may_be_late=LOOP_MAY_BE_LATE)
    return added_lateness(ticks, host, every=TICK), latest(ticks, every=TICK)


@pytest.mark.loop_timing
async def test_a_reader_lets_a_pod_sized_version_go_without_holding_the_loop(
    tmp_path: Path, parts: tuple[pa.Table, pa.Table, pa.Table]
) -> None:
    """a reader that read version 1 lets it go when a publish replaces every part."""
    first, second, _ = parts

    def before(store: SharedArrowStore, reader: SharedArrowReader) -> None:
        store.publish("v1", {"long": _whole(first)})
        _read(reader)

    added, late = await _judged(lambda: _measured(tmp_path, before, _whole(second)))
    print(f"let go of a held version: late {late * 1000:.1f} ms, past the host {added * 1000:.1f} ms")
    assert added < LOOP_MAY_BE_LATE, (
        f"the loop ran {added * 1000:.1f} ms late past the host's own idle loop ({late * 1000:.1f} ms in all) while a "
        f"reader let a version of {PARTS} files of {COLUMNS} columns go"
    )


@pytest.mark.loop_timing
async def test_an_idle_reader_lets_its_lent_files_go_without_holding_the_loop(
    tmp_path: Path, parts: tuple[pa.Table, pa.Table, pa.Table]
) -> None:
    """a reader that read version 1, then saw version 2 change a few parts while idle, keeps version 1's other files
    to lend its next read; a publish replacing every part leaves it none to lend, and it lets them all go."""
    first, second, third = parts

    def before(store: SharedArrowStore, reader: SharedArrowReader) -> None:
        store.publish("v1", {"long": _whole(first)})
        _read(reader)
        changed = Parts(schema=second.schema, parts=dict.fromkeys(_NAMES[:CHANGED], second))
        store.publish("v2", {"long": changed}, base="v1")
        # v1 let go; the files v2 still names kept to lend
        store.settle()

    added, late = await _judged(lambda: _measured(tmp_path, before, _whole(third)))
    print(f"let go of lent files: late {late * 1000:.1f} ms, past the host {added * 1000:.1f} ms")
    assert added < LOOP_MAY_BE_LATE, (
        f"the loop ran {added * 1000:.1f} ms late past the host's own idle loop ({late * 1000:.1f} ms in all) while an "
        f"idle reader let {PARTS - CHANGED} lent files of {COLUMNS} columns go"
    )
