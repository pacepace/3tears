"""A reader lets a superseded version go without holding the serving process's event loop.

The contract. A publish tells the store's readers (:meth:`SharedArrowStore.publish`), and a reader holding the version
it replaced lets that version go on the store's settling thread: its DuckDB connection closed, its mapped tables freed,
its files unmapped and its lease released. At the reports pod's size (2240 files a version, each a table part of about
200 columns: candidates 195, geographies 201, snapshots 197) that thread never holds the event loop more than 20 ms
past what the host's own idle loop is late at the same moments, while a publish replaces every part.

Why: cobalt dev's two reports pods both stalled their loops 134 ms and 160 ms at 15:49Z on 2026-11-19, the loop idle
while the settling thread held the interpreter lock in ``MappedTables.close``, reached from ``publish ->
_tell_readers -> refresh``. Its tables were dropped all at once (``self.tables``, ``self._by_inode``): freeing a
dictionary of 2240 tables of 200 columns is one cascade of deallocations in C, with no turn for any other thread until
its last array is freed (2240 tables of 24 columns: 36 ms on an M-series laptop, the loop as late; 200 columns, over
200 ms). The files' unmapping (``pa.memory_map``'s close) and the lease's unlinks let the lock go on their own.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pyarrow as pa
import pytest

from threetears.core.shared_arrow import Parts, SharedArrowStore
from threetears.core.testing.loop_lateness import HostTicker, Tick, added_lateness, latest, ticks_while, trusted

#: the most the event loop may run late while a version is let go, past what the host's own idle loop, ticking in a
#: process of its own, was late at the same moments (the loop watchdog's bound is 50 ms)
LOOP_MAY_BE_LATE = 0.020

#: what the test loop's ticker asks to sleep
TICK = 0.002

#: the reports pod's files a version (a full start's result, 1.98 GB over 2240 files, 2026-11-17) and its tables'
#: width: what freeing a version costs is its arrays, a file's columns over every file, not its rows
PARTS = 2240
COLUMNS = 200

#: each part's rows: few, so the test writes little; the arrays freed are as many as at the pod's rows
ROWS = 8


def _part(seed: int) -> pa.Table:
    """one part: ``COLUMNS`` integer columns of ``ROWS`` rows.

    :param seed: the first value
    :ptype seed: int
    :return: the part
    :rtype: pa.Table
    """
    return pa.table({f"c{index:03d}": pa.array(range(seed, seed + ROWS), pa.int64()) for index in range(COLUMNS)})


@pytest.fixture(scope="module")
def versions() -> tuple[Parts, Parts]:
    """two versions of one table at the pod's size, every part of the second other than the first's.

    :return: the first version's parts and the second's
    :rtype: tuple[Parts, Parts]
    """
    first, second = _part(0), _part(1)
    names = [f"p{index:05d}" for index in range(PARTS)]
    return (
        Parts(schema=first.schema, parts=dict.fromkeys(names, first)),
        Parts(schema=second.schema, parts=dict.fromkeys(names, second)),
    )


async def _let_go(tmp_path: Path, versions: tuple[Parts, Parts]) -> tuple[list[Tick], list[Tick]]:
    """the test loop's ticks and the host's idle loop's while a publish replaces every part of the version a reader
    holds, and the reader lets that version go on the store's settling thread.

    :return: the loop's ticks and the host's
    :rtype: tuple[list[Tick], list[Tick]]
    """
    # a store of its own each measurement: a retaken one publishes its versions afresh
    store = SharedArrowStore(Path(tempfile.mkdtemp(dir=tmp_path)) / "shm", max_bytes=8 << 30)
    first, second = versions
    await asyncio.to_thread(store.publish, "v1", {"long": first})
    reader = store.reader(config={"threads": 2})

    def read() -> None:
        with reader.read() as held:
            assert held.cursor.execute("SELECT count(*) FROM long").fetchone() == (PARTS * ROWS,)

    await asyncio.to_thread(read)

    def publish() -> None:
        store.publish("v2", {"long": second})
        # the reader told and v1 let go: its tables freed, its files unmapped, its lease released
        store.settle()

    async def run() -> None:
        await asyncio.to_thread(publish)
        await asyncio.sleep(0.05)

    try:
        with HostTicker() as host:
            ticks = await ticks_while(run, every=TICK)
        assert reader.status()["version"] is None, "the reader let v1 go"
    finally:
        reader.close()
    return ticks, host.ticks


@pytest.mark.loop_timing
async def test_a_reader_lets_a_pod_sized_version_go_without_holding_the_loop(
    tmp_path: Path, versions: tuple[Parts, Parts]
) -> None:
    ticks, host = await trusted(
        lambda: _let_go(tmp_path, versions), lambda measured: measured, host_may_be_late=LOOP_MAY_BE_LATE
    )
    added = added_lateness(ticks, host, every=TICK)
    late = latest(ticks, every=TICK)
    print(f"let go: late {late * 1000:.1f} ms, past the host {added * 1000:.1f} ms")
    assert added < LOOP_MAY_BE_LATE, (
        f"the loop ran {added * 1000:.1f} ms late past the host's own idle loop ({late * 1000:.1f} ms in all) while a "
        f"reader let a version of {PARTS} files of {COLUMNS} columns go"
    )
