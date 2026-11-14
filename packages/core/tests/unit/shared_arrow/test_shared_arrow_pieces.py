"""A result slot that grows piece by piece, every piece checked against the store's room, and published once.

The contracts (reports #53, ``SPEC-from-reports53-slot.md``): (a) a store with less room than a piece, but room for
the change, applies the change; (b) a result several pieces big is written in pieces and published as one version, a
lease on the old version still seeing the old rows, no piece left open once it is claimed; (c) a store with no room
for the next piece fails the whole result loudly -- refused counted, every piece discarded, nothing published -- and
a later result that fits applies. A part bigger than one piece is the only hard failure, named with its bytes.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import pyarrow as pa
import pytest

from packages.core.tests.unit.shared_arrow import jobs
from threetears.core.shared_arrow import (
    MappedSegment,
    Parts,
    SharedArrowFullError,
    SharedArrowPool,
    SharedArrowStore,
    read_version,
)

TABLE = "t"

#: rows of one part: about 0.75 MB of Arrow
PART_ROWS = 50_000


def _part(value: int, rows: int = PART_ROWS) -> pa.Table:
    """one part's rows: ``a`` is ``value`` in every row, a short text beside it.

    :return: the part
    :rtype: pa.Table
    """
    return jobs.part_rows(value, rows)


async def _until(condition: Callable[[], bool], what: str, seconds: float = 30.0) -> None:
    """wait until ``condition()`` holds.

    :return: nothing
    :rtype: None
    """
    deadline = time.monotonic() + seconds
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError(f"never: {what}")
        await asyncio.sleep(0.01)


def _published_parts(segment: MappedSegment) -> Parts:
    """the claimed result's parts of :data:`TABLE`, linked as they are (no row copied).

    :return: the parts
    :rtype: Parts
    """
    prefix = f"{TABLE}@"
    parts = {
        name[len(prefix) :]: path for name, path in segment.files.items() if name.startswith(prefix) and name != prefix
    }
    return Parts(schema=segment.tables[TABLE].schema, parts=parts)


@pytest.fixture
async def pool(tmp_path: Path) -> AsyncIterator[tuple[SharedArrowStore, SharedArrowPool]]:
    """a 24 MiB store and one worker over it.

    :return: the store and the pool
    :rtype: tuple[SharedArrowStore, SharedArrowPool]
    """
    store = SharedArrowStore(tmp_path / "shm", max_bytes=24 << 20)
    made = SharedArrowPool(store, workers=1, private_bytes=1 << 40, name="pieces")
    yield store, made
    await made.close()


class TestASmallStore:
    def test_a_small_change_applies_where_a_whole_piece_has_no_room(self, tmp_path: Path) -> None:
        store = SharedArrowStore(tmp_path / "shm", max_bytes=12 << 20)
        store.publish("v1", {TABLE: Parts(schema=_part(0).schema, parts={"p0": _part(0, rows=400_000)})})
        # a whole piece (64 MiB) has no room in a 12 MiB store holding ~5.6 MB: one fixed slot would be refused
        with pytest.raises(SharedArrowFullError):
            store.reserve(64 << 20)
        refused = store.refused
        slot = store.reserve_pieces(64 << 20, minimum=1 << 20)
        with slot.writing() as write:
            write({TABLE: Parts(schema=_part(1).schema, parts={"p1": _part(1, rows=10)})})
        segment = store.claim(slot)
        assert segment is not None
        try:
            store.publish("v2", {TABLE: _published_parts(segment)}, base="v1")
        finally:
            segment.close()
        assert store.refused == refused
        with store.lease("v2") as version:
            assert set(version.tables) == {TABLE}
        assert store.status()["slots_open"] == 0
        assert store.status()["pieces_open"] == 0

    def test_a_store_without_the_minimum_refuses_the_first_piece(self, tmp_path: Path) -> None:
        store = SharedArrowStore(tmp_path / "shm", max_bytes=4 << 20)
        with pytest.raises(SharedArrowFullError):
            store.reserve_pieces(64 << 20, minimum=8 << 20)
        assert store.refused == 1
        assert store.status()["slots_open"] == 0


class TestABigResult:
    async def test_a_result_three_pieces_big_is_published_as_one_version(
        self, pool: tuple[SharedArrowStore, SharedArrowPool]
    ) -> None:
        store, workers = pool
        store.publish("v1", {TABLE: Parts(schema=_part(0).schema, parts={"p0": _part(0, rows=10)})})
        old = store.lease("v1")
        try:
            # 8 parts of ~0.75 MB in 2 MiB pieces: 3 pieces at least, in a worker process
            done = await workers.run(jobs.write_parts, TABLE, 8, PART_ROWS, read=False, result_pieces=2 << 20)
            assert done.value == 8
            segment = done.segment
            assert segment is not None
            assert store.status()["slots_open"] == 0
            assert store.status()["pieces_open"] == 0
            assert store.status()["pieces_used"] >= 3
            try:
                store.publish("v2", {TABLE: _published_parts(segment)}, base="v1")
            finally:
                segment.close()
            with read_version(store.root, "v2") as new:
                assert new.tables[TABLE].num_rows == 10 + 8 * PART_ROWS
            with read_version(store.root, "v1") as kept:
                assert kept.tables[TABLE].num_rows == 10
        finally:
            old.release()

    def test_steps_of_whole_tables_fill_pieces_in_turn(self, tmp_path: Path) -> None:
        store = SharedArrowStore(tmp_path / "shm", max_bytes=24 << 20)
        slot = store.reserve_pieces(2 << 20, minimum=1 << 20)
        with slot.writing() as write:
            for value in range(5):
                write({f"w{value}": _part(value)})
        segment = store.claim(slot)
        assert segment is not None
        try:
            assert sorted(segment.tables) == [f"w{value}" for value in range(5)]
            assert all(segment.tables[f"w{value}"].num_rows == PART_ROWS for value in range(5))
        finally:
            segment.close()
        assert store.status()["pieces_used"] >= 3
        assert store.segments() == ()

    def test_a_part_bigger_than_one_piece_fails_naming_it(self, tmp_path: Path) -> None:
        store = SharedArrowStore(tmp_path / "shm", max_bytes=24 << 20)
        slot = store.reserve_pieces(1 << 20, minimum=1 << 20)
        with pytest.raises(SharedArrowFullError, match=r"part 'huge'.* bytes"):
            with slot.writing() as write:
                write(
                    {
                        TABLE: Parts(
                            schema=_part(0).schema,
                            parts={"small": _part(0, rows=10), "huge": _part(1, rows=4 * PART_ROWS)},
                        )
                    }
                )
        store.discard(slot)
        assert store.segments() == ()
        assert store.status()["slots_open"] == 0


class TestAFullStore:
    async def test_no_room_for_the_next_piece_fails_the_whole_result_and_a_later_one_applies(
        self, pool: tuple[SharedArrowStore, SharedArrowPool]
    ) -> None:
        store, workers = pool
        store.publish("v1", {TABLE: Parts(schema=_part(0).schema, parts={"p0": _part(0, rows=400_000)})})
        refused = store.refused
        # ~5.6 MB held of 24 MiB, and a result of ~30 MB in pieces of 6 MiB, each at least 4 MiB: a piece finds no room
        with pytest.raises(SharedArrowFullError):
            await workers.run(
                jobs.write_parts, TABLE, 40, PART_ROWS, read=False, result_pieces=6 << 20, result_minimum=4 << 20
            )
        assert store.refused == refused + 1
        await _until(lambda: store.status()["slots_open"] == 0, "every piece discarded")
        await _until(lambda: store.segments() == (), "every piece's files gone")
        assert store.current is not None and store.current.name == "v1"
        assert store.versions() == ("v1",)
        done = await workers.run(
            jobs.write_parts, TABLE, 2, PART_ROWS, read=False, result_pieces=6 << 20, result_minimum=4 << 20
        )
        assert done.segment is not None
        try:
            store.publish("v2", {TABLE: _published_parts(done.segment)}, base="v1")
        finally:
            done.segment.close()
        assert store.current is not None and store.current.name == "v2"
