"""A version of the shared copy holds few, large files (``design-copy-worker-backend.md``, decision 2b;
``research-shared-arrow-free.md``): a split table's small scopes share packs (``copy_parts.pack_name``) filled to
``PACK_TARGET_BYTES``, so letting a version go frees few arrays.

The contracts pinned here:

- many small scopes land in a few packs, and a version's arrays (files times columns) stay few;
- after many small changes, each bringing a scope of its own, a version's file count stays under its bound;
- a change rewrites only the packs and parts it lands in, every other file the version before's own;
- a pack outgrown is cut again as it is rewritten, its big scope moving to a part of its own;
- a version over its file bound makes a merge due, and the merge packs more until it is under;
- a per-scope reload of a packed scope replaces its rows alone, the pack's other scopes kept;
- a reload staging in its worker reserves room for its own rows, never the store's, so an apply beside it lands;
- an apply reserves room for the files its change rewrites, never the store's, so a reload stages beside it, and a
  part bigger than the change foretells grows its piece and still applies.
"""

from __future__ import annotations

import asyncio
import math
import os
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from packages.core.tests.unit.collections import copy_hooks
from threetears.core.collections.copy_lane import ApplyLane
from threetears.core.collections.copy_parts import (
    PACK_BELOW_BYTES,
    PACK_TARGET_BYTES,
    PartRule,
    is_pack,
)
from threetears.core.collections.copy_reload import ReloadLane
from threetears.core.collections.copy_pieces import apply_piece_bytes, reload_piece_bytes
from threetears.core.collections.shared_copy import (
    ROWS,
    ChangePage,
    ReloadScope,
    SharedCopyBackend,
    held_parts,
)
from threetears.core.shared_arrow import SharedArrowStore, read_version, version_files

#: the most files a version holds by default (``SharedCopyBackend``'s ``max_files``)
_MAX_VERSION_FILES = 512

#: the prefix a split table's packs index is published beside it under
_PACKED = "__packed__"

SCHEMA = pa.schema(
    [
        pa.field("state", pa.string()),
        pa.field("office_key", pa.string()),
        pa.field("candidate_id", pa.int64()),
        pa.field("votes", pa.int64()),
    ]
)
_KEY = ("state", "office_key", "candidate_id")
BY_STATE = PartRule("results", _KEY, scope_column="state")


def _state(n: int) -> str:
    return f"S{n:03d}"


def _rows(state: str, count: int, votes: int = 1, *, start: int = 0) -> pa.Table:
    """``count`` rows of one state, keyed by candidate, each ~40 bytes."""
    return pa.table(
        {
            "state": pa.array([state] * count, pa.string()),
            "office_key": pa.array(["G"] * count, pa.string()),
            "candidate_id": pa.array(np.arange(start, start + count), pa.int64()),
            "votes": pa.array(np.full(count, votes), pa.int64()),
        },
        schema=SCHEMA,
    )


def _rows_of(states: Sequence[str], count: int) -> pa.Table:
    return pa.concat_tables([_rows(state, count) for state in states])


def _held(store: SharedArrowStore) -> pa.Table:
    current = store.current
    assert current is not None
    with read_version(store.root, current.name) as mapped:
        rows = mapped.tables["results"]
        return pa.table({name: rows[name].combine_chunks() for name in SCHEMA.names}).sort_by(
            [(column, "ascending") for column in _KEY]
        )


def _sorted(rows: pa.Table) -> pa.Table:
    return rows.sort_by([(column, "ascending") for column in _KEY]).combine_chunks()


def _parts(store: SharedArrowStore) -> dict[str, int]:
    """each part of ``results`` the current version holds, and its file's inode."""
    current = store.current
    assert current is not None
    return {part: os.stat(path).st_ino for part, path in held_parts(store.root, current.name, "results").items()}


def _files(store: SharedArrowStore) -> int:
    current = store.current
    assert current is not None
    return len(version_files(store.root, current.name))


def _arrays(store: SharedArrowStore) -> int:
    """the arrays a reader of the current version maps (and frees as it lets it go): each table's columns times its
    chunks, a chunk a file."""
    current = store.current
    assert current is not None
    with read_version(store.root, current.name) as mapped:
        return sum(table.num_columns * max(table.column(0).num_chunks, 1) for table in mapped.tables.values())


@pytest.fixture
def store(tmp_path: Path) -> SharedArrowStore:
    return SharedArrowStore(tmp_path / "shm", max_bytes=1 << 30)


@pytest.fixture
async def backend(store: SharedArrowStore) -> AsyncIterator[SharedCopyBackend]:
    made = SharedCopyBackend(store, {"results": SCHEMA}, {"results": BY_STATE})
    yield made
    await made.close()


class TestFewFiles:
    async def test_many_small_scopes_land_in_a_few_packs(
        self, store: SharedArrowStore, backend: SharedCopyBackend
    ) -> None:
        """Flips if a split table holds a part a scope: 200 small states (one change, ~2,600 rows each, ~16 MiB in
        all) were 200 files and 800 arrays a version; packed they are one file a 2 MiB."""
        rows = _rows_of([_state(n) for n in range(200)], 2_600)
        await backend.apply([ChangePage(ROWS, "results", rows)])
        parts = _parts(store)
        assert all(is_pack(part) for part in parts), sorted(parts)
        packs = math.ceil(rows.nbytes / PACK_TARGET_BYTES)
        assert len(parts) <= packs + 1, (len(parts), packs)
        # the table's packs, its schema file and packs index, the version's mark at most
        assert _files(store) <= len(parts) + 3
        assert _arrays(store) <= len(SCHEMA) * (packs + 1) + 2 * 2
        assert _held(store).equals(_sorted(rows))

    async def test_small_changes_keep_the_file_count_under_the_bound(self, store: SharedArrowStore) -> None:
        """Flips if each new small scope stays a file of its own: 60 changes, each bringing a state, left 60 parts a
        version. Each change lands as a part, then the lane's merge packs it: the table holds two parts at most."""
        backend = SharedCopyBackend(store, {"results": SCHEMA}, {"results": BY_STATE})
        lane = ApplyLane(backend)
        try:
            most = 0
            for n in range(60):
                await lane.apply([ChangePage(ROWS, "results", _rows(_state(n), 50, votes=n))])
                await lane.drained()
                most = max(most, len(_parts(store)))
                assert _files(store) <= _MAX_VERSION_FILES
            assert most <= 2, most
            assert lane.status().merges >= 1
            held = _held(store)
            assert held.num_rows == 60 * 50
            assert sorted(set(held["state"].to_pylist())) == [_state(n) for n in range(60)]
        finally:
            await lane.close()
            await backend.close()


class TestAChangeRewritesOnlyItsParts:
    async def test_a_change_to_a_packed_scope_rewrites_its_pack_alone(
        self, store: SharedArrowStore, backend: SharedCopyBackend
    ) -> None:
        """Flips if a change rewrites more than the pack it lands in: a vote in one small state rewrites that state's
        pack, and every other pack and the big state's own part keep their files."""
        small = [_state(n) for n in range(50)]
        rows = pa.concat_tables([_rows_of(small, 2_600), _rows("BIG", 40_000)])
        await backend.apply([ChangePage(ROWS, "results", rows)], whole={"results"})
        before = _parts(store)
        packs = sorted(part for part in before if is_pack(part))
        assert len(packs) >= 2 and "BIG" in before, sorted(before)
        fix = _rows(_state(30), 1, votes=99)
        applied = await backend.apply([ChangePage(ROWS, "results", fix)])
        written = [part for part, wrote in applied.parts["results"].items() if wrote]
        assert len(written) == 1 and is_pack(written[0]), applied.parts
        assert set(applied.parts) == {"results"}
        after = _parts(store)
        assert set(after) == set(before)
        assert all(after[part] == before[part] for part in after if part != written[0])
        held = _held(store)
        assert held.filter(pc.equal(held["votes"], 99)).num_rows == 1
        # a change to the big state rewrites its own part alone
        applied = await backend.apply([ChangePage(ROWS, "results", _rows("BIG", 1, votes=7))])
        assert applied.parts == {"results": {"BIG": True}}

    async def test_an_outgrown_pack_is_cut_again(self, store: SharedArrowStore, backend: SharedCopyBackend) -> None:
        """Flips if a pack grows without bound: a packed state given 3 MiB more moves to a part of its own as its
        pack is rewritten, the pack's other states kept in a pack, and no row lost."""
        small = [_state(n) for n in range(20)]
        rows = _rows_of(small, 1_000)
        await backend.apply([ChangePage(ROWS, "results", rows)], whole={"results"})
        assert all(is_pack(part) for part in _parts(store))
        grown = _rows(_state(5), 120_000, start=1_000)
        assert grown.nbytes >= PACK_TARGET_BYTES + PACK_BELOW_BYTES
        await backend.apply([ChangePage(ROWS, "results", grown)])
        parts = _parts(store)
        assert _state(5) in parts, sorted(parts)
        assert any(is_pack(part) for part in parts)
        assert _held(store).equals(_sorted(pa.concat_tables([rows, grown])))
        current = store.current
        assert current is not None
        with read_version(store.root, current.name) as mapped:
            index = mapped.tables[f"{_PACKED}results"]
            assert _state(5) not in index.column(0).to_pylist()


class TestTheFileBound:
    async def test_a_version_over_its_bound_makes_a_merge_due_that_packs_more(self, store: SharedArrowStore) -> None:
        """Flips if a version may hold any number of files: 20 states of ~100 KiB each, over this backend's 64 KiB
        floor, are 20 parts; past its bound of 8 files the change says a merge is due, and the merge packs every scope
        under the 1 MiB target until the version is under."""
        backend = SharedCopyBackend(
            store, {"results": SCHEMA}, {"results": BY_STATE}, pack_below=64 << 10, pack_target=1 << 20, max_files=8
        )
        try:
            rows = _rows_of([_state(n) for n in range(20)], 2_600)
            applied = await backend.apply([ChangePage(ROWS, "results", rows)], whole={"results"})
            assert len(_parts(store)) == 20
            assert applied.files > 8 and applied.merge_due
            while await backend.merge() is not None:
                pass
            assert _files(store) <= 8, sorted(_parts(store))
            assert _held(store).equals(_sorted(rows))
        finally:
            await backend.close()


class TestReloads:
    async def test_a_packed_scopes_reload_replaces_its_rows_alone(self, store: SharedArrowStore) -> None:
        """Flips if a per-scope reload of a packed state drops the pack's other states, or keeps the reloaded
        state's old rows: its rows are replaced, every other state's kept."""
        backend = SharedCopyBackend(store, {"results": SCHEMA}, {"results": BY_STATE}, preload=(copy_hooks.__name__,))
        lane = ApplyLane(backend)
        reloads = ReloadLane(backend, lane, preload=(copy_hooks.__name__,))
        try:
            states = [_state(n) for n in range(10)]
            await lane.apply([ChangePage(ROWS, "results", _rows_of(states, 100))], whole={"results"})
            assert all(is_pack(part) for part in _parts(store))
            reload = await reloads.reload(
                [ChangePage(ROWS, "results", _rows(_state(3), 5, votes=42, start=200))],
                scopes={"results": ReloadScope("state", frozenset({_state(3)}))},
            )
            await reload.done
            held = _held(store)
            three = held.filter(pc.equal(held["state"], _state(3)))
            assert three["candidate_id"].to_pylist() == list(range(200, 205))
            assert set(three["votes"].to_pylist()) == {42}
            assert held.num_rows == 9 * 100 + 5
        finally:
            await reloads.close()
            await lane.close()
            await backend.close()

    async def test_a_staging_reload_leaves_room_for_an_apply(self, tmp_path: Path) -> None:
        """Flips if a reload's result piece takes the store's room: it reserved 1 GiB, or all a smaller store had, for
        as long as it staged, and an apply beside it found less than its 16 MiB and was refused."""
        store = SharedArrowStore(tmp_path / "shm", max_bytes=512 << 20)
        gate = tmp_path / "reload-go"
        backend = SharedCopyBackend(store, {"results": SCHEMA}, {"results": BY_STATE}, preload=(copy_hooks.__name__,))
        lane = ApplyLane(backend)
        reloads = ReloadLane(backend, lane, check=copy_hooks.WaitFor(str(gate)), preload=(copy_hooks.__name__,))
        try:
            await lane.apply([ChangePage(ROWS, "results", _rows("VA", 10))])
            reload = await reloads.reload([ChangePage(ROWS, "results", _rows("NC", 50))])
            deadline = asyncio.get_running_loop().time() + 20
            while reloads.status().staging != 1:
                assert asyncio.get_running_loop().time() < deadline, "the reload never staged"
                await asyncio.sleep(0.01)
            # the reload holds its piece while it waits on the gate: an apply lands beside it
            await lane.apply([ChangePage(ROWS, "results", _rows("VA", 1, votes=5))])
            gate.touch()
            await reload.done
            # the reload replaced the table whole (NC's 50 rows), and the apply beside it landed over it
            held = _held(store)
            assert held.num_rows == 50 + 1
            assert held.filter(pc.equal(held["state"], "VA"))["votes"].to_pylist() == [5]
        finally:
            gate.touch()
            await reloads.close()
            await lane.close()
            await backend.close()

    def test_a_reloads_piece_is_sized_to_its_largest_table(self) -> None:
        small = [ChangePage(ROWS, "results", _rows("VA", 10))]
        assert reload_piece_bytes(8 << 30, small) == (16 << 20) + small[0].rows.nbytes + small[0].rows.nbytes // 8
        big = [ChangePage(ROWS, "results", _rows("VA", 4_000_000))]
        wanted = big[0].rows.nbytes + big[0].rows.nbytes // 8 + (16 << 20)
        assert reload_piece_bytes(8 << 30, big) == wanted
        # never past a quarter of the store, nor 1 GiB, nor under 16 MiB
        assert reload_piece_bytes(256 << 20, big) == 64 << 20
        assert reload_piece_bytes(64 << 30, [ChangePage(ROWS, "results", _rows("VA", 40_000_000))]) == 1 << 30
        assert reload_piece_bytes(16 << 20, small) == 16 << 20


async def _until_queued(reloads: ReloadLane, reload_done: asyncio.Future[Any], what: str) -> None:
    """wait until the reload is staged and its publish waits its turn, or fails (raised here)."""
    deadline = asyncio.get_running_loop().time() + 30
    while reloads.status().queued != 1:
        if reload_done.done():
            reload_done.result()
            raise AssertionError(f"{what}: the reload ended before its publish's turn")
        assert asyncio.get_running_loop().time() < deadline, f"{what}: the reload never staged"
        await asyncio.sleep(0.01)


async def _until_piece_held(store: SharedArrowStore) -> None:
    """wait until a job holds a result piece (the apply waiting in its worker)."""
    deadline = asyncio.get_running_loop().time() + 20
    while store.status()["pieces_open"] < 1:
        assert asyncio.get_running_loop().time() < deadline, "the apply never reserved its piece"
        await asyncio.sleep(0.01)


class TestApplyPieces:
    async def test_an_apply_leaves_room_for_a_staging_reload(self, tmp_path: Path) -> None:
        """Flips if a change's result piece takes the store's room: it reserved 1 GiB, or all a 512 MiB store had,
        for as long as it applied, and a reload staging beside it found no room for its hand-off or its piece."""
        store = SharedArrowStore(tmp_path / "shm", max_bytes=512 << 20)
        gate = tmp_path / "apply-go"
        gate.touch()
        backend = SharedCopyBackend(
            store,
            {"results": SCHEMA},
            {"results": BY_STATE},
            hooks=copy_hooks.WaitFor(str(gate)),
            preload=(copy_hooks.__name__,),
        )
        lane = ApplyLane(backend)
        reloads = ReloadLane(backend, lane, preload=(copy_hooks.__name__,))
        try:
            await lane.apply([ChangePage(ROWS, "results", _rows("VA", 10))])
            gate.unlink()
            applying = await lane.submit([ChangePage(ROWS, "results", _rows("VA", 1, votes=5))])
            await _until_piece_held(store)
            # the apply holds its piece in the worker: a reload stages beside it
            reload = await reloads.reload([ChangePage(ROWS, "results", _rows("NC", 50))])
            await _until_queued(reloads, reload.done, "beside an apply")
            gate.touch()
            await applying
            await reload.done
            held = _held(store)
            assert held.num_rows == 50
            assert set(held["state"].to_pylist()) == {"NC"}
            assert store.refused == 0
        finally:
            gate.touch()
            await reloads.close()
            await lane.close()
            await backend.close()

    async def test_a_part_bigger_than_its_change_foretells_still_applies(self, tmp_path: Path) -> None:
        """Flips if a change's piece is its bound on one file: a hook derives a 40 MB part no table of the change
        foretells, past the 16 MiB piece the change is reserved at, and it fails; or if the apply takes the store, and
        the reload staging beside it is refused. The piece grows for the part, in a 512 MiB store beside a reload."""
        store = SharedArrowStore(tmp_path / "shm", max_bytes=512 << 20)
        gate = tmp_path / "apply-go"
        gate.touch()
        bulk_rows = 2_500_000
        backend = SharedCopyBackend(
            store,
            {"results": SCHEMA, copy_hooks.BULK: copy_hooks.BULK_SCHEMA},
            {"results": BY_STATE, copy_hooks.BULK: PartRule(copy_hooks.BULK, ("id",))},
            hooks=copy_hooks.WaitThenBulk(str(gate), bulk_rows),
            preload=(copy_hooks.__name__,),
        )
        lane = ApplyLane(backend)
        reloads = ReloadLane(backend, lane, preload=(copy_hooks.__name__,))
        try:
            page = [ChangePage(ROWS, "results", _rows("VA", 10))]
            assert apply_piece_bytes(store.max_bytes, page) < bulk_rows * 16
            await lane.apply(page)
            assert store.status()["pieces_grown"] >= 1
            gate.unlink()
            applying = await lane.submit([ChangePage(ROWS, "results", _rows("VA", 1, votes=5))])
            await _until_piece_held(store)
            reload = await reloads.reload(
                [ChangePage(ROWS, "results", _rows("NC", 50))],
                scopes={"results": ReloadScope("state", frozenset({"NC"}))},
            )
            await _until_queued(reloads, reload.done, "beside a big apply")
            grown = store.status()["pieces_grown"]
            gate.touch()
            await applying
            await reload.done
            assert store.status()["pieces_grown"] > grown
            held = _held(store)
            assert sorted(set(held["state"].to_pylist())) == ["NC", "VA"]
            assert held.filter(pc.equal(held["state"], "VA"))["votes"].to_pylist()[0] == 5
            current = store.current
            assert current is not None
            with read_version(store.root, current.name) as mapped:
                assert mapped.tables[copy_hooks.BULK].num_rows == bulk_rows
            assert store.refused == 0
        finally:
            gate.touch()
            await reloads.close()
            await lane.close()
            await backend.close()

    def test_an_applys_piece_is_sized_to_the_files_it_rewrites(self) -> None:
        small = [ChangePage(ROWS, "results", _rows("VA", 10))]
        rows = small[0].rows.nbytes
        assert apply_piece_bytes(8 << 30, small) == (16 << 20) + rows + rows // 8
        # the largest file the version holds of the table, plus the change's rows of it
        held = 40 << 20
        assert apply_piece_bytes(8 << 30, small, {"results": held}) == (16 << 20) + (held + rows) + (held + rows) // 8
        # a table the change does not touch counts nothing
        assert apply_piece_bytes(8 << 30, small, {"other": 1 << 30}) == apply_piece_bytes(8 << 30, small)
        # never past a quarter of the store, nor 1 GiB, nor under 16 MiB
        assert apply_piece_bytes(512 << 20, small, {"results": 1 << 30}) == 128 << 20
        assert apply_piece_bytes(64 << 30, small, {"results": 4 << 30}) == 1 << 30
        assert apply_piece_bytes(16 << 20, small) == 16 << 20
        assert apply_piece_bytes(8 << 30, []) == 16 << 20
