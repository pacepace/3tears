"""Columns a product computes over a change's rows in the apply worker before the change is written
(:class:`~threetears.core.collections.shared_copy.ColumnHooks`): reports' snapshots' ``most_recent_ind``, each row's
upload ranked newest first among its group's, over the rows the change brings and those the version holds. A row read
from L3 lacks the column; the held rows whose rank the change moves are written with it; a whole change ranks its own
rows alone. Each fails on the copy before the hook (reports R-1 moving ``ComputedColumns`` onto the shared copy)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from packages.core.tests.unit.collections import copy_computed
from packages.core.tests.unit.collections.copy_computed import (
    COLUMN,
    SNAPSHOTS,
    SNAPSHOTS_KEY,
    SNAPSHOTS_SCHEMA,
    NewestUpload,
    Reordered,
)
from threetears.core.collections.copy_parts import PartRule
from threetears.core.collections.shared_copy import KEYS, ROWS, ChangePage, ColumnHooks, SharedCopyBackend, held_parts
from threetears.core.shared_arrow import SharedArrowStore, read_version

BY_STATE = PartRule(SNAPSHOTS, SNAPSHOTS_KEY, scope_column="state")
#: L3's rows: every column but the one the copy computes
_READ = pa.schema([field for field in SNAPSHOTS_SCHEMA if field.name != COLUMN])


def _uploads(state: str, upload: int, candidates: int = 3) -> pa.Table:
    """one upload's rows of a state, as L3 holds them (no ``most_recent_ind``)."""
    return pa.table(
        {
            "state": pa.array([state] * candidates, pa.string()),
            "office_key": pa.array(["G"] * candidates, pa.string()),
            "candidate_id": pa.array(np.arange(candidates), pa.int64()),
            "upload": pa.array([upload] * candidates, pa.int64()),
            "votes": pa.array(np.full(candidates, upload * 10), pa.int64()),
        },
        schema=_READ,
    )


def _ranks(store: SharedArrowStore) -> dict[tuple[str, int, int], int]:
    """each held row's (state, candidate, upload) and its ``most_recent_ind``, as the current version holds them."""
    current = store.current
    assert current is not None
    with read_version(store.root, current.name) as mapped:
        rows = mapped.tables[SNAPSHOTS]
        found = zip(
            rows["state"].to_pylist(),
            rows["candidate_id"].to_pylist(),
            rows["upload"].to_pylist(),
            rows[COLUMN].to_pylist(),
            strict=True,
        )
        return {(state, candidate, upload): rank for state, candidate, upload, rank in found}


def _inodes(store: SharedArrowStore) -> dict[str, int]:
    current = store.current
    assert current is not None
    return {part: os.stat(path).st_ino for part, path in held_parts(store.root, current.name, SNAPSHOTS).items()}


@pytest.fixture
def store(tmp_path: Path) -> SharedArrowStore:
    return SharedArrowStore(tmp_path / "shm", max_bytes=1 << 30)


def _backend(store: SharedArrowStore, columns: ColumnHooks) -> SharedCopyBackend:
    return SharedCopyBackend(
        store,
        {SNAPSHOTS: SNAPSHOTS_SCHEMA},
        {SNAPSHOTS: BY_STATE},
        columns=columns,
        preload=(copy_computed.__name__,),
        pack_below=0,
    )


class TestComputedColumns:
    async def test_a_changes_rows_are_ranked_against_the_held_rows_and_the_held_ranks_it_moves_are_rewritten(
        self, store: SharedArrowStore
    ) -> None:
        backend = _backend(store, NewestUpload())
        try:
            await backend.apply([ChangePage(ROWS, SNAPSHOTS, pa.concat_tables([_uploads("GA", 1), _uploads("VA", 1)]))])
            assert set(_ranks(store).values()) == {1}
            va = _inodes(store)["VA"]
            # a newer upload of GA's candidates 0 and 1: theirs is 1 now, their upload 1 is 2; candidate 2 keeps 1
            await backend.apply([ChangePage(ROWS, SNAPSHOTS, _uploads("GA", 2, candidates=2))])
            ranks = _ranks(store)
            assert {key: rank for key, rank in ranks.items() if key[0] == "GA"} == {
                ("GA", 0, 2): 1,
                ("GA", 1, 2): 1,
                ("GA", 0, 1): 2,
                ("GA", 1, 1): 2,
                ("GA", 2, 1): 1,
            }
            assert {rank for key, rank in ranks.items() if key[0] == "VA"} == {1}
            assert _inodes(store)["VA"] == va, "a state the change does not touch keeps its file"
        finally:
            await backend.close()

    async def test_deleting_the_newest_upload_ranks_the_one_before_it_newest_again(
        self, store: SharedArrowStore
    ) -> None:
        backend = _backend(store, NewestUpload())
        try:
            await backend.apply([ChangePage(ROWS, SNAPSHOTS, _uploads("GA", 1))])
            await backend.apply([ChangePage(ROWS, SNAPSHOTS, _uploads("GA", 2))])
            assert _ranks(store)[("GA", 0, 1)] == 2
            newest = _uploads("GA", 2).select(list(SNAPSHOTS_KEY))
            await backend.apply([ChangePage(KEYS, SNAPSHOTS, newest)])
            assert _ranks(store) == {("GA", 0, 1): 1, ("GA", 1, 1): 1, ("GA", 2, 1): 1}
        finally:
            await backend.close()

    async def test_a_whole_change_ranks_its_own_rows_alone(self, store: SharedArrowStore) -> None:
        backend = _backend(store, NewestUpload())
        try:
            await backend.apply([ChangePage(ROWS, SNAPSHOTS, _uploads("GA", 7))])
            whole = pa.concat_tables([_uploads("GA", 1), _uploads("GA", 2), _uploads("VA", 3)])
            await backend.apply([ChangePage(ROWS, SNAPSHOTS, whole)], whole={SNAPSHOTS})
            ranks = _ranks(store)
            assert ("GA", 0, 7) not in ranks
            assert ranks[("GA", 0, 2)] == 1 and ranks[("GA", 0, 1)] == 2 and ranks[("VA", 0, 3)] == 1
        finally:
            await backend.close()

    async def test_rows_given_back_out_of_the_changes_order_are_refused_and_nothing_is_published(
        self, store: SharedArrowStore
    ) -> None:
        backend = _backend(store, Reordered())
        try:
            with pytest.raises(ValueError, match="one for one"):
                await backend.apply([ChangePage(ROWS, SNAPSHOTS, _uploads("GA", 1))])
            assert store.current is None
        finally:
            await backend.close()

    def test_a_computed_key_column_is_refused(self, store: SharedArrowStore) -> None:
        class OnTheKey:
            computed = {SNAPSHOTS: ("upload",)}

            def compute(self, table: str, change: object) -> object:
                raise AssertionError("never run")

        with pytest.raises(ValueError, match=r"snapshots\.upload"):
            _backend(store, OnTheKey())  # type: ignore[arg-type]
