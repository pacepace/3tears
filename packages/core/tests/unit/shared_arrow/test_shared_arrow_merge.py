"""A group of a table's parts merged into one part: readers see the same rows, and the merge writes only the group.

The contracts (``merge_parts``, ``merged_rows``). A tiered publish (a level-1 part per ten run parts, and so on) merges
neighbouring parts into one: every reader of the new version sees exactly the rows, in the same order, it saw before;
the parts outside the group are the base's own files (the same files, named again); and the only bytes written are the
merged part's, no more than the group's own files hold. A group that is not a run of neighbours, or a merged name that
sorts past another part, would reorder rows, so it is refused.
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path

import pyarrow as pa
import pytest

from threetears.core.shared_arrow import (
    Parts,
    SegmentSlot,
    SharedArrowStore,
    merge_parts,
    merged_rows,
    read_version,
    version_files,
)

#: the run parts of the table, and the rows each holds (a few batches each, as a run's load writes them)
RUNS = 30
ROWS = 2_000


def _run(index: int) -> pa.Table:
    """one run's rows: distinct in every run, three record batches.

    :return: the run
    :rtype: pa.Table
    """
    first = index * ROWS
    table = pa.table(
        {
            "run": pa.array([index] * ROWS, pa.int64()),
            "k": pa.array(range(first, first + ROWS), pa.int64()),
            "s": pa.array([f"v{n % 97}" for n in range(first, first + ROWS)], pa.string()),
        }
    )
    return pa.Table.from_batches(table.to_batches(max_chunksize=ROWS // 3 + 1))


def _name(index: int) -> str:
    return f"r{index:04d}"


@pytest.fixture
def store(tmp_path: Path) -> SharedArrowStore:
    """a store holding ``v1``: the table in thirty run parts, and a second table whole.

    :return: the store
    :rtype: SharedArrowStore
    """
    made = SharedArrowStore(tmp_path / "shm", max_bytes=256 << 20)
    runs = {_name(index): _run(index) for index in range(RUNS)}
    schema = next(iter(runs.values())).schema
    made.publish("v1", {"lsf": Parts(schema=schema, parts=runs), "other": pa.table({"a": [1, 2, 3]})})
    return made


def _version(store: SharedArrowStore, version: str) -> Path:
    return store.root / "versions" / version


def _inodes(store: SharedArrowStore, version: str) -> dict[str, tuple[int, int]]:
    return {name: (held.path.stat().st_ino, held.nbytes) for name, held in version_files(store.root, version).items()}


def _rows(store: SharedArrowStore, version: str) -> dict[str, list[object]]:
    with read_version(store.root, version) as mapped:
        return {name: mapped.tables[name].to_pylist() for name in sorted(mapped.tables)}


class TestAMergedGroup:
    def test_readers_see_identical_rows_and_the_merge_writes_only_the_groups_bytes(
        self, store: SharedArrowStore
    ) -> None:
        group = [_name(index) for index in range(10)]
        before = _rows(store, "v1")
        old = _inodes(store, "v1")
        parts = merge_parts(store, "v1", "lsf", group, into=f"{group[0]}.1")
        store.publish("v2", {"lsf": parts}, base="v1")
        after = _rows(store, "v2")
        assert after == before
        new = _inodes(store, "v2")
        # every part outside the group, and the other table, is the base's own file
        kept = {name for name in old if not any(name == f"lsf@{part}.arrow" for part in group)}
        kept.discard("lsf@.arrow")
        assert {name: new[name][0] for name in kept} == {name: old[name][0] for name in kept}
        assert not any(f"lsf@{part}.arrow" in new for part in group)
        written = {name: size for name, (inode, size) in new.items() if inode not in {i for i, _ in old.values()}}
        group_bytes = sum(old[f"lsf@{part}.arrow"][1] for part in group)
        merged = written.pop("lsf@r0000.1.arrow")
        # the only other file written is the table's schema file, rewritten by any publish in parts (no rows)
        assert set(written) == {"lsf@.arrow"} and written["lsf@.arrow"] < 4096
        assert 0.9 * group_bytes <= merged <= group_bytes, (merged, group_bytes)
        print(f"group {group_bytes} B in {len(group)} parts -> merged {merged} B; other files written {written}")

    def test_a_merged_part_merges_again_a_level_up(self, store: SharedArrowStore) -> None:
        before = _rows(store, "v1")
        store.publish(
            "v2", {"lsf": merge_parts(store, "v1", "lsf", [_name(i) for i in range(10)], "r0000.1")}, base="v1"
        )
        store.publish(
            "v3", {"lsf": merge_parts(store, "v2", "lsf", [_name(i) for i in range(10, 20)], "r0010.1")}, base="v2"
        )
        store.publish("v4", {"lsf": merge_parts(store, "v3", "lsf", ["r0000.1", "r0010.1"], "r0000.2")}, base="v3")
        assert _rows(store, "v4") == before

    def test_a_worker_merges_into_a_result_that_is_linked_as_the_part(self, store: SharedArrowStore) -> None:
        group = [_name(index) for index in range(3)]
        before = _rows(store, "v1")
        with store.lease("v1"):
            slot: SegmentSlot = store.reserve(64 << 20)
            # what a worker process runs: the slot and the opener pickle; the opener locks the version itself
            opener = merged_rows(store.root, "v1", "lsf", group)
            slot.write({"merged": opener()})
        claimed = store.claim(slot)
        assert claimed is not None
        merged_file = claimed.files["merged"]
        lsf_schema = claimed.tables["merged"].schema
        store.publish(
            "v2",
            {"lsf": Parts(schema=lsf_schema, parts={**dict.fromkeys(group), "r0000.1": merged_file})},
            base="v1",
        )
        claimed.close()
        assert _rows(store, "v2") == before

    def test_an_opener_never_read_holds_no_lock(self, store: SharedArrowStore) -> None:
        reader = merged_rows(store.root, "v1", "lsf", [_name(0), _name(1)])()
        assert reader.schema.names == ["run", "k", "s"]
        lock = _version(store, "v1") / ".lock"
        descriptor = os.open(lock, os.O_RDONLY)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(descriptor)


class TestAMergeThatWouldReorderRows:
    def test_a_group_that_is_not_a_run_of_neighbours_is_refused(self, store: SharedArrowStore) -> None:
        with pytest.raises(ValueError, match="would move parts"):
            merge_parts(store, "v1", "lsf", [_name(0), _name(2)], into="r0000.1")

    def test_a_merged_name_that_sorts_past_another_part_is_refused(self, store: SharedArrowStore) -> None:
        with pytest.raises(ValueError, match="would move parts"):
            merge_parts(store, "v1", "lsf", [_name(0), _name(1)], into="r0005")

    def test_a_part_the_version_does_not_hold_is_refused(self, store: SharedArrowStore) -> None:
        with pytest.raises(LookupError):
            merge_parts(store, "v1", "lsf", [_name(0), "nope"], into="r0000.1")

    def test_one_part_is_not_a_merge(self, store: SharedArrowStore) -> None:
        with pytest.raises(ValueError, match="at least two"):
            merge_parts(store, "v1", "lsf", [_name(0)], into="r0000.1")
