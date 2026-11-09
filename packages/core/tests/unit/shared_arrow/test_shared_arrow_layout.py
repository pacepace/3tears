"""A version is a manifest naming part files: nothing leaks, a crash leaves nothing a start cannot reclaim, and the
sweep a publish leaves runs off its path.

The contracts. The parts directory holds exactly the files the versions held name, each once, and no empty directory;
a file a reader's version names stays until the version goes. A publish that fails keeps nothing; one a process died in
the middle of, and the files of versions nobody holds, go when a new store starts, and a version a reader still holds
stays whole. A version written before manifests (its directory holding its files itself) is removed when nobody holds
it, and read, published on and removed in turn when a reader does. A publish returns before the sweep it leaves: a
slow removal never holds it, one sweeping thread runs at a time, and a publish short of room sweeps first itself.
"""

from __future__ import annotations

import fcntl
import os
import threading
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from threetears.core.shared_arrow import (
    MappedVersion,
    Parts,
    SharedArrowFullError,
    SharedArrowStore,
    version_files,
)

TABLE = "t"
SCHEMA = pa.schema([("a", pa.int64())])


def _rows(value: int, rows: int = 1_000) -> pa.Table:
    """a part whose ``a`` is ``value`` in every row.

    :return: the part
    :rtype: pa.Table
    """
    return pa.table({"a": pa.array([value] * rows, pa.int64())})


def _parts(values: dict[str, int | None]) -> Parts:
    """a table in parts, None dropping a part.

    :return: the parts
    :rtype: Parts
    """
    return Parts(schema=SCHEMA, parts={part: None if value is None else _rows(value) for part, value in values.items()})


def _on_disk(store: SharedArrowStore) -> set[Path]:
    """every file under the store's parts directory.

    :return: the files
    :rtype: set[Path]
    """
    return {file for directory in (store.root / "parts").iterdir() for file in directory.iterdir()}


def _named(store: SharedArrowStore) -> set[Path]:
    """every file a version the store holds names.

    :return: the files
    :rtype: set[Path]
    """
    return {held.path for version in store.versions() for held in version_files(store.root, version).values()}


def _sums(store: SharedArrowStore, version: str) -> list[int]:
    """the table's ``a`` column of a version, mapped.

    :return: the values
    :rtype: list[int]
    """
    with MappedVersion(store.root, version) as mapped:
        return list(mapped.tables[TABLE].column("a").to_pylist())


@pytest.fixture
def store(tmp_path: Path) -> SharedArrowStore:
    """a store with room for a handful of versions.

    :return: the store
    :rtype: SharedArrowStore
    """
    return SharedArrowStore(tmp_path / "shm", max_bytes=64 << 20)


class TestNothingLeaks:
    def test_the_parts_directory_holds_exactly_what_the_versions_held_name(self, store: SharedArrowStore) -> None:
        store.publish("v0", {TABLE: _parts({f"S{index}": index for index in range(6)}), "whole": _rows(9)})
        held = store.lease("v0")
        for step in range(1, 12):
            change: dict[str, int | None] = {f"S{step % 6}": 100 + step, f"S{(step + 2) % 6}": None}
            tables: dict[str, Any] = {TABLE: _parts(change)}
            if step % 4 == 0:
                tables["whole"] = _rows(step)
            store.publish(f"v{step}", tables, base=f"v{step - 1}")
            store.settle()
            assert _on_disk(store) == _named(store), f"after publish {step}"
            assert all(any(directory.iterdir()) for directory in (store.root / "parts").iterdir()), "an empty part dir"
        assert set(store.versions()) == {"v0", "v11"}
        held.release()
        assert store.versions() == ("v11",)
        assert _on_disk(store) == _named(store)
        assert store.status()["held_bytes"] == store.held_bytes()

    def test_a_file_a_readers_version_names_stays_until_the_version_goes(self, store: SharedArrowStore) -> None:
        store.publish("v1", {TABLE: _parts({"VA": 1, "GA": 2})})
        before = version_files(store.root, "v1")["t@GA.arrow"].path
        with MappedVersion(store.root, "v1") as mapped:
            store.publish("v2", {TABLE: _parts({"GA": 20})}, base="v1")
            store.settle()
            store.sweep()
            assert before.exists(), "a file the reader's version names was removed"
            # parts in name order: GA, then VA
            assert mapped.tables[TABLE].column("a").to_pylist() == [2] * 1_000 + [1] * 1_000
        store.sweep()
        assert not before.exists()
        assert _sums(store, "v2") == [20] * 1_000 + [1] * 1_000


class TestACrashLeavesNothingAStartCannotReclaim:
    def test_a_publish_that_fails_mid_write_keeps_nothing(self, store: SharedArrowStore) -> None:
        store.publish("v1", {TABLE: _parts({"VA": 1})})
        before = _on_disk(store)

        def broken() -> pa.Table:
            raise RuntimeError("the source failed")

        with pytest.raises(RuntimeError, match="source failed"):
            store.publish("v2", {TABLE: Parts(schema=SCHEMA, parts={"GA": _rows(2), "TX": broken})}, base="v1")
        assert _on_disk(store) == before
        assert store.versions() == ("v1",)
        assert [entry.name for entry in (store.root / "versions").iterdir() if entry.name.startswith(".")] == []

    def test_a_start_removes_what_a_dead_process_left_and_keeps_a_held_version_whole(self, tmp_path: Path) -> None:
        first = SharedArrowStore(tmp_path / "shm", max_bytes=64 << 20)
        first.publish("v1", {TABLE: _parts({"VA": 1, "GA": 2})})
        first.publish("v2", {TABLE: _parts({"GA": 20})}, base="v1")
        first.publish("v3", {TABLE: _parts({"TX": 3})}, base="v2")
        first.settle()
        # a worker of the old process still maps v3; the process died mid-publish of v4: its part files written, its
        # manifest staged, never renamed into place
        held = os.open(tmp_path / "shm" / "versions" / "v3" / ".lock", os.O_RDONLY)
        fcntl.flock(held, fcntl.LOCK_SH)
        dead = tmp_path / "shm" / "parts" / "0dead"
        dead.mkdir()
        (dead / "t@NY.arrow").write_bytes(b"x" * 4096)
        staged = tmp_path / "shm" / "versions" / ".staging-v4-0dead"
        staged.mkdir()
        (staged / "MANIFEST").write_text("shared-arrow manifest 1\nt@NY.arrow\t0dead\t4096\n")
        (staged / ".lock").touch()
        (tmp_path / "shm" / "versions" / ".staging-CURRENT-0dead").write_text("v4")
        try:
            again = SharedArrowStore(tmp_path / "shm", max_bytes=64 << 20)
            assert again.versions() == ("v3",)
            assert not dead.exists(), "a dead publish's part files were kept"
            assert [entry.name for entry in (tmp_path / "shm" / "versions").iterdir()] == ["v3"]
            assert _on_disk(again) == _named(again)
            assert _sums(again, "v3") == [20] * 1_000 + [3] * 1_000 + [1] * 1_000
            assert again.status()["held_bytes"] == again.held_bytes()
            # the new store publishes on the version it kept
            again.publish("v5", {TABLE: _parts({"VA": 10})}, base="v3")
            assert _sums(again, "v5") == [20] * 1_000 + [3] * 1_000 + [10] * 1_000
        finally:
            os.close(held)
        again.settle()
        again.sweep()
        assert again.versions() == ("v5",)
        assert _on_disk(again) == _named(again)
        assert again.held_bytes() == again.status()["held_bytes"]

    def test_a_start_with_nobody_holding_anything_leaves_the_directory_empty(self, tmp_path: Path) -> None:
        first = SharedArrowStore(tmp_path / "shm", max_bytes=64 << 20)
        first.publish("v1", {TABLE: _parts({"VA": 1, "GA": 2})})
        first.publish("v2", {TABLE: _parts({"GA": 20})}, base="v1")
        first.settle()
        again = SharedArrowStore(tmp_path / "shm", max_bytes=64 << 20)
        assert again.versions() == ()
        assert list((tmp_path / "shm" / "parts").iterdir()) == []
        assert again.held_bytes() == 0 == again.status()["held_bytes"]


def _flat_version(directory: Path, parts: dict[str, int]) -> None:
    """a version as stores wrote it before manifests: its directory holding a lock file, its table's schema file and
    a file per part.

    :return: nothing
    :rtype: None
    """
    directory.mkdir(parents=True)
    (directory / ".lock").touch()
    options = ipc.IpcWriteOptions(compression=None)
    files = {"t@.arrow": None, **{f"t@{part}.arrow": value for part, value in parts.items()}}
    for name, value in files.items():
        with pa.OSFile(str(directory / name), "wb") as sink, ipc.new_file(sink, SCHEMA, options=options) as writer:
            if value is not None:
                writer.write_table(_rows(value))


class TestAVersionWrittenBeforeManifests:
    def test_one_nobody_holds_is_removed_at_the_start(self, tmp_path: Path) -> None:
        _flat_version(tmp_path / "shm" / "versions" / "c-0", {"VA": 1})
        store = SharedArrowStore(tmp_path / "shm", max_bytes=64 << 20)
        assert store.versions() == ()
        assert store.held_bytes() == 0

    def test_one_a_reader_holds_is_read_published_on_and_removed_in_turn(self, tmp_path: Path) -> None:
        old = tmp_path / "shm" / "versions" / "c-0"
        _flat_version(old, {"VA": 1, "GA": 2})
        held = os.open(old / ".lock", os.O_RDONLY)
        fcntl.flock(held, fcntl.LOCK_SH)
        try:
            store = SharedArrowStore(tmp_path / "shm", max_bytes=64 << 20)
            assert store.versions() == ("c-0",)
            assert store.status()["held_bytes"] == store.held_bytes() > 0
            assert _sums(store, "c-0") == [2] * 1_000 + [1] * 1_000
            assert set(version_files(store.root, "c-0")) == {"t@.arrow", "t@GA.arrow", "t@VA.arrow"}
            with store.lease("c-0"):
                store.publish("c-1", {TABLE: _parts({"GA": 20, "TX": 3})}, base="c-0")
            assert _sums(store, "c-1") == [20] * 1_000 + [3] * 1_000 + [1] * 1_000
            store.settle()
            assert store.status()["held_bytes"] == store.held_bytes()
        finally:
            os.close(held)
        store.sweep()
        assert store.versions() == ("c-1",)
        assert not old.exists()
        assert _on_disk(store) == _named(store)
        assert _sums(store, "c-1") == [20] * 1_000 + [3] * 1_000 + [1] * 1_000
        assert store.status()["held_bytes"] == store.held_bytes()


class TestTheSweepLeavesThePublishPath:
    def test_a_publish_returns_while_the_sweep_it_left_is_held_and_readers_see_no_difference(
        self, store: SharedArrowStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store.publish("v1", {TABLE: _parts({"VA": 1, "GA": 2})})
        removing, go = threading.Event(), threading.Event()
        unlink = os.unlink

        def held_unlink(path: Any, *args: Any, **kwargs: Any) -> None:
            if threading.current_thread().name == "shared-arrow-settle":
                removing.set()
                go.wait(30)
            unlink(path, *args, **kwargs)

        monkeypatch.setattr(os, "unlink", held_unlink)
        threads = threading.active_count()
        store.publish("v2", {TABLE: _parts({"GA": 20})}, base="v1")
        assert removing.wait(10), "the publish's sweep never ran"
        # the sweep is held in its removal: publishes go on, and read whole
        for step in range(3, 8):
            store.publish(f"v{step}", {TABLE: _parts({"GA": step})}, base=f"v{step - 1}")
            assert _sums(store, f"v{step}") == [step] * 1_000 + [1] * 1_000
        settlers = [thread for thread in threading.enumerate() if thread.name == "shared-arrow-settle"]
        assert len(settlers) == 1, "the sweeps piled up"
        assert threading.active_count() <= threads + 1
        go.set()
        store.settle()
        assert store.versions() == ("v7",)
        assert _on_disk(store) == _named(store)
        assert not [thread for thread in threading.enumerate() if thread.name == "shared-arrow-settle"]

    def test_a_publish_short_of_room_sweeps_itself_before_it_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        one = _rows(1, rows=200_000)
        bounded = SharedArrowStore(tmp_path / "shm", max_bytes=int(one.nbytes * 2.5))
        # the store's thread never sweeps: what the publishes leave stays until a publish short of room sweeps
        go = threading.Event()
        sweep = bounded.sweep

        def held_sweep() -> list[str]:
            if threading.current_thread().name == "shared-arrow-settle":
                go.wait(30)
            return sweep()

        monkeypatch.setattr(bounded, "sweep", held_sweep)
        try:
            for step in range(5):
                bounded.publish(f"v{step}", {TABLE: _rows(step, rows=200_000)})
            # room for two versions: each publish short of it swept the ones superseded itself
            assert bounded.versions() == ("v3", "v4"), "a publish short of room swept the versions superseded"
            assert bounded.refused == 0
            held = bounded.lease()
            bounded.publish("v5", {TABLE: _rows(5, rows=200_000)})
            with pytest.raises(SharedArrowFullError):
                bounded.publish("v6", {TABLE: _rows(6, rows=200_000)})
            assert bounded.refused == 1
            held.release()
        finally:
            go.set()
            bounded.settle()
