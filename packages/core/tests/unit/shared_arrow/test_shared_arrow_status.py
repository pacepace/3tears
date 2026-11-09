"""A shared-Arrow store's status: read from its own tallies, never the file system, and exact.

The contracts. :meth:`SharedArrowStore.status` touches no file and waits on no lock, so an event loop may call it
(walking the directories held a serving loop 50-56 ms a call on cobalt dev, 2026-11-07); and its numbers are what
walking the directories finds -- ``held_bytes`` as :meth:`SharedArrowStore.held_bytes` counts it, a file linked into
several versions once, and the versions and hand-offs as :meth:`SharedArrowStore.versions` and
:meth:`SharedArrowStore.handoffs` list them -- after every publish, hand-off, claim, release and retirement.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from threetears.core.shared_arrow import Parts, SharedArrowStore
from threetears.observe import remove_tree as threetears_remove_tree

TABLE = "t"


def _table(value: int, rows: int = 1_000) -> pa.Table:
    """a table whose column ``a`` is ``value`` in every row.

    :return: the table
    :rtype: pa.Table
    """
    return pa.table({"a": pa.array([value] * rows, pa.int64())})


def _parts(values: dict[str, int], rows: int = 1_000) -> Parts:
    """a table in parts, each part's ``a`` its value.

    :return: the parts
    :rtype: Parts
    """
    return Parts(schema=_table(0, rows=0).schema, parts={part: _table(value, rows) for part, value in values.items()})


@pytest.fixture
def store(tmp_path: Path) -> SharedArrowStore:
    """a store with room for a handful of versions.

    :return: the store
    :rtype: SharedArrowStore
    """
    return SharedArrowStore(tmp_path / "shm", max_bytes=64 << 20)


def _walked(store: SharedArrowStore) -> dict[str, Any]:
    """what walking the store's directories finds, as status names it.

    :return: the values
    :rtype: dict[str, Any]
    """
    return {
        "held_bytes": store.held_bytes(),
        "versions": list(store.versions()),
        "handoffs_held": len(store.handoffs()),
    }


def _told(store: SharedArrowStore) -> dict[str, Any]:
    """what the store's status says of the same.

    :return: the values
    :rtype: dict[str, Any]
    """
    status = store.status()
    return {key: status[key] for key in ("held_bytes", "versions", "handoffs_held")}


@contextlib.contextmanager
def _no_file_system(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """every listing and stat of a path raises while the block runs.

    :return: a context manager
    :rtype: Iterator[None]
    """

    def refused(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("status touched the file system")

    with monkeypatch.context() as patched:
        for name in ("iterdir", "stat", "lstat", "exists", "is_dir", "is_file"):
            patched.setattr(Path, name, refused)
        yield


class TestStatusTouchesNoFile:
    def test_status_reads_its_tallies_with_every_listing_and_stat_refused(
        self, store: SharedArrowStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store.publish("v1", {TABLE: _parts({"VA": 1, "GA": 2}), "whole": _table(5)})
        with store.lease("v1"):
            store.publish("v2", {TABLE: _parts({"GA": 20})}, base="v1")
            handed = store.hand({"rows": _table(7)})
            walked = _walked(store)
            with _no_file_system(monkeypatch):
                status = store.status()
            assert {key: status[key] for key in walked} == walked
            assert status["current"] == "v2"
            assert status["versions"] == ["v1", "v2"]
            assert (status["handoffs_open"], status["handoffs_held"]) == (1, 1)
            assert status["held_bytes"] > 0
            store.release(handed)

    def test_status_answers_while_another_thread_holds_the_store_in_a_sweep(
        self, store: SharedArrowStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store.publish("v1", {TABLE: _table(1)})
        entered, go_on = threading.Event(), threading.Event()
        listing = Path.iterdir
        sweeper: list[threading.Thread] = []

        def slow(path: Path) -> Iterator[Path]:
            # the sweeping thread stops in its listing, holding whatever it holds then
            if sweeper and threading.current_thread() is sweeper[0]:
                entered.set()
                go_on.wait(10)
            return listing(path)

        monkeypatch.setattr(Path, "iterdir", slow)
        sweeper.append(threading.Thread(target=store.sweep))
        sweeper[0].start()
        try:
            assert entered.wait(10), "the sweep never listed the directory"
            answered: list[dict[str, Any]] = []
            reading = threading.Thread(target=lambda: answered.append(store.status()))
            reading.start()
            reading.join(5)
            assert answered, "status waited on the sweep"
            assert answered[0]["versions"] == ["v1"]
        finally:
            go_on.set()
            sweeper[0].join(10)


class TestStatusIsExact:
    def test_held_bytes_and_counts_match_the_walk_at_every_step(self, store: SharedArrowStore) -> None:
        steps: list[str] = []

        def check(step: str) -> None:
            steps.append(step)
            # what a publish leaves runs on the store's thread: compared once it has
            store.settle()
            assert _told(store) == _walked(store), f"after {step}"

        check("an empty store")
        store.publish("v1", {TABLE: _parts({"VA": 1, "GA": 2, "TX": 3}), "whole": _table(5)})
        check("a first version")
        v1 = store.lease("v1")
        # v2 links VA and whole from v1: those files counted once
        store.publish(
            "v2", {TABLE: Parts(schema=_table(0, rows=0).schema, parts={"GA": _table(20), "TX": None})}, base="v1"
        )
        check("a version sharing files with a held one")
        slot = store.reserve(1 << 20)
        slot.write({"result": _table(9)})
        # a worker's result counts once it is claimed (its open slot counts at its bound until then): the serving
        # process does not know a worker wrote it before it claims it
        written = sum(entry.stat().st_size for entry in (store.root / "segments" / slot.slot_id).iterdir())
        assert store.status()["held_bytes"] == store.held_bytes() - written
        assert store.status()["slots_open"] == 1
        segment = store.claim(slot)
        assert segment is not None
        check("a result claimed")
        with store.lease("v2"):
            # v3 links the claimed result's file as a part, and every other file from v2
            store.publish(
                "v3",
                {TABLE: Parts(schema=_table(0, rows=0).schema, parts={"NY": segment.files["result"]})},
                base="v2",
            )
            check("a version linking a claimed result")
        segment.close()
        check("the linked result closed: its file stays, in v3")
        first = store.hand({"rows": _table(7)})
        second = store.hand({"rows": _table(8), "more": _table(9)})
        check("two hand-offs")
        store.release(first)
        check("a hand-off released")
        v1.release()
        check("the first version let go and retired")
        store.publish("v4", {"other": _table(4)})
        check("a version sharing nothing, the rest retired")
        store.release(second)
        check("every hand-off released")
        store.publish("v3", {"ignored": _table(0)})
        check("a version already held made current again")
        assert _told(store)["versions"] == ["v3"]

    def test_a_store_started_over_a_held_directory_counts_what_it_finds(self, tmp_path: Path) -> None:
        first = SharedArrowStore(tmp_path / "shm", max_bytes=64 << 20)
        first.publish("v1", {TABLE: _parts({"VA": 1, "GA": 2})})
        with first.lease("v1"):
            first.publish("v2", {TABLE: _parts({"GA": 20})}, base="v1")
            first.settle()
            # a restart while a reader (a worker of the old process, say) still holds v1
            again = SharedArrowStore(tmp_path / "shm", max_bytes=64 << 20)
            assert _told(again) == _walked(again)
            assert "v1" in _told(again)["versions"]


class TestTwoSweepsRemovingOneDirectory:
    def test_a_partial_removal_recorded_after_another_sweep_removed_the_rest_counts_nothing_stale(
        self, store: SharedArrowStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store.publish("v1", {TABLE: _parts({"VA": 1, "GA": 2})})
        held = store.lease("v1")
        store.publish("v2", {"other": _table(4)})
        store.settle()
        # sweep A (the lease's release) retires v1; its removal leaves one file and its listing of what is left
        # waits for sweep B, which removes the directory whole and forgets it
        left_listed, b_done = threading.Event(), threading.Event()
        removing = threetears_remove_tree
        stat = Path.stat
        a_thread: list[threading.Thread] = []
        partial: list[Path] = []

        def on_a() -> bool:
            return bool(a_thread) and threading.current_thread() is a_thread[0]

        def remove_partly(path: Any, **kwargs: Any) -> None:
            if on_a() and Path(path).name.startswith(".retiring-") and not partial:
                # the file left is a table's, so a stale record of it would count real bytes
                files = sorted(Path(path).iterdir(), key=lambda entry: entry.name.startswith("."))
                for entry in files[1:]:
                    entry.unlink()
                partial.append(files[0])
                return
            removing(path, **kwargs)

        def stat_then_wait(path: Path, *args: Any, **kwargs: Any) -> Any:
            answer = stat(path, *args, **kwargs)
            if on_a() and partial and path == partial[0] and not left_listed.is_set():
                left_listed.set()
                b_done.wait(1)  # under the store's lock B cannot run: A goes on alone after a moment
            return answer

        monkeypatch.setattr("threetears.core.shared_arrow.remove_tree", remove_partly)
        monkeypatch.setattr(Path, "stat", stat_then_wait)
        a_thread.append(threading.Thread(target=held.release))
        a_thread[0].start()
        assert left_listed.wait(10), "sweep A never listed what its removal left"
        b_sweep = threading.Thread(target=store.sweep)
        b_sweep.start()
        b_sweep.join(10)
        b_done.set()
        a_thread[0].join(10)
        b_sweep.join(10)
        assert not a_thread[0].is_alive() and not b_sweep.is_alive()
        assert _told(store) == _walked(store)
        assert _told(store)["versions"] == ["v2"]
