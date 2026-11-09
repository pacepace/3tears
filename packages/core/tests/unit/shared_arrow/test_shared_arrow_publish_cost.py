"""A publish costs what it changes, never what the version holds.

The contracts. A publish of k changed parts on a version of N parts (``publish(base=)``, :class:`Parts`) does the same
file-system work at N parts and at 10N: counted, not timed, as the calls this process makes (an audit hook, and every
``os.stat``), on the publishing thread, and again with the sweep it leaves behind run to its end. Timed, it takes as
long at 10N as at N (within 1.5x), and within a calibrated multiple of writing the k parts' bytes bare, measured in
the test beside it.

Why: the reports pod applies a commit to its copy (140k rows into 9.5M, 2026-11-09) in 0.49 s against a bare DuckDB
upsert's 0.14 s; 0.20 s of it was the publish, which hard-linked every unchanged part (about 600) into the new
version's directory and ran two sweeps unlinking a retired version's links on its path.
"""

from __future__ import annotations

import collections
import contextlib
import os
import statistics
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from threetears.core.shared_arrow import Parts, SharedArrowStore

TABLE = "t"

#: the parts of the version at the small size, and ten times as many at the large
PARTS = 200
TEN_TIMES = 10 * PARTS

#: the parts each publish changes
CHANGED = 4

#: an unchanged part's rows (small: only their number is what is measured) and a changed part's (a state's rows of a
#: run, as a change writes them)
KEPT_ROWS = 100
CHANGED_ROWS = 64_000

#: the audit events of file-system work: opening, listing, linking, renaming, removing and making
_FS_EVENTS = frozenset(
    {
        "open",
        "os.listdir",
        "os.scandir",
        "os.link",
        "os.symlink",
        "os.rename",
        "os.remove",
        "os.rmdir",
        "os.mkdir",
        "os.truncate",
        "os.chmod",
        "os.utime",
        "shutil.rmtree",
    }
)

#: what is counting now: the counter, and the thread whose calls it counts (None for every thread)
_COUNTING: list[tuple[collections.Counter[str], int | None]] = []
_HOOKED: list[bool] = []


def _audited(event: str, _args: tuple[Any, ...]) -> None:
    """count a file-system event while a count is open (:func:`_counted`).

    :return: nothing
    :rtype: None
    """
    if _COUNTING and event in _FS_EVENTS:
        counter, thread = _COUNTING[-1]
        if thread is None or thread == threading.get_ident():
            counter[event] += 1


@contextlib.contextmanager
def _counted(monkeypatch: pytest.MonkeyPatch, *, thread: int | None) -> Iterator[collections.Counter[str]]:
    """count the file-system calls made in the block: every audited event and every ``os.stat`` / ``os.lstat``.

    :param thread: the thread whose calls count, or None for every thread
    :return: a context manager yielding the counter, filled as the block runs
    :rtype: Iterator[collections.Counter[str]]
    """
    if not _HOOKED:
        # an audit hook stays for the process; it counts only while a count is open
        sys.addaudithook(_audited)
        _HOOKED.append(True)
    counter: collections.Counter[str] = collections.Counter()
    stat, lstat = os.stat, os.lstat

    def counting(real: Any, name: str) -> Any:
        def call(*args: Any, **kwargs: Any) -> Any:
            if thread is None or thread == threading.get_ident():
                counter[name] += 1
            return real(*args, **kwargs)

        return call

    with monkeypatch.context() as patched:
        patched.setattr(os, "stat", counting(stat, "os.stat"))
        patched.setattr(os, "lstat", counting(lstat, "os.lstat"))
        _COUNTING.append((counter, thread))
        try:
            yield counter
        finally:
            _COUNTING.pop()


def _kept(index: int) -> pa.Table:
    """an unchanged part.

    :return: the part
    :rtype: pa.Table
    """
    return pa.table({"a": pa.array([index] * KEPT_ROWS, pa.int64()), "b": pa.array([index] * KEPT_ROWS, pa.float64())})


def _changed(value: int) -> pa.Table:
    """a changed part's rows.

    :return: the part
    :rtype: pa.Table
    """
    return pa.table(
        {"a": pa.array([value] * CHANGED_ROWS, pa.int64()), "b": pa.array([value] * CHANGED_ROWS, pa.float64())}
    )


class _Copy:
    """a store holding one version of ``parts`` parts, changed ``CHANGED`` parts a publish, each on the last.

    :param root: the store's directory
    :param parts: the parts of the version
    """

    def __init__(self, root: Path, parts: int) -> None:
        self.store = SharedArrowStore(root, max_bytes=4 << 30)
        self.schema = _kept(0).schema
        self.parts = parts
        self.last = "v0"
        self.publishes = 0
        self.store.publish(
            self.last, {TABLE: Parts(schema=self.schema, parts={f"p{i:05d}": _kept(i) for i in range(parts)})}
        )

    def change(self) -> Parts:
        """the next change: ``CHANGED`` parts spread across the version, new rows each.

        :return: the change
        :rtype: Parts
        """
        self.publishes += 1
        step = self.parts // CHANGED
        first = self.publishes % step
        return Parts(
            schema=self.schema,
            parts={f"p{first + n * step:05d}": _changed(self.publishes) for n in range(CHANGED)},
        )

    def publish(self, change: Parts) -> None:
        """publish ``change`` on the last version.

        :return: nothing
        :rtype: None
        """
        name = f"v{self.publishes}"
        self.store.publish(name, {TABLE: change}, base=self.last)
        self.last = name

    def swept(self) -> None:
        """the sweep a publish leaves behind, run to its end.

        :return: nothing
        :rtype: None
        """
        self.store.sweep()


@pytest.fixture
def copies(tmp_path: Path) -> tuple[_Copy, _Copy]:
    """a copy of :data:`PARTS` parts and one of :data:`TEN_TIMES`.

    :return: the two copies
    :rtype: tuple[_Copy, _Copy]
    """
    return _Copy(tmp_path / "small", PARTS), _Copy(tmp_path / "large", TEN_TIMES)


def _publish_counts(copy: _Copy, monkeypatch: pytest.MonkeyPatch) -> collections.Counter[str]:
    """the file-system calls one publish makes on its own thread, the sweep it left behind run first and after.

    :return: the counts
    :rtype: collections.Counter[str]
    """
    change = copy.change()
    copy.swept()
    with _counted(monkeypatch, thread=threading.get_ident()) as counter:
        copy.publish(change)
    copy.swept()
    return counter


class TestAPublishDoesWhatItChanges:
    def test_the_publishing_thread_does_the_same_file_work_at_n_parts_and_at_ten_n(
        self, copies: tuple[_Copy, _Copy], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        small, large = copies
        for copy in copies:
            _publish_counts(copy, monkeypatch)  # the first publish on a base
        at_n = _publish_counts(small, monkeypatch)
        at_ten_n = _publish_counts(large, monkeypatch)
        print(f"file calls of one publish: {PARTS} parts {dict(at_n)}; {TEN_TIMES} parts {dict(at_ten_n)}")
        assert at_ten_n == at_n, f"a publish's file work grew with the parts it left alone: {at_n} -> {at_ten_n}"

    def test_the_publish_and_the_sweep_it_leaves_do_the_same_file_work_at_n_parts_and_at_ten_n(
        self, copies: tuple[_Copy, _Copy], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        small, large = copies
        counts = []
        for copy in (small, large, small, large):
            change = copy.change()
            copy.store.settle()
            with _counted(monkeypatch, thread=None) as counter:
                copy.publish(change)
                copy.store.settle()
            counts.append(counter)
        at_n, at_ten_n = counts[2], counts[3]
        print(f"file calls of a publish and its sweep: {PARTS} parts {dict(at_n)}; {TEN_TIMES} parts {dict(at_ten_n)}")
        assert at_ten_n == at_n, f"a publish and its sweep grew with the parts left alone: {at_n} -> {at_ten_n}"


def _bare_write(directory: Path, tables: list[pa.Table]) -> float:
    """the seconds writing ``tables`` takes bare: each an uncompressed Arrow IPC file in a fresh directory, beside an
    empty schema file and a small text file, the directory renamed into place -- a version's own file work, with no
    store around it.

    :return: seconds
    :rtype: float
    """
    staging = directory / f".bare-{uuid.uuid4().hex}"
    started = time.perf_counter()
    staging.mkdir()
    options = ipc.IpcWriteOptions(compression=None)
    for index, table in enumerate(tables):
        path = str(staging / f"{index}.arrow")
        with pa.OSFile(path, "wb") as sink, ipc.new_file(sink, table.schema, options=options) as writer:
            for batch in table.to_batches():
                writer.write_batch(batch)
    with pa.OSFile(str(staging / "schema.arrow"), "wb") as sink, ipc.new_file(sink, tables[0].schema, options=options):
        pass
    (staging / "names").write_text("x" * 64)
    os.rename(staging, directory / f"done-{uuid.uuid4().hex}")
    return time.perf_counter() - started


#: rounds of each measure, after one to warm
ROUNDS = 9

#: the most a publish at ten times the parts may take against one at the parts
FLAT = 1.5

#: the most a publish may take against its bare write (:func:`_bare_write`): the store's own bookkeeping, a lock file
#: and the version made current
BARE_MULTIPLE = 2.0


@pytest.mark.loop_timing
class TestAPublishTakesAsLongAsWhatItChanges:
    """timed: run apart and serially, as the ``loop_timing`` suite runs its tests, since a bound on wall-clock time is
    a bound on the machine beside it too. The two sizes, and the bare write, are measured in turn, round by round, so
    whatever else the machine does falls on each alike."""

    def test_a_publish_at_ten_n_parts_takes_as_long_as_at_n_and_near_its_bare_write(
        self, copies: tuple[_Copy, _Copy], tmp_path: Path
    ) -> None:
        small, large = copies
        bare_dir = tmp_path / "bare"
        bare_dir.mkdir()
        times: dict[str, list[float]] = {"n": [], "ten_n": [], "bare": []}
        for round_ in range(ROUNDS + 1):
            for key, copy in (("n", small), ("ten_n", large)):
                change = copy.change()
                copy.swept()
                started = time.perf_counter()
                copy.publish(change)
                elapsed = time.perf_counter() - started
                copy.swept()
                if round_:
                    times[key].append(elapsed)
            bare = _bare_write(bare_dir, [_changed(round_) for _ in range(CHANGED)])
            if round_:
                times["bare"].append(bare)
        n, ten_n, bare = (statistics.median(times[key]) for key in ("n", "ten_n", "bare"))
        print(
            f"publish of {CHANGED} parts: {PARTS} parts {n * 1000:.2f} ms; {TEN_TIMES} parts {ten_n * 1000:.2f} ms; "
            f"bare write {bare * 1000:.2f} ms; ratio {ten_n / bare:.2f}"
        )
        assert ten_n <= FLAT * n, f"a publish at {TEN_TIMES} parts took {ten_n / n:.2f}x one at {PARTS}"
        assert ten_n <= BARE_MULTIPLE * bare, f"a publish took {ten_n / bare:.2f}x its bare write"
