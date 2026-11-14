"""the scans' on-disk memo answers only for the content it was computed from, and never fails a scan.

:mod:`tests.enforcement.scan_cache` keeps what the workspace scans computed across processes: xdist
workers, a pool's batches. A cache that answered for content it no longer has would let a gate pass
over code it never read, so these hold the key to the content and the code, a damaged or unwritable
store to a recompute, and concurrent writers to whole stores.
"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path

from tests.enforcement.scan_cache import CACHE_DIRECTORY, ScanCache, content_key, files_key

_REPO_ROOT = Path(__file__).resolve().parents[2]


class _Counted:
    """a computation that records each item it was asked for."""

    def __init__(self, inputs: Path) -> None:
        self.inputs = inputs
        self.calls: list[str] = []

    def key(self, item: str) -> str:
        return content_key(item, (self.inputs / item).read_bytes())

    def compute(self, item: str) -> str:
        self.calls.append(item)
        return (self.inputs / item).read_text().upper()


def _inputs(tmp_path: Path, **files: str) -> Path:
    inputs = tmp_path / "inputs"
    inputs.mkdir(exist_ok=True)
    for name, text in files.items():
        (inputs / name).write_text(text)
    return inputs


def test_a_second_process_takes_the_stored_value_and_computes_nothing(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, a="alpha", b="beta")
    first, second = _Counted(inputs), _Counted(inputs)

    assert ScanCache(tmp_path / "cache", "code").memoised("s", ["a", "b"], first.key, first.compute) == {
        "a": "ALPHA",
        "b": "BETA",
    }
    # a fresh instance reads the directory, as another worker or batch would
    assert ScanCache(tmp_path / "cache", "code").memoised("s", ["a", "b"], second.key, second.compute) == {
        "a": "ALPHA",
        "b": "BETA",
    }
    assert first.calls == ["a", "b"]
    assert second.calls == []


def test_an_edited_file_is_computed_again_and_only_it(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, a="alpha", b="beta")
    cache = ScanCache(tmp_path / "cache", "code")
    cache.memoised("s", ["a", "b"], _Counted(inputs).key, _Counted(inputs).compute)

    (inputs / "a").write_text("gamma")
    again = _Counted(inputs)

    assert cache.memoised("s", ["a", "b"], again.key, again.compute) == {"a": "GAMMA", "b": "BETA"}
    assert again.calls == ["a"]


def test_a_store_written_by_other_code_is_never_read(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, a="alpha")
    ScanCache(tmp_path / "cache", "old code").memoised("s", ["a"], _Counted(inputs).key, lambda _: "STALE")
    fresh = _Counted(inputs)

    assert ScanCache(tmp_path / "cache", "new code").memoised("s", ["a"], fresh.key, fresh.compute) == {"a": "ALPHA"}
    assert fresh.calls == ["a"]
    # and the old code's store is gone, so the directory does not grow with every change of code
    assert len(list((tmp_path / "cache").glob("s-*.pickle"))) == 1


def test_a_value_whose_input_moved_while_it_was_computed_is_not_stored(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, a="alpha")
    cache = ScanCache(tmp_path / "cache", "code")
    counted = _Counted(inputs)

    def edits_while_reading(item: str) -> str:
        value = counted.compute(item)
        (inputs / item).write_text("edited mid-scan")
        return value

    assert cache.memoised("s", ["a"], counted.key, edits_while_reading) == {"a": "ALPHA"}
    (inputs / "a").write_text("alpha")
    later = _Counted(inputs)
    cache.memoised("s", ["a"], later.key, later.compute)

    assert later.calls == ["a"], "a value computed while its input moved was stored under the old content"


def test_a_truncated_or_corrupt_store_reads_as_a_miss(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, a="alpha")
    cache = ScanCache(tmp_path / "cache", "code")
    cache.memoised("s", ["a"], _Counted(inputs).key, _Counted(inputs).compute)
    (store,) = (tmp_path / "cache").glob("s-*.pickle")
    whole = store.read_bytes()

    for damaged in (whole[: len(whole) // 2], whole[:-1] + bytes([whole[-1] ^ 0xFF]), b"", b"not a store"):
        store.write_bytes(damaged)
        counted = _Counted(inputs)
        assert cache.memoised("s", ["a"], counted.key, counted.compute) == {"a": "ALPHA"}
        assert counted.calls == ["a"]
        # and the recompute replaced it with a whole store
        assert store.read_bytes() == whole


def test_a_directory_that_cannot_be_written_recomputes_and_fails_nothing(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, a="alpha")
    blocker = tmp_path / "a file"
    blocker.write_text("")
    # a cache directory below a regular file can be neither created nor read
    cache = ScanCache(blocker / "cache", "code")

    for _ in range(2):
        counted = _Counted(inputs)
        assert cache.memoised("s", ["a"], counted.key, counted.compute) == {"a": "ALPHA"}
        assert counted.calls == ["a"]


def test_concurrent_writers_leave_a_whole_store_and_readers_see_only_whole_ones(tmp_path: Path) -> None:
    """eight writers, each with its own item set, race to replace one store while readers read it."""
    texts = {f"f{index}": f"text {index} " * (index * 2000 + 1) for index in range(8)}
    inputs = _inputs(tmp_path, **texts)
    directory = tmp_path / "cache"
    start = threading.Barrier(12)
    failures: list[str] = []

    def write(index: int) -> None:
        start.wait()
        for round_ in range(20):
            items = [f"f{(index + round_ + step) % 8}" for step in range(index % 4 + 1)]
            counted = _Counted(inputs)
            got = ScanCache(directory, "code").memoised("s", items, counted.key, counted.compute)
            if got != {item: texts[item].upper() for item in items}:
                failures.append(f"writer {index} got a wrong value")

    def read() -> None:
        start.wait()
        for _ in range(40):
            counted = _Counted(inputs)
            got = ScanCache(directory, "code").memoised("s", ["f0", "f7"], counted.key, counted.compute)
            if got != {"f0": texts["f0"].upper(), "f7": texts["f7"].upper()}:
                failures.append("a reader got a wrong value")

    threads = [threading.Thread(target=write, args=(index,)) for index in range(8)]
    threads += [threading.Thread(target=read) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not failures, failures
    # one store, no temporary file left behind, and it reads back whole
    assert len(list(directory.glob("s-*.pickle"))) == 1
    assert not list(directory.glob(".s-*.tmp"))
    last = _Counted(inputs)
    ScanCache(directory, "code").memoised("s", ["f0"], last.key, last.compute)
    assert last.calls in ([], ["f0"])


def test_files_key_moves_with_a_name_or_a_byte_and_with_nothing_else(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, a="alpha", b="beta")
    original = files_key([inputs / "b", inputs / "a"], tmp_path)

    assert files_key([inputs / "a", inputs / "b"], tmp_path) == original, "the order files are named in is not content"
    (inputs / "a").write_text("alphA")
    assert files_key([inputs / "a", inputs / "b"], tmp_path) != original
    (inputs / "a").write_text("alpha")
    (inputs / "a").rename(inputs / "c")
    assert files_key([inputs / "c", inputs / "b"], tmp_path) != original


def test_the_gates_cache_directory_is_ignored_by_git() -> None:
    """a clean clone that ran the gates shows no new untracked file."""
    probe = (_REPO_ROOT / CACHE_DIRECTORY / "probe.pickle").relative_to(_REPO_ROOT).as_posix()
    ignored = subprocess.run(["git", "check-ignore", "--quiet", probe], cwd=_REPO_ROOT, check=False)

    assert ignored.returncode == 0, f"{probe} is not gitignored"


def test_the_gates_cache_is_keyed_by_their_code() -> None:
    cache = ScanCache.for_repo(_REPO_ROOT)

    assert cache.directory == _REPO_ROOT / CACHE_DIRECTORY
    assert cache.code == ScanCache.for_repo(_REPO_ROOT).code
