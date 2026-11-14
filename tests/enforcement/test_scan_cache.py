"""the scans' on-disk memo answers only for what it was computed from, and never fails a scan.

:mod:`tests.enforcement.scan_cache` keeps what the workspace scans computed across processes: xdist
workers, a pool's batches. A cache that answered for content it no longer has would let a gate pass
over code it never read. So these warm the cache over a planted repo, change it by every route a
gate's answer can move (a new, edited or deleted file, a ledger edit, a ruff config change, a new
``src`` root, a moved release tag), and hold the cached answer to a fresh scan's and to the
violation planted. They hold a computation that reads outside its declared inputs to being answered
and not stored, a damaged or unwritable store to a recompute, and concurrent writers to whole stores.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import types
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.enforcement.scan_cache import (
    CACHE_DIRECTORY,
    ScanCache,
    ScanCacheWarning,
    ScanInputs,
    code_key,
    files_key,
    module_names,
    private_binding_scan,
    surfaces_at,
    surfaces_now,
)
from threetears.enforcement.release import surface_at, surface_now
from threetears.enforcement.underscore_access import PrivateBindingScan, scan_private_bindings

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _write(root: Path, files: dict[str, str]) -> None:
    for relative, text in files.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(text)


class _Counted:
    """a computation over the files of one directory that records each item it was asked for."""

    def __init__(self, inputs: Path) -> None:
        self.inputs = inputs
        self.calls: list[str] = []

    def declared(self, item: str) -> ScanInputs:
        return ScanInputs(root=self.inputs, files=(self.inputs / item,))

    def compute(self, item: str) -> str:
        self.calls.append(item)
        return (self.inputs / item).read_text().upper()


def _inputs(tmp_path: Path, **files: str) -> Path:
    inputs = tmp_path / "inputs"
    _write(inputs, files)
    return inputs


# -- the store ---------------------------------------------------------------------------------


def test_a_second_process_takes_the_stored_value_and_computes_nothing(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, a="alpha", b="beta")
    first, second = _Counted(inputs), _Counted(inputs)

    got = ScanCache(tmp_path / "cache", "code").memoised("s", ["a", "b"], first.declared, first.compute)
    # a fresh instance reads the directory, as another worker or batch would
    again = ScanCache(tmp_path / "cache", "code").memoised("s", ["a", "b"], second.declared, second.compute)

    assert got == again == {"a": "ALPHA", "b": "BETA"}
    assert first.calls == ["a", "b"]
    assert second.calls == []


def test_an_edited_file_is_computed_again_and_only_it(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, a="alpha", b="beta")
    cache = ScanCache(tmp_path / "cache", "code")
    warm = _Counted(inputs)
    cache.memoised("s", ["a", "b"], warm.declared, warm.compute)

    (inputs / "a").write_text("gamma")
    again = _Counted(inputs)

    assert cache.memoised("s", ["a", "b"], again.declared, again.compute) == {"a": "GAMMA", "b": "BETA"}
    assert again.calls == ["a"]


def test_a_store_written_by_other_code_is_never_read(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, a="alpha")
    old = _Counted(inputs)
    ScanCache(tmp_path / "cache", "old code").memoised("s", ["a"], old.declared, lambda _: "STALE")
    fresh = _Counted(inputs)

    got = ScanCache(tmp_path / "cache", "new code").memoised("s", ["a"], fresh.declared, fresh.compute)

    assert got == {"a": "ALPHA"}
    assert fresh.calls == ["a"]


def test_the_code_key_moves_with_the_source_of_a_loaded_module(tmp_path: Path) -> None:
    source = tmp_path / "probe.py"
    source.write_text("ANSWER = 1\n")
    module = types.ModuleType("threetears.probe")
    module.__file__ = str(source)
    before = code_key([module], [], tmp_path)

    source.write_text("ANSWER = 2\n")

    assert code_key([module], [], tmp_path) != before


def test_the_gates_cache_is_keyed_by_the_code_loaded() -> None:
    cache = ScanCache.for_repo(_REPO_ROOT)

    assert cache.directory == _REPO_ROOT / CACHE_DIRECTORY
    assert cache.code == ScanCache.for_repo(_REPO_ROOT).code


def test_a_value_whose_input_moved_while_it_was_computed_is_not_stored(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, a="alpha")
    cache = ScanCache(tmp_path / "cache", "code")
    counted = _Counted(inputs)

    def edits_while_reading(item: str) -> str:
        value = counted.compute(item)
        (inputs / item).write_text("edited mid-scan")
        return value

    with pytest.warns(ScanCacheWarning, match="changed while it was computed"):
        assert cache.memoised("s", ["a"], counted.declared, edits_while_reading) == {"a": "ALPHA"}
    (inputs / "a").write_text("alpha")
    later = _Counted(inputs)
    cache.memoised("s", ["a"], later.declared, later.compute)

    assert later.calls == ["a"], "a value computed while its input moved was stored under the old content"


def test_a_truncated_or_corrupt_store_reads_as_a_miss(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, a="alpha")
    cache = ScanCache(tmp_path / "cache", "code")
    warm = _Counted(inputs)
    cache.memoised("s", ["a"], warm.declared, warm.compute)
    (store,) = (tmp_path / "cache").glob("s-*.pickle")
    whole = store.read_bytes()

    for damaged in (whole[: len(whole) // 2], whole[:-1] + bytes([whole[-1] ^ 0xFF]), b"", b"not a store"):
        store.write_bytes(damaged)
        counted = _Counted(inputs)
        assert cache.memoised("s", ["a"], counted.declared, counted.compute) == {"a": "ALPHA"}
        assert counted.calls == ["a"]
        # and the recompute replaced it with a whole store
        assert store.read_bytes() == whole


def test_a_directory_that_cannot_be_written_recomputes_and_warns_once(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, a="alpha")
    blocker = tmp_path / "a file"
    blocker.write_text("")
    # a cache directory below a regular file can be neither created nor read
    cache = ScanCache(blocker / "cache", "code")

    with pytest.warns(ScanCacheWarning, match="cannot be written") as warned:
        for _ in range(2):
            counted = _Counted(inputs)
            assert cache.memoised("s", ["a"], counted.declared, counted.compute) == {"a": "ALPHA"}
            assert counted.calls == ["a"]

    assert len(warned) == 1
    assert len(cache.unstored) == 2


def test_concurrent_writers_leave_a_whole_store_and_readers_see_only_whole_ones(tmp_path: Path) -> None:
    """eight writers, each with its own item set, race to replace one store while four readers read it."""
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
            got = ScanCache(directory, "code").memoised("s", items, counted.declared, counted.compute)
            if got != {item: texts[item].upper() for item in items}:
                failures.append(f"writer {index} got a wrong value")

    def read() -> None:
        start.wait()
        for _ in range(40):
            counted = _Counted(inputs)
            got = ScanCache(directory, "code").memoised("s", ["f0", "f7"], counted.declared, counted.compute)
            if got != {"f0": texts["f0"].upper(), "f7": texts["f7"].upper()}:
                failures.append("a reader got a wrong value")

    threads = [threading.Thread(target=write, args=(index,)) for index in range(8)]
    threads += [threading.Thread(target=read) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not failures, failures
    # one store, no temporary file left behind
    assert len(list(directory.glob("s-*.pickle"))) == 1
    assert not list(directory.glob(".s-*.tmp"))


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
    probe = (CACHE_DIRECTORY / "probe.pickle").as_posix()
    ignored = subprocess.run(["git", "check-ignore", "--quiet", probe], cwd=_REPO_ROOT, check=False)

    assert ignored.returncode == 0, f"{probe} is not gitignored"


# -- a computation that reads outside its key is answered and not stored -----------------------


def test_outside_its_inputs_only_the_interpreters_own_code_may_be_opened(tmp_path: Path) -> None:
    """the one allowance beyond the declared inputs: a module, or its bytecode, under the interpreter's prefix."""
    inputs = ScanInputs(root=tmp_path)
    accepted = [
        Path(sys.prefix, "lib", "python3.14", "site-packages", "x", "__pycache__", "m.cpython-314.pyc.12345"),
        Path(sys.base_prefix, "lib", "python3.14", "json", "decoder.py"),
    ]
    refused = [
        Path(sys.prefix, "pyvenv.cfg"),
        _REPO_ROOT / "packages" / "core" / "src" / "threetears" / "core" / "__pycache__" / "x.cpython-314.pyc",
        tmp_path / "__pycache__" / "m.cpython-314.pyc",
        tmp_path / "m.py",
    ]

    for path in accepted:
        assert inputs.outside([("open", (str(path), "rb", 0))]) == [], f"{path} is the interpreter's code"
    for path in refused:
        assert inputs.outside([("open", (str(path), "rb", 0))]) == [f"open {path}"], f"{path} is not"


_HOOKS_ADDED = """
import json, sys
added = []
sys.addaudithook(lambda event, args: added.append(event) if event == "sys.addaudithook" else None)
from pathlib import Path
from tests.enforcement.scan_cache import ScanCache, ScanInputs
directory, inputs = Path(sys.argv[1]), Path(sys.argv[2])
cache = ScanCache(directory, "code")
declared = lambda item: ScanInputs(root=inputs, files=(inputs / item,))
counts = {"imported": len(added)}
cache.memoised("s", ["a"], declared, lambda item: "stale")
counts["after a hit"] = len(added)
cache.memoised("s", ["a", "b"], declared, lambda item: item.upper())
counts["after a miss"] = len(added)
cache.memoised("s", ["a", "b", "c"], declared, lambda item: item.upper())
counts["after another miss"] = len(added)
print(json.dumps(counts))
"""


def test_the_audit_hook_is_added_on_the_first_computation_and_never_before(tmp_path: Path) -> None:
    """a process that imports the cache and only hits its store runs without the hook, and one that computes adds it once."""
    inputs = _inputs(tmp_path, a="alpha", b="beta", c="gamma")
    warm = _Counted(inputs)
    ScanCache(tmp_path / "cache", "code").memoised("s", ["a"], warm.declared, warm.compute)

    # a fresh interpreter: this one may have added the hook already, and a hook is never removed
    ran = subprocess.run(
        [sys.executable, "-c", _HOOKS_ADDED, str(tmp_path / "cache"), str(inputs)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    counts = json.loads(ran.stdout)

    assert counts["after a hit"] == counts["imported"], "importing the cache or reading its store added a hook"
    assert counts["after a miss"] == counts["imported"] + 1
    assert counts["after another miss"] == counts["imported"] + 1, "the hook was added more than once"


def _undeclared(tmp_path: Path, compute: Callable[[str], str], commands: frozenset[str] = frozenset()) -> ScanCache:
    """run *compute* twice through one cache with only ``inputs/a`` declared; return the cache."""
    inputs = _inputs(tmp_path, a="alpha")
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "data").write_text("unkeyed")
    cache = ScanCache(tmp_path / "cache", "code")

    def declared(item: str) -> ScanInputs:
        return ScanInputs(root=inputs, files=(inputs / item,), commands=commands)

    for _ in range(2):
        assert cache.memoised("s", ["a"], declared, compute) == {"a": "ALPHA"}
    return cache


def test_a_computation_that_opens_an_undeclared_file_is_not_stored(tmp_path: Path) -> None:
    calls: list[str] = []

    def reads_elsewhere(item: str) -> str:
        calls.append(item)
        (tmp_path / "elsewhere" / "data").read_text()
        return (tmp_path / "inputs" / item).read_text().upper()

    with pytest.warns(ScanCacheWarning, match="read outside its key: open"):
        cache = _undeclared(tmp_path, reads_elsewhere)

    assert calls == ["a", "a"], "a value read from an unkeyed file was served from the store"
    assert len(cache.unstored) == 2


def test_a_computation_that_lists_an_undeclared_directory_is_not_stored(tmp_path: Path) -> None:
    calls: list[str] = []

    def lists_elsewhere(item: str) -> str:
        calls.append(item)
        os.listdir(tmp_path / "elsewhere")
        list((tmp_path / "elsewhere").iterdir())
        return (tmp_path / "inputs" / item).read_text().upper()

    with pytest.warns(ScanCacheWarning, match="read outside its key: os.(listdir|scandir)"):
        _undeclared(tmp_path, lists_elsewhere)

    assert calls == ["a", "a"]


def test_a_computation_that_starts_an_undeclared_program_is_not_stored_and_a_declared_one_is(tmp_path: Path) -> None:
    calls: list[str] = []

    def runs_git(item: str) -> str:
        calls.append(item)
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        return "ALPHA"

    with pytest.warns(ScanCacheWarning, match="read outside its key: subprocess.Popen git"):
        _undeclared(tmp_path / "undeclared", runs_git)
    assert calls == ["a", "a"]

    calls.clear()
    _undeclared(tmp_path / "declared", runs_git, commands=frozenset({"git"}))
    assert calls == ["a"]


# -- the private-binding gate's key, against every route its answer moves by -------------------

_LEDGER_TEXT = (
    "# rationale: the planted confinement module reaches the vendor library's private query class\n"
    "packages/lib/src/lib/_vendor_internals.py:<module>#0:_internal\n"
)
_RUFF_IGNORE = '\n[tool.ruff.lint.per-file-ignores]\n"packages/lib/src/lib/_vendor_internals.py" = ["SLF001"]\n'

#: a repo with one recorded confinement module (its third-party private import is sanctioned) and no violation
_BINDING_REPO = {
    "pyproject.toml": '[project]\nname = "planted"\n' + _RUFF_IGNORE,
    "ledger.txt": _LEDGER_TEXT,
    "packages/lib/src/lib/__init__.py": "",
    "packages/lib/src/lib/_vendor_internals.py": "from vendor._internal import Query\n\nQUERY = Query\n",
    "packages/lib/src/lib/mod.py": "_hidden = 1\nvisible = 2\n",
    "packages/lib/tests/test_mod.py": "from lib.mod import visible\n\n\ndef test_it():\n    assert visible\n",
}


def _found(scan: PrivateBindingScan, repo: Path) -> set[tuple[str, str]]:
    return {(violation.file.relative_to(repo).as_posix(), violation.symbol) for violation in scan.violations}


_ROUTES: dict[str, tuple[Callable[[Path], None], set[tuple[str, str]]]] = {
    "a new file": (
        lambda repo: _write(repo, {"packages/lib/tests/test_new.py": "from lib.mod import _hidden\n"}),
        {("packages/lib/tests/test_new.py", "_hidden")},
    ),
    "an edited file": (
        lambda repo: _write(repo, {"packages/lib/tests/test_mod.py": "from lib.mod import _hidden, visible\n"}),
        {("packages/lib/tests/test_mod.py", "_hidden")},
    ),
    "a ledger edit": (
        lambda repo: _write(repo, {"ledger.txt": "# no entries\n"}),
        {("packages/lib/src/lib/_vendor_internals.py", "_internal")},
    ),
    "a ruff config change": (
        lambda repo: _write(repo, {"pyproject.toml": '[project]\nname = "planted"\n'}),
        {("packages/lib/src/lib/_vendor_internals.py", "_internal")},
    ),
    "a new src root": (
        lambda repo: _write(
            repo,
            {
                "packages/other/src/other/__init__.py": "",
                "packages/other/src/other/user.py": "from lib.mod import _hidden\n",
            },
        ),
        {("packages/other/src/other/user.py", "_hidden")},
    ),
}


@pytest.mark.parametrize("route", sorted(_ROUTES))
def test_a_warm_private_binding_cache_reports_a_violation_planted_by(route: str, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write(repo, _BINDING_REPO)
    cache = ScanCache(tmp_path / "cache", "code")
    warm = private_binding_scan(cache, repo, repo / "ledger.txt")
    assert _found(warm, repo) == set(), "the planted repo should start clean"
    assert private_binding_scan(cache, repo, repo / "ledger.txt") == warm
    assert not cache.unstored, f"the clean scan was not stored: {cache.unstored}"

    plant, expected = _ROUTES[route]
    plant(repo)
    cached = private_binding_scan(cache, repo, repo / "ledger.txt")

    assert _found(cached, repo) == expected
    assert cached == scan_private_bindings(repo, repo / "ledger.txt")
    assert not cache.unstored, f"a scan read outside its key: {cache.unstored}"


def test_a_warm_private_binding_cache_drops_the_violation_of_a_deleted_file(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write(repo, {**_BINDING_REPO, "packages/lib/tests/test_new.py": "from lib.mod import _hidden\n"})
    cache = ScanCache(tmp_path / "cache", "code")
    assert _found(private_binding_scan(cache, repo, repo / "ledger.txt"), repo) == {
        ("packages/lib/tests/test_new.py", "_hidden")
    }

    (repo / "packages/lib/tests/test_new.py").unlink()

    assert _found(private_binding_scan(cache, repo, repo / "ledger.txt"), repo) == set()


# -- the public-API gate's keys ----------------------------------------------------------------


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=planted",
            "-c",
            "user.email=planted@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "tag.gpgsign=false",
            *args,
        ],
        cwd=repo,
        capture_output=True,
        check=True,
    )


def test_a_warm_baseline_surface_follows_a_moved_tag(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write(repo, {"packages/p/src/p/__init__.py": "", "packages/p/src/p/m.py": "a = 1\n"})
    _git(repo, "init", "--quiet")
    _git(repo, "add", ".")
    _git(repo, "commit", "--quiet", "-m", "one")
    _git(repo, "tag", "v0.1.0")
    cache = ScanCache(tmp_path / "cache", "code")
    root = "packages/p/src"
    assert surfaces_at(cache, repo, "v0.1.0", [root])[root]["packages/p/src/p/m.py"] == {"a"}

    _write(repo, {"packages/p/src/p/m.py": "a = 1\nb = 2\n"})
    _git(repo, "commit", "--quiet", "-am", "two")
    _git(repo, "tag", "--force", "v0.1.0")

    assert surfaces_at(cache, repo, "v0.1.0", [root])[root] == surface_at(repo, "v0.1.0", root).modules
    assert surfaces_at(cache, repo, "v0.1.0", [root])[root]["packages/p/src/p/m.py"] == {"a", "b"}
    assert not cache.unstored, f"a scan read outside its key: {cache.unstored}"


def test_a_warm_working_tree_surface_follows_new_edited_and_deleted_modules(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    root = "packages/p/src"
    _write(repo, {"packages/p/src/p/__init__.py": "", "packages/p/src/p/m.py": "a = 1\n"})
    cache = ScanCache(tmp_path / "cache", "code")
    surfaces_now(cache, repo, [root])

    for change in (
        {"packages/p/src/p/n.py": "fresh = 1\n"},
        {"packages/p/src/p/m.py": "a = 1\nb = 2\n"},
    ):
        _write(repo, change)
        assert surfaces_now(cache, repo, [root])[root] == surface_now(repo, root).modules
    (repo / "packages/p/src/p/m.py").unlink()
    surface = surfaces_now(cache, repo, [root])[root]

    assert surface == surface_now(repo, root).modules
    assert "packages/p/src/p/m.py" not in surface
    assert surface["packages/p/src/p/n.py"] == {"fresh"}
    assert not cache.unstored, f"a scan read outside its key: {cache.unstored}"


def test_a_warm_module_names_cache_follows_an_edit(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write(repo, {"m.py": "from x import used_before\n"})
    cache = ScanCache(tmp_path / "cache", "code")
    assert "used_before" in module_names(cache, repo, [("m.py", True)])[("m.py", True)][0]

    _write(repo, {"m.py": "from x import used_after\n"})
    uses, _ = module_names(cache, repo, [("m.py", True)])[("m.py", True)]

    assert "used_after" in uses
    assert "used_before" not in uses
    assert not cache.unstored, f"a scan read outside its key: {cache.unstored}"
