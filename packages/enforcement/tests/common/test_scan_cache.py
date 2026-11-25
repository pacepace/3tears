"""the scans' on-disk memo answers only for what it was computed from, and never fails a scan.

:mod:`threetears.enforcement.common.scan_cache` keeps what the enforcement scans computed across
processes: xdist workers, a pool's batches. A cache that answered for content it no longer has would
let a gate pass over code it never read. So these hold a value to its inputs' bytes and to the code
that computed it, a computation that reads outside its declared inputs to being answered and not
stored, a damaged or unwritable store to a recompute, and concurrent writers to whole stores. The
gates' own keys, against every route their answers move by, are held in
``tests/enforcement/test_scan_cache.py``.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
import warnings
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from threetears.enforcement.common import parse_python_path
from threetears.enforcement.common.scan_cache import (
    CACHE_DIRECTORY,
    ImportIndex,
    ScanCache,
    ScanInputs,
    code_key,
    file_facts,
    module_closure,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]


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


def test_the_code_key_follows_the_scanners_import_closure_and_nothing_else(tmp_path: Path) -> None:
    """a dependency's edit moves the key; a module the scanner never imports does not."""
    _write(
        tmp_path,
        {
            "threetears/scanner/__init__.py": "from threetears.scanner.walk import walk\n",
            "threetears/scanner/walk.py": "from threetears.dep import util\nfrom .rules import RULES\n",
            "threetears/scanner/rules.py": "RULES = ()\n",
            "threetears/dep/__init__.py": "",
            "threetears/dep/util.py": "def helper():\n    return 1\n",
            "threetears/unrelated/__init__.py": "X = 1\n",
        },
    )

    def locate(name: str) -> Path | None:
        base = tmp_path.joinpath(*name.split("."))
        return next((path for path in (base / "__init__.py", base.with_suffix(".py")) if path.is_file()), None)

    def key() -> str:
        return code_key(module_closure(["threetears.scanner.walk"], locate), [], tmp_path)

    assert sorted(module_closure(["threetears.scanner.walk"], locate)) == [
        "threetears.dep",
        "threetears.dep.util",
        "threetears.scanner",
        "threetears.scanner.rules",
        "threetears.scanner.walk",
    ]
    before = key()
    _write(tmp_path, {"threetears/unrelated/__init__.py": "X = 2\n"})
    assert key() == before, "a module outside the closure moved the key"
    _write(tmp_path, {"threetears/dep/util.py": "def helper():\n    return 2\n"})
    assert key() != before, "an edit to a dependency of the scanner left the key"


def test_a_value_whose_input_moved_while_it_was_computed_is_not_stored(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, a="alpha")
    cache = ScanCache(tmp_path / "cache", "code")
    counted = _Counted(inputs)

    def edits_while_reading(item: str) -> str:
        value = counted.compute(item)
        (inputs / item).write_text("edited mid-scan")
        return value

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a file edited mid-scan is the environment's doing: never a warning
        assert cache.memoised("s", ["a"], counted.declared, edits_while_reading) == {"a": "ALPHA"}
    assert [reason for reason in cache.unstored if "changed while it was computed" in reason]
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


def test_a_directory_that_cannot_be_written_recomputes_and_never_warns(tmp_path: Path) -> None:
    """a read-only checkout runs its gates uncached, and a run that treats warnings as errors still passes."""
    inputs = _inputs(tmp_path, a="alpha")
    blocker = tmp_path / "a file"
    blocker.write_text("")
    # a cache directory below a regular file can be neither created nor read
    cache = ScanCache(blocker / "cache", "code")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for _ in range(2):
            counted = _Counted(inputs)
            assert cache.memoised("s", ["a"], counted.declared, counted.compute) == {"a": "ALPHA"}
            assert counted.calls == ["a"]

    assert len(cache.unstored) == 2
    assert all("cannot be written" in reason for reason in cache.unstored)
    assert not cache.defects


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


def test_an_inputs_key_moves_with_a_files_name_or_byte_and_with_nothing_else(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, a="alpha", b="beta")

    def key(*names: str) -> str:
        return ScanInputs(root=tmp_path, files=tuple(inputs / name for name in names)).key()

    original = key("b", "a")
    assert key("a", "b") == original, "the order files are named in is not content"
    (inputs / "a").write_text("alphA")
    assert key("a", "b") != original
    (inputs / "a").write_text("alpha")
    (inputs / "a").rename(inputs / "c")
    assert key("c", "b") != original


def test_a_store_leaves_any_checkout_clean_without_its_gitignore_naming_it(tmp_path: Path) -> None:
    """a consumer repo whose gates cache under its root sees no untracked file, though its .gitignore says nothing."""
    repo = tmp_path / "consumer"
    _write(repo, {"a": "alpha", ".gitignore": "*.log\n"})
    subprocess.run(["git", "init", "--quiet"], cwd=repo, check=True)
    counted = _Counted(repo)

    ScanCache(repo / ".cache" / "enforcement-scans", "code").memoised("s", ["a"], counted.declared, counted.compute)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=repo, capture_output=True, text=True, check=True
    )

    assert counted.calls == ["a"]
    assert list((repo / ".cache" / "enforcement-scans").glob("s-*.pickle")), "nothing was stored"
    assert sorted(status.stdout.splitlines()) == ["?? .gitignore", "?? a"], status.stdout


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
from threetears.enforcement.common.scan_cache import ScanCache, ScanInputs
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

    with pytest.warns(UserWarning, match="enforcement scan cache: .*read outside its key: open"):
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

    with pytest.warns(UserWarning, match="enforcement scan cache: .*read outside its key: os.(listdir|scandir)"):
        _undeclared(tmp_path, lists_elsewhere)

    assert calls == ["a", "a"]


def test_a_module_imported_for_the_first_time_while_computing_does_not_refuse_the_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """the import system lists ``sys.path`` and opens the module; that is the interpreter loading code, not a read.

    a scan whose code imports a module lazily read outside its key only in a process that had not imported it yet,
    which is every fresh process: CI's unit suite on a cold runner refused a warm-cache test the pool had passed.
    """
    modules = tmp_path / "modules"
    modules.mkdir()
    (modules / "first_import_only_here.py").write_text("VALUE = 'loaded'\n")
    monkeypatch.syspath_prepend(str(modules))
    monkeypatch.delitem(sys.modules, "first_import_only_here", raising=False)

    def imports_lazily(item: str) -> str:
        import first_import_only_here  # noqa: PLC0415  (the first import is the point)

        assert first_import_only_here.VALUE == "loaded"
        return (tmp_path / "inputs" / item).read_text().upper()

    cache = _undeclared(tmp_path, imports_lazily)

    assert not cache.unstored, f"a first import refused the scan: {cache.unstored}"


def test_a_computation_that_starts_an_undeclared_program_is_not_stored_and_a_declared_one_is(tmp_path: Path) -> None:
    calls: list[str] = []

    def runs_git(item: str) -> str:
        calls.append(item)
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        return "ALPHA"

    with pytest.warns(UserWarning, match="enforcement scan cache: .*read outside its key: subprocess.Popen git"):
        _undeclared(tmp_path / "undeclared", runs_git)
    assert calls == ["a", "a"]

    calls.clear()
    _undeclared(tmp_path / "declared", runs_git, commands=frozenset({"git"}))
    assert calls == ["a"]


# -- the per-file walk --------------------------------------------------------------------------


class _Names:
    """an extractor that names a module's top-level functions, and records each file it was asked about."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, path: Path, tree: ast.Module) -> tuple[str, ...]:
        self.calls.append(path.name)
        return tuple(node.name for node in tree.body if isinstance(node, ast.FunctionDef))


def _modules(tmp_path: Path, **files: str) -> tuple[Path, list[Path]]:
    root = tmp_path / "repo"
    _write(root, {f"{name}.py": text for name, text in files.items()})
    return root, [root / f"{name}.py" for name in files]


def test_a_second_walk_parses_nothing_and_answers_the_same(tmp_path: Path) -> None:
    root, paths = _modules(tmp_path, a="def f(): pass\n", b="def g(): pass\ndef h(): pass\n")
    first, second = _Names(), _Names()

    got = ScanCache(tmp_path / "cache", "code").per_file("facts", first, paths, root=root)
    again = ScanCache(tmp_path / "cache", "code").per_file("facts", second, paths, root=root)

    assert got == again == {paths[0]: ("f",), paths[1]: ("g", "h")}
    assert list(got) == paths, "the facts come back in the order the files were given"
    assert first.calls == ["a.py", "b.py"]
    assert second.calls == []


def test_an_edited_or_renamed_file_is_parsed_again_and_only_it(tmp_path: Path) -> None:
    root, (a, b) = _modules(tmp_path, a="def f(): pass\n", b="def g(): pass\n")
    cache = ScanCache(tmp_path / "cache", "code")
    cache.per_file("facts", _Names(), [a, b], root=root)

    a.write_text("def f2(): pass\n")
    edited = _Names()
    assert cache.per_file("facts", edited, [a, b], root=root) == {a: ("f2",), b: ("g",)}
    assert edited.calls == ["a.py"]

    c = b.rename(root / "c.py")
    renamed = _Names()
    assert cache.per_file("facts", renamed, [a, c], root=root) == {a: ("f2",), c: ("g",)}
    assert renamed.calls == ["c.py"], "a fact is keyed by the file's name as well as its bytes"


def test_walks_over_different_files_share_one_store(tmp_path: Path) -> None:
    """a gate that walks the source trees and one that walks the tests each keep the other's facts."""
    root, (a, b) = _modules(tmp_path, a="def f(): pass\n", b="def g(): pass\n")
    cache = ScanCache(tmp_path / "cache", "code")
    cache.per_file("facts", _Names(), [a], root=root)
    cache.per_file("facts", _Names(), [b], root=root)

    both = _Names()
    cache.per_file("facts", both, [a, b], root=root)

    assert both.calls == []


def test_a_file_rewritten_while_it_is_extracted_is_answered_for_the_bytes_it_was_keyed_by(tmp_path: Path) -> None:
    root, (a,) = _modules(tmp_path, a="def before(): pass\n")
    cache = ScanCache(tmp_path / "cache", "code")
    names = _Names()

    def rewrites(path: Path, tree: ast.Module) -> tuple[str, ...]:
        path.write_text("def after(): pass\n")
        return names(path, tree)

    assert cache.per_file("facts", rewrites, [a], root=root) == {a: ("before",)}
    later = _Names()

    assert cache.per_file("facts", later, [a], root=root) == {a: ("after",)}
    assert later.calls == ["a.py"], "a fact of the old bytes was served for the new ones"


@pytest.mark.parametrize(
    ("source", "error", "reason"),
    [
        (b"def broken(:\n", SyntaxError, "syntax error at line 1"),
        (b"x = '\xff'\n", UnicodeDecodeError, "not valid utf-8"),
    ],
)
def test_an_unparsable_file_raises_or_is_noted_on_every_call_warm_or_cold(
    source: bytes, error: type[Exception], reason: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, (good,) = _modules(tmp_path, good="def f(): pass\n")
    bad = root / "bad.py"
    bad.write_bytes(source)
    cache = ScanCache(tmp_path / "cache", "code")
    noted: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "threetears.enforcement.common.scan_cache.note_unscanned",
        lambda target, why: noted.append((Path(target).name, why)),
    )

    for _ in range(2):
        with pytest.raises(error):
            cache.per_file("strict", _Names(), [good, bad], root=root)
    for _ in range(2):
        assert cache.per_file("lenient", _Names(), [good, bad], root=root, unparsable="note") == {good: ("f",)}

    assert noted == [("bad.py", reason), ("bad.py", reason)], "a warm walk skipped a file without saying so"
    with pytest.raises(error):
        parse_python_path(bad)  # the error parse_python_path raises, so a converted gate fails as it did


def test_a_file_is_read_as_parse_python_path_reads_it(tmp_path: Path) -> None:
    """carriage returns become newlines, so a gate converted to the cache sees the string constants it saw before."""
    root = tmp_path / "repo"
    root.mkdir()
    path = root / "crlf.py"
    path.write_bytes(b'TEXT = """one\r\ntwo\rthree"""\r\n')

    def constant(_: Path, tree: ast.Module) -> object:
        return next(node.value for node in ast.walk(tree) if isinstance(node, ast.Constant))

    assert ScanCache(tmp_path / "cache", "code").per_file("facts", constant, [path], root=root) == {
        path: constant(path, parse_python_path(path))
    }


def test_an_extractor_that_reads_another_file_is_answered_and_not_stored(tmp_path: Path) -> None:
    root, (a,) = _modules(tmp_path, a="def f(): pass\n")
    (root / "ledger.txt").write_text("unkeyed")
    calls: list[str] = []

    def reads_a_ledger(path: Path, tree: ast.Module) -> str:
        calls.append(path.name)
        return (root / "ledger.txt").read_text()

    cache = ScanCache(tmp_path / "cache", "code")
    with pytest.warns(UserWarning, match="enforcement scan cache: .*read outside its key: open"):
        for _ in range(2):
            assert cache.per_file("facts", reads_a_ledger, [a], root=root) == {a: "unkeyed"}

    assert calls == ["a.py", "a.py"]


def test_a_path_outside_the_root_is_refused(tmp_path: Path) -> None:
    root, _ = _modules(tmp_path, a="def f(): pass\n")
    outside = tmp_path / "elsewhere.py"
    outside.write_text("")

    with pytest.raises(ValueError, match="elsewhere.py"):
        ScanCache(tmp_path / "cache", "code").per_file("facts", _Names(), [outside], root=root)


def test_a_per_file_walk_inside_a_scan_leaves_the_scan_storable_and_its_stray_read_spoils_both(tmp_path: Path) -> None:
    """a whole-workspace scan whose scanner walks through the per-file cache: the walk's own store and code are
    not the scan's inputs, but a file the walk's extractor read outside its own file is."""
    root, paths = _modules(tmp_path, a="def f(): pass\n", b="def g(): pass\n")
    (root / "ledger.txt").write_text("unkeyed")
    outer = ScanCache(tmp_path / "outer", "code")
    walk = ScanCache(tmp_path / "walk", "code")

    def declared(_: str) -> ScanInputs:
        return ScanInputs(root=root, files=tuple(paths))

    def scan(_: str) -> dict[Path, tuple[str, ...]]:
        return walk.per_file("facts", _Names(), paths, root=root)

    for _ in range(2):
        assert outer.memoised("scan", ["workspace"], declared, scan) == {
            "workspace": {paths[0]: ("f",), paths[1]: ("g",)}
        }
    assert not outer.unstored, f"the walk's own store or code refused the scan: {outer.unstored}"
    assert not walk.unstored

    def strays(path: Path, tree: ast.Module) -> str:
        return (root / "ledger.txt").read_text()

    def stray_scan(_: str) -> dict[Path, str]:
        return walk.per_file("strays", strays, paths, root=root)

    with pytest.warns(UserWarning, match="enforcement scan cache: .*read outside its key"):
        outer.memoised("stray scan", ["workspace"], declared, stray_scan)
    assert any(reason.startswith("stray scan") for reason in outer.unstored), (
        "the scan kept a value its walk read stray"
    )
    assert walk.unstored, "the walk kept a fact read stray"


_EXTRACTS_AT_IMPORT = """
from pathlib import Path
from threetears.enforcement.common.scan_cache import ScanCache

ROOT = Path(__file__).parent / "repo"


def strays(path, tree):
    return (ROOT / "ledger.txt").read_text()


CACHE = ScanCache(Path(__file__).parent / "cache", "code")
FACTS = CACHE.per_file("strays", strays, [ROOT / "a.py"], root=ROOT)
"""


def test_an_extraction_run_while_its_module_imports_is_still_audited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """a gate that builds its table when pytest imports it has the import system below it on the stack; its own
    stray read is still its own, not the interpreter loading code."""
    _write(tmp_path, {"repo/a.py": "def f(): pass\n", "repo/ledger.txt": "unkeyed", "gate.py": _EXTRACTS_AT_IMPORT})

    with pytest.warns(UserWarning, match="enforcement scan cache: .*read outside its key: open"):
        module = _load("extracts_at_import", tmp_path / "gate.py", monkeypatch)

    assert module.FACTS == {tmp_path / "repo" / "a.py": "unkeyed"}
    assert module.CACHE.unstored, "a fact read stray during an import was stored"


#: the files :func:`_top_level_functions` was asked about, so a test can see what it parsed
_EXTRACTED: list[str] = []


def _top_level_functions(path: Path, tree: ast.Module) -> tuple[str, ...]:
    _EXTRACTED.append(path.name)
    return tuple(node.name for node in tree.body if isinstance(node, ast.FunctionDef))


def test_file_facts_keeps_an_extractors_facts_under_its_own_code(tmp_path: Path) -> None:
    root, paths = _modules(tmp_path, a="def f(): pass\n", b="def g(): pass\n")
    _EXTRACTED.clear()

    got = file_facts(_top_level_functions, paths, root=root)
    again = file_facts(_top_level_functions, paths, root=root)

    assert got == again == {paths[0]: ("f",), paths[1]: ("g",)}
    assert _EXTRACTED == ["a.py", "b.py"]
    (store,) = (root / CACHE_DIRECTORY).glob("facts-*.pickle")
    assert "_top_level_functions" in store.name, "the store is not named for its extractor"
    assert re.fullmatch(r"facts-.*_top_level_functions-[0-9a-f]{24}\.pickle", store.name), (
        "the store is not keyed by its code"
    )


# -- the code a fact comes from -----------------------------------------------------------------


def test_a_closure_resolves_relative_imports_against_the_package(tmp_path: Path) -> None:
    """``from . import x`` in a module is its package's sibling; in a package's ``__init__`` it is the package's child."""
    _write(
        tmp_path,
        {
            "threetears/pkg/__init__.py": "",
            "threetears/pkg/up.py": "",
            "threetears/pkg/sub/__init__.py": "from . import child\n",
            "threetears/pkg/sub/child.py": "",
            "threetears/pkg/sub/mod.py": "from . import sibling\nfrom ..up import thing\n",
            "threetears/pkg/sub/sibling.py": "",
            "threetears/pkg/sub/unreached.py": "",
        },
    )

    def locate(name: str) -> Path | None:
        base = tmp_path.joinpath(*name.split("."))
        return next((path for path in (base / "__init__.py", base.with_suffix(".py")) if path.is_file()), None)

    assert sorted(module_closure(["threetears.pkg.sub.mod"], locate)) == [
        "threetears.pkg",
        "threetears.pkg.sub",
        "threetears.pkg.sub.child",
        "threetears.pkg.sub.mod",
        "threetears.pkg.sub.sibling",
        "threetears.pkg.up",
    ]


def test_an_import_index_parses_each_module_once_across_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, {"threetears/one.py": "from threetears import two\n", "threetears/two.py": "X = 1\n"})
    parsed: list[str] = []
    real = ast.parse

    def counting(source: str | bytes, filename: str = "<unknown>", *args: object, **kwargs: object) -> ast.AST:
        if Path(filename).is_relative_to(tmp_path):
            parsed.append("threetears." + Path(filename).stem)
        return real(source, filename, *args, **kwargs)  # type: ignore[call-overload]

    monkeypatch.setattr(ast, "parse", counting)

    def locate(name: str) -> Path | None:
        found = tmp_path.joinpath(*name.split(".")).with_suffix(".py")
        return found if found.is_file() else None

    first = ImportIndex(tmp_path / "cache")
    cold = module_closure(["threetears.one"], locate, first)
    first.save()
    # a fresh index reads the store, as the next process would
    warm = module_closure(["threetears.one"], locate, ImportIndex(tmp_path / "cache"))

    assert (
        cold
        == warm
        == {"threetears.one": tmp_path / "threetears/one.py", "threetears.two": tmp_path / "threetears/two.py"}
    )
    assert sorted(parsed) == ["threetears.one", "threetears.two"], "a warm index parsed a module again"

    _write(tmp_path, {"threetears/two.py": "X = 2\n"})
    edited = ImportIndex(tmp_path / "cache")
    module_closure(["threetears.one"], locate, edited)
    assert sorted(parsed) == ["threetears.one", "threetears.two", "threetears.two"], "an edit was not indexed again"


_GATE = """
from pathlib import Path
import ast
from helpers.marks import marks_of
import threetears.enforcement.common.ast_helpers

def extract(path: Path, tree: ast.Module) -> object:
    return marks_of(tree)
"""

_LOADED_BY_PATH = """
from pathlib import Path
import ast
import importlib.util
spec = importlib.util.spec_from_file_location("loaded_reader", Path(__file__).parent / "scripts" / "reader.py")
READER = importlib.util.module_from_spec(spec)
spec.loader.exec_module(READER)

def extract(path: Path, tree: ast.Module) -> object:
    return READER.read(tree)
"""


def _load(name: str, path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_an_extractors_code_key_follows_what_it_imports_and_refers_to_and_nothing_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """an edit to the gate, a helper it imports or one it loads by path re-keys its facts; an unrelated file does not.

    Each look loads the gate under a fresh name, as a fresh process would meet it, and reads the code key off the
    name of the store its facts land in.
    """
    _write(
        tmp_path,
        {
            "uv.lock": "version = 1\n",
            "gate_by_import.py": _GATE,
            "helpers/__init__.py": "",
            "helpers/marks.py": "def marks_of(tree):\n    return 1\n",
            "gate_by_path.py": _LOADED_BY_PATH,
            "scripts/reader.py": "def read(tree):\n    return 1\n",
            "unrelated.py": "X = 1\n",
            "data/m.py": "x = 1\n",
        },
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in ("helpers", "helpers.marks"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    loads = iter(range(1_000))

    def code_of(gate: str) -> str:
        name = f"{gate}_{next(loads)}"
        extract = _load(name, tmp_path / f"{gate}.py", monkeypatch).extract
        file_facts(extract, [tmp_path / "data" / "m.py"], root=tmp_path / "data", directory=tmp_path / "cache")
        (store,) = (tmp_path / "cache").glob(f"facts-{name}.extract-*.pickle")
        return store.name.removesuffix(".pickle").rsplit("-", 1)[1]

    def keys() -> tuple[str, str]:
        return code_of("gate_by_import"), code_of("gate_by_path")

    before = keys()
    assert before == keys(), "the key is not stable"
    _write(tmp_path, {"unrelated.py": "X = 2\n"})
    assert keys() == before, "a file nothing refers to moved a key"

    for edit, moved in (
        ({"helpers/marks.py": "def marks_of(tree):\n    return 2\n"}, (True, False)),
        ({"scripts/reader.py": "def read(tree):\n    return 2\n"}, (False, True)),
        ({"gate_by_import.py": _GATE + "\n# edited\n"}, (True, False)),
        ({"uv.lock": "version = 2\n"}, (True, True)),
    ):
        previous = keys()
        _write(tmp_path, edit)
        now = keys()
        assert (now[0] != previous[0], now[1] != previous[1]) == moved, f"{sorted(edit)}: {moved} expected"
