"""the gates' scan keys answer only for what their scans read, against every route an answer moves by.

The cache under them is held in ``packages/enforcement/tests/common/test_scan_cache.py``. Here: the
gates' cache is keyed by the scanners' code whatever else a session imported, its directory is
gitignored, and the private-binding scan, the public API's surfaces and the module names it reads,
warmed over a planted repo and changed by every route their answer can move (a new, edited or
deleted file, a ledger edit, a ruff config change, a new ``src`` root, a moved release tag), still
match a fresh scan and the violation planted.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.enforcement.scan_cache import (
    SCANNER_MODULES,
    gate_scan_cache,
    module_names,
    private_binding_scan,
    surfaces_at,
    surfaces_now,
)
from threetears.enforcement.common.scan_cache import (
    CACHE_DIRECTORY,
    ScanCache,
    installed_module_file,
    module_closure,
)
from threetears.enforcement.release import surface_at, surface_now
from threetears.enforcement.underscore_access import PrivateBindingScan, scan_private_bindings

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _write(root: Path, files: dict[str, str]) -> None:
    for relative, text in files.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(text)


_CODE_KEY = """
import importlib, importlib.util, json, sys
from pathlib import Path
extra = []
if sys.argv[2] == "extras":
    for name in sys.argv[3:]:
        if importlib.util.find_spec(name) is not None:
            importlib.import_module(name)
            extra.append(name)
from tests.enforcement.scan_cache import gate_scan_cache
print(json.dumps({"code": gate_scan_cache(Path(sys.argv[1])).code, "extra": extra}))
"""


def test_two_processes_that_imported_different_modules_compute_one_code_key() -> None:
    """the key is the scanners' code, not whatever else the session collected first."""
    closure = module_closure(SCANNER_MODULES, installed_module_file)
    unrelated = [
        name
        for name in ("threetears.geo", "threetears.agent.memory", "threetears.evals", "threetears.models")
        if name not in closure
    ]

    def code(*argv: str) -> dict[str, object]:
        ran = subprocess.run(
            [sys.executable, "-c", _CODE_KEY, str(_REPO_ROOT), *argv],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        loaded: dict[str, object] = json.loads(ran.stdout)
        return loaded

    plain, loaded = code("plain"), code("extras", *unrelated)

    assert loaded["extra"], "no unrelated threetears package was importable; the comparison proves nothing"
    assert plain["code"] == loaded["code"]
    assert gate_scan_cache(_REPO_ROOT).code == plain["code"]
    assert gate_scan_cache(_REPO_ROOT).directory == _REPO_ROOT / CACHE_DIRECTORY


def test_the_gates_cache_directory_is_ignored_by_git() -> None:
    """a clean clone that ran the gates shows no new untracked file."""
    probe = (CACHE_DIRECTORY / "probe.pickle").as_posix()
    ignored = subprocess.run(["git", "check-ignore", "--quiet", probe], cwd=_REPO_ROOT, check=False)

    assert ignored.returncode == 0, f"{probe} is not gitignored"


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
