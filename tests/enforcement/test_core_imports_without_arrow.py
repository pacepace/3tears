"""enforcement: 3tears core imports without pyarrow and NumPy, except the modules listed here as Arrow's.

**The ruling** (Pace 2026-11-15): pyarrow stays an optional extra of core (``3tears[arrow]``, or ``[snapshot]`` with
DuckDB). 26dabec04 made it a base dependency by accident: ``core.collections.change_notes`` imported the restore's
Arrow fold (``core.change_note_fold``) at module top, so every process following the epoch log needed pyarrow, and the
SDK's unit suite failed 111 tests on ``ModuleNotFoundError: pyarrow``. The keeper, the one importer of the fold, now
lives in ``core.collections.change_note_keeper``; this gate keeps the rest of core free of Arrow.

**How.** ``uv export --frozen`` names, from ``uv.lock``, the distributions core and the epoch package install with
every core extra but the Arrow ones. A fresh venv (stdlib ``venv``, no pip) gets exactly those: each one's installed
files (its ``RECORD``) linked from this environment, which ``uv sync --frozen`` built from the same lock, so the
versions are the lock's (checked). A child interpreter there imports every module of ``threetears.core`` and
``threetears.epoch`` (the epoch follow, the change notes' writers and readers among them) and reports each that fails.
The venv must hold neither pyarrow nor NumPy (else the check proves nothing), and exactly the modules in
:data:`_ARROW_MODULES` may fail, each on pyarrow or NumPy: one listed that imports cleanly is stale and goes.

**Why a venv, not a blocked import in this interpreter:** it holds only what the lock gives core without the extras,
so a module reaching pyarrow through a distribution core does not declare fails too. **Why linked, not ``uv pip
install``:** the pool's runners have no package cache to install from offline (measured 2026-11-15: ``uv pip install
--offline`` found none of the lock's wheels), and a gate that downloads would fail with the network, not the code.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import textwrap
import tomllib
import venv
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import TypedDict, cast

import pytest
from packaging.markers import Marker

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: the Arrow extras, which the venv does not get
_ARROW_EXTRAS = frozenset({"arrow", "snapshot"})

#: what the Arrow extras bring: a module failing on another name is a real failure
_ARROW_DISTRIBUTIONS = frozenset({"pyarrow", "numpy"})

#: the modules that import pyarrow or NumPy at module top, each with its reason. Everything else in core and epoch
#: imports without them.
_ARROW_MODULES: dict[str, str] = {
    "threetears.core.change_note_fold": (
        "a restore's serving side: hands its note pages to a worker as Arrow tables (shared_arrow), never pickled"
    ),
    "threetears.core.change_note_fold_job": "a restore's worker side: parses the notes' CSV and folds them in Arrow",
    "threetears.core.collections.change_note_keeper": (
        "the change-note keeper: its restore folds through change_note_fold; run by the hub alone"
    ),
    "threetears.core.testing.note_ring": "test support installing the ring through ChangeNoteKeeper.install",
    "threetears.core.arrow_keys": (
        "Arrow rows matched by a key in Arrow's hash kernels (never Acero): Arrow tables in, NumPy row numbers out"
    ),
    "threetears.core.collections.copy_parts": (
        "a copied table's parts: its rows split by its scope column in Arrow, its locator an Arrow table"
    ),
    "threetears.core.collections.shared_copy": (
        "the copy's apply in its worker process: a change's parts rewritten in Arrow and published as shared Arrow"
    ),
}

#: a requirement line of ``uv export``: ``name==version`` (with a marker), or ``-e ./packages/<dir>``
_PINNED = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+)==(?P<version>[^ ;]+)(?: ; (?P<marker>.+))?$")
_EDITABLE = re.compile(r"^-e \./(?P<path>\S+)$")

#: the child's probe: import every module under the roots, report the failures and what the venv holds
_PROBE = textwrap.dedent(
    """
    import importlib
    import importlib.util
    import json
    import pkgutil
    import sys

    failed = {}
    imported = []
    for root_name in sys.argv[1:]:
        root = importlib.import_module(root_name)
        names = [root_name] + [
            info.name
            for info in pkgutil.walk_packages(root.__path__, root_name + ".", onerror=lambda name: None)
        ]
        for name in names:
            try:
                importlib.import_module(name)
            except ImportError as exc:
                failed[name] = getattr(exc, "name", None) or repr(exc)
            else:
                imported.append(name)
    print(json.dumps({
        "failed": failed,
        "imported": imported,
        "has": {each: importlib.util.find_spec(each) is not None for each in ("pyarrow", "numpy")},
    }))
    """
)

#: the child's purelib
_PATHS = "import json, sys, sysconfig; print(json.dumps(sysconfig.get_paths()['purelib']))"


class _Probe(TypedDict):
    """the child's report."""

    #: each module that failed to import, to the module its ImportError names
    failed: dict[str, str]
    #: each module imported
    imported: list[str]
    #: whether pyarrow / NumPy are installed in the venv
    has: dict[str, bool]


def _core_extras() -> list[str]:
    """every extra core declares but the Arrow ones."""
    declared = tomllib.loads((_REPO_ROOT / "packages/core/pyproject.toml").read_text(encoding="utf-8"))
    return sorted(set(declared["project"]["optional-dependencies"]) - _ARROW_EXTRAS)


def _run(*argv: str, cwd: Path = _REPO_ROOT) -> str:
    done = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, check=False)
    assert done.returncode == 0, f"{' '.join(argv)} failed ({done.returncode}):\n{done.stdout}\n{done.stderr}"
    return done.stdout


def _locked() -> dict[str, str | None]:
    """each distribution the lock installs for core + epoch without the Arrow extras, to its pinned version (``None``
    for a workspace member, installed editable). A line whose marker does not hold here is left out."""
    uv = shutil.which("uv")
    assert uv is not None, "uv is not on PATH: the gate reads the lock with it"
    extras = [arg for extra in _core_extras() for arg in ("--extra", extra)]
    text = _run(
        uv, "export", "--frozen", "--package", "3tears", "--package", "3tears-epoch", "--no-dev", "--no-hashes", *extras
    )
    locked: dict[str, str | None] = {}
    for line in text.splitlines():
        line = line.strip()
        if editable := _EDITABLE.match(line):
            project = tomllib.loads((_REPO_ROOT / editable["path"] / "pyproject.toml").read_text(encoding="utf-8"))
            locked[project["project"]["name"]] = None
        elif pinned := _PINNED.match(line):
            if pinned["marker"] and not Marker(pinned["marker"]).evaluate():
                continue
            locked[pinned["name"]] = pinned["version"]
    return locked


def _link(name: str, version: str | None, site: Path) -> None:
    """``name``'s installed files, linked into ``site`` as installing it would place them."""
    try:
        installed = distribution(name)
    except PackageNotFoundError:
        pytest.fail(f"{name} is in the lock's list but not installed here: run uv sync --frozen")
    if version is not None:
        assert installed.version == version, f"{name} is {installed.version} here, the lock pins {version}"
    files = installed.files
    assert files, f"{name} names no installed files (no RECORD)"
    for each in files:
        if each.parts[0] == ".." or "__pycache__" in each.parts:
            continue
        target = site / each
        source = Path(str(installed.locate_file(each)))
        if not source.exists() or target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source)


@pytest.fixture(scope="module")
def probe(tmp_path_factory: pytest.TempPathFactory) -> _Probe:
    work = tmp_path_factory.mktemp("no-arrow")
    home = work / "venv"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(home)
    python = home / "bin" / "python"
    site = Path(json.loads(_run(str(python), "-c", _PATHS, cwd=work)))
    for name, version in _locked().items():
        assert name not in _ARROW_DISTRIBUTIONS, f"the lock gives core {name} without the Arrow extras"
        _link(name, version, site)
    out = _run(str(python), "-c", _PROBE, "threetears.core", "threetears.epoch", cwd=work)
    return cast(_Probe, json.loads(out))


def test_the_venv_holds_no_arrow(probe: _Probe) -> None:
    """non-vacuity: with pyarrow or NumPy installed, every import below would pass."""
    assert probe["has"] == {"pyarrow": False, "numpy": False}, probe["has"]
    assert len(probe["imported"]) > 200, f"only {len(probe['imported'])} modules imported; discovery has collapsed"


def test_only_the_arrow_modules_need_arrow(probe: _Probe) -> None:
    """exactly the listed modules fail, each on pyarrow or NumPy."""
    failed = probe["failed"]
    unexpected = {name: missing for name, missing in failed.items() if name not in _ARROW_MODULES}
    assert not unexpected, (
        "these core/epoch modules fail to import without the Arrow extras; keep Arrow out of them (move the Arrow "
        f"part to a module of its own) or, if the module IS Arrow's, list it in _ARROW_MODULES with why: {unexpected}"
    )
    stale = sorted(set(_ARROW_MODULES) - set(failed))
    assert not stale, f"listed as Arrow's but import without it; drop them from _ARROW_MODULES: {stale}"
    wrong = {name: failed[name] for name in _ARROW_MODULES if failed[name] not in _ARROW_DISTRIBUTIONS}
    assert not wrong, f"listed Arrow modules failing on something else: {wrong}"
