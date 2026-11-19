"""strict mypy over every workspace package, one test per package, as testament's ``typecheck`` suite.

These are tests so that testament can serve their results from its cache: a tree that passed here once,
from a laptop or from CI, is not checked again. Its cache serves only a result its recorder judged
cacheable, which rules out four things mypy would otherwise do:

- start a process: mypy runs in this one, through ``mypy.api.run``;
- read the clock: mypy times its own phases with ``time.time()``, which the recorder counts as live
  data, so the test freezes it. mypy is mypyc-compiled, and mypyc reaches ``time.time`` through the
  module's attribute, so the frozen one is what it calls; pytest bound its own timer at import;
- draw entropy: every build draws ``os.urandom(4)`` as an id that names worker files, and mypy runs no
  workers here; its cache writer draws a temp name for each file it replaces, in a directory only this
  process writes. Neither changes a finding, so the test hands mypy fixed bytes;
- read a file outside the tree, the venv and temp: ``--config-file`` names the repo's pyproject, so mypy
  probes no user config, and ``--cache-dir`` is under pytest's temp, so mypy's incremental cache never
  carries one run's state into another (testament's cache is the one that saves the work). mypy also
  lists ``/usr/local/lib/mypy`` on every run, a stub directory it falls back on for a broken install;
  the test answers that it is not there, so no host's stubs there can change a result.

Each test reports everything mypy printed, a dependency's errors included, since mypy checks what a
package imports: an error in core fails core's test and its dependents', never none of them.

Run it with ``./scripts/typecheck.sh`` (the pool, or here with ``TESTAMENT=off``); the unit suite never
collects this directory.
"""

from __future__ import annotations

import errno
import os
import time
from pathlib import Path

import pytest
from mypy import api

from typecheck.targets import EXAMPLE_HOSTS, REPO_ROOT, Package, packages

_PACKAGES = packages()

#: the stub directory mypy appends to its search path on every POSIX run (``modulefinder.default_lib_path``)
_FALLBACK_STUBS = "/usr/local/lib/mypy"


@pytest.fixture(scope="session")
def mypy_cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """one mypy cache for the session, under pytest's temp: a later package reuses what an earlier
    one analysed (core, which most import), and nothing outlives the run.

    :param tmp_path_factory: pytest's session temp factory
    :ptype tmp_path_factory: pytest.TempPathFactory
    :return: the cache directory
    :rtype: Path
    """
    return tmp_path_factory.mktemp("mypy-cache")


def _mypy(args: list[str], cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """run mypy in this process from the repo root, with nothing live to read, and fail on any finding.

    :param args: what to check, and any flags beyond the config file's
    :ptype args: list[str]
    :param cache: the session's mypy cache directory
    :ptype cache: Path
    :param monkeypatch: the test's monkeypatch
    :ptype monkeypatch: pytest.MonkeyPatch
    :return: nothing
    :rtype: None
    """
    monkeypatch.chdir(REPO_ROOT)
    # mypy reads the clock only to time itself, and entropy only to name files; neither changes a finding
    monkeypatch.setattr(time, "time", lambda: 0.0)
    monkeypatch.setattr(os, "urandom", lambda size: bytes(size))
    listdir = os.listdir

    def listdir_in_tree(path: str = ".") -> list[str]:
        if os.fspath(path) == _FALLBACK_STUBS:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), _FALLBACK_STUBS)
        return listdir(path)

    monkeypatch.setattr(os, "listdir", listdir_in_tree)
    stdout, stderr, status = api.run(
        ["--config-file", str(REPO_ROOT / "pyproject.toml"), "--cache-dir", str(cache), *args]
    )
    if status != 0:
        pytest.fail(f"mypy exit {status}\n{stdout}{stderr}", pytrace=False)


@pytest.mark.parametrize("package", _PACKAGES, ids=[p.test_id for p in _PACKAGES])
def test_package_typechecks(package: Package, mypy_cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """a package's import roots pass strict mypy.

    Modules are addressed with ``-p``, resolved through the config's ``mypy_path``: the agent packages
    nest ``src/`` two levels deep, which mypy cannot resolve from file paths alone.

    :param package: the package
    :ptype package: Package
    :param mypy_cache: the session's mypy cache directory
    :ptype mypy_cache: Path
    :param monkeypatch: the test's monkeypatch
    :ptype monkeypatch: pytest.MonkeyPatch
    :return: nothing
    :rtype: None
    """
    roots = [arg for root in package.roots for arg in ("-p", root)]
    _mypy(["--explicit-package-bases", *roots], mypy_cache, monkeypatch)


def test_evals_example_hosts_typecheck(mypy_cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """the evals example hosts pass strict mypy importing only the engine's public roots.

    That is what proves a host can implement the engine's protocols (a completion client, a candidate
    kind, a document store) from the exported names alone. The repository root goes on mypy's path,
    because the hosts import their own fixture package by its repository path.

    :param mypy_cache: the session's mypy cache directory
    :ptype mypy_cache: Path
    :param monkeypatch: the test's monkeypatch
    :ptype monkeypatch: pytest.MonkeyPatch
    :return: nothing
    :rtype: None
    """
    files = sorted(str(f.relative_to(REPO_ROOT)) for host in EXAMPLE_HOSTS for f in (REPO_ROOT / host).glob("*.py"))
    assert len(files) > len(EXAMPLE_HOSTS), f"the example hosts hold no modules: {files}"
    monkeypatch.setenv("MYPYPATH", str(REPO_ROOT))
    _mypy(["--explicit-package-bases", "--namespace-packages", *files], mypy_cache, monkeypatch)


def test_a_type_error_fails_the_check(tmp_path: Path, mypy_cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """the harness fails on a finding: a pass here is cached and served to every later run of the tree, so a
    harness that stopped failing would pass every tree after it.

    The module is written under the test's temp directory, which the recorder reads as the test's own.

    :param tmp_path: the test's temp directory
    :ptype tmp_path: Path
    :param mypy_cache: the session's mypy cache directory
    :ptype mypy_cache: Path
    :param monkeypatch: the test's monkeypatch
    :ptype monkeypatch: pytest.MonkeyPatch
    :return: nothing
    :rtype: None
    """
    broken = tmp_path / "broken.py"
    broken.write_text('def half(n: int) -> int:\n    return n / 2\n\n\nhalf("two")\n')
    with pytest.raises(pytest.fail.Exception) as failed:
        _mypy([str(broken)], mypy_cache, monkeypatch)
    said = str(failed.value)
    assert said.startswith("mypy exit 1"), said
    assert 'Incompatible return value type (got "float", expected "int")' in said
    assert 'Argument 1 to "half" has incompatible type "str"' in said
