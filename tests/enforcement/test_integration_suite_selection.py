"""Enforcement: every integration test is in the pool's integration suite or in the laptop's list, and runs.

``scripts/test-integration.sh`` runs the integration tests on two routes while they move to the testament pool
one package at a time: the paths in ``.testament.toml``'s ``[suites.integration] default_selection`` on the pool,
and the script's ``LOCAL_DIRS`` here, through testcontainers. A test in neither runs nowhere, and nothing says so:
the pool and the laptop both go green without it. So:

- every ``test_*.py`` that marks a test ``integration`` lies under a suite path or a ``LOCAL_DIRS`` path;
- one that also marks a test ``docker`` or ``secrets`` lies under a ``LOCAL_DIRS`` path and under no suite path:
  the suite's command deselects both, and the local run leaves out every suite path, so such a file under a suite
  path would run nowhere;
- every listed path exists and holds an integration test, so a moved or deleted directory leaves no dead entry;
- no ``LOCAL_DIRS`` path is, or lies inside, a suite path (the script leaves a suite path out of a local run that
  holds it, not the other way round).

A test counts as marked when ``pytest.mark.<name>`` appears in a decorator or a ``pytestmark`` assignment, at module
or class level, as ``scripts/integration_marks.py`` reads it; the script reads marks through the same module.
AST-based and side-effect-free.
"""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path
from types import ModuleType

from threetears.enforcement.common import parse_python_path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "test-integration.sh"


def _suite_paths() -> list[str]:
    """The integration suite's ``default_selection`` paths."""
    manifest = tomllib.loads((_REPO_ROOT / ".testament.toml").read_text(encoding="utf-8"))
    selection = manifest["suites"]["integration"]["default_selection"]
    return [path.rstrip("/") for path in selection if not path.startswith("-")]


def _local_dirs() -> list[str]:
    """``scripts/test-integration.sh``'s ``LOCAL_DIRS``, one path per line between ``LOCAL_DIRS=(`` and ``)``."""
    text = _SCRIPT.read_text(encoding="utf-8")
    assert "\nLOCAL_DIRS=(\n" in text, "scripts/test-integration.sh no longer declares LOCAL_DIRS=( one path per line )"
    block = text.split("\nLOCAL_DIRS=(\n", 1)[1].split("\n)\n", 1)[0]
    return [line.strip().rstrip("/") for line in block.splitlines() if line.strip()]


def _test_files() -> list[Path]:
    """Every ``test_*.py`` under a workspace package's ``tests`` tree or the repo's ``tests/``."""
    found = [*(_REPO_ROOT / "packages").glob("**/tests/**/test_*.py"), *(_REPO_ROOT / "tests").glob("**/test_*.py")]
    return sorted(path for path in found if "__pycache__" not in path.parts)


def _mark_reader() -> ModuleType:
    """``scripts/integration_marks.py``, the mark reader ``scripts/test-integration.sh`` runs too, so both agree."""
    spec = importlib.util.spec_from_file_location("integration_marks", _REPO_ROOT / "scripts" / "integration_marks.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MARK_READER = _mark_reader()


def _integration_files() -> dict[str, set[str]]:
    """Each test file marking a test ``integration``, repo-relative, with every mark it applies."""
    found: dict[str, set[str]] = {}
    for path in _test_files():
        marks = _MARK_READER.marks(parse_python_path(path))
        if "integration" in marks:
            found[str(path.relative_to(_REPO_ROOT))] = marks
    return found


def _under(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(f"{prefix}/")


def test_the_walk_finds_the_integration_tests_it_must() -> None:
    """The control: files known to be integration, docker and secrets are found as such, so an empty walk can't pass."""
    found = _integration_files()
    assert "packages/scheduled-jobs/tests/integration/test_background_dispatch.py" in found
    assert "docker" in found["packages/nats/tests/integration/test_lame_duck_handover_live.py"]
    assert "secrets" in found["packages/datasources/tests/integration/test_redshift_driver_live.py"]


def test_every_integration_test_runs_on_the_pool_or_here() -> None:
    suite, local = _suite_paths(), _local_dirs()
    nowhere = sorted(
        path
        for path in _integration_files()
        if not any(_under(path, prefix) for prefix in suite) and not any(_under(path, prefix) for prefix in local)
    )
    assert not nowhere, (
        "integration test files in neither .testament.toml's [suites.integration] default_selection nor "
        f"scripts/test-integration.sh's LOCAL_DIRS, so nothing runs them: {nowhere}. Add the directory to the suite "
        "when it runs on the pool (0 skips), else to LOCAL_DIRS."
    )


def test_docker_and_secrets_tests_run_here() -> None:
    local = _local_dirs()
    off_the_laptop = sorted(
        path
        for path, marks in _integration_files().items()
        if marks & {"docker", "secrets"} and not any(_under(path, prefix) for prefix in local)
    )
    assert not off_the_laptop, (
        "integration test files marking a test `docker` or `secrets` outside scripts/test-integration.sh's "
        f"LOCAL_DIRS: {off_the_laptop}. The pool's integration suite deselects both, so only the local run can "
        "run them; list the file in LOCAL_DIRS."
    )


def test_no_docker_or_secrets_test_lies_under_a_suite_path() -> None:
    suite = _suite_paths()
    deselected = sorted(
        path
        for path, marks in _integration_files().items()
        if marks & {"docker", "secrets"} and any(_under(path, prefix) for prefix in suite)
    )
    assert not deselected, (
        f"integration test files marking a test `docker` or `secrets` under a path the pool's integration suite runs: "
        f"{deselected}. The suite deselects them and scripts/test-integration.sh leaves suite paths out of its local "
        "run, so they would run nowhere; name the suite's other files individually instead of their directory."
    )


def test_every_listed_path_exists_and_holds_an_integration_test() -> None:
    files = _integration_files()
    dead = [
        path
        for path in [*_suite_paths(), *_local_dirs()]
        if not (_REPO_ROOT / path).exists() or not any(_under(file, path) for file in files)
    ]
    assert not dead, f"suite or LOCAL_DIRS paths that are missing or hold no integration test: {dead}"


def test_no_local_path_is_a_suite_path_or_inside_one() -> None:
    suite = _suite_paths()
    overlapping = [path for path in _local_dirs() if any(_under(path, prefix) for prefix in suite)]
    assert not overlapping, (
        f"LOCAL_DIRS paths that are, or lie inside, a path the pool's integration suite runs: {overlapping}. "
        "A package that moved to the suite leaves LOCAL_DIRS."
    )
