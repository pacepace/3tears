"""Enforcement: every integration test runs on the pool, or in ``--here``, and the two lists say exactly which.

``scripts/test-integration.sh`` runs the integration suite on the testament pool: the paths in ``.testament.toml``'s
``[suites.integration] default_selection``, whose command deselects ``docker`` and ``secrets``. Its ``--here`` runs the
rest on the laptop: ``HERE_FILES``, with ``-m "docker or secrets"``. A test in neither runs nowhere, and nothing says
so: the pool and the laptop both go green without it. So:

- every ``test_*.py`` that marks a test ``integration`` lies under a suite path or is one of ``HERE_FILES``;
- ``HERE_FILES`` is exactly the test files that mark a test ``docker`` or ``secrets`` (integration or not: the live
  Claude CLI and Tavily tests are ``secrets`` too), and none of them lies under a suite path, where it would be
  deselected;
- every listed path exists, and every suite path holds an integration test, so a moved or deleted directory leaves
  no dead entry;
- ``process_only`` marks only integration tests (it is deselected outside the pool's process mode, which only the
  integration suite runs in).

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


def _here_files() -> list[str]:
    """``scripts/test-integration.sh``'s ``HERE_FILES``, one path per line between ``HERE_FILES=(`` and ``)``."""
    text = _SCRIPT.read_text(encoding="utf-8")
    assert "\nHERE_FILES=(\n" in text, "scripts/test-integration.sh no longer declares HERE_FILES=( one path per line )"
    block = text.split("\nHERE_FILES=(\n", 1)[1].split("\n)\n", 1)[0]
    return [line.strip() for line in block.splitlines() if line.strip()]


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
    suite, here = _suite_paths(), set(_here_files())
    nowhere = sorted(
        path for path in _integration_files() if path not in here and not any(_under(path, prefix) for prefix in suite)
    )
    assert not nowhere, (
        "integration test files in neither .testament.toml's [suites.integration] default_selection nor "
        f"scripts/test-integration.sh's HERE_FILES, so nothing runs them: {nowhere}. Add the path to the suite; a file "
        "marked `docker` or `secrets` goes in HERE_FILES."
    )


def test_here_is_exactly_the_docker_and_secrets_files() -> None:
    marked = sorted(
        str(path.relative_to(_REPO_ROOT))
        for path in _test_files()
        if _MARK_READER.marks(parse_python_path(path)) & {"docker", "secrets"}
    )
    assert sorted(_here_files()) == marked, (
        "scripts/test-integration.sh's HERE_FILES must list exactly the test files marking a test `docker` or "
        f"`secrets`, which only `--here` runs.\n  marked: {marked}\n  listed: {sorted(_here_files())}"
    )


def test_no_docker_or_secrets_test_lies_under_a_suite_path() -> None:
    suite = _suite_paths()
    deselected = sorted(path for path in _here_files() if any(_under(path, prefix) for prefix in suite))
    assert not deselected, (
        f"test files marking a test `docker` or `secrets` under a path the pool's integration suite runs: {deselected}. "
        "The suite deselects them; name the suite's other files individually instead of their directory."
    )


def test_every_listed_path_exists_and_every_suite_path_holds_an_integration_test() -> None:
    files = _integration_files()
    dead = [
        path
        for path in _suite_paths()
        if not (_REPO_ROOT / path).exists() or not any(_under(file, path) for file in files)
    ]
    dead += [path for path in _here_files() if not (_REPO_ROOT / path).is_file()]
    assert not dead, f"suite paths or HERE_FILES that are missing or hold no integration test: {dead}"


def test_process_only_marks_only_integration_tests() -> None:
    """``process_only`` is deselected outside process mode and only the integration suite runs in it, so a
    ``process_only`` module that is not also ``integration`` would run nowhere."""
    nowhere = sorted(
        str(path.relative_to(_REPO_ROOT))
        for path in _test_files()
        if {"process_only"} <= (marks := _MARK_READER.marks(parse_python_path(path))) and "integration" not in marks
    )
    assert not nowhere, f"modules marking `process_only` without `integration`, which run nowhere: {nowhere}"
    assert any("process_only" in _MARK_READER.marks(parse_python_path(path)) for path in _test_files()), (
        "the walk found no `process_only` module at all; the check above proves nothing"
    )


def test_no_plain_integration_test_sits_in_a_here_file_off_the_suite() -> None:
    """``--here`` selects ``docker or secrets``, and the pool never sees a ``HERE_FILES`` module: an integration test
    in one that carries neither mark runs nowhere."""
    suite = _suite_paths()
    stranded = sorted(
        f"{path}::{test}"
        for path in _here_files()
        if not any(_under(path, prefix) for prefix in suite)
        for test, marks in _MARK_READER.marks_by_test(parse_python_path(_REPO_ROOT / path)).items()
        if "integration" in marks and not marks & {"docker", "secrets"}
    )
    assert not stranded, (
        f"integration tests marked neither `docker` nor `secrets` in a HERE_FILES module, which run nowhere: {stranded}. "
        "Mark them, or move them to a module the suite runs."
    )
    tests = _MARK_READER.marks_by_test(
        parse_python_path(_REPO_ROOT / "packages/datasources/tests/integration/test_transaction_redshift_live.py")
    )
    assert tests and all({"integration", "secrets"} <= marks for marks in tests.values()), (
        "the control: a known secrets module's tests read as integration and secrets"
    )
