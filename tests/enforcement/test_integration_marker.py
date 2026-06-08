"""Enforcement: docker-backed tests declare the ``integration`` marker.

CI runs the unit + enforcement job with ``-m "not integration"`` on a
runner with **no docker**. A test that requests a docker-backed
(testcontainer) fixture but forgets
``pytestmark = pytest.mark.integration`` is therefore NOT deselected:
its fixture chain runs anyway and ERRORs the whole job. The failure is
invisible locally — where docker is present the test simply passes — so
the drift only surfaces in CI (and the file is silently excluded from
the dedicated ``-m integration`` job too, since that job deselects
everything *without* the marker).

This walker pins the contract: any ``test_*.py`` that references a
docker-rooted fixture (the testcontainer fixtures rooted at
:mod:`threetears.core.testing.fixtures` — ``db_container``, ``db_image``,
``pg_url``, ``pg_schema``, ``nats_container``) MUST carry a module-level
``pytestmark`` that references ``pytest.mark.integration``. In-process
"integration" tests that need no docker are deliberately NOT required to
carry the marker — they run correctly in the no-docker job.

**Heuristic boundary.** The walker parses each test file in isolation
and detects a docker dependency only when a docker-rooted fixture name
appears as a parameter *in that file* (directly, or via a local fixture
defined in the same file — the live case today). It does NOT resolve the
fixture graph across conftests, so a test that requests *only* a
conftest-defined fixture which itself chains off docker — without naming
a docker-rooted fixture in its own source — would escape the guard. Nor
does it detect ``@pytest.mark.usefixtures(...)`` injection. Both are
currently moot (every docker-chained conftest fixture is itself named
``pg_url`` / ``pg_schema`` and is already tracked; no suite uses
``usefixtures`` for docker), but if either pattern appears, add the new
fixture name to ``_DOCKER_ROOTED_FIXTURES`` or extend the detector.

AST-based, side-effect-free, well under the 15s budget.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# repo root: this file lives at tests/enforcement/, two levels down.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# fixtures whose setup requires a running docker daemon (testcontainers).
# rooted in threetears.core.testing.fixtures + the per-package aliases /
# derived fixtures that chain off them. requesting any of these — directly
# or through a local fixture that depends on one — pulls in docker.
_DOCKER_ROOTED_FIXTURES: frozenset[str] = frozenset(
    {
        "db_container",
        "db_image",
        "pg_url",
        "pg_schema",
        "nats_container",
    }
)


def _test_files() -> list[Path]:
    """every ``test_*.py`` under any workspace package ``tests`` tree.

    :return: sorted list of test file paths
    :rtype: list[Path]
    """
    packages = _REPO_ROOT / "packages"
    found = [path for path in packages.glob("**/tests/**/test_*.py") if "__pycache__" not in path.parts]
    return sorted(found)


def _requests_docker_fixture(tree: ast.Module) -> bool:
    """true when any function in the module takes a docker-rooted fixture param.

    pytest injects fixtures by parameter name, so a docker dependency
    shows up as a function/fixture argument named after one of the
    docker-rooted fixtures.

    :param tree: parsed module
    :ptype tree: ast.Module
    :return: whether the module requests a docker-rooted fixture
    :rtype: bool
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            params = [*args.posonlyargs, *args.args, *args.kwonlyargs]
            if any(p.arg in _DOCKER_ROOTED_FIXTURES for p in params):
                return True
    return False


def _declares_integration_marker(tree: ast.Module) -> bool:
    """true when the module assigns ``pytestmark`` referencing ``mark.integration``.

    accepts the single-marker form (``pytestmark = pytest.mark.integration``)
    and the list form (``pytestmark = [pytest.mark.integration, ...]``).

    :param tree: parsed module
    :ptype tree: ast.Module
    :return: whether the module declares the integration marker at module level
    :rtype: bool
    """
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets):
            continue
        for sub in ast.walk(node.value):
            if (
                isinstance(sub, ast.Attribute)
                and sub.attr == "integration"
                and isinstance(sub.value, ast.Attribute)
                and sub.value.attr == "mark"
            ):
                return True
    return False


def test_docker_backed_tests_declare_the_integration_marker() -> None:
    """a test requesting a docker-rooted fixture must be marked integration."""
    offenders: list[str] = []
    for path in _test_files():
        tree = ast.parse(path.read_text())
        if _requests_docker_fixture(tree) and not _declares_integration_marker(tree):
            offenders.append(str(path.relative_to(_REPO_ROOT)))
    if offenders:
        listing = "\n".join(f"  - {o}" for o in offenders)
        pytest.fail(
            "test file(s) request a docker-backed fixture but omit a module-level "
            "`pytestmark = pytest.mark.integration`. Without it the no-docker CI job "
            '(`-m "not integration"`) runs them and ERRORs:\n'
            f"{listing}"
        )
