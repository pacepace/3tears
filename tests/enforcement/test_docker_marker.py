"""Enforcement: a test module that drives a Docker daemon itself carries the ``docker`` marker.

The testament pool runs the tests in a container with no Docker daemon, and ``.testament.toml``'s suites
deselect ``docker``. A test that starts its own container (``DockerContainer(...)``, ``PostgresContainer(...)``),
runs ``docker exec``, or requests ``searxng_container`` (a SearXNG container that has no process form) can only run
where a daemon answers: the laptop, through ``scripts/test-integration.sh``. The marker is what keeps it off the pool.
Without it, the pool would run the test and fail it, or a skip would hide that it never ran.

This walker pins the contract: any ``test_*.py`` under a package's ``tests`` tree or the repo's ``tests/`` that does
one of those things carries a module-level ``pytestmark`` naming ``pytest.mark.docker``. The mark is per module, not
per test, because each of these modules builds its containers in module or session fixtures that every test in it
shares.

**Heuristic boundary.** Each file is parsed alone. A container started by a helper in another module, or by a
conftest fixture under a name other than ``searxng_container``, escapes the walk; the session fixtures of
``threetears.core.testing.fixtures`` (``db_container``, ``nats_container``, ``s3_container``) are deliberately not
docker indicators, since they are the services the pool provides. If a new container starter appears, add it to
``_STARTERS`` or ``_FIXTURES``.

AST-based and side-effect-free.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from threetears.enforcement.common import parse_python_path


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

#: callables that start a container of their own
_STARTERS: frozenset[str] = frozenset({"DockerContainer", "PostgresContainer"})

#: fixtures that only exist as a container
_FIXTURES: frozenset[str] = frozenset({"searxng_container"})

#: callables that run a command line (subprocess's, asyncio's and os's)
_RUNNERS: frozenset[str] = frozenset(
    {
        "run",
        "Popen",
        "call",
        "check_call",
        "check_output",
        "create_subprocess_exec",
        "create_subprocess_shell",
        "system",
    }
)


def _test_files() -> list[Path]:
    """Every ``test_*.py`` under a workspace package's ``tests`` tree or the repo's ``tests/``."""
    found = [*(_REPO_ROOT / "packages").glob("**/tests/**/test_*.py"), *(_REPO_ROOT / "tests").glob("**/test_*.py")]
    return sorted(path for path in found if "__pycache__" not in path.parts)


def _callee(call: ast.Call) -> str | None:
    """The bare name a call invokes: ``f`` for ``f(...)`` and ``mod.f(...)``."""
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _runs_docker_exec(call: ast.Call) -> bool:
    """Whether ``call`` runs ``docker exec``: an argv list (``run(["docker", "exec", ...])``), separate arguments
    (``create_subprocess_exec("docker", "exec", ...)``) or a command line (``system("docker exec ...")``)."""
    if _callee(call) not in _RUNNERS or not call.args:
        return False
    first = call.args[0]
    words = first.elts if isinstance(first, (ast.List, ast.Tuple)) else call.args
    head = [w.value if isinstance(w, ast.Constant) else None for w in words[:2]]
    if head == ["docker", "exec"]:
        return True
    return isinstance(first, ast.Constant) and isinstance(first.value, str) and first.value.startswith("docker exec ")


def _drives_docker(tree: ast.Module) -> list[str]:
    """What in ``tree`` drives Docker, one line each (empty when nothing does)."""
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _callee(node)
            if name in _STARTERS:
                found.append(f"{name}(...) at line {node.lineno}")
            if _runs_docker_exec(node):
                found.append(f"docker exec at line {node.lineno}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            for param in (*args.posonlyargs, *args.args, *args.kwonlyargs):
                if param.arg in _FIXTURES:
                    found.append(f"{param.arg} requested by {node.name}")
    return found


def _declares_docker_marker(tree: ast.Module) -> bool:
    """Whether the module assigns a ``pytestmark`` naming ``mark.docker`` (alone or in a list)."""
    return _declares_module_marker(tree, "docker")


def _declares_module_marker(tree: ast.Module, name: str) -> bool:
    """Whether the module assigns a ``pytestmark`` naming ``mark.<name>`` (alone or in a list)."""
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets):
            continue
        for sub in ast.walk(node.value):
            if (
                isinstance(sub, ast.Attribute)
                and sub.attr == name
                and isinstance(sub.value, ast.Attribute)
                and sub.value.attr == "mark"
            ):
                return True
    return False


def test_modules_that_drive_docker_carry_the_docker_marker() -> None:
    """A module that starts a container or runs ``docker exec`` must be marked ``docker``."""
    offenders: list[str] = []
    for path in _test_files():
        tree = parse_python_path(path)
        found = _drives_docker(tree)
        if found and not _declares_docker_marker(tree):
            offenders.append(f"{path.relative_to(_REPO_ROOT)}: {found[0]}")
    if offenders:
        listing = "\n".join(f"  - {o}" for o in offenders)
        pytest.fail(
            "test module(s) drive a Docker daemon without a module-level `pytestmark` naming "
            "`pytest.mark.docker`. The testament pool has no daemon and deselects `docker`; without the "
            "marker it runs them and they fail:\n"
            f"{listing}"
        )


_PLANTED = """
import subprocess
import pytest
from testcontainers.core.container import DockerContainer
from testcontainers.community import postgres

def test_starts_one():
    DockerContainer("img").start()

def test_starts_postgres():
    postgres.PostgresContainer("pg").start()

def test_execs(cid):
    subprocess.run(["docker", "exec", cid, "true"])

def test_searxng(searxng_container):
    assert searxng_container

def test_names_docker_but_drives_nothing(db_container):
    subprocess.run(["docker", "ps"])
    print("docker exec is a phrase here, not a command line")
    assert db_container
"""


def test_the_walker_finds_each_kind_of_docker_use() -> None:
    """The control: every planted kind is found, and a session service fixture is not one."""
    assert sorted(_drives_docker(ast.parse(_PLANTED))) == [
        "DockerContainer(...) at line 8",
        "PostgresContainer(...) at line 11",
        "docker exec at line 14",
        "searxng_container requested by test_searxng",
    ]
    assert not _declares_docker_marker(ast.parse(_PLANTED))


def test_a_module_pytestmark_list_holding_docker_counts() -> None:
    marked = "import pytest\npytestmark = [pytest.mark.integration, pytest.mark.docker]\n"
    assert _declares_docker_marker(ast.parse(marked))
    assert not _declares_docker_marker(ast.parse("import pytest\npytestmark = pytest.mark.integration\n"))


def test_every_docker_module_is_also_an_integration_module() -> None:
    """A ``docker`` module runs only through ``scripts/test-integration.sh``, which selects ``-m integration``.

    The pool deselects ``docker``, and the integration route selects ``integration``: a module marked ``docker`` and
    not ``integration`` would run on neither, only on CI's fallback shards. So the one implies the other.
    """
    offenders = [
        str(path.relative_to(_REPO_ROOT))
        for path in _test_files()
        if (tree := parse_python_path(path)) is not None
        and _declares_docker_marker(tree)
        and not _declares_module_marker(tree, "integration")
    ]
    assert not offenders, (
        "module(s) marked `docker` but not `integration` run on no local or pool route; "
        f"add `pytest.mark.integration` to their `pytestmark`: {offenders}"
    )


def test_the_docker_and_integration_check_reads_each_marker_apart() -> None:
    """The control: a module marked only ``docker`` is told apart from one marked both."""
    only_docker = ast.parse("import pytest\npytestmark = pytest.mark.docker\n")
    both = ast.parse("import pytest\npytestmark = [pytest.mark.integration, pytest.mark.docker]\n")
    assert _declares_docker_marker(only_docker) and not _declares_module_marker(only_docker, "integration")
    assert _declares_module_marker(both, "integration") and _declares_docker_marker(both)


def test_the_walker_sees_the_known_docker_modules() -> None:
    """The control on the tree: the modules known to drive Docker are found, so a broken detector can't pass."""
    known = {
        "packages/nats/tests/integration/test_lame_duck_handover_live.py",
        "packages/scrape/tests/integration/test_operator_pipe_sidecar.py",
        "packages/search/tests/test_searxng_live_scoring.py",
    }
    seen = {str(path.relative_to(_REPO_ROOT)) for path in _test_files() if _drives_docker(parse_python_path(path))}
    assert known <= seen, known - seen
