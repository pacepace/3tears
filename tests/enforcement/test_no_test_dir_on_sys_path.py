"""
enforcement: no test tree puts a directory on ``sys.path``.

Every test module and helper is named from the repo root -- the root
``pyproject.toml`` sets ``pythonpath = ["."]`` and ``--import-mode=importlib``, so
``packages/agent/tools/tests/unit/tools/_pod_auth.py`` is
``packages.agent.tools.tests.unit.tools._pod_auth`` -- and a helper is imported by
that name. Names built that way cannot collide across packages.

A conftest that inserted its own ``tests`` directory into ``sys.path`` made every
child of that directory a TOP-LEVEL name. The agent-tools and agent-workspace
suites both have ``unit``, ``integration`` and ``enforcement`` directories, so in
one process the first suite imported owned ``unit`` and the other's
``from unit.tools._pod_auth import ...`` looked in the wrong directory and failed
to collect. The workspace-wide run passed only because ``agent/tools`` sorts
before ``agent/workspace``; running the two in the other order, or a subset that
happened to import workspace first, broke.

The scrape sidecar is exempt: it is a separate deployable with its own venv,
``--ignore``d by the workspace configuration, and its own pytest run.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SEPARATE_DEPLOYABLES = (Path("packages/scrape/sidecar"),)
_PATH_MUTATORS = frozenset({"insert", "append", "extend"})


def _test_tree_sources() -> list[Path]:
    """every python file in a package's test tree, plus every package conftest.

    :return: repo-relative paths
    :rtype: list[Path]
    """
    found: set[Path] = set()
    for path in (_REPO_ROOT / "packages").rglob("*.py"):
        relative = path.relative_to(_REPO_ROOT)
        if ".venv" in relative.parts or any(relative.is_relative_to(skip) for skip in _SEPARATE_DEPLOYABLES):
            continue
        if "tests" in relative.parts or relative.name == "conftest.py":
            found.add(relative)
    return sorted(found)


def _mutates_sys_path(node: ast.AST) -> bool:
    """whether ``node`` is ``sys.path.insert/append/extend(...)`` or an assignment to ``sys.path``.

    :param node: any AST node
    :ptype node: ast.AST
    :return: whether it changes ``sys.path``
    :rtype: bool
    """

    def _is_sys_path(expr: ast.AST) -> bool:
        return (
            isinstance(expr, ast.Attribute)
            and expr.attr == "path"
            and isinstance(expr.value, ast.Name)
            and expr.value.id == "sys"
        )

    result = False
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        result = node.func.attr in _PATH_MUTATORS and _is_sys_path(node.func.value)
    elif isinstance(node, (ast.Assign, ast.AugAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        result = any(_is_sys_path(target) for target in targets)
    return result


def test_no_test_tree_puts_a_directory_on_sys_path() -> None:
    sources = _test_tree_sources()
    assert len(sources) > 100, "found too few test sources; the walk is not seeing the test trees"

    offenders = [
        f"{path}:{node.lineno}"
        for path in sources
        for node in ast.walk(ast.parse((_REPO_ROOT / path).read_text(), filename=str(path)))
        if _mutates_sys_path(node)
    ]

    assert offenders == [], (
        "these test files change sys.path, which makes a test directory's children top-level names "
        "that collide with another suite's in the same process; import the helper by its repo-root "
        f"name instead (from packages.<pkg>.tests... import ...): {offenders}"
    )
