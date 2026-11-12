"""Enforcement: a test that drives headless Chromium carries the ``browser`` marker.

The evals render checks need headless Chromium (playwright), and no Docker. They carry ``browser``, not
``integration``, so they run in the unit suite: on the testament pool, whose image holds Chromium and whose manifest
sets ``THREETEARS_REQUIRE_CHROMIUM=1``, and on the GitHub shards, which install it. The marker is what names them:
``-m "not browser"`` is the one way to leave them out of a run that has no browser, and ``-m browser`` the one way to
run only them. A test that reaches Chromium without it is invisible to both.

This walker pins the contract: in any ``test_*.py`` under a package's ``tests`` tree, a test function that reaches a
Chromium helper carries ``@pytest.mark.browser``, or sits under a class or module ``pytestmark`` that names it. A
Chromium helper is a name the file imports from ``report_kit_support`` (``chromium_unavailable``, ``rendered``,
``render_checked``) or playwright's own entry points (``sync_playwright``, ``async_playwright``). A test reaches one
when it names it, or names a function of the same file that reaches one (followed to a fixed point, so a local
``_browser()`` wrapper counts).

**Heuristic boundary.** Each file is parsed alone: a helper reached only through another module's function, or
through a fixture defined in a conftest, escapes the walk. Neither happens today; if one appears, extend
``_HELPERS`` or the detector.

AST-based and side-effect-free.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from threetears.enforcement.common import parse_python_path


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

#: what each Chromium helper is imported from, by name
_HELPERS: dict[str, tuple[str, ...]] = {
    "chromium_unavailable": ("report_kit_support",),
    "rendered": ("report_kit_support",),
    "render_checked": ("report_kit_support",),
    "sync_playwright": ("playwright.sync_api",),
    "async_playwright": ("playwright.async_api",),
}

_Function = ast.FunctionDef | ast.AsyncFunctionDef


def _test_files() -> list[Path]:
    """Every ``test_*.py`` under a workspace package's ``tests`` tree."""
    found = (_REPO_ROOT / "packages").glob("**/tests/**/test_*.py")
    return sorted(path for path in found if "__pycache__" not in path.parts)


def _imported_helpers(tree: ast.Module) -> set[str]:
    """The names this module binds to a Chromium helper, by importing it from where the helper lives."""
    bound: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        for alias in node.names:
            homes = _HELPERS.get(alias.name, ())
            if any(node.module == home or node.module.endswith(f".{home}") for home in homes):
                bound.add(alias.asname or alias.name)
    return bound


def _names(node: ast.AST) -> set[str]:
    """Every bare name a function's body loads."""
    return {sub.id for sub in ast.walk(node) if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load)}


def _browser_mark(node: ast.expr) -> bool:
    """Whether a decorator or a ``pytestmark`` value names ``mark.browser``."""
    return any(
        isinstance(sub, ast.Attribute)
        and sub.attr == "browser"
        and isinstance(sub.value, ast.Attribute)
        and sub.value.attr == "mark"
        for sub in ast.walk(node)
    )


def _pytestmark_names_browser(body: list[ast.stmt]) -> bool:
    """Whether a module's or a class's body assigns a ``pytestmark`` that names ``mark.browser``."""
    for node in body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets):
            if _browser_mark(node.value):
                return True
    return False


def _unmarked_browser_tests(tree: ast.Module) -> list[str]:
    """The tests in ``tree`` that reach a Chromium helper without the ``browser`` marker, by name."""
    helpers = _imported_helpers(tree)
    if not helpers:
        return []
    functions = [node for node in ast.walk(tree) if isinstance(node, _Function)]
    # the module's own functions that reach a helper, directly or through each other, to a fixed point
    reaching = set(helpers)
    grew = True
    while grew:
        grew = False
        for function in functions:
            if function.name not in reaching and _names(function) & reaching:
                reaching.add(function.name)
                grew = True
    module_marked = _pytestmark_names_browser(tree.body)
    offenders: list[str] = []

    def visit(body: list[ast.stmt], prefix: str, marked: bool) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                visit(node.body, f"{prefix}{node.name}::", marked or _pytestmark_names_browser(node.body))
            elif isinstance(node, _Function) and node.name.startswith("test"):
                if _names(node) & reaching and not (marked or any(_browser_mark(d) for d in node.decorator_list)):
                    offenders.append(f"{prefix}{node.name}")

    visit(tree.body, "", module_marked)
    return offenders


def test_tests_that_drive_chromium_carry_the_browser_marker() -> None:
    """A test reaching a Chromium helper must be marked ``browser``."""
    offenders: list[str] = []
    for path in _test_files():
        for name in _unmarked_browser_tests(parse_python_path(path)):
            offenders.append(f"{path.relative_to(_REPO_ROOT)}::{name}")
    if offenders:
        listing = "\n".join(f"  - {o}" for o in offenders)
        pytest.fail(
            "test(s) drive headless Chromium without `@pytest.mark.browser`. Without it a run that has no browser "
            '(`-m "not browser"`) cannot leave them out, and a run of the browser tests (`-m browser`) misses them:\n'
            f"{listing}"
        )


_PLANTED = """
import pytest
from packages.evals.tests.report_kit_support import chromium_unavailable, rendered as drawn

def _browser():
    if chromium_unavailable():
        pytest.skip("no browser")

def test_unmarked_through_a_local_wrapper():
    _browser()

@pytest.mark.browser
def test_marked():
    drawn("page", "() => 1")

def test_no_browser():
    rendered = "a local name, not the helper"
    assert rendered

class TestHeld:
    pytestmark = pytest.mark.browser

    def test_marked_by_its_class(self):
        _browser()

class TestLoose:
    def test_unmarked_in_a_class(self):
        drawn("page", "() => 1")
"""


def test_the_walker_finds_an_unmarked_test_through_a_local_wrapper_and_passes_a_marked_one() -> None:
    """The control: the detector flags exactly the planted unmarked tests."""
    assert _unmarked_browser_tests(ast.parse(_PLANTED)) == [
        "test_unmarked_through_a_local_wrapper",
        "TestLoose::test_unmarked_in_a_class",
    ]


def test_a_module_pytestmark_marks_every_test_in_it() -> None:
    marked = "import pytest\npytestmark = [pytest.mark.browser]\n" + _PLANTED.split("import pytest", 1)[1]
    assert _unmarked_browser_tests(ast.parse(marked)) == []


def test_the_walker_sees_the_render_tests() -> None:
    """The control on the tree: the evals render tests are found and every one of them is marked."""
    tests = files = 0
    for path in _test_files():
        tree = parse_python_path(path)
        if _imported_helpers(tree):
            files += 1
            tests += sum(
                1
                for node in ast.walk(tree)
                if isinstance(node, _Function) and any(_browser_mark(d) for d in node.decorator_list)
            )
    assert files >= 11 and tests >= 41, (files, tests)


def test_every_browser_test_lives_in_packages_evals() -> None:
    """CI's GitHub shards install Chromium on the evals shards only (test_ci_unit_shards.py), so a browser test
    elsewhere would run there without it. Move the install with it before moving one out."""
    outside = [
        str(path.relative_to(_REPO_ROOT))
        for path in _test_files()
        if _imported_helpers(parse_python_path(path)) and not path.is_relative_to(_REPO_ROOT / "packages" / "evals")
    ]
    assert outside == [], outside
