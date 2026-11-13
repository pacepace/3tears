"""Structural gate: one import direction between the report input's modules and the page.

The analysis package holds two report layers:

- **The facts**: every top-level ``analysis/report_*.py`` and ``analysis/reporting.py``. They state what a campaign
  measured (the report input, its margins, the read tier's projection) and know nothing of how a page prints it.
- **The page**: ``analysis/report/`` and everything below it. It reads the facts and decides what a reader is shown:
  the figure-key grammar, the number forms, the words, the options and the call, the blocks and the serializers.

The page may import the facts; a fact module never imports the page. So the input can be built, checked and stored
without the page's grammar loaded, and a change to how a page prints never reaches what a campaign measured.

The facts side is matched by path, so a new flat ``report_*.py`` is held to the rule the day it lands.

* **Directly**: no fact module names a page module, by any import (runtime, ``TYPE_CHECKING``, function-level,
  relative, ``from package import submodule``) or by a string literal that is a dotted page path (an
  ``importlib`` call, a patch target).
* **Through a third module**: following every module's imports from each fact module, over the whole package, never
  reaches the page. A fact module importing an ``analysis`` module that imports the page is the same reach,
  laundered. Only imports a module writes are followed: a parent package's ``__init__``, which Python runs first, is
  not, because every module's parent would then reach every sibling. An import that names a package is an import
  of that package's ``__init__`` and is followed like any other.
* **The floor**: the facts side is exactly the three modules that hold the input and its arithmetic, and the page
  imports at least one of them, so the gate never passes on an empty side.
"""

from __future__ import annotations

import ast
import re
from collections import deque
from collections.abc import Iterator
from fnmatch import fnmatch
from pathlib import Path

import pytest

from packages.evals.tests.import_resolution import absolute_module
from packages.evals.tests.package_placement import discover

#: The directory holding the ``threetears`` namespace.
SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"

#: The analysis package, under which both sides live.
_ANALYSIS = "threetears.evals.analysis"

#: The page side: this package and everything below it.
_PAGE = f"{_ANALYSIS}.report"

#: The facts side, by file name, directly in the analysis package's directory.
_FACT_FILES = ("report_*.py", "reporting.py")

#: What the facts side is today: the input's shape, its margins and the read tier's projection.
_FACTS_TODAY = frozenset({f"{_ANALYSIS}.report_input", f"{_ANALYSIS}.report_margins", f"{_ANALYSIS}.reporting"})

#: A string literal that is a dotted path into the engine, matched whole, so a sentence naming a module is not one.
_DOTTED_ENGINE_PATH = re.compile(r"threetears\.evals(?:\.\w+)+")


def _is_page(module: str) -> bool:
    return module == _PAGE or module.startswith(f"{_PAGE}.")


def _owner(target: str, modules: dict[str, Path]) -> str | None:
    """The longest prefix of ``target`` that is a module of the tree, or None when no prefix is one."""
    parts = target.split(".")
    while parts and ".".join(parts) not in modules:
        parts.pop()
    return ".".join(parts) or None


def _named_modules(path: Path, modules: dict[str, Path], root: Path) -> Iterator[tuple[int, str]]:
    """``(line, module)`` for every engine module ``path`` names, at any depth, by import or by dotted string."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        targets: list[str] = []
        if isinstance(node, ast.Import):
            targets = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = absolute_module(path, node, root=root)
            targets = [f"{base}.{alias.name}" if f"{base}.{alias.name}" in modules else base for alias in node.names]
        elif (
            isinstance(node, ast.Constant) and isinstance(node.value, str) and _DOTTED_ENGINE_PATH.fullmatch(node.value)
        ):
            targets = [node.value]
        for target in targets:
            owner = _owner(target, modules)
            if owner is None:
                continue
            if isinstance(node, ast.Constant) and owner != target and not _is_page(owner):
                # A dotted string that names no module (an event name, a logger) is no reach, unless it names
                # something on the page, as a patch target does.
                continue
            yield getattr(node, "lineno", 0), owner


def fact_modules(root: Path) -> set[str]:
    """The facts side of ``root``'s tree: the modules whose files match :data:`_FACT_FILES` in ``analysis/``."""
    analysis = root / "threetears" / "evals" / "analysis"
    return {
        module
        for module, path in discover(root).items()
        if path.parent == analysis and any(fnmatch(path.name, pattern) for pattern in _FACT_FILES)
    }


def direct_violations(root: Path) -> list[str]:
    """Each place a fact module names a page module itself: ``module:line -> page module``."""
    modules = discover(root)
    return sorted(
        f"{fact}:{line} -> {target}"
        for fact in fact_modules(root)
        for line, target in _named_modules(modules[fact], modules, root)
        if _is_page(target)
    )


def laundered_reaches(root: Path) -> list[str]:
    """Each page module a fact module reaches through other modules' imports, with the chain that reaches it."""
    modules = discover(root)
    edges = {
        module: sorted({target for _, target in _named_modules(path, modules, root) if target != module})
        for module, path in modules.items()
    }
    reaches: list[str] = []
    for fact in sorted(fact_modules(root)):
        came_from: dict[str, str | None] = {fact: None}
        queue = deque([fact])
        while queue:
            module = queue.popleft()
            for target in edges.get(module, []):
                if target in came_from:
                    continue
                came_from[target] = module
                if _is_page(target):
                    chain = [target]
                    while (step := came_from[chain[-1]]) is not None:
                        chain.append(step)
                    reaches.append(" -> ".join(reversed(chain)))
                    continue  # the page's own imports are the page's business
                queue.append(target)
    return sorted(reaches)


# --- the gate over today's tree --------------------------------------------------------------------


def test_no_fact_module_imports_the_page() -> None:
    violations = direct_violations(SOURCE_ROOT)
    assert violations == [], (
        "a module of the report input imports the page; the page reads the facts, never the reverse. Move the code "
        "that needs the page into analysis/report/:\n  " + "\n  ".join(violations)
    )


def test_no_fact_module_reaches_the_page_through_another_module() -> None:
    reaches = laundered_reaches(SOURCE_ROOT)
    assert reaches == [], (
        "a module of the report input reaches the page through another module's imports:\n  " + "\n  ".join(reaches)
    )


def test_the_facts_side_is_the_input_its_margins_and_the_read_tier() -> None:
    """A new flat ``report_*.py`` joins the facts side by its path, so it is named here, on purpose."""
    assert fact_modules(SOURCE_ROOT) == _FACTS_TODAY


def test_the_page_reads_the_facts() -> None:
    """Non-vacuity: the page imports a fact module, so the two sides are what the walk thinks they are."""
    modules = discover(SOURCE_ROOT)
    facts = fact_modules(SOURCE_ROOT)
    readers = {
        module
        for module, path in modules.items()
        if _is_page(module)
        for _, target in _named_modules(path, modules, SOURCE_ROOT)
        if target in facts
    }
    assert readers, "no page module imports a fact module: the sides are misnamed, or the walk reads nothing"


# --- the checker fires: synthetic trees ------------------------------------------------------------


def _synthetic_tree(tmp_path: Path, files: dict[str, str]) -> Path:
    """Build a ``threetears/evals/`` tree under ``tmp_path`` with the given sources and every package's init."""
    for relative, source in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    engine = tmp_path / "threetears" / "evals"
    for directory in {p for p in engine.rglob("*") if p.is_dir()} | {engine}:
        init = directory / "__init__.py"
        if not init.exists():
            init.write_text("", encoding="utf-8")
    return tmp_path


_SYNTHETIC_BASE = {
    "threetears/evals/analysis/report_input.py": "",
    "threetears/evals/analysis/stats.py": "",
    "threetears/evals/analysis/report/forms.py": "",
    "threetears/evals/analysis/report/figure_keys.py": "",
}

_PLANTED = "threetears/evals/analysis/report_planted.py"


def test_a_type_checking_and_a_function_level_import_are_both_reported(tmp_path: Path) -> None:
    source = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from threetears.evals.analysis.report.forms import PageForms\n"
        "def f():\n"
        "    from .report import figure_keys\n"
    )
    root = _synthetic_tree(tmp_path, {**_SYNTHETIC_BASE, _PLANTED: source})
    assert direct_violations(root) == [
        "threetears.evals.analysis.report_planted:3 -> threetears.evals.analysis.report.forms",
        "threetears.evals.analysis.report_planted:5 -> threetears.evals.analysis.report.figure_keys",
    ]


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("import threetears.evals.analysis.report.forms\n", "threetears.evals.analysis.report.forms"),
        ("from .report.forms import PageForms\n", "threetears.evals.analysis.report.forms"),
        ("from . import report\n", "threetears.evals.analysis.report"),
        ("from threetears.evals.analysis.report import forms\n", "threetears.evals.analysis.report.forms"),
        (
            "import importlib\nFORMS = importlib.import_module('threetears.evals.analysis.report.forms')\n",
            "threetears.evals.analysis.report.forms",
        ),
    ],
)
def test_each_way_of_naming_the_page_is_reported(tmp_path: Path, source: str, target: str) -> None:
    root = _synthetic_tree(tmp_path, {**_SYNTHETIC_BASE, _PLANTED: source})
    assert [violation.rpartition(" -> ")[2] for violation in direct_violations(root)] == [target]


def test_a_new_flat_report_module_joins_the_facts_side_by_its_path(tmp_path: Path) -> None:
    root = _synthetic_tree(tmp_path, {**_SYNTHETIC_BASE, _PLANTED: ""})
    assert fact_modules(root) == {"threetears.evals.analysis.report_input", "threetears.evals.analysis.report_planted"}


def test_a_page_import_laundered_through_a_middle_module_is_reported(tmp_path: Path) -> None:
    files = {
        **_SYNTHETIC_BASE,
        "threetears/evals/analysis/report_planted2.py": "from threetears.evals.analysis import middle\n",
        "threetears/evals/analysis/middle.py": "from threetears.evals.analysis.report.forms import PageForms\n",
    }
    root = _synthetic_tree(tmp_path, files)
    assert direct_violations(root) == []
    assert laundered_reaches(root) == [
        "threetears.evals.analysis.report_planted2 -> threetears.evals.analysis.middle"
        " -> threetears.evals.analysis.report.forms"
    ]


def test_the_page_importing_a_fact_module_is_admitted(tmp_path: Path) -> None:
    files = {
        **_SYNTHETIC_BASE,
        "threetears/evals/analysis/report/x.py": "from threetears.evals.analysis.report_input import ReportInput\n",
        "threetears/evals/analysis/report_input.py": "from threetears.evals.analysis.stats import mean\n",
    }
    root = _synthetic_tree(tmp_path, files)
    assert (direct_violations(root), laundered_reaches(root)) == ([], [])
