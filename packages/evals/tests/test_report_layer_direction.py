"""Structural gate: one import direction between the report input's modules and the page.

The analysis package holds two report layers:

- **The facts**: every top-level ``analysis/report_*.py`` and ``analysis/reporting.py``. They state what a campaign
  measured (the report input, its margins, the read tier's projection) and know nothing of how a page prints it.
- **The page**: ``analysis/report/`` and everything below it. It reads the facts and decides what a reader is shown:
  the figure-key grammar, the number forms, the words, the options and the call, the blocks and the serializers.

The page may import the facts; a fact module never imports the page. So a change to how a page prints never reaches
the code that states what a campaign measured.

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

**Numbers never import presentation.** The same direction, over the whole analysis package:

- **Numbers** compute, decide or assemble: the statistics, the bundle, the lenses, the report input.
- **Presentation** turns numbers into something read: the page, the analyst's tools, the modules that only print,
  and the engine packages above the analysis package.

Which side a module is on is written once, under "the two sides" below: a rule by directory, plus each module that
sits directly in ``analysis/``, by name. The modules directly in ``analysis/`` are named on both sides, and none
is numbers by default, because a default would place a new printing module on the numbers side unread. A module there that neither the rule nor a name places fails the gate, so
nothing joins the package unclassified.

A wording or layout change must never move a measured result, so no numbers module imports a presentation module,
by any of the ways above, directly or through a module that is on neither side. The rule has no exemption: code a
numbers module needs from presentation moves down to numbers (a decision, a typed value, a fixed-form speller in
``kernel/numbers.py``), or the wording that needs presentation moves up.
"""

from __future__ import annotations

import ast
import functools
import re
from collections import deque
from collections.abc import Iterator
from fnmatch import fnmatch
from pathlib import Path

import pytest

from packages.evals.tests.import_resolution import absolute_module
from packages.evals.tests.package_placement import discover, eval_relative
from threetears.enforcement.common import clear_parse_cache, parse_python_path

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
    for node in ast.walk(parse_python_path(path)):
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
            if isinstance(node, ast.Constant) and owner != target and side(owner) != PRESENTATION:
                # A dotted string that names no module (an event name, a logger) is no reach, unless it names
                # something in a presentation module, as a patch target does.
                continue
            yield getattr(node, "lineno", 0), owner


@functools.cache
def import_graph(root: Path) -> dict[str, tuple[tuple[int, str], ...]]:
    """Each module of ``root``'s tree, with the ``(line, module)`` of every other engine module it names.

    Read once for a tree and kept, so every gate in this file walks the same graph and the source is parsed once.
    """
    modules = discover(root)
    return {
        module: tuple(sorted({(line, target) for line, target in _named_modules(path, modules, root)}))
        for module, path in modules.items()
    }


@pytest.fixture(scope="module", autouse=True)
def _release_the_parsed_trees() -> Iterator[None]:
    """Give back the graphs and the parsed trees when this file is done: a package's trees are large."""
    yield
    import_graph.cache_clear()
    clear_parse_cache()


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
    graph = import_graph(root)
    return sorted(
        f"{fact}:{line} -> {target}" for fact in fact_modules(root) for line, target in graph[fact] if _is_page(target)
    )


def laundered_reaches(root: Path) -> list[str]:
    """Each page module a fact module reaches through other modules' imports, with the chain that reaches it."""
    edges = _edges(root)
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


def _edges(root: Path) -> dict[str, list[str]]:
    """Each module of ``root``'s tree, with the other modules it names."""
    return {
        module: sorted({target for _, target in named if target != module})
        for module, named in import_graph(root).items()
    }


# --- the two sides ---------------------------------------------------------------------------------

NUMBERS = "numbers"
PRESENTATION = "presentation"

#: Directories of ``analysis/`` whose every module is numbers.
_NUMBERS_TREES = frozenset({"bundle", "lenses"})

#: Directories of ``analysis/`` whose every module is presentation.
_PRESENTATION_TREES = frozenset({"report", "analyst"})

#: The numbers modules directly in ``analysis/``.
_NUMBERS_MODULES = frozenset(
    {
        "agreement",
        "arms",
        "aspects",
        "bar_proposals",
        "campaign_bundle",
        "campaign_readings",
        "campaigns",
        "cassette_mode",
        "cell_mean",
        "cells",
        "completeness",
        "confusion",
        "contention",
        "depth",
        "depth_reading",
        "errors",
        "floors",
        "generation_tally",
        "judge_drift",
        "judge_kind_readings",
        "judge_profiles",
        "judge_temperature",
        "kinds",
        "latency_partition",
        "measurement_windows",
        "out_of_run_spend",
        "quantities",
        "reads",
        "references",
        "remainder",
        "report_input",
        "report_margins",
        "reporting",
        "revisions",
        "run_ending",
        "said",
        "separation",
        "significance",
        "stats",
        "style",
    }
)

#: The presentation modules directly in ``analysis/``: each builds a report, reads a built one, or only prints.
_SURFACES = frozenset({"analyst_gaps", "campaign_title", "doc_analysis", "gate", "lens_text", "service", "summary"})

#: The engine packages above the analysis package. All of them are presentation to a numbers module.
_PRESENTATION_PACKAGES = frozenset({"actions", "harness", "ops", "quick", "transports"})


def side(module: str) -> str | None:
    """Which side ``module`` is on: :data:`NUMBERS`, :data:`PRESENTATION`, or None when it is on neither.

    ``analysis/__init__.py`` is on neither: it is the public root and re-exports both sides. A module directly in
    ``analysis/`` that no name above places is on neither too, and :func:`unclassified` reports it.
    """
    package, _, below = (eval_relative(module) or "").partition(".")
    if package in _PRESENTATION_PACKAGES:
        return PRESENTATION
    if package != "analysis" or not below:
        return None
    first, _, deeper = below.partition(".")
    if first in _NUMBERS_TREES or (first in _NUMBERS_MODULES and not deeper):
        return NUMBERS
    if first in _PRESENTATION_TREES or (first in _SURFACES and not deeper):
        return PRESENTATION
    return None


def unclassified(root: Path) -> list[str]:
    """The modules below ``analysis/`` in ``root``'s tree that the two sides do not place."""
    return sorted(module for module in discover(root) if module.startswith(f"{_ANALYSIS}.") and side(module) is None)


def stale_placements(root: Path) -> list[str]:
    """The names the two sides place that ``root``'s tree no longer holds."""
    modules = discover(root)
    placed = _NUMBERS_TREES | _PRESENTATION_TREES | _NUMBERS_MODULES | _SURFACES
    return sorted(name for name in placed if f"{_ANALYSIS}.{name}" not in modules)


def presentation_imports(root: Path) -> dict[tuple[str, str], int]:
    """Each ``(numbers module, presentation module)`` pair where the first names the second, with its first line.

    Both names are relative to ``threetears.evals``.
    """
    pairs: dict[tuple[str, str], int] = {}
    for module, named in import_graph(root).items():
        if side(module) != NUMBERS:
            continue
        for line, target in named:
            if side(target) == PRESENTATION:
                pairs.setdefault((eval_relative(module) or "", eval_relative(target) or ""), line)
    return pairs


def laundered_presentation(root: Path) -> list[str]:
    """Each presentation module a numbers module reaches through modules on neither side, with the chain.

    A chain through another numbers module is that module's own import, so it is not followed here.
    """
    edges = _edges(root)
    reaches: list[str] = []
    for start in sorted(module for module in edges if side(module) == NUMBERS):
        came_from: dict[str, str | None] = {start: None}
        queue = deque(target for target in edges[start] if side(target) is None)
        came_from.update(dict.fromkeys(queue, start))
        while queue:
            module = queue.popleft()
            for target in edges.get(module, []):
                if target in came_from or side(target) == NUMBERS:
                    continue
                came_from[target] = module
                if side(target) == PRESENTATION:
                    chain = [target]
                    while (step := came_from[chain[-1]]) is not None:
                        chain.append(step)
                    reaches.append(" -> ".join(reversed(chain)))
                    continue
                queue.append(target)
    return sorted(reaches)


def pair_findings(found: dict[tuple[str, str], int]) -> list[str]:
    """One sentence for each pair in the tree: the importer, the line, what it imports and what to do instead."""
    return [
        f"{importer} (line {line}) imports {imported}: numbers never import presentation. Move the code that "
        f"needs {imported} to the presentation side, or move what both sides need down to numbers."
        for (importer, imported), line in sorted(found.items())
    ]


# --- the gate over today's tree --------------------------------------------------------------------


def test_every_analysis_module_is_on_a_side() -> None:
    missing = unclassified(SOURCE_ROOT)
    assert missing == [], (
        "a module in analysis/ is on neither side. Name each in _NUMBERS_MODULES if it computes, decides or "
        "assembles, or in _SURFACES if it builds a page or prints:\n  " + "\n  ".join(missing)
    )


def test_every_placed_name_is_a_module() -> None:
    stale = stale_placements(SOURCE_ROOT)
    assert stale == [], (
        "the two sides name a module or directory analysis/ no longer holds. Delete each name, or rename it with "
        "its file:\n  " + "\n  ".join(stale)
    )


def test_no_numbers_module_imports_presentation() -> None:
    findings = pair_findings(presentation_imports(SOURCE_ROOT))
    assert findings == [], "\n".join(findings)


def test_no_numbers_module_reaches_presentation_through_a_module_on_neither_side() -> None:
    reaches = laundered_presentation(SOURCE_ROOT)
    assert reaches == [], (
        "a numbers module reaches presentation through a module on neither side. Break each chain at its first "
        "import:\n  " + "\n  ".join(reaches)
    )


def test_presentation_reads_numbers() -> None:
    """Non-vacuity: both sides hold modules and presentation imports numbers."""
    graph = import_graph(SOURCE_ROOT)
    sides = {module: side(module) for module in graph}
    readers = {
        module
        for module, named in graph.items()
        if sides[module] == PRESENTATION and any(sides[target] == NUMBERS for _, target in named)
    }
    assert readers, "no presentation module imports a numbers module: the sides are misnamed, or the walk reads nothing"


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
    facts = fact_modules(SOURCE_ROOT)
    readers = {
        module
        for module, named in import_graph(SOURCE_ROOT).items()
        if _is_page(module)
        for _, target in named
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
        (
            "TARGET = 'threetears.evals.analysis.report.forms.PageForms'\n",
            "threetears.evals.analysis.report.forms",
        ),
    ],
)
def test_each_way_of_naming_the_page_is_reported(tmp_path: Path, source: str, target: str) -> None:
    root = _synthetic_tree(tmp_path, {**_SYNTHETIC_BASE, _PLANTED: source})
    assert [violation.rpartition(" -> ")[2] for violation in direct_violations(root)] == [target]


def test_a_dotted_string_naming_no_module_off_the_page_is_no_reach(tmp_path: Path) -> None:
    """an event or logger name shaped like a path into the engine names no module, so it reaches nothing.

    Its longest module prefix is ``analysis`` itself, whose init imports the page, as the real one does: read as a
    reach, the name would launder the whole page into the facts side.
    """
    files = {
        **_SYNTHETIC_BASE,
        "threetears/evals/analysis/__init__.py": "from threetears.evals.analysis import report\n",
        _PLANTED: "EVENT = 'threetears.evals.analysis.stats_ready'\n",
    }
    root = _synthetic_tree(tmp_path, files)
    assert (direct_violations(root), laundered_reaches(root)) == ([], [])


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


# --- numbers and presentation: the checker fires ---------------------------------------------------

_PLANTED_NUMBERS = "threetears/evals/analysis/stats.py"


def _pairs(root: Path) -> list[tuple[str, str]]:
    return sorted(presentation_imports(root))


@pytest.mark.parametrize(
    ("source", "imported"),
    [
        ("from threetears.evals.analysis.report.forms import PageForms\n", "analysis.report.forms"),
        ("from .report import forms\n", "analysis.report.forms"),
        ("def f():\n    from threetears.evals.analysis.analyst import skill\n", "analysis.analyst.skill"),
        ("from threetears.evals.analysis import summary\n", "analysis.summary"),
        ("import threetears.evals.harness.writer_eval\n", "harness.writer_eval"),
        ("TARGET = 'threetears.evals.analysis.report.forms.PageForms'\n", "analysis.report.forms"),
        ("TARGET = 'threetears.evals.analysis.analyst.skill.render'\n", "analysis.analyst.skill"),
        ("TARGET = 'threetears.evals.analysis.summary.print_summary'\n", "analysis.summary"),
        ("TARGET = 'threetears.evals.harness.writer_eval.WriterEval'\n", "harness.writer_eval"),
        ("TARGET = 'threetears.evals.ops.report_writer.write'\n", "ops.report_writer"),
    ],
)
def test_each_way_a_numbers_module_names_presentation_is_a_pair(tmp_path: Path, source: str, imported: str) -> None:
    files = {
        **_SYNTHETIC_BASE,
        "threetears/evals/analysis/analyst/skill.py": "",
        "threetears/evals/analysis/summary.py": "",
        "threetears/evals/harness/writer_eval.py": "",
        "threetears/evals/ops/report_writer/__init__.py": "",
        _PLANTED_NUMBERS: source,
    }
    root = _synthetic_tree(tmp_path, files)
    assert _pairs(root) == [("analysis.stats", imported)]


def test_a_module_in_a_numbers_directory_is_held_by_its_path(tmp_path: Path) -> None:
    files = {
        **_SYNTHETIC_BASE,
        "threetears/evals/analysis/bundle/planted.py": "from threetears.evals.analysis.report.forms import PageForms\n",
    }
    root = _synthetic_tree(tmp_path, files)
    assert unclassified(root) == []
    assert _pairs(root) == [("analysis.bundle.planted", "analysis.report.forms")]


def test_a_new_module_directly_in_analysis_is_on_neither_side_until_named(tmp_path: Path) -> None:
    files = {
        **_SYNTHETIC_BASE,
        "threetears/evals/analysis/planted.py": "",
        "threetears/evals/analysis/planted_tree/inner.py": "",
        "threetears/evals/analysis/stats/inner.py": "",
    }
    root = _synthetic_tree(tmp_path, files)
    assert unclassified(root) == [
        "threetears.evals.analysis.planted",
        "threetears.evals.analysis.planted_tree",
        "threetears.evals.analysis.planted_tree.inner",
        "threetears.evals.analysis.stats.inner",
    ]


def test_a_placed_name_with_no_module_is_reported(tmp_path: Path) -> None:
    root = _synthetic_tree(tmp_path, dict(_SYNTHETIC_BASE))
    stale = stale_placements(root)
    assert "agreement" in stale
    assert "bundle" in stale
    assert not {"report", "report_input", "stats"} & set(stale)


def test_presentation_importing_numbers_or_presentation_is_no_pair(tmp_path: Path) -> None:
    files = {
        **_SYNTHETIC_BASE,
        "threetears/evals/analysis/report/x.py": "from threetears.evals.analysis.stats import mean\n",
        "threetears/evals/analysis/summary.py": "from threetears.evals.analysis.report.forms import PageForms\n",
        "threetears/evals/kernel/numbers.py": "",
        _PLANTED_NUMBERS: "from threetears.evals.kernel.numbers import spell_count\n",
    }
    root = _synthetic_tree(tmp_path, files)
    assert (_pairs(root), laundered_presentation(root)) == ([], [])


def test_presentation_reached_through_a_module_on_neither_side_is_reported(tmp_path: Path) -> None:
    files = {
        **_SYNTHETIC_BASE,
        "threetears/evals/kernel/middle.py": "from threetears.evals.analysis.report.forms import PageForms\n",
        _PLANTED_NUMBERS: "from threetears.evals.kernel import middle\n",
    }
    root = _synthetic_tree(tmp_path, files)
    assert _pairs(root) == []
    assert laundered_presentation(root) == [
        "threetears.evals.analysis.stats -> threetears.evals.kernel.middle -> threetears.evals.analysis.report.forms"
    ]


def test_the_analysis_root_launders_presentation_to_a_numbers_module_that_imports_it(tmp_path: Path) -> None:
    files = {
        **_SYNTHETIC_BASE,
        "threetears/evals/analysis/__init__.py": "from threetears.evals.analysis.report import forms\n",
        _PLANTED_NUMBERS: "from threetears.evals.analysis import tier_sentence\n",
    }
    root = _synthetic_tree(tmp_path, files)
    assert laundered_presentation(root) == [
        "threetears.evals.analysis.stats -> threetears.evals.analysis -> threetears.evals.analysis.report.forms"
    ]


def test_every_pair_is_a_finding_and_none_is_exempt() -> None:
    planted = ("analysis.stats", "analysis.report.forms")
    assert pair_findings({}) == []
    (finding,) = pair_findings({planted: 7})
    assert finding.startswith(
        "analysis.stats (line 7) imports analysis.report.forms: numbers never import presentation."
    )


def test_a_planted_pair_fails_the_gate_as_the_gate_reads_the_tree(tmp_path: Path) -> None:
    """The gate's own two calls, over a tree with one numbers module importing presentation: one finding, so the
    rule is held by what the tree imports and by no list."""
    files = {**_SYNTHETIC_BASE, _PLANTED_NUMBERS: "from threetears.evals.analysis.report.forms import PageForms\n"}
    root = _synthetic_tree(tmp_path, files)
    (finding,) = pair_findings(presentation_imports(root))
    assert finding.startswith("analysis.stats (line 1) imports analysis.report.forms: numbers never import")
