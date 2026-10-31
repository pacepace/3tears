"""The claims renderer's entry points are exported once production code calls them, and not before.

``analysis/report/doc_page``'s builder, its three writers and the page text are this build's report-document page.
The builder and the page text are exported (:data:`_CALLED`): the store gate, the one checker, builds a draft's page
strict and checks its text above the appendix, the analyst's ``report_check`` (``ops/analyst.py``) returns that page's
text, the writing loop's editor reads it and the report writer builds the page it stores. The HTML and Markdown writers
are the report read's (``report_read``, through the campaign page's serializers). ``CLAUDE.md`` holds that no public API exists only
for a test, and the caller gate (``tests/enforcement/test_public_api_has_a_caller.py``) reads a module's surface from
its ``__all__``, so each module's ``__all__`` stays empty until its caller lands. This test keeps that checkable, as
``test_report_doc_exports.py`` does for the document's own names: :data:`_AWAITING_A_CALLER` lists each entry point
left out, and goes red the moment a ``src`` module outside the package imports one, so the change that gives it a
caller also adds it to its module's ``__all__`` and takes it off this list.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

#: Each entry point awaiting a production caller, by the module that defines it.
_AWAITING_A_CALLER: dict[str, frozenset[str]] = {}

#: Each entry point exported with its production caller, by the module that defines it.
_CALLED: dict[str, frozenset[str]] = {
    "threetears.evals.analysis.report.doc_page.build": frozenset({"build"}),
    "threetears.evals.analysis.report.doc_page.html": frozenset({"doc_page_html"}),
    "threetears.evals.analysis.report.doc_page.markdown": frozenset({"doc_page_markdown"}),
    "threetears.evals.analysis.report.doc_page.text": frozenset({"above_appendix_text", "doc_page_text"}),
}

#: Every workspace member's ``src`` tree.
_REPO = Path(__file__).resolve().parents[3]
_SOURCES = sorted([*_REPO.glob("packages/*/src"), *_REPO.glob("packages/agent/*/src")])

#: The package's own directory: a caller is a module outside it.
_PACKAGE = "threetears/evals/analysis/report/doc_page/"


def _defined_functions(module: str) -> set[str]:
    path = Path(str(importlib.import_module(module).__file__))
    return {
        node.name
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    }


def test_each_module_exports_exactly_its_called_entry_points_and_defines_no_others() -> None:
    """A module's public functions are its entry points: exported once called, listed as awaiting until then."""
    for module in sorted({*_AWAITING_A_CALLER, *_CALLED}):
        exported = _CALLED.get(module, frozenset())
        assert set(importlib.import_module(module).__all__) == exported
        assert _defined_functions(module) == exported | _AWAITING_A_CALLER.get(module, frozenset())


def test_no_entry_point_has_a_production_caller_yet() -> None:
    """A ``src`` module outside the package importing a listed entry point gives it its caller: add it to its
    module's ``__all__`` and take it off this list."""
    called = sorted(
        f"{path.relative_to(_REPO)}: {node.module}.{alias.name}"
        for root in _SOURCES
        for path in root.rglob("*.py")
        if _PACKAGE not in path.as_posix()
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.ImportFrom) and node.module in _AWAITING_A_CALLER
        for alias in node.names
        if alias.name in _AWAITING_A_CALLER[node.module]
    )
    assert not called, "these entry points now have a production caller:\n  " + "\n  ".join(called)
