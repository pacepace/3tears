"""The report document's modules export a name once production code calls it, and not before.

``kernel/report_doc.py``'s ``__all__`` holds :class:`~threetears.evals.kernel.report_doc.EvalReportDoc`, whose caller
is the store, and the names its production callers read: the invariants (``analysis/report_doc_checks.py``), the
editor's facts (``analysis/report_doc_facts.py``), the claims renderer (``analysis/report/doc_page/build.py``) and the
analyst's session (``ops/report_writer``). Its other public names are this build's first ``report-doc/1`` surface,
waiting for their production callers. ``analysis/report_doc_checks.py`` exports ``check_report_doc``,
``DocVerdict`` and ``InvariantNumber``, whose first caller is the ``report_check`` action's operation
(``ops/analyst.py``). ``CLAUDE.md`` holds that no
public API exists only for a test, and the caller gate (``tests/enforcement/test_public_api_has_a_caller.py``) reads a
module's surface from its ``__all__``, so it cannot see the names left out. This test keeps the promise checkable:
:data:`_AWAITING_A_CALLER` lists every public name left out, per module, and goes red the moment a ``src`` module
imports one, so the change that gives it a caller also adds it to ``__all__`` (and, where a public root re-exports it,
to the API snapshot as provisional), and takes it off this list.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import ModuleType

import pytest

import threetears.evals.analysis.report_doc_checks as report_doc_checks
import threetears.evals.kernel.report_doc as report_doc

#: Every workspace member's ``src`` tree.
_REPO = Path(__file__).resolve().parents[3]
_SOURCES = sorted([*_REPO.glob("packages/*/src"), *_REPO.glob("packages/agent/*/src")])

#: Each module under test, to the public names it defines and leaves out of its ``__all__`` until a production caller
#: imports them.
_AWAITING_A_CALLER: dict[ModuleType, frozenset[str]] = {
    report_doc: frozenset(
        {
            "ANALYST_CAVEAT_PREFIX",
            "REPORT_DOC_SCHEMA",
            "Claim",
            "ClaimWeight",
            "ConfidenceForm",
            "DecisionEntry",
            "DocConfidence",
            "ExploreRow",
            "MethodItem",
            "NextStep",
            "Omitted",
            "Story",
        }
    ),
    report_doc_checks: frozenset(),
}


def _defined_public_names(module: ModuleType) -> set[str]:
    """Every public name the module defines at its top level: classes, functions and assigned names."""
    names: set[str] = set()
    for node in ast.parse(Path(str(module.__file__)).read_text()).body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names |= {target.id for target in node.targets if isinstance(target, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return {name for name in names if not name.startswith("_")}


def _imports_from_the_module(path: Path, imported_as: str) -> set[str]:
    """The names a source file takes from the module: ``from ... import`` of them, or attributes of the module."""
    tree = ast.parse(path.read_text())
    parent, _, leaf = imported_as.rpartition(".")
    aliases = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == imported_as and alias.asname
    } | {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == parent
        for alias in node.names
        if alias.name == leaf
    }
    taken = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == imported_as
        for alias in node.names
    }
    taken |= {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in aliases
    }
    return taken


_MODULES = pytest.mark.parametrize("module", list(_AWAITING_A_CALLER), ids=lambda module: module.__name__)


@_MODULES
def test_every_public_name_left_out_of_all_is_listed_as_awaiting_a_caller(module: ModuleType) -> None:
    """A public name added to the module joins ``__all__`` with its caller, or this list until it has one."""
    assert _defined_public_names(module) - set(module.__all__) == _AWAITING_A_CALLER[module]


@_MODULES
def test_no_name_awaiting_a_caller_has_one_yet(module: ModuleType) -> None:
    """A ``src`` module importing a listed name gives it its caller: add it to ``__all__`` and the API snapshot."""
    defined = Path(str(module.__file__)).resolve()
    awaiting = _AWAITING_A_CALLER[module]
    called = {
        f"{path.relative_to(_REPO)}: {name}"
        for root in _SOURCES
        for path in root.rglob("*.py")
        if path.resolve() != defined
        for name in _imports_from_the_module(path, module.__name__) & awaiting
    }
    assert not called, (
        f"these names now have a production caller: add each to {module.__name__}'s __all__ (and, where a public root "
        "re-exports it, regenerate the API snapshot, where it lands provisional), and take it off _AWAITING_A_CALLER "
        "here:\n  " + "\n  ".join(sorted(called))
    )
