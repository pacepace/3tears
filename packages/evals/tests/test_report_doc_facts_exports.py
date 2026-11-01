"""``analysis/report_doc_facts.py`` exports a name once production code calls it, and not before.

The module's ``__all__`` holds :func:`~threetears.evals.analysis.report_doc_facts.editor_facts`, whose caller is the
writing loop's editor turn (its user message carries the facts). :class:`~threetears.evals.analysis.report_doc_facts.EditorFact`
and :data:`~threetears.evals.analysis.report_doc_facts.EditorFactKind` wait for a production caller outside the module. ``CLAUDE.md`` holds that no public API exists
only for a test, and the caller gate (``tests/enforcement/test_public_api_has_a_caller.py``) reads a module's surface
from its ``__all__``, so it cannot see the names left out. This test keeps the promise checkable:
:data:`_AWAITING_A_CALLER` lists every public name left out, and goes red the moment a ``src`` module imports one, so
the change that gives it a caller also adds it to ``__all__`` and to the API snapshot as provisional, and takes it off
this list.
"""

from __future__ import annotations

import ast
from pathlib import Path

import threetears.evals.analysis.report_doc_facts as report_doc_facts

#: The module under test, and the import path its callers name.
_MODULE = Path(report_doc_facts.__file__)
_IMPORTED_AS = "threetears.evals.analysis.report_doc_facts"

#: Every workspace member's ``src`` tree.
_REPO = Path(__file__).resolve().parents[3]
_SOURCES = sorted([*_REPO.glob("packages/*/src"), *_REPO.glob("packages/agent/*/src")])

#: The public names defined in the module and left out of its ``__all__`` until a production caller imports them.
_AWAITING_A_CALLER = frozenset({"EditorFact", "EditorFactKind"})


def _defined_public_names() -> set[str]:
    """Every public name the module defines at its top level: classes, functions and assigned names."""
    names: set[str] = set()
    for node in ast.parse(_MODULE.read_text()).body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names |= {target.id for target in node.targets if isinstance(target, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return {name for name in names if not name.startswith("_")}


def _imports_from_the_module(path: Path) -> set[str]:
    """The names a source file takes from the module: ``from ... import`` of them, or attributes of the module."""
    tree = ast.parse(path.read_text())
    aliases = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == _IMPORTED_AS and alias.asname
    } | {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "threetears.evals.analysis"
        for alias in node.names
        if alias.name == "report_doc_facts"
    }
    taken = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == _IMPORTED_AS
        for alias in node.names
    }
    taken |= {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in aliases
    }
    return taken


def test_every_public_name_left_out_of_all_is_listed_as_awaiting_a_caller() -> None:
    """A public name added to the module joins ``__all__`` with its caller, or this list until it has one."""
    assert _defined_public_names() - set(report_doc_facts.__all__) == _AWAITING_A_CALLER


def test_no_name_awaiting_a_caller_has_one_yet() -> None:
    """A ``src`` module importing a listed name gives it its caller: add it to ``__all__`` and the API snapshot."""
    called = {
        f"{path.relative_to(_REPO)}: {name}"
        for root in _SOURCES
        for path in root.rglob("*.py")
        if path.resolve() != _MODULE.resolve()
        for name in _imports_from_the_module(path) & _AWAITING_A_CALLER
    }
    assert not called, (
        "these names now have a production caller: add each to analysis/report_doc_facts.py's __all__, regenerate "
        "the API snapshot (it lands provisional), and take it off _AWAITING_A_CALLER here:\n  "
        + "\n  ".join(sorted(called))
    )
