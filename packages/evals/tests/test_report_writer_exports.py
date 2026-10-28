"""The report writer's modules export a name once production code outside them calls it, and not before.

``ops/report_writer/analyst.py`` exports what the writing loop calls and leaves ``InvariantVerdict`` out (the loop
names the checker, not its verdicts). ``actions/report_writing.py``, the report writer's entry point, exports nothing
yet: its caller is the analysis job of plan §9 (W4), which Discodon's factory runs; the dev command line that writes a
report today is a test fixture, not production code. ``CLAUDE.md`` holds that no public API exists only for a test, and the caller gate
(``tests/enforcement/test_public_api_has_a_caller.py``) reads a module's surface from its ``__all__``, so it cannot
see the names left out. This test keeps the promise checkable: :data:`_AWAITING_A_CALLER` lists every public name
left out, and goes red the moment another ``src`` module imports one, so the change that gives it a caller also adds
it to its module's ``__all__`` and takes it off this list.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import ModuleType

import pytest

import threetears.evals.actions.report_writing as report_writing
import threetears.evals.ops.report_writer.analyst as analyst

#: Every workspace member's ``src`` tree.
_REPO = Path(__file__).resolve().parents[3]
_SOURCES = sorted([*_REPO.glob("packages/*/src"), *_REPO.glob("packages/agent/*/src")])

#: Per module, the public names it defines and leaves out of its ``__all__`` until a production caller imports them.
_AWAITING_A_CALLER: dict[ModuleType, frozenset[str]] = {
    analyst: frozenset({"InvariantVerdict"}),
    report_writing: frozenset({"ReportNotStored", "ReportWritten", "write_report"}),
}


def _defined_public_names(module: ModuleType) -> set[str]:
    """Every public name the module defines at its top level: classes, functions and assigned names."""
    names: set[str] = set()
    for node in ast.parse(Path(str(module.__file__)).read_text()).body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names |= {target.id for target in node.targets if isinstance(target, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return {name for name in names if not name.startswith("_")}


def _imports_from(path: Path, module: ModuleType) -> set[str]:
    """The names a source file takes from the module, by ``from ... import`` or as attributes of an alias."""
    tree = ast.parse(path.read_text())
    package, _, leaf = module.__name__.rpartition(".")
    aliases = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == module.__name__ and alias.asname
    } | {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == package
        for alias in node.names
        if alias.name == leaf
    }
    taken = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == module.__name__
        for alias in node.names
    }
    taken |= {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in aliases
    }
    return taken


@pytest.mark.parametrize("module", list(_AWAITING_A_CALLER), ids=lambda module: module.__name__.rpartition(".")[2])
def test_every_public_name_left_out_of_all_is_listed_as_awaiting_a_caller(module: ModuleType) -> None:
    """A public name added to the module joins ``__all__`` with its caller, or this list until it has one."""
    assert _defined_public_names(module) - set(module.__all__) == _AWAITING_A_CALLER[module]


@pytest.mark.parametrize("module", list(_AWAITING_A_CALLER), ids=lambda module: module.__name__.rpartition(".")[2])
def test_no_name_awaiting_a_caller_has_one_yet(module: ModuleType) -> None:
    """A ``src`` module importing a listed name gives it its caller: add it to ``__all__`` and take it off the list."""
    own = Path(str(module.__file__)).resolve()
    called = {
        f"{path.relative_to(_REPO)}: {name}"
        for root in _SOURCES
        for path in root.rglob("*.py")
        if path.resolve() != own
        for name in _imports_from(path, module) & _AWAITING_A_CALLER[module]
    }
    assert not called, (
        f"these names now have a production caller: add each to {module.__name__}'s __all__ and take it off "
        "_AWAITING_A_CALLER here:\n  " + "\n  ".join(sorted(called))
    )
