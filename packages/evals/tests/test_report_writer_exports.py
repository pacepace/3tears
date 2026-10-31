"""The report writer's modules export a name once production code outside them calls it, and not before.

``ops/report_writer/analyst.py`` exports what the writing loop calls and leaves ``InvariantVerdict`` out (the loop
names the checker, not its verdicts). ``actions/report_writing.py``, the report writer's entry point, leaves out the
writer's launch registry entry and what only it builds, whose caller is a host's registry. The writer kind's modules
(``writer_kind.py``, ``writer_contract.py``, ``writer_launch.py``) export what the kind's factory, the kind, its
launcher and its case bank call; the rest awaits a host's profile. The case bank (``writer_bank.py``) exports what the
launch reads and what the engine's writer actions call (the mint, the retirement and the template creation). The
promotion read (``writer_promotion.py``) exports the read its action calls, and the rubric (``rubric.py``) exports only
the read its callers make.
``CLAUDE.md`` holds that no public API exists only for a test, and the caller gate
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
import threetears.evals.ops.report_writer.writer_bank as writer_bank
import threetears.evals.ops.report_writer.writer_contract as writer_contract
import threetears.evals.ops.report_writer.writer_kind as writer_kind
import threetears.evals.ops.report_writer.writer_launch as writer_launch
import threetears.evals.ops.report_writer.writer_promotion as writer_promotion
import threetears.evals.ops.report_writer.rubric as rubric

#: Every workspace member's ``src`` tree.
_REPO = Path(__file__).resolve().parents[3]
_SOURCES = sorted([*_REPO.glob("packages/*/src"), *_REPO.glob("packages/agent/*/src")])

#: Per module, the public names it defines and leaves out of its ``__all__`` until a production caller imports them.
_AWAITING_A_CALLER: dict[ModuleType, frozenset[str]] = {
    analyst: frozenset({"InvariantVerdict"}),
    # A host's launch registry is the launcher's caller; the kind's factory is called only by the launcher, here.
    report_writing: frozenset({"launchable_report_writer_kind", "report_writer_kind", "write_report"}),
    # A host's profile declares the contract and the measures, outside this repo and not written yet.
    writer_contract: frozenset({"EffortWord", "WRITER_KIND_CONTRACT", "WRITER_KIND_MEASURES", "writer_kind_contract"}),
    writer_kind: frozenset({"PreparedWriter", "WRITER_CASE_KEY", "WRITER_CASE_VERSION", "log"}),
    writer_launch: frozenset({"log"}),
    # The engine's writer actions call the mint, the retirement and the template creation; the split's own reads and
    # the bank's type wait for a caller.
    writer_bank: frozenset(
        {"HELD_OUT_FRACTION", "WriterCaseBank", "WriterSplitSets", "log", "writer_group_side", "writer_split"}
    ),
    # The engine's writer_promotion action reads the decision; nothing names its literal type.
    writer_promotion: frozenset({"PromotionDecision"}),
    # The rubric's readers call shipped_rubric and never name its types.
    rubric: frozenset({"EditorRubric", "RubricItem"}),
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
