"""The package speaks one vocabulary in its identifiers: analyst, editor, writer, invariants, scorers.

The memo writer, the report planner, its planned page and the analysis reporter kind are retired. What survived them
(the page's parts, the chart vocabulary, the answer section's lines, the prose checks a report document runs) is named
for what it is now, so no identifier the package ships names a retired generation, except a name kept for one minor
so a host's import does not break. Held here, on identifiers only (prose is a Critic's read, never a word list):

- **no module path** under ``src/`` names one;
- **no top-level public name** any module defines names one, outside the modules that serve the retired names
  (each root's ``_retired``) and the one that says why they retired (``kernel/retired.py``);
- **no public name a root exports**, in the tree or in the checked-in snapshot (``api/surface.json``), names one,
  except a name its root serves for one minor with a ``DeprecationWarning`` (``kernel.retired.served_retired``);
- **no action** in the catalogue, and **no document type**, names one.

The stored and wire literals stay as they are (``"memo"``, ``"plan"``, ``"eval_report_plan"``, the ``"memo"`` report
section), and so do the words for things that were never the retired generations: memoisation, ``InMemory*``, launch
planning (``plan_judge``, ``PlannedCall``) and the case generators of ``threetears.evals.gen``.
"""

from __future__ import annotations

import ast
import importlib
import json
import re
import warnings
from pathlib import Path
from typing import Final

import pytest

from threetears.evals import PUBLIC_ROOTS
from threetears.evals.actions import eval_catalogue
from threetears.evals.schema.doc_types import DOC_TYPE_FACTS

#: A retired generation's word, as an identifier would hold it: never ``memory``, ``memoised`` or ``InMemory``.
RETIRED: Final = re.compile(r"memo(?![ri])|planner|plan_page|reporter", re.IGNORECASE)

PACKAGE: Final = Path(__file__).resolve().parents[1]
SOURCE: Final = PACKAGE / "src" / "threetears" / "evals"

#: The modules whose subject is the retired names themselves: each root's ``_retired`` serves them, and
#: ``kernel/retired.py`` says why they retired, so their own names may say what retired.
_SERVING: Final = re.compile(r"(^|/)_retired\.py$|^kernel/retired\.py$")


def _modules() -> list[Path]:
    return sorted(path for path in SOURCE.rglob("*.py") if "__pycache__" not in path.parts)


def _top_level_names(tree: ast.Module) -> list[str]:
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.append(node.name)
        elif isinstance(node, ast.Assign):
            names += [target.id for target in node.targets if isinstance(target, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.append(node.target.id)
        elif isinstance(node, ast.TypeAlias):
            names.append(node.name.id)
    return [name for name in names if not name.startswith("_")]


def _served_retired(root: str, name: str) -> bool:
    """Whether ``root`` serves ``name`` as a retired name: not bound on the module, and read with a deprecation."""
    module = importlib.import_module(root)
    if name in vars(module):
        return False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        getattr(module, name)
    return any(issubclass(warning.category, DeprecationWarning) for warning in caught)


def test_the_pattern_reads_the_retired_words_and_none_of_their_neighbours() -> None:
    for retired in ("ReporterKind", "memo_blocks", "plan_page", "report_planner", "MEMO_WRITER"):
        assert RETIRED.search(retired), retired
    for kept in ("InMemoryDocumentStore", "memory", "memoised", "plan_judge", "PlannedCall", "generate_variations"):
        assert not RETIRED.search(kept), kept


def test_no_module_path_under_src_names_a_retired_generation() -> None:
    named = [str(path.relative_to(SOURCE)) for path in _modules() if RETIRED.search(str(path.relative_to(SOURCE)))]
    assert named == []


def test_no_top_level_public_name_outside_the_serving_modules_names_a_retired_generation() -> None:
    named = [
        f"{path.relative_to(SOURCE)}:{name}"
        for path in _modules()
        if not _SERVING.search(path.relative_to(SOURCE).as_posix())
        for name in _top_level_names(ast.parse(path.read_text(encoding="utf-8")))
        if RETIRED.search(name)
    ]
    assert named == []


@pytest.mark.parametrize("root", PUBLIC_ROOTS)
def test_a_root_exports_a_retired_generation_s_name_only_as_a_deprecated_one(root: str) -> None:
    exported: list[str] = list(getattr(importlib.import_module(root), "__all__", []))
    named = [name for name in exported if RETIRED.search(name) and not _served_retired(root, name)]
    assert named == []


def test_the_snapshot_records_a_retired_generation_s_name_only_as_a_deprecated_one() -> None:
    names: dict[str, object] = json.loads((PACKAGE / "api" / "surface.json").read_text(encoding="utf-8"))["names"]
    named = [
        qualified
        for qualified in names
        if RETIRED.search(qualified.rsplit(".", 1)[1]) and not _served_retired(*qualified.rsplit(".", 1))
    ]
    assert named == []


def test_no_action_and_no_document_type_names_a_retired_generation() -> None:
    actions = [action.name for action in eval_catalogue().actions if RETIRED.search(action.name)]
    doc_types = [doc_type for doc_type in DOC_TYPE_FACTS if RETIRED.search(doc_type)]
    assert (actions, doc_types) == ([], [])
