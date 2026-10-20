"""enforcement walker: a collection that forces ``l3_pool`` off says whether L2 is its store.

``BaseCollection.l2_is_the_store`` decides what an L2 failure means. ``False`` (the default) treats
it as a cache miss: a refused write is logged and the save succeeds, a failed read answers "absent".
That is right when a durable tier stands behind L2 and wrong when nothing does -- then the save
reported a row no store holds, and the read reported a live row gone.

A collection that writes ``self.l3_pool = None`` has removed the one tier the default leans on, so
inheriting the default silently is exactly the defect this gate exists to stop. **The rule:** such
a class sets ``l2_is_the_store`` in its own body, ``True`` or ``False``, as a class attribute. A
collection whose L2 entries are derived (recomputed or rebuilt on a miss) legitimately says
``False``; it has to say it.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final

from threetears.enforcement.common import find_local_src_roots, iter_python_files, parse_python_file

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]


def _forces_l3_off(cls: ast.ClassDef) -> bool:
    for node in ast.walk(cls):
        if not isinstance(node, ast.Assign):
            continue
        sets_pool = any(
            isinstance(target, ast.Attribute)
            and target.attr == "l3_pool"
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
            for target in node.targets
        )
        if sets_pool and isinstance(node.value, ast.Constant) and node.value.value is None:
            return True
    return False


def _declares_the_store(cls: ast.ClassDef) -> bool:
    for statement in cls.body:
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            if statement.target.id == "l2_is_the_store" and statement.value is not None:
                return True
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "l2_is_the_store" for target in statement.targets
        ):
            return True
    return False


def find_undeclared(tree: ast.Module, label: str) -> list[str]:
    """every class in ``tree`` that forces ``l3_pool`` to ``None`` without declaring ``l2_is_the_store``.

    :param tree: a parsed module
    :ptype tree: ast.Module
    :param label: the module's path, for the message
    :ptype label: str
    :return: one ``path:line Class`` entry per offender
    :rtype: list[str]
    """
    return [
        f"{label}:{node.lineno} {node.name}"
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and _forces_l3_off(node) and not _declares_the_store(node)
    ]


def test_every_collection_that_forces_l3_off_declares_whether_l2_is_its_store() -> None:
    offenders: list[str] = []
    for root in find_local_src_roots(_REPO_ROOT):
        for source in iter_python_files(root):
            tree = parse_python_file(source)
            if tree is not None:
                offenders.extend(find_undeclared(tree, str(source.relative_to(_REPO_ROOT))))
    assert not offenders, (
        "these collections set self.l3_pool = None but inherit l2_is_the_store; set it in the class "
        "body (True where L2 is the only store, False where its entries are derived):\n  " + "\n  ".join(offenders)
    )


def test_the_walker_catches_an_undeclared_class_and_accepts_a_declared_one() -> None:
    undeclared = ast.parse(
        "class A(BaseCollection):\n    def __init__(self):\n        super().__init__()\n        self.l3_pool = None\n"
    )
    declared = ast.parse(
        "class B(BaseCollection):\n    l2_is_the_store: bool = True\n"
        "    def __init__(self):\n        self.l3_pool = None\n"
    )
    assert find_undeclared(undeclared, "a.py") == ["a.py:1 A"]
    assert find_undeclared(declared, "b.py") == []
