"""
enforcement: a collection's declared pk arity must match how it unpacks ids.

``BaseCollection.primary_key_column`` is the SINGLE declaration of a table's
key shape. Everything downstream reads it: ``normalize_pk`` validates arity
against it, ``l2_key`` composes the cache key from it, ``BaseEntity`` derives
its addressing ``_id`` from it, and the L1 backends build their WHERE clauses
from it.

A collection can nonetheless contradict its own declaration, because
``fetch_from_store`` / ``delete_from_store`` are hand-written SQL and nothing
forced them to agree. ``TableTemplateCollection`` did exactly that: its
docstring said "composite PK ``(customer_id, id)``", its ``fetch_from_store``
opened with ``customer_id, template_id = entity_id``, and the class declared
no ``primary_key_column`` at all -- so it inherited the scalar ``"id"``.

That inconsistency was invisible for as long as its ENTITY hand-wrote a
composite ``_id`` override, because the override supplied the tuple the SQL
wanted while the declaration was never consulted for anything that failed
loudly. Delete the override -- which is correct, the declaration is supposed
to be the only statement of key shape -- and the collection addresses rows by
the bare id: L3 raises ``TypeError`` on unpacking, and L2 silently caches the
row under a key one component short.

So the declaration and the unpacking are checked against each other here. A
``fetch_from_store`` that unpacks ``entity_id`` into N names is a statement
that the key has N parts, and it must agree with ``primary_key_column``.

Static AST parsing only (no imports, no execution), consistent with the rest
of ``tests/enforcement``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_GLOBS = ("packages/*/src", "packages/agent/*/src")
_UNPACKING_METHODS = ("fetch_from_store", "delete_from_store")


def _source_files() -> list[Path]:
    """return every shipped python module across the workspace packages.

    :return: module paths, sorted for stable test ids
    :rtype: list[Path]
    """
    found: list[Path] = []
    for glob in _PACKAGE_GLOBS:
        for src_dir in sorted(_REPO_ROOT.glob(glob)):
            found.extend(sorted(src_dir.rglob("*.py")))
    return found


def _declared_pk_arity(cls: ast.ClassDef) -> int | None:
    """return the arity of a class's ``primary_key_column`` declaration.

    :param cls: class node under inspection
    :ptype cls: ast.ClassDef
    :return: number of pk columns declared, or ``None`` when the class
        makes no declaration of its own
    :rtype: int | None
    """
    for node in cls.body:
        target = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target = node.target.id
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target = node.targets[0].id
        if target != "primary_key_column":
            continue
        value = node.value
        if isinstance(value, ast.Tuple):
            return len(value.elts)
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return 1
    return None


def _unpack_arity(cls: ast.ClassDef) -> dict[str, int]:
    """return, per method, how many names ``entity_id`` is unpacked into.

    only a direct ``a, b = entity_id`` statement counts -- the shape that
    silently disagrees with a scalar declaration.

    :param cls: class node under inspection
    :ptype cls: ast.ClassDef
    :return: mapping of method name to unpack arity
    :rtype: dict[str, int]
    """
    found: dict[str, int] = {}
    for node in cls.body:
        if not isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef):
            continue
        if node.name not in _UNPACKING_METHODS:
            continue
        for stmt in ast.walk(node):
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                continue
            target = stmt.targets[0]
            if not isinstance(target, ast.Tuple):
                continue
            if isinstance(stmt.value, ast.Name) and stmt.value.id == "entity_id":
                found[node.name] = len(target.elts)
    return found


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: str(p.relative_to(_REPO_ROOT)))
def test_declared_pk_arity_matches_id_unpacking(path: Path) -> None:
    """every collection unpacking ``entity_id`` declares a matching pk arity.

    :param path: module under inspection
    :ptype path: Path
    :return: nothing
    :rtype: None
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        unpacked = _unpack_arity(node)
        if not unpacked:
            continue
        declared = _declared_pk_arity(node)
        for method, arity in unpacked.items():
            if declared is None:
                violations.append(
                    f"{node.name}.{method} unpacks entity_id into {arity} names but the class "
                    f"declares no primary_key_column, so it inherits the scalar default"
                )
            elif declared != arity:
                violations.append(
                    f"{node.name}.{method} unpacks entity_id into {arity} names but "
                    f"primary_key_column declares {declared}"
                )

    assert not violations, (
        f"{path.relative_to(_REPO_ROOT)} contradicts its own key declaration:\n"
        + "\n".join(f"  {line}" for line in violations)
        + "\n\nThe declaration is the only place a key shape may be stated. Fix "
        "primary_key_column; do NOT reintroduce an entity-side _id override to "
        "paper over it."
    )
