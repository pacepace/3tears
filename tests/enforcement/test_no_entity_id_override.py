"""
enforcement: no entity may hand-write its own ``_id`` derivation.

``BaseEntity.__init__`` derives ``_id`` -- the key every tier addresses a
row by -- from the owning collection's declared ``primary_key_columns``:
the scalar pk value on a single-pk table, the declared-order tuple on a
composite-pk one. Before that derivation existed, every composite-pk
entity carried its own::

    def __init__(self, data, is_new=True, collection=None):
        super().__init__(data, is_new=is_new, collection=collection)
        object.__setattr__(self, "_id", (data["customer_id"], data["id"]))

Twelve copies of that block existed across this repo and the hub, and the
copy is not a style problem. It is a second, independent statement of the
table's key shape sitting next to the collection's declaration, free to
disagree with it. When it does, the disagreement is silent and
asymmetric: L3 keeps addressing the right row (its SQL is generated from
the declaration) while L1 and L2 address the wrong one, so a stale or
cross-tenant read comes back from cache with no error anywhere. That is
the failure the composite key was adopted to make structurally
impossible, reintroduced one entity at a time.

So the declaration is the only place a key shape may be stated. An entity
that needs a different addressing key needs its collection's
``primary_key_column`` changed, not an override.

Static AST parsing only (no imports, no execution), consistent with the
rest of ``tests/enforcement``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
#: shipped source AND tests. a test fixture reintroducing the override is
#: not harmless: the two suites best placed to catch an addressing
#: regression (``test_cache_coherence``, ``test_composite_pk_three_tier``)
#: each carried one, so both were exercising the retired path rather than
#: the derivation they existed to cover.
_PACKAGE_GLOBS = (
    "packages/*/src",
    "packages/agent/*/src",
    "packages/*/tests",
    "packages/agent/*/tests",
    "tests",
)

#: the banned attribute names. ``_id`` is the addressing key; ``_row_id``
#: backs ``.id``. both are statements of key shape and both are the
#: framework's to set.
_BANNED_ATTRS = frozenset({"_id", "_row_id"})

#: the one module allowed to assign them: it owns the derivation every
#: other entity inherits.
_DERIVATION_OWNER = Path("packages/core/src/threetears/core/entities/base.py")


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


def _is_setattr_call(node: ast.Call) -> bool:
    """report whether ``node`` is a ``setattr``-shaped call.

    covers ``object.__setattr__(self, "_id", v)``, the builtin
    ``setattr(self, "_id", v)``, and ``SomeBase.__setattr__(...)``. all
    three reach the same attribute, and the first review of this guard
    found it caught only the first.

    :param node: call node under inspection
    :ptype node: ast.Call
    :return: whether the callee sets an attribute by name
    :rtype: bool
    """
    func = node.func
    if isinstance(func, ast.Name) and func.id == "setattr":
        return True
    return isinstance(func, ast.Attribute) and func.attr == "__setattr__"


def _id_assignments(tree: ast.Module) -> list[int]:
    """return line numbers assigning a banned identity attribute.

    four spellings reach the attribute and all of them work, because
    ``_id`` / ``_row_id`` are in ``BaseEntity._INTERNAL_ATTRS`` and so
    route straight to ``object.__setattr__``:

    - ``object.__setattr__(self, "_id", v)`` / ``setattr(self, "_id", v)``
    - ``self._id = v`` -- the most natural one, and the one the first
      version of this guard missed entirely
    - ``self.__dict__["_id"] = v``

    a non-literal name (``key = "_id"; setattr(self, key, v)``) is not
    caught and cannot be without dataflow analysis; it is also not a
    spelling anyone reaches for by accident.

    :param tree: parsed module
    :ptype tree: ast.Module
    :return: line numbers of offending statements
    :rtype: list[int]
    """
    lines: list[int] = []
    for node in ast.walk(tree):
        # setattr(self, "_id", ...) / object.__setattr__(self, "_id", ...)
        if isinstance(node, ast.Call) and _is_setattr_call(node) and len(node.args) >= 2:
            target = node.args[1]
            if isinstance(target, ast.Constant) and target.value in _BANNED_ATTRS:
                lines.append(node.lineno)
            continue
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            # self._id = ...
            if isinstance(tgt, ast.Attribute) and tgt.attr in _BANNED_ATTRS:
                lines.append(node.lineno)
            # self.__dict__["_id"] = ...
            elif (
                isinstance(tgt, ast.Subscript)
                and isinstance(tgt.value, ast.Attribute)
                and tgt.value.attr == "__dict__"
                and isinstance(tgt.slice, ast.Constant)
                and tgt.slice.value in _BANNED_ATTRS
            ):
                lines.append(node.lineno)
    return sorted(set(lines))


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: str(p.relative_to(_REPO_ROOT)))
def test_no_entity_writes_its_own_id(path: Path) -> None:
    """no shipped module assigns ``_id`` outside the derivation owner.

    :param path: module under inspection
    :ptype path: Path
    :return: nothing
    :rtype: None
    """
    relative = path.relative_to(_REPO_ROOT)
    if relative == _DERIVATION_OWNER:
        pytest.skip("owns the derivation every entity inherits")

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders = _id_assignments(tree)

    assert not offenders, (
        f"{relative} assigns a banned identity attribute (_id / _row_id) at line(s) {offenders}. "
        "``BaseEntity`` derives it from the collection's declared "
        "``primary_key_columns``; an override is a second statement of the "
        "key shape that can silently disagree with the declaration, giving "
        "correct L3 addressing and wrong L1/L2 addressing. Change the "
        "collection's ``primary_key_column`` instead."
    )
