"""
enforcement: a composite-keyed collection is addressed with a full key.

``test_collection_pk_arity_matches_unpacking`` guards the DECLARATION side --
a collection may not contradict its own ``primary_key_column``. This one
guards the CALL side, which is where the same defect has now surfaced four
separate times:

1. ``ToolServer.deregister_tool`` deleted a ``namespaces`` row through
   ``NamespaceCollection.delete(namespace_id)`` -- a bare uuid against
   ``(row_scope, namespace_id)`` -- so every deregister raised and stale-tool
   pruning threw.
2. and 3. ``ensure_memory_owner_assignment`` read ``groups`` with
   ``get(group_id)`` against ``(row_scope, group_id)``, then ``group_members``
   with ``get(membership_id)`` against ``(group_id, id)``.
4. ``ensure_conversation_owner_assignment`` did both of the same.

``normalize_pk`` raises ``primary key arity mismatch`` on all of them, at the
FIRST statement that touches the collection -- so the whole rest of the
function had never executed in any process. What kept that invisible was
coverage: every test of those two functions supplied a duck-typed stand-in
whose ``get`` took one argument of any shape, which cannot tell a correct call
from a broken one. A gate that reads the declaration is the only thing that
can, because the declaration is the only place the arity is stated.

**Resolution is by ANNOTATION, so an ``Any``-typed collection is invisible to
this gate.** That is not a hole to work around at the call site -- it is the
finding. Both authorizer bundles typed their five rbac Collections ``Any`` on
a rationale that had gone stale, and that ``Any`` is precisely what let two
arity bugs sit in shipped code. Type the collection.

Static AST parsing only (no imports, no execution), consistent with the rest
of ``tests/enforcement``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_GLOBS = ("packages/*/src", "packages/agent/*/src")

#: methods whose FIRST positional argument is an entity id addressed through
#: ``BaseCollection.normalize_pk``. every one of them raises on an arity
#: mismatch, or -- for the sync cache accessors -- reads and writes a row at a
#: key one component short.
_ADDRESSING_METHODS = frozenset(
    {
        "delete",
        "ensure",
        "evict_from_cache_sync",
        "exists_in_cache_sync",
        "get",
        "get_field_sync",
        "get_row_sync",
        "set_field_sync",
    }
)


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
    """return the arity of a class's own ``primary_key_column`` declaration.

    :param cls: class node under inspection
    :ptype cls: ast.ClassDef
    :return: number of pk columns declared, or ``None`` when the class makes
        no declaration of its own
    :rtype: int | None
    """
    result: int | None = None
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
            result = len(value.elts)
        elif isinstance(value, ast.Constant) and isinstance(value.value, str):
            result = 1
    return result


def _bare_name(annotation: ast.expr | None) -> str | None:
    """return the terminal name of an annotation, ignoring subscripts.

    ``GroupCollection``, ``acl.GroupCollection`` and ``"GroupCollection"`` all
    reduce to ``GroupCollection``. an optional or unioned annotation is not
    reduced -- a name that may be ``None`` is still that collection.

    :param annotation: annotation node, or ``None`` when unannotated
    :ptype annotation: ast.expr | None
    :return: terminal class name, or ``None`` when there is no annotation
    :rtype: str | None
    """
    if annotation is None:
        return None
    text = ast.unparse(annotation).strip("'\"")
    text = text.replace("| None", "").strip()
    return text.split("[")[0].split(".")[-1]


def _pk_arities() -> dict[str, int]:
    """return ``{collection class name: pk arity}`` for composite-pk classes.

    a subclass that redeclares wins; one that does not inherits its base's
    arity, which is how ``HubGroupCollection`` is covered by
    ``GroupCollection``'s declaration.

    :return: mapping of class name to declared pk arity, composite only
    :rtype: dict[str, int]
    """
    declared: dict[str, int] = {}
    bases: dict[str, list[str]] = {}
    for path in _source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases[node.name] = [_bare_name(b) or "" for b in node.bases]
            arity = _declared_pk_arity(node)
            if arity is not None:
                declared[node.name] = arity
    resolved = dict(declared)
    changed = True
    while changed:
        changed = False
        for name, parents in bases.items():
            if name in resolved:
                continue
            for parent in parents:
                if parent in resolved:
                    resolved[name] = resolved[parent]
                    changed = True
                    break
    return {name: arity for name, arity in resolved.items() if arity > 1}


def _annotated_collections(tree: ast.AST, arities: dict[str, int]) -> dict[str, str]:
    """return ``{binding name: collection class}`` for annotated bindings.

    covers the three shapes a collection reaches a call site through: a
    function or method parameter, a ``self.x: T`` / ``x: T`` annotated
    assignment, and a class-body field declaration.

    :param tree: module syntax tree
    :ptype tree: ast.AST
    :param arities: composite-pk class arities
    :ptype arities: dict[str, int]
    :return: mapping of binding name to collection class name
    :rtype: dict[str, str]
    """
    found: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            cls = _bare_name(node.annotation)
            if cls in arities:
                found[node.arg] = cls
        elif isinstance(node, ast.AnnAssign):
            cls = _bare_name(node.annotation)
            if cls not in arities:
                continue
            if isinstance(node.target, ast.Name):
                found[node.target.id] = cls
            elif isinstance(node.target, ast.Attribute):
                found[node.target.attr] = cls
    return found


def _tuple_bindings(tree: ast.AST) -> dict[str, int]:
    """return ``{name: length}`` for every name bound to a tuple literal.

    a call site that composes its key on the line above -- ``pk = (scope,
    group_id)`` -- is passing a tuple, and reading the binding is what lets
    the gate say so without demanding the literal be inlined into the call.

    :param tree: module syntax tree
    :ptype tree: ast.AST
    :return: mapping of bound name to tuple length
    :rtype: dict[str, int]
    """
    found: dict[str, int] = {}
    for node in ast.walk(tree):
        value = None
        names: list[str] = []
        if isinstance(node, ast.Assign):
            value = node.value
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            value = node.value
            names = [node.target.id]
        if isinstance(value, ast.Tuple):
            for name in names:
                found[name] = len(value.elts)
    return found


def _receiver_name(node: ast.Attribute) -> str | None:
    """return the binding name a method call is made on.

    ``deps.group_collection.get(...)`` answers ``group_collection``;
    ``collection.get(...)`` answers ``collection``. anything more indirect
    (a subscript, a call result) answers ``None`` and is not judged.

    :param node: the ``Attribute`` node naming the called method
    :ptype node: ast.Attribute
    :return: receiver binding name, or ``None`` when it is not a plain name
    :rtype: str | None
    """
    receiver = node.value
    if isinstance(receiver, ast.Attribute):
        return receiver.attr
    if isinstance(receiver, ast.Name):
        return receiver.id
    return None


_ARITIES = _pk_arities()


def test_the_arity_table_is_not_empty() -> None:
    """the gate is non-vacuous: composite-pk collections were actually found.

    a walker that silently resolves nothing passes every file, which is the
    failure mode an enforcement test cannot detect from inside its own
    assertions.

    :return: nothing
    :rtype: None
    """
    assert "GroupCollection" in _ARITIES, "GroupCollection should resolve to a composite pk arity"
    assert _ARITIES["GroupCollection"] == 2
    assert "GroupMemberCollection" in _ARITIES, "GroupMemberCollection should resolve to a composite pk arity"
    assert _ARITIES["GroupMemberCollection"] == 2
    # a floor rather than the exact count: the point is to catch a walker that
    # resolved NOTHING, and pinning the exact number would fail on any package
    # split. 26 resolve today.
    assert len(_ARITIES) > 15, f"expected the workspace's composite-pk collections, found {len(_ARITIES)}"


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: str(p.relative_to(_REPO_ROOT)))
def test_composite_pk_collections_are_addressed_with_a_full_key(path: Path) -> None:
    """every annotated composite-pk collection is addressed with a tuple.

    :param path: module under inspection
    :ptype path: Path
    :return: nothing
    :rtype: None
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    bindings = _annotated_collections(tree, _ARITIES)
    tuples = _tuple_bindings(tree)
    violations: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in _ADDRESSING_METHODS or not node.args:
            continue
        receiver = _receiver_name(node.func)
        if receiver is None:
            continue
        cls = bindings.get(receiver)
        if cls is None:
            continue
        arity = _ARITIES[cls]
        argument = node.args[0]
        if isinstance(argument, ast.Tuple):
            supplied = len(argument.elts)
        elif isinstance(argument, ast.Name) and argument.id in tuples:
            supplied = tuples[argument.id]
        else:
            supplied = 1
        if supplied != arity:
            violations.append(
                f"line {node.lineno}: {receiver}.{node.func.attr}({ast.unparse(argument)}) addresses "
                f"{cls}, whose primary_key_column declares {arity} columns, with {supplied} value(s)"
            )

    assert not violations, (
        f"{path.relative_to(_REPO_ROOT)} addresses a composite-keyed collection with a partial key:\n"
        + "\n".join(f"  {line}" for line in violations)
        + "\n\nnormalize_pk raises 'primary key arity mismatch' on every one of these, at the first "
        "statement that touches the collection. Pass the full key in declared column order; for the "
        "row_scope-partitioned rbac tables, threetears.agent.acl.row_scope_for_customer states the "
        "partition rule so no call site has to restate it."
    )
