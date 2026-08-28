"""enforcement: nothing in the memory package writes ``platform.namespaces``.

The memory-namespace create moved to the hub so that no AGENT process writes the
platform control plane -- an agent able to insert a namespace row chooses what
the control plane says about its own storage and its own ownership, and the L3
broker's platform-write gate stayed open on ``namespace_type='memory'`` purely
to permit the call this package used to make.

The regression this guards is specific and tempting: an absent namespace is a
denial now, and the obvious "fix" for a denial is to build the row locally
again. That would restore the writer and silently re-open the gate's reason to
exist, with every behavioural test still green -- the namespace would resolve,
authorization would proceed, and only the deployment posture would have changed.
A behavioural test cannot see that; a structural one can.

Static AST parsing: no import, no install.
"""

from __future__ import annotations

import ast
from pathlib import Path

#: mutating Collection methods. a namespace row can only be created, updated or
#: removed through one of these, so naming them is naming the whole write
#: surface rather than one spelling of it.
_WRITE_METHODS = frozenset({"save_entity", "save", "delete", "delete_entity", "create"})

#: attribute names by which this package reaches the namespaces Collection. the
#: bundle field, the parameter it is threaded through, and the private handle a
#: future collaborator might store it under.
_NAMESPACE_HANDLES = frozenset({"namespace_collection", "_namespace_collection", "namespaces", "_namespaces"})

_MEMORY_SRC = Path(__file__).resolve().parents[1] / "src" / "threetears" / "agent" / "memory"


def _receiver_name(node: ast.expr) -> str | None:
    """name the object a method is being called on, when it has a simple name.

    handles ``namespace_collection.save_entity(...)`` and
    ``deps.namespace_collection.save_entity(...)`` alike, since only the last
    segment before the method identifies the handle.

    :param node: the call's receiver expression
    :ptype node: ast.expr
    :return: receiver name, or ``None`` when it is not a simple name/attribute
    :rtype: str | None
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _namespace_writes(path: Path) -> list[str]:
    """collect every ``<namespace handle>.<write method>(...)`` call in a module.

    :param path: module to scan
    :ptype path: Path
    :return: ``file:line: expression`` per violation
    :rtype: list[str]
    """
    tree = ast.parse(path.read_text())
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in _WRITE_METHODS:
            continue
        receiver = _receiver_name(node.func.value)
        if receiver in _NAMESPACE_HANDLES:
            found.append(f"{path.name}:{node.lineno}: {receiver}.{node.func.attr}(...)")
    return found


class TestMemoryPackageNeverWritesNamespaces:
    def test_no_module_writes_a_namespace_row(self) -> None:
        """no module under the memory package mutates the namespaces Collection."""
        violations: list[str] = []
        for path in sorted(_MEMORY_SRC.rglob("*.py")):
            violations.extend(_namespace_writes(path))
        assert violations == [], (
            "the memory package must not write platform.namespaces; the hub owns that "
            f"create (Subjects.hub_memory_namespace_ensure). found: {violations}"
        )

    def test_the_walker_sees_a_write_when_one_exists(self) -> None:
        """self-check: an empty result must mean "none", not "cannot see any".

        without this the test above passes just as happily against a walker that
        matches nothing at all, which is the failure mode that makes a
        structural gate worthless.
        """
        probe = _MEMORY_SRC.parent / "__enforcement_probe__.py"
        probe.write_text("async def f(deps):\n    await deps.namespace_collection.save_entity(x)\n")
        try:
            found = _namespace_writes(probe)
        finally:
            probe.unlink()
        assert len(found) == 1
        assert "namespace_collection.save_entity" in found[0]
