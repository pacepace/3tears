"""A scoped snapshot rebuilds every scope from L3 only when NATS holds no index (a first start, or every pointer lost
with its bucket). Anything else that makes one scope's copy in L2 unusable is that scope's: it alone is read from L3.

Cobalt prod, 2026-11-05: one chunk holding another count than its pointer made every new pod rebuild all 52 states
whole from L3, about four hours each. This pins, by the code's own shape, that the whole rebuild
(``_rebuild_scopes(None, ...)``) is reached from ``_rebuild`` alone, and ``_rebuild`` from ``_sync``'s
``index is None`` branch alone.
"""

from __future__ import annotations

import ast
from pathlib import Path

_SOURCE = Path(__file__).parents[1] / "src" / "threetears" / "core" / "collections" / "scoped_snapshot.py"


def _calls(tree: ast.AST, method: str) -> list[tuple[ast.Call, list[ast.AST]]]:
    """every ``self.<method>(...)`` call, with the nodes enclosing it, outermost first."""
    found: list[tuple[ast.Call, list[ast.AST]]] = []

    def walk(node: ast.AST, above: list[ast.AST]) -> None:
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == method
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
        ):
            found.append((node, above))
        for child in ast.iter_child_nodes(node):
            walk(child, [*above, node])

    walk(tree, [])
    return found


def _function(above: list[ast.AST]) -> str:
    return next(n.name for n in reversed(above) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef))


def _is_index_none(test: ast.expr) -> bool:
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "index"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Is)
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value is None
    )


def test_a_whole_rebuild_of_every_scope_is_reached_only_from_rebuild() -> None:
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
    whole = [
        (call, above)
        for call, above in _calls(tree, "_rebuild_scopes")
        if call.args and isinstance(call.args[0], ast.Constant) and call.args[0].value is None
    ]
    assert whole, "no whole rebuild found: the shape this test reads has changed"
    callers = {_function(above) for _, above in whole}
    assert callers == {"_rebuild"}, f"every scope is rebuilt from L3 outside _rebuild: {sorted(callers)}"


def test_the_whole_rebuild_runs_only_when_nats_holds_no_index() -> None:
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
    calls = _calls(tree, "_rebuild")
    assert calls, "no call of _rebuild found: the shape this test reads has changed"
    for call, above in calls:
        assert _function(above) == "_sync", f"_rebuild is called from {_function(above)}, not _sync's empty index"
        enclosing = [n for n in above if isinstance(n, ast.If)]
        assert enclosing and _is_index_none(enclosing[-1].test), (
            "_rebuild is reached other than under `if index is None:` (a whole rebuild while NATS holds an index)"
        )
        assert any(call in ast.walk(stmt) for stmt in enclosing[-1].body), (
            "_rebuild sits in the else of `if index is None:`"
        )
