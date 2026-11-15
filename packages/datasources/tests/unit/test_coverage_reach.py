"""where a column-coverage aggregate is read: only by a datasource driver, over the relation its datasource names.

``_build_coverage_by_dimension_sql`` counts every row of a relation grouped by a dimension, in one statement the
warehouse answers a row a dimension value; it declares that ``whole_read(kind="unbounded_by_warehouse")`` (Pace's
2026-11-12 kind, for a client's warehouse relation, which keeps no change record of ours). This holds it to that: the
builder is called from ``Driver.column_value_coverage_by_dimension`` alone, which reads through the driver's own
``fetch`` (the datasource's connection), so no 3tears code aggregates one of our own L3 tables with it.
"""

from __future__ import annotations

import ast
from pathlib import Path

from threetears.datasources.drivers.base import _build_coverage_by_dimension_sql

_REPO = Path(__file__).resolve().parents[4]
_BASE = "packages/datasources/src/threetears/datasources/drivers/base.py"


def _calls(path: Path, name: str) -> list[str]:
    """the qualified function each call of ``name`` in ``path`` is written in."""
    found: list[str] = []

    def walk(node: ast.AST, scope: tuple[str, ...]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                walk(child, (*scope, child.name))
                continue
            if isinstance(child, ast.Call):
                called = child.func.attr if isinstance(child.func, ast.Attribute) else getattr(child.func, "id", "")
                if called == name:
                    found.append(".".join(scope))
            walk(child, scope)

    walk(ast.parse(path.read_text(encoding="utf-8")), ())
    return found


def test_coverage_is_read_only_by_a_driver_over_its_datasources_relation() -> None:
    """``_build_coverage_by_dimension_sql`` is called only by ``Driver.column_value_coverage_by_dimension``, which
    sends it through the driver's own ``fetch`` (its datasource's connection), and it reads the relation it is
    named, quoted, grouped by the dimension."""
    built = {}
    for path in sorted(_REPO.glob("packages/*/src/**/*.py")):
        functions = _calls(path, "_build_coverage_by_dimension_sql")
        if functions:
            built[path.relative_to(_REPO).as_posix()] = functions
    assert built == {_BASE: ["Driver.column_value_coverage_by_dimension"]}
    source = (_REPO / _BASE).read_text(encoding="utf-8")
    method = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "column_value_coverage_by_dimension"
    )
    fetched = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "fetch"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
    ]
    assert len(fetched) == 1
    assert _build_coverage_by_dimension_sql("sales", "orders", "state", ["amount"]) == (
        'SELECT "state" AS dim_value, COUNT(*) AS total_rows, COUNT(NULLIF("amount", 0)) AS nz_0 '
        'FROM "sales"."orders" GROUP BY "state"'
    )
