"""where a column-coverage aggregate is read: only by a datasource driver, over the relation its datasource names.

``_build_coverage_sql`` and ``_build_coverage_by_dimension_sql`` count every row of a relation (grouped by a
dimension, for the second), in one statement the warehouse answers in a row (a row a dimension value); each declares ``whole_read(kind="unbounded_by_warehouse")`` (Pace's
2026-11-12 kind, for a client's warehouse relation, which keeps no change record of ours). This holds them to that: each
builder is called from one driver method alone (``Driver.column_value_coverage``,
``Driver.column_value_coverage_by_dimension``), which reads through the driver's own ``fetch`` (the datasource's
connection), so no 3tears code aggregates one of our own L3 tables with them.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from threetears.datasources.drivers.base import Driver

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
    """``_build_coverage_sql`` is called only by ``Driver.column_value_coverage`` and
    ``_build_coverage_by_dimension_sql`` only by ``Driver.column_value_coverage_by_dimension``, each of which sends it
    through the driver's own ``fetch`` (its datasource's connection)."""
    builders = {
        "_build_coverage_sql": "column_value_coverage",
        "_build_coverage_by_dimension_sql": "column_value_coverage_by_dimension",
    }
    source = (_REPO / _BASE).read_text(encoding="utf-8")
    for builder, method_name in builders.items():
        built = {}
        for path in sorted(_REPO.glob("packages/*/src/**/*.py")):
            functions = _calls(path, builder)
            if functions:
                built[path.relative_to(_REPO).as_posix()] = functions
        assert built == {_BASE: [f"Driver.{method_name}"]}, builder
        method = next(
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.AsyncFunctionDef) and node.name == method_name
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
        assert len(fetched) == 1, method_name


class _Fetching:
    """a driver's ``fetch``, recording the statement the coverage read sends."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def fetch(self, sql: str) -> list[dict[str, Any]]:
        self.sent.append(sql)
        return []


async def test_the_coverage_read_names_the_relation_it_is_given() -> None:
    """Flips if either coverage read reads anything but the datasource relation it is named (grouped by the dimension,
    for the per-dimension one)."""
    driver = _Fetching()
    await Driver.column_value_coverage(driver, "sales", "orders", ["amount"])  # type: ignore[arg-type]
    await Driver.column_value_coverage_by_dimension(driver, "sales", "orders", "state", ["amount"])  # type: ignore[arg-type]
    assert driver.sent == [
        'SELECT COUNT(*) AS total_rows, COUNT(NULLIF("amount", 0)) AS nz_0 FROM "sales"."orders"',
        'SELECT "state" AS dim_value, COUNT(*) AS total_rows, COUNT(NULLIF("amount", 0)) AS nz_0 '
        'FROM "sales"."orders" GROUP BY "state"',
    ]
