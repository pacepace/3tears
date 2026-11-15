"""where a relation fingerprint is read: only in a datasource driver, over the warehouse relation its datasource names.

A fingerprint (``threetears.core.fingerprint``) counts a relation and sums its keys' hashes in one statement the
warehouse answers with one row. The drivers declare that read ``whole_read(kind="unbounded_by_warehouse")`` (coordinator
ruling 2026-11-14, under Pace's 2026-11-12 kind): a client's warehouse relation keeps no change record of ours, and our
speed bar is our own side, not the warehouse's. This holds them to it: every statement built by the fingerprint
builders anywhere in 3tears' sources is built in one of those driver functions, so no 3tears code fingerprints one of
our own L3 tables (those stay keyed).
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]

#: the builders, and the driver functions each read is declared in
_BUILDERS = frozenset({"fingerprint_sql", "postgres_fingerprint_sql"})
_DECLARED = {
    "packages/datasources/src/threetears/datasources/drivers/asyncpg_driver.py": {
        "AsyncpgDriver.relation_fingerprint",
        "AsyncpgDriver.relation_fingerprint_groups",
    },
    "packages/datasources/src/threetears/datasources/drivers/redshift_driver.py": {
        "RedshiftDriver._fingerprint_rows._do_sync",
    },
}


def _builder_calls(path: Path) -> list[str]:
    """the qualified function each fingerprint builder call in ``path`` is written in."""
    found: list[str] = []

    def walk(node: ast.AST, scope: tuple[str, ...]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                walk(child, (*scope, child.name))
                continue
            if isinstance(child, ast.Call):
                name = child.func.attr if isinstance(child.func, ast.Attribute) else getattr(child.func, "id", "")
                if name in _BUILDERS:
                    found.append(".".join(scope))
            walk(child, scope)

    walk(ast.parse(path.read_text(encoding="utf-8")), ())
    return found


def test_a_fingerprint_is_read_only_by_a_driver_over_its_datasources_warehouse_relation() -> None:
    """``AsyncpgDriver.relation_fingerprint``, ``AsyncpgDriver.relation_fingerprint_groups`` and
    ``RedshiftDriver._fingerprint_rows._do_sync`` are the only functions building a fingerprint statement."""
    built: dict[str, set[str]] = {}
    for path in sorted(_REPO.glob("packages/*/src/**/*.py")):
        relative = path.relative_to(_REPO).as_posix()
        if relative.endswith("threetears/core/fingerprint.py"):
            continue  # the builders themselves (postgres_fingerprint_sql spells fingerprint_sql for Postgres)
        functions = _builder_calls(path)
        if functions:
            built[relative] = set(functions)
    assert built == _DECLARED
    assert {"relation_fingerprint", "relation_fingerprint_groups", "_do_sync"} <= {
        name.rsplit(".", 1)[-1] for names in built.values() for name in names
    }
