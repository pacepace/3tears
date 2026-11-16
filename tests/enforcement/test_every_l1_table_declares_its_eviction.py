"""Every table 3tears declares for an L1 declares its eviction policy where it is declared, and the list is reviewed.

The SQLite L1 refuses a table with no policy (``EVICTABLE`` or ``KEEP``, with why: threetears.core.eviction), at
startup. This holds it before then: every ``TableSchema(...)`` and SQLAlchemy ``Table(...)`` in a package's source
names its policy (``l1_eviction=`` / ``info={L1_EVICTION_INFO: ...}``), and the policies are listed in
``_l1_table_policies.txt``, so a new table, or a table whose policy changes, shows in review with its reason
(Pace, 2026-11-15: "some tables can't be LRU'd": the policy decides by what the table is doing). A table a call names
by a variable (a factory's) is listed by the variable.
"""

from __future__ import annotations

import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_LIST = Path(__file__).with_name("_l1_table_policies.txt")


def _policy(node: ast.Call) -> str | None:
    """the policy a TableSchema or Table call declares (``KEEP``/``EVICTABLE``), ``PASSED_ON`` for one it passes on."""
    for keyword in node.keywords:
        value = keyword.value
        if keyword.arg == "info" and not isinstance(value, ast.Dict):
            return "PASSED_ON"
        if keyword.arg == "info" and isinstance(value, ast.Dict):
            for key, each in zip(value.keys, value.values, strict=True):
                if isinstance(key, ast.Name) and key.id == "L1_EVICTION_INFO":
                    value = each
                    break
            else:
                continue
        elif keyword.arg != "l1_eviction":
            continue
        if isinstance(value, ast.Call) and value.args and isinstance(value.args[0], ast.Attribute):
            return value.args[0].attr
        # a declaration passed on (TableSchema.to_sqlalchemy_table carries its schema's own)
        return "PASSED_ON" if not (isinstance(value, ast.Constant) and value.value is None) else None
    return None


def _declarations() -> dict[str, str]:
    found: dict[str, str] = {}
    for path in sorted((_ROOT / "packages").glob("**/src/**/*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
            if name == "TableSchema":
                named = next((k.value for k in node.keywords if k.arg == "name"), None)
            elif name == "Table" and len(node.args) >= 2:
                named = node.args[0]
            else:
                continue
            if named is None:
                continue
            label = ast.literal_eval(named) if isinstance(named, ast.Constant) else ast.unparse(named)
            where = f"{path.relative_to(_ROOT)}:{label}"
            found[where] = _policy(node) or "UNDECLARED"
    return found


def test_every_table_declares_a_policy() -> None:
    undeclared = sorted(where for where, policy in _declarations().items() if policy == "UNDECLARED")
    assert not undeclared, (
        "these L1 tables declare no eviction policy (TableSchema(l1_eviction=Declared(Policy.KEEP|EVICTABLE, why)) "
        "or Table(..., info={L1_EVICTION_INFO: ...})); the L1 refuses them at startup:\n  " + "\n  ".join(undeclared)
    )


def test_the_list_of_policies_is_reviewed() -> None:
    listed = dict(line.rsplit(" ", 1) for line in _LIST.read_text().splitlines() if line and not line.startswith("#"))
    now = _declarations()
    assert listed == now, (
        f"{_LIST.name} differs from the tables' declarations: review the change, then write the list again:\n"
        + "\n".join(f"{where} {policy}" for where, policy in sorted(now.items()))
    )
