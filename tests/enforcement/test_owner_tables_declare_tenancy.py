"""Every table schema 3tears hands an owner to declare among its own tables declares its tenancy on the schema.

A pod's change-log reader picks each table's keys by the table's declared tenancy (``PodTenants.keys_for``). A
schema an owner declares among its own tables (:func:`~threetears.core.collections.scope_epochs.scope_epochs_schema`,
:func:`~threetears.core.coordination.source_token.source_tokens_schema`, ``COORDINATION_TABLE_SCHEMAS``) lives in
the owner's own schema, and the hub's broker writes its changes under that schema's key (``s-<schema>``). Declaring
no tenancy left it ``undeclared``, so a tool pod of a provider space asked for ``_all.<table>.e``: refused by the
grant the hub gives it (``s-ns_<hex>.>`` and the access tables' ``_all`` keys), and missing the records even if
granted (2026-11-20, the reports pod's ``scope_epochs``).

The collection census holds a concrete collection CLASS to a tenancy (``test_write_generation_declarations.py``); a
schema built by a helper, or exported bare in a tuple, reaches an owner with no class around it. So this walks every
``TableSchema(...)`` in a package's source and holds it to one of:

- it passes ``tenancy=`` itself;
- it is the ``schema`` of a class whose body assigns ``tenancy`` (directly, or as a module constant the class binds);
- it is listed in :data:`_NOT_AN_OWNERS_TABLE` with why no owner declares it as its table.

A class-body schema of a class that declares no tenancy of its own fails too: the bare schema escapes the class
(the coordination tables' do, through ``COORDINATION_TABLE_SCHEMAS``).
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final

_ROOT = Path(__file__).resolve().parents[2]

#: ``path:where:table`` (``where`` the enclosing function or class, ``<module>`` at module level) of each schema no owner declares as its own table, with why
_NOT_AN_OWNERS_TABLE: Final[dict[str, str]] = {
    "packages/core/src/threetears/core/collections/versioned_answers.py:_answers_schema:table_name": (
        "the L1 table of an L2-only derived collection's answers: no L3 table, no change-log record (its write "
        "generation is kept in L2, beside its rows); no owner declares it"
    ),
    "packages/core/src/threetears/core/collections/versioned_answers.py:_index_schema:table_name": (
        "the L1 table of the answers' L2-only index: no L3 table, no change-log record (its write generation is "
        "kept in L2, beside its rows); no owner declares it"
    ),
    "packages/agent/acl/src/threetears/agent/acl/shape_answers.py:_schema:table_name": (
        "the L1 table of an L2-only derived collection's answers: no L3 table, no change-log record (its write "
        "generation is kept in L2, beside its rows); no owner declares it"
    ),
    "packages/datasources/src/threetears/datasources/collections.py:<module>:datasource_customers": (
        "the hub's own platform table (hub v001), declared here only so a keyed read of it is judged against its "
        "key; the hub's collection declares its tenancy"
    ),
}


def _called(node: ast.Call) -> str | None:
    func = node.func
    return func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None


def _parents(tree: ast.Module) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _label(path: Path, node: ast.Call, parents: dict[ast.AST, ast.AST]) -> str:
    named = next((k.value for k in node.keywords if k.arg == "name"), None)
    label = "?" if named is None else ast.literal_eval(named) if isinstance(named, ast.Constant) else ast.unparse(named)
    where: ast.AST | None = parents.get(node)
    while where is not None and not isinstance(where, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        where = parents.get(where)
    scope = where.name if isinstance(where, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) else "<module>"
    return f"{path.relative_to(_ROOT)}:{scope}:{label}"


def _assigns(body: list[ast.stmt], name: str) -> bool:
    for stmt in body:
        if isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in stmt.targets):
            return True
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id == name:
            return True
    return False


def _bound_names(cls: ast.ClassDef) -> set[str]:
    """the module names a class body binds as its ``schema``."""
    bound: set[str] = set()
    for stmt in cls.body:
        value = stmt.value if isinstance(stmt, ast.Assign | ast.AnnAssign) else None
        targets = (
            stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target] if isinstance(stmt, ast.AnnAssign) else []
        )
        if isinstance(value, ast.Name) and any(isinstance(t, ast.Name) and t.id == "schema" for t in targets):
            bound.add(value.id)
    return bound


def _undeclared() -> list[str]:
    found: list[str] = []
    for path in sorted((_ROOT / "packages").glob("**/src/**/*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        tenanted_constants: set[str] = set()
        for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
            if _assigns(cls.body, "tenancy"):
                tenanted_constants |= _bound_names(cls)
        parents = _parents(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _called(node) != "TableSchema":
                continue
            if any(k.arg == "tenancy" for k in node.keywords):
                continue
            holder = parents.get(node)
            owner = parents.get(holder) if holder is not None else None
            if isinstance(owner, ast.ClassDef) and _assigns(owner.body, "tenancy"):
                continue  # the class's own declaration: the census holds it
            if isinstance(holder, ast.Assign | ast.AnnAssign) and isinstance(owner, ast.Module):
                targets = holder.targets if isinstance(holder, ast.Assign) else [holder.target]
                if any(isinstance(t, ast.Name) and t.id in tenanted_constants for t in targets):
                    continue  # a module constant a tenancy-declaring class binds as its schema
            where = _label(path, node, parents)
            if where not in _NOT_AN_OWNERS_TABLE:
                found.append(where)
    return found


def test_every_owner_table_schema_declares_its_tenancy() -> None:
    undeclared = _undeclared()
    assert not undeclared, (
        "these table schemas declare no tenancy, so a pod's change-log reader reads them under _all, where the "
        "hub neither grants an owner's pod nor writes an owner's table. Pass tenancy=TENANT_BY_SCHEMA for a table "
        "that lives in its owner's schema (tenant_column(...) or no_tenant(reason) for a platform table), or list "
        "it in _NOT_AN_OWNERS_TABLE with why no owner declares it:\n  " + "\n  ".join(undeclared)
    )


def test_every_exemption_is_still_a_schema() -> None:
    """an exemption whose schema moved or was declared is dropped, so the list never covers a new table."""
    walked: set[str] = set()
    for path in sorted((_ROOT / "packages").glob("**/src/**/*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        parents = _parents(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _called(node) == "TableSchema":
                if not any(k.arg == "tenancy" for k in node.keywords):
                    walked.add(_label(path, node, parents))
    stale = sorted(set(_NOT_AN_OWNERS_TABLE) - walked)
    assert not stale, "these exemptions name no undeclared TableSchema any more; drop them:\n  " + "\n  ".join(stale)
