"""templates over a declared closed map (:func:`threetears.core.declared.sentinel_templates`), judged once per entry.

A module that writes one SQL template for several tables (the hub's broker shadows a Class-B table with one of three
bodies, its table and columns sentinels filled from a closed map) declares the templates, the sentinels and the map.
The walk reads the declaration and the map as data (no import of the declaring code): the map is a module-level dict
literal of table name onto a call whose keyword arguments are the entry's columns. Each named template is then judged
once per entry that fills every column sentinel it holds, the sentinels replaced by the entry's table and columns.

A template the declaration does not name, a map the walk cannot read, or a template no entry fills is judged as
written: its table is a sentinel, ``unknown-schema``. Nothing is let through unjudged.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

__all__ = ["SentinelTemplates", "declared_templates"]

#: the declaration's name, as the declaring module calls it
_DECLARATION = "sentinel_templates"


@dataclass(frozen=True)
class SentinelTemplates:
    """one module's declared templates and the map they range over.

    :ivar templates: the module-level names the templates are bound to
    :ivar table: the table's sentinel
    :ivar columns: each column sentinel onto the keyword of an entry that fills it
    :ivar entries: each entry's table and its keywords' values (a column name, or None), in the map's order; empty
        when the map could not be read
    """

    templates: frozenset[str]
    table: str
    columns: Mapping[str, str]
    entries: tuple[tuple[str, Mapping[str, str | None]], ...]

    def instances(self, text: str) -> list[str] | None:
        """``text`` once per entry that fills every column sentinel it holds, the sentinels replaced; None when it is
        to be judged as written (no entry, or none fills it).

        :param text: the template's SQL
        :ptype text: str
        :return: each instance, or None
        :rtype: list[str] | None
        """
        held = [sentinel for sentinel in self.columns if sentinel in text]
        made: list[str] = []
        for table, values in self.entries:
            filled = {sentinel: values.get(self.columns[sentinel]) for sentinel in held}
            if any(value is None for value in filled.values()):
                continue
            instance = text.replace(self.table, table)
            for sentinel, value in filled.items():
                instance = instance.replace(sentinel, str(value))
            made.append(instance)
        return made or None


def _string(node: ast.expr | None, strings: Mapping[str, str]) -> str | None:
    """a literal string, or a module-level name bound to one.

    :param node: the expression
    :ptype node: ast.expr | None
    :param strings: the module's names bound to strings
    :ptype strings: Mapping[str, str]
    :return: the string, or None
    :rtype: str | None
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return strings.get(node.id)
    return None


def _map_entries(
    dotted: str, modules: Sequence[tuple[str, ast.Module]]
) -> tuple[tuple[str, Mapping[str, str | None]], ...]:
    """the entries of the map at ``dotted`` (``package.module.NAME``), or none when the walk cannot read it.

    :param dotted: the map's dotted path
    :ptype dotted: str
    :param modules: every module, repo-relative path onto its parse
    :ptype modules: Sequence[tuple[str, ast.Module]]
    :return: each entry's table and keyword values
    :rtype: tuple[tuple[str, Mapping[str, str | None]], ...]
    """
    module, _, name = dotted.rpartition(".")
    base = module.replace(".", "/")
    tree = next(
        (
            tree
            for path, tree in modules
            if path.endswith((f"/{base}.py", f"/{base}/__init__.py")) or path in {f"{base}.py", f"{base}/__init__.py"}
        ),
        None,
    )
    if tree is None:
        return ()
    for statement in tree.body:
        target = statement.targets[0] if isinstance(statement, ast.Assign) and len(statement.targets) == 1 else None
        target = statement.target if isinstance(statement, ast.AnnAssign) else target
        value = statement.value if isinstance(statement, (ast.Assign, ast.AnnAssign)) else None
        if not (isinstance(target, ast.Name) and target.id == name and isinstance(value, ast.Dict)):
            continue
        entries: list[tuple[str, Mapping[str, str | None]]] = []
        for key, entry in zip(value.keys, value.values, strict=True):
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str) and isinstance(entry, ast.Call)):
                return ()
            values: dict[str, str | None] = {}
            for keyword in entry.keywords:
                if keyword.arg is None or not isinstance(keyword.value, ast.Constant):
                    return ()
                if keyword.value.value is not None and not isinstance(keyword.value.value, str):
                    return ()
                values[keyword.arg] = keyword.value.value
            entries.append((key.value, values))
        return tuple(entries)
    return ()


def declared_templates(
    modules: Sequence[tuple[str, ast.Module]], strings: Mapping[str, Mapping[str, str]]
) -> dict[str, SentinelTemplates]:
    """every module's :func:`~threetears.core.declared.sentinel_templates` declaration, by its path.

    :param modules: every module, repo-relative path onto its parse
    :ptype modules: Sequence[tuple[str, ast.Module]]
    :param strings: each module's names bound to strings, by its path
    :ptype strings: Mapping[str, Mapping[str, str]]
    :return: the declarations
    :rtype: dict[str, SentinelTemplates]
    """
    found: dict[str, SentinelTemplates] = {}
    for path, tree in modules:
        own = strings.get(path, {})
        for statement in tree.body:
            call = statement.value if isinstance(statement, ast.Expr) else None
            if not isinstance(call, ast.Call):
                continue
            callee = call.func.id if isinstance(call.func, ast.Name) else getattr(call.func, "attr", None)
            if callee != _DECLARATION:
                continue
            given = {keyword.arg: keyword.value for keyword in call.keywords if keyword.arg}
            table = _string(given.get("table"), own)
            tables = _string(given.get("tables"), own)
            listed = given.get("templates")
            columns = given.get("columns")
            if table is None or tables is None or not isinstance(listed, ast.Tuple | ast.List):
                continue
            names = [_string(element, own) for element in listed.elts]
            pairs: dict[str, str] = {}
            if isinstance(columns, ast.Dict):
                for key, value in zip(columns.keys, columns.values, strict=True):
                    sentinel, keyword = _string(key, own), _string(value, own)
                    if sentinel is not None and keyword is not None:
                        pairs[sentinel] = keyword
            found[path] = SentinelTemplates(
                templates=frozenset(name for name in names if name is not None),
                table=table,
                columns=pairs,
                entries=_map_entries(tables, modules),
            )
    return found
