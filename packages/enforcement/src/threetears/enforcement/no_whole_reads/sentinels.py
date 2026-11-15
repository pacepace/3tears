"""templates over a declared closed map (:func:`threetears.core.declared.sentinel_templates`), judged once per entry.

A module that writes one SQL template for several tables (the hub's broker shadows a Class-B table with one of three
bodies, its table and columns sentinels filled from a closed map) declares the templates, the sentinels and the map.
The walk reads the declaration and the map as data (no import of the declaring code): the map is a module-level dict
literal of table name onto a call whose keyword arguments are the entry's columns. Each named template is then judged
once per entry that fills every column sentinel it holds, the sentinels replaced by the entry's table and columns.

A template the declaration does not name, a map the walk cannot read, or a template no entry fills is judged as
written: its table is a sentinel, ``unknown-schema``. Nothing is let through unjudged.

A declaration naming a ``renderer`` (the module's function rendering a template, filled from the map, into SQL text)
lets a keyed helper written in the module read a rendered template as a derived relation,
``Derived(<renderer>(<template>, "<table>", ...), params)``: the walk reads that relation as the template's instance
for that entry (:meth:`SentinelTemplates.instance_for`), so the helper's read over it is judged (the instance on its
own where its template is written). A ``Derived`` over any other run-time text, or a renderer call not naming a
declared template and an entry as literals, stays unseen.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

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
    :ivar renderer: the module's function rendering a template into SQL text, or None
    :ivar texts: each declared template's text, by its name, where the walk could read it
    """

    templates: frozenset[str]
    table: str
    columns: Mapping[str, str]
    entries: tuple[tuple[str, Mapping[str, str | None]], ...]
    renderer: str | None = None
    texts: Mapping[str, str] = field(default_factory=dict)

    def instance_for(self, template: str, table: str) -> str | None:
        """the text the renderer answers for ``template`` filled with the map's entry ``table``
        (``<renderer>(<template>, "<table>", ...)``): the template's instance for that entry. None when there is no
        renderer, the template is not declared or its text was not read, or the entry is not in the map or does not
        fill every column sentinel the template holds.

        :param template: the declared template's module-level name
        :ptype template: str
        :param table: the map entry's table
        :ptype table: str
        :return: the instance, or None
        :rtype: str | None
        """
        text = self.texts.get(template) if self.renderer is not None and template in self.templates else None
        if text is None:
            return None
        entry = [(name, values) for name, values in self.entries if name == table]
        made = SentinelTemplates(self.templates, self.table, self.columns, tuple(entry)).instances(text)
        return made[0] if made else None

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


def _template_texts(
    tree: ast.Module,
    names: frozenset[str],
    own: Mapping[str, str],
    text_of: Callable[[ast.expr, Mapping[str, str]], str | None] | None,
) -> dict[str, str]:
    """each declared template's text: the string a module-level name is bound to, or the string a one-argument call
    is handed (``_parse_template(f"SELECT ...")``), as ``text_of`` reads it in the module's scope.

    :param tree: the module
    :ptype tree: ast.Module
    :param names: the declared templates' names
    :ptype names: frozenset[str]
    :param own: the module's names bound to strings
    :ptype own: Mapping[str, str]
    :param text_of: how the walk reads a string expression's text; none read when None
    :ptype text_of: Callable[[ast.expr, Mapping[str, str]], str | None] | None
    :return: name onto text, for each template read
    :rtype: dict[str, str]
    """
    texts: dict[str, str] = {}
    if text_of is None:
        return texts
    for statement in tree.body:
        target = statement.targets[0] if isinstance(statement, ast.Assign) and len(statement.targets) == 1 else None
        if not (isinstance(target, ast.Name) and target.id in names) or not isinstance(statement, ast.Assign):
            continue
        value = statement.value
        if isinstance(value, ast.Call) and len(value.args) == 1 and not value.keywords:
            value = value.args[0]
        text = text_of(value, own)
        if text is not None:
            texts[target.id] = text
    return texts


def declared_templates(
    modules: Sequence[tuple[str, ast.Module]],
    strings: Mapping[str, Mapping[str, str]],
    text_of: Callable[[ast.expr, Mapping[str, str]], str | None] | None = None,
) -> dict[str, SentinelTemplates]:
    """every module's :func:`~threetears.core.declared.sentinel_templates` declaration, by its path.

    :param modules: every module, repo-relative path onto its parse
    :ptype modules: Sequence[tuple[str, ast.Module]]
    :param strings: each module's names bound to strings, by its path
    :ptype strings: Mapping[str, Mapping[str, str]]
    :param text_of: how the walk reads a string expression's text in a scope, for a declaration naming a renderer
        (its templates' texts); None reads none
    :ptype text_of: Callable[[ast.expr, Mapping[str, str]], str | None] | None
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
            declared = frozenset(name for name in names if name is not None)
            renderer = _string(given.get("renderer"), own)
            if renderer is not None and not any(
                isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == renderer for node in tree.body
            ):
                # a renderer the module does not define is no renderer: its Derived reads stay unseen
                renderer = None
            found[path] = SentinelTemplates(
                templates=declared,
                table=table,
                columns=pairs,
                entries=_map_entries(tables, modules),
                renderer=renderer,
                texts=_template_texts(tree, declared, own, text_of) if renderer is not None else {},
            )
    return found
