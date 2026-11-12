"""bounded relation producers: a copy's relations already cut to a key, checked where they are made.

A repo whose copy accessor (``snapshot.relation(table)``) answers relations a producer has cut to a key (reports'
``limited_snapshot``: every relation of a reader of some states is ``(SELECT * FROM <view> WHERE join_state_code IN
(<their states>))``) declares the producer, :class:`BoundBy`, in ``NoWholeReadsConfig.bounded_producers``. A read
through the accessor is then judged as reading that cut -- ``(SELECT * FROM copy.<table> WHERE <key> IN (...))`` -- with
no condition on the key in its own SQL. The declaration is checked, never taken on trust; each of these is a finding
naming its site (``bounded-producer <accessor>``):

- the producer: every relation it returns (each value of the mapping it hands the snapshot's attribute) carries a
  condition on the key's columns, ``WHERE <column> IN (...)`` / ``= ...``, written in the f-string or held in a local
  the producer binds once and handed to a helper of its module that writes it after ``WHERE``. The column is the key's
  as the walk resolves it: a literal, a constant, or the producer's parameter defaulting to one, which no caller of
  the producer overrides with another. Only a branch testing ``<table> in <unlimited>`` (the declared constant, its
  members resolved by the walk) may return a relation uncut: a read of one of those tables by name is judged as
  before;
- every other place the snapshot's attribute is set (a constructor or ``replace`` handed it, a store to it): a
  snapshot's own relations handed on (``relations=views.relations``), a copy of them (``dict(views.relations)``,
  ``{**views.relations, ...}``) whose every added relation reads through the accessor, or a function the declaration
  names in ``sources`` with its reason (the every-state base the producer cuts). A source named that sets nothing is
  stale;
- each function named in ``every_state`` (where a reader of every state is handed the uncut relations) skips the
  producer on exactly that branch: ``if <states> is None`` on its parameter, the producer called with that parameter
  only where it is not None, never where it is.

A bound of no column is no bound: ``relation_producers``' ``()`` declares a read whole, and stays that.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

from threetears.enforcement.no_whole_reads.helpers import COPY_CALLS
from threetears.enforcement.no_whole_reads.schema import MIN_REASON

__all__ = ["BoundBy", "Bounded", "bounded_relations"]

#: how deep a relation's condition is followed through helpers and local names
_DEPTH: Final[int] = 4


@dataclass(frozen=True)
class BoundBy:
    """a producer whose returned relations are cut to a key, declared for the copy accessor it feeds.

    :ivar producer: the producer's dotted name, ``<module>.<function>`` (``aibots_reports.pod.report_data.limited_snapshot``)
    :ivar key: the key every relation it returns is cut by; its first column is the one checked
    :ivar reason: why a read through the accessor is bounded by it (30+ characters)
    :ivar unlimited: a constant of the producer's module (or one it imports) naming the tables it leaves uncut, tested
        as ``<table> in <unlimited>``; empty for none
    :ivar unlimited_reason: why those are left uncut
    :ivar sources: each other function that sets the snapshot's relations, ``<module>.<function>`` or
        ``<module>.<Class>.<method>``, onto why the relations it sets are the uncut every-state ones
    :ivar every_state: each function, ``Class.method`` or a module function's name, where a reader of every state is
        handed the uncut relations: the producer skipped on its ``<states> is None`` branch
    :ivar attribute: the snapshot's attribute holding the relations
    """

    producer: str
    key: tuple[str, ...]
    reason: str
    unlimited: str = ""
    unlimited_reason: str = ""
    sources: Mapping[str, str] = field(default_factory=dict)
    every_state: tuple[str, ...] = ()
    attribute: str = "relations"


@dataclass(frozen=True)
class Bounded:
    """what the walk reads a copy accessor's relation as, and the declaration's findings.

    :ivar key: the key columns a relation through the accessor is cut by; empty when none is declared
    :ivar unlimited: the tables the producer leaves uncut, by name (lower-case)
    :ivar findings: each check that failed, naming its site
    """

    key: tuple[str, ...] = ()
    unlimited: frozenset[str] = frozenset()
    findings: tuple[str, ...] = ()

    def relation(self, copy: str) -> str | None:
        """the relation a read through the accessor reads: the copy's relation cut by the key, or None for one the
        producer leaves uncut (or with no bound declared).

        :param copy: the copy's relation as the walk names it (``copy.<name>``, ``copy.{}``)
        :ptype copy: str
        :return: ``(SELECT * FROM copy.<name> WHERE <column> IN ({}) ...)``, or None
        :rtype: str | None
        """
        if not self.key or copy.split(".", 1)[-1].strip('"').lower() in self.unlimited:
            return None
        conditions = " AND ".join(f'"{column}" IN ({{}})' for column in self.key)
        return f"(SELECT * FROM {copy} WHERE {conditions})"


def _final(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _dotted(relative: str) -> str:
    return relative.removesuffix(".py").removesuffix("/__init__").replace("/", ".")


def _functions(tree: ast.Module) -> Iterator[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """every function of a module with its qualified name (``Class.method``), nested ones included."""

    def walk(body: Sequence[ast.stmt], prefix: str) -> Iterator[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
        for statement in body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}{statement.name}"
                yield name, statement
                yield from walk(statement.body, f"{name}.")
            elif isinstance(statement, ast.ClassDef):
                yield from walk(statement.body, f"{prefix}{statement.name}.")

    yield from walk(tree.body, "")


def _own_nodes(function: ast.FunctionDef | ast.AsyncFunctionDef) -> Iterator[ast.AST]:
    """the nodes of a function's own body, nested functions and classes left out."""
    stack: list[ast.AST] = list(function.body)
    while stack:
        node = stack.pop()
        yield node
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                stack.append(child)


def _bound_once(function: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, ast.expr]:
    """the local names a function binds exactly once, onto the value."""
    found: dict[str, ast.expr] = {}
    twice: set[str] = set()
    for node in _own_nodes(function):
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets, value = [node.target], node.value if isinstance(node, ast.AnnAssign) else None
            if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                twice.add(node.target.id)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.With, ast.AsyncWith, ast.NamedExpr)):
            names = [
                found_name.id
                for found_name in ast.walk(
                    node.target if isinstance(node, (ast.For, ast.AsyncFor, ast.NamedExpr)) else node
                )
                if isinstance(found_name, ast.Name) and isinstance(found_name.ctx, ast.Store)
            ]
            twice.update(names)
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                if target.id in found or value is None:
                    twice.add(target.id)
                else:
                    found[target.id] = value
    return {name: value for name, value in found.items() if name not in twice}


def _parameters(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.arg]:
    arguments = function.args
    return [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]


def _default(function: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> tuple[bool, ast.expr | None]:
    """whether ``name`` is a parameter of the function, and its default (None when it has none)."""
    arguments = function.args
    positional = [*arguments.posonlyargs, *arguments.args]
    defaults: dict[str, ast.expr] = {}
    for argument, default in zip(
        positional[len(positional) - len(arguments.defaults) :], arguments.defaults, strict=True
    ):
        defaults[argument.arg] = default
    for argument, keyword_default in zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True):
        if keyword_default is not None:
            defaults[argument.arg] = keyword_default
    known = any(argument.arg == name for argument in _parameters(function))
    return known, defaults.get(name)


def _position(function: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> int | None:
    positional = [argument.arg for argument in (*function.args.posonlyargs, *function.args.args)]
    return positional.index(name) if name in positional else None


def _argument(call: ast.Call, name: str, position: int | None) -> ast.expr | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    if position is not None and position < len(call.args):
        return call.args[position]
    return None


@dataclass
class _Module:
    relative: str
    tree: ast.Module
    env: dict[str, str]
    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef]


class _Check:
    """one declaration's checks over the walked modules."""

    def __init__(
        self, accessor: str, bound: BoundBy, modules: list[_Module], module_values: Mapping[str, Mapping[str, ast.expr]]
    ) -> None:
        self.accessor = accessor
        self.bound = bound
        self.modules = modules
        self.module_values = module_values
        self.findings: list[str] = []
        self.unlimited: frozenset[str] = frozenset()
        #: the parameter the condition's column is read from, and its position, when it is one
        self.column_parameter: tuple[str, int | None] | None = None

    def finding(self, module: _Module | None, node: ast.AST | None, function: str, message: str) -> None:
        where = f"{module.relative}:{getattr(node, 'lineno', 1)}" if module is not None else "NoWholeReadsConfig:1"
        self.findings.append(f"{where} in {function}: bounded-producer {self.accessor} -- {message}")

    def run(self) -> None:
        bound = self.bound
        owner, _, method = self.accessor.rpartition(".")
        if not owner or method not in COPY_CALLS:
            self.finding(
                None,
                None,
                "bounded_producers",
                f"{self.accessor!r} is no copy accessor ({sorted(COPY_CALLS)} on a class)",
            )
            return
        if not bound.key:
            self.finding(
                None,
                None,
                "bounded_producers",
                "a bound names its key's columns; () declares a read whole (relation_producers), never a bound",
            )
            return
        if len(bound.reason) < MIN_REASON:
            self.finding(
                None,
                None,
                "bounded_producers",
                f"the reason says why the producer bounds every read ({MIN_REASON}+ characters)",
            )
        found = self._function(bound.producer)
        if found is None:
            self.finding(
                None, None, "bounded_producers", f"the producer {bound.producer!r} names no function the walk reads"
            )
            return
        module, name, function = found
        self._producer(module, name, function)
        self._callers(function.name)
        self._sources(owner, function.name, (module.relative, name))
        self._every_state(function.name)

    def _function(self, dotted: str) -> tuple[_Module, str, ast.FunctionDef | ast.AsyncFunctionDef] | None:
        """a function by ``<module>.<qualified name>``: the walked module whose dotted path ends with the module part."""
        parts = dotted.split(".")
        for split in range(len(parts) - 1, 0, -1):
            module_part, qualified = ".".join(parts[:split]), ".".join(parts[split:])
            for module in self.modules:
                path = _dotted(module.relative)
                if (path == module_part or path.endswith(f".{module_part}")) and qualified in module.functions:
                    return module, qualified, module.functions[qualified]
        return None

    # -- the producer --------------------------------------------------------------------------------------------

    def _producer(self, module: _Module, name: str, function: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        bound = self.bound
        if bound.unlimited:
            members = self._members(module, bound.unlimited)
            if members is None:
                self.finding(
                    module,
                    function,
                    name,
                    f"the unlimited constant {bound.unlimited!r} is no set of table names the walk resolves",
                )
            else:
                self.unlimited = members
            if len(bound.unlimited_reason) < MIN_REASON:
                self.finding(
                    module, function, name, f"the unlimited tables need their reason ({MIN_REASON}+ characters)"
                )
        local = _bound_once(function)
        binders = {local_name for local_name, value in local.items() if self._condition(module, function, value)}
        returns = [node for node in _own_nodes(function) if isinstance(node, ast.Return) and node.value is not None]
        handed = [
            keyword.value
            for node in returns
            if isinstance(node.value, ast.Call)
            for keyword in node.value.keywords
            if keyword.arg == bound.attribute
        ]
        if not returns or len(handed) != len(returns):
            self.finding(
                module,
                function,
                name,
                f"every return hands the snapshot its {bound.attribute}= mapping, cut by the key",
            )
            return
        for value in handed:
            if isinstance(value, ast.Name) and value.id in local:
                value = local[value.id]
            if isinstance(value, ast.DictComp):
                targets = {
                    node.id
                    for generator in value.generators
                    for node in ast.walk(generator.target)
                    if isinstance(node, ast.Name)
                }
                self._cut(module, function, name, value.value, binders, targets, local)
            elif isinstance(value, ast.Dict):
                for item in value.values:
                    self._cut(module, function, name, item, binders, set(), local)
            else:
                self.finding(
                    module, value, name, f"its {bound.attribute}= is no mapping the walk can follow to each relation"
                )

    def _cut(
        self,
        module: _Module,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        name: str,
        value: ast.expr,
        binders: set[str],
        targets: set[str],
        local: dict[str, ast.expr],
        depth: int = 0,
    ) -> None:
        """a finding unless every relation ``value`` can be is cut by the key (or is an unlimited table's)."""
        column = self.bound.key[0]
        if isinstance(value, ast.IfExp):
            test = value.test
            unlimited_branch = (
                bool(self.bound.unlimited)
                and isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id in targets
                and len(test.ops) == 1
                and isinstance(test.ops[0], ast.In)
                and isinstance(test.comparators[0], ast.Name)
                and test.comparators[0].id == self.bound.unlimited
            )
            if not unlimited_branch:
                self._cut(module, function, name, value.body, binders, targets, local, depth + 1)
            self._cut(module, function, name, value.orelse, binders, targets, local, depth + 1)
            return
        if depth < _DEPTH and isinstance(value, ast.Name) and value.id in local:
            self._cut(module, function, name, local[value.id], binders, targets, local, depth + 1)
            return
        if isinstance(value, ast.JoinedStr) and self._where_binds(module, function, value, binders):
            return
        if depth < _DEPTH and isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            callee = module.functions.get(value.func.id)
            if callee is not None:
                handed = {
                    parameter.arg
                    for parameter, argument in zip(
                        (*callee.args.posonlyargs, *callee.args.args), value.args, strict=False
                    )
                    if isinstance(argument, ast.Name) and argument.id in binders
                } | {
                    keyword.arg
                    for keyword in value.keywords
                    if keyword.arg is not None and isinstance(keyword.value, ast.Name) and keyword.value.id in binders
                }
                callee_local = _bound_once(callee)
                callee_binders = handed | {
                    local_name for local_name, held in callee_local.items() if self._condition(module, callee, held)
                }
                returns = [
                    node.value for node in _own_nodes(callee) if isinstance(node, ast.Return) and node.value is not None
                ]
                before = len(self.findings)
                for returned in returns:
                    self._cut(module, callee, name, returned, callee_binders, set(), callee_local, depth + 1)
                if returns and len(self.findings) == before:
                    return
                del self.findings[before:]
        self.finding(
            module,
            value,
            name,
            f"returns a relation not cut by {column!r}: each is written WHERE {column} IN (...) (or a condition the "
            "producer binds once on that column), or tested `<table> in <unlimited>`",
        )

    def _where_binds(
        self, module: _Module, function: ast.FunctionDef | ast.AsyncFunctionDef, value: ast.JoinedStr, binders: set[str]
    ) -> bool:
        """whether an f-string's ``WHERE`` is followed by a condition on the key: a local holding one, or one written
        there (``WHERE {quote(column)} IN (`` / ``WHERE join_state_code = ``)."""
        parts = value.values
        for index, part in enumerate(parts):
            if not (
                isinstance(part, ast.Constant)
                and isinstance(part.value, str)
                and part.value.rstrip().upper().endswith("WHERE")
            ):
                continue
            following = parts[index + 1] if index + 1 < len(parts) else None
            if (
                isinstance(following, ast.FormattedValue)
                and isinstance(following.value, ast.Name)
                and following.value.id in binders
            ):
                return True
            if self._condition(module, function, ast.JoinedStr(values=list(parts[index + 1 :]))):
                return True
        return any(
            isinstance(part, ast.Constant)
            and isinstance(part.value, str)
            and self._literal_condition(
                part.value.upper().rpartition("WHERE")[2] if "WHERE" in part.value.upper() else ""
            )
            for part in parts
        )

    def _literal_condition(self, text: str) -> bool:
        column = self.bound.key[0].upper()
        stripped = text.strip().replace('"', "")
        return stripped.startswith((f"{column} IN (", f"{column} =", f"{column}="))

    def _condition(self, module: _Module, function: ast.FunctionDef | ast.AsyncFunctionDef, value: ast.expr) -> bool:
        """whether an expression is a condition on the key's first column: an f-string opening with the column (a
        literal, a constant, or a parameter defaulting to one; quoted by a one-argument call or not), then ``IN (`` or
        ``=``."""
        if not isinstance(value, ast.JoinedStr) or not value.values:
            return False
        first, rest = value.values[0], value.values[1:]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return self._literal_condition(first.value)
        if not isinstance(first, ast.FormattedValue) or not rest:
            return False
        after = rest[0]
        if not (isinstance(after, ast.Constant) and isinstance(after.value, str)):
            return False
        if not after.value.lstrip().upper().startswith(("IN (", "=")):
            return False
        named = first.value
        if isinstance(named, ast.Call) and len(named.args) == 1 and not named.keywords:
            named = named.args[0]
        text = self._column_text(module, function, named)
        return text is not None and text.strip('"').lower() == self.bound.key[0].lower()

    def _column_text(
        self, module: _Module, function: ast.FunctionDef | ast.AsyncFunctionDef, named: ast.expr
    ) -> str | None:
        if isinstance(named, ast.Constant) and isinstance(named.value, str):
            return named.value
        if not isinstance(named, ast.Name):
            return None
        known, default = _default(function, named.id)
        if known:
            if default is None:
                return None
            self.column_parameter = (named.id, _position(function, named.id))
            return (
                self._column_text(module, function, default)
                if not isinstance(default, ast.Name)
                else module.env.get(default.id)
            )
        return module.env.get(named.id)

    def _members(self, module: _Module, constant: str) -> frozenset[str] | None:
        """the table names a module constant (``frozenset({A, "b"})``) holds, each resolved, or None."""
        value = self.module_values.get(module.relative, {}).get(constant)
        if isinstance(value, ast.Call) and _final(value) in {"frozenset", "set", "tuple"} and len(value.args) == 1:
            value = value.args[0]
        if not isinstance(value, (ast.Set, ast.List, ast.Tuple)):
            return None
        names: set[str] = set()
        for element in value.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                names.add(element.value.lower())
            elif isinstance(element, ast.Name) and element.id in module.env:
                names.add(module.env[element.id].lower())
            else:
                return None
        return frozenset(names)

    # -- callers, sources, every-state readers -------------------------------------------------------------------

    def _callers(self, producer: str) -> None:
        """a finding at each call of the producer handing its column parameter anything but the key's column."""
        if self.column_parameter is None:
            return
        parameter, position = self.column_parameter
        for module in self.modules:
            for qualified, function in module.functions.items():
                for node in _own_nodes(function):
                    if not (isinstance(node, ast.Call) and _final(node) == producer):
                        continue
                    handed = _argument(node, parameter, position)
                    if handed is None:
                        continue
                    text = (
                        handed.value
                        if isinstance(handed, ast.Constant) and isinstance(handed.value, str)
                        else module.env.get(handed.id)
                        if isinstance(handed, ast.Name)
                        else None
                    )
                    if text is None or text.strip('"').lower() != self.bound.key[0].lower():
                        self.finding(
                            module,
                            node,
                            qualified,
                            f"hands the producer's {parameter}= another column than {self.bound.key[0]!r}",
                        )

    def _sources(self, owner: str, producer: str, producer_at: tuple[str, str]) -> None:
        """a finding at each place the snapshot's relations are set from a source the gate does not check."""
        attribute = self.bound.attribute
        declared = {}
        for dotted, reason in self.bound.sources.items():
            found = self._function(dotted)
            if found is None:
                self.finding(None, None, "bounded_producers", f"the source {dotted!r} names no function the walk reads")
                continue
            if len(reason) < MIN_REASON:
                self.finding(
                    found[0],
                    found[2],
                    found[1],
                    f"the source's reason says why its relations are uncut ({MIN_REASON}+ characters)",
                )
            declared[(found[0].relative, found[1])] = dotted
        matched: set[tuple[str, str]] = set()
        for module in self.modules:
            for qualified, function in module.functions.items():
                at = (module.relative, qualified)
                local = _bound_once(function)
                for node in _own_nodes(function):
                    set_value: ast.expr | None = None
                    if isinstance(node, ast.Call) and _final(node) in {owner, "replace"}:
                        set_value = next((keyword.value for keyword in node.keywords if keyword.arg == attribute), None)
                    elif isinstance(node, (ast.Assign, ast.AugAssign)):
                        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                        stored = [
                            target
                            for target in targets
                            if (isinstance(target, ast.Attribute) and target.attr == attribute)
                            or (
                                isinstance(target, ast.Subscript)
                                and isinstance(target.value, ast.Attribute)
                                and target.value.attr == attribute
                            )
                        ]
                        if stored and not (qualified.startswith(f"{owner}.") and qualified.count(".") == 1):
                            set_value = node.value
                    if set_value is None or at == producer_at:
                        continue
                    if at in declared:
                        matched.add(at)
                        continue
                    if not self._derived(module, function, set_value, local):
                        self.finding(
                            module,
                            node,
                            qualified,
                            f"sets {owner}.{attribute} from a source the gate does not check: cut it with "
                            f"{producer}, hand on a snapshot's own (or a copy adding relations read through "
                            f"its accessor), or name the source in bounded_producers' sources with its reason",
                        )
        for at, dotted in declared.items():
            if at not in matched:
                self.finding(
                    None, None, "bounded_producers", f"the source {dotted!r} sets no {owner}.{attribute}: delete it"
                )

    def _derived(
        self,
        module: _Module,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        value: ast.expr,
        local: dict[str, ast.expr],
    ) -> bool:
        """whether relations set from ``value`` are a snapshot's own, or a copy of them adding only relations read
        through the accessor."""
        attribute = self.bound.attribute
        name = value.id if isinstance(value, ast.Name) else None
        if name is not None and name in local:
            value = local[name]
        if isinstance(value, ast.Attribute) and value.attr == attribute:
            return True
        if isinstance(value, ast.Call) and _final(value) == "dict" and len(value.args) == 1 and not value.keywords:
            if not (isinstance(value.args[0], ast.Attribute) and value.args[0].attr == attribute):
                return False
            # each relation stored into the copy reads through the accessor
            stores = [
                node.value
                for node in _own_nodes(function)
                if isinstance(node, ast.Assign)
                for target in node.targets
                if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id == name
            ]
            return all(self._through_accessor(module, stored) for stored in stores)
        if isinstance(value, ast.Dict):
            for key, item in zip(value.keys, value.values, strict=True):
                if key is None:
                    if isinstance(item, ast.Attribute) and item.attr == attribute:
                        continue
                    if isinstance(item, ast.DictComp) and self._through_accessor(module, item.value):
                        continue
                    if isinstance(item, ast.Dict) and all(self._through_accessor(module, held) for held in item.values):
                        continue
                    return False
                if not self._through_accessor(module, item):
                    return False
            return any(
                key is None and isinstance(item, ast.Attribute) and item.attr == attribute
                for key, item in zip(value.keys, value.values, strict=True)
            )
        return False

    def _through_accessor(self, module: _Module, value: ast.expr, depth: int = 0) -> bool:
        """whether a relation reads through the copy accessor: an accessor call in it, or a call of a function of the
        module whose every return holds one."""
        if any(isinstance(node, ast.Call) and _final(node) in COPY_CALLS for node in ast.walk(value)):
            return True
        if depth < _DEPTH and isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            callee = module.functions.get(value.func.id)
            if callee is not None:
                returns = [
                    node.value for node in _own_nodes(callee) if isinstance(node, ast.Return) and node.value is not None
                ]
                return bool(returns) and all(
                    self._through_accessor(module, returned, depth + 1) for returned in returns
                )
        return False

    def _every_state(self, producer: str) -> None:
        """a finding unless each declared every-state reader skips the producer exactly where its states are None."""
        for qualified in self.bound.every_state:
            found = [(module, module.functions[qualified]) for module in self.modules if qualified in module.functions]
            if not found:
                self.finding(
                    None,
                    None,
                    "bounded_producers",
                    f"the every-state reader {qualified!r} names no function the walk reads",
                )
                continue
            for module, function in found:
                if not self._skips_for_every_state(function, producer):
                    self.finding(
                        module,
                        function,
                        qualified,
                        f"a reader of every state is handed the uncut relations only where its states are None: "
                        f"`if <states> is None:` skips {producer}, which the other branch calls with those states",
                    )
                for node in _own_nodes(function):
                    if isinstance(node, ast.Call) and _final(node) == producer and not self._guarded(function, node):
                        self.finding(
                            module, node, qualified, f"calls {producer} where a reader of every state reaches it"
                        )

    def _none_tests(
        self, function: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> Iterator[tuple[ast.If, str, list[ast.stmt], list[ast.stmt]]]:
        """each ``if <parameter> is None`` / ``is not None`` of the function: the parameter, then its None branch and its
        other branch."""
        parameters = {argument.arg for argument in _parameters(function)}
        for node in _own_nodes(function):
            if not isinstance(node, ast.If):
                continue
            test = node.test
            if not (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id in parameters
                and len(test.ops) == 1
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value is None
            ):
                continue
            if isinstance(test.ops[0], ast.Is):
                yield node, test.left.id, node.body, node.orelse
            elif isinstance(test.ops[0], ast.IsNot):
                yield node, test.left.id, node.orelse, node.body

    def _skips_for_every_state(self, function: ast.FunctionDef | ast.AsyncFunctionDef, producer: str) -> bool:
        for node, parameter, none_branch, other in self._none_tests(function):
            none_calls = [
                call
                for statement in none_branch
                for call in ast.walk(statement)
                if isinstance(call, ast.Call) and _final(call) == producer
            ]
            other_calls = [
                call
                for statement in (other or self._after(function, node))
                for call in ast.walk(statement)
                if isinstance(call, ast.Call) and _final(call) == producer
            ]
            if (
                none_branch
                and not none_calls
                and other_calls
                and all(self._hands(call, parameter) for call in other_calls)
            ):
                return True
        return False

    def _after(self, function: ast.FunctionDef | ast.AsyncFunctionDef, node: ast.If) -> list[ast.stmt]:
        """the statements after an ``if`` whose None branch always returns: its other branch."""
        for body in (function.body,):
            if node in body and node.body and isinstance(node.body[-1], (ast.Return, ast.Raise)):
                return body[body.index(node) + 1 :]
        return []

    def _hands(self, call: ast.Call, parameter: str) -> bool:
        states = _argument(call, "states", 1)
        return isinstance(states, ast.Name) and states.id == parameter

    def _guarded(self, function: ast.FunctionDef | ast.AsyncFunctionDef, call: ast.Call) -> bool:
        """whether a call of the producer sits where the function's states are not None."""
        for node, _, none_branch, other in self._none_tests(function):
            region = other or self._after(function, node)
            if any(call in ast.walk(statement) for statement in region) and not any(
                call in ast.walk(statement) for statement in none_branch
            ):
                return True
        return False


def bounded_relations(
    bounds: Mapping[str, BoundBy],
    modules: Sequence[tuple[str, ast.Module]],
    envs: Mapping[str, dict[str, str]],
) -> Bounded:
    """check every declared bound against the walked modules, and say what a copy accessor's relation reads as.

    :param bounds: the accessor (``Snapshot.relation``) onto its declaration (``NoWholeReadsConfig.bounded_producers``)
    :ptype bounds: Mapping[str, BoundBy]
    :param modules: each walked module, repo-relative path and tree
    :ptype modules: Sequence[tuple[str, ast.Module]]
    :param envs: each module's strings in scope (its own and those it imports)
    :ptype envs: Mapping[str, dict[str, str]]
    :return: the bound and its findings
    :rtype: Bounded
    """
    if not bounds:
        return Bounded()
    walked = [_Module(relative, tree, envs.get(relative, {}), dict(_functions(tree))) for relative, tree in modules]
    values: dict[str, dict[str, ast.expr]] = {}
    for module in walked:
        held: dict[str, ast.expr] = {}
        for statement in module.tree.body:
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
            ):
                held[statement.targets[0].id] = statement.value
            elif (
                isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and statement.value is not None
            ):
                held[statement.target.id] = statement.value
        values[module.relative] = held
    findings: list[str] = []
    key: tuple[str, ...] = ()
    unlimited: frozenset[str] = frozenset()
    if len(bounds) > 1:
        findings.append(
            "NoWholeReadsConfig:1 in bounded_producers: bounded-producer * -- one bound for the copy accessor, not several"
        )
    for accessor, bound in bounds.items():
        check = _Check(accessor, bound, walked, values)
        check.run()
        findings.extend(check.findings)
        if bound.key and accessor.rpartition(".")[2] in COPY_CALLS:
            key, unlimited = bound.key, check.unlimited
    return Bounded(key=key, unlimited=unlimited, findings=tuple(findings))
