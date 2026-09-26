"""static resolution of an audit call's ``event_type`` argument, the way a reader resolves it.

A family's safe keys (``declare_safe_detail_keys`` and the built-in families) are credited
to a site only for the event types the site can be shown to publish. A literal is shown at
once. Producers also name the event type by a constant or pass it into a helper, and a
reader resolves those without running anything; so does this module.

**what it resolves.**

- a ``str`` literal;
- a name bound exactly once at module level to a ``str`` literal (``EVENT = "..."`` or
  ``EVENT: Final = "..."``), in the same module or reached through ``from m import NAME``
  (absolute or relative, aliased or not, followed through re-exports) into another module
  of the scanned source roots;
- an attribute on an imported module of the scanned roots (``events.EVENT`` after ``from
  pkg import events``, ``pkg.events.EVENT`` after ``import pkg.events``);
- a parameter of an enclosing function, as the set of values every call of that function
  in the scanned roots passes for it -- each resolved by these same rules, so a caller
  passing its own parameter is followed to its callers -- or the parameter's default for a
  caller that omits it.

**what it refuses.** Anything else resolves to nothing, and a site with nothing resolved
gets no family credit -- today's behaviour, reported as before. In particular: a name
bound more than once at module level, or rebound by ``global``; a local variable,
including a parameter the helper reassigns; an import from outside the scanned roots; a
value computed by a call or an expression; a helper nothing calls; a caller that passes the
parameter positionally (positions are not mapped to parameters) or splats ``**kwargs``; a
cycle of helpers passing the value round. **One unresolvable caller makes the whole
parameter unresolved**, never a partial answer.

**how callers are found.** By name: every call whose callee is spelled with the helper's
name, bare or as an attribute. A same-named unrelated function is therefore treated as a
caller too. That can only withhold credit, never grant it: the site is credited with the
keys safe for EVERY resolved value (the intersection), so an extra value narrows it.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from threetears.enforcement.common import callee_names, dotted

__all__ = [
    "EventTypeResolver",
    "FunctionNode",
    "SourceModule",
    "module_name",
    "parameter_names",
    "source_module",
]

FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef

#: a resolution that failed. distinct from an empty set, which no resolution produces.
_UNRESOLVED: None = None


@dataclass(frozen=True)
class SourceModule:
    """one parsed module of the scanned source roots, indexed for resolution.

    :ivar name: dotted module name relative to its source root (``pkg.events``)
    :ivar is_package: whether the module is a package ``__init__``
    :ivar tree: the parsed module
    :ivar chains: ``id(node)`` -> the functions enclosing it, outermost first
    :ivar constants: names bound exactly once at module level to a ``str`` literal
    :ivar imports: module-level import alias -> ``(module, member)``; ``member`` is ``None``
        for ``import a.b [as c]``
    :ivar methods: ``id`` of every function defined directly in a class body
    """

    name: str
    is_package: bool
    tree: ast.Module
    chains: Mapping[int, tuple[FunctionNode, ...]]
    constants: Mapping[str, str]
    imports: Mapping[str, tuple[str, str | None]]
    methods: frozenset[int] = field(default_factory=frozenset)


def source_module(tree: ast.Module, *, name: str, is_package: bool = False) -> SourceModule:
    """index one parsed module for resolution.

    :param tree: the parsed module
    :ptype tree: ast.Module
    :param name: its dotted name relative to its source root
    :ptype name: str
    :param is_package: whether it is a package ``__init__``
    :ptype is_package: bool
    :return: the indexed module
    :rtype: SourceModule
    """
    return SourceModule(
        name=name,
        is_package=is_package,
        tree=tree,
        chains=_enclosing_functions(tree),
        constants=_module_constants(tree),
        imports=_module_imports(tree, name=name, is_package=is_package),
        methods=frozenset(
            id(item)
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
            for item in node.body
            if isinstance(item, FunctionNode)
        ),
    )


def module_name(path: Path, root: Path) -> tuple[str, bool]:
    """the dotted module name a file has under its source root.

    :param path: the ``.py`` file
    :ptype path: Path
    :param root: the source root it sits under
    :ptype root: Path
    :return: ``(dotted name, is_package)``
    :rtype: tuple[str, bool]
    """
    parts = list(path.relative_to(root).with_suffix("").parts)
    is_package = parts[-1] == "__init__"
    if is_package:
        parts = parts[:-1]
    return ".".join(parts), is_package


class EventTypeResolver:
    """resolves ``event_type`` arguments across the scanned modules, with its caller index built once."""

    def __init__(self, modules: Mapping[str, SourceModule]) -> None:
        """index every call in every module by callee name.

        :param modules: every scanned module, by dotted name
        :ptype modules: Mapping[str, SourceModule]
        """
        self._modules = modules
        self._calls: dict[str, list[tuple[SourceModule, ast.Call]]] = {}
        for module in modules.values():
            for node in ast.walk(module.tree):
                if isinstance(node, ast.Call):
                    for name in callee_names(node):
                        self._calls.setdefault(name, []).append((module, node))

    def resolve_site(
        self,
        expr: ast.expr | None,
        module: SourceModule,
        chain: tuple[FunctionNode, ...],
    ) -> frozenset[str]:
        """every event type an audit call's ``event_type`` argument can be shown to carry.

        :param expr: the ``event_type=`` argument, or ``None`` when the call passes none
        :ptype expr: ast.expr | None
        :param module: the module holding the call
        :ptype module: SourceModule
        :param chain: the functions enclosing the call, outermost first
        :ptype chain: tuple[FunctionNode, ...]
        :return: the resolved values; empty when any part could not be resolved
        :rtype: frozenset[str]
        """
        resolved = None if expr is None else self._resolve(expr, module, chain, frozenset())
        return resolved if resolved is not None else frozenset()

    def _resolve(
        self,
        expr: ast.expr,
        module: SourceModule,
        chain: tuple[FunctionNode, ...],
        visiting: frozenset[int],
    ) -> frozenset[str] | None:
        """resolve one expression in its module and enclosing functions.

        :param expr: the expression
        :ptype expr: ast.expr
        :param module: the module it is written in
        :ptype module: SourceModule
        :param chain: the functions enclosing it, outermost first
        :ptype chain: tuple[FunctionNode, ...]
        :param visiting: ``id`` of every helper already being resolved, to stop a cycle
        :ptype visiting: frozenset[int]
        :return: the values, or ``None`` when unresolvable
        :rtype: frozenset[str] | None
        """
        result: frozenset[str] | None = _UNRESOLVED
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            result = frozenset({expr.value})
        elif isinstance(expr, ast.Name):
            result = self._resolve_name(expr.id, module, chain, visiting)
        elif isinstance(expr, ast.Attribute):
            head = dotted(expr.value)
            target = self._module_for(head, module) if head is not None else None
            if target is not None:
                result = self._resolve_global(expr.attr, target, frozenset())
        return result

    def _resolve_name(
        self,
        name: str,
        module: SourceModule,
        chain: tuple[FunctionNode, ...],
        visiting: frozenset[int],
    ) -> frozenset[str] | None:
        """resolve a bare name: the nearest enclosing parameter, else a module-level binding.

        :param name: the name
        :ptype name: str
        :param module: the module it is written in
        :ptype module: SourceModule
        :param chain: the functions enclosing it, outermost first
        :ptype chain: tuple[FunctionNode, ...]
        :param visiting: helpers already being resolved
        :ptype visiting: frozenset[int]
        :return: the values, or ``None`` when unresolvable
        :rtype: frozenset[str] | None
        """
        for function in reversed(chain):
            if name in parameter_names(function):
                return self._resolve_parameter(function, name, module, visiting)
            if _stores(function, name):
                return _UNRESOLVED
        return self._resolve_global(name, module, frozenset())

    def _resolve_global(
        self, name: str, module: SourceModule, seen: frozenset[tuple[str, str]]
    ) -> frozenset[str] | None:
        """resolve a module-level name: its own ``str`` constant, or through its import.

        :param name: the name
        :ptype name: str
        :param module: the module it is looked up in
        :ptype module: SourceModule
        :param seen: ``(module, name)`` pairs already followed, to stop an import cycle
        :ptype seen: frozenset[tuple[str, str]]
        :return: the value, or ``None`` when unresolvable
        :rtype: frozenset[str] | None
        """
        result: frozenset[str] | None = _UNRESOLVED
        key = (module.name, name)
        if key in seen:
            return result
        if name in module.constants:
            result = frozenset({module.constants[name]})
        elif name in module.imports:
            source, member = module.imports[name]
            target = self._modules.get(source)
            is_module = member is None or f"{source}.{member}" in self._modules
            if target is not None and member is not None and not is_module:
                result = self._resolve_global(member, target, seen | {key})
        return result

    def _module_for(self, head: str, module: SourceModule) -> SourceModule | None:
        """the scanned module a dotted head (``events``, ``pkg.events``) names, if any.

        :param head: the dotted spelling before the final attribute
        :ptype head: str
        :param module: the module it is written in
        :ptype module: SourceModule
        :return: the module, or ``None``
        :rtype: SourceModule | None
        """
        first, _, rest = head.partition(".")
        target: SourceModule | None = None
        if first in module.imports:
            source, member = module.imports[first]
            base = source if member is None else f"{source}.{member}"
            target = self._modules.get(f"{base}.{rest}" if rest else base)
        return target

    def _resolve_parameter(
        self,
        function: FunctionNode,
        parameter: str,
        module: SourceModule,
        visiting: frozenset[int],
    ) -> frozenset[str] | None:
        """resolve a helper's parameter to the union of what every caller passes.

        :param function: the helper
        :ptype function: ast.FunctionDef | ast.AsyncFunctionDef
        :param parameter: the parameter name
        :ptype parameter: str
        :param module: the module defining the helper
        :ptype module: SourceModule
        :param visiting: helpers already being resolved
        :ptype visiting: frozenset[int]
        :return: the values, or ``None`` when any caller is unresolvable or none exists
        :rtype: frozenset[str] | None
        """
        callers = self._calls.get(function.name, [])
        if id(function) in visiting or _stores(function, parameter) or not callers:
            return _UNRESOLVED
        inner = visiting | {id(function)}
        values: set[str] = set()
        for caller_module, call in callers:
            passed = self._passed(call, function, parameter, module, caller_module, inner)
            if passed is None:
                return _UNRESOLVED
            values |= passed
        return frozenset(values)

    def _passed(
        self,
        call: ast.Call,
        function: FunctionNode,
        parameter: str,
        module: SourceModule,
        caller_module: SourceModule,
        visiting: frozenset[int],
    ) -> frozenset[str] | None:
        """what one call passes for the helper's parameter.

        :param call: the call
        :ptype call: ast.Call
        :param function: the helper
        :ptype function: ast.FunctionDef | ast.AsyncFunctionDef
        :param parameter: the parameter name
        :ptype parameter: str
        :param module: the module defining the helper, where its defaults are evaluated
        :ptype module: SourceModule
        :param caller_module: the module holding the call
        :ptype caller_module: SourceModule
        :param visiting: helpers already being resolved
        :ptype visiting: frozenset[int]
        :return: the values, or ``None`` when unresolvable
        :rtype: frozenset[str] | None
        """
        if any(keyword.arg is None for keyword in call.keywords):
            return _UNRESOLVED
        for keyword in call.keywords:
            if keyword.arg == parameter:
                return self._resolve(keyword.value, caller_module, caller_module.chains.get(id(call), ()), visiting)
        index = _positional_index(function, parameter, is_method=id(function) in module.methods)
        positional = any(isinstance(argument, ast.Starred) for argument in call.args) or (
            index is not None and len(call.args) > index
        )
        default = _default(function, parameter)
        if positional or default is None:
            return _UNRESOLVED
        return self._resolve(default, module, module.chains.get(id(function), ()), visiting)


def parameter_names(function: FunctionNode) -> set[str]:
    """every parameter name a function declares.

    :param function: a function definition
    :ptype function: ast.FunctionDef | ast.AsyncFunctionDef
    :return: its parameter names
    :rtype: set[str]
    """
    arguments = function.args
    names = {arg.arg for arg in [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]}
    for variadic in (arguments.vararg, arguments.kwarg):
        if variadic is not None:
            names.add(variadic.arg)
    return names


def _positional_index(function: FunctionNode, parameter: str, *, is_method: bool) -> int | None:
    """the position a caller would pass ``parameter`` at, or ``None`` for a keyword-only one.

    :param function: the helper
    :ptype function: ast.FunctionDef | ast.AsyncFunctionDef
    :param parameter: the parameter name
    :ptype parameter: str
    :param is_method: whether the helper is defined in a class body (its first parameter
        is bound by the attribute access, not passed)
    :ptype is_method: bool
    :return: the zero-based caller position, or ``None``
    :rtype: int | None
    """
    positional = [arg.arg for arg in [*function.args.posonlyargs, *function.args.args]]
    if parameter not in positional:
        return None
    index = positional.index(parameter)
    return index - 1 if is_method and index > 0 else index


def _default(function: FunctionNode, parameter: str) -> ast.expr | None:
    """the default expression a parameter takes when a caller omits it.

    :param function: the helper
    :ptype function: ast.FunctionDef | ast.AsyncFunctionDef
    :param parameter: the parameter name
    :ptype parameter: str
    :return: the default, or ``None`` when it has none
    :rtype: ast.expr | None
    """
    arguments = function.args
    for arg, default in zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True):
        if arg.arg == parameter:
            return default
    positional = [*arguments.posonlyargs, *arguments.args]
    offset = len(positional) - len(arguments.defaults)
    for index, arg in enumerate(positional):
        if arg.arg == parameter and index >= offset:
            return arguments.defaults[index - offset]
    return None


def _stores(function: FunctionNode, name: str) -> bool:
    """whether a function body binds ``name`` (an assignment, a loop target, ``global``).

    :param function: the function
    :ptype function: ast.FunctionDef | ast.AsyncFunctionDef
    :param name: the name
    :ptype name: str
    :return: ``True`` when the body rebinds it
    :rtype: bool
    """
    return any(
        (isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Store | ast.Del))
        or (isinstance(node, ast.Global | ast.Nonlocal) and name in node.names)
        for statement in function.body
        for node in ast.walk(statement)
    )


def _module_constants(tree: ast.Module) -> dict[str, str]:
    """names bound exactly once at module level, to a ``str`` literal.

    :param tree: the parsed module
    :ptype tree: ast.Module
    :return: name -> value
    :rtype: dict[str, str]
    """
    bindings: dict[str, int] = {}
    literal: dict[str, str] = {}
    for statement in tree.body:
        for name in _bound_names(statement):
            bindings[name] = bindings.get(name, 0) + 1
        target, value = _single_assignment(statement)
        if target is not None and isinstance(value, ast.Constant) and isinstance(value.value, str):
            literal[target] = value.value
    rebound_globally = {name for node in ast.walk(tree) if isinstance(node, ast.Global) for name in node.names}
    return {name: value for name, value in literal.items() if bindings.get(name) == 1 and name not in rebound_globally}


def _single_assignment(statement: ast.stmt) -> tuple[str | None, ast.expr | None]:
    """the name and value of ``NAME = value`` / ``NAME: T = value``, else nothing.

    :param statement: a module-level statement
    :ptype statement: ast.stmt
    :return: ``(name, value)`` or ``(None, None)``
    :rtype: tuple[str | None, ast.expr | None]
    """
    result: tuple[str | None, ast.expr | None] = (None, None)
    if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
        result = (statement.targets[0].id, statement.value)
    elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
        result = (statement.target.id, statement.value)
    return result


def _bound_names(statement: ast.stmt) -> Iterable[str]:
    """every name one module-level statement binds, at any depth of that statement.

    a conditional or ``try`` binding counts: a name bound on two paths is bound twice.

    :param statement: a module-level statement
    :ptype statement: ast.stmt
    :return: the bound names, with repeats
    :rtype: Iterable[str]
    """
    names: list[str] = []
    for node in ast.walk(statement):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.append(node.id)
        elif isinstance(node, ast.alias):
            names.append((node.asname or node.name).split(".")[0])
        elif isinstance(node, FunctionNode | ast.ClassDef) and node is statement:
            names.append(node.name)
    if isinstance(statement, FunctionNode | ast.ClassDef):
        # a function body's own assignments are its locals, not module bindings.
        names = [statement.name]
    return names


def _module_imports(tree: ast.Module, *, name: str, is_package: bool) -> dict[str, tuple[str, str | None]]:
    """module-level import aliases, resolved to absolute module names.

    :param tree: the parsed module
    :ptype tree: ast.Module
    :param name: the module's dotted name, the base of a relative import
    :ptype name: str
    :param is_package: whether the module is a package ``__init__``
    :ptype is_package: bool
    :return: alias -> ``(module, member)``
    :rtype: dict[str, tuple[str, str | None]]
    """
    imports: dict[str, tuple[str, str | None]] = {}
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                if alias.asname is not None:
                    imports[alias.asname] = (alias.name, None)
                else:
                    first = alias.name.split(".")[0]
                    imports[first] = (first, None)
        elif isinstance(statement, ast.ImportFrom):
            source = _absolute(statement, name=name, is_package=is_package)
            if source is None:
                continue
            for alias in statement.names:
                imports[alias.asname or alias.name] = (source, alias.name)
    return imports


def _absolute(statement: ast.ImportFrom, *, name: str, is_package: bool) -> str | None:
    """the absolute module a ``from ... import`` names.

    :param statement: the import
    :ptype statement: ast.ImportFrom
    :param name: the importing module's dotted name
    :ptype name: str
    :param is_package: whether the importing module is a package ``__init__``
    :ptype is_package: bool
    :return: the absolute module name, or ``None`` when the relative import climbs past the root
    :rtype: str | None
    """
    if statement.level == 0:
        return statement.module
    parts = name.split(".") if name else []
    base = parts if is_package else parts[:-1]
    climb = statement.level - 1
    if climb > len(base):
        return None
    anchor = base[: len(base) - climb]
    return ".".join([*anchor, statement.module] if statement.module else anchor)


def _enclosing_functions(tree: ast.Module) -> dict[int, tuple[FunctionNode, ...]]:
    """map every node to the functions enclosing it, outermost first.

    :param tree: a parsed module
    :ptype tree: ast.Module
    :return: ``id(node)`` -> its enclosing function chain
    :rtype: dict[int, tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...]]
    """
    chains: dict[int, tuple[FunctionNode, ...]] = {}

    def visit(node: ast.AST, chain: tuple[FunctionNode, ...]) -> None:
        for child in ast.iter_child_nodes(node):
            chains[id(child)] = chain
            visit(child, (*chain, child) if isinstance(child, FunctionNode) else chain)

    visit(tree, ())
    return chains
