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
  caller that omits it;
- a conditional expression (``A if cond else B``), as the values of both branches;
- for a call to a declared forwarder that passes no ``event_type`` at all, what every
  construction (or forwarder call) inside the forwarder resolves to: a forwarder that
  fixes its own event type publishes it for every call site that does not name one.

every multi-valued answer (callers, branches, inner constructions) is a set, and a site is
credited only with the keys safe for every value in it.

**what it refuses.** Anything else resolves to nothing, and a site with nothing resolved
gets no family credit -- today's behaviour, reported as before. In particular: a name
bound more than once at module level, or rebound by ``global``; a local variable,
including a parameter the helper reassigns; an import from outside the scanned roots; a
value computed by a call or an expression; a helper nothing calls; a caller that passes the
parameter positionally (positions are not mapped to parameters) or splats ``**kwargs``; a
cycle of helpers passing the value round; a conditional with an unreadable branch; a
forwarder defined outside the scanned roots, constructing nothing, calling itself, or with
an inner construction that is itself unresolvable. **One unresolvable caller, branch or
inner construction makes the whole answer unresolved**, never a partial answer.

**how callers are found.** By name: every call whose callee is spelled with the helper's
name, bare or as an attribute. A same-named unrelated function is therefore treated as a
caller too. That can only withhold credit, never grant it: the site is credited with the
keys safe for EVERY resolved value (the intersection), so an extra value narrows it.

**what is parsed.** Over :class:`ScannedModules`, only the modules resolution reaches: each
file's call and function names are kept in the scan cache by the file's bytes (``file_facts``),
so the resolver finds a name's callers and definitions without parsing the rest, and the answer
is the same as over every module parsed.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from threetears.enforcement.common import callee_names, dotted, iter_python_files, parse_python_path
from threetears.enforcement.common.scan_cache import CACHE_DIRECTORY, file_facts

__all__ = [
    "EventTypeResolver",
    "FunctionNode",
    "ScannedModules",
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


@dataclass(frozen=True)
class _ModuleNames:
    """what one module calls and defines, by name: the fact the scan cache keeps per file.

    :ivar callees: every name a call in the module is spelled with (:func:`callee_names`)
    :ivar functions: the name of every function the module defines, at any depth
    """

    callees: frozenset[str]
    functions: frozenset[str]


def _module_names(path: Path, tree: ast.Module) -> _ModuleNames:
    """one module's call and function names.

    :param path: the module's file
    :ptype path: Path
    :param tree: its parsed source
    :ptype tree: ast.Module
    :return: its names
    :rtype: _ModuleNames
    """
    callees: set[str] = set()
    functions: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            callees |= callee_names(node)
        elif isinstance(node, FunctionNode):
            functions.add(node.name)
    return _ModuleNames(callees=frozenset(callees), functions=frozenset(functions))


class ScannedModules(Mapping[str, SourceModule]):
    """every module under a set of source roots, by dotted name, each parsed only when it is looked up.

    the modules are the ones :func:`iter_python_files` yields under each root, in that order; a
    name two roots both hold is the first root's, and a file that does not parse goes through
    ``note_unscanned`` and is left out. what each module calls and defines is read through the
    scan cache, so :class:`EventTypeResolver` over this mapping parses only the modules its
    answers reach. a module is indexed once per mapping, so its nodes keep their identity.
    """

    def __init__(self, src_roots: Iterable[Path], repo_root: Path) -> None:
        """walk the roots and read each file's names, from the scan cache where its bytes were seen.

        :param src_roots: the source roots, each the base of its modules' dotted names
        :ptype src_roots: Iterable[Path]
        :param repo_root: the repo's root: the cache lives under it, and keys the files under it
            relative to it (a root outside it keys its own files)
        :ptype repo_root: Path
        """
        located: list[tuple[str, bool, Path]] = []
        by_anchor: dict[Path, list[Path]] = {}
        for root in src_roots:
            paths = list(iter_python_files(root))
            located.extend((*module_name(path, root), path) for path in paths)
            by_anchor.setdefault(repo_root if root.is_relative_to(repo_root) else root, []).extend(paths)
        names: dict[Path, _ModuleNames] = {}
        for anchor, paths in by_anchor.items():
            names.update(
                file_facts(_module_names, paths, root=anchor, directory=repo_root / CACHE_DIRECTORY, unparsable="note")
            )
        self._files: dict[str, tuple[Path, bool]] = {}
        self._names: dict[str, _ModuleNames] = {}
        for name, is_package, path in located:
            if path in names and name not in self._files:
                self._files[name] = (path, is_package)
                self._names[name] = names[path]
        self._indexed: dict[str, SourceModule] = {}

    def __getitem__(self, name: str) -> SourceModule:
        """the module ``name``, parsed and indexed on its first lookup.

        :param name: a dotted module name
        :ptype name: str
        :return: the indexed module
        :rtype: SourceModule
        :raises KeyError: when no scanned file is that module
        """
        module = self._indexed.get(name)
        if module is None:
            path, is_package = self._files[name]
            module = source_module(parse_python_path(path), name=name, is_package=is_package)
            self._indexed[name] = module
        return module

    def __contains__(self, name: object) -> bool:
        """whether a scanned file is the module ``name``, without parsing it.

        :param name: a dotted module name
        :ptype name: object
        :return: whether it is scanned
        :rtype: bool
        """
        return name in self._files

    def __iter__(self) -> Iterator[str]:
        """every module name, in walk order.

        :return: the names
        :rtype: Iterator[str]
        """
        return iter(self._files)

    def __len__(self) -> int:
        """how many modules were scanned.

        :return: the count
        :rtype: int
        """
        return len(self._files)

    def path(self, name: str) -> Path:
        """the file the module ``name`` was read from.

        :param name: a dotted module name
        :ptype name: str
        :return: its file
        :rtype: Path
        :raises KeyError: when no scanned file is that module
        """
        return self._files[name][0]

    def calling(self, names: frozenset[str]) -> list[str]:
        """every module holding a call spelled with one of ``names``, in walk order, none of them parsed.

        :param names: callee names, as :func:`callee_names` reads a call
        :ptype names: frozenset[str]
        :return: the module names
        :rtype: list[str]
        """
        return [module for module, found in self._names.items() if found.callees & names]

    def defining(self, name: str) -> list[str]:
        """every module defining a function called ``name``, in walk order, none of them parsed.

        :param name: a function name
        :ptype name: str
        :return: the module names
        :rtype: list[str]
        """
        return [module for module, found in self._names.items() if name in found.functions]


class EventTypeResolver:
    """resolves ``event_type`` arguments across the scanned modules, with its caller index built once."""

    def __init__(
        self,
        modules: Mapping[str, SourceModule],
        *,
        constructors: frozenset[str] = frozenset(),
        forwarders: frozenset[str] = frozenset(),
    ) -> None:
        """index every call and every function definition in every module by name.

        :param modules: every scanned module, by dotted name
        :ptype modules: Mapping[str, SourceModule]
        :param constructors: callee names that build an audit event
        :ptype constructors: frozenset[str]
        :param forwarders: callee names of declared wrapper helpers, whose own inner
            construction fixes the event type for a call site that passes none
        :ptype forwarders: frozenset[str]
        """
        self._modules = modules
        self._targets = constructors | forwarders
        self._forwarders = forwarders
        self._calls: dict[str, list[tuple[SourceModule, ast.Call]]] = {}
        self._definitions: dict[str, list[tuple[SourceModule, FunctionNode]]] = {}
        # over ScannedModules, a name's callers and definitions are read when it is first asked for,
        # from the modules the cached names say hold it
        self._scanned = modules if isinstance(modules, ScannedModules) else None
        for module in modules.values() if self._scanned is None else ():
            for node in ast.walk(module.tree):
                if isinstance(node, ast.Call):
                    for name in callee_names(node):
                        self._calls.setdefault(name, []).append((module, node))
                elif isinstance(node, FunctionNode):
                    self._definitions.setdefault(node.name, []).append((module, node))

    def _calls_to(self, name: str) -> list[tuple[SourceModule, ast.Call]]:
        """every call spelled with ``name``, with its module, in module then walk order.

        :param name: a callee name
        :ptype name: str
        :return: the calls
        :rtype: list[tuple[SourceModule, ast.Call]]
        """
        if self._scanned is not None and name not in self._calls:
            modules = [self._scanned[module] for module in self._scanned.calling(frozenset({name}))]
            self._calls[name] = [
                (module, node)
                for module in modules
                for node in ast.walk(module.tree)
                if isinstance(node, ast.Call) and name in callee_names(node)
            ]
        return self._calls.get(name, [])

    def _definitions_of(self, name: str) -> list[tuple[SourceModule, FunctionNode]]:
        """every function called ``name``, with its module, in module then walk order.

        :param name: a function name
        :ptype name: str
        :return: the definitions
        :rtype: list[tuple[SourceModule, FunctionNode]]
        """
        if self._scanned is not None and name not in self._definitions:
            modules = [self._scanned[module] for module in self._scanned.defining(name)]
            self._definitions[name] = [
                (module, node)
                for module in modules
                for node in ast.walk(module.tree)
                if isinstance(node, FunctionNode) and node.name == name
            ]
        return self._definitions.get(name, [])

    def resolve_call(self, call: ast.Call, module: SourceModule) -> frozenset[str]:
        """every event type one audit construction or forwarder call can be shown to publish.

        the call's own ``event_type=`` argument when it passes one. otherwise, for a call to a
        declared forwarder, what the forwarder's own inner constructions resolve to -- a
        forwarder that fixes its event type (``_audited_failure`` building ``admin.action``)
        publishes that type for every call site that does not name one.

        :param call: the constructor or forwarder call
        :ptype call: ast.Call
        :param module: the module holding it
        :ptype module: SourceModule
        :return: the resolved values; empty when any part could not be resolved
        :rtype: frozenset[str]
        """
        resolved = self._resolve_call(call, module, frozenset())
        return resolved if resolved is not None else frozenset()

    def _resolve_call(
        self, call: ast.Call, module: SourceModule, visiting: frozenset[tuple[str, int]]
    ) -> frozenset[str] | None:
        """resolve one call's published event types.

        :param call: the constructor or forwarder call
        :ptype call: ast.Call
        :param module: the module holding it
        :ptype module: SourceModule
        :param visiting: ``(kind, id)`` of every function already being resolved as a forwarder
            or through a parameter, to stop a cycle; the two kinds are separate because a
            forwarder's inner construction may legitimately resolve that same forwarder's
            parameter
        :ptype visiting: frozenset[tuple[str, int]]
        :return: the values, or ``None`` when unresolvable
        :rtype: frozenset[str] | None
        """
        expr = next((keyword.value for keyword in call.keywords if keyword.arg == "event_type"), None)
        result: frozenset[str] | None = _UNRESOLVED
        if expr is not None:
            result = self._resolve(expr, module, module.chains.get(id(call), ()), visiting)
        else:
            forwarders = callee_names(call) & self._forwarders
            if len(forwarders) == 1:
                result = self._resolve_forwarder(next(iter(forwarders)), visiting)
        return result

    def _resolve_forwarder(self, name: str, visiting: frozenset[tuple[str, int]]) -> frozenset[str] | None:
        """the union of what every inner construction of every forwarder named ``name`` publishes.

        :param name: the forwarder's name
        :ptype name: str
        :param visiting: functions already being resolved
        :ptype visiting: frozenset[tuple[str, int]]
        :return: the values, or ``None`` when the forwarder is unknown, constructs nothing, or
            any inner construction is unresolvable
        :rtype: frozenset[str] | None
        """
        definitions = self._definitions_of(name)
        if not definitions:
            return _UNRESOLVED
        values: set[str] = set()
        for module, function in definitions:
            inner_calls = [
                node for node in ast.walk(function) if isinstance(node, ast.Call) and callee_names(node) & self._targets
            ]
            if ("forwarder", id(function)) in visiting or not inner_calls:
                return _UNRESOLVED
            for inner in inner_calls:
                resolved = self._resolve_call(inner, module, visiting | {("forwarder", id(function))})
                if resolved is None:
                    return _UNRESOLVED
                values |= resolved
        return frozenset(values)

    def _resolve(
        self,
        expr: ast.expr,
        module: SourceModule,
        chain: tuple[FunctionNode, ...],
        visiting: frozenset[tuple[str, int]],
    ) -> frozenset[str] | None:
        """resolve one expression in its module and enclosing functions.

        :param expr: the expression
        :ptype expr: ast.expr
        :param module: the module it is written in
        :ptype module: SourceModule
        :param chain: the functions enclosing it, outermost first
        :ptype chain: tuple[FunctionNode, ...]
        :param visiting: ``(kind, id)`` of every function already being resolved, to stop a cycle
        :ptype visiting: frozenset[tuple[str, int]]
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
        elif isinstance(expr, ast.IfExp):
            # either branch may be the one published, so the value is both; one branch the
            # walker cannot read makes the whole expression unknown.
            body = self._resolve(expr.body, module, chain, visiting)
            orelse = self._resolve(expr.orelse, module, chain, visiting)
            if body is not None and orelse is not None:
                result = body | orelse
        return result

    def _resolve_name(
        self,
        name: str,
        module: SourceModule,
        chain: tuple[FunctionNode, ...],
        visiting: frozenset[tuple[str, int]],
    ) -> frozenset[str] | None:
        """resolve a bare name: the nearest enclosing parameter, else a module-level binding.

        :param name: the name
        :ptype name: str
        :param module: the module it is written in
        :ptype module: SourceModule
        :param chain: the functions enclosing it, outermost first
        :ptype chain: tuple[FunctionNode, ...]
        :param visiting: helpers already being resolved
        :ptype visiting: frozenset[tuple[str, int]]
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
        visiting: frozenset[tuple[str, int]],
    ) -> frozenset[str] | None:
        """resolve a helper's parameter to the union of what every caller passes.

        :param function: the helper
        :ptype function: ast.FunctionDef | ast.AsyncFunctionDef
        :param parameter: the parameter name
        :ptype parameter: str
        :param module: the module defining the helper
        :ptype module: SourceModule
        :param visiting: helpers already being resolved
        :ptype visiting: frozenset[tuple[str, int]]
        :return: the values, or ``None`` when any caller is unresolvable or none exists
        :rtype: frozenset[str] | None
        """
        callers = self._calls_to(function.name)
        if ("parameter", id(function)) in visiting or _stores(function, parameter) or not callers:
            return _UNRESOLVED
        inner = visiting | {("parameter", id(function))}
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
        visiting: frozenset[tuple[str, int]],
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
        :ptype visiting: frozenset[tuple[str, int]]
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
