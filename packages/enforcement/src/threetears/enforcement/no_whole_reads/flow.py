"""where the relation a keyed step reads comes from, as one module writes it.

A step (``keyed.semi``, ``read_step_arrow``, ...) is judged by the relations it names. A relation named by a
literal is judged as that table; this module says what the walk can tell of any other expression a step names:

- **clean**: a change relation (a declared transient producer's return, ``change_relation(...)``), or a step or
  derived select (judged where it is written), or a call of a function every return of which is one of these;
  through a ``self`` attribute, a local name or a container every value of which is one: ``self.new = work.put(rows)``,
  ``self.let_go[name] = work.put(rows)``, ``touched = distinct(...)`` then ``touched = union(touched, ...)``,
  ``parts.append(semi(...))``, ``union(*parts)``, ``for name, rel in sorted(self.removed.items())``;
- **handed**: a parameter of a *forwarder*, a private function (or a method or constructor of a private class) whose
  every call in its module the walk sees, which hands the parameter on to a step (or returns it as one): the step
  is judged at each call site instead, by the argument that site hands it (``_Work.step(step)``,
  ``_union(parts)``, ``_Lsf(work, sources)`` whose ``self.sources[name]`` a step reads);
- **None**: anything else, judged as the walk judges a relation it cannot read (an interpolation).

A forwarder is private by its name, so its callers are in its module; one of the same name as another definition,
or referenced other than by a call (handed on as a value), is none. A declared transient producer is never one.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
import re
from typing import Final, Literal

from threetears.enforcement.common import dotted
from threetears.enforcement.no_whole_reads.helpers import COPY_CALLS, STEP_CALLS, STEP_RELATIONS, STEP_TARGETS

__all__ = [
    "CLEAN",
    "HANDED",
    "Callee",
    "RelationFlow",
    "default_of",
    "local_names",
    "step_builders",
    "step_relation_arguments",
]

#: what an expression a step names is: a change relation or a step judged where it is written (``clean``), a
#: forwarder's parameter judged at its call sites (``handed``), or None for anything the walk cannot tell
_Kind = Literal["clean", "handed"] | None

CLEAN: Final = "clean"
HANDED: Final = "handed"

#: a binding that says nothing of its own (a cycle back to the name being read): left out of a combination
_NEUTRAL: Final = "neutral"

#: how one parameter is read: as a relation itself (``rel``), or as a container of relations (``each``)
Mode = Literal["rel", "each"]

#: the steps that are relations a step may read; the readers (``read_step``) and writes are not
_RELATION_STEPS: Final = STEP_CALLS - {
    "read_step",
    "read_step_arrow",
    "count_step",
    "any_step",
    "materialize",
    "delete_matching",
    "first_difference",
}

#: calls answering their one argument's elements (``sorted(self.removed.items())``)
_PASSING: Final = frozenset({"sorted", "list", "tuple", "set", "frozenset", "dict", "reversed", "iter"})

#: the deepest the walk follows bindings
_DEPTH: Final = 24

Function = ast.FunctionDef | ast.AsyncFunctionDef


@dataclass(frozen=True)
class Callee:
    """a forwarder a call reaches: the function, its class (None for a module function), and how many leading
    parameters the call does not bind (``self`` of a method, of a constructor).

    :ivar function: the forwarder
    :ivar owner: its class
    :ivar skipped: leading parameters a call leaves out
    """

    function: Function
    owner: ast.ClassDef | None
    skipped: int

    def parameters(self) -> list[tuple[str, int | None]]:
        """each parameter a call binds, with its position among the call's positional arguments (None for
        keyword-only).

        :return: the parameters
        :rtype: list[tuple[str, int | None]]
        """
        arguments = self.function.args
        positional = [*arguments.posonlyargs, *arguments.args][self.skipped :]
        named: list[tuple[str, int | None]] = [(argument.arg, index) for index, argument in enumerate(positional)]
        return named + [(argument.arg, None) for argument in arguments.kwonlyargs]


@dataclass
class _Binding:
    """what one statement binds a name (or a ``self`` attribute) to.

    :ivar value: the expression bound, None where the walk cannot read it
    :ivar how: ``value`` (the name is the expression), ``element`` (one value the container holds), ``elements``
        (the container holds the expression's values), ``iterated`` (the name is a value of the expression),
        ``items`` (the name is a value of the mapping ``value``)
    :ivar scope: the function the binding is written in
    """

    value: ast.expr | None
    how: str
    scope: Function


@dataclass
class _Scope:
    function: Function
    owner: ast.ClassDef | None
    comprehension: dict[str, list[_Binding]] = field(default_factory=dict)


def _final(call: ast.Call) -> str:
    func = call.func
    return func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""


def _argument(call: ast.Call, keyword: str, position: int | None) -> ast.expr | None:
    for given in call.keywords:
        if given.arg == keyword:
            return given.value
    if position is None or any(isinstance(arg, ast.Starred) for arg in call.args[: position + 1]):
        return None
    return call.args[position] if len(call.args) > position else None


def step_relation_arguments(call: ast.Call) -> Iterator[tuple[ast.expr, Mode]]:
    """each argument a step call names a relation by, and how: a ``union``'s ``*parts`` names each of its values.

    :param call: a step call
    :ptype call: ast.Call
    :return: each argument and its mode
    :rtype: Iterator[tuple[ast.expr, Mode]]
    """
    name = _final(call)
    if name == "union":
        for argument in call.args:
            if isinstance(argument, ast.Starred):
                yield argument.value, "each"
            else:
                yield argument, "rel"
    for position, keyword in (*STEP_RELATIONS.get(name, ()), *STEP_TARGETS.get(name, ())):
        found = _argument(call, keyword, position)
        if found is not None:
            yield found, "rel"


def _combine(kinds: list[_Kind | str]) -> _Kind:
    if any(kind is None for kind in kinds):
        return None
    told = [kind for kind in kinds if kind != _NEUTRAL]
    if not told:
        return None
    return HANDED if HANDED in told else CLEAN


def _static(function: Function) -> bool:
    return any(isinstance(d, ast.Name) and d.id == "staticmethod" for d in function.decorator_list)


def _private(function: Function, owner: ast.ClassDef | None) -> bool:
    if owner is not None and owner.name.startswith("_"):
        return function.name == "__init__" or not function.name.startswith("__")
    return function.name.startswith("_") and not function.name.startswith("__")


def _annotations(tree: ast.Module) -> set[int]:
    """the ids of every node inside an annotation (a parameter's, a return's, an annotated assignment's)."""
    inside: set[int] = set()
    for node in ast.walk(tree):
        annotations: list[ast.expr | None] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            arguments = node.args
            every = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs, arguments.vararg, arguments.kwarg]
            annotations += [argument.annotation for argument in every if argument is not None]
            annotations.append(node.returns)
        elif isinstance(node, ast.AnnAssign):
            annotations.append(node.annotation)
        for annotation in annotations:
            if annotation is not None:
                inside.update(id(part) for part in ast.walk(annotation))
    return inside


#: the nodes that open a scope of their own: a name bound inside one is that scope's
_SCOPES: Final = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Lambda,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)


def _scope_nodes(scope: ast.AST) -> Iterator[ast.AST]:
    """every node inside ``scope`` that is not inside a scope nested in it (a nested scope's node itself is given)."""
    for child in ast.iter_child_nodes(scope):
        yield child
        if not isinstance(child, (*_SCOPES, ast.ClassDef)):
            yield from _scope_nodes(child)


def _bound_in(scope: ast.AST) -> set[str]:
    """the names a function, lambda or comprehension binds for itself: its parameters (its comprehension targets),
    and every name it assigns, imports, defines or catches, less those it declares ``global`` / ``nonlocal``."""
    bound: set[str] = set()
    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        arguments = scope.args
        every = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs, arguments.vararg, arguments.kwarg]
        bound.update(argument.arg for argument in every if argument is not None)
        body: list[ast.AST] = [scope.body] if isinstance(scope, ast.Lambda) else list(scope.body)
    else:
        assert isinstance(scope, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp))
        body = [generator.target for generator in scope.generators]
    declared: set[str] = set()
    for top in body:
        for node in [top, *_scope_nodes(top)]:
            if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
                bound.add(node.id)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                bound.update((alias.asname or alias.name).split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                bound.add(node.name)
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                declared.update(node.names)
    return bound - declared


def local_names(tree: ast.Module) -> set[int]:
    """the ids of every bare name in ``tree`` that is a function's (a lambda's, a comprehension's) own binding or one
    of an enclosing function's: a local, never the module definition of that name.

    A local that shares a module function's name (``namesakes = [...]`` in another function) is neither a use of
    that function as a value nor a rebinding of it.

    :param tree: the module
    :ptype tree: ast.Module
    :return: the ids of the local names
    :rtype: set[int]
    """
    found: set[int] = set()

    def visit(node: ast.AST, enclosing: tuple[frozenset[str], ...]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _SCOPES):
                visit(child, (*enclosing, frozenset(_bound_in(child))))
            elif isinstance(child, ast.ClassDef):
                # a class body is no function scope: its methods see the module's names, not the class's
                visit(child, ())
            else:
                if isinstance(child, ast.Name) and any(child.id in scope for scope in enclosing):
                    found.add(id(child))
                visit(child, enclosing)

    visit(tree, ())
    return found


#: a return or field annotation naming a keyed step (``keyed.Step``, ``Step | None``)
_STEP_ANNOTATION: Final = re.compile(r"\bStep\b")


def _names_step(annotation: ast.expr | None) -> bool:
    """whether an annotation names a keyed step."""
    return annotation is not None and bool(_STEP_ANNOTATION.search(ast.unparse(annotation)))


def _callable_of_step(annotation: ast.expr | None) -> bool:
    """whether an annotation is (or holds) a callable returning a keyed step (``Callable[[...], keyed.Step]``)."""
    return annotation is not None and any(
        isinstance(node, ast.Subscript)
        and (dotted(node.value) or "").rsplit(".", 1)[-1] == "Callable"
        and isinstance(node.slice, ast.Tuple)
        and len(node.slice.elts) == 2
        and _names_step(node.slice.elts[1])
        for node in ast.walk(annotation)
    )


def step_builders(modules: list[ast.Module]) -> frozenset[str]:
    """the names of the repo's step builders: each function or method whose return annotation names a keyed step
    (``-> keyed.Step``, ``-> keyed.Step | None``), and each class field annotated a callable returning one
    (``select: str | Callable[[Sequence[str]], keyed.Step]``). A name some other definition of the repo shares
    without such an annotation is none.

    A step a builder returns is judged where the builder writes it: every keyed step the builder makes is judged in
    its body (or, a forwarder's, at its calls by what each hands it), so a step got by calling one (``made.step(())``,
    ``newest_projections(states)``, a field holding one called as ``self.select(...)``) is a step judged where it is
    written, as a step call is (R11).

    :param modules: every walked module
    :ptype modules: list[ast.Module]
    :return: the names
    :rtype: frozenset[str]
    """
    building: set[str] = set()
    other: set[str] = set()
    for tree in modules:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                (building if _names_step(node.returns) and not _callable_of_step(node.returns) else other).add(
                    node.name
                )
            elif isinstance(node, ast.ClassDef):
                for member in node.body:
                    if isinstance(member, ast.AnnAssign) and isinstance(member.target, ast.Name):
                        if _callable_of_step(member.annotation):
                            building.add(member.target.id)
    return frozenset(building - other)


def _origin_module(origin: str) -> str:
    """the module file an imported name's origin (``pkg.mod.name``) is defined in, as a path suffix."""
    return origin.rsplit(".", 1)[0].replace(".", "/") + ".py"


@dataclass
class RepoForwarders:
    """the repo's public forwarders, followed across its modules: for a repo whose public functions only its own walked
    code calls (an application; a library's are called from repos this walk never reads). Each module's
    :class:`RelationFlow` reads and writes the shared state below, keyed by the definition's node, until it settles.

    :ivar eligible: ``(path, qualified name)`` of each public definition every reference to which in the repo is a
        call the walk matches (a module function or a constructor by its imported name, a method by a name no other
        definition of the repo shares), with at least one such call
    :ivar functions: a module function's or a class's name onto each eligible one, with its module's path
    :ivar methods: a method's name onto the eligible one
    :ivar forwarded: per forwarder, each parameter it hands a step and how
    :ivar returned: per relation-returning forwarder, each parameter only its returns hand on
    :ivar returning: the forwarders every return of which is a relation
    :ivar wanted: the forwarders a call of which stands where a relation is named
    """

    eligible: frozenset[tuple[str, str]] = frozenset()
    functions: dict[str, list[tuple[str, Callee]]] = field(default_factory=dict)
    methods: dict[str, Callee] = field(default_factory=dict)
    forwarded: dict[int, dict[str, set[Mode]]] = field(default_factory=dict)
    returned: dict[int, dict[str, set[Mode]]] = field(default_factory=dict)
    returning: set[int] = field(default_factory=set)
    wanted: set[int] = field(default_factory=set)

    def size(self) -> tuple[int, int, int, int]:
        """how much the shared state holds, to tell when following it has settled.

        :return: counts of the forwarded and returned parameters, the returning and the wanted forwarders
        :rtype: tuple[int, int, int, int]
        """

        def count(found: dict[int, dict[str, set[Mode]]]) -> int:
            return sum(len(modes) for handed in found.values() for modes in handed.values())

        return count(self.forwarded), count(self.returned), len(self.returning), len(self.wanted)

    def external(self, call: ast.Call, imports: dict[str, str]) -> Callee | None:
        """the eligible definition of another module a call reaches: a name imported from it, or a method's name.

        :param call: the call
        :ptype call: ast.Call
        :param imports: the calling module's imported names, onto ``<module>.<name>``
        :ptype imports: dict[str, str]
        :return: the definition, or None
        :rtype: Callee | None
        """
        func = call.func
        if isinstance(func, ast.Name) and func.id in imports:
            origin = imports[func.id]
            module, name = _origin_module(origin), origin.rsplit(".", 1)[-1]
            found = [callee for path, callee in self.functions.get(name, []) if path.endswith(module)]
            return found[0] if len(found) == 1 else None
        if isinstance(func, ast.Attribute):
            return self.methods.get(func.attr)
        return None

    @classmethod
    def collect(cls, modules: list[tuple[str, ast.Module]]) -> RepoForwarders:
        """which public definitions of the repo may be forwarders (:attr:`eligible`).

        :param modules: each walked module's path and parse
        :ptype modules: list[tuple[str, ast.Module]]
        :return: the repo's forwarders, nothing yet followed
        :rtype: RepoForwarders
        """
        imports: dict[str, dict[str, str]] = {}
        relative: set[str] = set()
        loose: dict[str, set[str]] = {}
        loose_attributes: set[str] = set()
        named: dict[str, int] = {}
        for path, tree in modules:
            mine: dict[str, str] = {}
            for statement in ast.walk(tree):
                if isinstance(statement, ast.ImportFrom) and statement.module and not statement.level:
                    mine.update({a.asname or a.name: f"{statement.module}.{a.name}" for a in statement.names})
                elif isinstance(statement, ast.ImportFrom):
                    relative.update(a.asname or a.name for a in statement.names)
                elif isinstance(statement, ast.Import):
                    mine.update({a.asname or a.name: a.name for a in statement.names})
            imports[path] = mine
            called = {id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
            annotated = _annotations(tree)
            skipped = called | annotated | local_names(tree)
            names: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and id(node) not in skipped:
                    names.add(node.id)
                elif isinstance(node, ast.Attribute) and id(node) not in called:
                    loose_attributes.add(node.attr)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    named[node.name] = named.get(node.name, 0) + 1
            loose[path] = names
        as_modules = {_origin_module(origin + ".x") for mine in imports.values() for origin in mine.values()}
        eligible: set[tuple[str, str]] = set()
        functions: dict[str, list[tuple[str, Callee]]] = {}
        methods: dict[str, Callee] = {}
        for path, tree in modules:
            imported_whole = any(path.endswith(module) for module in as_modules)

            def referenced(name: str, path: str = path) -> bool:
                if name in loose[path] or name in relative:
                    return True
                return any(
                    alias in loose[other]
                    for other, mine in imports.items()
                    for alias, origin in mine.items()
                    if origin.rsplit(".", 1)[-1] == name and path.endswith(_origin_module(origin))
                )

            for statement in tree.body:
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if statement.name.startswith("_") or imported_whole or referenced(statement.name):
                        continue
                    eligible.add((path, statement.name))
                    functions.setdefault(statement.name, []).append((path, Callee(statement, None, 0)))
                elif isinstance(statement, ast.ClassDef):
                    for member in statement.body:
                        if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            continue
                        if member.name == "__init__":
                            if statement.name.startswith("_") or imported_whole or referenced(statement.name):
                                continue
                            eligible.add((path, f"{statement.name}.__init__"))
                            functions.setdefault(statement.name, []).append((path, Callee(member, statement, 1)))
                        elif not member.name.startswith("__") and not _private(member, statement):
                            if named.get(member.name, 0) > 1 or member.name in loose_attributes:
                                continue
                            eligible.add((path, f"{statement.name}.{member.name}"))
                            methods[member.name] = Callee(member, statement, 0 if _static(member) else 1)
        return cls(eligible=frozenset(eligible), functions=functions, methods=methods)


class RelationFlow:
    """what each expression a step names in one module is (:data:`_Kind`), and the module's forwarders.

    :param tree: the module
    :ptype tree: ast.Module
    :param keyed_step: whether a call is one of 3tears' keyed steps (imported from ``threetears.core.keyed``)
    :ptype keyed_step: Callable[[ast.Call], bool]
    :param produces: whether an expression is a declared transient producer's call, in a function of a class
    :ptype produces: Callable[[ast.expr, Function | None, ast.ClassDef | None], bool]
    :param produced: the ``self`` attributes of a class every assignment of which is a producer's return
    :ptype produced: Callable[[ast.ClassDef], frozenset[str]]
    :param producer: whether a definition is a declared transient producer (never a forwarder)
    :ptype producer: Callable[[Function, ast.ClassDef | None], bool]
    :param builders: the repo's step builders (:func:`step_builders`)
    :ptype builders: frozenset[str]
    """

    def __init__(
        self,
        tree: ast.Module,
        *,
        keyed_step: Callable[[ast.Call], bool],
        produces: Callable[[ast.expr, Function | None, ast.ClassDef | None], bool],
        produced: Callable[[ast.ClassDef], frozenset[str]],
        producer: Callable[[Function, ast.ClassDef | None], bool],
        path: str = "",
        imports: dict[str, str] | None = None,
        repo: RepoForwarders | None = None,
        builders: frozenset[str] = frozenset(),
    ) -> None:
        self._path = path
        #: the repo's step builders (:func:`step_builders`): a call of one is a step judged where it is written
        self._builders = builders
        self._imports = imports or {}
        self._repo = repo
        self._keyed_step = keyed_step
        self._produces = produces
        self._produced = produced
        self._definitions: list[tuple[Function, ast.ClassDef | None]] = []
        for statement in tree.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._definitions.append((statement, None))
            elif isinstance(statement, ast.ClassDef):
                self._definitions += [
                    (member, statement)
                    for member in statement.body
                    if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
        self._owners = {id(function): owner for function, owner in self._definitions}
        self._functions: dict[str, Function] = {}
        self._methods: dict[str, tuple[Function, ast.ClassDef]] = {}
        self._constructors: dict[str, tuple[Function, ast.ClassDef]] = {}
        self._candidates: set[int] = set()
        self._bindings: dict[int, dict[str, list[_Binding]]] = {}
        self._fields: dict[int, dict[str, list[_Binding]]] = {}
        #: per forwarder, each parameter it hands a step and how (the repo's, shared, where it follows them across
        #: modules)
        self.forwarded: dict[int, dict[str, set[Mode]]] = repo.forwarded if repo is not None else {}
        #: the forwarders every return of which is a relation (clean or handed)
        self._returning: set[int] = repo.returning if repo is not None else set()
        #: per relation-returning forwarder, each parameter only its returns hand on, and how: judged where a call of
        #: it stands as a relation, by the argument that call hands it
        self._returned: dict[int, dict[str, set[Mode]]] = repo.returned if repo is not None else {}
        #: the relation-returning forwarder whose returns the walk reads: its own parameters read there are recorded
        #: as returned, every other one as forwarded
        self._returning_now: int | None = None
        #: the forwarders a call of which stands where a relation is named: only these may return relations
        self._wanted: set[int] = repo.wanted if repo is not None else set()
        self._recording = True
        #: a parameter of a forwarder reads as handed without being recorded (whether a function returns relations)
        self._dry = False
        self._busy: set[tuple[int, str]] = set()
        self._find_forwarders(tree, producer)
        self._settle()
        self._recording = False

    # -- forwarders --------------------------------------------------------------------------------------------

    def _find_forwarders(self, tree: ast.Module, producer: Callable[[Function, ast.ClassDef | None], bool]) -> None:
        names: dict[str, int] = {}
        for function, _ in self._definitions:
            names[function.name] = names.get(function.name, 0) + 1
        called: set[int] = set()
        loose_names: set[str] = set()
        loose_attributes: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                called.add(id(node.func))
        local = local_names(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and id(node) not in called and id(node) not in local:
                loose_names.add(node.id)
            elif isinstance(node, ast.Attribute) and id(node) not in called:
                loose_attributes.add(node.attr)
        eligible = self._repo.eligible if self._repo is not None else frozenset()
        external: set[int] = set()
        for function, owner in self._definitions:
            if producer(function, owner):
                continue
            qualified = function.name if owner is None else f"{owner.name}.{function.name}"
            public = not _private(function, owner)
            if public and (self._path, qualified) not in eligible:
                continue
            if owner is None:
                if function.name in loose_names:
                    continue
                self._functions[function.name] = function
            elif function.name == "__init__":
                self._constructors[owner.name] = (function, owner)
            else:
                if names[function.name] > 1 or function.name in loose_attributes:
                    continue
                self._methods[function.name] = (function, owner)
            if public:
                external.add(id(function))
            else:
                self._candidates.add(id(function))
        sited: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and (matched := self._match(node)) is not None:
                sited.add(id(matched.function))
        self._candidates &= sited
        # a public one of the repo's: its calls are in any module, each judged by that module's walk
        self._candidates |= external

    def callee(self, call: ast.Call) -> Callee | None:
        """the forwarder a call reaches, if it is one's call.

        :param call: the call
        :ptype call: ast.Call
        :return: the forwarder, or None
        :rtype: Callee | None
        """
        found = self._match(call)
        if found is not None and id(found.function) in self._candidates:
            return found
        if self._repo is not None:
            return self._repo.external(call, self._imports)
        return None

    def _match(self, call: ast.Call) -> Callee | None:
        """the private definition a call names, by its name: a module function, a constructor, a method of a name no
        other definition shares."""
        func = call.func
        if isinstance(func, ast.Name):
            if func.id in self._functions:
                return Callee(self._functions[func.id], None, 0)
            if func.id in self._constructors:
                function, owner = self._constructors[func.id]
                return Callee(function, owner, 1)
        elif isinstance(func, ast.Attribute) and func.attr in self._methods:
            function, owner = self._methods[func.attr]
            return Callee(function, owner, 0 if _static(function) else 1)
        return None

    def _settle(self) -> None:
        """follow forwarders to a fixed point: a parameter is handed once a step, a forwarder or a relation-returning
        forwarder's return reads it."""
        for _ in range(_DEPTH):
            before = self._size()
            for function, owner in self._definitions:
                scope = _Scope(function, owner)
                for expression, mode in self._positions(function):
                    self._judged(expression, mode, scope)
                if id(function) in self._returning:
                    # a return hands its parameters on only where a call of the function stands as a relation
                    self._returning_now = id(function)
                    try:
                        for node in ast.walk(function):
                            if isinstance(node, ast.Return) and node.value is not None:
                                self._judged(node.value, "rel", scope)
                    finally:
                        self._returning_now = None
                elif id(function) in self._wanted:
                    returned = [
                        node.value
                        for node in ast.walk(function)
                        if isinstance(node, ast.Return) and node.value is not None
                    ]
                    self._dry = True
                    try:
                        kinds = [self._kind(value, scope) for value in returned]
                    finally:
                        self._dry = False
                    if returned and _combine(kinds) is not None:
                        self._returning.add(id(function))
            if self._size() == before:
                return

    def _size(self) -> tuple[int, int, int, int]:
        def count(found: dict[int, dict[str, set[Mode]]]) -> int:
            return sum(len(modes) for handed in found.values() for modes in handed.values())

        return count(self.forwarded), count(self._returned), len(self._returning), len(self._wanted)

    def _positions(self, function: Function) -> Iterator[tuple[ast.expr, Mode]]:
        """each expression a function names a relation by: a step's relation arguments, a forwarder's handed
        arguments, and its returns where it is a relation-returning forwarder."""
        for node in ast.walk(function):
            if isinstance(node, ast.Call):
                # a step, or a keyed write from a relation (apply_stage's stage, update_from's source)
                if (_final(node) in STEP_CALLS or _final(node) in STEP_RELATIONS) and self._keyed_step(node):
                    yield from step_relation_arguments(node)
                callee = self.callee(node)
                if callee is not None:
                    yield from self.handed_arguments(node, callee)

    def handed_arguments(self, call: ast.Call, callee: Callee) -> Iterator[tuple[ast.expr, Mode]]:
        """each argument a forwarder's call hands a parameter the forwarder hands a step, and how it is read.

        :param call: the call
        :ptype call: ast.Call
        :param callee: the forwarder it reaches
        :ptype callee: Callee
        :return: each argument and its mode
        :rtype: Iterator[tuple[ast.expr, Mode]]
        """
        handed = self.forwarded.get(id(callee.function), {})
        for name, position in callee.parameters():
            modes = handed.get(name)
            if not modes:
                continue
            # a parameter the call leaves to its default is handed the default: judged at the call as if passed
            argument = _argument(call, name, position)
            if argument is None:
                argument = default_of(callee.function, name)
            if argument is None:
                continue
            for mode in sorted(modes):
                yield argument, mode

    # -- kinds -----------------------------------------------------------------------------------------------

    def kind(self, expression: ast.expr, function: Function | None, owner: ast.ClassDef | None) -> _Kind:
        """what a relation expression is where the walk is.

        :param expression: the expression a step names
        :ptype expression: ast.expr
        :param function: the function it is written in, None at module level
        :ptype function: Function | None
        :param owner: that function's class, None for a module function
        :ptype owner: ast.ClassDef | None
        :return: its kind
        :rtype: _Kind
        """
        if function is None:
            return None
        return self._judged(expression, "rel", _Scope(function, owner))

    def elements(self, expression: ast.expr, function: Function | None, owner: ast.ClassDef | None) -> _Kind:
        """what every value of a container of relations is where the walk is.

        :param expression: the container (``parts``, ``self.sources``)
        :ptype expression: ast.expr
        :param function: the function it is written in, None at module level
        :ptype function: Function | None
        :param owner: that function's class, None for a module function
        :ptype owner: ast.ClassDef | None
        :return: their kind
        :rtype: _Kind
        """
        if function is None:
            return None
        return self._judged(expression, "each", _Scope(function, owner))

    def owner_of(self, function: Function | None) -> ast.ClassDef | None:
        """the class a method is defined in, None for a module function (or a nested one).

        :param function: the function
        :ptype function: Function | None
        :return: its class
        :rtype: ast.ClassDef | None
        """
        return None if function is None else self._owners.get(id(function))

    def _judged(self, expression: ast.expr, mode: Mode, scope: _Scope) -> _Kind:
        found = self._kind(expression, scope) if mode == "rel" else self._elements(expression, scope)
        return CLEAN if found == CLEAN else HANDED if found == HANDED else None

    def _kind(self, node: ast.expr, scope: _Scope, depth: int = 0) -> _Kind | str:
        if depth > _DEPTH:
            return None
        if isinstance(node, ast.Constant):
            return CLEAN if node.value is None else None
        if isinstance(node, ast.Call):
            return self._call_kind(node, scope, depth)
        if isinstance(node, ast.Attribute):
            if self._self_attribute(node, scope):
                if scope.owner is not None and node.attr in self._produced(scope.owner):
                    return CLEAN
                return self._field(node.attr, "rel", scope, depth)
            return None
        if isinstance(node, ast.Name):
            return self._name(node.id, "rel", scope, depth)
        if isinstance(node, ast.Subscript):
            return self._elements(node.value, scope, depth + 1)
        if isinstance(node, ast.IfExp):
            return _combine([self._kind(node.body, scope, depth + 1), self._kind(node.orelse, scope, depth + 1)])
        if isinstance(node, ast.BoolOp):
            return _combine([self._kind(value, scope, depth + 1) for value in node.values])
        if isinstance(node, ast.NamedExpr):
            return self._kind(node.value, scope, depth + 1)
        return None

    def _call_kind(self, node: ast.Call, scope: _Scope, depth: int) -> _Kind | str:
        name = _final(node)
        if (
            name == "Derived"
            and node.args
            and isinstance(node.args[0], ast.Call)
            and _final(node.args[0]) in COPY_CALLS
        ):
            # a derived select the code is handed as a copy's relation (a limited copy's): the copy relation it names
            return None
        if name in {"change_relation", "Derived"}:
            return CLEAN
        if name == "Quoted" and len(node.args) == 1:
            # a name already quoted is what it names
            return self._kind(node.args[0], scope, depth + 1)
        if name in _RELATION_STEPS and self._keyed_step(node):
            return CLEAN
        if self._produces(node, scope.function, scope.owner):
            return CLEAN
        callee = self.callee(node)
        if callee is not None and (self._recording or self._dry):
            self._wanted.add(id(callee.function))
        if callee is not None and id(callee.function) in self._returning:
            # a relation-returning forwarder: what it returns is what each call hands the parameters its returns hand on
            returned = self._returned.get(id(callee.function), {})
            kinds: list[_Kind | str] = [CLEAN]
            for parameter, position in callee.parameters():
                for mode in sorted(returned.get(parameter, set())):
                    argument = _argument(node, parameter, position)
                    if argument is not None:
                        kinds.append(
                            self._kind(argument, scope, depth + 1)
                            if mode == "rel"
                            else self._elements(argument, scope, depth + 1)
                        )
            return _combine(kinds)
        if name == "get" and isinstance(node.func, ast.Attribute) and node.args:
            kinds = [self._elements(node.func.value, scope, depth + 1)]
            if len(node.args) > 1:
                kinds.append(self._kind(node.args[1], scope, depth + 1))
            return _combine(kinds)
        if name in self._builders:
            # a step builder's return: judged where the builder writes it (R11)
            return CLEAN
        return None

    def _elements(self, node: ast.expr, scope: _Scope, depth: int = 0) -> _Kind | str:
        if depth > _DEPTH:
            return None
        if isinstance(node, ast.Constant):
            return CLEAN if node.value is None else None
        if isinstance(node, ast.Dict):
            kinds: list[_Kind | str] = [
                self._elements(value, scope, depth + 1) if key is None else self._kind(value, scope, depth + 1)
                for key, value in zip(node.keys, node.values, strict=True)
            ]
            return _combine(kinds) if kinds else CLEAN
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            kinds = [
                self._elements(element.value, scope, depth + 1)
                if isinstance(element, ast.Starred)
                else self._kind(element, scope, depth + 1)
                for element in node.elts
            ]
            return _combine(kinds) if kinds else CLEAN
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            inner = _Scope(scope.function, scope.owner, dict(scope.comprehension))
            for generator in node.generators:
                for target, binding in self._targets(generator.target, generator.iter, scope.function):
                    inner.comprehension[target] = [binding]
            value = node.value if isinstance(node, ast.DictComp) else node.elt
            return self._kind(value, inner, depth + 1)
        if isinstance(node, ast.Call):
            name = _final(node)
            if isinstance(node.func, ast.Name) and name in _PASSING and len(node.args) == 1:
                return self._elements(node.args[0], scope, depth + 1)
            if isinstance(node.func, ast.Attribute) and name in {"values", "copy"} and not node.args:
                return self._elements(node.func.value, scope, depth + 1)
            return None
        if isinstance(node, ast.Attribute):
            if self._self_attribute(node, scope):
                return self._field(node.attr, "each", scope, depth)
            return None
        if isinstance(node, ast.Name):
            return self._name(node.id, "each", scope, depth)
        if isinstance(node, ast.IfExp):
            return _combine(
                [self._elements(node.body, scope, depth + 1), self._elements(node.orelse, scope, depth + 1)]
            )
        if isinstance(node, ast.BoolOp):
            return _combine([self._elements(value, scope, depth + 1) for value in node.values])
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return _combine([self._elements(node.left, scope, depth + 1), self._elements(node.right, scope, depth + 1)])
        return None

    @staticmethod
    def _self_attribute(node: ast.Attribute, scope: _Scope) -> bool:
        return isinstance(node.value, ast.Name) and node.value.id == "self" and scope.owner is not None

    def _name(self, name: str, mode: Mode, scope: _Scope, depth: int) -> _Kind | str:
        if name in scope.comprehension:
            return self._bound(scope.comprehension[name], mode, scope, depth)
        function = scope.function
        arguments = function.args
        parameters = {argument.arg for argument in [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]}
        bindings = self._local_bindings(function).get(name, [])
        if name in parameters:
            if bindings:
                # a parameter the function rebinds: what it holds is no longer only what it was handed
                return None
            return self._parameter(function, name, mode)
        if not bindings:
            return None
        return self._bound(bindings, mode, scope, depth)

    def _parameter(self, function: Function, name: str, mode: Mode) -> _Kind:
        if id(function) not in self._candidates or name in {"self", "cls"}:
            return None
        if self._dry:
            return HANDED
        if self._recording:
            into = self._returned if id(function) == self._returning_now else self.forwarded
            into.setdefault(id(function), {}).setdefault(name, set()).add(mode)
            return HANDED
        handed = {**self._returned.get(id(function), {})}
        for parameter, modes in self.forwarded.get(id(function), {}).items():
            handed[parameter] = handed.get(parameter, set()) | modes
        return HANDED if mode in handed.get(name, set()) else None

    def _field(self, attribute: str, mode: Mode, scope: _Scope, depth: int) -> _Kind | str:
        owner = scope.owner
        if owner is None:
            return None
        bindings = self._field_bindings(owner).get(attribute, [])
        if not bindings:
            return None
        key = (id(owner), f"self.{attribute}:{mode}")
        if key in self._busy:
            return _NEUTRAL
        self._busy.add(key)
        try:
            kinds = [self._one(binding, mode, _Scope(binding.scope, owner), depth + 1) for binding in bindings]
        finally:
            self._busy.discard(key)
        return _combine(kinds)

    def _bound(self, bindings: list[_Binding], mode: Mode, scope: _Scope, depth: int) -> _Kind | str:
        key = (id(bindings), mode)
        if key in self._busy:
            return _NEUTRAL
        self._busy.add(key)
        try:
            kinds = [self._one(binding, mode, scope, depth + 1) for binding in bindings]
        finally:
            self._busy.discard(key)
        return _combine(kinds)

    def _one(self, binding: _Binding, mode: Mode, scope: _Scope, depth: int) -> _Kind | str:
        """what one binding says of a name read as a relation (``rel``) or as a container of them (``each``)."""
        value = binding.value
        if value is None:
            return None
        if binding.how == "value":
            return self._kind(value, scope, depth) if mode == "rel" else self._elements(value, scope, depth)
        if binding.how == "element":
            # ``parts.append(x)``, ``held[name] = x``: one value of the container
            return self._kind(value, scope, depth) if mode == "each" else _NEUTRAL
        if binding.how == "elements":
            # ``parts.extend(xs)``, ``parts += xs``
            return self._elements(value, scope, depth) if mode == "each" else _NEUTRAL
        if binding.how == "iterated":
            return self._elements(value, scope, depth) if mode == "rel" else None
        if binding.how == "items":
            return self._elements(value, scope, depth) if mode == "rel" else None
        return None

    # -- bindings --------------------------------------------------------------------------------------------

    def _local_bindings(self, function: Function) -> dict[str, list[_Binding]]:
        cached = self._bindings.get(id(function))
        if cached is not None:
            return cached
        found: dict[str, list[_Binding]] = {}

        def add(name: str, binding: _Binding) -> None:
            found.setdefault(name, []).append(binding)

        for node in ast.walk(function):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    self._assigned(target, node.value, function, add)
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                self._assigned(node.target, node.value, function, add)
            elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                add(
                    node.target.id,
                    _Binding(node.value, "elements" if isinstance(node.op, ast.Add) else "value", function),
                )
                if not isinstance(node.op, ast.Add):
                    add(node.target.id, _Binding(None, "value", function))
            elif isinstance(node, ast.NamedExpr):
                add(node.target.id, _Binding(node.value, "value", function))
            elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
                # a comprehension's target too: a step written inside it reads the name the comprehension binds
                for name, binding in self._targets(node.target, node.iter, function):
                    add(name, binding)
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    for name in _names(item.optional_vars):
                        add(name, _Binding(None, "value", function))
            elif isinstance(node, ast.ExceptHandler) and node.name:
                add(node.name, _Binding(None, "value", function))
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                receiver = node.func.value
                if isinstance(receiver, ast.Name):
                    for binding in _mutation(node, function):
                        add(receiver.id, binding)
        self._bindings[id(function)] = found
        return found

    def _assigned(
        self,
        target: ast.expr,
        value: ast.expr,
        function: Function,
        add: Callable[[str, _Binding], None],
    ) -> None:
        if isinstance(target, ast.Name):
            add(target.id, _Binding(value, "value", function))
        elif isinstance(target, (ast.Tuple, ast.List)):
            values = value.elts if isinstance(value, (ast.Tuple, ast.List)) else None
            # unpacked from a comprehension (``race, county = (work.put(x) for x in ...)``): each a value it makes
            made = isinstance(value, (ast.ListComp, ast.GeneratorExp)) and not any(
                isinstance(element, ast.Starred) for element in target.elts
            )
            for index, element in enumerate(target.elts):
                matched = values[index] if values is not None and len(values) == len(target.elts) else None
                if isinstance(element, ast.Name) and made:
                    add(element.id, _Binding(value, "iterated", function))
                elif isinstance(element, ast.Name):
                    add(element.id, _Binding(matched, "value", function))
                else:
                    for name in _names(element):
                        add(name, _Binding(None, "value", function))
        elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            add(target.value.id, _Binding(value, "element", function))
        elif isinstance(target, ast.Starred):
            for name in _names(target):
                add(name, _Binding(None, "value", function))

    def _targets(self, target: ast.expr, iterable: ast.expr, function: Function) -> list[tuple[str, _Binding]]:
        """what a loop's (or a comprehension's) target names are bound to: each a value of the iterable; of a
        mapping's ``.items()`` (sorted or not), the second the mapping's value; of ``enumerate``, the second the
        iterable's value; any other unpacking, nothing the walk can tell."""
        while isinstance(iterable, ast.Call) and isinstance(iterable.func, ast.Name) and _final(iterable) in _PASSING:
            if len(iterable.args) != 1:
                break
            iterable = iterable.args[0]
        if isinstance(target, ast.Name):
            return [(target.id, _Binding(iterable, "iterated", function))]
        if isinstance(target, ast.Tuple) and len(target.elts) == 2 and isinstance(iterable, ast.Call):
            name = _final(iterable)
            first, second = target.elts
            if isinstance(iterable.func, ast.Attribute) and name == "items" and not iterable.args:
                pairs: list[tuple[str, _Binding]] = [(n, _Binding(None, "value", function)) for n in _names(first)]
                if isinstance(second, ast.Name):
                    return [*pairs, (second.id, _Binding(iterable.func.value, "items", function))]
                return pairs + [(n, _Binding(None, "value", function)) for n in _names(second)]
            if isinstance(iterable.func, ast.Name) and name == "enumerate" and iterable.args:
                pairs = [(n, _Binding(None, "value", function)) for n in _names(first)]
                if isinstance(second, ast.Name):
                    return [*pairs, (second.id, _Binding(iterable.args[0], "iterated", function))]
                return pairs + [(n, _Binding(None, "value", function)) for n in _names(second)]
        return [(name, _Binding(None, "value", function)) for name in _names(target)]

    def _field_bindings(self, owner: ast.ClassDef) -> dict[str, list[_Binding]]:
        cached = self._fields.get(id(owner))
        if cached is not None:
            return cached
        found: dict[str, list[_Binding]] = {}

        def attribute(node: ast.expr) -> str | None:
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
                return node.attr
            return None

        for method in owner.body:
            if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(method):
                if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    value = node.value
                    for target in targets:
                        named = attribute(target)
                        if named is not None and value is not None:
                            how = "elements" if isinstance(node, ast.AugAssign) else "value"
                            if isinstance(node, ast.AugAssign) and not isinstance(node.op, ast.Add):
                                value = None
                            found.setdefault(named, []).append(_Binding(value, how, method))
                        elif isinstance(target, ast.Subscript) and (named := attribute(target.value)) is not None:
                            found.setdefault(named, []).append(_Binding(value, "element", method))
                        elif isinstance(target, (ast.Tuple, ast.List)):
                            for element in target.elts:
                                if (named := attribute(element)) is not None:
                                    found.setdefault(named, []).append(_Binding(None, "value", method))
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    named = attribute(node.func.value)
                    if named is not None:
                        for binding in _mutation(node, method):
                            found.setdefault(named, []).append(binding)
        self._fields[id(owner)] = found
        return found


def _names(node: ast.expr | None) -> list[str]:
    return [] if node is None else [n.id for n in ast.walk(node) if isinstance(n, ast.Name)]


#: a container's methods adding one value, or the values of their argument
_ADDS_ONE: Final = frozenset({"append", "add", "appendleft"})
_ADDS_MANY: Final = frozenset({"extend", "update"})
_ADDS_AT: Final = frozenset({"insert", "setdefault"})


def _mutation(call: ast.Call, function: Function) -> list[_Binding]:
    """what a method call on a container adds to it: ``append(x)`` one value, ``extend(xs)`` / ``update(m)`` their
    values, ``insert(i, x)`` / ``setdefault(k, x)`` one; nothing for any other call."""
    name = _final(call)
    if name in _ADDS_ONE and len(call.args) == 1:
        return [_Binding(call.args[0], "element", function)]
    if name in _ADDS_MANY and len(call.args) == 1:
        return [_Binding(call.args[0], "elements", function)]
    if name in _ADDS_AT and len(call.args) == 2:
        return [_Binding(call.args[1], "element", function)]
    return []


def default_of(function: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> ast.expr | None:
    """the default a function's parameter takes when a call leaves it out; None when it has none.

    :param function: the function
    :ptype function: ast.FunctionDef | ast.AsyncFunctionDef
    :param name: the parameter
    :ptype name: str
    :return: the default expression, or None
    :rtype: ast.expr | None
    """
    arguments = function.args
    positional = [*arguments.posonlyargs, *arguments.args]
    named = [argument.arg for argument in positional][len(positional) - len(arguments.defaults) :]
    defaults: dict[str, ast.expr] = dict(zip(named, arguments.defaults, strict=True))
    defaults.update(
        {a.arg: d for a, d in zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True) if d is not None}
    )
    return defaults.get(name)
