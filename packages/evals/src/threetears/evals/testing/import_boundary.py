"""A check an adopter runs over its own source tree: every reach into ``threetears.evals`` goes through a public root.

The package's API is its public roots (:data:`threetears.evals.PUBLIC_ROOTS`) and the names each root's
``__all__`` declares. Everything below a root is implementation: it moves, splits and is renamed in any
release, and a host that reached it breaks with an ``ImportError`` the package's own version gate never
warned about. The package holds its own tree to this (``tests/test_package_matrix.py``); this is the same
rule for a host's tree, so a host's suite fails on the reach rather than on the upgrade.

Use it from any test runner::

    from pathlib import Path

    from threetears.evals.testing import nonpublic_evals_imports


    def test_every_evals_import_is_public() -> None:
        assert nonpublic_evals_imports(Path("src"), Path("tests")) == ()

**What it reads, rule by rule.** Each finding names the rule it breaks (:data:`BoundaryRule`), so a host can
tell an import to move from a test that patches the engine:

* ``static_import`` — every ``import`` and ``from ... import`` statement: a module below a root, a name a
  root's ``__all__`` does not declare, a star import from a root.
* ``dynamic_import`` — ``importlib.import_module`` and ``__import__`` called with a literal module name
  that is below a root or is a module of the package that is not a root (a literal ``fromlist`` to
  ``__import__`` is read like a ``from`` import).
* ``attribute_reach`` — an attribute chain through a bound root (``from threetears.evals import run;
  run.metering``; ``import threetears.evals.kernel as k; k.helper``; ``threetears.evals.schema.models``
  after ``import threetears.evals``; ``evals.run.metering`` after ``from threetears import evals``), and ``getattr``/``hasattr`` with a literal name on one, each
  checked against that root's ``__all__``.
* ``private_reach`` — a ``_name`` attribute on a chain rooted at a name bound from a ``threetears.evals``
  import, or ``getattr``/``hasattr`` with a literal ``"_..."`` name on one. Dunders (``__name__``,
  ``__all__``) are admitted, here and under ``attribute_reach``.
* ``patch`` — replacing something in the package: ``monkeypatch.setattr``/``delattr``/``setitem``/
  ``delitem``, ``mock.patch``/``patch.object``/``patch.dict``/``patch.multiple`` (as a call or a
  decorator, through any receiver), builtin ``setattr``/``delattr``, and assignment or ``del`` of an
  attribute — whenever the target is a ``"threetears.evals..."`` string or an object bound from a
  ``threetears.evals`` import. Patching a name where the host's own module binds it is the host's
  business and is not reported.

Imports are read at module level, under ``TYPE_CHECKING`` and inside functions alike, since each is a name
the host's code binds to. It parses; it does not execute the host's code. It imports each public root a
reach names, to read that root's ``__all__``, so a root that needs an extra (``vega``,
``transports.fastmcp``) must be importable where the check runs — as it must be where the host's code runs.

**What it cannot see.**

* A module name or attribute name computed at run time: ``importlib.import_module(name)``, an f-string,
  ``getattr(k, name)``, a patch target built from parts.
* A reach through anything but a name bound by an import: the result of a call or a subscript
  (``k.make()._cache``), a name re-bound from an imported one (``alias = k; alias._x``), an instance
  (``EvalRun(...)._x``), ``sys.modules``, ``exec`` and ``eval``.
* A string annotation (``"k._Private"``): annotations are read only where they are expressions.
* Control flow. Which binding of a name is in effect is decided by scope and source order, not by the
  paths through a scope: a rebinding inside one branch of an ``if`` counts on every line after it, a
  rebinding later in a loop is not seen by an earlier line's next iteration, and a function reads its
  enclosing scope's last binding of a free name, whenever it is called.
* A shadowed ``importlib``: the names ``importlib`` and ``import_module`` are recognised wherever a file
  imports them, without regard to scope.

**Bindings are per scope.** A name is read as a ``threetears.evals`` binding only where the import's
binding is the one in effect, as Python resolves it: a function parameter, a local assignment, a ``for``,
``with``, ``except`` or ``match`` target, a comprehension's target, a nested ``def`` or ``class``, and
another import all shadow it within their scope, and ``global``/``nonlocal`` send a rebinding to the scope
they name. So ``def f(run): run._x`` is not a reach, while a nested function that does not rebind ``run``
still reads the module's import. A name read before its scope binds it is not a reach either.
"""

from __future__ import annotations

import ast
import importlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from threetears.evals import PUBLIC_ROOTS

#: The package itself. Importing it is allowed (it declares :data:`~threetears.evals.PUBLIC_ROOTS`), and
#: so is ``from threetears.evals import <root>`` for a public root directly below it.
_PACKAGE = "threetears.evals"

#: Which rule a :class:`BoundaryFinding` breaks — a stable identifier a host may filter or count on. The
#: module docstring states each.
BoundaryRule = Literal["static_import", "dynamic_import", "attribute_reach", "private_reach", "patch"]

#: Every :data:`BoundaryRule`, in the order the module docstring states them.
BOUNDARY_RULES: tuple[BoundaryRule, ...] = (
    "static_import",
    "dynamic_import",
    "attribute_reach",
    "private_reach",
    "patch",
)

#: Method names that patch their first argument: pytest's ``monkeypatch``, under any receiver name.
_MONKEYPATCH_METHODS = frozenset({"setattr", "delattr", "setitem", "delitem"})

#: ``patch.<these>`` patch their first argument, as ``patch`` itself does.
_PATCH_FORMS = frozenset({"object", "dict", "multiple"})

#: Builtins that read an attribute named by their second argument.
_ATTRIBUTE_READERS = frozenset({"getattr", "hasattr"})

#: Builtins that replace or remove an attribute named by their second argument.
_ATTRIBUTE_WRITERS = frozenset({"setattr", "delattr"})


@dataclass(frozen=True)
class BoundaryFinding:
    """One reach into ``threetears.evals`` that does not go through a public root's declared names.

    Attributes:
        path: The file it is in.
        line: Its line number.
        rule: Which rule it breaks.
        target: The dotted name reached: the module imported (``threetears.evals.schema.models``), the name
            bound or read (``threetears.evals.run.metering``; ``threetears.evals.schema.*`` for a star import),
            or what is patched (``threetears.evals.run.metering.x``).
        reason: Why it is not public, and where the name is public when it is anywhere.
    """

    path: Path
    line: int
    rule: BoundaryRule
    target: str
    reason: str

    def __str__(self) -> str:
        """The finding as one line: where, which rule, what, and why.

        Returns:
            ``path:line: rule: target — reason``.
        """
        return f"{self.path}:{self.line}: {self.rule}: {self.target} — {self.reason}"


def nonpublic_evals_imports(*sources: Path | str) -> tuple[BoundaryFinding, ...]:
    """Every reach into ``threetears.evals`` under ``sources`` that goes below a public root's declared names.

    A static import is public when it imports from a public root a name in that root's ``__all__``, imports
    a public root itself (``import threetears.evals.run``; ``from threetears.evals import run``), or
    imports the package's own root. The other rules (:data:`BoundaryRule`) are stated in the module
    docstring.

    Args:
        sources: Files and directories to read; a directory is read recursively for ``.py`` files.

    Returns:
        Every finding, in path, line, rule and target order; empty when every reach is public.

    Raises:
        FileNotFoundError: A source does not exist — a check over a path that is not there would pass
            by reading nothing.
        ValueError: No source is named, or a directory named holds no ``.py`` file: a check over nothing
            passes by reading nothing, which is the failure the check exists to prevent.
        SyntaxError: A file does not parse.
    """
    if not sources:
        raise ValueError(
            "nonpublic_evals_imports names no file or directory, so it would pass having read nothing; name the "
            "host's source and test trees"
        )
    exported: dict[str, frozenset[str]] = {}
    findings: list[BoundaryFinding] = []
    for path in _python_files(sources):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        findings.extend(_FileCheck(path, tree, exported).findings())
    return tuple(sorted(set(findings), key=lambda found: (str(found.path), found.line, found.rule, found.target)))


# --- what a bound name stands for ----------------------------------------------------------------------

#: What a name bound by an import stands for, as a chain walks it: a prefix of the package's dotted name
#: (``threetears``), a public root or the package itself, a module of the package that is not a root, or
#: any other object of the package.
_Kind = Literal["namespace", "root", "below", "object"]


@dataclass(frozen=True)
class _Bound:
    """What an expression resolves to inside the package: its dotted name and what kind of thing it is."""

    dotted: str
    kind: _Kind


def _bound_module(module: str) -> _Bound | None:
    """What ``import <module>`` makes reachable, as a module, or None outside the package.

    Args:
        module: The dotted module name.

    Returns:
        The binding, or None.
    """
    if module == _PACKAGE or module in PUBLIC_ROOTS:
        return _Bound(module, "root")
    if _in_package(module):
        return _Bound(module, "below")
    if f"{_PACKAGE}.".startswith(f"{module}."):
        return _Bound(module, "namespace")
    return None


def _is_dunder(name: str) -> bool:
    """Whether ``name`` is a dunder (``__name__``), which every module and object carries and none hides."""
    return len(name) > 4 and name.startswith("__") and name.endswith("__")


def _is_private(name: str) -> bool:
    """Whether ``name`` is an underscored implementation name: leading ``_``, not a dunder."""
    return name.startswith("_") and not _is_dunder(name)


# --- which binding of a name is in effect where it is read ---------------------------------------------

#: Where a binding happens, as ``(line, column)``: a read sees the last binding in its scope that comes
#: before it. A function's parameters and a comprehension's targets bind at :data:`_SCOPE_START`, before
#: anything in the scope is read.
_Position = tuple[int, int]

#: The position a scope's own parameters and targets bind at: before every read in it.
_SCOPE_START: _Position = (0, -1)

#: A scope that reads its free names when it is CALLED, not where it is written: a name it reads from an
#: enclosing scope is whatever that scope bound last.
_DEFERRED_KINDS = frozenset({"function"})


@dataclass
class _Scope:
    """One Python scope: the module, a function or lambda, a class body, or a comprehension.

    Attributes:
        kind: ``module``, ``function`` (a lambda too), ``class`` or ``comprehension``.
        parent: The enclosing scope; None for the module.
        events: Each name bound in this scope, to its bindings in source order: where, and what it binds
            (None for anything that is not a ``threetears.evals`` import — an assignment, a parameter, an
            import of another package).
        declared_global: Names a ``global`` statement hands to the module scope.
        declared_nonlocal: Names a ``nonlocal`` statement hands to an enclosing function.
    """

    kind: Literal["module", "function", "class", "comprehension"]
    parent: _Scope | None
    events: dict[str, list[tuple[_Position, _Bound | None]]] = field(default_factory=dict)
    declared_global: set[str] = field(default_factory=set)
    declared_nonlocal: set[str] = field(default_factory=set)


def _end(node: ast.AST) -> _Position:
    """Where ``node`` ends — the position a binding made from its value takes effect at."""
    return (getattr(node, "end_lineno", None) or 0, getattr(node, "end_col_offset", None) or 0)


def _start(node: ast.AST) -> _Position:
    """Where ``node`` starts — the position a read at it happens at."""
    return (getattr(node, "lineno", 0), getattr(node, "col_offset", 0))


def _target_names(target: ast.AST) -> Iterator[str]:
    """Every name an assignment target binds: a name, or the names inside a tuple, list or starred target.

    An attribute or subscript target binds no name; it changes an object, which the patch rule reads.
    """
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, ast.Tuple | ast.List):
        for element in target.elts:
            yield from _target_names(element)
    elif isinstance(target, ast.Starred):
        yield from _target_names(target.value)


class _ScopeBuilder(ast.NodeVisitor):
    """Walk a module once, recording every scope, every binding in it, and the scope each node is read in.

    The parts of a definition Python evaluates in the ENCLOSING scope — decorators, defaults, annotations,
    a class's bases, a comprehension's first iterable — are visited there, and only the body in the new one.
    """

    def __init__(self) -> None:
        self.module = _Scope("module", None)
        self.current = self.module
        self.scope_of: dict[int, _Scope] = {}

    def visit(self, node: ast.AST) -> None:
        self.scope_of[id(node)] = self.current
        super().visit(node)

    def _visit_all(self, nodes: Sequence[ast.AST]) -> None:
        for node in nodes:
            self.visit(node)

    def _bind(self, name: str, at: _Position, bound: _Bound | None, scope: _Scope | None = None) -> None:
        """Record that ``name`` is bound at ``at``, in the scope a ``global``/``nonlocal`` hands it to."""
        scope = scope or self.current
        if name in scope.declared_global:
            scope = self.module
        elif name in scope.declared_nonlocal:
            scope = self._enclosing_owner(scope, name)
        scope.events.setdefault(name, []).append((at, bound))

    def _enclosing_owner(self, scope: _Scope, name: str) -> _Scope:
        """The enclosing function scope a ``nonlocal name`` refers to."""
        cursor = scope.parent
        while cursor is not None:
            if cursor.kind == "function" and name in cursor.events:
                return cursor
            cursor = cursor.parent
        return scope

    def _enter(self, kind: Literal["function", "class", "comprehension"]) -> _Scope:
        scope = _Scope(kind, self.current)
        self.current = scope
        return scope

    def _leave(self, scope: _Scope) -> None:
        assert scope.parent is not None
        self.current = scope.parent

    # --- imports bind --------------------------------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            head = alias.name.split(".")[0]
            bound = _bound_module(alias.name if alias.asname is not None else head)
            self._bind(alias.asname or head, _end(node), bound)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module if node.level == 0 else None
        for alias in node.names:
            if alias.name == "*":
                continue
            bound: _Bound | None = None
            if module is not None:
                dotted = f"{module}.{alias.name}"
                if _in_package(module):
                    bound = _Bound(dotted, "root") if dotted in PUBLIC_ROOTS else _Bound(dotted, "object")
                else:
                    # ``from threetears import evals`` binds the package as ``import threetears.evals as evals``
                    # does: a module named off a namespace above it is that module.
                    bound = _bound_module(dotted)
            self._bind(alias.asname or alias.name, _end(node), bound)
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        self.current.declared_global.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.current.declared_nonlocal.update(node.names)

    # --- every other binding shadows -----------------------------------------------------------------

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for target in node.targets:
            for name in _target_names(target):
                self._bind(name, _end(node.value), None)
        self._visit_all(node.targets)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.target)
        self.visit(node.value)
        for name in _target_names(node.target):
            self._bind(name, _end(node.value), None)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.visit(node.annotation)
        if node.value is not None:
            self.visit(node.value)
        for name in _target_names(node.target):
            self._bind(name, _end(node), None)
        self.visit(node.target)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        owner = self.current
        while owner.kind == "comprehension" and owner.parent is not None:
            owner = owner.parent
        self._bind(node.target.id, _end(node.value), None, owner)
        self.visit(node.target)

    def visit_For(self, node: ast.For | ast.AsyncFor) -> None:
        self.visit(node.iter)
        for name in _target_names(node.target):
            self._bind(name, _end(node.iter), None)
        self.visit(node.target)
        self._visit_all(node.body)
        self._visit_all(node.orelse)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit_For(node)

    def visit_With(self, node: ast.With | ast.AsyncWith) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                for name in _target_names(item.optional_vars):
                    self._bind(name, _end(item.context_expr), None)
                self.visit(item.optional_vars)
        self._visit_all(node.body)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self.visit_With(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        self._visit_all(node.targets)
        for target in node.targets:
            for name in _target_names(target):
                self._bind(name, _end(node), None)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is not None:
            self.visit(node.type)
        if node.name is not None:
            self._bind(node.name, _end(node.type) if node.type is not None else _start(node), None)
        self._visit_all(node.body)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        self.generic_visit(node)
        if node.name is not None:
            self._bind(node.name, _end(node), None)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        if node.name is not None:
            self._bind(node.name, _end(node), None)

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        self.generic_visit(node)
        if node.rest is not None:
            self._bind(node.rest, _end(node), None)

    # --- new scopes ----------------------------------------------------------------------------------

    def _parameters(self, arguments: ast.arguments) -> list[str]:
        every = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
        every += [arg for arg in (arguments.vararg, arguments.kwarg) if arg is not None]
        return [arg.arg for arg in every]

    def _outer_parts_of(self, arguments: ast.arguments) -> None:
        """A signature's defaults and annotations, which the enclosing scope evaluates."""
        self._visit_all(arguments.defaults)
        self._visit_all([default for default in arguments.kw_defaults if default is not None])
        every = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
        every += [arg for arg in (arguments.vararg, arguments.kwarg) if arg is not None]
        self._visit_all([arg.annotation for arg in every if arg.annotation is not None])

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._visit_all(node.decorator_list)
        self._outer_parts_of(node.args)
        if node.returns is not None:
            self.visit(node.returns)
        self._visit_all(node.type_params)
        scope = self._enter("function")
        for name in self._parameters(node.args):
            self._bind(name, _SCOPE_START, None)
        self._visit_all(node.body)
        self._leave(scope)
        self._bind(node.name, _end(node), None)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self._outer_parts_of(node.args)
        scope = self._enter("function")
        for name in self._parameters(node.args):
            self._bind(name, _SCOPE_START, None)
        self.visit(node.body)
        self._leave(scope)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_all(node.decorator_list)
        self._visit_all(node.bases)
        self._visit_all(node.keywords)
        self._visit_all(node.type_params)
        scope = self._enter("class")
        self._visit_all(node.body)
        self._leave(scope)
        self._bind(node.name, _end(node), None)

    def _comprehension(self, node: ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp) -> None:
        first, *rest = node.generators
        self.visit(first.iter)
        scope = self._enter("comprehension")
        for generator in node.generators:
            for name in _target_names(generator.target):
                self._bind(name, _SCOPE_START, None)
        for at, generator in enumerate([first, *rest]):
            self.scope_of[id(generator)] = scope
            self.visit(generator.target)
            if at:
                self.visit(generator.iter)
            self._visit_all(generator.ifs)
        if isinstance(node, ast.DictComp):
            self.visit(node.key)
            self.visit(node.value)
        else:
            self.visit(node.elt)
        self._leave(scope)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._comprehension(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._comprehension(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._comprehension(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._comprehension(node)


@dataclass
class _Scopes:
    """A module's scopes, answering which binding of a name is in effect where it is read."""

    module: _Scope
    scope_of: dict[int, _Scope]

    @classmethod
    def of(cls, tree: ast.Module) -> _Scopes:
        builder = _ScopeBuilder()
        builder.visit(tree)
        return cls(builder.module, builder.scope_of)

    def resolve(self, node: ast.Name) -> _Bound | None:
        """What ``node`` stands for in the package where it is read, or None when the binding in effect is not
        a ``threetears.evals`` import.

        A name bound in the reading scope is the last binding there before the read (none before it is an
        unbound local, or — in a class body — a read of the module's). A free name is looked up outward,
        skipping class bodies as Python does; once the lookup leaves a function, the read happens when the
        function is called, so the enclosing scope's LAST binding is the one in effect.

        Args:
            node: A name being read.

        Returns:
            Its binding, or None.
        """
        name = node.id
        scope = self.scope_of.get(id(node), self.module)
        at: _Position | None = _start(node)
        if name in scope.declared_global:
            return self._last(self.module, name, None)[1]
        if name not in scope.declared_nonlocal and name in scope.events:
            found, bound = self._last(scope, name, at)
            if found or scope.kind != "class":
                return bound
        cursor = scope
        while cursor.parent is not None:
            if cursor.kind in _DEFERRED_KINDS:
                at = None
            cursor = cursor.parent
            if cursor.kind == "class":
                continue
            if name in cursor.events:
                return self._last(cursor, name, at)[1]
        return None

    @staticmethod
    def _last(scope: _Scope, name: str, at: _Position | None) -> tuple[bool, _Bound | None]:
        """The last binding of ``name`` in ``scope`` before ``at`` (any, when ``at`` is None).

        Returns:
            Whether there is one, and what it binds.
        """
        before = [bound for where, bound in scope.events.get(name, []) if at is None or where <= at]
        return (True, before[-1]) if before else (False, None)


@dataclass
class _FileCheck:
    """The rules, over one parsed file."""

    path: Path
    tree: ast.Module
    exported: dict[str, frozenset[str]]
    scopes: _Scopes = field(init=False)
    importlib_names: set[str] = field(default_factory=set)
    import_module_names: set[str] = field(default_factory=set)
    out: list[BoundaryFinding] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.scopes = _Scopes.of(self.tree)

    def findings(self) -> list[BoundaryFinding]:
        """Every finding in the file: the static imports first, then every reach through a bound name.

        Returns:
            The findings, unordered.
        """
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                self._import(node)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None:
                self._import_from(node, node.module)
        inner = {id(node.value) for node in ast.walk(self.tree) if isinstance(node, ast.Attribute)}
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call):
                self._call(node)
            elif isinstance(node, ast.Attribute) and id(node) not in inner:
                self._chain(node)
        return self.out

    def _find(self, node: ast.AST, rule: BoundaryRule, target: str, reason: str) -> None:
        self.out.append(
            BoundaryFinding(path=self.path, line=getattr(node, "lineno", 0), rule=rule, target=target, reason=reason)
        )

    # --- static imports, and the bindings they make ---------------------------------------------------

    def _import(self, node: ast.Import) -> None:
        for alias in node.names:
            if (reason := _module_import_defect(alias.name)) is not None:
                self._find(node, "static_import", alias.name, reason)
            if alias.name == "importlib" and alias.asname is not None:
                self.importlib_names.add(alias.asname)
            elif alias.name.split(".")[0] == "importlib" and alias.asname is None:
                # ``import importlib.util`` binds ``importlib`` itself, as ``import importlib`` does; an alias of a
                # submodule (``import importlib.util as u``) binds that submodule, which has no import_module.
                self.importlib_names.add("importlib")

    def _import_from(self, node: ast.ImportFrom, module: str) -> None:
        for alias in node.names:
            if module == "importlib" and alias.name == "import_module":
                self.import_module_names.add(alias.asname or alias.name)
            if not _in_package(module):
                continue
            if (reason := _from_import_defect(module, alias.name, self.exported)) is not None:
                self._find(node, "static_import", f"{module}.{alias.name}", reason)

    # --- attribute chains -----------------------------------------------------------------------------

    def _resolve(self, node: ast.expr) -> _Bound | None:
        """What ``node`` reaches in the package, reporting nothing — or None when it is not a bound reach.

        Used for the object a call is handed (``getattr``'s, a patch's), whose own chain is reported as a
        chain; a link that breaks a rule is followed here as an object, so a patch through it is still a patch.

        Args:
            node: A name, or an attribute chain of names.

        Returns:
            The binding the chain ends at, or None when it leaves the package, passes through a dunder, or
            does not start at a bound name.
        """
        if isinstance(node, ast.Name):
            return self.scopes.resolve(node)
        if isinstance(node, ast.Attribute):
            base = self._resolve(node.value)
            return None if base is None else self._step(base, node.attr, node=None)
        return None

    def _step(self, base: _Bound, name: str, *, node: ast.AST | None) -> _Bound | None:
        """Reach ``name`` on ``base``, reporting the reach at ``node`` when it breaks a rule.

        Args:
            base: What the chain has reached so far.
            name: The attribute read.
            node: Where to report; None resolves without reporting, following a link that breaks a rule
                as an object of the package.

        Returns:
            What the chain reaches next, or None when it stops: it left the package, passed through a
            dunder, or broke a rule (when reporting).
        """
        dotted = f"{base.dotted}.{name}"
        if base.kind == "namespace":
            return _bound_module(dotted)
        if _is_dunder(name):
            return None
        if _is_private(name):
            if node is None:
                return _Bound(dotted, "object")
            self._find(
                node,
                "private_reach",
                dotted,
                f"{name} is an underscored implementation name of {base.dotted}; nothing outside the package "
                "may bind to it, and it moves without notice",
            )
            return None
        if base.kind != "root":
            return _Bound(dotted, "object")
        if dotted in PUBLIC_ROOTS:
            return _Bound(dotted, "root")
        if node is None or name in _exports(base.dotted, self.exported):
            return _Bound(dotted, "object")
        self._find(node, "attribute_reach", dotted, _undeclared(base.dotted, name, self.exported))
        return None

    def _chain(self, node: ast.Attribute) -> None:
        """Walk one whole attribute chain from its base name, reporting the first link that breaks a rule.

        An assigned or deleted last link is a patch of what the chain reached before it.
        """
        links: list[ast.Attribute] = []
        cursor: ast.expr = node
        while isinstance(cursor, ast.Attribute):
            links.append(cursor)
            cursor = cursor.value
        if not isinstance(cursor, ast.Name) or (bound := self.scopes.resolve(cursor)) is None:
            return
        links.reverse()
        for at, link in enumerate(links):
            if at == len(links) - 1 and isinstance(link.ctx, ast.Store | ast.Del) and bound.kind != "namespace":
                self._patch(link, f"{bound.dotted}.{link.attr}", "assigns or deletes it")
                return
            step = self._step(bound, link.attr, node=link)
            if step is None:
                return
            bound = step

    # --- calls: dynamic imports, literal attribute names, patches -------------------------------------

    def _call(self, node: ast.Call) -> None:
        func = node.func
        if self._is_import_module(func):
            self._dynamic_import(node)
        elif isinstance(func, ast.Name) and func.id == "__import__":
            self._dunder_import(node)
        elif isinstance(func, ast.Name) and func.id in _ATTRIBUTE_READERS:
            self._literal_attribute(node)
        elif isinstance(func, ast.Name) and func.id in _ATTRIBUTE_WRITERS:
            self._patch_call(node, named=True)
        elif isinstance(func, ast.Attribute) and func.attr in _MONKEYPATCH_METHODS:
            self._patch_call(node, named=func.attr in _ATTRIBUTE_WRITERS)
        elif _is_patch(func):
            self._patch_call(node, named=isinstance(func, ast.Attribute) and func.attr == "object")

    def _is_import_module(self, func: ast.expr) -> bool:
        if isinstance(func, ast.Name):
            return func.id in self.import_module_names
        return (
            isinstance(func, ast.Attribute)
            and func.attr == "import_module"
            and isinstance(func.value, ast.Name)
            and func.value.id in self.importlib_names
        )

    def _dynamic_import(self, node: ast.Call) -> None:
        name = _literal(_argument(node, 0, "name"))
        if name is None:
            return
        if name.startswith("."):
            package = _literal(_argument(node, 1, "package"))
            if package is None:
                return
            name = _resolve_relative(name, package)
        if (reason := _module_import_defect(name)) is not None:
            self._find(node, "dynamic_import", name, reason)

    def _dunder_import(self, node: ast.Call) -> None:
        name = _literal(_argument(node, 0, "name"))
        if name is None or not _in_package(name):
            return
        fromlist = _argument(node, 3, "fromlist")
        names = (
            [element.value for element in fromlist.elts if isinstance(element, ast.Constant)]
            if isinstance(fromlist, ast.List | ast.Tuple)
            else []
        )
        if not names:
            if (reason := _module_import_defect(name)) is not None:
                self._find(node, "dynamic_import", name, reason)
            return
        for member in names:
            if isinstance(member, str) and (reason := _from_import_defect(name, member, self.exported)) is not None:
                self._find(node, "dynamic_import", f"{name}.{member}", reason)

    def _literal_attribute(self, node: ast.Call) -> None:
        target = _argument(node, 0, "obj")
        name = _literal(_argument(node, 1, "name"))
        if target is None or name is None:
            return
        bound = self._resolve(target)
        if bound is not None:
            self._step(bound, name, node=node)

    def _patch_call(self, node: ast.Call, *, named: bool) -> None:
        """Report a patch whose target is in the package.

        Args:
            node: The call.
            named: Whether the call names the attribute it patches as its second argument
                (``setattr(obj, "name", ...)``, ``patch.object(obj, "name")``).
        """
        target = _argument(node, 0, "target")
        if target is None:
            return
        text = _literal(target)
        if text is not None:
            if _in_package(text):
                self._patch(node, text, "patches it by name")
            return
        bound = self._resolve(target)
        if bound is None or bound.kind == "namespace":
            return
        name = _literal(_argument(node, 1, "name") or _argument(node, 1, "attribute")) if named else None
        self._patch(node, bound.dotted if name is None else f"{bound.dotted}.{name}", "patches it")

    def _patch(self, node: ast.AST, target: str, how: str) -> None:
        self._find(
            node,
            "patch",
            target,
            f"{how}, replacing the engine's own object, which no release keeps at that path; patch the name where "
            "your code binds it, or reach the behaviour through a seam a public root exports",
        )


def _is_patch(func: ast.expr) -> bool:
    """Whether ``func`` is ``patch``, ``<anything>.patch`` or ``<...>patch.object|dict|multiple``.

    Args:
        func: A call's function.

    Returns:
        Whether it is one of the mock module's patchers, under any receiver.
    """
    if isinstance(func, ast.Name):
        return func.id == "patch"
    if not isinstance(func, ast.Attribute):
        return False
    if func.attr == "patch":
        return True
    owner = func.value
    named_patch = (isinstance(owner, ast.Name) and owner.id == "patch") or (
        isinstance(owner, ast.Attribute) and owner.attr == "patch"
    )
    return func.attr in _PATCH_FORMS and named_patch


def _argument(node: ast.Call, position: int, keyword: str) -> ast.expr | None:
    """A call's argument, by position or by keyword.

    Args:
        node: The call.
        position: Its position.
        keyword: Its keyword.

    Returns:
        The argument's expression, or None when the call does not pass it.
    """
    if len(node.args) > position and not any(isinstance(arg, ast.Starred) for arg in node.args[: position + 1]):
        return node.args[position]
    return next((kw.value for kw in node.keywords if kw.arg == keyword), None)


def _literal(node: ast.expr | None) -> str | None:
    """The string ``node`` is when it is a string literal, else None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _resolve_relative(name: str, package: str) -> str:
    """``importlib.import_module``'s resolution of a relative ``name`` against ``package``.

    Args:
        name: A name with leading dots.
        package: The anchor package.

    Returns:
        The absolute name.
    """
    level = len(name) - len(name.lstrip("."))
    parts = package.split(".")
    base = ".".join(parts[: len(parts) - (level - 1)]) if level - 1 < len(parts) else ""
    rest = name[level:]
    return f"{base}.{rest}" if rest else base


# --- the static rules ------------------------------------------------------------------------------------


def _python_files(sources: tuple[Path | str, ...]) -> Iterator[Path]:
    """Every ``.py`` file the sources name, each once, sorted.

    Args:
        sources: Files and directories.

    Yields:
        The files.

    Raises:
        FileNotFoundError: A source does not exist.
        ValueError: A directory holds no ``.py`` file.
    """
    seen: set[Path] = set()
    for source in map(Path, sources):
        if not source.exists():
            raise FileNotFoundError(f"{source} does not exist, so nothing under it could be checked")
        files = sorted(source.rglob("*.py")) if source.is_dir() else [source]
        if not files:
            raise ValueError(
                f"{source} holds no .py file, so the check would pass having read nothing under it; name a directory "
                "holding the host's Python source, or the files themselves"
            )
        for path in files:
            if path not in seen:
                seen.add(path)
                yield path


def _in_package(module: str) -> bool:
    """Whether ``module`` is the package or a module below it.

    Args:
        module: An absolute module name.

    Returns:
        Whether it is ``threetears.evals`` or starts with ``threetears.evals.``.
    """
    return module == _PACKAGE or module.startswith(f"{_PACKAGE}.")


def _module_import_defect(module: str) -> str | None:
    """Why importing ``module`` reaches below a public root, or None when it does not.

    Args:
        module: The module imported.

    Returns:
        One sentence, or None.
    """
    if not _in_package(module) or module == _PACKAGE or module in PUBLIC_ROOTS:
        return None
    return f"{module} is below a public root; import from {_root_above(module)} instead"


def _from_import_defect(module: str, name: str, exported: dict[str, frozenset[str]]) -> str | None:
    """Why ``from <module> import <name>`` reaches below a public root, or None when it does not.

    Args:
        module: The module imported from.
        name: The name bound.
        exported: Each public root's ``__all__``, filled as roots are first named.

    Returns:
        One sentence, or None.
    """
    if not _in_package(module):
        return None
    if f"{module}.{name}" in PUBLIC_ROOTS:
        return None
    if module == _PACKAGE:
        return None if name in _exports(_PACKAGE, exported) else _undeclared(module, name, exported)
    if module not in PUBLIC_ROOTS:
        homes = _public_homes(name, exported)
        where = f"import it from {' or '.join(homes)}" if homes else "no public root exports it"
        return f"{module} is below a public root; {where}"
    if name == "*" or name not in _exports(module, exported):
        return _undeclared(module, name, exported)
    return None


def _undeclared(module: str, name: str, exported: dict[str, frozenset[str]]) -> str:
    """The reason for a name a root does not declare.

    Args:
        module: The root.
        name: The name.
        exported: The roots' ``__all__``.

    Returns:
        One sentence, naming where the name is public when it is.
    """
    if name == "*":
        return f"a star import binds whatever {module} happens to hold; import the names its __all__ declares"
    homes = [home for home in _public_homes(name, exported) if home != module]
    where = f"; it is public from {' or '.join(homes)}" if homes else "; no public root exports it"
    return f"{name} is not in {module}.__all__{where}"


def _exports(root: str, exported: dict[str, frozenset[str]]) -> frozenset[str]:
    """A root's ``__all__``, imported once.

    Args:
        root: A public root, or the package itself.
        exported: The cache.

    Returns:
        The names it declares.
    """
    if root not in exported:
        exported[root] = frozenset(getattr(importlib.import_module(root), "__all__", ()))
    return exported[root]


def _public_homes(name: str, exported: dict[str, frozenset[str]]) -> list[str]:
    """Every public root whose ``__all__`` declares ``name``.

    A root whose extra is not installed cannot be read and is passed over: it is a suggestion, and the
    finding stands either way.

    Args:
        name: The name.
        exported: The cache.

    Returns:
        The roots, in :data:`~threetears.evals.PUBLIC_ROOTS` order.
    """
    homes = []
    for root in PUBLIC_ROOTS:
        try:
            names = _exports(root, exported)
        # NOSILENT: a root whose extra is not installed drops out of the suggestion only; the finding stands.
        except ImportError:
            continue
        if name in names:
            homes.append(root)
    return homes


def _root_above(module: str) -> str:
    """The nearest public root above ``module``, or the package when none is.

    Args:
        module: A module below a root.

    Returns:
        The root's name.
    """
    parts = module.split(".")
    for end in range(len(parts) - 1, 1, -1):
        candidate = ".".join(parts[:end])
        if candidate in PUBLIC_ROOTS:
            return candidate
    return _PACKAGE


__all__ = ["BOUNDARY_RULES", "BoundaryFinding", "BoundaryRule", "nonpublic_evals_imports"]
