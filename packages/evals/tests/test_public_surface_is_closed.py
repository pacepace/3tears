"""Structural gate: every engine type a public name hands a host has a public name of its own.

A host imports from the public roots and from nothing below them (``test_package_matrix.py``). That
rule is only livable if the roots are CLOSED: a host implements the protocols a root's signatures
name, receives the values they return, catches the exceptions they document and annotates its own
code with all of them — under strict mypy, every one of those is a name it has to import. A type
reachable from a public signature but exported from no root leaves the host two choices, both
wrong: reach below the root, or type the value as ``Any``.

**What is walked.** Starting from every name in every public root's ``__all__``, and following each
engine type found:

* every parameter and return annotation of a public function, method, property or ``__init__``;
* every Pydantic field and dataclass field, and every annotation a Protocol declares — the
  class's own and those its engine-defined bases declare (a base itself is never named by a host,
  so it is walked through, not required);
* every exception named in a ``:raises`` field of a public docstring.

An annotation token counts when it resolves to a class, or a module-level type alias, defined
under ``threetears.evals``. It resolves against the namespace of the module that wrote it, together with the
engine names that module imports only under ``if TYPE_CHECKING:``: a type a signature names is the host's to name
however its module imports it, so a checker-only import hides nothing from the walk. Exempt, because a host never
needs the name: a ``TypeVar``, and an ``Annotated`` alias over builtins (``ModelProse`` is a ``str`` to every
reader; its metadata is a validator). A PRIVATE engine class reachable from a public signature fails as a leak — it
cannot be exported, so the signature has to change.

**What this cannot see.** A type that appears only inside a function body, a ``dict[str, Any]`` a
lens returns (its keys are not types), and a type named only in prose outside a ``:raises`` field.
"""

from __future__ import annotations

import annotationlib
import ast
import importlib
import importlib.util
import inspect
import pkgutil
import re
import sys
import typing
from collections.abc import Iterator
from dataclasses import fields, is_dataclass
from typing import Annotated, Any, TypeVar, get_args, get_origin

import threetears.evals
from threetears.evals.gen import ProposedDraft
from threetears.evals.kernel import DefinitionStore
from threetears.evals.run import create_rubric_dim
from threetears.evals.schema import RubricProposal

#: The public roots a host imports from: the package itself, whose ``__all__`` holds the quick path's names
#: (``Eval``, ``Result``, ...), then the package's own tuple, which the package matrix test holds equal to the roots
#: it enforces. Read rather than copied, because a copy here once dropped ``analysis.viz`` and left two of its
#: reachable types unexported with this test green.
PUBLIC_ROOTS = ("threetears.evals", *threetears.evals.PUBLIC_ROOTS)

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_RAISES_ENTRY = re.compile(r"^:raises ([A-Za-z_][A-Za-z0-9_.]*):", re.MULTILINE)
_BUILTIN_BASES = (str, int, float, bool, bytes, dict, list, tuple, set, frozenset, type(None))


def _engine_modules() -> Iterator[Any]:
    for info in pkgutil.walk_packages(threetears.evals.__path__, "threetears.evals."):
        yield importlib.import_module(info.name)


def _module_level_aliases() -> dict[int, tuple[str, str]]:
    """``id(alias) -> (module, name)`` for every module-level type alias the engine defines."""
    aliases: dict[int, tuple[str, str]] = {}
    for module in _engine_modules():
        tree = ast.parse(inspect.getsource(module))
        for node in tree.body:
            names: list[str] = []
            if isinstance(node, ast.Assign):
                names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names = [node.target.id]
            elif isinstance(node, ast.TypeAlias):
                names = [node.name.id]
            for name in names:
                value = getattr(module, name, None)
                if _is_type_alias(value):
                    aliases.setdefault(id(value), (module.__name__, name))
    return aliases


def _is_type_checking(test: ast.expr) -> bool:
    """``TYPE_CHECKING`` or ``typing.TYPE_CHECKING``: the test of a block only a type checker reads."""
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _checker_only_names(module: Any) -> dict[str, Any]:
    """Each engine name ``module`` imports under ``if TYPE_CHECKING:``, resolved to what the import would bind.

    Only imports from ``threetears.evals`` are resolved: an annotation token counts only when it names an engine
    class or alias, so a third-party name a module imports for its checker can never be one.
    """
    names: dict[str, Any] = {}
    for node in ast.walk(ast.parse(inspect.getsource(module))):
        if not (isinstance(node, ast.If) and _is_type_checking(node.test)):
            continue
        for statement in node.body:
            if isinstance(statement, ast.ImportFrom):
                source_name = importlib.util.resolve_name(
                    "." * statement.level + (statement.module or ""), module.__package__
                )
                if not source_name.startswith("threetears.evals"):
                    continue
                source = importlib.import_module(source_name)
                for alias in statement.names:
                    bound = getattr(source, alias.name, None)
                    if bound is None:
                        bound = importlib.import_module(f"{source_name}.{alias.name}")
                    names[alias.asname or alias.name] = bound
            elif isinstance(statement, ast.Import):
                for alias in statement.names:
                    if alias.name.startswith("threetears.evals") and alias.asname is not None:
                        names[alias.asname] = importlib.import_module(alias.name)
    return names


def _is_type_alias(value: object) -> bool:
    if isinstance(value, typing.TypeAliasType):
        return True
    origin = get_origin(value)
    return origin is not None and not isinstance(value, type)


def _exempt_alias(value: object) -> bool:
    """An ``Annotated`` alias over builtins: a host types it as the builtin."""
    if get_origin(value) is not Annotated:
        return False
    base = get_args(value)[0]
    base_origin = get_origin(base) or base
    return isinstance(base_origin, type) and issubclass(base_origin, _BUILTIN_BASES)


def _engine_classes_by_name() -> dict[str, list[type]]:
    found: dict[str, list[type]] = {}
    for module in _engine_modules():
        for name, value in vars(module).items():
            if inspect.isclass(value) and value.__module__ == module.__name__:
                found.setdefault(name, []).append(value)
    return found


class _Walk:
    """The reachability walk, recording each engine type reached and the first path to it."""

    def __init__(self) -> None:
        self.aliases = _module_level_aliases()
        self.classes_by_name = _engine_classes_by_name()
        self.exported: set[int] = set()
        for root in PUBLIC_ROOTS:
            module = importlib.import_module(root)
            self.exported |= {id(getattr(module, name)) for name in module.__all__}
        self.reached: dict[int, tuple[str, str]] = {}
        self._queue: list[tuple[str, object]] = []
        self._visited: set[int] = set()
        self._namespaces: dict[str, dict[str, Any]] = {}

    def _namespace(self, module_name: str) -> dict[str, Any]:
        """The names an annotation written in ``module_name`` resolves against: its globals, over its checker-only
        imports when it is an engine module (empty for a module that is not loaded)."""
        if module_name not in self._namespaces:
            module = sys.modules.get(module_name)
            engine = module_name == "threetears.evals" or module_name.startswith("threetears.evals.")
            checker_only = _checker_only_names(module) if module and engine else {}
            self._namespaces[module_name] = {**checker_only, **vars(module)} if module else {}
        return self._namespaces[module_name]

    def run(self) -> None:
        for root in PUBLIC_ROOTS:
            module = importlib.import_module(root)
            for name in module.__all__:
                self._enqueue(name, getattr(module, name))
        while self._queue:
            name, value = self._queue.pop()
            self._scan(name, value)

    def _enqueue(self, name: str, value: object) -> None:
        if id(value) not in self._visited:
            self._visited.add(id(value))
            self._queue.append((name, value))

    def _reach(self, value: object, label: str, via: str) -> None:
        self.reached.setdefault(id(value), (label, via))
        self._enqueue(label.rsplit(".", 1)[-1], value)

    def _token(self, token: str, namespace: dict[str, Any], via: str) -> None:
        value = namespace.get(token)
        if value is None or isinstance(value, TypeVar):
            return
        if inspect.isclass(value) and value.__module__.startswith("threetears.evals"):
            self._reach(value, f"{value.__module__}.{value.__qualname__}", via)
        elif id(value) in self.aliases and not _exempt_alias(value):
            module, name = self.aliases[id(value)]
            self._reach(value, f"{module}.{name}", via)

    def _annotation(self, annotation: object, namespace: dict[str, Any], via: str) -> None:
        if annotation is inspect.Parameter.empty or annotation is None:
            return
        if isinstance(annotation, str):
            tokens = _IDENT.findall(annotation)
        elif inspect.isclass(annotation):
            tokens = [annotation.__name__]
            namespace = {**namespace, annotation.__name__: annotation}
        else:
            tokens = _IDENT.findall(repr(annotation))
        for token in tokens:
            self._token(token, namespace, via)

    def _callable(self, label: str, function: Any, namespace: dict[str, Any]) -> None:
        try:
            signature = inspect.signature(function, annotation_format=annotationlib.Format.STRING)
        except TypeError, ValueError:
            return
        for parameter in signature.parameters.values():
            self._annotation(parameter.annotation, namespace, f"{label}({parameter.name})")
        self._annotation(signature.return_annotation, namespace, f"{label} -> return")
        self._raises(label, inspect.getdoc(function) or "", namespace)

    def _raises(self, label: str, doc: str, namespace: dict[str, Any]) -> None:
        for entry in _RAISES_ENTRY.finditer(doc):
            name = entry.group(1).rsplit(".", 1)[-1]
            candidates = [namespace[name]] if inspect.isclass(namespace.get(name)) else []
            candidates = candidates or self.classes_by_name.get(name, [])
            for value in candidates:
                if value.__module__.startswith("threetears.evals"):
                    self._reach(value, f"{value.__module__}.{value.__qualname__}", f"{label} raises")

    def _scan(self, name: str, value: object) -> None:
        # An alias's own ``__module__`` names where its ORIGIN lives (``collections.abc`` for a
        # ``Callable``), so its tokens are resolved where the engine defined it.
        home = self.aliases[id(value)][0] if id(value) in self.aliases else getattr(value, "__module__", "")
        namespace = self._namespace(home or "")
        if not inspect.isclass(value):
            if callable(value) and not _is_type_alias(value):
                self._callable(name, value, namespace)
            elif _is_type_alias(value):
                self._annotation(value, namespace, name)
            return
        for klass in value.__mro__:
            if not klass.__module__.startswith("threetears.evals"):
                continue
            klass_namespace = self._namespace(klass.__module__)
            # Read through ``annotationlib`` as strings: a ``NamedTuple``, and any class whose module does not defer
            # its annotations, keeps them behind an annotate function rather than in its ``__dict__``, and a
            # checker-only name cannot be evaluated.
            own_annotations = annotationlib.get_annotations(klass, format=annotationlib.Format.STRING)
            for field_name, annotation in own_annotations.items():
                if field_name.startswith("_"):
                    continue
                self._annotation(annotation, klass_namespace, f"{name}.{field_name}")
        # A copy: reading a member's signature may store the annotations it computes in the class's ``__dict__``.
        for member_name, member in list(vars(value).items()):
            if member_name.startswith("_") and member_name not in ("__init__", "__call__"):
                continue
            label = f"{name}.{member_name}"
            if isinstance(member, property) and member.fget is not None:
                self._callable(label, member.fget, namespace)
            elif isinstance(member, (staticmethod, classmethod)):
                self._callable(label, member.__func__, namespace)
            elif inspect.isfunction(member):
                self._callable(label, member, namespace)
        self._raises(name, inspect.getdoc(value) or "", namespace)
        if is_dataclass(value):
            for field in fields(value):
                if field.init and field.name.startswith("_"):
                    self.reached.setdefault(id(field), (f"{value.__qualname__}.{field.name}", f"{name}.__init__"))


def _walk() -> _Walk:
    walk = _Walk()
    walk.run()
    return walk


def test_every_engine_type_a_public_name_reaches_is_exported_from_a_public_root() -> None:
    walk = _walk()
    unexported = sorted(
        f"{label}  (reached via {via})"
        for key, (label, via) in walk.reached.items()
        if key not in walk.exported and not label.rsplit(".", 1)[-1].startswith("_")
    )
    assert not unexported, (
        "engine types a host receives, implements, catches or must name are exported from no public root:\n  "
        + "\n  ".join(unexported)
    )


def test_no_private_engine_type_is_reachable_from_a_public_name() -> None:
    walk = _walk()
    leaked = sorted(
        f"{label}  (reached via {via})"
        for key, (label, via) in walk.reached.items()
        if label.rsplit(".", 1)[-1].startswith("_")
    )
    assert not leaked, "a public signature hands a host a private engine type:\n  " + "\n  ".join(leaked)


def test_the_walk_reads_a_signature_type_imported_only_for_the_type_checker() -> None:
    # ``create_rubric_dim`` types its ``storage`` as a ``DefinitionStore`` its module imports only under
    # ``if TYPE_CHECKING:``, so the module's own namespace never holds the name and only the checker-only read finds it.
    assert "DefinitionStore" not in vars(sys.modules[create_rubric_dim.__module__]), (
        "create_rubric_dim's module imports DefinitionStore at run time now: pin a signature whose type is checker-only"
    )
    assert id(DefinitionStore) in _walk().reached, "the walk did not reach a type a signature imports for the checker"


def test_the_walk_reads_the_fields_of_a_named_tuple() -> None:
    # A ``NamedTuple`` keeps its fields' annotations behind an annotate function, not in its ``__dict__``, so
    # ``ProposedDraft.proposal`` is the only path to ``RubricProposal`` that reading the class's dict would miss.
    assert tuple in ProposedDraft.__mro__ and "__annotations__" not in ProposedDraft.__dict__
    assert (
        "RubricProposal" in annotationlib.get_annotations(ProposedDraft, format=annotationlib.Format.STRING)["proposal"]
    )
    assert id(RubricProposal) in _walk().reached, "the walk did not reach a type named only in a NamedTuple's field"
