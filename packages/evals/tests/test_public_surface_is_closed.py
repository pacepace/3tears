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
* every exception named in a ``Raises:`` section of a public docstring.

An annotation token counts when it resolves to a class, or a module-level type alias, defined
under ``threetears.evals``. Exempt, because a host never needs the name: a ``TypeVar``, and an
``Annotated`` alias over builtins (``ModelProse`` is a ``str`` to every reader; its metadata is a
validator). A PRIVATE engine class reachable from a public signature fails as a leak — it cannot be
exported, so the signature has to change.

**What this cannot see.** A type that appears only inside a function body, a ``dict[str, Any]`` a
lens returns (its keys are not types), and a type named only in prose outside a ``Raises:`` section.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import re
import sys
import typing
from collections.abc import Iterator
from dataclasses import fields, is_dataclass
from typing import Annotated, Any, TypeVar, get_args, get_origin

import threetears.evals

#: The public roots a host imports from (the package matrix's list).
PUBLIC_ROOTS = (
    "threetears.evals.contracts",
    "threetears.evals.contracts.host",
    "threetears.evals.run",
    "threetears.evals.analysis",
    "threetears.evals.gen",
)

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_RAISES_ENTRY = re.compile(r"^\s{4}([A-Za-z_][A-Za-z0-9_.]*):")
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
            signature = inspect.signature(function)
        except TypeError, ValueError:
            return
        for parameter in signature.parameters.values():
            self._annotation(parameter.annotation, namespace, f"{label}({parameter.name})")
        self._annotation(signature.return_annotation, namespace, f"{label} -> return")
        self._raises(label, inspect.getdoc(function) or "", namespace)

    def _raises(self, label: str, doc: str, namespace: dict[str, Any]) -> None:
        section = re.search(r"^Raises:\n((?:[ \t]+.*\n?)+)", doc, re.MULTILINE)
        if not section:
            return
        for line in section.group(1).splitlines():
            entry = _RAISES_ENTRY.match(line)
            if not entry:
                continue
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
        module = sys.modules.get(home or "")
        namespace = dict(vars(module)) if module else {}
        if not inspect.isclass(value):
            if callable(value) and not _is_type_alias(value):
                self._callable(name, value, namespace)
            elif _is_type_alias(value):
                self._annotation(value, namespace, name)
            return
        for klass in value.__mro__:
            if not klass.__module__.startswith("threetears.evals"):
                continue
            klass_namespace = dict(vars(sys.modules[klass.__module__]))
            for field_name, annotation in klass.__dict__.get("__annotations__", {}).items():
                if field_name.startswith("_"):
                    continue
                self._annotation(annotation, klass_namespace, f"{name}.{field_name}")
        for member_name, member in vars(value).items():
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
