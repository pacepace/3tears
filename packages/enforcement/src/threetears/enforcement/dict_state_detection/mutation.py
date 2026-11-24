"""walker: an object's dict or set state changed in place after ``__init__``.

The ``__init__`` walker (:mod:`.walkers`) asks where state lives (an L1 backend or NATS KV, not a raw dict). This one
asks how it changes: in the trees a repo names (:attr:`DictStateConfig.immutable_roots`), a ``self._x`` that holds a
dict or a set is never changed in place outside ``__init__``. Its owner builds the next value (a
:class:`~types.MappingProxyType`, a ``frozenset``, a tuple) and assigns it whole, so a reader holding the old value (a
status, a coroutine across an await, a thread) never sees it change under it. There is no allowlist: "small, owner-only,
under the lock" is no exception (Pace, 2026-11-23).

An attribute counts as a dict or a set when any method of its class assigns it one: a dict or set display or
comprehension, a ``dict()`` / ``set()`` / ``OrderedDict()`` / ``defaultdict()`` / ``Counter()`` / ``dict.fromkeys()``
call, or an annotation naming one of those (or ``MutableMapping`` / ``MutableSet``). A change in place is, outside
``__init__``:

- ``self._x[k] = v``, ``self._x[k][j] = v``, ``self._x[k] += v``, ``del self._x[k]``;
- ``self._x |= ...`` (every augmented assignment to the attribute itself, which a dict or set does in place);
- a mutating method call on it or on an entry of it: ``self._x.pop(k)``, ``self._x[k].append(v)``, ...;
- the same through a name bound in the method to it or to an entry of it: ``waits = self._x.setdefault(t, {})`` then
  ``waits[f] = w``.
"""

from __future__ import annotations

import ast
from pathlib import Path

from threetears.enforcement.common import Violation, iter_python_files, parse_python_file, relative_posix_path
from threetears.enforcement.dict_state_detection.config import DictStateAllowlistEntry

__all__ = ["find_mutated_state_violations", "find_stale_immutable_debts"]

_CATEGORY = "dict_state_detection.mutated_in_place"
_STALE_CATEGORY = "dict_state_detection.stale_immutable_debt"

#: the constructors of a dict or set
_CONSTRUCTORS: frozenset[str] = frozenset({"dict", "set", "OrderedDict", "defaultdict", "Counter"})

#: the annotations of a dict or set
_ANNOTATIONS: frozenset[str] = frozenset(
    {"dict", "set", "OrderedDict", "defaultdict", "Counter", "MutableMapping", "MutableSet"}
)

#: the methods that change a dict, a set or a list in place
_MUTATORS: frozenset[str] = frozenset(
    {
        "add",
        "append",
        "clear",
        "difference_update",
        "discard",
        "extend",
        "insert",
        "intersection_update",
        "pop",
        "popitem",
        "remove",
        "setdefault",
        "symmetric_difference_update",
        "update",
        "__setitem__",
        "__delitem__",
    }
)

#: the reads that hand back an entry of a dict (bound to a name, a change of the name is a change of the entry)
_ENTRY_READS: frozenset[str] = frozenset({"get", "setdefault"})


def _self_attr(node: ast.expr) -> str | None:
    """``x`` of ``self._x``; None for anything else.

    :param node: the expression
    :ptype node: ast.expr
    :return: the attribute's name
    :rtype: str | None
    """
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr.startswith("_")
    ):
        return node.attr
    return None


def _is_mutable_value(node: ast.expr) -> bool:
    """whether ``node`` builds a dict or a set.

    :param node: the expression
    :ptype node: ast.expr
    :return: whether it does
    :rtype: bool
    """
    if isinstance(node, (ast.Dict, ast.DictComp, ast.Set, ast.SetComp)):
        return True
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name) and func.id in _CONSTRUCTORS:
            return True
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "dict":
            return True
    if isinstance(node, ast.BoolOp):
        return any(_is_mutable_value(value) for value in node.values)
    return False


def _is_mutable_annotation(node: ast.expr) -> bool:
    """whether an annotation names a dict or a set (``dict[str, int]``, ``set[str] | None``, ...).

    :param node: the annotation
    :ptype node: ast.expr
    :return: whether it does
    :rtype: bool
    """
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _is_mutable_annotation(node.left) or _is_mutable_annotation(node.right)
    if isinstance(node, ast.Subscript):
        node = node.value
    if isinstance(node, ast.Name):
        return node.id in _ANNOTATIONS
    if isinstance(node, ast.Attribute):
        return node.attr in _ANNOTATIONS
    return False


def _methods(cls: ast.ClassDef) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [child for child in cls.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _mutable_attrs(cls: ast.ClassDef) -> frozenset[str]:
    """the attributes any method of ``cls`` assigns a dict or a set.

    :param cls: the class
    :ptype cls: ast.ClassDef
    :return: their names
    :rtype: frozenset[str]
    """
    found: set[str] = set()
    for method in _methods(cls):
        for stmt in ast.walk(method):
            if isinstance(stmt, ast.AnnAssign):
                name = _self_attr(stmt.target)
                if name is not None and (
                    _is_mutable_annotation(stmt.annotation)
                    or (stmt.value is not None and _is_mutable_value(stmt.value))
                ):
                    found.add(name)
            elif isinstance(stmt, ast.Assign) and _is_mutable_value(stmt.value):
                for target in stmt.targets:
                    name = _self_attr(target)
                    if name is not None:
                        found.add(name)
    return frozenset(found)


def _reached(node: ast.expr, attrs: frozenset[str], aliases: dict[str, str]) -> str | None:
    """the state attribute ``node`` is, or is an entry of (through subscripts, or a name bound to either).

    :param node: the expression
    :ptype node: ast.expr
    :param attrs: the dict or set attributes
    :ptype attrs: frozenset[str]
    :param aliases: the method's names bound to one, onto its attribute
    :ptype aliases: dict[str, str]
    :return: the attribute's name, or None
    :rtype: str | None
    """
    while isinstance(node, ast.Subscript):
        node = node.value
    if isinstance(node, ast.Name):
        return aliases.get(node.id)
    name = _self_attr(node)
    return name if name in attrs else None


def _entry_of(node: ast.expr, attrs: frozenset[str], aliases: dict[str, str]) -> str | None:
    """the state attribute ``node`` hands back (itself, an entry by subscript or by ``.get`` / ``.setdefault``).

    :param node: the bound value
    :ptype node: ast.expr
    :param attrs: the dict or set attributes
    :ptype attrs: frozenset[str]
    :param aliases: the method's names bound to one, onto its attribute
    :ptype aliases: dict[str, str]
    :return: the attribute's name, or None
    :rtype: str | None
    """
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _ENTRY_READS:
        return _reached(node.func.value, attrs, aliases)
    return _reached(node, attrs, aliases)


def _changed(stmt: ast.AST, attrs: frozenset[str], aliases: dict[str, str]) -> str | None:
    """the state attribute ``stmt`` changes in place, or None.

    :param stmt: the node
    :ptype stmt: ast.AST
    :param attrs: the dict or set attributes
    :ptype attrs: frozenset[str]
    :param aliases: the method's names bound to one, onto its attribute
    :ptype aliases: dict[str, str]
    :return: the attribute's name, or None
    :rtype: str | None
    """
    if isinstance(stmt, (ast.Assign, ast.Delete)):
        for target in stmt.targets:
            if isinstance(target, ast.Subscript):
                name = _reached(target, attrs, aliases)
                if name is not None:
                    return name
    elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Subscript):
        return _reached(stmt.target, attrs, aliases)
    elif isinstance(stmt, ast.AugAssign):
        # a name bound to an entry is rebound by an augmented assignment (``count += 1``), never changed in place
        target = stmt.target
        if isinstance(target, ast.Subscript):
            return _reached(target, attrs, aliases)
        name = _self_attr(target)
        return name if name in attrs else None
    elif isinstance(stmt, ast.Call) and isinstance(stmt.func, ast.Attribute) and stmt.func.attr in _MUTATORS:
        return _entry_of(stmt.func.value, attrs, aliases)
    return None


def _scan_method(
    cls: ast.ClassDef,
    method: ast.FunctionDef | ast.AsyncFunctionDef,
    attrs: frozenset[str],
    module_path: Path,
) -> list[Violation]:
    """the changes in place of ``attrs`` in one method.

    :param cls: the class
    :ptype cls: ast.ClassDef
    :param method: the method
    :ptype method: ast.FunctionDef | ast.AsyncFunctionDef
    :param attrs: the dict or set attributes
    :ptype attrs: frozenset[str]
    :param module_path: the module
    :ptype module_path: Path
    :return: the violations, in source order
    :rtype: list[Violation]
    """
    aliases: dict[str, str] = {}
    for stmt in ast.walk(method):
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            name = _entry_of(stmt.value, attrs, aliases)
            if name is not None:
                aliases[stmt.targets[0].id] = name
    found: list[Violation] = []
    for stmt in ast.walk(method):
        name = _changed(stmt, attrs, aliases)
        if name is None:
            continue
        found.append(
            Violation(
                category=_CATEGORY,
                file=module_path,
                line=getattr(stmt, "lineno", method.lineno),
                symbol=f"{cls.name}.{name}",
                reason=(
                    f"{cls.name}.{name} holds a dict or set changed in place in {method.name}(): keep it an "
                    f"immutable value (MappingProxyType, frozenset, tuple; threetears.core.collections.frozen_state) "
                    f"and assign the next one whole"
                ),
            )
        )
    return sorted(found, key=lambda violation: violation.line)


def find_mutated_state_violations(roots: tuple[Path, ...], repo_root: Path) -> list[Violation]:
    """every change in place of an object's dict or set state outside ``__init__`` under ``roots``.

    :param roots: the trees (directories or single modules) where state is immutable
    :ptype roots: tuple[Path, ...]
    :param repo_root: the repo's root (kept for parity with the sibling walkers)
    :ptype repo_root: Path
    :return: the violations, root-major, then file, then line
    :rtype: list[Violation]
    """
    _ = repo_root
    violations: list[Violation] = []
    for root in roots:
        modules = [root] if root.is_file() else list(iter_python_files(root))
        for module_path in modules:
            tree = parse_python_file(module_path)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                attrs = _mutable_attrs(node)
                if not attrs:
                    continue
                for method in _methods(node):
                    if method.name != "__init__":
                        violations.extend(_scan_method(node, method, attrs, module_path))
    return violations


def find_stale_immutable_debts(
    roots: tuple[Path, ...], debts: tuple[DictStateAllowlistEntry, ...], repo_root: Path
) -> list[Violation]:
    """each tracked change in place that no longer happens: the redesign landed, so the entry goes.

    :param roots: the trees where state is immutable
    :ptype roots: tuple[Path, ...]
    :param debts: the tracked changes in place
    :ptype debts: tuple[DictStateAllowlistEntry, ...]
    :param repo_root: the repo's root, which the entries' files are relative to
    :ptype repo_root: Path
    :return: one violation a stale entry
    :rtype: list[Violation]
    """
    if not debts:
        return []
    real = {
        (relative_posix_path(violation.file, repo_root), violation.symbol)
        for violation in find_mutated_state_violations(roots, repo_root)
    }
    return [
        Violation(
            category=_STALE_CATEGORY,
            file=repo_root / debt.file,
            line=1,
            symbol=f"{debt.class_name}.{debt.attr_name}",
            reason=(
                f"immutable_known_violations entry {debt.file}:{debt.class_name}.{debt.attr_name} no longer changes "
                f"in place; drop the entry"
            ),
        )
        for debt in debts
        if (debt.file, f"{debt.class_name}.{debt.attr_name}") not in real
    ]
