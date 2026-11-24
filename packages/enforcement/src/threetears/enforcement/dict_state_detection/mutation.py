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

The widened walk (``widened=True``; Pace's ruling of 2026-11-24, opt-in per repo through
:attr:`DictStateConfig.immutable_widened` until each repo's sites are paid) closes three gaps:

- (a) public attributes count too (``self.documents``, not only ``self._documents``);
- (b) lists and deques count too (a list display or comprehension, ``list()`` / ``deque()``, an annotation naming
  ``list`` / ``deque`` / ``MutableSequence`` or a ``typing`` alias), and so do their changes: ``popleft``,
  ``appendleft``, ``extendleft``, ``rotate``, ``sort``, ``reverse``, slice assignment, and ``OrderedDict.move_to_end``;
- (c) class-body fields count too: ``x: dict[str, int] = field(default_factory=dict)``, an annotated field with no
  value (a dataclass's), ``ClassVar`` / ``Final`` / ``Optional`` wrappers, and a mutable class attribute (``_seen =
  set()``), changed through ``self`` or ``cls`` outside ``__init__`` / ``__post_init__``; and a change made from
  anywhere else in the same module through another object (``held.order.move_to_end(k)``) to a class-body field or a
  private attribute of a class there.

Immutable values (tuple, frozenset, ``MappingProxyType``, ``Mapping`` / ``Sequence`` annotations) never count, and
neither does a function-local container. Nor, in the widened walk, does the state of a call-scoped class (Pace,
2026-11-24: function-local and call-scoped mutation is safe): one whose every instance lives and dies inside one call,
as :mod:`.call_scope` decides across all the walked trees, so no other coroutine or thread can reach it.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from threetears.enforcement.common import Violation, iter_python_files, parse_python_file, relative_posix_path
from threetears.enforcement.dict_state_detection.call_scope import call_scoped_classes
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

#: gap (b): the constructors of a list or a deque
_SEQUENCE_CONSTRUCTORS: frozenset[str] = frozenset({"list", "deque"})

#: gap (b): the annotations of a list or a deque, and the ``typing`` aliases of every mutable container
_SEQUENCE_ANNOTATIONS: frozenset[str] = frozenset(
    {"list", "deque", "MutableSequence", "List", "Deque", "Dict", "Set", "DefaultDict"}
)

#: gap (b): the methods that change a list, a deque or an OrderedDict in place
_SEQUENCE_MUTATORS: frozenset[str] = frozenset(
    {"appendleft", "extendleft", "popleft", "rotate", "sort", "reverse", "move_to_end"}
)

#: gap (c): annotation wrappers whose first argument is the field's real type
_WRAPPERS: frozenset[str] = frozenset({"ClassVar", "Final", "Optional", "Annotated"})

#: the reads that hand back an entry of a dict (bound to a name, a change of the name is a change of the entry)
_ENTRY_READS: frozenset[str] = frozenset({"get", "setdefault"})


@dataclass(frozen=True)
class _Rules:
    """what one walk counts: the narrow (enforced) walk, or the widened one of 2026-11-24."""

    widened: bool
    constructors: frozenset[str]
    annotations: frozenset[str]
    mutators: frozenset[str]
    receivers: frozenset[str]
    builders: frozenset[str]


_NARROW = _Rules(
    widened=False,
    constructors=_CONSTRUCTORS,
    annotations=_ANNOTATIONS,
    mutators=_MUTATORS,
    receivers=frozenset({"self"}),
    builders=frozenset({"__init__"}),
)

_WIDE = _Rules(
    widened=True,
    constructors=_CONSTRUCTORS | _SEQUENCE_CONSTRUCTORS,
    annotations=_ANNOTATIONS | _SEQUENCE_ANNOTATIONS,
    mutators=_MUTATORS | _SEQUENCE_MUTATORS,
    receivers=frozenset({"self", "cls"}),
    builders=frozenset({"__init__", "__post_init__"}),
)


@dataclass(frozen=True)
class _Scope:
    """what a change in one function can reach: its class's own attributes, and the module's shared fields.

    :ivar owner: the function's class, or None for a module function
    :ivar own: the attributes of ``owner`` whose change through ``self`` / ``cls`` counts (empty in a builder)
    :ivar shared: gap (c): each class-body field or private attribute of the module, with the classes holding it
    """

    owner: str | None
    own: frozenset[str]
    shared: tuple[tuple[str, str], ...]


def _receiver_attr(node: ast.expr, rules: _Rules) -> str | None:
    """``x`` of ``self._x`` (widened: of ``self.x`` / ``cls.x`` too); None for anything else.

    :param node: the expression
    :ptype node: ast.expr
    :param rules: the walk
    :ptype rules: _Rules
    :return: the attribute's name
    :rtype: str | None
    """
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in rules.receivers
        and (rules.widened or node.attr.startswith("_"))
    ):
        return node.attr
    return None


def _is_mutable_value(node: ast.expr, rules: _Rules) -> bool:
    """whether ``node`` builds a dict or a set (widened: a list or a deque too).

    :param node: the expression
    :ptype node: ast.expr
    :param rules: the walk
    :ptype rules: _Rules
    :return: whether it does
    :rtype: bool
    """
    if isinstance(node, (ast.Dict, ast.DictComp, ast.Set, ast.SetComp)):
        return True
    if rules.widened and isinstance(node, (ast.List, ast.ListComp)):
        return True
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name) and func.id in rules.constructors:
            return True
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "dict":
            return True
        if rules.widened:
            if isinstance(func, ast.Attribute) and func.attr in rules.constructors:
                return True
            if isinstance(func, ast.Name) and func.id == "field":
                return any(
                    keyword.arg == "default_factory" and _is_factory(keyword.value, rules) for keyword in node.keywords
                )
    if isinstance(node, ast.BoolOp):
        return any(_is_mutable_value(value, rules) for value in node.values)
    if rules.widened and isinstance(node, ast.IfExp):
        return _is_mutable_value(node.body, rules) or _is_mutable_value(node.orelse, rules)
    return False


def _is_factory(node: ast.expr, rules: _Rules) -> bool:
    """whether a ``default_factory`` builds a mutable container (``dict``, ``deque``, ``lambda: {...}``).

    :param node: the factory
    :ptype node: ast.expr
    :param rules: the walk
    :ptype rules: _Rules
    :return: whether it does
    :rtype: bool
    """
    if isinstance(node, ast.Name):
        return node.id in rules.constructors
    if isinstance(node, ast.Attribute):
        return node.attr in rules.constructors
    if isinstance(node, ast.Lambda):
        return _is_mutable_value(node.body, rules)
    return False


def _is_mutable_annotation(node: ast.expr, rules: _Rules) -> bool:
    """whether an annotation names a dict or a set (``dict[str, int]``, ``set[str] | None``, ...).

    :param node: the annotation
    :ptype node: ast.expr
    :param rules: the walk
    :ptype rules: _Rules
    :return: whether it does
    :rtype: bool
    """
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _is_mutable_annotation(node.left, rules) or _is_mutable_annotation(node.right, rules)
    if rules.widened and isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            parsed = ast.parse(node.value, mode="eval").body
        except SyntaxError:
            return False
        return _is_mutable_annotation(parsed, rules)
    if isinstance(node, ast.Subscript):
        wrapper = node.value
        wrapper_name = wrapper.id if isinstance(wrapper, ast.Name) else getattr(wrapper, "attr", None)
        if rules.widened and wrapper_name in _WRAPPERS:
            inner = node.slice
            if isinstance(inner, ast.Tuple) and inner.elts:
                inner = inner.elts[0]
            return _is_mutable_annotation(inner, rules)
        node = node.value
    if isinstance(node, ast.Name):
        return node.id in rules.annotations
    if isinstance(node, ast.Attribute):
        return node.attr in rules.annotations
    return False


def _methods(cls: ast.ClassDef) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [child for child in cls.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _mutable_attrs(cls: ast.ClassDef, rules: _Rules) -> frozenset[str]:
    """the attributes any method of ``cls`` assigns a dict or a set (widened: a list or a deque too).

    :param cls: the class
    :ptype cls: ast.ClassDef
    :param rules: the walk
    :ptype rules: _Rules
    :return: their names
    :rtype: frozenset[str]
    """
    found: set[str] = set()
    for method in _methods(cls):
        for stmt in ast.walk(method):
            if isinstance(stmt, ast.AnnAssign):
                name = _receiver_attr(stmt.target, rules)
                if name is not None and (
                    _is_mutable_annotation(stmt.annotation, rules)
                    or (stmt.value is not None and _is_mutable_value(stmt.value, rules))
                ):
                    found.add(name)
            elif isinstance(stmt, ast.Assign) and _is_mutable_value(stmt.value, rules):
                for target in stmt.targets:
                    name = _receiver_attr(target, rules)
                    if name is not None:
                        found.add(name)
    return frozenset(found)


def _class_fields(cls: ast.ClassDef, rules: _Rules) -> frozenset[str]:
    """gap (c): the class-body fields of ``cls`` holding a mutable container (annotated as one, or assigned one).

    :param cls: the class
    :ptype cls: ast.ClassDef
    :param rules: the walk
    :ptype rules: _Rules
    :return: their names
    :rtype: frozenset[str]
    """
    found: set[str] = set()
    for stmt in cls.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            if _is_mutable_annotation(stmt.annotation, rules) or (
                stmt.value is not None and _is_mutable_value(stmt.value, rules)
            ):
                found.add(stmt.target.id)
        elif isinstance(stmt, ast.Assign) and _is_mutable_value(stmt.value, rules):
            found.update(target.id for target in stmt.targets if isinstance(target, ast.Name))
    return frozenset(found)


def _reached(node: ast.expr, scope: _Scope, aliases: dict[str, str], rules: _Rules) -> str | None:
    """the state ``node`` is, or is an entry of (through subscripts, or a name bound to either), as ``Class.attr``.

    :param node: the expression
    :ptype node: ast.expr
    :param scope: what the function can reach
    :ptype scope: _Scope
    :param aliases: the function's names bound to state, onto its ``Class.attr``
    :ptype aliases: dict[str, str]
    :param rules: the walk
    :ptype rules: _Rules
    :return: the state's ``Class.attr``, or None
    :rtype: str | None
    """
    while isinstance(node, ast.Subscript):
        node = node.value
    if isinstance(node, ast.Name):
        return aliases.get(node.id)
    if not isinstance(node, ast.Attribute):
        return None
    if isinstance(node.value, ast.Name) and node.value.id in rules.receivers:
        name = _receiver_attr(node, rules)
        if name is not None and name in scope.own:
            return f"{scope.owner}.{name}"
        return None
    for attr, holders in scope.shared:
        if attr == node.attr:
            return f"{holders}.{attr}"
    return None


def _entry_of(node: ast.expr, scope: _Scope, aliases: dict[str, str], rules: _Rules) -> str | None:
    """the state ``node`` hands back (itself, an entry by subscript or by ``.get`` / ``.setdefault``).

    :param node: the bound value
    :ptype node: ast.expr
    :param scope: what the function can reach
    :ptype scope: _Scope
    :param aliases: the function's names bound to state, onto its ``Class.attr``
    :ptype aliases: dict[str, str]
    :param rules: the walk
    :ptype rules: _Rules
    :return: the state's ``Class.attr``, or None
    :rtype: str | None
    """
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _ENTRY_READS:
        return _reached(node.func.value, scope, aliases, rules)
    return _reached(node, scope, aliases, rules)


def _changed(stmt: ast.AST, scope: _Scope, aliases: dict[str, str], rules: _Rules) -> str | None:
    """the state ``stmt`` changes in place (as ``Class.attr``), or None.

    :param stmt: the node
    :ptype stmt: ast.AST
    :param scope: what the function can reach
    :ptype scope: _Scope
    :param aliases: the function's names bound to state, onto its ``Class.attr``
    :ptype aliases: dict[str, str]
    :param rules: the walk
    :ptype rules: _Rules
    :return: the state's ``Class.attr``, or None
    :rtype: str | None
    """
    if isinstance(stmt, (ast.Assign, ast.Delete)):
        for target in stmt.targets:
            if isinstance(target, ast.Subscript):
                name = _reached(target, scope, aliases, rules)
                if name is not None:
                    return name
    elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Subscript):
        return _reached(stmt.target, scope, aliases, rules)
    elif isinstance(stmt, ast.AugAssign):
        # a name bound to an entry is rebound by an augmented assignment (``count += 1``), never changed in place
        target = stmt.target
        if isinstance(target, (ast.Subscript, ast.Attribute)):
            return _reached(target, scope, aliases, rules)
    elif isinstance(stmt, ast.Call) and isinstance(stmt.func, ast.Attribute) and stmt.func.attr in rules.mutators:
        return _entry_of(stmt.func.value, scope, aliases, rules)
    return None


def _scan_function(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    scope: _Scope,
    module_path: Path,
    rules: _Rules,
) -> list[Violation]:
    """the changes in place of the state ``scope`` reaches, in one function.

    :param function: the method or module function
    :ptype function: ast.FunctionDef | ast.AsyncFunctionDef
    :param scope: what it can reach
    :ptype scope: _Scope
    :param module_path: the module
    :ptype module_path: Path
    :param rules: the walk
    :ptype rules: _Rules
    :return: the violations, in source order
    :rtype: list[Violation]
    """
    aliases: dict[str, str] = {}
    for stmt in ast.walk(function):
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            name = _entry_of(stmt.value, scope, aliases, rules)
            if name is not None:
                aliases[stmt.targets[0].id] = name
    kinds = "dict, set, list or deque" if rules.widened else "dict or set"
    found: list[Violation] = []
    for stmt in ast.walk(function):
        symbol = _changed(stmt, scope, aliases, rules)
        if symbol is None:
            continue
        found.append(
            Violation(
                category=_CATEGORY,
                file=module_path,
                line=getattr(stmt, "lineno", function.lineno),
                symbol=symbol,
                reason=(
                    f"{symbol} holds a {kinds} changed in place in {function.name}(): keep it an "
                    f"immutable value (MappingProxyType, frozenset, tuple; threetears.core.frozen_state) "
                    f"and assign the next one whole"
                ),
            )
        )
    return sorted(found, key=lambda violation: violation.line)


def _module_functions(body: list[ast.stmt]) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """the functions of a module outside every class (under ``if`` / ``try`` / ``with`` too).

    :param body: the statements
    :ptype body: list[ast.stmt]
    :return: the functions
    :rtype: list[ast.FunctionDef | ast.AsyncFunctionDef]
    """
    found: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for stmt in body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found.append(stmt)
        elif not isinstance(stmt, ast.ClassDef):
            for field_name in ("body", "orelse", "finalbody"):
                inner = getattr(stmt, field_name, None)
                if isinstance(inner, list):
                    found.extend(_module_functions([child for child in inner if isinstance(child, ast.stmt)]))
            for handler in getattr(stmt, "handlers", None) or ():
                if isinstance(handler, ast.ExceptHandler):
                    found.extend(_module_functions(handler.body))
    return found


def _scan_module(tree: ast.Module, module_path: Path, rules: _Rules) -> list[Violation]:
    """every change in place in one module.

    :param tree: the module
    :ptype tree: ast.Module
    :param module_path: its path
    :ptype module_path: Path
    :param rules: the walk
    :ptype rules: _Rules
    :return: the violations
    :rtype: list[Violation]
    """
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    own_by_class: list[tuple[ast.ClassDef, frozenset[str]]] = []
    holders: dict[str, set[str]] = {}
    for cls in classes:
        fields = _class_fields(cls, rules) if rules.widened else frozenset()
        attrs = _mutable_attrs(cls, rules) | fields
        own_by_class.append((cls, attrs))
        for attr in attrs:
            if attr in fields or attr.startswith("_"):
                holders.setdefault(attr, set()).add(cls.name)
    shared: tuple[tuple[str, str], ...] = (
        tuple((attr, "/".join(sorted(names))) for attr, names in sorted(holders.items())) if rules.widened else ()
    )
    violations: list[Violation] = []
    for cls, attrs in own_by_class:
        if not attrs and not shared:
            continue
        for method in _methods(cls):
            own = frozenset() if method.name in rules.builders else attrs
            if not own and not shared:
                continue
            violations.extend(_scan_function(method, _Scope(cls.name, own, shared), module_path, rules))
    if shared:
        for function in _module_functions(tree.body):
            violations.extend(_scan_function(function, _Scope(None, frozenset(), shared), module_path, rules))
    if not rules.widened:
        return violations
    # a function nested in a class nested in a function is walked from both; each change counts once
    seen: set[tuple[int, str]] = set()
    unique: list[Violation] = []
    for violation in violations:
        key = (violation.line, violation.symbol)
        if key not in seen:
            seen.add(key)
            unique.append(violation)
    return unique


def find_mutated_state_violations(
    roots: tuple[Path, ...], repo_root: Path, *, widened: bool = False
) -> list[Violation]:
    """every change in place of an object's dict or set state outside ``__init__`` under ``roots``.

    :param roots: the trees (directories or single modules) where state is immutable
    :ptype roots: tuple[Path, ...]
    :param repo_root: the repo's root (kept for parity with the sibling walkers)
    :ptype repo_root: Path
    :param widened: also count public attributes, lists and deques, class-body fields, and changes made through
        another object in the same module, except the state of call-scoped classes (Pace's rulings of 2026-11-24)
    :ptype widened: bool
    :return: the violations, root-major, then file, then line
    :rtype: list[Violation]
    """
    _ = repo_root
    rules = _WIDE if widened else _NARROW
    parsed: list[tuple[Path, ast.Module]] = []
    for root in roots:
        modules = [root] if root.is_file() else list(iter_python_files(root))
        for module_path in modules:
            tree = parse_python_file(module_path)
            if tree is not None:
                parsed.append((module_path, tree))
    violations: list[Violation] = []
    for module_path, tree in parsed:
        violations.extend(_scan_module(tree, module_path, rules))
    if not widened:
        return violations
    scoped = call_scoped_classes(tree for _, tree in parsed)
    return [violation for violation in violations if not _call_scoped(violation.symbol, scoped)]


def _call_scoped(symbol: str, scoped: frozenset[str]) -> bool:
    """whether every class holding ``symbol``'s state (``A.x``, or ``A/B.x`` by a shared field name) is call-scoped.

    :param symbol: the violation's ``Class.attr``
    :ptype symbol: str
    :param scoped: the call-scoped classes
    :ptype scoped: frozenset[str]
    :return: whether they all are
    :rtype: bool
    """
    holders = symbol.rsplit(".", 1)[0].split("/")
    return all(holder in scoped for holder in holders)


def find_stale_immutable_debts(
    roots: tuple[Path, ...],
    debts: tuple[DictStateAllowlistEntry, ...],
    repo_root: Path,
    *,
    widened: bool = False,
) -> list[Violation]:
    """each tracked change in place that no longer happens: the redesign landed, so the entry goes.

    :param roots: the trees where state is immutable
    :ptype roots: tuple[Path, ...]
    :param debts: the tracked changes in place
    :ptype debts: tuple[DictStateAllowlistEntry, ...]
    :param repo_root: the repo's root, which the entries' files are relative to
    :ptype repo_root: Path
    :param widened: the walk the debts are tracked under (see :func:`find_mutated_state_violations`)
    :ptype widened: bool
    :return: one violation a stale entry
    :rtype: list[Violation]
    """
    if not debts:
        return []
    real = {
        (relative_posix_path(violation.file, repo_root), violation.symbol)
        for violation in find_mutated_state_violations(roots, repo_root, widened=widened)
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
