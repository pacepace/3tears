"""Tenancy is one opaque ``scope_id`` — required everywhere, named on every port call, never read.

Three properties, each over a population derived from the code rather than listed, so a model, a
store method or a branch added later is checked without anyone remembering to add a row:

1. **Every stored model requires a non-blank ``scope_id``.** The stored models are every package
   model declaring a ``doc_type`` (:mod:`packages.evals.tests.stored_models`). A default would let
   a document be written without its writer choosing a scope, and a blank one is a partition no
   scoped read can return.
2. **Every storage port method names the scope it acts in, or takes it from the document it
   writes.** The ports are every ``Protocol`` in the package whose name ends in ``Store`` —
   :class:`~threetears.evals.schema.store_port.DocumentStore` and the narrow ports cut from it alike.
   A method with neither is a scope-free read or write, the shape the multi-tenant norm forbids.
3. **No code in ``src/`` interprets a scope's value.** No comparison of a scope against a literal
   or a named constant (an ALL-CAPS name), no membership test in either direction, no truthiness
   test of one wherever it appears (``scope_id or "default"`` is a fallback scope, whether or not
   an ``if`` holds it), no method called on one, no slicing, ``len()`` or numeric/boolean
   conversion of one, no ``match`` over one, no read of one through ``getattr``/``.get`` with a
   fallback, no lookup keyed by one in a literal or constant table, and no ``scope_id`` parameter
   with a default. A value derived from a scope (``str(scope_id)``, ``scope_id[:7]``) compared to
   anything is a read of it too. Comparing two scopes for equality is allowed — that is identity,
   which an opaque value still has.

   **The one named exception: a scope no other writer can share.** A function marked
   ``# scope-default: <rationale>`` on the line above its ``def`` (or its first decorator) may give its
   scope parameter a default and test whether one was named (``scope_id is None``) — nothing else:
   every other read in it is flagged as anywhere. The rule exists so a writer to a store that outlives
   the call chooses the scope its documents land in; the quick path's own in-memory store lives and
   dies with one call, so the scope it falls back to partitions nothing anyone else reads. The marker
   sits on the function, as ``# parity-exempt:`` does on a fake, so it moves with the code; the
   rationale is one line of at least 30 characters, and the marked functions are pinned by name below
   so a new one is a reviewed change.

   **What the detector cannot see**, so its green is not read as wider than it is: a scope is
   recognised by its name — ``scope_id`` or ``_scope_id``, as a variable, an attribute, a
   ``["scope_id"]`` item, a ``.get("scope_id")`` or a ``getattr(x, "scope_id")`` — and by a local
   alias assigned from one (``scope = run.scope_id``), within the function that assigns it. A scope
   handed to a function under another parameter name (``def f(tenant): ...``), stored on an
   attribute of another name, or carried inside a container, is out of reach; so is a lookup keyed
   by a scope in a lowercase table, which reads the same as an identity-keyed partition.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import typing
import pkgutil
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from annotated_types import MinLen

import threetears.evals
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.schema.store_port import DocumentStore

from packages.evals.tests.stored_models import stored_models

_SRC = Path(threetears.evals.__file__).parent


# =============================================================================
# 1. Every stored model requires a non-blank scope
# =============================================================================


def _scope_defects(model: type[EvalBaseModel]) -> list[str]:
    field = model.model_fields.get("scope_id")
    if field is None:
        return ["declares no scope_id"]
    defects = []
    if not field.is_required():
        defects.append(f"scope_id has a default ({field.get_default()!r}), so a writer need not choose one")
    if not any(isinstance(meta, MinLen) and meta.min_length >= 1 for meta in field.metadata):
        defects.append("scope_id has no min_length, so a blank scope is representable")
    return defects


def test_the_stored_models_are_found() -> None:
    """The positive control: the derivation reaches both tiers, so property 1 is not vacuous."""
    names = {model.__name__ for model in stored_models()}

    assert {"EvalTemplate", "EvalCampaign", "EvalRun", "EvalResult"} <= names


@pytest.mark.parametrize("model", stored_models(), ids=lambda model: model.__name__)
def test_every_stored_model_requires_a_non_blank_scope(model: type[EvalBaseModel]) -> None:
    assert _scope_defects(model) == []


# =============================================================================
# 2. Every storage port method names its scope, or takes it from what it writes
# =============================================================================


def _store_ports() -> list[type]:
    """Every ``*Store`` protocol defined in the package, after importing all of it."""
    for module in pkgutil.walk_packages(threetears.evals.__path__, prefix="threetears.evals."):
        importlib.import_module(module.name)
    ports = {
        obj
        for name, module in sys.modules.items()
        if name.startswith("threetears.evals")
        for obj in vars(module).values()
        if inspect.isclass(obj)
        and typing.is_protocol(obj)
        and obj.__module__ == name
        and obj.__name__.endswith("Store")
    }
    return sorted(ports, key=lambda port: port.__qualname__)


#: The annotation of a raw document: the one write that is not of a model still carries ``scope_id``.
_DOCUMENT = "dict[str, Any]"


def _derives_scope_from_what_it_writes(method: Any) -> bool:
    """A write whose first argument is a stored model, or a raw document, carries its scope with it.

    Read off the annotation as written (the package defers annotations), against the stored models'
    names — the population property 1 holds to requiring a scope.
    """
    parameters = list(inspect.signature(method).parameters.values())[1:]
    if not parameters:
        return False
    written = parameters[0].annotation
    return written == _DOCUMENT or written in {model.__name__ for model in stored_models()}


def _port_defects(port: type) -> list[str]:
    defects = []
    for name, member in vars(port).items():
        if name.startswith("_") or not inspect.isfunction(member):
            continue
        if "scope_id" in inspect.signature(member).parameters:
            continue
        if _derives_scope_from_what_it_writes(member):
            continue
        defects.append(f"{port.__qualname__}.{name} neither takes scope_id nor writes a document carrying one")
    return defects


def test_the_ports_are_found() -> None:
    """The positive control: the document store and the narrow ports are both in the population."""
    names = {port.__qualname__ for port in _store_ports()}

    assert DocumentStore in _store_ports()
    assert {"CampaignStore", "CurationStore", "LensStore"} <= names


def test_the_document_store_derives_its_scope_only_on_upsert() -> None:
    """Pinned by name as well: the one port every host implements has exactly one scope-free signature."""
    scope_free = [
        name
        for name, member in vars(DocumentStore).items()
        if not name.startswith("_")
        and inspect.isfunction(member)
        and "scope_id" not in inspect.signature(member).parameters
    ]

    assert scope_free == ["upsert"]


@pytest.mark.parametrize("port", _store_ports(), ids=lambda port: port.__qualname__)
def test_every_port_method_names_its_scope_or_writes_one(port: type) -> None:
    assert _port_defects(port) == []


# =============================================================================
# 3. No code reads a scope's value
# =============================================================================


#: The field name a scope travels under; ``_scope_id`` (a private attribute holding one) is the same.
_SCOPE = "scope_id"

#: Builtins whose result is a reading of their argument's value — its length, its truth, a number.
_VALUE_READERS = frozenset({"len", "bool", "int", "float", "ord", "list", "tuple", "set", "sorted", "reversed", "iter"})

#: Builtins that render their argument; harmless as carriage, a read once the rendering is compared.
_RENDERERS = frozenset({"str", "repr", "format"})

_FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda


def _is_scope_name(name: str) -> bool:
    return name.lstrip("_") == _SCOPE


def _is_constant_str(node: ast.AST, value: str) -> bool:
    return isinstance(node, ast.Constant) and node.value == value


def _is_scope(node: ast.AST, aliases: frozenset[str] = frozenset()) -> bool:
    """A scope: ``scope_id``, ``x.scope_id``, ``x["scope_id"]``, ``x.get("scope_id")``, ``getattr(x, "scope_id")``.

    ``aliases`` are local names assigned from one in the function being read.
    """
    if isinstance(node, ast.Name):
        return _is_scope_name(node.id) or node.id in aliases
    if isinstance(node, ast.Attribute):
        return _is_scope_name(node.attr)
    if isinstance(node, ast.Subscript):
        return _is_constant_str(node.slice, _SCOPE)
    if isinstance(node, ast.Call) and node.args and _is_constant_str(node.args[0], _SCOPE):
        return isinstance(node.func, ast.Attribute) and node.func.attr == "get"
    if isinstance(node, ast.Call) and len(node.args) >= 2 and _is_constant_str(node.args[1], _SCOPE):
        return isinstance(node.func, ast.Name) and node.func.id == "getattr"
    return False


def _derived_from_scope(node: ast.AST, aliases: frozenset[str]) -> bool:
    """A value computed FROM a scope's value: a slice or index of one, or a builtin reading or rendering it."""
    if isinstance(node, ast.Subscript):
        return _is_scope(node.value, aliases)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.args:
        return node.func.id in _VALUE_READERS | _RENDERERS and _is_scope(node.args[0], aliases)
    return False


def _is_literal(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant | ast.JoinedStr):
        return True
    if isinstance(node, ast.List | ast.Tuple | ast.Set):
        return all(_is_literal(element) for element in node.elts)
    if isinstance(node, ast.Dict):
        return all(key is not None and _is_literal(key) for key in node.keys)
    return False


def _is_named_constant(node: ast.AST) -> bool:
    """An ALL-CAPS name or attribute — ``DEFAULT_SCOPE``, ``config.ALLOWED_SCOPES`` — the spelling of a constant."""
    name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else ""
    return any(ch.isalpha() for ch in name) and name == name.upper()


def _is_table(node: ast.AST) -> bool:
    """A table whose entries are fixed values: a literal, or a named constant."""
    return _is_literal(node) or _is_named_constant(node)


def _reads_with_fallback(node: ast.Call) -> bool:
    """``x.get("scope_id", default)`` or ``getattr(x, "scope_id", default)`` — a scope that may be a default."""
    if isinstance(node.func, ast.Attribute) and node.func.attr == "get":
        return len(node.args) >= 2 and _is_constant_str(node.args[0], _SCOPE)
    if isinstance(node.func, ast.Name) and node.func.id == "getattr":
        return len(node.args) >= 3 and _is_constant_str(node.args[1], _SCOPE)
    return False


#: The marker naming a function's scope default as the deliberate exception, with its reason.
_DEFAULT_MARKER = "# scope-default:"

#: The shortest rationale the marker accepts — the bar ``# parity-exempt:`` sets.
_MIN_RATIONALE = 30


def _marked_default(fn: ast.AST, lines: list[str]) -> bool:
    """Whether the line above ``fn`` (or above its first decorator) carries the marker with a real rationale."""
    if not isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
        return False
    first = min([fn.lineno, *(decorator.lineno for decorator in fn.decorator_list)])
    above = lines[first - 2].strip() if first >= 2 else ""
    return above.startswith(_DEFAULT_MARKER) and len(above.removeprefix(_DEFAULT_MARKER).strip()) >= _MIN_RATIONALE


def _is_none_check(node: ast.Compare, aliases: frozenset[str]) -> bool:
    """``scope_id is None`` / ``is not None``: whether a scope was named, and nothing about its value."""
    operands = [node.left, *node.comparators]
    return (
        all(isinstance(op, ast.Is | ast.IsNot) for op in node.ops)
        and any(_is_scope(o, aliases) for o in operands)
        and all(_is_scope(o, aliases) or (isinstance(o, ast.Constant) and o.value is None) for o in operands)
    )


def _defaulted_scope_parameters(fn: _FunctionNode) -> list[str]:
    """The scope parameters of ``fn`` that carry a default, so a caller need not name the scope."""
    args = fn.args
    positional = args.posonlyargs + args.args
    defaulted = positional[len(positional) - len(args.defaults) :]
    defaulted += [arg for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True) if default is not None]
    return [arg.arg for arg in defaulted if _is_scope_name(arg.arg)]


def _local_aliases(fn: ast.AST, inherited: frozenset[str]) -> frozenset[str]:
    """Names ``fn``'s own body assigns from a scope — ``scope = run.scope_id`` — closed over chains of them.

    Flow-insensitive within the function, so a name assigned from a scope anywhere in it is one
    everywhere in it: the detector errs towards flagging. Nested functions are read on their own.
    """
    params: set[str] = set()
    if isinstance(fn, _FunctionNode):
        args = fn.args
        params = {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs}
        params |= {a.arg for a in (args.vararg, args.kwarg) if a is not None}
    aliases = frozenset(inherited - params)

    def own_nodes(node: ast.AST) -> Iterator[ast.AST]:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _FunctionNode | ast.ClassDef):
                continue
            yield child
            yield from own_nodes(child)

    while True:
        found = set(aliases)
        for node in own_nodes(fn):
            if isinstance(node, ast.Assign | ast.AnnAssign | ast.NamedExpr) and node.value is not None:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if _is_scope(node.value, aliases):
                    found |= {target.id for target in targets if isinstance(target, ast.Name)}
        if found == aliases:
            return aliases
        aliases = frozenset(found)


def _reads_at(
    node: ast.AST, aliases: frozenset[str], lines: list[str], *, in_marked: bool = False
) -> Iterator[tuple[int, str]]:
    """The violations ``node`` itself commits, given the scope aliases in force where it sits.

    ``in_marked`` is whether ``node`` sits directly in a function marked ``# scope-default:``, where a
    test of whether a scope was named is the exception's own business.
    """

    def scope(n: ast.AST) -> bool:
        return _is_scope(n, aliases)

    def scope_or_derived(n: ast.AST) -> bool:
        return scope(n) or _derived_from_scope(n, aliases)

    if isinstance(node, ast.Compare) and in_marked and _is_none_check(node, aliases):
        pass
    elif isinstance(node, ast.Compare):
        operands = [node.left, *node.comparators]
        if any(isinstance(op, ast.In | ast.NotIn) for op in node.ops) and any(scope_or_derived(o) for o in operands):
            yield node.lineno, "tests membership with a scope"
        elif any(scope_or_derived(o) for o in operands) and any(_is_table(o) for o in operands):
            yield node.lineno, "compares a scope against a literal or a named constant"
        elif any(_derived_from_scope(o, aliases) for o in operands):
            yield node.lineno, "compares a value derived from a scope"
    elif isinstance(node, ast.BoolOp) and any(scope(value) for value in node.values):
        yield node.lineno, "uses a scope's truthiness in a boolean operation"
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not) and scope(node.operand):
        yield node.lineno, "negates a scope's truthiness"
    elif isinstance(node, ast.If | ast.IfExp | ast.While | ast.Assert) and scope(node.test):
        yield node.lineno, "branches on a scope's truthiness"
    elif isinstance(node, ast.comprehension) and any(scope(test) for test in node.ifs):
        yield node.target.lineno, "filters on a scope's truthiness"
    elif isinstance(node, ast.Match) and scope_or_derived(node.subject):
        yield node.lineno, "matches over a scope"
    elif isinstance(node, ast.Subscript) and scope(node.value):
        yield node.lineno, "slices or indexes a scope"
    elif isinstance(node, ast.Subscript) and scope(node.slice) and _is_table(node.value):
        yield node.lineno, "looks a scope up in a fixed table"
    elif isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and scope_or_derived(func.value):
            yield node.lineno, f"calls .{func.attr}() on a scope"
        elif isinstance(func, ast.Attribute) and node.args and scope(node.args[0]) and _is_table(func.value):
            yield node.lineno, f"looks a scope up in a fixed table with .{func.attr}()"
        elif isinstance(func, ast.Name) and func.id in _VALUE_READERS and node.args and scope(node.args[0]):
            yield node.lineno, f"reads a scope's value with {func.id}()"
        elif _reads_with_fallback(node):
            yield node.lineno, "reads a scope with a fallback default"
    if isinstance(node, _FunctionNode) and not _marked_default(node, lines):
        for name in _defaulted_scope_parameters(node):
            yield node.lineno, f"defaults the scope parameter {name!r}"


def scope_value_reads(source: str) -> Iterator[tuple[int, str]]:
    """Every place ``source`` interprets a scope's value, as ``(line, what)``.

    Args:
        source: Python source.

    Yields:
        One entry per violation.
    """

    lines = source.splitlines()

    def visit(node: ast.AST, aliases: frozenset[str], in_marked: bool) -> Iterator[tuple[int, str]]:
        for child in ast.iter_child_nodes(node):
            yield from _reads_at(child, aliases, lines, in_marked=in_marked)
            if isinstance(child, _FunctionNode):
                yield from visit(child, _local_aliases(child, aliases), _marked_default(child, lines))
            elif isinstance(child, ast.ClassDef):
                yield from visit(child, aliases, False)
            else:
                yield from visit(child, aliases, in_marked)

    tree = ast.parse(source)
    yield from visit(tree, _local_aliases(tree, frozenset()), False)


def marked_scope_defaults(source: str) -> list[str]:
    """The names of the functions in ``source`` that carry the ``# scope-default:`` exception."""
    lines = source.splitlines()
    return [node.name for node in ast.walk(ast.parse(source)) if _marked_default(node, lines) and hasattr(node, "name")]


class TestTheDetector:
    """The detector, against shapes it must and must not flag — so a green scan means something."""

    @pytest.mark.parametrize(
        "source",
        [
            'if scope_id == "prod":\n    pass',
            'x = run.scope_id != "a"',
            'ok = doc["scope_id"] in ("a", "b")',
            "if not scope_id:\n    pass",
            "x = 1 if campaign.scope_id else 2",
            'x = scope_id.startswith("tenant-")',
            "match scope_id:\n    case 'a':\n        pass",
            "y = [r for r in rows if r.scope_id]",
            'x = doc.get("scope_id") == "a"',
            "if scope_id is None:\n    pass",
            # A fallback scope, with no ``if`` anywhere near it, in each spelling a scope arrives in.
            'x = scope_id or "default"',
            'x = run.scope_id or "default"',
            'store.get(run_id, scope_id or "default")',
            "x = ready and scope_id",
            "x = not scope_id",
            'x = doc.get("scope_id", "default")',
            'x = getattr(run, "scope_id", "default")',
            # Compared against a named constant, or a membership test either way round.
            "ok = scope_id == DEFAULT_SCOPE",
            "ok = run.scope_id != config.HOME_SCOPE",
            "ok = scope_id in ALLOWED_SCOPES",
            "ok = scope_id in allowed",
            'ok = "prod" in scope_id',
            # A value derived from a scope, compared to anything — or a read in itself.
            'ok = scope_id[:7] == "tenant-"',
            "ok = len(scope_id) > 3",
            'ok = str(scope_id) == "x"',
            "x = scope_id[0]",
            "n = len(run.scope_id)",
            "flag = bool(scope_id)",
            'ok = getattr(run, "scope_id") == "x"',
            'ok = str(scope_id).startswith("tenant-")',
            # A scope looked up in a table of fixed values.
            'v = {"prod": 1}[scope_id]',
            "v = DEFAULTS.get(scope_id)",
            "v = LIMITS[run.scope_id]",
            # A scope parameter a caller need not name.
            'def f(scope_id: str = "default"):\n    pass',
            "def f(*, scope_id: str | None = None):\n    pass",
            "async def f(run_id, scope_id=None):\n    pass",
            "f = lambda scope_id='x': scope_id",
            # A local alias carries the scope, so the parameter's name alone cannot be what the check reads.
            'scope = scope_id\nif scope == "prod":\n    pass',
            'def f(run):\n    tenant = run.scope_id\n    other = tenant\n    return other or "default"',
            # The exception is narrow: a short rationale exempts nothing, and a marked function's other reads,
            # a nested function in it, and the line below the marker's own function, are all still read.
            "# scope-default: too short\ndef f(*, scope_id=None):\n    pass",
            "# scope-default: the call builds and drops its own in-memory store\ndef f(*, scope_id=None):\n"
            '    return scope_id == "prod"',
            "# scope-default: the call builds and drops its own in-memory store\ndef f(*, scope_id=None):\n"
            '    return scope_id or "quick"',
            "# scope-default: the call builds and drops its own in-memory store\ndef f(*, scope_id=None):\n"
            "    def g(scope_id=None):\n        return scope_id is None\n    return g",
            "# scope-default: the call builds and drops its own in-memory store\ndef f(*, scope_id=None):\n"
            "    pass\ndef h(scope_id=None):\n    pass",
            "# scope-default: the call builds and drops its own in-memory store\n\ndef f(*, scope_id=None):\n    pass",
        ],
    )
    def test_it_flags_a_read_of_the_value(self, source: str) -> None:
        assert list(scope_value_reads(source)), source

    @pytest.mark.parametrize(
        "source",
        [
            "same = insight.scope_id == scope_id",
            "stores = load(run_id, scope_id)",
            "x = {d.id for d in docs if d.scope_id != self.scope_id}",
            'log.info("scope=%s", scope_id)',
            "if storage.delete(doc_id, scope_id):\n    pass",
            # Carriage in every new spelling the detector now reads, and an alias used only as carriage.
            'msg = f"no runs in scope {scope_id!r}"',
            "key = (str(scope_id), run_id)",
            "rows = by_scope[scope_id]",
            "rows = self._partitions.get(run.scope_id, [])",
            "def f(run_id: str, scope_id: str, *, limit: int = 10):\n    pass",
            "def f(run):\n    scope = run.scope_id\n    return load(run.id, scope)",
            # A name spelled like an alias in one function is not one in its sibling.
            "def a(run):\n    s = run.scope_id\n    return s\ndef b(s):\n    return s or 'x'",
            'same = getattr(a, "scope_id") == getattr(b, "scope_id")',
            # The named exception: a default, and a test of whether a scope was named, in a marked function.
            "# scope-default: the call builds and drops its own in-memory store\nasync def f(*, scope_id=None):\n"
            "    if scope_id is not None:\n        return scope_id\n    return DEFAULT_QUICK_SCOPE",
            "# scope-default: the call builds and drops its own in-memory store\n@decorated\ndef f(scope_id=None):\n"
            "    return load(scope_id)",
        ],
    )
    def test_it_passes_identity_and_carriage(self, source: str) -> None:
        assert list(scope_value_reads(source)) == [], source


def _src_files() -> list[Path]:
    return sorted(_SRC.rglob("*.py"))


def test_the_scan_reaches_the_storage_layer() -> None:
    """The positive control: the files that handle scopes most are in the scanned population."""
    scanned = {path.relative_to(_SRC).as_posix() for path in _src_files()}

    assert {"kernel/storage.py", "schema/store_port.py", "analysis/campaigns.py"} <= scanned


def test_the_scope_defaults_are_only_the_quick_path_s_own_store() -> None:
    """The exception is pinned: a new ``# scope-default:`` marker fails here until it is reviewed and named.

    Each is the quick path, whose store is the call's own in-memory one unless a ``host=`` or ``store=`` is
    passed — and :func:`~threetears.evals.quick.one_call.quick_scope` refuses a missing scope with either.
    """
    marked = {
        f"{path.relative_to(_SRC).as_posix()}:{name}"
        for path in _src_files()
        for name in marked_scope_defaults(path.read_text(encoding="utf-8"))
    }

    assert marked == {
        "quick/compare.py:compare",
        "quick/one_call.py:quick_scope",
        "quick/one_call.py:run_eval",
        "quick/one_call.py:run_arms",
    }


def test_no_code_in_src_reads_a_scope_s_value() -> None:
    reads = [
        f"{path.relative_to(_SRC)}:{line}: {what}"
        for path in _src_files()
        for line, what in scope_value_reads(path.read_text(encoding="utf-8"))
    ]

    assert reads == []
