"""Structural gate: the engine keeps no host state at module level.

Every entrypoint takes its host as an argument, so two hosts in one process are two values. That
holds only while nothing under ``src/`` keeps state a second host would read: a module-level profile,
a registry installed at import, a dict filled at runtime. This walks every module for the shapes
such state takes, and refuses each unless it is on :data:`ALLOWED` with a rationale.

1. **A ``global`` statement** — a function rebinding a module name is the mechanism of every
   installed-host design (its setter rebound a module-level ``_ACTIVE``).
2. **A module-level binding of a host-typed value** — annotated or constructed as one of
   :data:`HOST_TYPES`, ``None`` included: ``_ACTIVE: HostProfile | None = None`` is the shape, and its
   value is beside the point.
3. **A module-level object mutated after import** — a method that changes a container
   (``append``, ``update``, ``setdefault``, ``__setitem__`` …), an item or attribute assignment, a
   ``del``, or a ``setattr``/``delattr``, on anything reached from a module-level name — a variable,
   or a module-level ``def`` or ``class``: ``_Memo.seen[k] = v`` mutates a class attribute,
   ``get.host = h`` a function attribute — from anywhere in its module, unless that name is bound
   locally where the mutation happens. A constant table is never mutated, so it is not state and is
   not refused.
4. **A mutable default argument** — a default is evaluated once, when the ``def`` runs, so
   ``def f(x, _memo: dict = {})`` is a container every call in the process shares: a cache by
   another name, keyed by whatever the function writes into it. Refused for a literal container or
   comprehension, a call to a builtin container constructor, and a call constructing a host type
   or a state constructor.

Plus one for memoisation, which is module state by another name: a cached function may not take a
parameter that could be a host — one annotated with a host type, one annotated ``Any`` or
``object``, or one with no annotation at all, which could be anything — or its cache would answer
one host with another's result. A method's ``self``/``cls`` is exempt: a cache keyed by the
instance answers each instance from its own call.

**What this cannot see**, stated so the green is not read as wider than it is:

- state reached through an IMPORTED name (``module.thing.items.append``, ``os.environ[k] = v``) and
  mutation from another module — the walk resolves names within one module only;
- an alias: ``memo = _Memo.seen`` then ``memo[k] = v`` mutates module state through a local name,
  and the local name is all the walk sees; the same for ``cls.seen[k] = v`` inside a classmethod
  and ``self.seen[k] = v`` on a mutable class attribute, which read as instance state;
- a default constructed from any other class (``options: Options = Options()``): whether it is
  shared mutable state is a property of that class, which a syntax walk cannot read;
- state a third-party library keeps.

The first two have no instance in the tree that anyone has found; the walk's own fabrication test
shows each shape it does claim to catch going red.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"

#: Types whose instance at module level would be a host's state, or a default standing in for one.
HOST_TYPES: frozenset[str] = frozenset(
    {
        "EvalHost",
        "LaunchHost",
        "LaunchSettings",
        "HostProfile",
        "CompletionClients",
        "TraceSink",
        "EvalStorage",
        "DocumentStore",
        "EvalJobManager",
        "SweepableRegistry",
        "MeasureRegistry",
        "BarRegistry",
        "WorldRegistry",
        "StyleProfile",
    }
)

#: Constructions that hold process state by their nature, refused at module level like a host type.
STATE_CONSTRUCTORS: frozenset[str] = frozenset({"Lock", "RLock", "Semaphore", "Event", "Condition", "ContextVar"})

#: Methods that change a container (or an object's attributes) in place.
MUTATORS: frozenset[str] = frozenset(
    {
        "append",
        "add",
        "update",
        "setdefault",
        "pop",
        "popitem",
        "clear",
        "extend",
        "insert",
        "remove",
        "discard",
        "__setitem__",
        "__delitem__",
        "__setattr__",
        "__delattr__",
        "__iadd__",
        "__ior__",
    }
)

#: Builtins that set or delete an attribute on their first argument.
ATTRIBUTE_SETTERS: frozenset[str] = frozenset({"setattr", "delattr"})

#: Constructors whose result is a mutable container, so a default built by one is shared state.
MUTABLE_CONSTRUCTORS: frozenset[str] = frozenset(
    {"dict", "list", "set", "bytearray", "defaultdict", "OrderedDict", "Counter", "deque", "ChainMap"}
)

#: Annotations that say nothing about what a parameter holds, so a host could arrive through one.
UNTYPED: frozenset[str] = frozenset({"Any", "object"})

#: The decorators that memoise a function at module level.
CACHES: frozenset[str] = frozenset({"cache", "lru_cache", "cached_property"})

#: ``(module, name)`` -> why this module state holds nothing a host could read. The only entries.
ALLOWED: dict[tuple[str, str], str] = {
    ("threetears/evals/vega/render.py", "_fonts_registered"): (
        "whether the renderer's bundled fonts were registered with the chart library — a process-wide "
        "font cache shared by every host because the library's font table is process-wide"
    ),
    ("threetears/evals/vega/render.py", "_warned_unfonted"): (
        "whether the one-time warning that a render ran with no bundled fonts has been logged — a log "
        "de-duplication flag about the process, not about any host"
    ),
    ("threetears/evals/vega/render.py", "_font_lock"): (
        "serialises the one-time font registration above; guards process-wide library state"
    ),
    ("threetears/evals/vega/render.py", "_registered_font_dirs"): (
        "the font directories already handed to the chart library, so none is registered twice — "
        "part of the same process-wide font cache"
    ),
    ("threetears/evals/contracts/campaign_writes.py", "_CAMPAIGN_WRITES"): (
        "serialises every read-modify-write of a campaign in the process. Shared across hosts, which "
        "over-serialises and never leaks: it holds no value, and a per-store lock would be weaker, "
        "since a host may build two storages over one document store"
    ),
    ("threetears/evals/contracts/host/sweepables.py", "SHARED_CORE"): (
        "the engine's own core declarations every host extends — engine vocabulary, built once from "
        "constants and never mutated (extend returns a new registry)"
    ),
}


class Finding(NamedTuple):
    """One piece of module state the walk found."""

    module: str
    line: int
    name: str
    shape: str

    def render(self) -> str:
        """The finding on one line, with the line to look at."""
        return f"{self.module}:{self.line} {self.name} ({self.shape})"


def _names_in(node: ast.AST | None) -> set[str]:
    """Every bare and attribute name an annotation or constructor expression mentions."""
    if node is None:
        return set()
    found: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            found.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            found.add(sub.attr)
        elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            # A string annotation, written for a forward reference.
            found |= {part.strip("[]|, ") for part in sub.value.replace("|", " ").split()}
    return found


def _constructed(value: ast.AST | None) -> str | None:
    """The callee's name when ``value`` is a call, else ``None``."""
    if isinstance(value, ast.Call):
        func = value.func
        return func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
    return None


def _module_bindings(tree: ast.Module) -> dict[str, tuple[int, ast.AST | None, ast.AST | None]]:
    """Module-level names -> (line, annotation, value): assignments, and every ``def`` and ``class``.

    A ``def`` or ``class`` binds a module name exactly as an assignment does, and the object it binds
    carries attributes a function can write — a class attribute, a function attribute — so it is
    module state the moment anything mutates it. Its annotation and value are ``None``: neither is a
    host type, so it is refused only when mutated.
    """
    bound: dict[str, tuple[int, ast.AST | None, ast.AST | None]] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bound[target.id] = (node.lineno, None, node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            bound[node.target.id] = (node.lineno, node.annotation, node.value)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            bound[node.name] = (node.lineno, None, None)
    return bound


def _root_name(node: ast.AST) -> str | None:
    """The bare name an attribute/item chain starts from (``a`` in ``a.b[c].d``), else ``None``.

    Stops at a call: ``a.copy().update()`` changes what ``copy`` returned, not ``a``.
    """
    while isinstance(node, ast.Attribute | ast.Subscript):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _locally_bound(fn: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> set[str]:
    """Names a function binds for itself, which shadow a module name of the same spelling."""
    args = fn.args
    names = {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs}
    names |= {a.arg for a in (args.vararg, args.kwarg) if a is not None}
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
    return names


def _mutated_names(tree: ast.Module) -> Iterator[tuple[str, int, str]]:
    """``(name, line, how)`` for every in-place change of anything reached from a bare name, module-wide.

    The name is the root of the changed chain, so ``_Memo.seen[k] = v`` reports ``_Memo``: the
    module-level object whose state changed, however deep in it the change landed.
    """

    def walk(node: ast.AST, shadowed: frozenset[str]) -> Iterator[tuple[str, int, str]]:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
                yield from walk(child, shadowed | _locally_bound(child))
                continue
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                name = _root_name(child.func.value)
                if name and child.func.attr in MUTATORS and name not in shadowed:
                    yield name, child.lineno, f".{child.func.attr}()"
            elif isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.args:
                name = _root_name(child.args[0])
                if child.func.id in ATTRIBUTE_SETTERS and name and name not in shadowed:
                    yield name, child.lineno, f"{child.func.id}()"
            elif isinstance(child, ast.Assign | ast.AugAssign | ast.AnnAssign | ast.Delete):
                targets = child.targets if isinstance(child, ast.Assign | ast.Delete) else [child.target]
                for target in targets:
                    if isinstance(target, ast.Subscript | ast.Attribute):
                        name = _root_name(target.value)
                        if name and name not in shadowed:
                            how = "item" if isinstance(target, ast.Subscript) else "attribute"
                            yield name, child.lineno, f"{how} {'deleted' if isinstance(child, ast.Delete) else 'set'}"
            yield from walk(child, shadowed)

    yield from walk(tree, frozenset())


def _is_mutable_default(default: ast.AST) -> bool:
    """A default whose one evaluated value every call would share and could change."""
    if isinstance(default, ast.Dict | ast.List | ast.Set | ast.DictComp | ast.ListComp | ast.SetComp):
        return True
    return (_constructed(default) or "") in MUTABLE_CONSTRUCTORS | HOST_TYPES | STATE_CONSTRUCTORS


def _mutable_defaults(fn: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> list[str]:
    """The parameters of ``fn`` whose default is shared mutable state."""
    args = fn.args
    positional = args.posonlyargs + args.args
    pairs = list(zip(positional[len(positional) - len(args.defaults) :], args.defaults, strict=True))
    pairs += [(arg, default) for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True) if default]
    return [arg.arg for arg, default in pairs if _is_mutable_default(default)]


def _could_hold_a_host(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """The parameters of ``fn`` a host could arrive through: host-typed, ``Any``/``object``, or unannotated.

    A method's leading ``self``/``cls`` is not one: a cache keyed by the instance answers each
    instance from its own call.
    """
    args = fn.args
    params = args.posonlyargs + args.args + args.kwonlyargs
    params += [a for a in (args.vararg, args.kwarg) if a is not None]
    if params and params[0].arg in {"self", "cls"}:
        params = params[1:]
    return [a.arg for a in params if a.annotation is None or _names_in(a.annotation) & (HOST_TYPES | UNTYPED)]


def module_state(root: Path) -> list[Finding]:
    """Every piece of module state under ``root``'s ``threetears`` tree, allowed or not.

    Args:
        root: The directory holding the ``threetears`` namespace.

    Returns:
        The findings, in module and line order.
    """
    findings: list[Finding] = []
    for path in sorted((root / "threetears").rglob("*.py")):
        module = path.relative_to(root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        bindings = _module_bindings(tree)

        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                findings += [Finding(module, node.lineno, name, "rebound through `global`") for name in node.names]
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
                label = node.name if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) else "<lambda>"
                findings += [
                    Finding(module, node.lineno, label, f"mutable default argument: {param}")
                    for param in _mutable_defaults(node)
                ]
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                cached = {n for d in node.decorator_list for n in _names_in(d)} & CACHES
                hosted = _could_hold_a_host(node)
                if cached and hosted:
                    findings.append(
                        Finding(module, node.lineno, node.name, f"memoised over a possible host: {', '.join(hosted)}")
                    )

        for name, (line, annotation, value) in bindings.items():
            typed = (_names_in(annotation) | {_constructed(value) or ""}) & (HOST_TYPES | STATE_CONSTRUCTORS)
            if typed:
                findings.append(Finding(module, line, name, f"bound to a {', '.join(sorted(typed))}"))

        findings += [
            Finding(module, line, name, f"mutated: {how}")
            for name, line, how in _mutated_names(tree)
            if name in bindings
        ]
    return sorted(set(findings), key=lambda f: (f.module, f.line, f.name))


def test_the_engine_keeps_no_host_state_at_module_level() -> None:
    """The gate: every piece of module state is on the allow-list, which names why it holds no host."""
    unexplained = [f.render() for f in module_state(_SRC) if (f.module, f.name) not in ALLOWED]

    assert not unexplained, (
        "Module-level state in the engine, which a second host in the same process would share: "
        + "; ".join(unexplained)
        + ". Pass it in on the EvalHost (or as an argument) instead. If it genuinely holds nothing a host "
        "could read, add it to ALLOWED with the reason."
    )


def test_every_allowance_is_still_state_the_walk_finds() -> None:
    """An allowance whose state is gone is slack: the next state under that name would pass unread."""
    found = {(f.module, f.name) for f in module_state(_SRC)}

    assert set(ALLOWED) <= found, sorted(set(ALLOWED) - found)


@pytest.mark.parametrize(
    ("source", "name"),
    [
        (
            "from threetears.evals.contracts.host import HostProfile\n"
            "_ACTIVE: HostProfile | None = None\n"
            "def install(p):\n    global _ACTIVE\n    _ACTIVE = p\n",
            "_ACTIVE",
        ),
        ('_INSTALLED: "EvalHost | None" = None\n', "_INSTALLED"),
        ("_HOSTS = {}\ndef register(host):\n    _HOSTS[host.id] = host\n", "_HOSTS"),
        ("_SEEN = []\ndef note(x):\n    _SEEN.append(x)\n", "_SEEN"),
        ("import contextvars\n_CURRENT = contextvars.ContextVar('host')\n", "_CURRENT"),
        (
            "from functools import cache\nfrom threetears.evals.contracts.host import HostProfile\n"
            "@cache\ndef levers(profile: HostProfile):\n    return profile.sweepables\n",
            "levers",
        ),
        # The shapes a module-level ``def`` or ``class`` carries, and the defaults a ``def`` evaluates once.
        ("class _Memo:\n    seen = {}\ndef note(k, v):\n    _Memo.seen[k] = v\n", "_Memo"),
        ("def get():\n    return get.host\ndef put(h):\n    get.host = h\n", "get"),
        ("class _S:\n    pass\ndef put(h):\n    setattr(_S, 'host', h)\n", "_S"),
        ("class _S:\n    host = None\ndef drop():\n    delattr(_S, 'host')\n", "_S"),
        ("class _C(dict):\n    pass\ndef put(k, v):\n    _C.__setitem__(_C, k, v)\n", "_C"),
        ("class _R:\n    hosts = []\ndef keep(h):\n    _R.hosts.append(h)\n", "_R"),
        ("def catalog(name, _memo: dict = {}):\n    return _memo.setdefault(name, name)\n", "catalog"),
        ("def seen(name, *, _log=[]):\n    _log.append(name)\n", "seen"),
        ("from collections import defaultdict\ndef tally(k, _n=defaultdict(int)):\n    _n[k] += 1\n", "tally"),
        ("from functools import cache\n@cache\ndef levers(profile):\n    return profile.sweepables\n", "levers"),
        (
            "from functools import lru_cache\nfrom typing import Any\n"
            "@lru_cache\ndef levers(profile: Any):\n    return profile.sweepables\n",
            "levers",
        ),
    ],
)
def test_each_shape_of_module_state_reddens_the_walk(tmp_path: Path, source: str, name: str) -> None:
    """Fabricated red: each shape the docstring claims to catch, in a tree of its own."""
    (tmp_path / "threetears").mkdir()
    (tmp_path / "threetears" / "planted.py").write_text(source, encoding="utf-8")

    assert name in {f.name for f in module_state(tmp_path)}


def test_a_constant_table_and_a_shadowing_local_are_not_state(tmp_path: Path) -> None:
    """The half that keeps the gate usable: a table nobody mutates, and a local of the same name, pass."""
    (tmp_path / "threetears").mkdir()
    (tmp_path / "threetears" / "fine.py").write_text(
        "_TABLE = {'a': 1}\n__all__ = ['x']\ndef build():\n    _TABLE = {}\n    _TABLE['b'] = 2\n    return _TABLE\n",
        encoding="utf-8",
    )

    assert module_state(tmp_path) == []


def test_instance_state_immutable_defaults_and_a_typed_cache_are_not_state(tmp_path: Path) -> None:
    """The other half for the def/class shapes: what they must NOT refuse, so the gate stays usable.

    An instance writing its own attributes, a class whose attributes are only read, immutable
    defaults, a copy changed in place, and a cache over typed, host-free parameters — and a method
    cached on ``self`` — are all ordinary code, not module state.
    """
    (tmp_path / "threetears").mkdir()
    (tmp_path / "threetears" / "fine.py").write_text(
        "from functools import cache\n"
        "class Box:\n"
        "    def __init__(self, items: list[str]) -> None:\n"
        "        self.items = items\n"
        "        self.items.append('x')\n"
        "        object.__setattr__(self, 'frozen', True)\n"
        "    @cache\n"
        "    def size(self) -> int:\n"
        "        return len(self.items)\n"
        "def make(name: str, tags: tuple[str, ...] = (), limit: int | None = None, *, sep: str = ',') -> Box:\n"
        "    return Box(list(tags))\n"
        "def grow(seen: frozenset[str] = frozenset(), unit: str | None = None) -> None:\n"
        "    _ = Box.__name__\n"
        "def build() -> dict[str, int]:\n"
        "    out = _TABLE.copy()\n"
        "    out.update({'b': 2})\n"
        "    return out\n"
        "_TABLE = {'a': 1}\n"
        "@cache\n"
        "def width(name: str, size: int) -> int:\n"
        "    return len(name) * size\n",
        encoding="utf-8",
    )

    assert module_state(tmp_path) == []
