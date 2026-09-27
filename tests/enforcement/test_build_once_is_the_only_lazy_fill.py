"""enforcement: a module-level object built on first use goes through ``BuildOnce``.

**The class this closes.** A getter that checks a module-level cache and fills it when the
value is missing is a race the moment two threads call it first: each one that looked before
the other stored its value builds another. It has bitten as ``ValueError: Duplicated timeseries
in CollectorRegistry`` out of ``create_chat_model`` (two threads each registering the
``threetears_llm_*`` instruments), as a second process-wide Claude CLI pool running twice the
CLIs its limits allow, as several isolation roots for one credential, and as a second
sync-to-async bridge loop stranding work queued on the first. Seven sites were fixed by
hand-copying the same check-lock-recheck idiom into each, which made the fix exactly as
complete as the sweep that found them, and left nothing to stop an eighth getter being written
without the lock.

So there is one construction, :class:`threetears.observe.build_once.BuildOnce` (which keeps its
values on the instance, so it is not itself a module-level fill), and this guard refuses a
hand-rolled one: any function in a package's ``src`` that STORES into a module-level
name, where the store is conditional on having read that same name -- which is what "fill it
if it is missing" is, locked or not. The spellings it recognises, each pinned by a mutation
case below:

- ``value = _CACHE.get(key)`` / ``if value is None:`` ... ``_CACHE[key] = value``, with or
  without a lock and a recheck around it;
- ``if key not in _CACHE: _CACHE[key] = build()``;
- ``global _VALUE`` / ``if _VALUE is None: _VALUE = build()``;
- ``if (value := _CACHE.get(key)) is None: _CACHE[key] = ...``;
- ``try: return _CACHE[key]`` / ``except KeyError: _CACHE[key] = build()``;
- ``_CACHE.setdefault(key, build())``, which evaluates ``build()`` on every call and so
  builds a second value whenever two callers race, even though only one is kept.

- a guard clause that returns early -- ``if key in _CACHE: return _CACHE[key]`` or
  ``if _LOOP is not None and _LOOP.is_running(): return _LOOP`` -- then builds and stores;
- any of the above filling a cache imported from another module.

What it does not refuse, because it is not a fill: a setter that rebinds a module-level
name unconditionally (``set_default_usage_tracker``); a store of a literal (a probe caching
``True`` / ``False``, a reset storing ``None``), since racing callers store equal values with
nothing to duplicate; and an update through ``X.get(key, default)``, which cannot tell a
missing value from a present one and so is never a check for something to build.

Static parsing only -- no import, no network -- consistent with the rest of
``tests/enforcement``.
"""

from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_GLOBS = ("packages/*/src", "packages/agent/*/src")


@dataclass(frozen=True)
class _Fill:
    """one hand-rolled lazy fill of a module-level name.

    :ivar line: the line of the store
    :ivar function: the enclosing function's name
    :ivar name: the module-level name being filled
    """

    line: int
    function: str
    name: str


def _module_level_names(tree: ast.Module) -> set[str]:
    """names bound at a module's top level, by an assignment or a ``from`` import.

    the import matters: a cache declared in one module and filled from another
    (``from .base import _instrument_cache``) is the same cache, filled by hand.

    :param tree: parsed module
    :ptype tree: ast.Module
    :return: the module-level names
    :rtype: set[str]
    """
    names: set[str] = set()
    for node in tree.body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names)
        for target in targets:
            names.update(sub.id for sub in ast.walk(target) if isinstance(sub, ast.Name))
    return names


def _names_read(node: ast.AST) -> set[str]:
    """every plain name an expression or statement mentions.

    :param node: the node to scan
    :ptype node: ast.AST
    :return: the names
    :rtype: set[str]
    """
    return {sub.id for sub in ast.walk(node) if isinstance(sub, ast.Name)}


def _is_literal(node: ast.expr | None) -> bool:
    """true for a literal constant -- ``None``, ``True``, a number, a string.

    storing a literal builds nothing: two racing callers store equal values with no identity to
    tell apart and no side effect to repeat, so a probe that caches ``True`` / ``False`` and a
    reset that stores ``None`` are not the race this guard exists for.

    :param node: the stored expression, when known
    :ptype node: ast.expr | None
    :return: whether it is a literal
    :rtype: bool
    """
    return isinstance(node, ast.Constant)


def _is_lookup_with_default(node: ast.expr) -> bool:
    """true for ``X.get(key, default)`` with a default that is not ``None``.

    such a lookup cannot tell "missing" from "present", so a local bound from it is a value
    being updated (a last-logged timestamp, a counter), never a check for a value to build.

    :param node: the expression a local is bound from
    :ptype node: ast.expr
    :return: whether it is a lookup with a non-``None`` default
    :rtype: bool
    """
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and len(node.args) == 2
        and not (isinstance(node.args[1], ast.Constant) and node.args[1].value is None)
    )


def _exits(statements: list[ast.stmt]) -> bool:
    """true when a block always leaves its enclosing block: it ends in return, raise, break or continue.

    :param statements: the block
    :ptype statements: list[ast.stmt]
    :return: whether the block exits
    :rtype: bool
    """
    return bool(statements) and isinstance(statements[-1], (ast.Return, ast.Raise, ast.Break, ast.Continue))


class _FillFinder(ast.NodeVisitor):
    """finds, in one function, every store into a module-level name guarded by a read of it.

    "Guarded by a read of it" means the store sits inside an ``if`` or ``while`` whose test
    mentions the name -- directly, or through a local bound from an expression that mentions
    it (``value = _CACHE.get(key)``, then ``if value is None``) -- or after such an ``if`` whose
    body leaves early (``if key in _CACHE: return _CACHE[key]``, then the store), or inside the
    ``except`` handler of a ``try`` whose body mentions it.
    """

    def __init__(self, function: ast.FunctionDef | ast.AsyncFunctionDef, module_names: set[str]) -> None:
        """
        prepares a finder for one function.

        :param function: the function to scan
        :ptype function: ast.FunctionDef | ast.AsyncFunctionDef
        :param module_names: names bound at the module's top level
        :ptype module_names: set[str]
        """
        self._function = function
        declared_global = {name for node in ast.walk(function) if isinstance(node, ast.Global) for name in node.names}
        locally_bound = {
            sub.id
            for node in ast.walk(function)
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.For, ast.With, ast.NamedExpr))
            for sub in ast.walk(node)
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store)
        } | self._parameters(function)
        #: module-level names this function can reach: every global it declares, and every
        #: module name it never rebinds locally (a subscript store needs no ``global``).
        self._reachable = declared_global | (module_names - locally_bound)
        self._declared_global = declared_global
        self._params = self._parameters(function)
        #: local name -> the module-level names the expression it was bound from mentions
        self._derived: dict[str, set[str]] = {}
        self._guards: list[set[str]] = []
        self.fills: list[_Fill] = []

    @staticmethod
    def _parameters(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
        """the names of *function*'s parameters.

        :param function: the function
        :ptype function: ast.FunctionDef | ast.AsyncFunctionDef
        :return: its parameter names
        :rtype: set[str]
        """
        return {arg.arg for arg in ast.walk(function.args) if isinstance(arg, ast.arg)}

    def _builds_nothing(self, value: ast.expr | None) -> bool:
        """true when storing *value* cannot be a lazy build.

        a literal has no identity to duplicate, and a parameter is an object the CALLER already
        built -- ``register_scheme(scheme, resolver)`` storing ``resolver`` is a registration, not
        a build.

        :param value: the stored expression, when known
        :ptype value: ast.expr | None
        :return: whether the store builds nothing
        :rtype: bool
        """
        return _is_literal(value) or (isinstance(value, ast.Name) and value.id in self._params)

    def _guarded_names(self, test: ast.AST) -> set[str]:
        """the module-level names a guard mentions, directly or through a derived local.

        :param test: the guard
        :ptype test: ast.AST
        :return: the module-level names it depends on
        :rtype: set[str]
        """
        mentioned = _names_read(test)
        found = mentioned & self._reachable
        for name in mentioned:
            found |= self._derived.get(name, set())
        return found

    def _record(self, node: ast.AST, name: str) -> None:
        """record a store into *name* when a guard around it read *name*.

        :param node: the storing node
        :ptype node: ast.AST
        :param name: the module-level name stored into
        :ptype name: str
        """
        if any(name in guard for guard in self._guards):
            self.fills.append(_Fill(line=getattr(node, "lineno", 0), function=self._function.name, name=name))

    def _note_derivations(self, targets: list[ast.expr], value: ast.expr) -> None:
        """remember which module-level names a local was bound from.

        :param targets: the assignment's targets
        :ptype targets: list[ast.expr]
        :param value: the assigned expression
        :ptype value: ast.expr
        """
        sources = set() if _is_lookup_with_default(value) else self._guarded_names(value)
        if not sources:
            return
        for target in targets:
            for sub in ast.walk(target):
                if isinstance(sub, ast.Name) and sub.id not in self._reachable:
                    self._derived[sub.id] = self._derived.get(sub.id, set()) | sources

    def _check_store(self, target: ast.expr, value: ast.expr | None) -> None:
        """check one assignment target for a guarded fill.

        :param target: the target
        :ptype target: ast.expr
        :param value: the value stored, when known
        :ptype value: ast.expr | None
        """
        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            if target.value.id in self._reachable and not self._builds_nothing(value):
                self._record(target, target.value.id)
        elif isinstance(target, ast.Name) and target.id in self._declared_global:
            if not self._builds_nothing(value):
                self._record(target, target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._check_store(element, None)

    def visit_Assign(self, node: ast.Assign) -> None:
        """a plain assignment: a possible fill, and a possible derived local.

        :param node: the assignment
        :ptype node: ast.Assign
        """
        for target in node.targets:
            self._check_store(target, node.value)
        self._note_derivations(list(node.targets), node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """an annotated assignment.

        :param node: the assignment
        :ptype node: ast.AnnAssign
        """
        if node.value is not None:
            self._check_store(node.target, node.value)
            self._note_derivations([node.target], node.value)
        self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        """a walrus binds a derived local exactly as an assignment does.

        :param node: the named expression
        :ptype node: ast.NamedExpr
        """
        self._note_derivations([node.target], node.value)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """``_CACHE.setdefault(...)`` is a fill whatever surrounds it.

        :param node: the call
        :ptype node: ast.Call
        """
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "setdefault"
            and isinstance(func.value, ast.Name)
            and func.value.id in self._reachable
        ):
            self.fills.append(_Fill(line=node.lineno, function=self._function.name, name=func.value.id))
        self.generic_visit(node)

    def generic_visit(self, node: ast.AST) -> None:
        """visit children, walking every statement list as a block so an early exit guards the rest.

        :param node: the node whose children to visit
        :ptype node: ast.AST
        """
        for _field, value in ast.iter_fields(node):
            if isinstance(value, list) and value and all(isinstance(item, ast.stmt) for item in value):
                self._visit_block(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self.visit(item)
            elif isinstance(value, ast.AST):
                self.visit(value)

    def _visit_block(self, statements: list[ast.stmt]) -> None:
        """visit a block; after an ``if`` that reads a name and leaves early, the rest is guarded by it.

        :param statements: the block
        :ptype statements: list[ast.stmt]
        """
        pushed = 0
        for statement in statements:
            self.visit(statement)
            if isinstance(statement, ast.If) and (_exits(statement.body) or _exits(statement.orelse)):
                self._guards.append(self._guarded_names(statement.test))
                pushed += 1
        for _ in range(pushed):
            self._guards.pop()

    def _visit_guarded(self, test: ast.expr, node: ast.If | ast.While) -> None:
        """visit a conditional's test first, so a walrus in it derives, then its body under guard.

        :param test: the conditional's test
        :ptype test: ast.expr
        :param node: the conditional
        :ptype node: ast.If | ast.While
        """
        self.visit(test)
        self._guards.append(self._guarded_names(test))
        self._visit_block(node.body)
        self._visit_block(node.orelse)
        self._guards.pop()

    def visit_If(self, node: ast.If) -> None:
        """an ``if`` guards its body.

        :param node: the statement
        :ptype node: ast.If
        """
        self._visit_guarded(node.test, node)

    def visit_While(self, node: ast.While) -> None:
        """a ``while`` guards its body.

        :param node: the statement
        :ptype node: ast.While
        """
        self._visit_guarded(node.test, node)

    def visit_Try(self, node: ast.Try) -> None:
        """a ``try`` that read the name guards its ``except`` handlers.

        :param node: the statement
        :ptype node: ast.Try
        """
        self._visit_block(node.body)
        read: set[str] = set()
        for statement in node.body:
            read |= self._guarded_names(statement)
        self._guards.append(read)
        for handler in node.handlers:
            self.visit(handler)
        self._guards.pop()
        self._visit_block(node.orelse)
        self._visit_block(node.finalbody)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """a nested function is scanned on its own, not as part of this one.

        :param node: the nested function
        :ptype node: ast.FunctionDef
        """
        if node is self._function:
            self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """a nested coroutine function is scanned on its own, not as part of this one.

        :param node: the nested function
        :ptype node: ast.AsyncFunctionDef
        """
        if node is self._function:
            self.generic_visit(node)


def lazy_fills(source: str) -> list[_Fill]:
    """every hand-rolled lazy fill of a module-level name in *source*.

    :param source: python source
    :ptype source: str
    :return: the fills, in source order
    :rtype: list[_Fill]
    """
    tree = ast.parse(source)
    module_names = _module_level_names(tree)
    fills: list[_Fill] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            finder = _FillFinder(node, module_names)
            finder.visit(node)
            fills.extend(finder.fills)
    return sorted(fills, key=lambda fill: fill.line)


#: module-level state that the walker reads as a fill and that builds nothing, keyed
#: ``path::function::name``. each entry says why; a stale one fails
#: :func:`test_every_listed_exception_still_names_a_real_site`.
_NOT_BUILDS: dict[str, str] = {
    "packages/nats/src/threetears/nats/kv.py::_log_timeout_remedy::_last_timeout_remedy_log": (
        "a per-bucket last-logged timestamp, rewritten every time the throttle lets a remedy "
        "through rather than built once; the absence check exists because absence means 'never "
        "logged' (see the comment there), and two racing timeouts logging the remedy twice is "
        "the throttle's accepted imprecision, with no object built and nothing registered"
    ),
}


def _key(relative: str, fill: _Fill) -> str:
    """the :data:`_NOT_BUILDS` key for *fill* in the file at *relative*.

    :param relative: repo-relative posix path
    :ptype relative: str
    :param fill: the finding
    :ptype fill: _Fill
    :return: ``path::function::name``
    :rtype: str
    """
    return f"{relative}::{fill.function}::{fill.name}"


def _source_files() -> list[Path]:
    """every shipped python file across the workspace's source trees.

    :return: python files under each package's ``src``
    :rtype: list[Path]
    """
    found: list[Path] = []
    for glob in _PACKAGE_GLOBS:
        for root in sorted(_REPO_ROOT.glob(glob)):
            found.extend(sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts))
    return found


def test_the_source_trees_were_discovered() -> None:
    """the globs must actually match -- a silent zero would pass the guard below.

    :return: none
    :rtype: None
    :raises AssertionError: if almost nothing was discovered
    """
    assert len(_source_files()) > 100, (
        f"only {len(_source_files())} source files matched {_PACKAGE_GLOBS}; the layout changed "
        "and this guard is now inspecting almost nothing."
    )


def test_no_module_level_object_is_filled_by_hand() -> None:
    """every lazily built module-level object goes through ``BuildOnce``.

    :return: none
    :rtype: None
    :raises AssertionError: if a function fills a module-level name after checking it
    """
    violations: list[str] = []
    for path in _source_files():
        relative = path.relative_to(_REPO_ROOT).as_posix()
        violations.extend(
            f"{relative}:{fill.line}: {fill.function}() fills module-level {fill.name} by hand"
            for fill in lazy_fills(path.read_text(encoding="utf-8"))
            if _key(relative, fill) not in _NOT_BUILDS
        )
    assert not violations, (
        "these functions fill a module-level cache after checking it, by hand:\n  "
        + "\n  ".join(violations)
        + "\n\nTwo threads that both look before either stores each build one. Hold the value "
        "in a `threetears.observe.BuildOnce` and fetch it with `.get(key, build)`, which builds "
        "at most once per key however many threads ask first and stays lock-free once built."
    )


def test_every_listed_exception_still_names_a_real_site() -> None:
    """an entry in :data:`_NOT_BUILDS` that matches nothing is removed, not left to excuse a newcomer.

    :return: none
    :rtype: None
    :raises AssertionError: if an entry names a site the walker no longer reports
    """
    found = {
        _key(path.relative_to(_REPO_ROOT).as_posix(), fill)
        for path in _source_files()
        for fill in lazy_fills(path.read_text(encoding="utf-8"))
    }
    assert found, "the walker reported nothing at all across the source trees, so it is checking nothing"
    stale = sorted(set(_NOT_BUILDS) - found)
    assert not stale, f"these _NOT_BUILDS entries match no site any more; delete them: {stale}"


_MUTATIONS: dict[str, str] = {
    "double-checked lock over .get()": """
        import threading
        _CACHE = {}
        _LOCK = threading.Lock()
        def get(key):
            value = _CACHE.get(key)
            if value is None:
                with _LOCK:
                    value = _CACHE.get(key)
                    if value is None:
                        value = object()
                        _CACHE[key] = value
            return value
    """,
    "unlocked check then fill": """
        _CACHE = {}
        def get(key):
            value = _CACHE.get(key)
            if value is None:
                value = object()
                _CACHE[key] = value
            return value
    """,
    "membership test": """
        _CACHE: dict = {}
        def get(key):
            if key not in _CACHE:
                _CACHE[key] = object()
            return _CACHE[key]
    """,
    "global rebound after an is-None check": """
        _VALUE = None
        def get():
            global _VALUE
            if _VALUE is None:
                _VALUE = object()
            return _VALUE
    """,
    "global rebound after a staleness check": """
        _LOOP = None
        def get():
            global _LOOP
            if _LOOP is None or not _LOOP.is_running():
                _LOOP = object()
            return _LOOP
    """,
    "guard clause that returns early, then the fill under a lock": """
        import threading
        _LOOP = None
        _LOCK = threading.Lock()
        def get():
            global _LOOP
            if _LOOP is not None and _LOOP.is_running():
                return _LOOP
            with _LOCK:
                if _LOOP is not None and _LOOP.is_running():
                    return _LOOP
                loop = object()
                _LOOP = loop
                return _LOOP
    """,
    "membership guard clause that returns early": """
        _CACHE = {}
        def get(key):
            if key in _CACHE:
                return _CACHE[key]
            verdict = object()
            _CACHE[key] = verdict
            return verdict
    """,
    "a cache imported from another module": """
        from pkg.base import _instrument_cache
        def get(key):
            instrument = _instrument_cache.get(key)
            if instrument is None:
                instrument = object()
                _instrument_cache[key] = instrument
            return instrument
    """,
    "walrus in the guard": """
        _CACHE = {}
        def get(key):
            if (value := _CACHE.get(key)) is None:
                value = _CACHE[key] = object()
            return value
    """,
    "try / except KeyError": """
        _CACHE = {}
        def get(key):
            try:
                return _CACHE[key]
            except KeyError:
                _CACHE[key] = object()
                return _CACHE[key]
    """,
    "setdefault": """
        _CACHE = {}
        def get(key):
            return _CACHE.setdefault(key, object())
    """,
    "inside a method": """
        _CACHE = {}
        class Owner:
            def get(self, key):
                if key not in _CACHE:
                    _CACHE[key] = object()
                return _CACHE[key]
    """,
    "in a coroutine function": """
        _CACHE = {}
        async def get(key):
            value = _CACHE.get(key)
            if value is None:
                value = _CACHE[key] = object()
            return value
    """,
}


@pytest.mark.parametrize("source", list(_MUTATIONS.values()), ids=list(_MUTATIONS))
def test_every_spelling_of_a_hand_rolled_fill_is_refused(source: str) -> None:
    """each known spelling of the idiom, reintroduced, is caught.

    :param source: a module holding one hand-rolled fill
    :ptype source: str
    :return: none
    :rtype: None
    :raises AssertionError: if the walker misses it
    """
    assert lazy_fills(textwrap.dedent(source)), "the walker missed this spelling of a hand-rolled lazy fill"


_NOT_FILLS: dict[str, str] = {
    "an unconditional setter": """
        _DEFAULT = None
        def set_default(value):
            global _DEFAULT
            _DEFAULT = value
    """,
    "a reset to None": """
        _VALUE = None
        def reset():
            global _VALUE
            if _VALUE is not None:
                _VALUE = None
    """,
    "a local that shadows the module name": """
        _CACHE = {}
        def build(key):
            _CACHE = {}
            if key not in _CACHE:
                _CACHE[key] = object()
            return _CACHE
    """,
    "a probe that caches a literal": """
        _AVAILABLE = None
        def available():
            global _AVAILABLE
            if _AVAILABLE is None:
                try:
                    import prometheus_client
                    _AVAILABLE = True
                except ImportError:
                    _AVAILABLE = False
            return _AVAILABLE
    """,
    "a throttle that updates a timestamp": """
        import time
        _LAST: dict = {}
        def should_log(key):
            now = time.monotonic()
            last = _LAST.get(key, 0.0)
            if now - last >= 60.0:
                _LAST[key] = now
                return True
            return False
    """,
    "a registry storing what its caller passed": """
        _BACKENDS = {}
        def register(scheme, resolver):
            if scheme in _BACKENDS:
                raise ValueError(scheme)
            _BACKENDS[scheme] = resolver
    """,
    "going through BuildOnce": """
        from threetears.observe import BuildOnce
        _CACHE = BuildOnce()
        def get(key):
            return _CACHE.get(key, object)
    """,
}


@pytest.mark.parametrize("source", list(_NOT_FILLS.values()), ids=list(_NOT_FILLS))
def test_what_is_not_a_fill_is_not_refused(source: str) -> None:
    """a setter, a reset, a shadowing local and the helper itself are left alone.

    :param source: a module with no hand-rolled fill
    :ptype source: str
    :return: none
    :rtype: None
    :raises AssertionError: if the walker flags it
    """
    assert not lazy_fills(textwrap.dedent(source))
