"""which classes are call-scoped: every instance lives and dies inside one call, so its containers may change in place.

Pace's ruling of 2026-11-24: function-local and call-scoped mutation is safe. The changed-in-place rule targets only
state that is shared or outlives its call (attributes of long-lived objects, module state, anything another coroutine
or thread can reach). A class is call-scoped when, across every tree the walk is given, all of these hold:

- it is constructed somewhere, only inside functions (or as the ``field(default_factory=C)`` of a class-body field of
  another call-scoped class, so it lives exactly as long as that holder), and never subclassed;
- no method lets ``self`` out: ``self`` is only ever read through an attribute (never passed, assigned, returned,
  yielded, or captured by a nested function or lambda);
- each construction (``C(...)``, ``cls(...)`` returned by a factory called as ``C.factory(...)``) goes only to:
  a discarded expression; a comparison or a format (consumed there); a local name; ``return`` from a function whose
  every call site, in turn, does the same (an undecorated function, never referenced but called); an ``await``; an
  argument of a call that keeps it local: an unresolved callee that is no known long-lived sink (``create_task``,
  ``append``, ``register``, ``partial``, ...), each same-named function in the trees whose matching parameter is
  itself kept local (an ``async def`` only when awaited at the call), each function a dispatch table can hand back
  (``TABLE[key](...)``, or a local bound once to ``TABLE[key]`` / ``TABLE.get(key)``, where ``TABLE`` is a dict
  display bound once, in the function or at module level, only ever read by key, whose every value names a function
  in the trees), the function an awaited ``asyncio.to_thread(f, ...)`` runs (one worker thread, awaited, at a time),
  or the constructor of another call-scoped class;
- each local name it is bound to is never declared ``global`` / ``nonlocal``, and only read: through an attribute
  (unless that attribute's value is stored on an attribute or entry, or handed to a sink), in a test, a format, an
  iteration, a ``with``, or a flow above (alias, return, argument). A read inside a nested function counts only when
  that closure itself stays in the call: an undecorated ``def`` whose name is such a local in turn, or a lambda whose
  value flows as above (never one a task or callback could keep). Calling a local that holds a function keeps the
  function; the call's result is followed instead.

Anything else, or anything unresolved past a few hops, means not call-scoped: unsure is flagged.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable

__all__ = ["call_scoped_classes"]

#: calls that keep what they are handed beyond the call (a task, a callback, a registry, a long-lived container)
_SINKS: frozenset[str] = frozenset(
    {
        "Process",
        "Thread",
        "Timer",
        "WeakValueDictionary",
        "add",
        "add_callback",
        "add_done_callback",
        "add_listener",
        "append",
        "appendleft",
        "call_at",
        "call_later",
        "call_soon",
        "call_soon_threadsafe",
        "connect",
        "copy",
        "create_task",
        "deepcopy",
        "ensure_future",
        "extend",
        "gather",
        "insert",
        "model_copy",
        "on",
        "partial",
        "put",
        "put_nowait",
        "ref",
        "register",
        "replace",
        "run_coroutine_threadsafe",
        "run_in_executor",
        "send",
        "set",
        "setattr",
        "setdefault",
        "start_soon",
        "submit",
        "subscribe",
        "to_thread",
        "update",
    }
)

#: the methods that put what they are handed into the container they are called on
_STORE_IN: frozenset[str] = frozenset({"add", "append", "appendleft", "extend", "extendleft", "insert", "setdefault"})

#: the sinks that copy a container's entries in, never keeping the container handed to them
_COPY_IN: frozenset[str] = frozenset({"update", "extend", "extendleft"})

#: unresolved calls whose result is a fresh value that never holds the argument itself
_FRESH: frozenset[str] = frozenset(
    {
        "all",
        "any",
        "asdict",
        "astuple",
        "bool",
        "dumps",
        "float",
        "format",
        "hash",
        "id",
        "int",
        "len",
        "print",
        "repr",
        "str",
        "sum",
    }
)

#: calls whose arguments are type checks, never a flow of the value
_TYPE_CHECKS: frozenset[str] = frozenset({"isinstance", "issubclass", "cast", "type"})

#: the generic types whose subscript names a class only as a type (``Callable[[C], R]``), never handing it anywhere
_GENERICS: frozenset[str] = frozenset(
    {
        "AsyncIterator",
        "Awaitable",
        "Callable",
        "ClassVar",
        "Coroutine",
        "Final",
        "Iterable",
        "Iterator",
        "Mapping",
        "Optional",
        "Sequence",
        "Union",
        "dict",
        "frozenset",
        "list",
        "set",
        "tuple",
        "type",
    }
)

#: how many calls deep a flow is followed before it counts as unsure. 14 is the least that sees the deepest truly
#: call-scoped object met so far: the hub compiler's _Emission, one accumulator handed down the emitter 14 calls deep
#: (measured 2026-11-23: 13 refuses it; _Walk needs 8, _Compilation 8). A flow returning to the call it came in by
#: resumes at that call's depth, so the limit bounds how deep calls nest, not how long a flow is.
_MAX_DEPTH = 14

_Function = ast.FunctionDef | ast.AsyncFunctionDef

#: the call path a value came in by: each function it entered as an argument, with the call it entered by
_Sites = tuple[tuple[_Function, ast.Call, int], ...]


def _callee_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


class _Trees:
    """the trees, indexed for the flow questions (built and dropped inside one :func:`call_scoped_classes`)."""

    def __init__(self, trees: Iterable[ast.Module]) -> None:
        self.parents: dict[int, ast.AST] = {}
        self.functions: dict[str, list[_Function]] = {}
        self.calls: dict[str, list[ast.Call]] = {}
        self.classes: dict[str, list[ast.ClassDef]] = {}
        self.name_loads: dict[str, list[ast.Name]] = {}
        self.attr_loads: dict[str, list[ast.Attribute]] = {}
        self.annotation_nodes: set[int] = set()
        self.subclassed: set[str] = set()
        for tree in trees:
            for node in ast.walk(tree):
                for child in ast.iter_child_nodes(node):
                    self.parents[id(child)] = node
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.functions.setdefault(node.name, []).append(node)
                    annotations = [arg.annotation for arg in _all_args(node.args)]
                    annotations.append(node.returns)
                    for annotation in annotations:
                        if annotation is not None:
                            self.annotation_nodes.update(id(inner) for inner in ast.walk(annotation))
                elif isinstance(node, ast.AnnAssign):
                    self.annotation_nodes.update(id(inner) for inner in ast.walk(node.annotation))
                elif isinstance(node, ast.ClassDef):
                    self.classes.setdefault(node.name, []).append(node)
                    for base in node.bases:
                        base_name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", None)
                        if base_name is not None:
                            self.subclassed.add(base_name)
                elif isinstance(node, ast.Call):
                    name = _callee_name(node)
                    if name is not None:
                        self.calls.setdefault(name, []).append(node)
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    self.name_loads.setdefault(node.id, []).append(node)
                elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                    self.attr_loads.setdefault(node.attr, []).append(node)
        self.verdicts: dict[int, bool] = {}
        self.in_progress: set[int] = set()
        self.locals_seen: set[tuple[int, str, bool]] = set()
        self.containers_seen: set[tuple[int, str, int]] = set()
        # per question: the functions a ``return`` handed the value back from, and the calls into each other one
        # still waiting to learn whether it does (a local already walked, revisited through another call)
        self.returning: set[int] = set()
        self.waiting: dict[int, list[tuple[ast.Call, int, _Sites]]] = {}

    def parent(self, node: ast.AST) -> ast.AST | None:
        return self.parents.get(id(node))

    def scope_of(self, node: ast.AST) -> ast.AST | None:
        """the innermost function or lambda around ``node`` (None at module or class level)."""
        current = self.parent(node)
        while current is not None:
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                return current
            if isinstance(current, ast.ClassDef):
                return None
            current = self.parent(current)
        return None

    def class_of(self, node: ast.AST) -> ast.ClassDef | None:
        current = self.parent(node)
        while current is not None:
            if isinstance(current, ast.ClassDef):
                return current
            current = self.parent(current)
        return None

    # ------------------------------------------------------------------ flows of a value
    #
    # ``sites`` is the call path a value came in by: each entry is a function the value entered as an argument, and
    # the call it entered by. A value that function returns goes back to that one call only (its other callers never
    # handed it this value), so the flow resumes there; with no entry for the function, every caller is followed.

    def value_ok(self, node: ast.expr, depth: int, sites: _Sites = ()) -> bool:
        """whether the value ``node`` produces stays inside its call."""
        if depth > _MAX_DEPTH:
            return False
        parent = self.parent(node)
        if isinstance(parent, (ast.Expr, ast.Compare, ast.FormattedValue)):
            # discarded, compared or formatted: consumed where it stands
            return True
        if isinstance(parent, (ast.For, ast.AsyncFor, ast.comprehension)) and parent.iter is node:
            # iterated where it stands, as a read of a local is
            return True
        if isinstance(parent, ast.Await):
            return self.value_ok(parent, depth, sites)
        if isinstance(parent, ast.IfExp) and node is not parent.test:
            # the branch's value is the conditional's value
            return self.value_ok(parent, depth, sites)
        if isinstance(parent, (ast.ListComp, ast.SetComp, ast.GeneratorExp)) and node is parent.elt:
            # each element is held by the comprehension's value
            return self.value_ok(parent, depth, sites)
        if isinstance(parent, ast.Starred):
            return self.starred_ok(parent, depth, sites)
        if isinstance(parent, ast.Lambda) and node is parent.body:
            # what a lambda hands back reaches whoever calls it: the lambda itself is followed, and so is the result
            # of each call made through a name holding it
            return self.value_ok(parent, depth + 1, sites)
        if isinstance(parent, (ast.Assign, ast.AnnAssign)) and parent.value is node:
            targets = parent.targets if isinstance(parent, ast.Assign) else [parent.target]
            scope = self.scope_of(parent)
            if len(targets) == 1 and isinstance(targets[0], ast.Subscript):
                # stored in an entry of a local container (``by_name[k] = x``): that container must stay local
                return self.local_container_ok(targets[0].value, scope, depth, sites)
            if len(targets) != 1 or not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return False
            # a name, or an unpacking into names (``term, label = f()``): each name may hold what was unpacked
            target = targets[0]
            names = target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target]
            names = [name.value if isinstance(name, ast.Starred) else name for name in names]
            return all(isinstance(name, ast.Name) and self.local_ok(scope, name.id, depth, sites) for name in names)
        if isinstance(parent, ast.Return):
            scope = self.scope_of(parent)
            if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return False
            self.returning.add(id(scope))
            for waited, at, outer in self.waiting.pop(id(scope), []):
                if not self.value_ok(waited, at, outer):
                    return False
            if sites and sites[-1][0] is scope:
                return self.value_ok(sites[-1][1], sites[-1][2], sites[:-1])
            return self.returns_ok(scope, depth + 1)
        if isinstance(parent, ast.keyword):
            call = self.parent(parent)
            return isinstance(call, ast.Call) and self.arg_ok(call, parent.arg, None, depth + 1, sites)
        if isinstance(parent, ast.Call) and node in parent.args:
            if isinstance(parent.func, ast.Attribute) and parent.func.attr in _STORE_IN:
                # put in a container (``planned.append(x)``): fine only when that container is a local that stays
                return self.local_container_ok(parent.func.value, self.scope_of(parent), depth, sites)
            return self.arg_ok(parent, None, parent.args.index(node), depth + 1, sites)
        if isinstance(parent, ast.Tuple) and isinstance(self.parent(parent), ast.Return):
            scope = self.scope_of(parent)
            if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return False
            slot = parent.elts.index(node)
            if sites and sites[-1][0] is scope:
                return self.slot_ok([sites[-1][1]], slot, len(parent.elts), sites[-1][2], sites[:-1])
            if not self.returns_ok_shape(scope):
                return False
            return self.slot_ok(self.calls_of(scope), slot, len(parent.elts), depth + 1, ())
        if isinstance(parent, (ast.List, ast.Tuple, ast.Set)) and not isinstance(
            getattr(parent, "ctx", None), ast.Store
        ):
            # an element of a display: the display holds it
            return self.value_ok(parent, depth, sites)
        return False

    def starred_ok(self, starred: ast.Starred, depth: int, sites: _Sites) -> bool:
        """whether values unpacked by ``starred`` stay: in a display, the display holds them; in a call, an unresolved
        callee that is no sink is followed through its result, and each resolved one must keep every parameter the
        values can land in."""
        holder = self.parent(starred)
        if isinstance(holder, (ast.List, ast.Tuple, ast.Set)):
            return self.value_ok(holder, depth, sites)
        if not isinstance(holder, ast.Call) or starred not in holder.args:
            return False
        name = _callee_name(holder)
        if name is None or name in _SINKS or name in self.classes or self.dispatched(holder) is not None:
            return False
        if name not in self.functions:
            return name in _FRESH or self.read_ok(holder, depth + 1, sites)
        return self.spread_ok(holder, holder.args.index(starred), self.callees(holder, name), depth + 1, sites)

    def local_container_ok(self, container: ast.expr, scope: ast.AST | None, depth: int, sites: _Sites = ()) -> bool:
        """whether ``container`` is a local of ``scope`` that stays inside the call: bound there, or a parameter
        whose every argument (the one call it came in by, when known) is such a container in turn; or a field of
        a call-scoped object (see :meth:`field_container_ok`). An entry of such a container (``d[k]``,
        ``d.setdefault(k, [])``, ``d.get(k)``) is held by it."""
        if isinstance(container, ast.Subscript):
            return self.local_container_ok(container.value, scope, depth, sites)
        if (
            isinstance(container, ast.Call)
            and isinstance(container.func, ast.Attribute)
            and container.func.attr in {"get", "setdefault"}
        ):
            return self.local_container_ok(container.func.value, scope, depth, sites)
        if isinstance(container, ast.Attribute) and isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return self.field_container_ok(container, scope, depth + 1)
        if not isinstance(container, ast.Name) or not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return False
        if container.id in {arg.arg for arg in _all_args(scope.args)}:
            return self.param_container_ok(scope, container.id, depth + 1, sites)
        bound = any(
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(target, ast.Name) and target.id == container.id
                for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
            )
            for node in ast.walk(scope)
        )
        return bound and self.local_ok(scope, container.id, depth + 1, sites)

    def field_container_ok(self, container: ast.Attribute, scope: _Function, depth: int) -> bool:
        """whether ``container`` (``x.field``) is a field of a call-scoped object whose entries never leave the call.

        ``x`` must be a parameter or local of ``scope`` annotated with a class of the trees that is call-scoped, so the
        container lives in the call; and every read of an entry of any ``.field`` anywhere (``o.field[k]``,
        ``o.field.get(k)``) must keep that entry in the call too. Any other use of a ``.field`` is unsure.
        """
        if depth > _MAX_DEPTH or not isinstance(container.value, ast.Name):
            return False
        owner = _annotated_class(scope, container.value.id)
        if owner is None or owner not in self.classes or not self.class_ok(owner, scope):
            return False
        for load in self.attr_loads.get(container.attr, []):
            parent = self.parent(load)
            if isinstance(parent, ast.Subscript) and parent.value is load:
                if isinstance(parent.ctx, ast.Load) and not self.value_ok(parent, depth + 1):
                    return False
            elif isinstance(parent, ast.Attribute) and parent.value is load and parent.attr == "get":
                call = self.parent(parent)
                if not (isinstance(call, ast.Call) and call.func is parent and self.value_ok(call, depth + 1)):
                    return False
            elif not (isinstance(parent, (ast.Compare, ast.UnaryOp)) or isinstance(parent, ast.BoolOp)):
                return False
        return True

    def param_container_ok(self, function: _Function, param: str, depth: int, sites: _Sites) -> bool:
        """whether every container handed to ``function`` as ``param`` is a local container that stays."""
        if depth > _MAX_DEPTH:
            return False
        # one walk per call it came in by (0: every caller), each assumed fine while it is on the way
        key = (id(function), param, id(sites[-1][1]) if sites and sites[-1][0] is function else 0)
        if key in self.containers_seen:
            return True
        self.containers_seen.add(key)
        if not self.local_ok(function, param, depth, sites):
            return False
        if sites and sites[-1][0] is function:
            calls, outer = [sites[-1][1]], sites[:-1]
        elif self.returns_ok_shape(function):
            calls, outer = self.calls_of(function), ()
        else:
            return False
        for call in calls:
            argument = _argument_for(function, call, param)
            if argument is None or not self.local_container_ok(argument, self.scope_of(call), depth, outer):
                return False
        return True

    def slot_ok(self, calls: list[ast.Call], slot: int, width: int, depth: int, sites: _Sites) -> bool:
        """whether each of ``calls`` keeps the ``slot`` of the tuple it hands back inside the call.

        A caller unpacking the tuple into names (``hits, scan = f()``) is followed through the name at ``slot``; any
        other use of the whole tuple is followed as one value.
        """
        if depth > _MAX_DEPTH:
            return False
        for call in calls:
            value: ast.AST = call
            parent = self.parent(call)
            if isinstance(parent, ast.Await):
                value, parent = parent, self.parent(parent)
            if (
                isinstance(parent, ast.Assign)
                and parent.value is value
                and len(parent.targets) == 1
                and isinstance(parent.targets[0], ast.Tuple)
                and len(parent.targets[0].elts) == width
            ):
                target = parent.targets[0].elts[slot]
                scope = self.scope_of(parent)
                if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    return False
                # a name, or a nested unpacking of names (``(reading,), _ = f()``): each name holds what it unpacks
                names = target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target]
                names = [name.value if isinstance(name, ast.Starred) else name for name in names]
                if not all(
                    isinstance(name, ast.Name) and self.local_ok(scope, name.id, depth, sites) for name in names
                ):
                    return False
            elif not self.value_ok(call, depth, sites):
                return False
        return True

    def returns_ok_shape(self, function: _Function) -> bool:
        """whether ``function``'s results can all be followed: undecorated (or a class or static method), called
        somewhere, and never referenced except to be called."""
        if not all(
            isinstance(decorator, ast.Name) and decorator.id in {"classmethod", "staticmethod"}
            for decorator in function.decorator_list
        ):
            return False
        if not self.calls_of(function):
            return False
        for ref in [*self.name_loads.get(function.name, []), *self.attr_loads.get(function.name, [])]:
            parent = self.parent(ref)
            if not self.may_reach(function, ref) or (isinstance(parent, ast.Call) and parent.func is ref):
                continue
            # a value of a provable dispatch table: its calls are the table's (see :meth:`calls_of`)
            if isinstance(ref, ast.Name) and self.table_calls(ref) is not None:
                continue
            return False
        return True

    def calls_of(self, function: _Function) -> list[ast.Call]:
        """the calls that may reach ``function``: every same-named call, less those that certainly cannot, and every
        call through a provable dispatch table holding it."""
        found = [call for call in self.calls.get(function.name, []) if self.may_reach(function, call.func)]
        for ref in self.name_loads.get(function.name, []):
            if self.may_reach(function, ref):
                found.extend(self.table_calls(ref) or [])
        return found

    def may_reach(self, function: _Function, ref: ast.expr) -> bool:
        """whether the reference ``ref`` (a name, or an attribute) may name ``function``.

        Only a function at the top of its module is told apart: a bare name in another module names it only where
        that module imports the name, and ``self.f`` / ``cls.f`` never does. Anything else may.
        """
        home = self.parent(function)
        if isinstance(home, ast.ClassDef):
            return self.may_reach_method(home, ref)
        if not isinstance(home, ast.Module):
            return True
        if isinstance(ref, ast.Attribute):
            return not (isinstance(ref.value, ast.Name) and ref.value.id in {"self", "cls"})
        module = self.module_of(ref)
        if module is home or module is None:
            return True
        return any(
            isinstance(node, (ast.Import, ast.ImportFrom))
            and any(
                alias.name == "*" or (alias.asname or alias.name.split(".")[0]) == function.name for alias in node.names
            )
            for node in ast.walk(module)
        )

    def may_reach_method(self, owner: ast.ClassDef, ref: ast.expr) -> bool:
        """whether ``ref`` may name a method of ``owner``.

        A bare name names it only in ``owner``'s own body. ``self.m`` / ``cls.m`` in a method of a class D names it
        only when D is ``owner`` or one of its bases (by name, across the trees), unless ``owner`` is subclassed
        (then anything may). Any other receiver may.
        """
        if isinstance(ref, ast.Name):
            return self.scope_of(ref) is None and self.class_of(ref) is owner
        if not (isinstance(ref, ast.Attribute) and isinstance(ref.value, ast.Name) and ref.value.id in {"self", "cls"}):
            return True
        if owner.name in self.subclassed or len(self.classes.get(owner.name, [])) != 1:
            return True
        around = self.class_of(ref)
        if around is None:
            return True
        return around is owner or around.name in self.ancestors(owner)

    def ancestors(self, cls: ast.ClassDef) -> frozenset[str]:
        """the names of ``cls``'s bases, and theirs, as far as the trees define them."""
        found: set[str] = set()
        pending = [cls]
        while pending:
            current = pending.pop()
            for base in current.bases:
                name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", None)
                if isinstance(name, str) and name not in found:
                    found.add(name)
                    pending.extend(self.classes.get(name, []))
        return frozenset(found)

    def read_ok(self, name: ast.expr, depth: int, sites: _Sites = ()) -> bool:
        """whether one read of a local bound to the instance (or a value that may hold it) keeps it inside the call."""
        parent = self.parent(name)
        if isinstance(parent, ast.Attribute) and parent.value is name:
            holder = self.parent(parent)
            if isinstance(holder, (ast.Assign, ast.AnnAssign)) and holder.value is parent:
                targets = holder.targets if isinstance(holder, ast.Assign) else [holder.target]
                return all(isinstance(target, ast.Name) for target in targets)
            if isinstance(holder, ast.Call) and parent in holder.args:
                # a field's entries copied into another container (``d.update(report.ids)``) leave the field behind
                callee = _callee_name(holder)
                return callee not in _SINKS or callee in _COPY_IN
            return True
        if isinstance(parent, ast.Subscript) and parent.value is name:
            return True
        if isinstance(parent, (ast.Compare, ast.BoolOp, ast.UnaryOp, ast.FormattedValue, ast.withitem)):
            return True
        if isinstance(parent, (ast.If, ast.While, ast.Assert, ast.IfExp)) and getattr(parent, "test", None) is name:
            return True
        if isinstance(parent, (ast.For, ast.AsyncFor, ast.comprehension)) and parent.iter is name:
            return True
        if isinstance(parent, ast.Call) and name in parent.args and _callee_name(parent) in _TYPE_CHECKS:
            return True
        if isinstance(parent, ast.Call) and parent.func is name:
            # called: the call keeps nothing of the callable; what it hands back (a coroutine holding a closure's
            # cells, say) is followed instead
            return self.value_ok(parent, depth + 1, sites)
        return self.value_ok(name, depth, sites)

    def local_ok(self, function: _Function, local: str, depth: int, sites: _Sites = ()) -> bool:
        """whether the local ``local`` of ``function`` keeps what it holds inside the call.

        Each local is walked once per question. Its reads are the same however the value came in; only where a
        ``return`` of it goes differs, so a later visit through another call checks just that call's result.
        """
        entered = bool(sites) and sites[-1][0] is function
        if (id(function), local, False) in self.locals_seen:
            # walked with every caller of a ``return`` followed: nothing left to check
            return True
        if entered and (id(function), local, True) in self.locals_seen:
            if id(function) in self.returning:
                return self.value_ok(sites[-1][1], sites[-1][2], sites[:-1])
            self.waiting.setdefault(id(function), []).append((sites[-1][1], sites[-1][2], sites[:-1]))
            return True
        self.locals_seen.add((id(function), local, entered))
        for node in ast.walk(function):
            if isinstance(node, (ast.Global, ast.Nonlocal)) and local in node.names:
                return False
            if isinstance(node, ast.Name) and node.id == local and isinstance(node.ctx, ast.Load):
                if not self.closures_ok(node, function, depth, sites):
                    return False
                if not self.read_ok(node, depth, sites):
                    return False
        return True

    def closures_ok(self, node: ast.AST, function: _Function, depth: int, sites: _Sites) -> bool:
        """whether every closure between a read ``node`` and the ``function`` whose local it reads stays in the call.

        A nested ``def`` must be undecorated, its name a local of the scope around it that stays (it is called, or
        handed to a callee that keeps it); a lambda's value must stay as any value does.
        """
        scope = self.scope_of(node)
        while scope is not function:
            if depth > _MAX_DEPTH or scope is None:
                return False
            outer = self.scope_of(scope)
            if isinstance(scope, ast.Lambda):
                if not self.value_ok(scope, depth + 1, sites):
                    return False
            elif (
                not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef))
                or scope.decorator_list
                or not isinstance(outer, (ast.FunctionDef, ast.AsyncFunctionDef))
            ):
                return False
            elif not self.local_ok(outer, scope.name, depth + 1, sites):
                return False
            scope = outer
        return True

    def returns_ok(self, function: _Function, depth: int) -> bool:
        """whether every caller of ``function`` keeps its result inside the call."""
        if depth > _MAX_DEPTH or not self.returns_ok_shape(function):
            return False
        return all(self.value_ok(call, depth) for call in self.calls_of(function))

    def arg_ok(self, call: ast.Call, keyword: str | None, position: int | None, depth: int, sites: _Sites = ()) -> bool:
        """whether the callee of ``call`` keeps the argument at ``position`` / ``keyword`` inside the call."""
        if depth > _MAX_DEPTH:
            return False
        parent = self.parent(call)
        dispatched = self.dispatched(call)
        name = _callee_name(call)
        if dispatched is not None:
            if not dispatched:
                # a dispatch through a table that cannot be proven: unsure
                return False
            defs = dispatched
        elif name == "to_thread" and isinstance(parent, ast.Await) and keyword is None and position:
            # ``await asyncio.to_thread(f, *args)``: one worker thread runs f(*args) while the call waits for it
            runner = call.args[0]
            if not isinstance(runner, ast.Name) or runner.id not in self.functions:
                return False
            if any(isinstance(function, ast.AsyncFunctionDef) for function in self.functions[runner.id]):
                return False
            return self.params_ok(self.functions[runner.id], call, None, position - 1, False, depth, sites)
        elif name is None or name in _SINKS:
            return False
        elif name in self.classes:
            return self.class_ok(name, call)
        else:
            defs = self.callees(call, name)
            if not defs:
                # an unresolved callee may hand back something holding the argument (a coroutine, a wrapper): its
                # result must stay inside the call too, unless the callee builds a fresh value from it
                return name in _FRESH or self.read_ok(call, depth, sites)
        if any(isinstance(function, ast.AsyncFunctionDef) for function in defs) and not isinstance(parent, ast.Await):
            return False
        return self.params_ok(defs, call, keyword, position, isinstance(call.func, ast.Attribute), depth, sites)

    def dispatched(self, call: ast.Call) -> list[_Function] | None:
        """the functions a dispatch-table call can reach, or None when ``call`` is no dispatch through a table.

        ``TABLE[key](...)``, or ``f(...)`` where ``f`` is a local bound only to ``TABLE[key]`` / ``TABLE.get(key)``.
        An unprovable table (rebound, changed, read otherwise, or holding anything but functions of the trees) gives
        an empty list, so no parameter is kept and the flow is unsure.
        """
        dispatch, display = self.dispatch_display(call)
        if not dispatch:
            return None
        if display is None:
            return []
        found: list[_Function] = []
        for value in display.values:
            if not isinstance(value, ast.Name) or value.id not in self.functions or value.id in self.classes:
                return []
            own = self.named_function(value)
            found.extend([own] if own is not None else self.functions[value.id])
        return found

    def dispatch_display(self, call: ast.Call) -> tuple[bool, ast.Dict | None]:
        """whether ``call`` dispatches through a table, and the table's dict display when it is provable."""
        func = call.func
        scope = self.scope_of(call)
        table: ast.expr | None = None
        if isinstance(func, ast.Subscript) and isinstance(func.ctx, ast.Load):
            table = func.value
        elif (
            isinstance(func, ast.Name)
            and func.id not in self.functions
            and func.id not in self.classes
            and isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef))
        ):
            bindings = _bindings(scope, func.id)
            lookups = [_looked_up(value) for value in bindings or []]
            if not lookups or all(looked_up is None for looked_up in lookups):
                return False, None
            for looked_up in lookups:
                if looked_up is None or (table is not None and ast.dump(looked_up) != ast.dump(table)):
                    return True, None
                table = looked_up
        if table is None:
            return False, None
        if not isinstance(table, ast.Name):
            return True, None
        return True, self.table(table.id, scope)

    def named_function(self, ref: ast.Name) -> _Function | None:
        """the one function a bare name reaches by Python's own lookup (see :meth:`module_function`), or None."""
        name = ref.id
        scope = self.scope_of(ref)
        while scope is not None:
            if isinstance(scope, ast.Lambda) or _bindings(scope, name) != []:
                return None
            scope = self.scope_of(scope)
        module = self.module_of(ref)
        if module is None:
            return None
        found = [
            stmt
            for stmt in module.body
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == name
        ]
        if len(found) != 1 or self.module_binds(module, name, found[0]):
            return None
        return found[0]

    def table_calls(self, ref: ast.Name) -> list[ast.Call] | None:
        """the calls that dispatch to the function ``ref`` names as a value of a provable table, or None when
        ``ref`` is no such value (or the table's calls cannot all be found)."""
        display = self.parent(ref)
        if not isinstance(display, ast.Dict) or ref not in display.values:
            return None
        binding = self.parent(display)
        if not isinstance(binding, (ast.Assign, ast.AnnAssign)) or binding.value is not display:
            return None
        targets = binding.targets if isinstance(binding, ast.Assign) else [binding.target]
        if len(targets) != 1 or not isinstance(targets[0], ast.Name):
            return None
        holder = self.scope_of(binding) or self.module_of(binding)
        if holder is None or self.table(targets[0].id, holder) is not display:
            return None
        # a module-level table read from another module (imported) could dispatch from there: unsure
        if isinstance(holder, ast.Module) and any(
            self.module_of(load) is not holder for load in self.name_loads.get(targets[0].id, [])
        ):
            return None
        return [
            node
            for node in ast.walk(holder)
            if isinstance(node, ast.Call) and self.dispatch_display(node) == (True, display)
        ]

    def table(self, name: str, scope: ast.AST | None) -> ast.Dict | None:
        """the dict display ``name`` is bound to once, in ``scope`` or else at module level, and only read by key."""
        if scope is None:
            return None
        bindings = _bindings(scope, name) if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)) else []
        if bindings is None:
            return None
        loads: list[ast.Name]
        if bindings:
            loads = [
                node
                for node in ast.walk(scope)
                if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load)
            ]
        else:
            # a module-level table: bound once, at the top of the module, and read only by key in every tree (a
            # same-named name anywhere else counts as a read of it: stricter, never laxer)
            module = self.module_of(scope)
            bindings = _bindings(module, name) if module is not None else None
            loads = self.name_loads.get(name, [])
        if bindings is None or len(bindings) != 1:
            return None
        display = bindings[0]
        if not isinstance(display, ast.Dict) or None in display.keys:
            return None
        if not all(self.keyed_read(load) for load in loads):
            return None
        return display

    def keyed_read(self, load: ast.Name) -> bool:
        """whether ``load`` reads one entry of the table it names: ``load[key]`` or ``load.get(...)``."""
        parent = self.parent(load)
        if isinstance(parent, ast.Subscript):
            return parent.value is load and isinstance(parent.ctx, ast.Load)
        if isinstance(parent, ast.Attribute) and parent.attr == "get":
            call = self.parent(parent)
            return isinstance(call, ast.Call) and call.func is parent
        return False

    def callees(self, call: ast.Call, name: str) -> list[_Function]:
        """the functions ``call`` (to ``name``) can reach: the one Python's lookup certainly picks, or else every
        same-named function in the trees (stricter)."""
        own = self.module_function(call) or self.own_method(call)
        return [own] if own is not None else self.functions.get(name, [])

    def own_method(self, call: ast.Call) -> _Function | None:
        """the method ``self.m(...)`` reaches when the class around the call defines ``m`` once and is never
        subclassed (so nothing overrides it); None otherwise."""
        func = call.func
        if not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "self"):
            return None
        method = self.scope_of(call)
        cls = self.parent(method) if method is not None else None
        if not isinstance(cls, ast.ClassDef) or cls.name in self.subclassed or len(self.classes[cls.name]) != 1:
            return None
        found = [
            stmt
            for stmt in cls.body
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == func.attr
        ]
        if len(found) != 1 or found[0].decorator_list:
            return None
        for node in ast.walk(cls):
            # an instance attribute of the same name would shadow the method
            if (
                isinstance(node, ast.Attribute)
                and node.attr == func.attr
                and not isinstance(node.ctx, ast.Load)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
            ):
                return None
        return found[0]

    def module_function(self, call: ast.Call) -> _Function | None:
        """the one function a bare-name call reaches by Python's own lookup: a ``def`` at the top of the call's
        module, bound there by nothing else, and by no scope around the call; None when that is not certain (every
        same-named function in the trees is then followed, which is stricter)."""
        if not isinstance(call.func, ast.Name):
            return None
        name = call.func.id
        scope = self.scope_of(call)
        while scope is not None:
            if isinstance(scope, ast.Lambda) or _bindings(scope, name) != []:
                return None
            scope = self.scope_of(scope)
        module = self.module_of(call)
        if module is None:
            return None
        found = [stmt for stmt in module.body if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))]
        found = [stmt for stmt in found if stmt.name == name]
        if len(found) != 1 or self.module_binds(module, name, found[0]):
            return None
        return found[0]

    def module_binds(self, module: ast.Module, name: str, function: _Function | ast.ClassDef) -> bool:
        """whether anything but ``function`` (a def or a class) binds ``name`` at module level (an import anywhere counts
        too)."""
        for node in ast.walk(module):
            if node is function:
                continue
            if isinstance(node, (ast.Global, ast.Nonlocal)) and name in node.names:
                return True
            if isinstance(node, (ast.Import, ast.ImportFrom)) and any(
                (alias.asname or alias.name.split(".")[0]) == name for alias in node.names
            ):
                return True
            bound = (isinstance(node, ast.Name) and node.id == name and not isinstance(node.ctx, ast.Load)) or (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name
            )
            if bound and self.at_module_level(node):
                return True
        return False

    def at_module_level(self, node: ast.AST) -> bool:
        """whether ``node`` binds in its module's own namespace (no function, lambda or class around it)."""
        current = self.parent(node)
        while current is not None and not isinstance(current, ast.Module):
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                return False
            current = self.parent(current)
        return True

    def in_type_expression(self, ref: ast.Name) -> bool:
        """whether ``ref`` stands inside the subscript of a generic type (``Callable[[C, int], R]``, ``list[C]``)."""
        current: ast.AST = ref
        parent = self.parent(current)
        while isinstance(parent, (ast.List, ast.Tuple)) or (
            isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.BitOr)
        ):
            current, parent = parent, self.parent(parent)
        if not isinstance(parent, ast.Subscript) or parent.slice is not current:
            return False
        generic = parent.value
        name = generic.id if isinstance(generic, ast.Name) else getattr(generic, "attr", None)
        return name in _GENERICS

    def field_holder(self, factory: ast.keyword) -> ast.ClassDef | None:
        """the class whose body field ``factory`` (``default_factory=C``) builds, or None anywhere else."""
        call = self.parent(factory)
        if not isinstance(call, ast.Call) or _callee_name(call) != "field":
            return None
        stmt = self.parent(call)
        if not isinstance(stmt, (ast.AnnAssign, ast.Assign)) or stmt.value is not call:
            return None
        holder = self.parent(stmt)
        if not isinstance(holder, ast.ClassDef) or stmt not in holder.body:
            return None
        return holder

    def module_of(self, node: ast.AST) -> ast.Module | None:
        current: ast.AST | None = node
        while current is not None:
            if isinstance(current, ast.Module):
                return current
            current = self.parent(current)
        return None

    def spread_ok(self, call: ast.Call, position: int, defs: list[_Function], depth: int, sites: _Sites) -> bool:
        """whether each of ``defs`` keeps local every parameter values unpacked at ``position`` of ``call`` can bind."""
        if depth > _MAX_DEPTH:
            return False
        awaited = isinstance(self.parent(call), ast.Await)
        for function in defs:
            if isinstance(function, ast.AsyncFunctionDef) and not awaited:
                return False
            params = [*function.args.posonlyargs, *function.args.args]
            bound = isinstance(call.func, ast.Attribute) and params and params[0].arg in {"self", "cls"}
            reachable = params[position + (1 if bound else 0) :]
            if function.args.vararg is not None:
                reachable.append(function.args.vararg)
            entered = (*sites, (function, call, depth))
            if not all(self.local_ok(function, param.arg, depth, entered) for param in reachable):
                return False
        return True

    def params_ok(
        self,
        defs: list[_Function],
        call: ast.Call,
        keyword: str | None,
        position: int | None,
        method_call: bool,
        depth: int,
        sites: _Sites,
    ) -> bool:
        """whether each of ``defs`` keeps the parameter the argument at ``position`` / ``keyword`` binds local.

        ``method_call`` says the callee was reached through an attribute, so a ``self`` / ``cls`` first parameter is
        bound by the receiver rather than by the first argument.
        """
        for function in defs:
            params = [*function.args.posonlyargs, *function.args.args]
            bound = method_call and params and params[0].arg in {"self", "cls"}
            if keyword is not None:
                if keyword not in {arg.arg for arg in (*params, *function.args.kwonlyargs)}:
                    return False
                param = keyword
            else:
                index = (position or 0) + (1 if bound else 0)
                if index >= len(params):
                    return False
                param = params[index].arg
            if not self.local_ok(function, param, depth, (*sites, (function, call, depth))):
                return False
        return True

    # ------------------------------------------------------------------ the class

    def leaks_self(self, cls: ast.ClassDef) -> bool:
        for method in cls.body:
            if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(method):
                if isinstance(node, ast.Name) and node.id == "self" and isinstance(node.ctx, ast.Load):
                    if self.scope_of(node) is not method:
                        return True
                    parent = self.parent(node)
                    if isinstance(parent, ast.Attribute) and parent.value is node:
                        continue
                    if isinstance(parent, ast.Call) and node in parent.args:
                        # handed to a call: only a known-short-lived one (unresolved, no sink) keeps it in the call
                        callee = _callee_name(parent)
                        if callee in _TYPE_CHECKS or (
                            callee not in _SINKS and callee not in self.functions and callee not in self.classes
                        ):
                            continue
                    return True
        return False

    def class_ok(self, name: str, at: ast.AST | None = None) -> bool:
        """whether every instance of the class ``name`` (as named at ``at``) lives and dies inside one call.

        A name Python's own lookup certainly resolves to one class (one defined at the top of ``at``'s module, bound
        there by nothing else) asks about that class alone; otherwise every same-named class in the trees must be
        call-scoped (stricter).
        """
        own = self.module_class(name, at) if at is not None else None
        if own is not None:
            return self.definition_ok(own)
        return all(self.definition_ok(cls) for cls in self.classes.get(name, [])) and name in self.classes

    def module_class(self, name: str, at: ast.AST) -> ast.ClassDef | None:
        """the one class ``name`` names at ``at`` by Python's own lookup, when that is certain; None otherwise."""
        scope = self.scope_of(at)
        while scope is not None:
            if isinstance(scope, ast.Lambda) or _bindings(scope, name) != []:
                return None
            scope = self.scope_of(scope)
        module = self.module_of(at)
        if module is None:
            return None
        found = [stmt for stmt in module.body if isinstance(stmt, ast.ClassDef) and stmt.name == name]
        if len(found) != 1 or self.module_binds(module, name, found[0]):
            return None
        return found[0]

    def definition_ok(self, cls: ast.ClassDef) -> bool:
        """whether every instance of the class ``cls`` lives and dies inside one call.

        Each class is its own question, with its own depth and its own record of locals found fine (every refusal
        ends a question, so nothing found fine in one is trusted in another). A class asked about inside another's
        question (a constructor argument, a field's holder) is answered as a question of its own, assuming the ones
        on the way fine; that answer is kept only for a class asked about first.
        """
        if id(cls) in self.verdicts:
            return self.verdicts[id(cls)]
        if id(cls) in self.in_progress:
            return True
        outer = (self.locals_seen, self.containers_seen, self.returning, self.waiting)
        self.locals_seen, self.containers_seen, self.returning, self.waiting = set(), set(), set(), {}
        nested = bool(self.in_progress)
        self.in_progress.add(id(cls))
        try:
            verdict = self._class_ok(cls, 0)
        finally:
            self.in_progress.discard(id(cls))
            self.locals_seen, self.containers_seen, self.returning, self.waiting = outer
        if not nested:
            self.verdicts[id(cls)] = verdict
        return verdict

    def may_name_class(self, cls: ast.ClassDef, ref: ast.Name) -> bool:
        """whether the bare name ``ref`` may name ``cls``: in its own module unless certainly another class, and in
        another module only where that module imports the name."""
        module = self.module_of(ref)
        home = self.module_of(cls)
        if module is home:
            own = self.module_class(cls.name, ref)
            return own is None or own is cls
        return module is None or any(
            isinstance(node, (ast.Import, ast.ImportFrom))
            and any(alias.name == "*" or (alias.asname or alias.name.split(".")[0]) == cls.name for alias in node.names)
            for node in ast.walk(module)
        )

    def _class_ok(self, cls: ast.ClassDef, depth: int) -> bool:
        name = cls.name
        classes = [cls]
        if name in self.subclassed or self.leaks_self(cls):
            return False
        constructed = False
        for ref in self.name_loads.get(name, []):
            if id(ref) in self.annotation_nodes or not self.may_name_class(cls, ref):
                continue
            parent = self.parent(ref)
            if isinstance(parent, ast.Call) and parent.func is ref:
                constructed = True
                if self.scope_of(parent) is None or not self.value_ok(parent, depth):
                    return False
            elif isinstance(parent, ast.keyword) and parent.arg == "default_factory":
                # ``x: C = field(default_factory=C)`` on a class body: each instance lives exactly as long as its
                # holder, so it is call-scoped when the holder is
                owner = self.field_holder(parent)
                if owner is None or owner is cls or not self.definition_ok(owner):
                    return False
                constructed = True
            elif isinstance(parent, (ast.Attribute, ast.ExceptHandler, ast.ClassDef)):
                # ``C.CONST``, ``C.factory(...)`` (each factory's calls are followed below), ``except C``, a base
                continue
            elif self.in_type_expression(ref):
                continue
            elif isinstance(parent, ast.Call) and ref in parent.args and _callee_name(parent) in _TYPE_CHECKS:
                continue
            elif isinstance(parent, ast.Tuple) and isinstance(self.parent(parent), (ast.Call, ast.ExceptHandler)):
                holder = self.parent(parent)
                if isinstance(holder, ast.ExceptHandler) or (
                    isinstance(holder, ast.Call) and _callee_name(holder) in _TYPE_CHECKS
                ):
                    continue
                return False
            else:
                return False
        for attr in self.attr_loads.get(name, []):
            parent = self.parent(attr)
            if isinstance(parent, ast.Call) and parent.func is attr:
                constructed = True
                if self.scope_of(parent) is None or not self.value_ok(parent, depth):
                    return False
        for cls in classes:
            for node in ast.walk(cls):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"cls", name}:
                    method = self.scope_of(node)
                    constructed = True
                    if isinstance(self.parent(node), ast.Return) and isinstance(
                        method, (ast.FunctionDef, ast.AsyncFunctionDef)
                    ):
                        # a factory: every call of its name, on any receiver, is followed (a shared name is stricter)
                        if method.decorator_list and not all(
                            isinstance(decorator, ast.Name) and decorator.id in {"classmethod", "staticmethod"}
                            for decorator in method.decorator_list
                        ):
                            return False
                        calls = self.calls.get(method.name, [])
                        if not all(self.value_ok(call, depth) for call in calls):
                            return False
                    elif not self.value_ok(node, depth):
                        return False
        return constructed


def _all_args(args: ast.arguments) -> list[ast.arg]:
    found = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    if args.vararg is not None:
        found.append(args.vararg)
    if args.kwarg is not None:
        found.append(args.kwarg)
    return found


def _annotated_class(function: _Function, name: str) -> str | None:
    """the class name a parameter or local ``name`` of ``function`` is annotated with (``x: C``), or None."""
    annotations = [arg.annotation for arg in _all_args(function.args) if arg.arg == name]
    annotations.extend(
        node.annotation
        for node in ast.walk(function)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name
    )
    found: set[str] = set()
    for annotation in annotations:
        if isinstance(annotation, ast.Name):
            found.add(annotation.id)
        elif isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
            found.add(annotation.value)
        else:
            return None
    return next(iter(found)) if len(found) == 1 else None


def _argument_for(function: _Function, call: ast.Call, param: str) -> ast.expr | None:
    """the argument ``call`` binds to ``function``'s parameter ``param``; None when unsure (unpacked arguments)."""
    if any(isinstance(arg, ast.Starred) for arg in call.args) or any(kw.arg is None for kw in call.keywords):
        return None
    for keyword in call.keywords:
        if keyword.arg == param:
            return keyword.value
    params = [arg.arg for arg in (*function.args.posonlyargs, *function.args.args)]
    if param not in params:
        return None
    bound = isinstance(call.func, ast.Attribute) and params[0] in {"self", "cls"}
    index = params.index(param) - (1 if bound else 0)
    return call.args[index] if 0 <= index < len(call.args) else None


def _bindings(holder: ast.AST, name: str) -> list[ast.expr] | None:
    """the values ``name`` is bound to in ``holder`` (a function, or a module's top level), each by a plain
    ``name = value`` / ``name: T = value``; None when it is bound any other way too (a parameter, a loop or ``with``
    target, an augmented or walrus assignment, an import, a ``del``, a nested scope).
    """
    if isinstance(holder, (ast.FunctionDef, ast.AsyncFunctionDef)) and name in {
        arg.arg for arg in _all_args(holder.args)
    }:
        return None
    values: list[ast.expr] = []
    plain: set[int] = set()
    nodes = ast.walk(holder) if not isinstance(holder, ast.Module) else iter(holder.body)
    for node in nodes:
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if len(targets) == 1 and isinstance(targets[0], ast.Name) and targets[0].id == name:
                values.append(node.value)
                plain.add(id(targets[0]))
    for node in ast.walk(holder):
        if isinstance(node, ast.Name) and node.id == name and not isinstance(node.ctx, ast.Load):
            if id(node) not in plain:
                return None
        elif isinstance(node, (ast.Global, ast.Nonlocal)) and name in node.names:
            return None
        elif isinstance(node, (ast.Import, ast.ImportFrom)) and any(
            (alias.asname or alias.name.split(".")[0]) == name for alias in node.names
        ):
            return None
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            if node is not holder:
                return None
    return values


def _looked_up(value: ast.expr) -> ast.expr | None:
    """the table ``value`` reads one entry of (``TABLE[key]``, ``TABLE.get(key)``), or None."""
    if isinstance(value, ast.Subscript) and isinstance(value.ctx, ast.Load):
        return value.value
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "get"
        and len(value.args) == 1
        and not value.keywords
    ):
        return value.func.value
    return None


def call_scoped_classes(trees: Iterable[ast.Module]) -> frozenset[str]:
    """the names of the classes in ``trees`` whose every instance lives and dies inside one call.

    :param trees: every module of the repo's walked trees (a class's uses are looked for in all of them)
    :ptype trees: Iterable[ast.Module]
    :return: the call-scoped class names
    :rtype: frozenset[str]
    """
    index = _Trees(trees)
    return frozenset(name for name in sorted(index.classes) if index.class_ok(name))
