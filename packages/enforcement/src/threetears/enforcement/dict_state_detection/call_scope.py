"""which classes are call-scoped: every instance lives and dies inside one call, so its containers may change in place.

Pace's ruling of 2026-11-24: function-local and call-scoped mutation is safe. The changed-in-place rule targets only
state that is shared or outlives its call (attributes of long-lived objects, module state, anything another coroutine
or thread can reach). A class is call-scoped when, across every tree the walk is given, all of these hold:

- it is constructed somewhere, only inside functions, and never subclassed;
- no method lets ``self`` out: ``self`` is only ever read through an attribute (never passed, assigned, returned,
  yielded, or captured by a nested function or lambda);
- each construction (``C(...)``, ``cls(...)`` returned by a factory called as ``C.factory(...)``) goes only to:
  a discarded expression; a local name; ``return`` from a function whose every call site, in turn, does the same (an
  undecorated function, never referenced but called); an ``await``; an argument of a call that keeps it local:
  an unresolved callee that is no known long-lived sink (``create_task``, ``append``, ``register``, ``partial``, ...),
  each same-named function in the trees whose matching parameter is itself kept local (an ``async def`` only when
  awaited at the call), or the constructor of another call-scoped class;
- each local name it is bound to is never declared ``global`` / ``nonlocal``, never read inside a nested function or
  lambda (a closure a task or callback could keep), and only read: through an attribute (unless that attribute's
  value is stored on an attribute or entry, or handed to a sink), in a test, a format, an iteration, a ``with``, or a
  flow above (alias, return, argument).

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
_STORE_IN: frozenset[str] = frozenset({"add", "append", "appendleft", "insert", "setdefault"})

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

#: how many calls deep a flow is followed before it counts as unsure
_MAX_DEPTH = 6

_Function = ast.FunctionDef | ast.AsyncFunctionDef


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
        self.verdicts: dict[str, bool] = {}
        self.in_progress: set[str] = set()
        self.locals_seen: set[tuple[int, str]] = set()

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

    def value_ok(self, node: ast.expr, depth: int) -> bool:
        """whether the value ``node`` produces stays inside its call."""
        if depth > _MAX_DEPTH:
            return False
        parent = self.parent(node)
        if isinstance(parent, ast.Expr):
            return True
        if isinstance(parent, ast.Await):
            return self.value_ok(parent, depth)
        if isinstance(parent, (ast.Assign, ast.AnnAssign)) and parent.value is node:
            targets = parent.targets if isinstance(parent, ast.Assign) else [parent.target]
            scope = self.scope_of(parent)
            if len(targets) == 1 and isinstance(targets[0], ast.Subscript):
                # stored in an entry of a local container (``by_name[k] = x``): that container must stay local
                return self.local_container_ok(targets[0].value, scope, depth)
            if len(targets) != 1 or not isinstance(targets[0], ast.Name):
                return False
            if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return False
            return self.local_ok(scope, targets[0].id, depth)
        if isinstance(parent, ast.Return):
            scope = self.scope_of(parent)
            if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return False
            return self.returns_ok(scope, depth + 1)
        if isinstance(parent, ast.keyword):
            call = self.parent(parent)
            return isinstance(call, ast.Call) and self.arg_ok(call, parent.arg, None, depth + 1)
        if isinstance(parent, ast.Call) and node in parent.args:
            if isinstance(parent.func, ast.Attribute) and parent.func.attr in _STORE_IN:
                # put in a container (``planned.append(x)``): fine only when that container is a local that stays
                return self.local_container_ok(parent.func.value, self.scope_of(parent), depth)
            return self.arg_ok(parent, None, parent.args.index(node), depth + 1)
        if isinstance(parent, ast.Tuple) and isinstance(self.parent(parent), ast.Return):
            scope = self.scope_of(parent)
            if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return False
            return self.returns_slot_ok(scope, parent.elts.index(node), len(parent.elts), depth + 1)
        return False

    def local_container_ok(self, container: ast.expr, scope: ast.AST | None, depth: int) -> bool:
        """whether ``container`` is a local name of ``scope`` (bound there, no parameter) that stays inside the call."""
        if not isinstance(container, ast.Name) or not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return False
        if container.id in {arg.arg for arg in _all_args(scope.args)}:
            return False
        bound = any(
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(target, ast.Name) and target.id == container.id
                for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
            )
            for node in ast.walk(scope)
        )
        return bound and self.local_ok(scope, container.id, depth + 1)

    def returns_slot_ok(self, function: _Function, slot: int, width: int, depth: int) -> bool:
        """whether every caller of ``function`` keeps the ``slot`` of the tuple it returns inside the call.

        A caller unpacking the tuple into names (``hits, scan = f()``) is followed through the name at ``slot``; any
        other use of the whole tuple is followed as one value.
        """
        if depth > _MAX_DEPTH or not self.returns_ok_shape(function):
            return False
        for call in self.calls.get(function.name, []):
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
                if not isinstance(target, ast.Name) or not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    return False
                if not self.local_ok(scope, target.id, depth):
                    return False
            elif not self.value_ok(call, depth):
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
        if not self.calls.get(function.name):
            return False
        for ref in self.name_loads.get(function.name, []):
            parent = self.parent(ref)
            if not (isinstance(parent, ast.Call) and parent.func is ref):
                return False
        for attr in self.attr_loads.get(function.name, []):
            parent = self.parent(attr)
            if not (isinstance(parent, ast.Call) and parent.func is attr):
                return False
        return True

    def read_ok(self, name: ast.expr, depth: int) -> bool:
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
        return self.value_ok(name, depth)

    def local_ok(self, function: _Function, local: str, depth: int) -> bool:
        """whether the local ``local`` of ``function`` keeps what it holds inside the call."""
        key = (id(function), local)
        if key in self.locals_seen:
            return True
        self.locals_seen.add(key)
        for node in ast.walk(function):
            if isinstance(node, (ast.Global, ast.Nonlocal)) and local in node.names:
                return False
            if isinstance(node, ast.Name) and node.id == local and isinstance(node.ctx, ast.Load):
                if self.scope_of(node) is not function:
                    return False
                if not self.read_ok(node, depth):
                    return False
        return True

    def returns_ok(self, function: _Function, depth: int) -> bool:
        """whether every caller of ``function`` keeps its result inside the call."""
        if depth > _MAX_DEPTH or not self.returns_ok_shape(function):
            return False
        return all(self.value_ok(call, depth) for call in self.calls[function.name])

    def arg_ok(self, call: ast.Call, keyword: str | None, position: int | None, depth: int) -> bool:
        """whether the callee of ``call`` keeps the argument at ``position`` / ``keyword`` inside the call."""
        if depth > _MAX_DEPTH:
            return False
        name = _callee_name(call)
        if name is None or name in _SINKS:
            return False
        if name in self.classes:
            return self.class_ok(name, depth)
        defs = self.functions.get(name, [])
        if not defs:
            # an unresolved callee may hand back something holding the argument (a coroutine, a wrapper): its
            # result must stay inside the call too, unless the callee builds a fresh value from it
            return name in _FRESH or self.read_ok(call, depth)
        parent = self.parent(call)
        for function in defs:
            if isinstance(function, ast.AsyncFunctionDef) and not isinstance(parent, ast.Await):
                return False
            params = [*function.args.posonlyargs, *function.args.args]
            bound = isinstance(call.func, ast.Attribute) and params and params[0].arg in {"self", "cls"}
            if keyword is not None:
                if keyword not in {arg.arg for arg in (*params, *function.args.kwonlyargs)}:
                    return False
                param = keyword
            else:
                index = (position or 0) + (1 if bound else 0)
                if index >= len(params):
                    return False
                param = params[index].arg
            if not self.local_ok(function, param, depth):
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

    def class_ok(self, name: str, depth: int) -> bool:
        """whether every instance of the class ``name`` lives and dies inside one call."""
        if name in self.verdicts:
            return self.verdicts[name]
        if name in self.in_progress:
            return True
        if depth == 0:
            # a local found fine earlier counts as fine only within one question: every refusal ends that question
            self.locals_seen.clear()
        self.in_progress.add(name)
        try:
            verdict = self._class_ok(name, depth)
        finally:
            self.in_progress.discard(name)
        if depth == 0:
            self.verdicts[name] = verdict
        return verdict

    def _class_ok(self, name: str, depth: int) -> bool:
        classes = self.classes[name]
        if name in self.subclassed or any(self.leaks_self(cls) for cls in classes):
            return False
        constructed = False
        for ref in self.name_loads.get(name, []):
            if id(ref) in self.annotation_nodes:
                continue
            parent = self.parent(ref)
            if isinstance(parent, ast.Call) and parent.func is ref:
                constructed = True
                if self.scope_of(parent) is None or not self.value_ok(parent, depth):
                    return False
            elif isinstance(parent, (ast.Attribute, ast.ExceptHandler, ast.ClassDef)):
                # ``C.CONST``, ``C.factory(...)`` (each factory's calls are followed below), ``except C``, a base
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


def call_scoped_classes(trees: Iterable[ast.Module]) -> frozenset[str]:
    """the names of the classes in ``trees`` whose every instance lives and dies inside one call.

    :param trees: every module of the repo's walked trees (a class's uses are looked for in all of them)
    :ptype trees: Iterable[ast.Module]
    :return: the call-scoped class names
    :rtype: frozenset[str]
    """
    index = _Trees(trees)
    return frozenset(name for name in sorted(index.classes) if index.class_ok(name, 0))
