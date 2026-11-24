"""walker: a long-lived value written back from a read taken before an ``await``.

The dict rule (:mod:`threetears.enforcement.dict_state_detection`) moved shared state to one shape: its owner replaces
an immutable value whole. That shape has one failure mode. An async function reads the long-lived value into a local
(or straight into an expression), awaits, and then assigns the same target a value computed from the stale read:
another coroutine's update made during the await is lost. Reports had exactly this (refresh_service, fixed
2026-11-23): ``state.loads = (*state.loads, await self._after_load(...))`` reads ``state.loads`` before the await, so
two loads ending together each rebuilt from the same tuple and one was lost.

A finding is, in one ``async def`` (nested functions are walked on their own):

- (a) a read of a target: an attribute chain rooted at a name (``self.x``, ``cls.x``, ``state.loads``,
  ``_HOLDER.value``), or a name the function declares ``global`` / ``nonlocal``;
- (b) an ``await`` (or an ``async for`` step, or an ``async with`` entry or exit) after that read, on the same path;
- (c) after that await, an assignment to the same target (``=``, annotated, or augmented) of a value derived from the
  pre-await read: the read itself, a local bound from it, or anything computed from either.

Fine: re-reading the target after the await (a value built on a fresh read of the target, ``x.v = without(x.v, key)``
with ``key`` picked before the await included: what is kept is the fresh value), and a read and a write with no await
between them. A comprehension's filter picks items and carries no value.

The walk follows evaluation order inside an expression (``x.v = (*x.v, await f())`` reads ``x.v`` before the await;
``x.v = (await f(), *x.v)`` reads it after), follows each path through ``if`` / ``try`` / loops (a path that returns
before the write carries nothing to it; a loop body is walked twice, so a read in one turn reaching a write in the next
counts), and forgets what a local held when the local, or the root of a chain read through it, is bound again.

What an awaited call hands back derives from its arguments only when the call computes from them: a function of
the module (or imported), a method of the same object (``self.m``), or a hand-off to a thread or a worker
(``asyncio.to_thread``, ``run_in_executor``, ``*_in_worker``, ...); so ``self.x = await asyncio.to_thread(build,
self.x)`` is a finding. A method of another object answers for that object (a store's compare-and-swap revision, a
page's next cursor) and carries nothing. A write the path has just checked against the target, with no await since
(``if self.x is expected: self.x = ...``, or ``if self.x is not expected: return`` before it), is a compare-and-set
and fine.

What it does not follow, so it never guesses: a value a called function reads for itself (``self._build()`` reading
``self.x``), a closure's reads, and aliases of the root (``s = self`` then ``s.x``). A condition
alone (``if self.x is None:``) carries no value, so the usual lazy fill is no finding.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from threetears.enforcement.common import Violation, iter_python_files, parse_python_file

__all__ = ["find_await_race_violations"]

_CATEGORY = "await_race.stale_write_back"


@dataclass(frozen=True)
class _Dep:
    """one read a value derives from.

    :ivar chain: the target read (``self.x``, ``state.loads``, a declared global's name)
    :ivar read_line: where it was read
    :ivar await_line: the first await after the read on this path, or None while none has run
    """

    chain: str
    read_line: int
    await_line: int | None


_Deps = frozenset[_Dep]
_Env = dict[str, _Deps]


def _chain(node: ast.expr) -> str | None:
    """``a.b.c`` for an attribute chain rooted at a name; None for anything else.

    :param node: the expression
    :ptype node: ast.expr
    :return: the dotted chain
    :rtype: str | None
    """
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name) or not parts:
        return None
    return ".".join([node.id, *reversed(parts)])


def _crossed(deps: _Deps, line: int) -> _Deps:
    """``deps`` with an await at ``line`` behind every read not yet behind one.

    :param deps: the reads
    :ptype deps: _Deps
    :param line: the await's line
    :ptype line: int
    :return: the reads, each now behind an await
    :rtype: _Deps
    """
    return frozenset(dep if dep.await_line is not None else _Dep(dep.chain, dep.read_line, line) for dep in deps)


def _merge(*envs: _Env | None) -> _Env | None:
    """the state where paths join: what any of them may hold (None for no path at all).

    :param envs: each path's state, None where the path ended
    :ptype envs: _Env | None
    :return: the joined state
    :rtype: _Env | None
    """
    live = [env for env in envs if env is not None]
    if not live:
        return None
    merged: _Env = {}
    for env in live:
        for name, deps in env.items():
            merged[name] = merged.get(name, frozenset()) | deps
    return merged


class _Flow:
    """the walk of one ``async def``: the reads each local holds, path by path, and the stale write-backs found."""

    def __init__(self, function: ast.AsyncFunctionDef, module_path: Path) -> None:
        self._module_path = module_path
        self._function = function
        self._declared = frozenset(
            name for node in _own_nodes(function) if isinstance(node, (ast.Global, ast.Nonlocal)) for name in node.names
        )
        self._env: _Env | None = {}
        #: the reads gathered by the expressions being evaluated now (an await inside one crosses them too)
        self._pending: list[set[_Dep]] = []
        #: the targets this path just checked are still what it expects, with no await since (a compare-and-set)
        self._guards: frozenset[str] = frozenset()
        self._breaks: list[list[_Env | None]] = []
        self._continues: list[list[_Env | None]] = []
        self.found: list[Violation] = []
        self._reported: set[tuple[int, str]] = set()

    # ---- expressions -------------------------------------------------------------------------------------------

    def _await(self, line: int) -> None:
        self._guards = frozenset()
        if self._env is not None:
            self._env = {name: _crossed(deps, line) for name, deps in self._env.items()}
        for gathered in self._pending:
            crossed = _crossed(frozenset(gathered), line)
            gathered.clear()
            gathered.update(crossed)

    def _value(self, node: ast.expr | None) -> _Deps:
        """the reads ``node``'s value derives from, evaluating it in order (its awaits cross what came before).

        :param node: the expression
        :ptype node: ast.expr | None
        :return: the reads
        :rtype: _Deps
        """
        gathered: set[_Dep] = set()
        if node is None:
            return frozenset()
        self._pending.append(gathered)
        try:
            self._visit(node, gathered)
        finally:
            self._pending.pop()
        return frozenset(gathered)

    def _visit(self, node: ast.AST, into: set[_Dep]) -> None:
        if isinstance(node, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return
        if isinstance(node, ast.Await):
            if isinstance(node.value, ast.Call) and _computes_from_arguments(node.value):
                self._visit(node.value, into)
            else:
                # another object's answer (a store's new revision, a page's next cursor), not a value built here
                self._value(node.value)
            self._await(node.lineno)
            return
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                if node.id in self._declared:
                    # a declared global or nonlocal is the target itself: reading it reads its value now
                    into.add(_Dep(node.id, node.lineno, None))
                elif self._env is not None:
                    into.update(self._env.get(node.id, frozenset()))
            return
        if isinstance(node, ast.Attribute):
            self._visit(node.value, into)
            chain = _chain(node)
            if chain is not None and isinstance(node.ctx, ast.Load):
                into.add(_Dep(chain, node.lineno, None))
            return
        if isinstance(node, ast.NamedExpr):
            deps = self._value(node.value)
            into.update(deps)
            self._bind_name(node.target.id, deps, node.lineno)
            return
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if key is not None:
                    self._visit(key, into)
                self._visit(value, into)
            return
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            self._comprehension(node, into)
            return
        for child in ast.iter_child_nodes(node):
            self._visit(child, into)

    def _comprehension(
        self, node: ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp, into: set[_Dep]
    ) -> None:
        saved = dict(self._env) if self._env is not None else None
        for generator in node.generators:
            deps = self._value(generator.iter)
            into.update(deps)
            if generator.is_async:
                self._await(generator.iter.lineno)
            self._bind_target(generator.target, deps, generator.iter.lineno, check=False)
            for condition in generator.ifs:
                # a filter picks which items come through; the items still derive from the iterable alone
                self._value(condition)
        if isinstance(node, ast.DictComp):
            self._visit(node.key, into)
            self._visit(node.value, into)
        else:
            self._visit(node.elt, into)
        if saved is not None and self._env is not None:
            # the comprehension's own names leave with it; what an await inside it crossed stays crossed
            names = {name for generator in node.generators for name in _target_names(generator.target)}
            for name in names:
                if name in saved:
                    self._env[name] = saved[name]
                else:
                    self._env.pop(name, None)

    # ---- bindings and writes -----------------------------------------------------------------------------------

    def _bind_name(self, name: str, deps: _Deps, line: int) -> None:
        if name in self._declared:
            self._check(name, deps, line)
        if self._env is None:
            return
        # a chain read through the old binding is about another object now
        prefix = name + "."
        rebound: _Env = {
            other: frozenset(dep for dep in held if not dep.chain.startswith(prefix))
            for other, held in self._env.items()
        }
        rebound[name] = frozenset(dep for dep in deps if not dep.chain.startswith(prefix))
        self._env = rebound

    def _bind_target(self, target: ast.expr, deps: _Deps, line: int, *, check: bool = True) -> None:
        if isinstance(target, ast.Name):
            self._bind_name(target.id, deps, line)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._bind_target(element, deps, line, check=check)
        elif isinstance(target, ast.Starred):
            self._bind_target(target.value, deps, line, check=check)
        elif isinstance(target, ast.Attribute):
            self._value(target.value)
            chain = _chain(target)
            if chain is not None and check:
                self._check(chain, deps, line)
        elif isinstance(target, ast.Subscript):
            self._value(target.value)
            self._value(target.slice)

    def _check(self, chain: str, deps: _Deps, line: int) -> None:
        stale = sorted(
            (dep for dep in deps if dep.chain == chain and dep.await_line is not None),
            key=lambda dep: (dep.read_line, dep.await_line or 0),
        )
        # a read with no await behind it is fresh only when it is another read: the same read reaching here with no
        # await on one path and behind one on another is the stale read still
        stale_reads = {dep.read_line for dep in stale}
        fresh = any(dep.chain == chain and dep.await_line is None and dep.read_line not in stale_reads for dep in deps)
        if not stale or fresh or chain in self._guards or (line, chain) in self._reported:
            return
        self._reported.add((line, chain))
        first = stale[0]
        self.found.append(
            Violation(
                category=_CATEGORY,
                file=self._module_path,
                line=line,
                symbol=f"{self._function.name}:{chain}",
                reason=(
                    f"{chain} is assigned in {self._function.name}() from its value read at line {first.read_line}, "
                    f"before the await at line {first.await_line}: an update made during the await is lost. Read it "
                    f"after the await (or keep the update with its one owner)"
                ),
            )
        )

    # ---- statements --------------------------------------------------------------------------------------------

    def run(self) -> None:
        self._block(self._function.body)

    def _block(self, body: list[ast.stmt]) -> None:
        for stmt in body:
            if self._env is None:
                return
            self._statement(stmt)

    def _branch(self, body: list[ast.stmt], start: _Env | None) -> _Env | None:
        self._env = dict(start) if start is not None else None
        self._block(body)
        return self._env

    def _statement(self, stmt: ast.stmt) -> None:
        if isinstance(stmt, ast.Assign):
            deps = self._value(stmt.value)
            for target in stmt.targets:
                self._bind_target(target, deps, stmt.lineno)
        elif isinstance(stmt, ast.AnnAssign):
            if stmt.value is not None:
                self._bind_target(stmt.target, self._value(stmt.value), stmt.lineno)
        elif isinstance(stmt, ast.AugAssign):
            self._augmented(stmt)
        elif isinstance(stmt, (ast.Return, ast.Raise)):
            for child in ast.iter_child_nodes(stmt):
                if isinstance(child, ast.expr):
                    self._value(child)
            self._env = None
        elif isinstance(stmt, ast.Break):
            if self._breaks:
                self._breaks[-1].append(self._env)
            self._env = None
        elif isinstance(stmt, ast.Continue):
            if self._continues:
                self._continues[-1].append(self._env)
            self._env = None
        elif isinstance(stmt, ast.If):
            self._if(stmt)
        elif isinstance(stmt, ast.While):
            self._loop(stmt, iterate=None)
        elif isinstance(stmt, (ast.For, ast.AsyncFor)):
            self._loop(stmt, iterate=self._value(stmt.iter))
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            for item in stmt.items:
                deps = self._value(item.context_expr)
                if isinstance(stmt, ast.AsyncWith):
                    self._await(stmt.lineno)
                if item.optional_vars is not None:
                    self._bind_target(item.optional_vars, deps, stmt.lineno, check=False)
            self._block(stmt.body)
            if isinstance(stmt, ast.AsyncWith) and self._env is not None:
                self._await(stmt.lineno)
        elif isinstance(stmt, (ast.Try, ast.TryStar)):
            self._try(stmt)
        elif isinstance(stmt, ast.Match):
            self._value(stmt.subject)
            start = self._env
            ends = [self._branch(case.body, start) for case in stmt.cases]
            self._guards = frozenset()
            self._env = _merge(start, *ends)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            self._bind_name(stmt.name, frozenset(), stmt.lineno)
        elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
            for alias in stmt.names:
                self._bind_name(alias.asname or alias.name.split(".")[0], frozenset(), stmt.lineno)
        else:
            for child in ast.iter_child_nodes(stmt):
                if isinstance(child, ast.expr):
                    self._value(child)

    def _if(self, stmt: ast.If) -> None:
        """both branches from one state, joined; a test comparing a target read now (``is`` / ``==``) guards the
        branch where it held, until the next await."""
        test = self._value(stmt.test)
        fresh = frozenset(dep.chain for dep in test if dep.await_line is None)
        held, broken = _compared(stmt.test, fresh)
        start, guards = self._env, self._guards
        self._guards = guards | held
        body = self._branch(stmt.body, start)
        body_guards = self._guards
        self._guards = guards | broken
        orelse = self._branch(stmt.orelse, start)
        orelse_guards = self._guards
        if body is None:
            self._guards = orelse_guards
        elif orelse is None:
            self._guards = body_guards
        else:
            self._guards = body_guards & orelse_guards
        self._env = _merge(body, orelse)

    def _augmented(self, stmt: ast.AugAssign) -> None:
        """``t op= v``: ``t`` is read first, then ``v`` evaluated (its awaits cross that read), then ``t`` stored."""
        target = stmt.target
        gathered: set[_Dep] = set()
        self._pending.append(gathered)
        try:
            if isinstance(target, ast.Name):
                if target.id in self._declared:
                    gathered.add(_Dep(target.id, stmt.lineno, None))
                elif self._env is not None:
                    gathered.update(self._env.get(target.id, frozenset()))
            elif isinstance(target, ast.Attribute):
                self._visit(target.value, gathered)
                chain = _chain(target)
                if chain is not None:
                    gathered.add(_Dep(chain, stmt.lineno, None))
            else:
                self._visit(target, gathered)
            self._visit(stmt.value, gathered)
        finally:
            self._pending.pop()
        self._bind_target(target, frozenset(gathered), stmt.lineno)

    def _loop(self, stmt: ast.While | ast.For | ast.AsyncFor, iterate: _Deps | None) -> None:
        head = self._env
        exits: list[_Env | None] = []
        breaks: list[_Env | None] = []
        for _turn in range(2):
            self._env = dict(head) if head is not None else None
            self._guards = frozenset()
            if isinstance(stmt, ast.While):
                self._value(stmt.test)
            elif isinstance(stmt, ast.AsyncFor):
                self._await(stmt.lineno)
            exits.append(self._env)
            if isinstance(stmt, (ast.For, ast.AsyncFor)) and iterate is not None and self._env is not None:
                self._bind_target(stmt.target, iterate, stmt.lineno, check=False)
            self._breaks.append([])
            self._continues.append([])
            self._block(stmt.body)
            breaks.extend(self._breaks.pop())
            continued = self._continues.pop()
            head = _merge(head, self._env, *continued)
        self._env = _merge(*exits)
        self._block(stmt.orelse)
        self._env = _merge(self._env, *breaks)

    def _try(self, stmt: ast.Try | ast.TryStar) -> None:
        start = self._env
        # a handler can start after any statement of the body: what each of them left may reach it
        seen: list[_Env | None] = [start]
        self._env = dict(start) if start is not None else None
        for child in stmt.body:
            if self._env is None:
                break
            self._statement(child)
            seen.append(self._env)
        after_body = self._env
        handler_entry = _merge(*seen)
        ends: list[_Env | None] = []
        for handler in stmt.handlers:
            self._guards = frozenset()
            ends.append(self._branch(handler.body, handler_entry))
        self._guards = frozenset()
        self._env = after_body
        self._block(stmt.orelse)
        joined = _merge(self._env, *ends)
        if stmt.finalbody:
            # the finally runs on every way out, an exception propagating from the body included
            self._env = _merge(joined, handler_entry)
            self._block(stmt.finalbody)
            if joined is None:
                self._env = None
        else:
            self._env = joined


#: the spellings of a hand-off to a thread or a worker: what it hands back is computed from what it was given
_OFFLOADS: tuple[str, ...] = ("to_thread", "run_in_executor", "run_sync", "in_worker", "offload", "in_process")


def _computes_from_arguments(call: ast.Call) -> bool:
    """whether an awaited call's answer is computed from its arguments: a function of this module (or imported), a
    method of this object, or a hand-off to a thread or a worker. A method of another object (``bucket.update``,
    ``self._bucket.purge``, ``reads.page``) answers for that object: a compare-and-swap's new revision, a page's next
    cursor, which no read here goes stale under.

    :param call: the awaited call
    :ptype call: ast.Call
    :return: whether it is
    :rtype: bool
    """
    func = call.func
    if isinstance(func, ast.Name):
        return True
    if not isinstance(func, ast.Attribute):
        return False
    if isinstance(func.value, ast.Name) and func.value.id in {"self", "cls"}:
        return True
    spelled = _chain(func) or func.attr
    return any(offload in spelled for offload in _OFFLOADS)


def _compared(test: ast.expr, fresh: frozenset[str]) -> tuple[frozenset[str], frozenset[str]]:
    """the targets ``test`` checks are what the path expects: as ``(held when true, held when false)``.

    ``target is x`` / ``target == x`` holds when true; ``target is not x`` / ``target != x`` when false; only a
    target read fresh by this test counts, and only a plain comparison or one joined by ``and`` (true) / ``or``
    (false).

    :param test: the ``if`` test
    :ptype test: ast.expr
    :param fresh: the targets the test read with no await since
    :ptype fresh: frozenset[str]
    :return: the targets checked on each side
    :rtype: tuple[frozenset[str], frozenset[str]]
    """
    if isinstance(test, ast.BoolOp):
        parts = [_compared(value, fresh) for value in test.values]
        if isinstance(test.op, ast.And):
            return frozenset().union(*(true for true, _false in parts)), frozenset()
        return frozenset(), frozenset().union(*(false for _true, false in parts))
    if not isinstance(test, ast.Compare) or len(test.ops) != 1:
        return frozenset(), frozenset()
    chains = {_chain(operand) for operand in (test.left, *test.comparators)}
    checked = frozenset(chain for chain in chains if chain is not None and chain in fresh)
    if isinstance(test.ops[0], (ast.Is, ast.Eq)):
        return checked, frozenset()
    if isinstance(test.ops[0], (ast.IsNot, ast.NotEq)):
        return frozenset(), checked
    return frozenset(), frozenset()


def _target_names(target: ast.expr) -> Iterator[str]:
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            yield from _target_names(element)
    elif isinstance(target, ast.Starred):
        yield from _target_names(target.value)


def _own_nodes(function: ast.AST) -> Iterator[ast.AST]:
    """every node of ``function``'s own body, not descending into nested functions, lambdas or classes.

    :param function: the function
    :ptype function: ast.AST
    :return: the nodes
    :rtype: Iterator[ast.AST]
    """
    pending = list(ast.iter_child_nodes(function))
    while pending:
        node = pending.pop()
        yield node
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            pending.extend(ast.iter_child_nodes(node))


def _awaits(function: ast.AsyncFunctionDef) -> bool:
    return any(isinstance(node, (ast.Await, ast.AsyncFor, ast.AsyncWith)) for node in _own_nodes(function)) or any(
        generator.is_async
        for node in _own_nodes(function)
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp))
        for generator in node.generators
    )


def _scan_module(tree: ast.Module, module_path: Path) -> list[Violation]:
    """every stale write-back in one module's async functions.

    :param tree: the module
    :ptype tree: ast.Module
    :param module_path: its path
    :ptype module_path: Path
    :return: the violations, by line
    :rtype: list[Violation]
    """
    found: list[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and _awaits(node):
            flow = _Flow(node, module_path)
            flow.run()
            found.extend(flow.found)
    return sorted(found, key=lambda violation: violation.line)


def _is_under(path: Path, roots: tuple[Path, ...]) -> bool:
    resolved = path.resolve()
    return any(resolved == root.resolve() or resolved.is_relative_to(root.resolve()) for root in roots)


def find_await_race_violations(
    roots: tuple[Path, ...],
    *,
    non_serving: tuple[Path, ...] = (),
) -> list[Violation]:
    """every stale write-back across an ``await`` under ``roots``, less the non-serving trees.

    :param roots: the trees (directories or single modules) walked
    :ptype roots: tuple[Path, ...]
    :param non_serving: trees that are not serving code; nothing under one is reported
    :ptype non_serving: tuple[Path, ...]
    :return: the violations, root-major, then file, then line
    :rtype: list[Violation]
    """
    violations: list[Violation] = []
    for root in roots:
        for module_path in [root] if root.is_file() else list(iter_python_files(root)):
            if _is_under(module_path, non_serving):
                continue
            tree = parse_python_file(module_path)
            if tree is not None:
                violations.extend(_scan_module(tree, module_path))
    return violations
