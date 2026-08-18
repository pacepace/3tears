"""enforcement: the checkpoint saver's scope cannot be skipped or bypassed.

static AST checks, well under a second, over
``src/threetears/langgraph/checkpoint.py``. two rules, one design:

1. the name ``thread_id`` never reaches the executor as a bound parameter --
   only ``storage_thread_id``, the value
   :meth:`ThreeTierCheckpointSaver.storage_thread_id` produces, may.
2. ``__init__`` declares ``scope`` with NO default, and no ``customer_id``
   parameter survives anywhere in the signature.

rule 2 is here because the defect this design corrected was not the mechanism
but the DEFAULT: an optional ``customer_id=None`` meant a caller who said
nothing addressed every customer's keyspace, so tenancy was a convention. a
default re-added later -- ``scope: CheckpointScope = _SOME_FALLBACK``, or a
``customer_id`` parameter restored "for compatibility" -- would restore exactly
that, and would do so without failing any behavioural test, because every such
test constructs a saver by passing what it wants.

rationale: the customer dimension on ``checkpoints`` / ``checkpoint_writes``
lives INSIDE the ``thread_id`` value rather than in a separate column, because
the primary key is ``(thread_id, checkpoint_ns, checkpoint_id)`` and a customer
column would have to join that key to make a row unique through its customer.
that design has exactly one failure mode: a statement added later that binds the
raw, caller-supplied ``thread_id``. such a statement reads and writes the
un-scoped keyspace, so a saver bound to one customer would address another's
rows -- silently, and only for the new statement, which is precisely the shape
no runtime test covers because the test would have to know the statement exists.

what this can and cannot see: it flags the bare name ``thread_id`` appearing in
the arguments of a ``self._exec.<method>(...)`` call, and in any list literal
assigned to ``params`` (the shape ``alist`` uses to build a variadic argument
list). it prunes at a ``.storage_thread_id(...)`` call, since the raw name
inside one is the argument being converted. it does NOT chase a raw thread id
laundered through an intermediate variable under another name. that limit is
recorded rather than papered over: this makes the common case impossible, not
every case.

a negative control below drives the walker over a synthetic violation, so a
regression that made the rule blind would fail rather than pass quietly -- the
failure mode every name-sweeping check is prone to.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_CHECKPOINT_MODULE = (
    Path(__file__).resolve().parent.parent.parent / "src" / "threetears" / "langgraph" / "checkpoint.py"
)

_RAW_NAME = "thread_id"
_PARAMS_NAME = "params"
_SCOPING_CALL = "storage_thread_id"
_SAVER_CLASS = "ThreeTierCheckpointSaver"
_SCOPE_PARAM = "scope"
_REMOVED_PARAM = "customer_id"


def _saver_init(tree: ast.Module) -> ast.FunctionDef:
    """locate ``ThreeTierCheckpointSaver.__init__`` in a parsed module.

    :param tree: the parsed checkpoint module
    :ptype tree: ast.Module
    :return: the constructor's function node
    :rtype: ast.FunctionDef
    :raises AssertionError: when the class or its constructor is missing, since
        a rule that silently found nothing would read as a pass
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == _SAVER_CLASS:
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and member.name == "__init__":
                    return member
    raise AssertionError(f"{_SAVER_CLASS}.__init__ not found in {_CHECKPOINT_MODULE.name}")


def _is_executor_call(node: ast.AST) -> bool:
    """is this a call on the saver's executor?

    :param node: any AST node
    :ptype node: ast.AST
    :return: True when the node is ``self._exec.<something>(...)``
    :rtype: bool
    """
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not isinstance(func, ast.Attribute):
        return False
    inner = func.value
    return isinstance(inner, ast.Attribute) and inner.attr == "_exec" and isinstance(inner.value, ast.Name)


def _is_scoping_call(node: ast.AST) -> bool:
    """is this the call that launders a raw thread id into a scoped one?

    :param node: any AST node
    :ptype node: ast.AST
    :return: True when the node is ``<something>.storage_thread_id(...)``
    :rtype: bool
    """
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == _SCOPING_CALL


def _mentions_raw_thread_id(node: ast.AST) -> bool:
    """does this expression reach a statement with an UNSCOPED ``thread_id``?

    walks the expression but prunes at a :func:`_is_scoping_call` subtree: the
    raw name inside ``self.storage_thread_id(thread_id)`` is the argument being
    converted, which is the correct shape rather than a violation.

    :param node: any AST node
    :ptype node: ast.AST
    :return: True when a ``Name`` load of ``thread_id`` survives that pruning
    :rtype: bool
    """
    if _is_scoping_call(node):
        return False
    if isinstance(node, ast.Name) and node.id == _RAW_NAME:
        return True
    return any(_mentions_raw_thread_id(child) for child in ast.iter_child_nodes(node))


def _offending_lines(tree: ast.Module) -> list[int]:
    """line numbers of every place a raw thread id reaches a statement.

    :param tree: the parsed checkpoint module
    :ptype tree: ast.Module
    :return: sorted line numbers, empty when the module is clean
    :rtype: list[int]
    """
    found: set[int] = set()
    for node in ast.walk(tree):
        if _is_executor_call(node):
            assert isinstance(node, ast.Call)
            for argument in [*node.args, *(kw.value for kw in node.keywords)]:
                if _mentions_raw_thread_id(argument):
                    found.add(argument.lineno)
        if isinstance(node, ast.AnnAssign | ast.Assign):
            targets = [node.target] if isinstance(node, ast.AnnAssign) else node.targets
            is_params = any(isinstance(target, ast.Name) and target.id == _PARAMS_NAME for target in targets)
            if is_params and node.value is not None and _mentions_raw_thread_id(node.value):
                found.add(node.value.lineno)
    return sorted(found)


class TestCheckpointSqlBindsTheScopedThreadId:
    """every checkpoint statement addresses the customer-scoped keyspace."""

    def test_no_statement_binds_the_raw_thread_id(self) -> None:
        """a raw bound thread id is a cross-customer read or write waiting to happen.

        :return: nothing
        :rtype: None
        """
        tree = ast.parse(_CHECKPOINT_MODULE.read_text(encoding="utf-8"))

        offending = _offending_lines(tree)

        assert not offending, (
            f"{_CHECKPOINT_MODULE.name} binds the caller's raw `thread_id` to a statement at "
            f"line(s) {', '.join(map(str, offending))}.\n\n"
            "The customer dimension lives inside the stored thread id, so a statement must bind "
            "`self.storage_thread_id(thread_id)` -- binding the raw value addresses the un-scoped "
            "keyspace and lets a saver bound to one customer reach another customer's rows."
        )

    def test_the_walker_flags_a_synthetic_violation(self) -> None:
        """negative control: a rule that sees nothing must not read as clean.

        :return: nothing
        :rtype: None
        """
        source = (
            "async def bad(self, thread_id):\n"
            "    await self._exec.execute('DELETE FROM checkpoints WHERE thread_id = $1', thread_id)\n"
        )

        assert _offending_lines(ast.parse(source)) == [2]

    def test_the_walker_accepts_the_scoped_form(self) -> None:
        """the conversion call is the fix, so it must not read as the violation.

        :return: nothing
        :rtype: None
        """
        source = (
            "async def good(self, thread_id):\n"
            "    await self._exec.execute(\n"
            "        'DELETE FROM checkpoints WHERE thread_id = $1', self.storage_thread_id(thread_id)\n"
            "    )\n"
        )

        assert _offending_lines(ast.parse(source)) == []

    def test_the_walker_flags_a_raw_id_in_a_params_list(self) -> None:
        """``alist`` builds its bound parameters in a list, so the list is checked too.

        :return: nothing
        :rtype: None
        """
        source = "def bad(self, thread_id, checkpoint_ns):\n    params = [thread_id, checkpoint_ns]\n"

        assert _offending_lines(ast.parse(source)) == [2]


class TestTheScopeDecisionHasNoDefault:
    """the corrected defect was the default, so the absence of one is enforced."""

    def test_scope_is_keyword_only_with_no_default(self) -> None:
        """a default on ``scope`` would restore "say nothing, see everything".

        checked structurally rather than by constructing a saver, because a
        default that is itself an unscoped scope would construct perfectly
        happily and pass every behavioural test in the suite.

        :return: nothing
        :rtype: None
        """
        init = _saver_init(ast.parse(_CHECKPOINT_MODULE.read_text(encoding="utf-8")))

        names = [argument.arg for argument in init.args.kwonlyargs]
        assert _SCOPE_PARAM in names, (
            f"{_SAVER_CLASS}.__init__ must take `{_SCOPE_PARAM}` as a keyword-only parameter; "
            "a second positional next to `executor` is a transposition no type checker would catch."
        )
        default = init.args.kw_defaults[names.index(_SCOPE_PARAM)]
        assert default is None, (
            f"{_SAVER_CLASS}.__init__ gives `{_SCOPE_PARAM}` a default. The default is the whole defect: "
            "a caller who says nothing about tenancy would get a saver again, and the scope would be a "
            "convention rather than a gate."
        )

    def test_the_removed_customer_id_parameter_stays_removed(self) -> None:
        """re-adding it "for compatibility" re-opens the same door.

        :return: nothing
        :rtype: None
        """
        init = _saver_init(ast.parse(_CHECKPOINT_MODULE.read_text(encoding="utf-8")))

        declared = [argument.arg for argument in [*init.args.posonlyargs, *init.args.args, *init.args.kwonlyargs]]

        assert _REMOVED_PARAM not in declared, (
            f"{_SAVER_CLASS}.__init__ declares `{_REMOVED_PARAM}` again. Tenancy is expressed by "
            "`scope: CheckpointScope`; a parallel parameter reintroduces the optional, defaulted form "
            "this replaced."
        )

    def test_the_locator_refuses_to_pass_on_a_missing_constructor(self) -> None:
        """negative control: a rule that finds nothing must not read as clean.

        :return: nothing
        :rtype: None
        """
        with pytest.raises(AssertionError, match="not found"):
            _saver_init(ast.parse("class Unrelated:\n    pass\n"))
