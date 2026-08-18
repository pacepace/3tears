"""enforcement: the checkpoint saver's scope cannot be skipped or bypassed.

static AST checks, well under a second, over
``src/threetears/langgraph/checkpoint.py`` and
``src/threetears/langgraph/checkpoint_scope.py``. four rules, one design:

1. the name ``thread_id`` never reaches the executor as a bound parameter --
   only ``storage_thread_id``, the value
   :meth:`ThreeTierCheckpointSaver.storage_thread_id` produces, may.
2. ``__init__`` declares ``scope`` with NO default, and no ``customer_id``
   parameter survives anywhere in the signature.
3. ``CheckpointScope.customer_for_config`` -- the resolver a ``from_config``
   scope reads each call's customer through -- contains no DEFAULTED lookup and
   does raise.
4. ``adelete_thread`` keeps ``thread_id`` as its only non-self positional
   parameter, so a live consumer that calls it positionally keeps working.

rule 2 is here because the defect this design corrected was not the mechanism
but the DEFAULT: an optional ``customer_id=None`` meant a caller who said
nothing addressed every customer's keyspace, so tenancy was a convention. a
default re-added later -- ``scope: CheckpointScope = _SOME_FALLBACK``, or a
``customer_id`` parameter restored "for compatibility" -- would restore exactly
that, and would do so without failing any behavioural test, because every such
test constructs a saver by passing what it wants.

rule 3 is rule 2 one level down, for the third scope. a ``from_config`` saver
serves many customers from one process and resolves each call's customer out of
``config["configurable"][key]``; its entire safety is that a missing, ``None``,
or non-``UUID`` value RAISES rather than degrading to the un-tenanted keyspace.
a two-argument ``.get(key, something)`` is precisely how that degradation gets
written -- one character of diff, no failing behavioural test unless someone
thought to write the missing-key case -- so the shape is refused structurally as
well as tested behaviourally.

rule 4 is a compatibility guarantee rather than a safety one. scriob's
delete-session route calls ``adelete_thread(str(session_id))`` positionally, so
the per-call customer that ``from_config`` needs had to arrive as a keyword-only
addition. a later change that made it positional would break that call site with
a wrong-argument bug rather than a type error.

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

_PACKAGE_SRC = Path(__file__).resolve().parent.parent.parent / "src" / "threetears" / "langgraph"
_CHECKPOINT_MODULE = _PACKAGE_SRC / "checkpoint.py"
_SCOPE_MODULE = _PACKAGE_SRC / "checkpoint_scope.py"

_RAW_NAME = "thread_id"
_PARAMS_NAME = "params"
_SCOPING_CALL = "storage_thread_id"
_SAVER_CLASS = "ThreeTierCheckpointSaver"
_SCOPE_CLASS = "CheckpointScope"
_SCOPE_PARAM = "scope"
_REMOVED_PARAM = "customer_id"
_RESOLVER_METHOD = "customer_for_config"
_PER_THREAD_PURGE = "adelete_thread"

_AnyFunctionDef = ast.FunctionDef | ast.AsyncFunctionDef


def _method_in_class(tree: ast.Module, class_name: str, method_name: str) -> _AnyFunctionDef:
    """locate one method of one class in a parsed module.

    :param tree: a parsed module
    :ptype tree: ast.Module
    :param class_name: the class to search inside
    :ptype class_name: str
    :param method_name: the method to find
    :ptype method_name: str
    :return: the method's function node, sync or async
    :rtype: ast.FunctionDef | ast.AsyncFunctionDef
    :raises AssertionError: when the class or the method is missing, since a
        rule that silently found nothing would read as a pass
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for member in node.body:
                if isinstance(member, ast.FunctionDef | ast.AsyncFunctionDef) and member.name == method_name:
                    return member
    raise AssertionError(f"{class_name}.{method_name} not found")


def _saver_init(tree: ast.Module) -> ast.FunctionDef:
    """locate ``ThreeTierCheckpointSaver.__init__`` in a parsed module.

    :param tree: the parsed checkpoint module
    :ptype tree: ast.Module
    :return: the constructor's function node
    :rtype: ast.FunctionDef
    :raises AssertionError: when the class or its constructor is missing, or the
        constructor is somehow async, since a rule that silently found nothing
        would read as a pass
    """
    node = _method_in_class(tree, _SAVER_CLASS, "__init__")
    assert isinstance(node, ast.FunctionDef), f"{_SAVER_CLASS}.__init__ is not a plain function"
    return node


def _defaulted_lookup_lines(node: ast.AST) -> list[int]:
    """line numbers of every two-argument ``.get(...)`` call under *node*.

    a one-argument ``.get(key)`` yields ``None`` on a miss, which the resolver
    then has to test and refuse; a two-argument one substitutes a value and
    carries on, which is the exact shape of a silent degradation.

    :param node: any AST node
    :ptype node: ast.AST
    :return: sorted line numbers, empty when nothing defaults
    :rtype: list[int]
    """
    found: set[int] = set()
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "get"
            and len(child.args) >= 2
        ):
            found.add(child.lineno)
    return sorted(found)


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


class TestConfigResolutionCannotSilentlyDefault:
    """a ``from_config`` saver must fail CLOSED, and that is one edit from false."""

    def test_the_resolver_uses_no_defaulted_lookup(self) -> None:
        """``.get(key, fallback)`` is how "fail closed" quietly becomes "fail open".

        a ``from_config`` scope reads each call's customer out of
        ``config["configurable"][key]``. substituting a fallback for a missing
        key turns a host that forgot to pass one into a host addressing the
        un-tenanted keyspace -- believing, throughout, that it was isolated.
        that is worse than no tenancy at all, so the shape is refused here as
        well as in the behavioural suite.

        :return: nothing
        :rtype: None
        """
        resolver = _method_in_class(
            ast.parse(_SCOPE_MODULE.read_text(encoding="utf-8")),
            _SCOPE_CLASS,
            _RESOLVER_METHOD,
        )

        offending = _defaulted_lookup_lines(resolver)

        assert not offending, (
            f"{_SCOPE_CLASS}.{_RESOLVER_METHOD} uses a defaulted lookup at line(s) "
            f"{', '.join(map(str, offending))}.\n\n"
            "A missing, None, or non-UUID customer must RAISE. Substituting a default degrades a "
            "multi-tenant saver to the un-tenanted keyspace on one forgotten dict key, which is the "
            "single failure this scope exists to prevent."
        )

    def test_the_resolver_raises(self) -> None:
        """a resolver that cannot refuse has nothing to fail closed with.

        :return: nothing
        :rtype: None
        """
        resolver = _method_in_class(
            ast.parse(_SCOPE_MODULE.read_text(encoding="utf-8")),
            _SCOPE_CLASS,
            _RESOLVER_METHOD,
        )

        assert any(isinstance(node, ast.Raise) for node in ast.walk(resolver)), (
            f"{_SCOPE_CLASS}.{_RESOLVER_METHOD} contains no raise. A resolver that cannot refuse a "
            "missing customer returns one anyway, and every key it builds addresses the wrong keyspace."
        )

    def test_the_defaulted_lookup_walker_flags_a_synthetic_violation(self) -> None:
        """negative control: a rule that sees nothing must not read as clean.

        :return: nothing
        :rtype: None
        """
        source = "def resolve(self, configurable):\n    return configurable.get('customer_id', None)\n"

        assert _defaulted_lookup_lines(ast.parse(source)) == [2]

    def test_the_defaulted_lookup_walker_accepts_an_undefaulted_lookup(self) -> None:
        """the correct shape must not read as the violation.

        :return: nothing
        :rtype: None
        """
        source = "def resolve(self, config):\n    return config.get('configurable')\n"

        assert _defaulted_lookup_lines(ast.parse(source)) == []


class TestThePerThreadPurgeStaysPositionallyCompatible:
    """scriob calls ``adelete_thread(str(session_id))``; that has to keep working."""

    def test_thread_id_is_the_only_positional_parameter(self) -> None:
        """the per-call customer had to arrive keyword-only, and must stay so.

        checked structurally because a positional second parameter would not
        fail any test in THIS repo -- it would fail in a consumer, at runtime, by
        binding a session id to the wrong argument.

        :return: nothing
        :rtype: None
        """
        purge = _method_in_class(
            ast.parse(_CHECKPOINT_MODULE.read_text(encoding="utf-8")),
            _SAVER_CLASS,
            _PER_THREAD_PURGE,
        )

        positional = [argument.arg for argument in [*purge.args.posonlyargs, *purge.args.args]]

        assert positional == ["self", _RAW_NAME], (
            f"{_SAVER_CLASS}.{_PER_THREAD_PURGE} declares positional parameters {positional}. "
            "A live consumer calls it as adelete_thread(str(session_id)), so anything beyond "
            "`thread_id` must be keyword-only."
        )

    def test_the_locator_refuses_to_pass_on_a_missing_method(self) -> None:
        """negative control: a rule that finds nothing must not read as clean.

        :return: nothing
        :rtype: None
        """
        with pytest.raises(AssertionError, match="not found"):
            _method_in_class(ast.parse("class Unrelated:\n    pass\n"), _SAVER_CLASS, _PER_THREAD_PURGE)
