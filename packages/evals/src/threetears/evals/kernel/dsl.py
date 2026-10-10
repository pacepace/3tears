"""The goal-state DSL's evaluator: an expression read against a cell's end state, call ledger and case.

The language itself — its surface, its three-valued *Missing values* rule, its static extraction and
its safety allowlist — is documented and parsed in :mod:`threetears.evals.schema.goal_grammar`. This
module evaluates a parsed expression (:func:`evaluate`, :func:`evaluate_with_detail`), and holds the
checks that need a host's world or action vocabulary to answer (``undefined_*``,
:func:`world_prose_matches`, :func:`call_parameter_matches`).
"""

from __future__ import annotations

import ast
import copy
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from threetears.evals.schema.call_ledger import CallLedger, is_pass
from threetears.evals.schema.goal_grammar import (
    FIRE_PREDICATES,
    DSLError,
    TextMatch,
    TextOperand,
    extract_text_matches,
    parse,
    parse_tool_action,
    referenced_actions,
    referenced_fires,
    text_operand,
)
from threetears.evals.schema.prose import schema_is_prose, schema_nodes_at

if TYPE_CHECKING:
    from threetears.evals.kernel.host.world import WorldRegistry
    from threetears.evals.schema.world_events import Firings


class _Missing:
    """Sentinel for unresolved paths.

    *Unknown*, not *known-absent*: the evaluator never asks it a question. Every operator in the
    language checks for it first and propagates it three-valued (the module's *Missing values*), so
    neither a comparison nor its negation can be satisfied by a path that resolved to nothing. The
    dunders below answer False only so that Python code holding the sentinel outside the evaluator
    cannot mistake it for a match either.
    """

    _instance: _Missing | None = None

    def __new__(cls) -> _Missing:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<Missing>"

    def __bool__(self) -> bool:
        return False

    # All comparison operators against Missing return False — the missing
    # value is unknown, not zero/empty, so it can't satisfy any predicate.
    def __eq__(self, other: object) -> bool:
        return False

    def __ne__(self, other: object) -> bool:
        return False  # also false: "Missing != x" is unknown

    def __lt__(self, other: object) -> bool:
        return False

    def __le__(self, other: object) -> bool:
        return False

    def __gt__(self, other: object) -> bool:
        return False

    def __ge__(self, other: object) -> bool:
        return False

    def __hash__(self) -> int:
        return 0

    def __contains__(self, _item: object) -> bool:
        return False

    def __len__(self) -> int:
        return 0

    def __iter__(self) -> Iterator[Any]:
        return iter(())


Missing = _Missing()


@dataclass
class _EvalContext:
    """Per-call evaluation context passed through the AST walker."""

    end_state: Mapping[str, Any]  # declared dimension name -> value
    ledger: CallLedger
    variation: dict[str, Any]
    binding: dict[str, Any]  # `it` -> current element inside any()/all()
    world: WorldRegistry | None  # how `state.<dimension>` resolves a name; None declares no world at all
    fired: Firings | None  # what fired, and which firings were armed; None when no world events were recorded
    unresolved: list[str] = field(default_factory=list)  # the paths that resolved to Missing, for the detail


def world_prose_matches(expression: str, registry: WorldRegistry | None) -> tuple[TextMatch, ...]:
    """The text predicates in a goal check that match against model prose in this host's world.

    For an authoring gate: every returned match is a deterministic string check over text a model
    wrote, which the gate refuses. ``contains()`` over a structured array (a list of orders, a
    list of enum values) is not returned — membership stays — and neither is an emptiness test.

    **Opt-in, where :func:`call_parameter_matches` fails closed, and the difference is who wrote the
    value.** Every call parameter is written by the model by construction, so only a schema keyword
    closing it can say it is not prose. World state is mostly not the model's: catalogue titles, ids,
    seeded fixtures and host-computed fields are strings the model never wrote, and refusing every
    unclosed one would refuse checks over data. So the host's world declaration names the fields a
    model writes (the prose marker), and a model-written field declared without it is a defect in
    that declaration, not something this function can infer.

    Args:
        expression: A goal-state expression.
        registry: The host's world vocabulary, or None for a host that declares no world (nothing
            is prose there, so nothing is returned).

    Returns:
        The offending matches, in source order.

    Raises:
        DSLError: The expression does not parse.
    """
    if registry is None:
        return ()
    offending: list[TextMatch] = []
    for match in extract_text_matches(expression):
        if match.call is not None or not match.operand:
            continue
        head = match.operand[: match.operand.index("[]")] if "[]" in match.operand else match.operand
        dimension = registry.resolve_path(".".join(head))
        if dimension is None:
            # Not this gate's to report: the vocabulary gate refuses a path naming no dimension.
            continue
        declared = registry.get(dimension)
        if declared is None:
            continue
        below = match.operand[len(dimension.split(".")) :]
        if any(schema_is_prose(node) for node in schema_nodes_at(declared.schema, below)):
            offending.append(match)
    return tuple(offending)


#: Schema keywords that close a string to a declared set of values, so equality against a literal is
#: a check of structure rather than a reading of text.
_CLOSING_KEYWORDS = ("enum", "const", "pattern")


def _is_free_text(schema: Mapping[str, Any]) -> bool:
    """Whether a parameter schema admits free text — a string its schema does not close.

    An array is judged by its items — every shape they can take, so one free-text branch makes it free text; a
    node with no ``type`` may hold a string, so it counts.
    """
    if schema.get("type") == "array" and (elements := schema_nodes_at(schema, ("[]",))):
        return any(_is_free_text(element) for element in elements)
    if any(keyword in schema for keyword in _CLOSING_KEYWORDS):
        return False
    declared = schema.get("type")
    if declared is None:
        return True
    return "string" in (declared if isinstance(declared, list) else [declared])


def call_parameter_matches(
    expression: str, reader: Callable[[str, str], Mapping[str, Any] | None] | None
) -> tuple[tuple[TextMatch, str], ...]:
    """The text predicates over recorded call parameters that an authoring gate refuses, and why.

    **Fail closed**, unlike :func:`world_prose_matches`: a call parameter is text the model wrote
    until the action's parameter schema — the one the model is shown — closes it (``enum``,
    ``const`` or ``pattern``). So a host with no reader, an action it does not describe and a
    parameter the schema does not name are each refused, since none can say the value is not prose.

    Args:
        expression: A goal-state expression.
        reader: The host's ``(tool, action) -> parameter schema`` reader, or None.

    Returns:
        ``(match, reason)`` per refused predicate, in source order.

    Raises:
        DSLError: The expression does not parse.
    """
    refused: list[tuple[TextMatch, str]] = []
    for match in extract_text_matches(expression):
        if match.call is None:
            continue
        tool, action = parse_tool_action(match.call)
        schema = reader(tool, action) if reader is not None else None
        if schema is None:
            refused.append((match, f"this host describes no parameters for {match.call}"))
            continue
        # Every shape the position can take, through any ``anyOf``: it is closed only when every one of them
        # closes it, so a single free-text branch refuses the predicate, and a position no shape describes does.
        nodes = schema_nodes_at(schema, match.operand)
        if not nodes:
            refused.append((match, f"{match.call} declares no parameter {'.'.join(match.operand)}"))
        elif any(_is_free_text(node) for node in nodes):
            refused.append((match, f"{match.call}'s {'.'.join(match.operand)} is free text"))
    return tuple(refused)


def undefined_action(tool: str, action: str, actions: Callable[[str], frozenset[str] | None] | None) -> str | None:
    """Why ``tool.action`` names nothing its host defines, or None when it does or the host cannot say.

    The one answer both gates that hold a recorded call's name to the host give: a goal check's
    ``calls()`` references (:func:`undefined_call_references`) and a control's recorded calls
    (``threetears.evals.run.check_controls``). Shared because a check and its control are proof of each
    other only if both are held to the same rule: a typo admitted by one and refused by the other
    still yields a check whose control agrees with it. The reader's None is "cannot list this
    tool's actions" and leaves it unchecked; an empty set is "this tool offers no action", which is
    what a host answers for a tool it does not have.

    Args:
        tool: The tool a call names.
        action: The action a call names.
        actions: The host's ``tool -> action names`` reader, or None.

    Returns:
        One sentence naming the undefined action, or None.
    """
    known = actions(tool) if actions is not None else None
    if known is None or action in known:
        return None
    if not known:
        return f"{tool} has no action {action!r} (this host lists no actions for {tool!r} — is it a tool the host has?)"
    return f"{tool} has no action {action!r} (its actions: {', '.join(sorted(known))})"


def undefined_fire_references(expression: str, world: WorldRegistry | None) -> tuple[str, ...]:
    """Every ``fired()`` name in a goal check that is not a triggered dimension of the host's world.

    A misspelled or untriggered name does not fail loudly at run time: nothing ever fires under it, so
    ``fired()`` is False on every trial and reads as the candidate's failure. A dimension that is
    declared but not triggered is refused too — it arrives at t=0 and has no condition to fire.

    Args:
        expression: A goal-state expression.
        world: The host's world registry, or None for a host that declares no world.

    Returns:
        One reason per name that is not a triggered dimension, in source order.

    Raises:
        DSLError: The expression does not parse.
    """
    return tuple(
        reason
        for name in referenced_fires(expression)
        if (reason := undefined_fired_dimension(name, world)) is not None
    )


def undefined_fired_dimension(name: str, world: WorldRegistry | None) -> str | None:
    """Why ``name`` is not a triggered dimension of the host's world, or None when it is.

    The one answer both gates that hold a fired name to the host give: a goal check's ``fired()``
    references (:func:`undefined_fire_references`) and a control's stated ``fired`` set
    (``threetears.evals.run.check_controls``) — shared for :func:`undefined_action`'s reason, so a check
    and its control are held to one rule.

    Args:
        name: A dimension named as having fired.
        world: The host's world registry, or None for a host that declares no world.

    Returns:
        One sentence naming the defect, or None.
    """
    if world is None:
        return f"fired({name!r}) reads world events, and this host declares no world"
    declared = world.get(name)
    if declared is None:
        return f"fired({name!r}) names no dimension this host's world declares"
    if declared.when == "initial":
        return f"fired({name!r}) names a dimension set at t=0 — only a triggered dimension fires"
    return None


def undefined_call_references(
    expression: str,
    actions: Callable[[str], frozenset[str] | None] | None,
    parameters: Callable[[str, str], Mapping[str, Any] | None] | None,
) -> tuple[str, ...]:
    """Every action, or ``calls()`` parameter, a goal check names that its host says does not exist.

    A misspelled name does not fail loudly at run time: ``calls("inventory.place_ordr")`` is an empty
    list and ``it.note_txt`` resolves to nothing, so the check scores False on every trial and
    reads as the candidate's failure. So each ``tool.action`` given to a call builtin is looked up
    in the host's action list, and each parameter read through ``calls()`` — compared or not — in
    that action's parameter schema.

    **Only what the host can describe is checked.** A tool whose actions the host cannot list
    (``actions`` returns None for it) and an action with no parameter schema pass unverified: no
    answer is not the answer "absent". A tool the host does not have is not that case — the host
    answers it with an empty set, and every action named on it is refused
    (:func:`undefined_action`). A trailing ``.length`` is the DSL's pseudo-attribute, not a
    parameter.

    Args:
        expression: A goal-state expression.
        actions: The host's ``tool -> action names`` reader, or None.
        parameters: The host's ``(tool, action) -> parameter schema`` reader, or None.

    Returns:
        One reason per undefined reference, in source order, each named once.

    Raises:
        DSLError: The expression does not parse.
    """
    body = parse(expression).body
    reasons: list[str] = []

    def refuse(reason: str) -> None:
        if reason not in reasons:
            reasons.append(reason)

    for tool, action in referenced_actions(expression):
        if (reason := undefined_action(tool, action, actions)) is not None:
            refuse(reason)

    operands: list[TextOperand] = []
    _collect_call_operands(body, None, operands)
    for spec, segments in operands:
        named = segments[1:] if segments[:1] == ("[]",) else segments
        if named[-1:] == ("length",):
            named = named[:-1]
        if not named:
            continue
        tool, action = parse_tool_action(spec)
        schema = parameters(tool, action) if parameters is not None else None
        if schema is not None and not schema_nodes_at(schema, named):
            refuse(f"{spec} declares no parameter {'.'.join(segment for segment in named if segment != '[]')}")
    return tuple(reasons)


def _collect_call_operands(node: ast.AST, it_binding: TextOperand | None, found: list[TextOperand]) -> None:
    """Record every path read through ``calls()`` under ``node``; ``it_binding`` is what ``it`` addresses.

    Args:
        node: The node to walk.
        it_binding: The operand ``it`` stands for inside the enclosing ``any()``/``all()``.
        found: Appended to in place.
    """
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("any", "all"):
        if node.args and isinstance(node.args[0], ast.GeneratorExp):
            generator = node.args[0]
            _collect_call_operands(generator.generators[0].iter, it_binding, found)
            iterable = text_operand(generator.generators[0].iter, it_binding)
            element = (iterable[0], (*iterable[1], "[]")) if iterable is not None else None
            _collect_call_operands(generator.elt, element, found)
            return
    if isinstance(node, (ast.Attribute, ast.Subscript)):
        operand = text_operand(node, it_binding)
        if operand is not None and operand[0] is not None:
            found.append(operand)
            if isinstance(node, ast.Subscript):
                _collect_call_operands(node.slice, it_binding, found)
            return
    for child in ast.iter_child_nodes(node):
        _collect_call_operands(child, it_binding, found)


def evaluate(
    expression: str,
    *,
    end_state: Mapping[str, Any],
    ledger: CallLedger,
    world: WorldRegistry | None,
    variation: dict[str, Any] | None = None,
    fired: Firings | None = None,
) -> bool:
    """Evaluate a goal-state expression and return its boolean result.

    Args:
        expression: DSL expression text.
        end_state: The world to read ``state.<dimension>`` against, keyed by declared dimension name.
        ledger: The calls the candidate made, read by the call predicates.
        world: The host's world registry, through which ``state.<path>`` resolves to a declared
            dimension — the resolution the authoring gate refuses an unknown path by. None for a
            host that declares no world, where any ``state`` path raises.
        variation: Variation parameter dict (test case's variation_params).
            Defaults to empty when omitted.
        fired: What fired during the cell — read by ``fired()`` — and which firings were of the event the
            seed armed — read by ``fired_armed()``. None when no world events were recorded for this
            evaluation, where both raise rather than answering False: an evaluation with no record cannot
            say nothing fired. When it cannot know which firings were armed (``armed_known=False`` — a
            witnessed cell, built by :meth:`Firings.of`), ``fired_armed()`` is :data:`Missing`.

    Returns:
        ``True`` if the expression's value is truthy after coercion; ``False`` otherwise, including
        when it is :data:`Missing` — not established, which is never a pass.

    Raises:
        DSLError: For malformed expressions, for a ``state`` path on a host that declares no world,
            and for ``fired()`` or ``fired_armed()`` with no world events recorded. A path that names no
            declared dimension, or that does not resolve inside one, is :data:`Missing` instead, and
            propagates three-valued (the module's *Missing values*).
    """
    tree = parse(expression)
    ctx = _EvalContext(
        end_state=end_state, ledger=ledger, variation=variation or {}, binding={}, world=world, fired=fired
    )
    result = _eval_node(tree.body, ctx)
    if result is Missing:
        return False
    return bool(result)


def evaluate_with_detail(
    expression: str,
    *,
    end_state: Mapping[str, Any],
    ledger: CallLedger,
    world: WorldRegistry | None,
    variation: dict[str, Any] | None = None,
    fired: Firings | None = None,
) -> tuple[bool, str]:
    """Evaluate and return ``(result, detail)`` for trace inspection.

    The detail string captures the resolved value(s) so the goal-state
    outcomes panel can show *why* a check passed or failed. Examples::

        (True, "True")
        (False, "not established: state.support.messages resolved to nothing")

    Args:
        expression: DSL expression text.
        end_state: The world to read, keyed by declared dimension name, as :func:`evaluate` takes it.
        ledger: The calls the candidate made, as :func:`evaluate` takes it.
        world: The host's world registry, as :func:`evaluate` takes it.
        variation: Variation parameter dict. Defaults to empty when omitted.
        fired: What fired, as :func:`evaluate` takes it.

    Returns:
        The boolean result and the resolved value's ``repr``; for a :data:`Missing` value, ``False``
        and a detail starting ``"not established"`` that names the paths that resolved to nothing.

    Raises:
        DSLError: As :func:`evaluate` raises it.
    """
    tree = parse(expression)
    ctx = _EvalContext(
        end_state=end_state, ledger=ledger, variation=variation or {}, binding={}, world=world, fired=fired
    )
    value = _eval_node(tree.body, ctx)
    if value is Missing:
        return False, not_established_detail(ctx.unresolved)
    return bool(value), repr(value)


#: The prefix of every detail recording a check that rested on a value the world did not hold. A
#: reader tells "not established" from "evaluated and false" by it, since both are ``passed=False``.
NOT_ESTABLISHED = "not established"


def not_established_detail(unresolved: list[str]) -> str:
    """The detail for a check whose value is :data:`Missing`, naming what resolved to nothing.

    Args:
        unresolved: The paths that resolved to :data:`Missing` while the expression was evaluated,
            in the order met; repeats are dropped.

    Returns:
        ``"not established: ..."`` naming each path, or saying a value resolved to nothing when no
        path did (``length()`` of a value that has no length).
    """
    named = list(dict.fromkeys(unresolved))
    if not named:
        return f"{NOT_ESTABLISHED}: a value it reads resolved to nothing"
    return f"{NOT_ESTABLISHED}: {', '.join(named)} resolved to nothing"


def _eval_node(node: ast.AST, ctx: _EvalContext) -> Any:
    """Walk one node of the parsed AST and return its value.

    Path resolution failures produce :data:`Missing`, which every operator propagates three-valued
    (the module's *Missing values*) rather than reading as False.
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return _resolve_root(node.id, ctx)
    if isinstance(node, ast.Attribute):
        return _resolve_attr(node, ctx)
    if isinstance(node, ast.Subscript):
        return _resolve_subscript(node, ctx)
    if isinstance(node, ast.List | ast.Tuple):
        # A literal holding an unknown element is itself unknown: Python's `==`, `in` and `set()`
        # would otherwise read the element as an ordinary value that equals nothing, so
        # `not contains([state.x], ...)` answered True over a world that never held `state.x`.
        elements = [_eval_node(elt, ctx) for elt in node.elts]
        return Missing if any(element is Missing for element in elements) else elements
    if isinstance(node, ast.UnaryOp):
        return _eval_unary(node, ctx)
    if isinstance(node, ast.BoolOp):
        return _eval_boolop(node, ctx)
    if isinstance(node, ast.Compare):
        return _eval_compare(node, ctx)
    if isinstance(node, ast.Call):
        return _eval_call(node, ctx)
    if isinstance(node, ast.GeneratorExp):
        # GeneratorExp on its own is only used inside any()/all() in this DSL.
        # If we get here, surface a useful error rather than silently iterating.
        raise DSLError("Bare generator expressions are not evaluable; use any()/all().")
    raise DSLError(f"Unhandled node type at eval time: {type(node).__name__}")


def _resolve_root(name: str, ctx: _EvalContext) -> Any:
    """Resolve a bare identifier — state / variation / it."""
    if name == "state":
        # Reached only by a `state` that does not head an attribute chain (`state[...]`, `length(state)`,
        # a bare `state`): a chain is resolved whole by _resolve_attr. The root is not a value, so there
        # is nothing to hand back — and a container of every dimension would let a check read state
        # without naming the dimension the authoring gate resolves.
        raise DSLError("'state' is not a value: name a declared world dimension under it (state.<dimension>).")
    if name == "variation":
        return ctx.variation
    if name == "it":
        if "it" not in ctx.binding:
            raise DSLError("'it' referenced outside an any()/all() generator.")
        return ctx.binding["it"]
    raise DSLError(f"Internal error: name {name!r} reached eval despite validation.")


def _resolve_attr(node: ast.Attribute, ctx: _EvalContext) -> Any:
    """Resolve an attribute path against the current value."""
    if (segments := _state_segments(node)) is not None:
        if ctx.world is None:
            raise DSLError(
                f"state.{'.'.join(segments)} reads world state, and this host declares no world: "
                "a state path names a declared world dimension, and there is none to name"
            )
        resolved = _resolve_dimension_path(segments, ctx.world, ctx.end_state)
        if resolved is Missing:
            ctx.unresolved.append("state." + ".".join(segments))
        return resolved
    base = _eval_node(node.value, ctx)
    found = _lookup(base, node.attr)
    if found is Missing and base is not Missing:
        ctx.unresolved.append(ast.unparse(node))
    return found


def _resolve_subscript(node: ast.Subscript, ctx: _EvalContext) -> Any:
    """Resolve an indexed lookup (list[int], dict[str], slicing not supported)."""
    base = _eval_node(node.value, ctx)
    key = _eval_node(node.slice, ctx)
    if base is Missing:
        return Missing
    try:
        return base[key]
    except KeyError, IndexError, TypeError:
        ctx.unresolved.append(ast.unparse(node))
        return Missing


def _lookup(base: Any, attr: str) -> Any:
    """Look up ``attr`` on ``base``. Returns Missing if not found.

    Underscore-prefixed attributes are rejected outright — DSL expressions
    should never reach into dunders (``__class__``, ``__bases__``, …) or
    private fields. The parser already blocks function calls to non-allowlisted
    names, but bare attribute access on resolved values is open by design
    (Pydantic field access on ``state.inventory.orders[0]``-style items).
    Rejecting ``_``-prefixed names closes the operator-hygiene gap.
    """
    if base is Missing:
        return Missing
    if attr.startswith("_"):
        return Missing
    # Synthetic .length attribute on lists / dicts / strings — DSL ergonomic helper.
    if attr == "length":
        try:
            return len(base)
        except TypeError:
            return Missing
    if isinstance(base, dict):
        return base.get(attr, Missing)
    # Attribute access on plain Python objects (Pydantic instances, etc.)
    # is allowed for known fields; missing attrs → Missing.
    return getattr(base, attr, Missing)


def _state_segments(node: ast.Attribute) -> list[str] | None:
    """The segments of a pure attribute chain rooted at ``state``, or None for any other node.

    The whole chain, because a dimension's name may span several segments (``inbox.messages``) and
    only the full path can be resolved by its longest declared prefix. An index ends the chain: what
    lies beyond it addresses inside a value, and the walker resolves it from the chain below.

    Args:
        node: An attribute node.

    Returns:
        ``["inbox", "messages", "length"]`` for ``state.inbox.messages.length``; None when the chain
        roots anywhere but ``state`` or passes through an index or a call.
    """
    segments: list[str] = []
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        segments.append(current.attr)
        current = current.value
    if not (isinstance(current, ast.Name) and current.id == "state"):
        return None
    segments.reverse()
    return segments


def _resolve_dimension_path(segments: list[str], world: WorldRegistry, end_state: Mapping[str, Any]) -> Any:
    """Read ``state.<segments>`` as a dimension name and a path inside its value.

    The dimension is the longest declared prefix — :meth:`WorldRegistry.resolve_path`, the same
    resolution the authoring gate refuses an unknown path by, so a check that passes the gate reads
    the value the gate resolved it to. The end state is keyed by that name, so no layout is consulted.

    Args:
        segments: The attribute chain below ``state``.
        world: The host's world registry.
        end_state: The world to read, keyed by declared dimension name.

    Returns:
        The value at the path, or :data:`Missing` when it names no declared dimension, the end state
        holds no value for the dimension, or the rest of the path does not resolve inside it.
    """
    name = world.resolve_path(".".join(segments))
    if name is None or name not in end_state:
        return Missing
    value: Any = end_state[name]
    for attr in segments[len(name.split(".")) :]:
        value = _lookup(value, attr)
    return value


def _eval_unary(node: ast.UnaryOp, ctx: _EvalContext) -> Any:
    """Evaluate unary not/+/-; each keeps :data:`Missing` as Missing, so ``not`` never turns unknown into True."""
    val = _eval_node(node.operand, ctx)
    if val is Missing:
        return Missing
    if isinstance(node.op, ast.Not):
        return not val
    if isinstance(node.op, ast.USub | ast.UAdd):
        try:
            return -val if isinstance(node.op, ast.USub) else +val
        except TypeError as mismatch:
            # The world holding a value the template cannot negate is the template's or the rig's fault,
            # raised like a comparison's type mismatch, so the cell is excluded rather than scored.
            raise DSLError(f"cannot apply unary {type(node.op).__name__} to {type(val).__name__}") from mismatch
    raise DSLError(f"Unhandled unary operator: {type(node.op).__name__}")


def _eval_boolop(node: ast.BoolOp, ctx: _EvalContext) -> Any:
    """Evaluate ``and``/``or`` by Kleene's three-valued logic, short-circuiting on the deciding value.

    ``and`` is False as soon as an operand is False, else :data:`Missing` if any operand was, else
    True; ``or`` is the dual. Reading ``Missing`` as False here — the obvious shortcut — is what lets
    ``not (a and b)`` hold over a world that holds neither ``a`` nor ``b``.
    """
    if not isinstance(node.op, ast.And | ast.Or):
        raise DSLError(f"Unhandled boolean operator: {type(node.op).__name__}")
    deciding = isinstance(node.op, ast.Or)  # the truth value that settles the whole expression
    unknown = False
    for v in node.values:
        value = _eval_node(v, ctx)
        if value is Missing:
            unknown = True
        elif bool(value) is deciding:
            return deciding
    return Missing if unknown else not deciding


def _eval_compare(node: ast.Compare, ctx: _EvalContext) -> Any:
    """Evaluate chained comparisons (a < b < c is two pairwise comparisons, joined by ``and``).

    A pair with a :data:`Missing` side is Missing, not False: the comparison was not established, and
    answering False would make its negation True. A chain is False when any pair is False, else
    Missing when any pair was, else True — the same Kleene ``and`` as :func:`_eval_boolop`.
    """
    left = _eval_node(node.left, ctx)
    unknown = False
    for op, comparator in zip(node.ops, node.comparators):
        right = _eval_node(comparator, ctx)
        if left is Missing or right is Missing:
            unknown = True
            left = right
            continue
        try:
            if isinstance(op, ast.Eq):
                ok = left == right
            elif isinstance(op, ast.NotEq):
                ok = left != right
            elif isinstance(op, ast.Lt):
                ok = left < right
            elif isinstance(op, ast.LtE):
                ok = left <= right
            elif isinstance(op, ast.Gt):
                ok = left > right
            elif isinstance(op, ast.GtE):
                ok = left >= right
            else:
                raise DSLError(f"Unhandled comparison operator: {type(op).__name__}")
        except TypeError as mismatch:
            # Values the world holds that cannot be compared are the template's or the rig's
            # fault, not the candidate's: raised, so the cell is excluded rather than scored.
            raise DSLError(
                f"cannot compare {type(left).__name__} with {type(right).__name__} ({type(op).__name__})"
            ) from mismatch
        if not ok:
            return False
        left = right
    return Missing if unknown else True


def _eval_call(node: ast.Call, ctx: _EvalContext) -> Any:
    """Dispatch a Call node to the DSL builtin handlers."""
    if not isinstance(node.func, ast.Name):
        # Unreachable past parse(), which refuses any call that is not a named builtin; restated so
        # the evaluator does not depend on that for its own soundness.
        raise DSLError(f"Unknown function call: {ast.dump(node.func)!r}")
    func_name = node.func.id
    if func_name in ("any", "all"):
        return _eval_any_all(func_name, node, ctx)
    args = [_eval_node(a, ctx) for a in node.args]
    if node.keywords:
        raise DSLError(f"{func_name}() does not accept keyword arguments.")
    if func_name == "contains":
        return _builtin_contains(*_check_arity("contains", args, 2))
    if func_name == "intersects":
        return _builtin_intersects(*_check_arity("intersects", args, 2))
    if func_name == "length":
        return _builtin_length(*_check_arity("length", args, 1))
    if func_name == "called_before":
        return _builtin_called_before(ctx.ledger, *_check_arity("called_before", args, 2))
    if func_name == "called_after":
        return _builtin_called_after(ctx.ledger, *_check_arity("called_after", args, 2))
    if func_name == "call_count":
        return _builtin_call_count(ctx.ledger, *_check_arity("call_count", args, 1))
    if func_name == "last_call_was":
        return _builtin_last_call_was(ctx.ledger, *_check_arity("last_call_was", args, 1))
    if func_name == "calls":
        return _builtin_calls(ctx.ledger, *_check_arity("calls", args, 1))
    if func_name == "passed":
        _check_arity("passed", args, 0)
        return _builtin_passed(ctx.ledger)
    if func_name in FIRE_PREDICATES:
        return _builtin_fired(func_name, ctx, ast.unparse(node), *_check_arity(func_name, args, 1))
    raise DSLError(f"Unknown DSL function: {func_name}")


def _builtin_fired(predicate: str, ctx: _EvalContext, call: str, name: Any) -> Any:
    """Whether the triggered dimension ``name`` fired during the cell — for ``fired_armed``, as the seed's armed event.

    Args:
        predicate: ``fired`` (any firing, whoever caused it) or ``fired_armed`` (a firing of the event
            the cell's seed armed, so not the world's own firing on the same dimension).
        ctx: The evaluation context, whose ``fired`` is what fired (None when no world events were
            recorded) and whose ``unresolved`` names what a :data:`Missing` rests on.
        call: The call as written, named in the detail when it is not established.
        name: The dimension.

    Returns:
        Whether it fired as the predicate asks; :data:`Missing` for ``fired_armed`` when which firings
        were armed cannot be known (:attr:`~threetears.evals.schema.world_events.Firings.armed_known`
        — a witnessed cell, which no seed armed), so ``not fired_armed(...)`` is not established either.

    Raises:
        DSLError: No world events were recorded for this evaluation. Answering False there would
            score a cell whose events nobody recorded as one in which nothing fired.
    """
    fired = ctx.fired
    if fired is None:
        raise DSLError(
            f"{predicate}({name!r}) reads the cell's world events, and none were recorded for this evaluation — "
            "a cell that opened no world session has no record to read"
        )
    if predicate == "fired_armed":
        if not fired.armed_known:
            ctx.unresolved.append(call)
            return Missing
        return name in fired.armed
    return name in fired.dimensions


def _check_arity(name: str, args: list[Any], expected: int) -> list[Any]:
    if len(args) != expected:
        raise DSLError(f"{name}() takes {expected} argument(s); got {len(args)}.")
    return args


def _builtin_contains(haystack: Any, needle: Any) -> Any:
    """Whether ``needle`` is in ``haystack`` — substring over a string, membership otherwise; :data:`Missing` over one.

    The substring half is refused over model prose where a template is AUTHORED, not here — see
    the module's *Text matches over model prose* — so a stored template evaluates as it always did.
    """
    if haystack is Missing or needle is Missing:
        return Missing
    try:
        return needle in haystack
    except TypeError as mismatch:
        raise DSLError(
            f"contains(): cannot test a {type(needle).__name__} in a {type(haystack).__name__}"
        ) from mismatch


def _builtin_intersects(a: Any, b: Any) -> Any:
    """Whether two sets-of-elements share at least one element; :data:`Missing` when either is Missing."""
    if a is Missing or b is Missing:
        return Missing
    try:
        set_a = set(a) if not isinstance(a, str) else {a}
        set_b = set(b) if not isinstance(b, str) else {b}
    except TypeError as mismatch:
        raise DSLError(
            f"intersects(): cannot take the elements of {type(a).__name__} and {type(b).__name__}"
        ) from mismatch
    return bool(set_a & set_b)


def _builtin_length(value: Any) -> Any:
    """Return ``len(value)`` or Missing if unsizable."""
    if value is Missing:
        return Missing
    try:
        return len(value)
    except TypeError:
        return Missing


def _all_calls_with_index(ledger: CallLedger) -> list[tuple[int, str, str]]:
    """Return ``[(index, tool, action), ...]`` over the ledger's calls, in recorded order across every tool.

    The engine's deliberate-pass entry is left out: it is not a call, and only ``passed()`` reads it.
    """
    return [(i, call.tool, call.action) for i, call in enumerate(ledger.calls) if not is_pass(call)]


def _builtin_passed(ledger: CallLedger) -> bool:
    """True when the cell recorded a deliberate pass and made no call.

    A pass beside a call is an act followed (or preceded) by a pass, which is not passing; an empty ledger is
    doing nothing without saying so, which is not a deliberate pass either.
    """
    return any(is_pass(call) for call in ledger.calls) and all(is_pass(call) for call in ledger.calls)


def _builtin_called_before(ledger: CallLedger, first: Any, second: Any) -> bool:
    """True when the first occurrence of ``first`` precedes the first occurrence of ``second``."""
    tool_a, action_a = parse_tool_action(first)
    tool_b, action_b = parse_tool_action(second)
    calls = _all_calls_with_index(ledger)
    first_idx_a = next((i for i, t, a in calls if t == tool_a and a == action_a), None)
    first_idx_b = next((i for i, t, a in calls if t == tool_b and a == action_b), None)
    if first_idx_a is None or first_idx_b is None:
        return False
    return first_idx_a < first_idx_b


def _builtin_called_after(ledger: CallLedger, first: Any, second: Any) -> bool:
    """True when the last occurrence of ``first`` follows the last occurrence of ``second``."""
    tool_a, action_a = parse_tool_action(first)
    tool_b, action_b = parse_tool_action(second)
    calls = _all_calls_with_index(ledger)
    last_idx_a = next((i for i, t, a in reversed(calls) if t == tool_a and a == action_a), None)
    last_idx_b = next((i for i, t, a in reversed(calls) if t == tool_b and a == action_b), None)
    if last_idx_a is None or last_idx_b is None:
        return False
    return last_idx_a > last_idx_b


def _builtin_call_count(ledger: CallLedger, spec: Any) -> int:
    """Return the count of recorded calls matching ``tool.action``."""
    tool, action = parse_tool_action(spec)
    calls = _all_calls_with_index(ledger)
    return sum(1 for _, t, a in calls if t == tool and a == action)


def _builtin_last_call_was(ledger: CallLedger, spec: Any) -> bool:
    """True when the most recent recorded call matches ``tool.action``."""
    tool, action = parse_tool_action(spec)
    calls = _all_calls_with_index(ledger)
    if not calls:
        return False
    _, last_tool, last_action = calls[-1]
    return last_tool == tool and last_action == action


def _builtin_calls(ledger: CallLedger, spec: Any) -> list[dict[str, Any]]:
    """Return the recorded parameters of every call matching ``tool.action``, in ledger order.

    Copies, so an expression cannot reach the ledger it reads.
    """
    tool, action = parse_tool_action(spec)
    return [
        copy.deepcopy(call.params)
        for call in ledger.calls
        if call.tool == tool and call.action == action and not is_pass(call)
    ]


def _eval_any_all(func_name: str, node: ast.Call, ctx: _EvalContext) -> Any:
    """Evaluate ``any(<body> for it in <path>)`` / ``all(...)`` three-valued.

    :data:`Missing` over a Missing iterable. Over elements, ``any`` is the Kleene ``or`` of the bodies
    and ``all`` the Kleene ``and``: an element whose body is Missing leaves the answer unknown unless
    another element settles it.
    """
    if len(node.args) != 1 or not isinstance(node.args[0], ast.GeneratorExp):
        raise DSLError(f"{func_name}() takes one generator expression argument.")
    genexp = node.args[0]
    gen = genexp.generators[0]
    iterable = _eval_node(gen.iter, ctx)
    if iterable is Missing:
        return Missing
    try:
        items = list(iterable)
    except TypeError as mismatch:
        raise DSLError(f"{func_name}(): cannot iterate a {type(iterable).__name__}") from mismatch

    deciding = func_name == "any"  # the element verdict that settles the whole expression
    body = genexp.elt

    def _eval_with_binding(item: Any) -> Any:
        previous = ctx.binding.get("it", _SENTINEL)
        ctx.binding["it"] = item
        try:
            return _eval_node(body, ctx)
        finally:
            if previous is _SENTINEL:
                ctx.binding.pop("it", None)
            else:
                ctx.binding["it"] = previous

    unknown = False
    for item in items:
        value = _eval_with_binding(item)
        if value is Missing:
            unknown = True
        elif bool(value) is deciding:
            return deciding
    return Missing if unknown else not deciding


_SENTINEL = object()


__all__ = [
    "Missing",
    "NOT_ESTABLISHED",
    "call_parameter_matches",
    "evaluate",
    "evaluate_with_detail",
    "not_established_detail",
    "undefined_action",
    "undefined_call_references",
    "undefined_fire_references",
    "undefined_fired_dimension",
    "world_prose_matches",
]
