"""Goal-state DSL — small expression language for eval scoring.

Expressions are evaluated against a :class:`~threetears.evals.contracts.world_state.WorldState`
(``state``) and a variation parameter dict (``variation``). They return a
:class:`bool` (or :class:`Missing` if a path resolution fails — comparisons
against ``Missing`` always return ``False`` rather than raising).

Surface
-------

Path access (Python-style)::

    state.inventory.orders           # list
    state.inventory.orders.length    # len(...)
    state.inventory.orders[-1]       # indexing
    state.support.messages[-1].content
    variation.region_pair            # variation parameter

With the host's world registry passed, ``state.<path>`` roots at a dimension NAME, as the
authoring gate reads it: ``state.inbox.messages`` and a flat ``state.ingest_backlog`` both resolve,
each found where the registry's addressing put it. Without one, the path reads the world's
``namespace.key`` layout directly.

Comparisons::

    state.inventory.orders.length >= 1
    state.inventory.orders[0].sku == "SKU-42"
    variation.tone != "hostile"

Predicates (function-call form)::

    contains(state.inventory.orders, "SKU-42")              # value in path
    intersects(state.inventory.orders, variation.categories) # non-empty intersection
    length(state.inventory.orders) >= 2                     # equivalent to .length

Ordering predicates (across all tools' recorded calls)::

    called_before("inventory.search", "inventory.place_order")
    called_after("inventory.place_order", "inventory.cancel_order")
    call_count("inventory.place_order") >= 2
    last_call_was("inventory.place_order")

Call parameters (``calls()`` returns the matching calls' recorded parameters, in order)::

    calls("inventory.place_order").length >= 3
    any(it.priority == "rush" for it in calls("inventory.add_note"))
    all(it.note_text.length > 0 for it in calls("inventory.place_order"))

Generator predicates (``it`` binds to each element)::

    any(it.sku == "X" for it in state.inventory.orders)
    all(intersects(it.tags, variation.categories) for it in state.inventory.orders)

Boolean composition::

    state.inventory.orders.length >= 1 and any(it.sku == "X" for it in state.inventory.orders)
    not state.support.messages[-1].content == ""

Static extraction
-----------------

:func:`extract_paths` parses an expression and reports what it reads without evaluating
anything, so an authoring gate can resolve a path against a host's declared vocabulary
before a run exists::

    state.inventory.orders.length >= 1      # reads the path inventory.orders.length
    state.inventory.orders[0].sku == "X"    # reads inventory.orders — an index addresses inside it

Roots are reported apart rather than merged: ``variation`` names a case parameter, not
world state, and a path below an index or bound to ``it`` addresses inside a value rather
than naming one. Every example in this docstring is extracted by
``tests/test_dsl.py``, so a surface documented here that the language cannot
actually parse fails there rather than misleading an author.

Text matches over model prose
-----------------------------

``contains()`` is generic: membership over a structured value (an array of orders) and a
substring test over a string. The first is a mechanical check; the second, over text a model
wrote, is a keyword test on prose — an eval dimension wearing a validator's clothes. The
language cannot tell them apart, because the difference is the operand's TYPE,
so :func:`extract_text_matches` reports every text comparison with the field it reads and the
authoring gate asks the host's vocabulary which of those fields are prose
(:func:`world_prose_matches`). Evaluation is unchanged: a stored template
keeps loading and running; the refusal is at authoring.

A call parameter is text the model wrote until its tool says otherwise. ``calls()`` exposes what
the subject passed, which includes what it said, so a text comparison over a parameter is allowed
only where the action's own parameter schema — the one the model is shown — closes the value
(``enum``, ``const`` or ``pattern``); :func:`call_parameter_matches` reports every other one, and
every one over an action or parameter the host does not describe. The schema is the host's
statement, read as given: the tool's own model-facing schema is the source. A ``pattern`` that admits free text would make that statement false, which is the host's
defect to fix in its schema; this module does not second-guess a regex.

Safety
------

The parser uses :mod:`ast` with a strict node allowlist. Disallowed: lambdas,
dict literals, conditional expressions, list comprehensions (only generator
expressions for ``any``/``all``), attribute access on non-allowlisted root
names, function calls to anything except the DSL builtins above. Malformed
or disallowed expressions raise :class:`DSLError` at parse time, not at
evaluation time — pinning down bad expressions in the template editor before
a run starts.
"""

from __future__ import annotations

import ast
import itertools
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from threetears.evals.contracts.prose import schema_is_prose, schema_nodes_at
from threetears.evals.contracts.world_state import WorldState

if TYPE_CHECKING:
    from threetears.evals.contracts.host.world import WorldRegistry


class DSLError(ValueError):
    """Raised when a DSL expression is malformed or disallowed."""


class _Missing:
    """Sentinel for unresolved paths.

    Compares unequal to everything (including itself for ordering operators)
    so a missing path does not silently match anything. Equality against
    ``None`` returns False (Missing is *unknown*, not *known-absent*).
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

#: Root names allowed at the head of a path. Anything else parses as a DSL error.
_ALLOWED_ROOTS = frozenset({"state", "variation", "it"})

#: Roots that address INSIDE a value rather than naming one, so extraction reports nothing under
#: them. ``it`` binds to elements of an iterable that is itself a path, so its segments sit below
#: something the extraction already reported and repeating them would invent a dimension per
#: element.
_ADDRESSES_WITHIN_A_VALUE = frozenset({"it"})

#: Roots :func:`extract_paths` reports, DERIVED from what the parser admits rather than listed
#: beside it. Two lists of one fact is what a root added to the validator alone would ride
#: through: the expression would parse, extract to nothing, and let a vocabulary check report
#: clean over paths nobody resolved — a false negative in a detector, which is the failure this
#: package refuses everywhere else.
_EXTRACTED_ROOTS = _ALLOWED_ROOTS - _ADDRESSES_WITHIN_A_VALUE

#: DSL builtin function names — recognized in Call nodes.
#: ``not`` is intentionally NOT here: Python's parser emits it as
#: :class:`ast.UnaryOp(ast.Not, ...)`, never as a function call.
_BUILTINS = frozenset(
    {
        "contains",
        "intersects",
        "length",
        "called_before",
        "called_after",
        "call_count",
        "last_call_was",
        "calls",
        "any",
        "all",
    }
)


@dataclass
class _EvalContext:
    """Per-call evaluation context passed through the AST walker."""

    state: WorldState
    variation: dict[str, Any]
    binding: dict[str, Any]  # `it` -> current element inside any()/all()
    world: WorldRegistry | None = None  # how `state.<dimension>` finds a value; None reads the layout raw


# =============================================================================
# Parser — ast.parse with allowlist validation
# =============================================================================


def parse(expression: str) -> ast.Expression:
    """Parse a goal-state expression into an AST and validate the node set.

    Raises:
        DSLError: If the expression is syntactically invalid or contains
            disallowed nodes.

    Returns:
        An :class:`ast.Expression` ready to evaluate with :func:`_eval_node`.
    """
    if not isinstance(expression, str) or not expression.strip():
        raise DSLError("Expression must be a non-empty string.")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise DSLError(f"Syntax error in expression: {e}") from e
    _validate(tree)
    return tree


def _validate(tree: ast.AST) -> None:
    """Walk the parsed tree and reject any disallowed nodes."""
    for node in ast.walk(tree):
        if isinstance(
            node,
            (
                ast.Expression,
                ast.BoolOp,
                ast.UnaryOp,
                ast.Compare,
                ast.Name,
                ast.Attribute,
                ast.Subscript,
                ast.Constant,
                ast.Call,
                ast.List,
                ast.Tuple,
                ast.Load,
                ast.Store,
                ast.And,
                ast.Or,
                ast.Not,
                ast.Eq,
                ast.NotEq,
                ast.Gt,
                ast.GtE,
                ast.Lt,
                ast.LtE,
                ast.GeneratorExp,
                ast.comprehension,
                ast.USub,
                ast.UAdd,
            ),
        ):
            continue
        raise DSLError(f"Disallowed node type in expression: {type(node).__name__}")

    # Additional structural checks
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            # Allowed: root names (state/variation/it), builtins, or loop targets.
            if node.id in _ALLOWED_ROOTS or node.id in _BUILTINS:
                continue
            # A Name inside a generator's comprehension target is bound — but
            # for that we need to know context; checked separately below.
            # Reject anything else.
            raise DSLError(
                f"Unknown identifier {node.id!r}; allowed roots: state, variation, it"
                f" — call DSL builtins by name: {sorted(_BUILTINS)}."
            )
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _BUILTINS:
                func_repr = getattr(node.func, "id", None) or ast.dump(node.func)
                raise DSLError(f"Unknown function call: {func_repr!r}")
            if node.func.id == "calls" and "." not in (_calls_spec(node) or ""):
                # A computed spec would leave the authoring gate unable to name the action whose
                # schema says which parameters are free text, so it could neither refuse nor admit.
                raise DSLError("calls() takes one 'tool.action' string literal.")
        if isinstance(node, ast.GeneratorExp):
            if len(node.generators) != 1:
                raise DSLError("any/all generators must have exactly one 'for' clause.")
            gen = node.generators[0]
            if gen.ifs:
                raise DSLError("any/all generators may not have 'if' filters in this DSL.")
            if not (isinstance(gen.target, ast.Name) and gen.target.id == "it"):
                raise DSLError("any/all generators must bind to the variable 'it'.")


# =============================================================================
# Static extraction — what an expression reads, without evaluating it
# =============================================================================


@dataclass(frozen=True)
class ExtractedPaths:
    """Where one expression reads from, grouped by the root it addresses through.

    Produced without evaluating anything and without resolving anything: this says what an
    expression *reaches for*, and whoever holds a vocabulary decides whether it exists. That
    split is what lets an authoring gate refuse a path before a run, which is the whole point of
    a closed grammar over an open vocabulary.
    """

    by_root: Mapping[str, tuple[str, ...]]
    """``{root: dotted paths}`` for every root extraction reports, in first-appearance order.

    **One store with the named accessors reading it**, rather than a field per root. A root the
    grammar grows lands here with no change to this class, so anything enumerating sees it; a
    field per root would silently drop it, which is the same two-lists-of-one-fact shape the root
    set itself is derived to avoid.
    """

    @property
    def world(self) -> tuple[str, ...]:
        """Dotted paths under ``state``.

        A path roots at a declared state dimension's name and the segments beyond it address
        inside that dimension's value — so a resolver takes the longest declared prefix rather
        than matching the whole string. Where the host declares ``inventory.orders``,
        ``state.inventory.orders.length`` and ``state.inventory.orders`` are both that dimension.

        Returns:
            The paths, in first-appearance order.
        """
        return self.by_root.get("state", ())

    @property
    def variation(self) -> tuple[str, ...]:
        """Dotted paths under ``variation``.

        Case parameters, not world state. Read apart rather than folded in because resolving one
        against a world registry would refuse every expression that reads its own case.

        Returns:
            The paths, in first-appearance order.
        """
        return self.by_root.get("variation", ())


def extract_paths(expression: str) -> ExtractedPaths:
    """Parse ``expression`` and return the paths it reads, without evaluating it.

    **Nothing here assumes a path roots at world state.** Roots are grouped, and anything that is
    not a path — a builtin call, a literal, a comparison — contributes nothing rather than being
    refused, so a builtin the grammar gains later is not rejected by an extractor written before
    it existed.

    Two kinds of segment are deliberately absent from what comes back:

    * **anything below an index.** ``state.inventory.orders[0].sku`` yields ``inventory.orders``, because
      a subscript addresses into a value and a dimension is the value, never one element of it.
      A resolver can say whether the dimension exists; only that dimension's own schema could
      speak for ``sku``, and this function does not have one.
    * **anything rooted at ``it``.** The loop variable binds to elements of an iterable that is
      itself a path, and that path is extracted where it is written — so ``any(it.sku == "X"
      for it in state.inventory.orders)`` reads ``inventory.orders`` and nothing else.

    Args:
        expression: DSL expression text.

    Returns:
        The paths it reads, deduplicated, in the order they first appear.

    Raises:
        DSLError: The expression is malformed or contains disallowed nodes. Extraction parses
            with the same validator evaluation does, so a path that cannot be extracted is one
            that could never have been evaluated either.
    """
    found: dict[str, list[str]] = {root: [] for root in _EXTRACTED_ROOTS}
    _collect_paths(parse(expression).body, found)
    return ExtractedPaths(by_root={root: tuple(dict.fromkeys(paths)) for root, paths in found.items()})


def _collect_paths(node: ast.AST, found: dict[str, list[str]]) -> None:
    """Record every rooted path under ``node`` into ``found``, in source order.

    Args:
        node: The node to walk.
        found: ``{root: [dotted path, ...]}``, appended to in place.
    """
    if isinstance(node, (ast.Attribute, ast.Subscript)):
        base, segments, indices = _unwind_chain(node)
        if isinstance(base, ast.Name):
            if base.id in found and segments:
                found[base.id].append(".".join(segments))
        else:
            # A chain over something that is not a bare name — `length(state.x).y` and its kin.
            # The chain contributes no path of its own; what it is built on may.
            _collect_paths(base, found)
        for index in indices:
            _collect_paths(index, found)
        return
    for child in ast.iter_child_nodes(node):
        _collect_paths(child, found)


def _unwind_chain(node: ast.AST) -> tuple[ast.AST, list[str], list[ast.AST]]:
    """Split an attribute/index chain into what it is rooted on and how it addresses.

    Walks outermost to innermost, so an index encountered on the way down discards the segments
    already collected: those sit *above* the index and therefore address inside one element,
    which no dimension name reaches.

    Args:
        node: An :class:`ast.Attribute` or :class:`ast.Subscript`.

    Returns:
        ``(base, segments, indices)`` — the innermost non-chain node, the dotted segments between
        it and the first index, and every index expression, each of which may hold paths of its
        own.
    """
    segments: list[str] = []
    indices: list[ast.AST] = []
    current: ast.AST = node
    while True:
        if isinstance(current, ast.Attribute):
            segments.append(current.attr)
            current = current.value
        elif isinstance(current, ast.Subscript):
            segments.clear()
            indices.append(current.slice)
            current = current.value
        else:
            break
    segments.reverse()
    return current, segments, indices


#: What a :class:`TextMatch` does with the text it reads.
TextPredicate = Literal["contains", "intersects", "equality"]


@dataclass(frozen=True)
class TextMatch:
    """One deterministic string comparison an expression performs, and what it reads.

    Reported statically, like :class:`ExtractedPaths`, and for the same kind of consumer: an
    authoring gate that holds a vocabulary this module does not. Whether the comparison is
    legitimate depends on what the operand IS — ``contains()`` over an array of orders is
    membership over structured values, the same call over a string a model wrote is a substring
    test over prose — and only the vocabulary's schema can say which.
    """

    source: str
    """The predicate as written back (``ast.unparse``), for an error an author can find."""

    operand: tuple[str, ...]
    """What the predicate reads, from the ``state`` root (stripped), one segment per step.

    Unlike :attr:`ExtractedPaths.world`, segments BELOW an index are kept, with ``"[]"`` marking
    the index — ``state.inventory.orders[0].note`` is ``("inventory", "orders", "[]", "note")`` — because
    the question here is about the field compared, not the dimension that holds it. ``it`` inside
    ``any()``/``all()`` is resolved to the iterable it binds to, plus ``"[]"``.
    """

    predicate: TextPredicate
    """``contains``/``intersects`` (the builtin), or ``equality`` (``==``/``!=``)."""

    call: str | None = None
    """The ``tool.action`` whose recorded parameters the operand reads through ``calls()``, if any.

    Set, :attr:`operand` addresses inside ONE call's parameters (``("position",)``), so it is
    resolved against that action's parameter schema rather than against the world.
    """


def extract_text_matches(expression: str) -> tuple[TextMatch, ...]:
    """Parse ``expression`` and report every deterministic string comparison it makes over state.

    Reported:

    * ``contains(X, ...)`` — ``X`` is the operand;
    * ``intersects(X, Y)`` — each side that is a state path is an operand;
    * ``X == v`` / ``X != v`` where ``v`` is a non-empty string literal, a ``variation`` path or
      another state path.

    **Emptiness is not reported**: ``X == ""`` asks whether anything was written, which is
    structure, not a reading of what was written. Neither is a comparison against a number.

    Args:
        expression: DSL expression text.

    Returns:
        The matches, in source order.

    Raises:
        DSLError: The expression is malformed or contains disallowed nodes.
    """
    found: list[TextMatch] = []
    _collect_text_matches(parse(expression).body, None, found)
    return tuple(found)


#: What an operand addresses: the ``calls()`` spec it reads through (None for world state), then its
#: segments. One shape for both roots, so ``it`` bound to either resolves by the same walk.
_Operand = tuple[str | None, tuple[str, ...]]


def _calls_spec(node: ast.AST) -> str | None:
    """The literal ``tool.action`` of a ``calls(...)`` node, or None when ``node`` is not one."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "calls"):
        return None
    if len(node.args) != 1 or node.keywords:
        return None
    (arg,) = node.args
    return arg.value if isinstance(arg, ast.Constant) and isinstance(arg.value, str) else None


def _match(node: ast.AST, operand: _Operand, predicate: TextPredicate) -> TextMatch | None:
    """A :class:`TextMatch` for ``operand``, or None when it addresses no field.

    A ``calls()`` operand drops the index that picks one call; what remains names a parameter.
    Nothing remaining is the call list or one call's whole parameter dict, which no string equals.
    """
    call, segments = operand
    if call is not None:
        segments = segments[1:] if segments[:1] == ("[]",) else segments
        if not segments:
            return None
    elif not segments:
        return None
    return TextMatch(source=ast.unparse(node), operand=segments, predicate=predicate, call=call)


def _collect_text_matches(node: ast.AST, it_binding: _Operand | None, found: list[TextMatch]) -> None:
    """Record each text match under ``node``; ``it_binding`` is what ``it`` addresses, if bound.

    Args:
        node: The node to walk.
        it_binding: The operand ``it`` stands for inside the enclosing ``any()``/``all()``.
        found: Appended to in place.
    """
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        if name in ("any", "all") and node.args and isinstance(node.args[0], ast.GeneratorExp):
            generator = node.args[0]
            iterable = _text_operand(generator.generators[0].iter, it_binding)
            _collect_text_matches(generator.generators[0].iter, it_binding, found)
            element = (iterable[0], (*iterable[1], "[]")) if iterable is not None else None
            _collect_text_matches(generator.elt, element, found)
            return
        if name == "contains" and node.args:
            operand = _text_operand(node.args[0], it_binding)
            if operand is not None and (match := _match(node, operand, "contains")) is not None:
                found.append(match)
        elif name == "intersects":
            for argument in node.args:
                operand = _text_operand(argument, it_binding)
                if operand is not None and (match := _match(node, operand, "intersects")) is not None:
                    found.append(match)
    elif isinstance(node, ast.Compare):
        sides = [node.left, *node.comparators]
        for op, (left, right) in zip(node.ops, itertools.pairwise(sides), strict=True):
            if not isinstance(op, (ast.Eq, ast.NotEq)):
                continue
            for subject, other in ((left, right), (right, left)):
                operand = _text_operand(subject, it_binding)
                if operand is not None and _is_text_comparand(other, it_binding):
                    if (match := _match(node, operand, "equality")) is not None:
                        found.append(match)
    for child in ast.iter_child_nodes(node):
        _collect_text_matches(child, it_binding, found)


def _text_operand(node: ast.AST, it_binding: _Operand | None) -> _Operand | None:
    """What ``node`` addresses in world state or in recorded call parameters, or None.

    Args:
        node: An expression node.
        it_binding: What ``it`` stands for, if bound.

    Returns:
        ``(calls spec or None, segments)`` with ``"[]"`` for each index, or None for a literal, a
        call other than ``calls()``, a ``variation`` path, or ``it`` outside any binding.
    """
    segments: list[str] = []
    current = node
    while isinstance(current, (ast.Attribute, ast.Subscript)):
        segments.append(current.attr if isinstance(current, ast.Attribute) else "[]")
        current = current.value
    segments.reverse()
    if (spec := _calls_spec(current)) is not None:
        return (spec, tuple(segments))
    if not isinstance(current, ast.Name):
        return None
    if current.id == "state":
        return (None, tuple(segments)) if segments else None
    if current.id == "it" and it_binding is not None:
        return (it_binding[0], (*it_binding[1], *segments))
    return None


def _is_text_comparand(node: ast.AST, it_binding: _Operand | None) -> bool:
    """Whether an equality's other side makes it a comparison of text rather than a count or emptiness.

    Args:
        node: The other side of ``==``/``!=``.
        it_binding: What ``it`` stands for, if bound.

    Returns:
        True for a non-empty string literal, a ``variation`` path, or another state path.
    """
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str) and node.value != ""
    if _text_operand(node, it_binding) is not None:
        return True
    base = node
    while isinstance(base, (ast.Attribute, ast.Subscript)):
        base = base.value
    return isinstance(base, ast.Name) and base.id == "variation" and base is not node


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
        tool, action = _parse_tool_action(match.call)
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


#: The builtins whose string arguments name a ``tool.action``.
_ACTION_BUILTINS = frozenset({"called_before", "called_after", "call_count", "last_call_was", "calls"})


def referenced_actions(expression: str) -> tuple[tuple[str, str], ...]:
    """Every ``(tool, action)`` a goal check names through a call builtin, in source order, each once.

    The one walk over what a check refers to: :func:`undefined_call_references` asks it whether each name
    exists for the host, and a launch asks it whether each is offered to the candidate.

    Args:
        expression: A goal-state expression.

    Returns:
        The named actions.

    Raises:
        DSLError: The expression does not parse.
    """
    named: list[tuple[str, str]] = []
    for node in ast.walk(parse(expression).body):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _ACTION_BUILTINS):
            continue
        for argument in node.args:
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str) and "." in argument.value:
                pair = _parse_tool_action(argument.value)
                if pair not in named:
                    named.append(pair)
    return tuple(named)


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

    operands: list[_Operand] = []
    _collect_call_operands(body, None, operands)
    for spec, segments in operands:
        named = segments[1:] if segments[:1] == ("[]",) else segments
        if named[-1:] == ("length",):
            named = named[:-1]
        if not named:
            continue
        tool, action = _parse_tool_action(spec)
        schema = parameters(tool, action) if parameters is not None else None
        if schema is not None and not schema_nodes_at(schema, named):
            refuse(f"{spec} declares no parameter {'.'.join(segment for segment in named if segment != '[]')}")
    return tuple(reasons)


def _collect_call_operands(node: ast.AST, it_binding: _Operand | None, found: list[_Operand]) -> None:
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
            iterable = _text_operand(generator.generators[0].iter, it_binding)
            element = (iterable[0], (*iterable[1], "[]")) if iterable is not None else None
            _collect_call_operands(generator.elt, element, found)
            return
    if isinstance(node, (ast.Attribute, ast.Subscript)):
        operand = _text_operand(node, it_binding)
        if operand is not None and operand[0] is not None:
            found.append(operand)
            if isinstance(node, ast.Subscript):
                _collect_call_operands(node.slice, it_binding, found)
            return
    for child in ast.iter_child_nodes(node):
        _collect_call_operands(child, it_binding, found)


# =============================================================================
# Evaluator
# =============================================================================


def evaluate(
    expression: str,
    *,
    state: WorldState,
    variation: dict[str, Any] | None = None,
    world: WorldRegistry | None = None,
) -> bool:
    """Evaluate a goal-state expression and return its boolean result.

    Args:
        expression: DSL expression text.
        state: The :class:`~threetears.evals.contracts.world_state.WorldState` to read
            path expressions against.
        variation: Variation parameter dict (test case's variation_params).
            Defaults to empty when omitted.
        world: The host's world registry. Given one, ``state.<path>`` roots at a dimension name and
            finds the value where the registry's addressing put it — the reading the authoring gate
            resolves the same path by. None (a host with no world, or the language exercised alone)
            reads ``state.<namespace>.<key>`` off the container's layout.

    Returns:
        ``True`` if the expression's value is truthy after coercion;
        ``False`` otherwise (including ``Missing`` results).

    Raises:
        DSLError: For malformed expressions. Evaluation errors against valid
            expressions return ``False`` via :data:`Missing` semantics — they
            never raise at run time.
    """
    tree = parse(expression)
    ctx = _EvalContext(state=state, variation=variation or {}, binding={}, world=world)
    result = _eval_node(tree.body, ctx)
    if result is Missing:
        return False
    return bool(result)


def evaluate_with_detail(
    expression: str,
    *,
    state: WorldState,
    variation: dict[str, Any] | None = None,
    world: WorldRegistry | None = None,
) -> tuple[bool, str]:
    """Evaluate and return ``(result, detail)`` for trace inspection.

    The detail string captures the resolved value(s) so the goal-state
    outcomes panel can show *why* a check passed or failed. Examples::

        ("queue.length=2 satisfies >= 1", True)
        ("messages[-1] = Missing", False)

    Args:
        expression: DSL expression text.
        state: The world to read path expressions against.
        variation: Variation parameter dict. Defaults to empty when omitted.
        world: The host's world registry, as :func:`evaluate` takes it.

    Returns:
        The boolean result and the resolved value's ``repr``, or ``"<Missing>"``.
    """
    tree = parse(expression)
    ctx = _EvalContext(state=state, variation=variation or {}, binding={}, world=world)
    value = _eval_node(tree.body, ctx)
    if value is Missing:
        return False, "<Missing>"
    return bool(value), repr(value)


def _eval_node(node: ast.AST, ctx: _EvalContext) -> Any:
    """Walk one node of the parsed AST and return its value.

    Path resolution failures produce :data:`Missing`; comparisons against
    Missing return False without raising.
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
        return [_eval_node(elt, ctx) for elt in node.elts]
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
        return _StateProxy(ctx.state)
    if name == "variation":
        return ctx.variation
    if name == "it":
        if "it" not in ctx.binding:
            raise DSLError("'it' referenced outside an any()/all() generator.")
        return ctx.binding["it"]
    raise DSLError(f"Internal error: name {name!r} reached eval despite validation.")


def _resolve_attr(node: ast.Attribute, ctx: _EvalContext) -> Any:
    """Resolve an attribute path against the current value."""
    if ctx.world is not None and (segments := _state_segments(node)) is not None:
        return _resolve_dimension_path(segments, ctx.world, ctx.state)
    base = _eval_node(node.value, ctx)
    attr = node.attr
    return _lookup(base, attr)


def _resolve_subscript(node: ast.Subscript, ctx: _EvalContext) -> Any:
    """Resolve an indexed lookup (list[int], dict[str], slicing not supported)."""
    base = _eval_node(node.value, ctx)
    key = _eval_node(node.slice, ctx)
    if base is Missing:
        return Missing
    try:
        return base[key]
    except KeyError, IndexError, TypeError:
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


def _resolve_dimension_path(segments: list[str], world: WorldRegistry, state: WorldState) -> Any:
    """Read ``state.<segments>`` as a dimension name and a path inside its value.

    The dimension is the longest declared prefix — :meth:`WorldRegistry.resolve_path`, the same
    resolution the authoring gate refuses an unknown path by, so a check that passes the gate reads
    the value the gate resolved it to. The value is found where the registry's addressing puts it in
    a world laid out as a seed is, whatever the host names its dimensions: ``ingest_backlog`` held
    under its carrier as ``ingest_backlog``, or ``inbox.messages`` held under ``inbox`` as ``messages``.

    Args:
        segments: The attribute chain below ``state``.
        world: The host's world registry.
        state: The world to read.

    Returns:
        The value at the path, or :data:`Missing` when it names no declared dimension, the world
        holds no value for the dimension, or the rest of the path does not resolve inside it.
    """
    name = world.resolve_path(".".join(segments))
    if name is None:
        return Missing
    dimension = world.get(name)
    if dimension is None:
        return Missing
    key = world.held_at(name, state.namespaces)
    if key is None:
        return Missing
    value: Any = state.namespaces[dimension.carrier][key]
    for attr in segments[len(name.split(".")) :]:
        value = _lookup(value, attr)
    return value


class _StateProxy:
    """Thin wrapper around WorldState that reads dotted paths off its layout.

    ``state.inventory.orders`` ->
        ``state_proxy.inventory`` returns the inventory namespace dict ->
        ``.orders`` calls _lookup on the dict.

    The reading with no world registry. With one, an attribute chain off ``state`` never reaches
    here: :func:`_resolve_dimension_path` reads it as a dimension name instead.
    """

    __slots__ = ("_state",)

    def __init__(self, state: WorldState):
        self._state = state

    def __getattr__(self, name: str) -> Any:
        # `length` is special even here — return the total number of
        # populated namespaces for `state.length` (rarely useful, but consistent).
        if name == "length":
            return len(self._state.namespaces)
        # Otherwise, return the namespace dict (or Missing if absent).
        return self._state.namespaces.get(name, Missing)


def _eval_unary(node: ast.UnaryOp, ctx: _EvalContext) -> Any:
    """Evaluate unary not/+/-."""
    val = _eval_node(node.operand, ctx)
    if isinstance(node.op, ast.Not):
        if val is Missing:
            return Missing
        return not val
    if isinstance(node.op, ast.USub):
        if val is Missing:
            return Missing
        return -val
    if isinstance(node.op, ast.UAdd):
        if val is Missing:
            return Missing
        return +val
    raise DSLError(f"Unhandled unary operator: {type(node.op).__name__}")


def _eval_boolop(node: ast.BoolOp, ctx: _EvalContext) -> Any:
    """Evaluate ``and``/``or`` with short-circuit semantics.

    Missing values are treated as falsy in `and`/`or` (don't propagate as
    Missing — that would collapse `true or missing` to Missing). This matches
    user intent: missing as 'cannot confirm', which is falsy in propositional
    logic.
    """
    if isinstance(node.op, ast.And):
        result: Any = True
        for v in node.values:
            value = _eval_node(v, ctx)
            if value is Missing or not value:
                return False
            result = value
        return bool(result)
    if isinstance(node.op, ast.Or):
        for v in node.values:
            value = _eval_node(v, ctx)
            if value is Missing:
                continue
            if value:
                return True
        return False
    raise DSLError(f"Unhandled boolean operator: {type(node.op).__name__}")


def _eval_compare(node: ast.Compare, ctx: _EvalContext) -> Any:
    """Evaluate chained comparisons (a < b < c is two pairwise comparisons)."""
    left = _eval_node(node.left, ctx)
    for op, comparator in zip(node.ops, node.comparators):
        right = _eval_node(comparator, ctx)
        if left is Missing or right is Missing:
            return False
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
    return True


# =============================================================================
# DSL builtins
# =============================================================================


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
        return _builtin_called_before(ctx.state, *_check_arity("called_before", args, 2))
    if func_name == "called_after":
        return _builtin_called_after(ctx.state, *_check_arity("called_after", args, 2))
    if func_name == "call_count":
        return _builtin_call_count(ctx.state, *_check_arity("call_count", args, 1))
    if func_name == "last_call_was":
        return _builtin_last_call_was(ctx.state, *_check_arity("last_call_was", args, 1))
    if func_name == "calls":
        return _builtin_calls(ctx.state, *_check_arity("calls", args, 1))
    raise DSLError(f"Unknown DSL function: {func_name}")


def _check_arity(name: str, args: list[Any], expected: int) -> list[Any]:
    if len(args) != expected:
        raise DSLError(f"{name}() takes {expected} argument(s); got {len(args)}.")
    return args


def _builtin_contains(haystack: Any, needle: Any) -> bool:
    """True when ``needle`` is in ``haystack``. Strings: substring; iterables: membership.

    The substring half is refused over model prose where a template is AUTHORED, not here — see
    the module's *Text matches over model prose* — so a stored template evaluates as it always did.
    """
    if haystack is Missing or needle is Missing:
        return False
    try:
        return needle in haystack
    except TypeError as mismatch:
        raise DSLError(
            f"contains(): cannot test a {type(needle).__name__} in a {type(haystack).__name__}"
        ) from mismatch


def _builtin_intersects(a: Any, b: Any) -> bool:
    """True when sets-of-elements share at least one element."""
    if a is Missing or b is Missing:
        return False
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


# ---- Ordering predicates -----------------------------------------------------


def _all_calls_with_index(state: WorldState) -> list[tuple[int, str, str]]:
    """Return ``[(global_index, tool, action), ...]`` from ``state.global_calls``.

    Cross-tool ordering relies on the global ledger maintained by
    :meth:`WorldState.record_call`. Per-tool ``calls`` lists are
    intentionally NOT used as a fallback — they don't preserve cross-tool
    order and would silently degrade the ordering-predicate semantics.
    """
    return [(i, entry.get("tool", ""), entry.get("action", "")) for i, entry in enumerate(state.global_calls)]


def _parse_tool_action(spec: Any) -> tuple[str, str]:
    """Parse a TOOL.ACTION spec from the DSL.

    Accepts either a string like ``"inventory.place_order"`` or a dict
    ``{"tool": "inventory", "action": "place_order"}`` (the AST resolves
    ``inventory.place_order`` as Attribute(Name('inventory'), 'place_order') which
    is not a valid path — so the user writes it as a string literal).

    Returns:
        ``(tool, action)`` tuple.

    Raises:
        DSLError: When the spec doesn't decode to a 2-segment dotted name.
    """
    if isinstance(spec, str):
        if "." not in spec:
            raise DSLError(f"tool.action spec must be 'tool.action'; got {spec!r}.")
        tool, action = spec.split(".", 1)
        return tool, action
    raise DSLError(f"Expected a 'tool.action' string spec; got {type(spec).__name__}.")


def _builtin_called_before(state: WorldState, first: Any, second: Any) -> bool:
    """True when the first occurrence of ``first`` precedes the first occurrence of ``second``."""
    tool_a, action_a = _parse_tool_action(first)
    tool_b, action_b = _parse_tool_action(second)
    calls = _all_calls_with_index(state)
    first_idx_a = next((i for i, t, a in calls if t == tool_a and a == action_a), None)
    first_idx_b = next((i for i, t, a in calls if t == tool_b and a == action_b), None)
    if first_idx_a is None or first_idx_b is None:
        return False
    return first_idx_a < first_idx_b


def _builtin_called_after(state: WorldState, first: Any, second: Any) -> bool:
    """True when the last occurrence of ``first`` follows the last occurrence of ``second``."""
    tool_a, action_a = _parse_tool_action(first)
    tool_b, action_b = _parse_tool_action(second)
    calls = _all_calls_with_index(state)
    last_idx_a = next((i for i, t, a in reversed(calls) if t == tool_a and a == action_a), None)
    last_idx_b = next((i for i, t, a in reversed(calls) if t == tool_b and a == action_b), None)
    if last_idx_a is None or last_idx_b is None:
        return False
    return last_idx_a > last_idx_b


def _builtin_call_count(state: WorldState, spec: Any) -> int:
    """Return the count of recorded calls matching ``tool.action``."""
    tool, action = _parse_tool_action(spec)
    calls = _all_calls_with_index(state)
    return sum(1 for _, t, a in calls if t == tool and a == action)


def _builtin_last_call_was(state: WorldState, spec: Any) -> bool:
    """True when the most recent recorded call matches ``tool.action``."""
    tool, action = _parse_tool_action(spec)
    calls = _all_calls_with_index(state)
    if not calls:
        return False
    _, last_tool, last_action = calls[-1]
    return last_tool == tool and last_action == action


def _builtin_calls(state: WorldState, spec: Any) -> list[dict[str, Any]]:
    """Return the recorded parameters of every call matching ``tool.action``, in ledger order.

    Copies, so an expression cannot reach the ledger it reads.
    """
    tool, action = _parse_tool_action(spec)
    return [
        dict(entry.get("params", {}))
        for entry in state.global_calls
        if entry.get("tool") == tool and entry.get("action") == action
    ]


# ---- any() / all() generators ------------------------------------------------


def _eval_any_all(func_name: str, node: ast.Call, ctx: _EvalContext) -> bool:
    """Evaluate ``any(<body> for it in <path>)`` / ``all(...)``."""
    if len(node.args) != 1 or not isinstance(node.args[0], ast.GeneratorExp):
        raise DSLError(f"{func_name}() takes one generator expression argument.")
    genexp = node.args[0]
    gen = genexp.generators[0]
    iterable = _eval_node(gen.iter, ctx)
    if iterable is Missing:
        return False
    try:
        items = list(iterable)
    except TypeError as mismatch:
        raise DSLError(f"{func_name}(): cannot iterate a {type(iterable).__name__}") from mismatch

    aggregator = any if func_name == "any" else all
    body = genexp.elt

    def _eval_with_binding(item: Any) -> bool:
        previous = ctx.binding.get("it", _SENTINEL)
        ctx.binding["it"] = item
        try:
            value = _eval_node(body, ctx)
            if value is Missing:
                return False
            return bool(value)
        finally:
            if previous is _SENTINEL:
                ctx.binding.pop("it", None)
            else:
                ctx.binding["it"] = previous

    return aggregator(_eval_with_binding(it) for it in items)


_SENTINEL = object()


__all__ = [
    "DSLError",
    "ExtractedPaths",
    "Missing",
    "TextMatch",
    "TextPredicate",
    "call_parameter_matches",
    "evaluate",
    "evaluate_with_detail",
    "extract_paths",
    "extract_text_matches",
    "parse",
    "referenced_actions",
    "undefined_action",
    "undefined_call_references",
    "world_prose_matches",
]
