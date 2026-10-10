"""Goal-state DSL — small expression language for eval scoring.

Expressions are evaluated against a cell's end state (``state``: declared world dimension name →
value), its :class:`~threetears.evals.contracts.call_ledger.CallLedger` (read by the call
predicates), and a variation parameter dict (``variation``). The evaluation is three-valued: True,
False, or :data:`~threetears.evals.contracts.dsl.Missing` — *not established* — when it rests on a value the end state does not hold.
A check whose value is ``Missing`` is never a pass; the grader records it as failed with a detail
saying it was not established and naming the path that held nothing (see *Missing values* below).

Surface
-------

Path access (Python-style)::

    state.inventory.orders           # list
    state.inventory.orders.length    # len(...)
    state.inventory.orders[-1]       # indexing
    state.support.messages[-1].content
    variation.region_pair            # variation parameter

``state.<path>`` roots at a declared world dimension's NAME and nothing else, as the authoring gate
reads it: ``state.inbox.messages`` and a flat ``state.ingest_backlog`` both resolve through the
host's world registry, by the longest declared prefix, and whatever follows the dimension addresses
inside its value. A path naming no declared dimension is :data:`~threetears.evals.contracts.dsl.Missing`. There is no reading of a
raw layout: a host that declares no world has no ``state`` to read, and a path under it raises.

Comparisons::

    state.inventory.orders.length >= 1
    state.inventory.orders[0].sku == "SKU-42"
    variation.tone != "hostile"

Predicates (function-call form)::

    contains(state.inventory.orders, "SKU-42")              # value in path
    contains(state.inventory.categories, variation.category) # a case parameter in path
    intersects(state.inventory.categories, ["toys", "games"]) # non-empty intersection
    length(state.inventory.orders) >= 2                     # equivalent to .length

Ordering predicates (across all tools' recorded calls — the cell's call ledger, never world state)::

    called_before("inventory.search", "inventory.place_order")
    called_after("inventory.place_order", "inventory.cancel_order")
    call_count("inventory.place_order") >= 2
    last_call_was("inventory.place_order")

Each ordering predicate is False when either action never happened, so "never acted" reads like "acted in
the wrong order", and ``not called_before(a, b)`` holds for a candidate that did neither. A check about order
pairs it with ``call_count(a) >= 1``. ``last_call_was`` reads only the cell's final call, across the whole
cell, so a candidate that acted and then called something else answers False for the action it did take.

A deliberate pass (the engine's reserved ledger entry, recorded by ``CallLedger.record_pass``) is not a call:
none of the call builtins sees it, and ``passed()`` reads it, the same for every host::

    passed()
    not passed() and call_count("inventory.place_order") >= 1

``passed()`` holds when the cell recorded a pass and no call: a candidate that acted and then passed did not
pass. A cell that did nothing and recorded no pass did not pass either.

Call parameters (``calls()`` returns the matching calls' recorded parameters, in order)::

    calls("inventory.place_order").length >= 3
    any(it.priority == "rush" for it in calls("inventory.add_note"))
    all(it.note_text.length > 0 for it in calls("inventory.place_order"))

World events (``fired()`` reads which triggered dimensions fired during the cell, by the rig or in the
world — never world state, and never at t=0; ``fired_armed()`` reads only a firing of the event the
cell's seed armed, so the world's own firing on the same dimension does not satisfy it)::

    fired("inventory.restock_alarm")
    not fired("support.escalation")
    fired_armed("inventory.restock_alarm")

Generator predicates (``it`` binds to each element)::

    any(it.sku == "X" for it in state.inventory.orders)
    all(intersects(it.tags, ["toys", "games"]) for it in state.inventory.orders)

A case parameter (``variation.<name>``) is one string, as the case stores it, and never a list: a
case's variation parameters are a flat string map. So it is compared (``==``, ``!=``) or looked for
(``contains(state.<path>, variation.<name>)``), and never read as a collection — ``intersects`` over
one, ``contains`` searching one, a generator over one or an index into one is refused where the
expression is parsed. Write a set of values in the check itself, as a list literal.

Boolean composition::

    state.inventory.orders.length >= 1 and any(it.sku == "X" for it in state.inventory.orders)
    not state.support.messages[-1].content == ""

Missing values
--------------

A path that resolves to nothing — a dimension the end state does not hold, an index past the end,
a field a value does not carry — is :data:`~threetears.evals.contracts.dsl.Missing`: *unknown*, not *absent* and not *empty*. It
propagates by Kleene's three-valued logic, so negating a question never turns "unknown" into "yes":

* a comparison with a ``Missing`` side is ``Missing`` (and so is ``contains``/``intersects`` over one,
  and ``.length``/``length()`` of one);
* a list or tuple literal holding a ``Missing`` element is ``Missing`` (``[state.x]`` is not a
  one-element list when ``state.x`` resolved to nothing);
* ``fired_armed(...)`` is ``Missing`` for a cell whose firings cannot say which were armed — a
  witnessed cell, which no seed armed (:meth:`~threetears.evals.contracts.world_events.Firings.of`);
* ``not Missing`` is ``Missing``;
* ``a and b`` is False when any operand is False, else ``Missing`` when any is ``Missing``, else True;
  ``a or b`` is True when any operand is True, else ``Missing`` when any is ``Missing``, else False;
* ``any(...)``/``all(...)`` over a ``Missing`` iterable is ``Missing``; over elements, ``any`` is True
  when some element's body is True, else ``Missing`` when some is ``Missing``, and ``all`` the dual.

So ``not state.support.messages[-1].content == ""`` holds when the last message has content, fails
when it is empty, and is *not established* when there are no messages at all — exactly as
``state.support.messages[-1].content != ""`` is. A top-level ``Missing`` is never a pass:
:func:`evaluate` returns False for it and :func:`evaluate_with_detail` says it was not established.
Note that ``.length`` is always ``len()`` of the value, so a mapping field literally named
``length`` cannot be reached by path.

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
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

# The language's static half: parsing, the node allowlist, and the extraction an authoring gate and a
# stored template's validators read. It imports only ``ast``, so a stored shape validates an expression
# without reaching the evaluator (threetears.evals.contracts.dsl), which this module's docstring documents
# alongside the grammar because a reader learns the language as one thing.


class DSLError(ValueError):
    """Raised when a DSL expression is malformed or disallowed."""


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
        "fired",
        "fired_armed",
        "passed",
        "any",
        "all",
    }
)


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
    called = {id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in _BUILTINS and id(node) not in called:
                # A builtin is not a value: used as one it would parse here and fail only at evaluation.
                raise DSLError(f"{node.id} is a DSL function, not a value: call it ({node.id}(...)).")
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
            if node.func.id in _ACTION_BUILTINS and not (
                node.args
                and all(
                    isinstance(arg, ast.Constant) and isinstance(arg.value, str) and "." in arg.value
                    for arg in node.args
                )
            ):
                # A computed spec would leave the authoring gate unable to name the action — to ask
                # whether the host defines it, or, for calls(), whose schema says which parameters are
                # free text — so a typo would score False on every trial.
                raise DSLError(f"{node.func.id}() takes 'tool.action' string literals, never a computed spec.")
            if node.func.id in FIRE_PREDICATES and _fired_name(node) is None:
                # A computed name would leave the authoring gate unable to ask whether the dimension is a
                # triggered one this host declares, so a typo would score False on every trial.
                raise DSLError(f"{node.func.id}() takes one dimension name as a string literal.")
        if isinstance(node, ast.GeneratorExp):
            if len(node.generators) != 1:
                raise DSLError("any/all generators must have exactly one 'for' clause.")
            gen = node.generators[0]
            if gen.ifs:
                raise DSLError("any/all generators may not have 'if' filters in this DSL.")
            if not (isinstance(gen.target, ast.Name) and gen.target.id == "it"):
                raise DSLError("any/all generators must bind to the variable 'it'.")
        _refuse_a_variation_read_as_a_collection(node)


def _variation_path(node: ast.AST) -> str | None:
    """The source of ``node`` when it is a path under ``variation`` (``variation.x``, ``variation.x[0]``), else None."""
    base = node
    while isinstance(base, (ast.Attribute, ast.Subscript)):
        base = base.value
    if isinstance(base, ast.Name) and base.id == "variation" and base is not node:
        return ast.unparse(node)
    return None


def _refuse_a_variation_read_as_a_collection(node: ast.AST) -> None:
    """Refuse an expression that reads a case parameter as a collection of elements.

    A case stores every variation parameter as ONE string (``EvalTestCase.variation_params`` is a
    flat string map), so a parameter is never a list, whatever it spells. Read as a collection it
    reads wrong without a word: ``intersects`` takes a string as a single element, so a parameter
    spelling several categories (``"a,b"``, or JSON) never intersects anything; ``contains`` over one
    is a substring test on its spelling, so ``"a"`` is found in ``'["ab"]'``; iterating or indexing
    one walks its characters. A control end state may still state a real list, so a check of this
    shape could be proven on its control and then fail every case — refused where it is parsed instead.

    Raises:
        DSLError: ``intersects`` with a variation operand, ``contains`` with a variation haystack, a
            generator over a variation path, or an index into one.
    """
    remedy = (
        "a case stores every variation parameter as one string, never a list, so it cannot be read as a "
        'collection; write the set in the check as a list literal (intersects(it.tags, ["kitchen", "garden"])), '
        "or test one parameter's value with contains(state.<path>, variation.<name>) or =="
    )
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "intersects":
            read = next((path for arg in node.args if (path := _variation_path(arg)) is not None), None)
            if read is not None:
                raise DSLError(f"intersects() over {read}: {remedy}.")
        if node.func.id == "contains" and node.args and (read := _variation_path(node.args[0])) is not None:
            raise DSLError(f"contains() with {read} as what is searched: {remedy}.")
    if isinstance(node, ast.comprehension) and (read := _variation_path(node.iter)) is not None:
        raise DSLError(f"a generator over {read}: {remedy}.")
    if isinstance(node, ast.Subscript) and (read := _variation_path(node.value)) is not None:
        raise DSLError(f"an index into {read}: {remedy}.")


def speaks_the_goal_language(expression: str) -> bool:
    """Whether ``expression`` is written in this language's words, whether or not today's rules admit it.

    True when it is Python expression syntax naming no identifier but the language's roots and
    builtins. A re-check asks this of a stored outcome :func:`parse` refuses, to tell a goal check
    written under an older, looser rule — which it must name as refused today — from a fact a kind
    computed and reported in its own words (``field_accuracy >= 0.92``), which was never a goal check.

    Args:
        expression: A stored outcome's expression.

    Returns:
        Whether it names only the language's own words.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return False
    return all(
        node.id in _ALLOWED_ROOTS or node.id in _BUILTINS for node in ast.walk(tree) if isinstance(node, ast.Name)
    )


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
TextOperand = tuple[str | None, tuple[str, ...]]


def _calls_spec(node: ast.AST) -> str | None:
    """The literal ``tool.action`` of a ``calls(...)`` node, or None when ``node`` is not one."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "calls"):
        return None
    if len(node.args) != 1 or node.keywords:
        return None
    (arg,) = node.args
    return arg.value if isinstance(arg, ast.Constant) and isinstance(arg.value, str) else None


def _match(node: ast.AST, operand: TextOperand, predicate: TextPredicate) -> TextMatch | None:
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


def _collect_text_matches(node: ast.AST, it_binding: TextOperand | None, found: list[TextMatch]) -> None:
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
            iterable = text_operand(generator.generators[0].iter, it_binding)
            _collect_text_matches(generator.generators[0].iter, it_binding, found)
            element = (iterable[0], (*iterable[1], "[]")) if iterable is not None else None
            _collect_text_matches(generator.elt, element, found)
            return
        if name == "contains" and node.args:
            operand = text_operand(node.args[0], it_binding)
            if operand is not None and (match := _match(node, operand, "contains")) is not None:
                found.append(match)
        elif name == "intersects":
            for argument in node.args:
                operand = text_operand(argument, it_binding)
                if operand is not None and (match := _match(node, operand, "intersects")) is not None:
                    found.append(match)
    elif isinstance(node, ast.Compare):
        sides = [node.left, *node.comparators]
        for op, (left, right) in zip(node.ops, itertools.pairwise(sides), strict=True):
            if not isinstance(op, (ast.Eq, ast.NotEq)):
                continue
            for subject, other in ((left, right), (right, left)):
                operand = text_operand(subject, it_binding)
                if operand is not None and _is_text_comparand(other, it_binding):
                    if (match := _match(node, operand, "equality")) is not None:
                        found.append(match)
    for child in ast.iter_child_nodes(node):
        _collect_text_matches(child, it_binding, found)


def text_operand(node: ast.AST, it_binding: TextOperand | None) -> TextOperand | None:
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


def _is_text_comparand(node: ast.AST, it_binding: TextOperand | None) -> bool:
    """Whether an equality's other side makes it a comparison of text rather than a count or emptiness.

    Args:
        node: The other side of ``==``/``!=``.
        it_binding: What ``it`` stands for, if bound.

    Returns:
        True for a non-empty string literal, a ``variation`` path, or another state path.
    """
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str) and node.value != ""
    if text_operand(node, it_binding) is not None:
        return True
    base = node
    while isinstance(base, (ast.Attribute, ast.Subscript)):
        base = base.value
    return isinstance(base, ast.Name) and base.id == "variation" and base is not node


#: The builtins whose string arguments name a ``tool.action``.
_ACTION_BUILTINS = frozenset({"called_before", "called_after", "call_count", "last_call_was", "calls"})


#: Every builtin that reads the cell's call ledger: the action builtins, and ``passed()``.
_LEDGER_BUILTINS = _ACTION_BUILTINS | {"passed"}


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
                pair = parse_tool_action(argument.value)
                if pair not in named:
                    named.append(pair)
    return tuple(named)


def reads_call_ledger(expression: str) -> bool:
    """Whether a goal check reads the cell's call ledger at all — through any call builtin, or ``passed()``.

    Args:
        expression: A goal-state expression.

    Returns:
        Whether it calls ``called_before``, ``called_after``, ``call_count``, ``last_call_was``, ``calls`` or
        ``passed``.

    Raises:
        DSLError: The expression does not parse.
    """
    return any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _LEDGER_BUILTINS
        for node in ast.walk(parse(expression).body)
    )


#: The predicates that read a cell's world events, each naming one triggered dimension as a string literal:
#: ``fired`` for any firing, ``fired_armed`` for a firing of the event the cell's seed armed.
FIRE_PREDICATES = frozenset({"fired", "fired_armed"})


def _fired_name(node: ast.Call) -> str | None:
    """The dimension a ``fired()`` or ``fired_armed()`` call names, or None when it names none as one string literal.

    Args:
        node: A ``fired`` or ``fired_armed`` call.

    Returns:
        The name.
    """
    if len(node.args) != 1 or node.keywords:
        return None
    (argument,) = node.args
    if isinstance(argument, ast.Constant) and isinstance(argument.value, str) and argument.value:
        return argument.value
    return None


def referenced_fires(expression: str) -> tuple[str, ...]:
    """Every dimension a goal check names through ``fired()`` or ``fired_armed()``, in source order, each once.

    The one walk over what a check reads of a cell's world events: the authoring gate asks whether each
    name is a triggered dimension the host declares (:func:`undefined_fire_references`), and a
    precondition refuses any — at t=0 nothing has fired.

    Args:
        expression: A goal-state expression.

    Returns:
        The named dimensions.

    Raises:
        DSLError: The expression does not parse.
    """
    calls = [
        node
        for node in ast.walk(parse(expression).body)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FIRE_PREDICATES
    ]
    named: list[str] = []
    # Sorted by position: ``ast.walk`` is breadth-first, so a name nested deeper would otherwise be
    # reported after a shallower one that comes later in the text.
    for node in sorted(calls, key=lambda call: (call.lineno, call.col_offset)):
        name = _fired_name(node)
        if name is not None and name not in named:
            named.append(name)
    return tuple(named)


def parse_tool_action(spec: Any) -> tuple[str, str]:
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


__all__ = [
    "FIRE_PREDICATES",
    "TextOperand",
    "parse_tool_action",
    "text_operand",
    "DSLError",
    "ExtractedPaths",
    "TextMatch",
    "TextPredicate",
    "extract_paths",
    "extract_text_matches",
    "parse",
    "reads_call_ledger",
    "referenced_actions",
    "referenced_fires",
    "speaks_the_goal_language",
]
