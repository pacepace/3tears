"""Unit tests for the goal-state DSL."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

import threetears.evals.kernel.dsl as _dsl
import threetears.evals.schema.goal_grammar as _grammar
from threetears.evals.schema import Firings, Precondition, WorldEvent
from threetears.evals.kernel.dsl import NOT_ESTABLISHED, Missing, evaluate, evaluate_with_detail
from threetears.evals.schema.goal_grammar import DSLError, extract_paths, parse
from threetears.evals.kernel.host import Triggered, WorldDimension, WorldRegistry
from threetears.evals.schema.call_ledger import CallLedger


def _state_with(**dimensions: dict[str, Any]) -> dict[str, Any]:
    """The keyword arguments :func:`evaluate` takes for a world holding ``dimensions``, and an empty ledger.

    Each keyword is one declared dimension whose value is an object, so ``state.shop.cart`` names the
    ``shop`` dimension and addresses ``cart`` inside its value — the longest-declared-prefix rule a
    real host's world resolves by. Calls go on ``["ledger"]``.

    Args:
        **dimensions: Dimension name -> its value.

    Returns:
        ``end_state``, ``ledger`` and ``world``, ready to splat into :func:`evaluate`.
    """
    world = WorldRegistry(
        [
            WorldDimension(
                name=name,
                carrier="test",
                schema={"type": "object"},
                matters="a DSL test presumes this value",
                seed="h.seed",
                read="h.read",
            )
            for name in dimensions
        ],
        bindings={"h.seed": lambda value: None, "h.read": dict},
    )
    return {
        "end_state": {name: dict(value) for name, value in dimensions.items()},
        "ledger": CallLedger(),
        "world": world,
    }


# =============================================================================
# Path access
# =============================================================================


def test_path_resolves_namespace_dict():
    state = _state_with(shop={"cart": [1, 2, 3]})
    assert evaluate("state.shop.cart.length == 3", **state) is True


def test_path_with_subscript_positive_index():
    state = _state_with(shop={"cart": [{"title": "A"}, {"title": "B"}]})
    assert evaluate('state.shop.cart[0].title == "A"', **state) is True


def test_path_with_subscript_negative_index():
    state = _state_with(chat={"messages": [{"content": "hi"}, {"content": "thanks"}]})
    assert evaluate('state.chat.messages[-1].content == "thanks"', **state) is True


# =============================================================================
# Paths root at a dimension name, whatever a host names its dimensions
# =============================================================================


def _shop_world(*, composed: bool) -> WorldRegistry:
    """A two-carrier world named one of the two ways a host may name it.

    Flat: ``stock`` and ``float``, each key naming its dimension as it is. Composed: ``shelf.stock`` and
    ``till.float``, the carrier composed in by the addressing the registry declares.

    Args:
        composed: Whether the names compose the carrier in.

    Returns:
        The registry.
    """

    def named(carrier: str, key: str) -> str:
        return f"{carrier}.{key}" if composed else key

    def dimension(carrier: str, key: str) -> WorldDimension:
        return WorldDimension(
            name=named(carrier, key),
            carrier=carrier,
            schema={"type": "integer"},
            matters="a shop scenario presumes what is on the shelf and in the till",
            seed="h.seed",
            read="h.read",
        )

    return WorldRegistry(
        [dimension("shelf", "stock"), dimension("till", "float")],
        bindings={"h.seed": lambda value: None, "h.read": lambda: 0},
        address=named if composed else None,
    )


#: The same world, laid out as a seed is: carrier -> key -> value, for either naming.
_SHOP = {"shelf": {"stock": 3}, "till": {"float": 20}}


@pytest.mark.parametrize(
    ("composed", "stock", "till_float"),
    [(False, "state.stock", "state.float"), (True, "state.shelf.stock", "state.till.float")],
    ids=["flat names", "composed names"],
)
def test_a_path_names_the_dimension_however_the_host_names_it(composed, stock, till_float):
    """``state.<dimension>`` reads the value the end state holds under that dimension's name."""
    world = _shop_world(composed=composed)
    end_state = world.named(_SHOP)

    assert evaluate(f"{stock} == 3 and {till_float} == 20", end_state=end_state, ledger=CallLedger(), world=world)
    assert evaluate_with_detail(f"{stock} > 2", end_state=end_state, ledger=CallLedger(), world=world) == (True, "True")


def test_a_path_inside_a_flat_dimension_s_value_resolves_below_it():
    """The rest of the path addresses inside the value, as it does below a composed name."""
    world = WorldRegistry(
        [
            WorldDimension(
                name="corrections",
                carrier="console",
                schema={"type": "array", "items": {"type": "string"}},
                matters="a deference scenario presumes a correction is in force",
                seed="h.seed",
                read="h.read",
            )
        ],
        bindings={"h.seed": lambda value: None, "h.read": list},
    )
    cell = {"end_state": {"corrections": ["reprice", "void"]}, "ledger": CallLedger(), "world": world}

    assert evaluate("state.corrections.length == 2", **cell) is True
    assert evaluate('state.corrections[-1] == "void"', **cell) is True
    assert evaluate('any(it == "reprice" for it in state.corrections)', **cell) is True


@pytest.mark.parametrize("composed", [False, True], ids=["flat names", "composed names"])
def test_a_dimension_the_world_does_not_declare_or_the_end_state_does_not_hold_is_missing(composed):
    """Undeclared or absent — each is Missing, never another dimension's value."""
    world = _shop_world(composed=composed)
    stock = "state.shelf.stock" if composed else "state.stock"
    full = world.named(_SHOP)

    assert (
        evaluate("state.shelf.mood == 1 or state.mood == 1", end_state=full, ledger=CallLedger(), world=world) is False
    )
    # A key the end state holds that no dimension declares is unreachable: the path resolves through
    # the registry, never through the end state's own keys.
    assert evaluate("state.mood == 1", end_state={**full, "mood": 1}, ledger=CallLedger(), world=world) is False
    assert evaluate(f"{stock} == 3", end_state={}, ledger=CallLedger(), world=world) is False


def test_a_state_path_on_a_host_that_declares_no_world_raises():
    """No layout is read in place of a registry: a worldless host has no dimension for a path to name."""
    with pytest.raises(DSLError, match="declares no world"):
        evaluate("state.shop.cart.length >= 1", end_state={"shop": {"cart": [1]}}, ledger=CallLedger(), world=None)


def test_a_check_reading_only_calls_and_the_case_needs_no_world():
    """The ledger and the case are not world state, so a worldless host grades them."""
    ledger = CallLedger()
    ledger.record("shop", "search", {"query": "linen"})

    assert evaluate(
        'call_count("shop.search") == 1 and variation.tone == "casual"',
        end_state={},
        ledger=ledger,
        world=None,
        variation={"tone": "casual"},
    )


@pytest.mark.parametrize("expression", ["length(state) == 0", 'state["shop"].length == 1', "not state"])
def test_state_alone_is_not_a_value(expression):
    """``state`` names no container of every dimension: a check reads one by name, as the gate resolves it."""
    with pytest.raises(DSLError, match="'state' is not a value"):
        evaluate(expression, **_state_with(shop={"cart": []}))


class TestNamingASeedShapedWorld:
    """``WorldRegistry.named`` turns a seed-shaped world into the name-keyed end state a check reads."""

    def test_each_key_is_named_through_the_host_s_addressing(self):
        assert _shop_world(composed=True).named(_SHOP) == {"shelf.stock": 3, "till.float": 20}
        assert _shop_world(composed=False).named(_SHOP) == {"stock": 3, "float": 20}

    def test_the_values_are_copies(self):
        seed = {"shelf": {"stock": [1]}}
        world = WorldRegistry(
            [
                WorldDimension(
                    name="stock",
                    carrier="shelf",
                    schema={"type": "array"},
                    matters="a shop scenario presumes what is on the shelf",
                    seed="h.seed",
                    read="h.read",
                )
            ],
            bindings={"h.seed": lambda value: None, "h.read": list},
        )
        world.named(seed)["stock"].append(2)
        assert seed == {"shelf": {"stock": [1]}}

    @pytest.mark.parametrize(
        ("namespaces", "refusal"),
        [
            ({"shelf": {"mood": 1}}, "addresses no dimension"),
            ({"till": {"stock": 3}}, "but 'shelf' supplies it"),
            ({"shelf": 3}, "is not a mapping"),
        ],
        ids=["undeclared", "misplaced", "malformed"],
    )
    def test_a_value_no_read_could_return_is_refused(self, namespaces, refusal):
        with pytest.raises(ValueError, match=refusal):
            _shop_world(composed=False).named(namespaces)


def test_variation_resolves_from_variation_dict():
    state = _state_with()
    assert (
        evaluate(
            'variation.tone == "casual"',
            **state,
            variation={"tone": "casual"},
        )
        is True
    )


def test_missing_namespace_resolves_to_missing_and_comparison_returns_false():
    state = _state_with()
    # No shop namespace at all
    assert evaluate("state.shop.cart.length >= 1", **state) is False


def test_missing_index_resolves_to_missing():
    state = _state_with(shop={"cart": []})
    assert evaluate('state.shop.cart[0].title == "X"', **state) is False


def test_missing_dict_key_resolves_to_missing():
    state = _state_with(shop={"cart": []})
    # No 'catalog' key
    assert evaluate('state.shop.catalog[0].title == "X"', **state) is False


# =============================================================================
# Comparisons
# =============================================================================


def test_int_equality():
    state = _state_with(shop={"cart": [1]})
    assert evaluate("state.shop.cart.length == 1", **state) is True


def test_int_ordering_operators():
    state = _state_with(shop={"cart": [1, 2, 3]})
    assert evaluate("state.shop.cart.length >= 2", **state) is True
    assert evaluate("state.shop.cart.length > 3", **state) is False
    assert evaluate("state.shop.cart.length <= 3", **state) is True
    assert evaluate("state.shop.cart.length < 3", **state) is False
    assert evaluate("state.shop.cart.length != 3", **state) is False


def test_string_comparison():
    state = _state_with(shop={"cart": [{"title": "Blue Kettle"}]})
    assert evaluate('state.shop.cart[0].title != "Other"', **state) is True


def test_comparison_type_mismatch_raises_rather_than_failing_the_check():
    """An uncomparable pair is a fault in the template or the rig, not the candidate failing it.

    Raised, so the runner excludes the cell; a False here was scored as a failed goal.
    """
    state = _state_with(shop={"cart": [1]})
    with pytest.raises(DSLError, match="cannot compare int with str"):
        evaluate('state.shop.cart[0] < "abc"', **state)
    # The positive case beside it: comparable values still compare.
    assert evaluate("state.shop.cart[0] < 2", **state) is True


@pytest.mark.parametrize(
    "expression",
    [
        "contains(state.shop.count, 1)",
        "intersects(state.shop.count, state.shop.count)",
        "any(it == 1 for it in state.shop.count)",
    ],
)
def test_a_builtin_over_a_value_of_the_wrong_type_raises(expression):
    """contains / intersects / any over an int: the world or the template is wrong, not the candidate."""
    with pytest.raises(DSLError):
        evaluate(expression, **_state_with(shop={"count": 3}))


# =============================================================================
# contains / intersects / length builtins
# =============================================================================


def test_contains_list_membership():
    state = _state_with(shop={"cart": ["a", "b", "c"]})
    assert evaluate('contains(state.shop.cart, "b")', **state) is True
    assert evaluate('contains(state.shop.cart, "z")', **state) is False


def test_contains_string_substring():
    state = _state_with(chat={"messages": [{"content": "thank you very much"}]})
    assert (
        evaluate(
            'contains(state.chat.messages[-1].content, "thank")',
            **state,
        )
        is True
    )


def test_intersects_lists():
    state = _state_with(shop={"cart": ["kitchen", "garden"]})
    assert evaluate('intersects(state.shop.cart, ["kitchen", "outdoor"])', **state) is True
    assert evaluate('intersects(state.shop.cart, ["toys"])', **state) is False


def test_intersects_missing_path_returns_false():
    state = _state_with()
    assert evaluate('intersects(state.shop.cart, ["kitchen"])', **state) is False


def test_a_case_parameter_is_looked_for_as_one_value():
    """A parameter is one string, as a case stores it, so it is the needle, never the haystack."""
    state = _state_with(shop={"cart": ["kitchen", "garden"]})
    assert evaluate("contains(state.shop.cart, variation.category)", **state, variation={"category": "kitchen"})
    assert not evaluate("contains(state.shop.cart, variation.category)", **state, variation={"category": "toys"})


@pytest.mark.parametrize(
    ("expression", "said"),
    [
        ("intersects(state.shop.cart, variation.categories)", r"intersects\(\) over variation.categories"),
        ("intersects(variation.categories, state.shop.cart)", r"intersects\(\) over variation.categories"),
        ("any(intersects(it.tags, variation.target) for it in state.shop.cart)", r"over variation.target"),
        ('contains(variation.categories, "kitchen")', r"contains\(\) with variation.categories as what is searched"),
        ('any(it == "kitchen" for it in variation.categories)', "a generator over variation.categories"),
        ('variation.categories[0] == "kitchen"', r"an index into variation.categories"),
    ],
)
def test_a_case_parameter_read_as_a_collection_is_refused_where_it_is_parsed(expression, said):
    """A case stores every parameter as one string (#665): read as a collection it never meant what it says.

    ``intersects`` took the string as ONE element, so a parameter spelling several categories never
    intersected anything; ``contains`` searched its spelling; a generator or an index walked its
    characters. Each evaluated False (or worse, True by accident) with no error, so it is refused.
    """
    with pytest.raises(DSLError, match=said) as refused:
        parse(expression)
    assert "never a list" in str(refused.value)
    for stored in ("kitchen,garden", '["garden", "kitchen"]'):
        with pytest.raises(DSLError):
            evaluate(
                expression,
                **_state_with(shop={"cart": ["kitchen"]}),
                variation={"categories": stored, "target": stored},
            )


def test_length_function_form_matches_attribute_form():
    state = _state_with(shop={"cart": [1, 2, 3]})
    assert evaluate("length(state.shop.cart) == 3", **state) is True
    assert evaluate("state.shop.cart.length == 3", **state) is True


# =============================================================================
# Boolean composition
# =============================================================================


def test_and_short_circuits_on_missing():
    state = _state_with(shop={"cart": []})
    # First clause is true; second's path is missing → overall false.
    assert (
        evaluate(
            "state.shop.cart.length == 0 and state.chat.messages.length >= 1",
            **state,
        )
        is False
    )


def test_or_ignores_missing_in_favor_of_truthy_sibling():
    state = _state_with(shop={"cart": ["x"]})
    assert (
        evaluate(
            "state.chat.messages.length >= 1 or state.shop.cart.length >= 1",
            **state,
        )
        is True
    )


def test_not_inverts_truthy():
    state = _state_with(shop={"cart": []})
    assert evaluate("not state.shop.cart.length >= 1", **state) is True


# =============================================================================
# Missing values — three-valued, so negation never turns "unknown" into a pass
# =============================================================================


def _chat_absent() -> dict[str, Any]:
    """A world declaring ``shop`` and ``chat`` whose end state holds ``shop`` and not ``chat``.

    The shape a dimension whose carrier was not attached leaves: declared, so every path below
    resolves, and absent from the end state, so ``state.chat.*`` is Missing.
    """
    state = _state_with(shop={"cart": ["x"], "items": [{"q": 1}, {"p": 1}]}, chat={})
    del state["end_state"]["chat"]
    return state


def _chat_present() -> dict[str, Any]:
    """The same world, with ``chat`` held: the positive half of every negated case below."""
    return _state_with(
        shop={"cart": ["x"], "items": [{"q": 1}, {"p": 1}]},
        chat={"messages": [], "status": "closed", "tags": ["y"]},
    )


#: Every negated shape the language has, each TRUE over ``_chat_present`` — so a case that reads False over
#: ``_chat_absent`` is failing for the missing value and not for the expression.
_NEGATED_FORMS = [
    "not state.chat.messages.length >= 1",
    'not state.chat.status == "open"',
    'not not state.chat.status == "closed"',
    'not (state.chat.status == "open" and state.shop.cart.length == 1)',
    'not (state.chat.status == "open" or state.shop.cart.length == 5)',
    'not contains(state.chat.tags, "x")',
    'not intersects(state.chat.tags, ["x"])',
    "not length(state.chat.tags) >= 2",
    'not any(it == "x" for it in state.chat.tags)',
    'not all(it == "x" for it in state.chat.tags)',
    'state.chat.status != "open"',
    # A list literal holding a missing element is itself missing, so building one around a path
    # does not launder the unknown into an answer.
    'not contains([state.chat.status], "open")',
    'not intersects([state.chat.status], ["open"])',
    'not ([state.chat.status] == ["open"])',
    'not ((state.chat.status, 1) == ("open", 1))',
]


@pytest.mark.parametrize("expression", _NEGATED_FORMS)
def test_a_negated_check_over_a_missing_value_is_not_established(expression: str) -> None:
    """``not`` keeps an unknown unknown: the check holds over the world that has the value, and over the
    world that does not it is never a pass — it is failed as *not established*, naming what was missing.

    Read as False, a comparison against a missing value made its negation True, so a check scored a
    candidate that did nothing as passing.
    """
    assert evaluate(expression, **_chat_present()) is True

    assert evaluate(expression, **_chat_absent()) is False
    passed, detail = evaluate_with_detail(expression, **_chat_absent())
    assert passed is False
    assert detail.startswith(_dsl.NOT_ESTABLISHED)
    assert "state.chat." in detail


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        # `and`: a False operand decides, whatever else is unknown; otherwise unknown wins over True.
        ('state.chat.status == "open" and state.shop.cart.length == 5', False),
        ('state.shop.cart.length == 5 and state.chat.status == "open"', False),
        ('state.chat.status == "open" and state.shop.cart.length == 1', None),
        # `or`: the dual.
        ('state.chat.status == "open" or state.shop.cart.length == 1', True),
        ('state.shop.cart.length == 1 or state.chat.status == "open"', True),
        ('state.chat.status == "open" or state.shop.cart.length == 5', None),
        # A chain is the `and` of its pairs.
        ("5 < state.shop.cart.length < state.chat.count", False),
        ("0 < state.shop.cart.length < state.chat.count", None),
        # any() is the `or` over elements, all() the `and`; an element without the field is unknown.
        ("any(it.p == 1 for it in state.shop.items)", True),
        ("any(it.p == 2 for it in state.shop.items)", None),
        ("all(it.p == 2 for it in state.shop.items)", False),
        ("all(it.p == 1 for it in state.shop.items)", None),
        ("any(it == 1 for it in state.chat.tags)", None),
        ("all(it == 1 for it in state.chat.tags)", None),
    ],
)
def test_each_connective_decides_by_kleene_logic(expression: str, expected: bool | None) -> None:
    """A deciding operand settles the expression; otherwise a missing operand leaves it unknown (None here)."""
    passed, detail = evaluate_with_detail(expression, **_chat_absent())

    if expected is None:
        assert passed is False
        assert detail.startswith(_dsl.NOT_ESTABLISHED)
    else:
        assert passed is expected
        assert detail == repr(expected)


def test_not_established_names_a_missing_variation_and_an_index_past_the_end() -> None:
    """The detail names whatever resolved to nothing, not only state paths."""
    state = _chat_present()

    _, by_variation = evaluate_with_detail("not variation.tone == 'hostile'", **state)
    _, by_index = evaluate_with_detail('not state.chat.messages[0] == "hi"', **state)

    assert by_variation == f"{_dsl.NOT_ESTABLISHED}: variation.tone resolved to nothing"
    assert by_index == f"{_dsl.NOT_ESTABLISHED}: state.chat.messages[0] resolved to nothing"


def test_a_negated_presumption_over_a_value_the_seeded_world_lacks_does_not_hold() -> None:
    """A precondition goes through the same evaluator: an unknown is never a presumption the world satisfied."""
    from packages.evals.tests.factories import make_template, make_test_case
    from threetears.evals.run import assert_preconditions

    template = make_template(
        preconditions=[Precondition(expression='not state.chat.status == "open"', presumes="the chat is not open")]
    )
    absent = _chat_absent()
    present = _chat_present()

    (failed,) = assert_preconditions(template, make_test_case(), absent["end_state"], world=absent["world"])
    assert failed.held is False and failed.detail.startswith(_dsl.NOT_ESTABLISHED)
    assert assert_preconditions(template, make_test_case(), present["end_state"], world=present["world"]) == []


def test_grading_records_a_not_established_check_as_failed_and_says_so() -> None:
    """The grader every kind uses: never a pass, and a detail telling it from a check that evaluated False."""
    from threetears.evals.run import grade_goal_checks

    absent = _chat_absent()
    unknown, false = grade_goal_checks(
        ['not state.chat.status == "open"', "state.shop.cart.length == 5"],
        ledger=absent["ledger"],
        end_state=absent["end_state"],
        fired=None,
        variation={},
        world=absent["world"],
    )

    assert (unknown.passed, unknown.detail) == (False, f"{_dsl.NOT_ESTABLISHED}: state.chat.status resolved to nothing")
    assert (false.passed, false.detail) == (False, "False")


# =============================================================================
# Ordering predicates (read the call ledger)
# =============================================================================


def test_called_before_finds_first_occurrences():
    state = _state_with()
    state["ledger"].record("shop", "search", {"query": "kitchen"})
    state["ledger"].record("shop", "add_item", {"item_ref": "r1"})
    state["ledger"].record("chat", "send_message", {"content": "added!"})

    assert evaluate('called_before("shop.search", "shop.add_item")', **state) is True
    assert evaluate('called_before("shop.add_item", "shop.search")', **state) is False


def test_called_before_cross_tool():
    state = _state_with()
    state["ledger"].record("shop", "add_item", {"item_ref": "r1"})
    state["ledger"].record("chat", "send_message", {"content": "added"})

    assert evaluate('called_before("shop.add_item", "chat.send_message")', **state) is True


def test_called_before_returns_false_if_either_missing():
    state = _state_with()
    state["ledger"].record("shop", "search", {})

    assert evaluate('called_before("shop.search", "shop.add_item")', **state) is False
    assert evaluate('called_before("shop.add_item", "shop.search")', **state) is False


def test_called_after_uses_last_occurrences():
    state = _state_with()
    state["ledger"].record("shop", "search", {})
    state["ledger"].record("shop", "add_item", {})
    state["ledger"].record("shop", "search", {})  # second search

    # last search (index 2) comes after last add_item (index 1)
    assert evaluate('called_after("shop.search", "shop.add_item")', **state) is True


def test_call_count_returns_int_for_comparison():
    state = _state_with()
    state["ledger"].record("shop", "add_item", {})
    state["ledger"].record("shop", "add_item", {})
    state["ledger"].record("shop", "add_item", {})

    assert evaluate('call_count("shop.add_item") >= 2', **state) is True
    assert evaluate('call_count("shop.add_item") == 3', **state) is True
    assert evaluate('call_count("shop.skip") == 0', **state) is True


# =============================================================================
# Restraint — the inverted goal-state check
#
# A template that scores a subject for NOT reaching for a tool asserts the
# opposite of the usual check. These pin the three properties that shape makes
# load-bearing: it is true of a state that did other things, false the moment
# the tool fires, and — the one that matters — true of a state where nothing
# happened at all. That last property is why an inverted check must never be a
# template's only check. Those three properties are pinned against a host's own shipped
# template and transcript renderer, so they live in that host's suite; what the DSL itself
# owes is below.
# =============================================================================


def test_negated_and_equality_restraint_forms_agree():
    """`not call_count(...) >= 1` and `call_count(...) == 0` are interchangeable.

    Both parse and both evaluate; the equality form is the one the committed
    template uses because it reads as a statement about the count rather than as
    a negation of a threshold, but nothing in the DSL prefers it.
    """
    quiet = _state_with()
    quiet["ledger"].record("chat", "send_message", {"content": "linen is nostalgia"})
    busy = _state_with()
    busy["ledger"].record("lookup", "lookup", {"query": "linen"})

    for state in (quiet, busy):
        assert evaluate('call_count("lookup.lookup") == 0', **state) == evaluate(
            'not call_count("lookup.lookup") >= 1', **state
        )


def test_last_call_was_matches_most_recent():
    state = _state_with()
    state["ledger"].record("shop", "search", {})
    state["ledger"].record("shop", "add_item", {})
    state["ledger"].record("chat", "send_message", {})

    assert evaluate('last_call_was("chat.send_message")', **state) is True
    assert evaluate('last_call_was("shop.add_item")', **state) is False


def test_last_call_was_empty_state_false():
    state = _state_with()
    assert evaluate('last_call_was("shop.search")', **state) is False


def test_ordering_predicate_rejects_unparseable_spec():
    state = _state_with()
    state["ledger"].record("shop", "search", {})

    with pytest.raises(DSLError):
        evaluate('called_before("not_a_tool_action", "shop.search")', **state)


@pytest.mark.parametrize(
    "expression",
    [
        "called_before(state.shop, state.chat)",
        'called_before(variation.a, "shop.search")',
        'called_after("shop.search", variation.b)',
        "call_count(state.shop) >= 1",
        "last_call_was(variation.a)",
        "calls(variation.a).length >= 1",
        'called_before(1, "shop.search")',
        "last_call_was([1, 2, 3])",
    ],
)
def test_every_call_builtin_refuses_a_computed_spec_at_parse(expression: str) -> None:
    """A spec that is not a ``'tool.action'`` literal is refused where the template is written.

    The authoring gate names the actions a check reads from its literals alone, so a computed spec
    would leave it unable to ask whether the host defines the action, and a typo would score False on
    every trial instead of being refused.
    """
    with pytest.raises(DSLError, match="never a computed spec"):
        parse(expression)


@pytest.mark.parametrize("expression", ["contains", "contains == 1", "length(calls)", "not fired"])
def test_a_builtin_used_as_a_value_is_refused_at_parse(expression: str) -> None:
    """A builtin named outside a call parses as nothing evaluable, so it is refused at parse, by name."""
    with pytest.raises(DSLError, match="is a DSL function, not a value"):
        parse(expression)


def test_a_unary_operator_over_a_value_it_cannot_apply_to_raises_a_dsl_error() -> None:
    """Negating a string is the template's fault, raised as one, never as a raw TypeError."""
    with pytest.raises(DSLError, match="cannot apply unary USub to str"):
        evaluate("-state.shop.name == 1", **_state_with(shop={"name": "x"}))


def test_dunder_attribute_access_blocked():
    """``_``-prefixed attrs resolve to Missing — closes the dunder-escape gap.

    Without this guard, an expression like ``state.shop.__class__`` would
    return the actual class object via Python's getattr fallback.
    A Missing value leaves the check not established — failed rather than
    leaking implementation details.
    """
    state = _state_with(shop={"cart": []})
    # __class__ on the dict would normally return <class 'dict'>; DSL returns Missing.
    assert evaluate("state.shop.__class__.length == 0", **state) is False
    assert evaluate("state.shop._private == 1", **state) is False


# =============================================================================
# any() / all() generators
# =============================================================================


def test_any_finds_matching_element():
    state = _state_with(
        shop={
            "cart": [
                {"title": "Item A", "tags": ["kitchen"]},
                {"title": "Item B", "tags": ["outdoor"]},
            ]
        }
    )
    assert (
        evaluate(
            'any(it.title == "Item A" for it in state.shop.cart)',
            **state,
        )
        is True
    )
    assert (
        evaluate(
            'any(it.title == "Item Z" for it in state.shop.cart)',
            **state,
        )
        is False
    )


def test_any_with_intersects_predicate():
    state = _state_with(
        shop={
            "cart": [
                {"title": "T1", "tags": ["kitchen", "garden"]},
                {"title": "T2", "tags": ["outdoor"]},
            ]
        }
    )
    assert evaluate('any(intersects(it.tags, ["kitchen", "klezmer"]) for it in state.shop.cart)', **state) is True


def test_all_requires_every_element_to_match():
    state = _state_with(
        shop={
            "cart": [
                {"tags": ["kitchen"]},
                {"tags": ["kitchen", "garden"]},
            ]
        }
    )
    assert (
        evaluate(
            'all(contains(it.tags, "kitchen") for it in state.shop.cart)',
            **state,
        )
        is True
    )
    state2 = _state_with(shop={"cart": [{"tags": ["kitchen"]}, {"tags": ["outdoor"]}]})
    assert (
        evaluate(
            'all(contains(it.tags, "kitchen") for it in state.shop.cart)',
            **state2,
        )
        is False
    )


def test_any_over_empty_iterable_is_false():
    state = _state_with(shop={"cart": []})
    assert evaluate('any(it.title == "X" for it in state.shop.cart)', **state) is False


def test_all_over_empty_iterable_is_true():
    """Vacuous truth — all(empty) is True (matches Python's all([]))."""
    state = _state_with(shop={"cart": []})
    assert evaluate('all(it.title == "X" for it in state.shop.cart)', **state) is True


def test_any_over_missing_iterable_is_false():
    state = _state_with()
    assert evaluate('any(it.title == "X" for it in state.shop.cart)', **state) is False


def test_any_requires_it_as_binding_variable():
    state = _state_with(shop={"cart": []})
    with pytest.raises(DSLError):
        evaluate('any(item.title == "X" for item in state.shop.cart)', **state)


def test_any_rejects_if_filters():
    state = _state_with(shop={"cart": []})
    with pytest.raises(DSLError):
        evaluate('any(it.title == "X" for it in state.shop.cart if it.tag)', **state)


# =============================================================================
# Parse-time errors (fail loud before run starts)
# =============================================================================


def test_parse_rejects_empty_string():
    with pytest.raises(DSLError):
        parse("")
    with pytest.raises(DSLError):
        parse("   ")


def test_parse_rejects_syntax_errors():
    with pytest.raises(DSLError):
        parse("state.shop.cart ==")


def test_parse_rejects_disallowed_root_names():
    with pytest.raises(DSLError):
        parse("os.environ == 'production'")


def test_parse_rejects_unknown_functions():
    with pytest.raises(DSLError):
        parse("eval(\"os.system('rm -rf /')\")")


def test_parse_rejects_lambda():
    with pytest.raises(DSLError):
        parse("(lambda x: x)(state)")


def test_parse_rejects_dict_literal():
    with pytest.raises(DSLError):
        parse('{"a": 1} == variation.thing')


def test_it_outside_generator_raises():
    state = _state_with()
    with pytest.raises(DSLError):
        evaluate('it.title == "X"', **state)


# =============================================================================
# Missing semantics
# =============================================================================


def test_missing_singleton_is_falsy():
    assert bool(Missing) is False


def test_missing_all_comparisons_return_false():
    # Every operator on Missing returns False — Missing is "unknown".
    assert (Missing == 1) is False
    assert (Missing != 1) is False
    assert (Missing < 1) is False
    assert (Missing > 1) is False


def test_missing_in_returns_false():
    assert (1 in Missing) is False


def test_missing_len_returns_zero():
    assert len(Missing) == 0


def test_missing_iter_yields_nothing():
    assert list(Missing) == []


# =============================================================================
# evaluate_with_detail — returns (result, detail)
# =============================================================================


def test_evaluate_with_detail_passes_through_value():
    state = _state_with(shop={"cart": [1, 2]})
    result, detail = evaluate_with_detail("state.shop.cart.length >= 1", **state)
    assert result is True
    assert "True" in detail


def test_evaluate_with_detail_reports_missing_leaf():
    """When the entire expression evaluates to Missing, detail says it was not established."""
    state = _state_with()
    # Bare path → resolves to Missing → detail labels it.
    result, detail = evaluate_with_detail("state.shop.cart", **state)
    assert result is False
    assert detail.startswith(_dsl.NOT_ESTABLISHED)


def test_evaluate_with_detail_reports_a_compared_missing_value_as_not_established():
    state = _state_with()
    result, detail = evaluate_with_detail("state.shop.cart.length >= 1", **state)
    assert result is False
    # A comparison against Missing is itself Missing — never False, whose negation would pass.
    assert detail == f"{_dsl.NOT_ESTABLISHED}: state.shop.cart.length resolved to nothing"


# =============================================================================
# Variation references inside any() body
# =============================================================================


def test_variation_resolves_inside_generator_body():
    state = _state_with(
        shop={
            "cart": [
                {"title": "A", "tags": ["kitchen"]},
                {"title": "B", "tags": ["garden"]},
            ]
        }
    )
    assert (
        evaluate(
            "any(it.title == variation.target_title for it in state.shop.cart)",
            **state,
            variation={"target_title": "A"},
        )
        is True
    )


# =============================================================================
# Static extraction — what an expression reads, without evaluating it
# =============================================================================


def _docstring_examples() -> list[str]:
    """Every indented example line in the dsl module docstring.

    The docstring is the surface an author is taught from, so its examples are the corpus this
    module owes extraction over. Derived rather than transcribed: a hand-copied list stops
    matching the docstring the first time somebody documents a new form, and nothing says so.

    Returns:
        The example expressions, in the order they are documented.
    """
    assert _grammar.__doc__, "the dsl module docstring is the corpus these tests read"
    return [
        line.strip()
        for line in _grammar.__doc__.splitlines()
        if line.startswith("    ") and line.strip() and not line.strip().startswith("#")
    ]


def test_every_documented_example_parses():
    """A documented form the language cannot read teaches an author an expression that fails.

    This found one: ``intersects`` was documented in infix position, which is not Python and
    therefore not this grammar — a form nobody could have written from the docs and had work.
    """
    assert _docstring_examples(), "the docstring examples are the corpus; finding none is the bug"

    unparseable = []
    for example in _docstring_examples():
        try:
            parse(example)
        except DSLError as malformed:
            unparseable.append((example, str(malformed)))

    assert not unparseable


def test_every_documented_example_is_extractable():
    """Extraction runs over the whole documented surface, not the shapes it was written against.

    Paths, indexing, generators, ledger predicates and bare literals all appear there, so a form
    the walker does not handle surfaces here rather than as a silently empty path set in a gate.
    """
    for example in _docstring_examples():
        extract_paths(example)


def test_a_path_reports_the_dimension_and_what_it_addresses_beneath_it():
    """The whole dotted path comes back; which prefix is the dimension is the registry's call."""
    assert extract_paths("state.shop.cart.length >= 1").world == ("shop.cart.length",)


def test_an_index_ends_the_path_because_a_dimension_is_the_value_not_one_element():
    """Nothing under a subscript can name a dimension, so reporting it would invent one."""
    assert extract_paths('state.shop.cart[0].title == "Blue Kettle"').world == ("shop.cart",)
    assert extract_paths("state.chat.messages[-1].content").world == ("chat.messages",)


def test_a_path_inside_an_index_is_extracted_too():
    """An index is an expression, and one that reads world state reads it as much as any other."""
    reads = extract_paths('state.shop.cart[variation.position].title == "x"')

    assert reads.world == ("shop.cart",)
    assert reads.variation == ("position",)


def test_the_generator_binding_contributes_nothing_and_its_iterable_contributes_everything():
    """``it`` addresses inside an element of a path that is itself already reported."""
    reads = extract_paths('any(it.title == "X" for it in state.shop.cart)')

    assert reads.world == ("shop.cart",)


def test_variation_is_reported_apart_from_world_state():
    """A case parameter is not something a world declares, and merging the two would refuse
    every expression that reads its own case."""
    reads = extract_paths('contains(state.shop.cart, variation.categories) and variation.tone != "hostile"')

    assert reads.world == ("shop.cart",)
    assert reads.variation == ("categories", "tone")


def test_a_builtin_call_is_not_a_path():
    """Extraction must not assume everything it sees roots at a dimension, or a ledger predicate
    reads as an unresolvable one and a gate refuses a correct expression."""
    reads = extract_paths('called_before("shop.search", "shop.add_item") and call_count("shop.skip") >= 2')

    assert reads.world == ()
    assert reads.variation == ()


def test_a_path_under_a_builtin_call_is_still_extracted():
    """``length(state.x) >= 2`` reads ``x`` as surely as ``state.x.length >= 2`` does."""
    assert extract_paths("length(state.shop.cart) >= 2").world == ("shop.cart",)


def test_repeated_paths_are_reported_once_in_first_appearance_order():
    """A gate resolving the same path four times reports the same defect four times."""
    reads = extract_paths(
        "state.shop.history.length >= 1 and state.shop.cart.length >= 1 and state.shop.history.length < 9"
    )

    assert reads.world == ("shop.history.length", "shop.cart.length")


def test_extraction_refuses_what_evaluation_would_refuse():
    """One parser, so a path that cannot be extracted is one that could never have been run."""
    with pytest.raises(DSLError):
        extract_paths("state.shop.cart.length >= ")

    with pytest.raises(DSLError):
        extract_paths("os.system('rm -rf /')")


# =============================================================================
# fired(): what the cell's world events say fired, never world state and never t=0
# =============================================================================


class TestFired:
    """``fired("<dimension>")`` reads the triggered dimensions that fired during the cell."""

    @staticmethod
    def _triggered_world() -> WorldRegistry:
        return WorldRegistry(
            [
                WorldDimension(
                    name="restock_alarm",
                    carrier="shelf",
                    schema={"type": "boolean"},
                    matters="the alarm a scenario presumes goes off mid-run",
                    seed="arm",
                    read="read",
                    when=Triggered(kind="event", condition="stock_runs_out"),
                ),
                WorldDimension(
                    name="stock",
                    carrier="shelf",
                    schema={"type": "integer"},
                    matters="what is on the shelf",
                    seed="set",
                    read="read",
                ),
            ],
            bindings={"arm": lambda value: None, "set": lambda value: None, "read": lambda: None},
        )

    def test_fired_is_true_exactly_for_a_dimension_that_fired(self) -> None:
        call = {"end_state": {}, "ledger": CallLedger(), "world": None}
        alarm = frozenset({"restock_alarm"})
        assert evaluate('fired("restock_alarm")', fired=Firings(dimensions=alarm), **call) is True
        assert evaluate('fired("restock_alarm")', fired=Firings(), **call) is False
        assert evaluate('not fired("restock_alarm")', fired=Firings(), **call) is True

    def test_fired_armed_is_true_only_for_a_firing_of_the_seed_s_armed_event(self) -> None:
        """Both directions on one dimension: the world's own firing satisfies fired() and not fired_armed()."""
        call = {"end_state": {}, "ledger": CallLedger(), "world": None}
        alarm = frozenset({"restock_alarm"})
        worlds_own = Firings(dimensions=alarm)
        seeds = Firings(dimensions=alarm, armed=alarm)

        assert evaluate('fired("restock_alarm")', fired=worlds_own, **call) is True
        assert evaluate('fired_armed("restock_alarm")', fired=worlds_own, **call) is False
        assert evaluate('fired_armed("restock_alarm")', fired=seeds, **call) is True
        assert evaluate('fired_armed("restock_alarm")', fired=Firings(), **call) is False

    def test_a_witnessed_cell_s_fired_armed_is_not_established_negated_or_not(self) -> None:
        """No seed armed a witnessed cell, so its armed=False events cannot say the seed's event did not fire."""
        call = {"end_state": {}, "ledger": CallLedger(), "world": None}
        firing = WorldEvent(
            kind="event", dimension="restock_alarm", condition="low_stock", caused_by="world", event="restock-1"
        )
        witnessed = Firings.of([firing], provenance="witnessed")
        launched = Firings.of([firing], provenance="commissioned")

        assert witnessed == Firings(dimensions=frozenset({"restock_alarm"}), armed_known=False)
        for expression in ('fired_armed("restock_alarm")', 'not fired_armed("restock_alarm")'):
            passed, detail = evaluate_with_detail(expression, fired=witnessed, **call)
            assert passed is False
            assert detail == f"{NOT_ESTABLISHED}: fired_armed('restock_alarm') resolved to nothing"
        assert evaluate('fired("restock_alarm")', fired=witnessed, **call) is True
        assert evaluate('not fired_armed("restock_alarm")', fired=launched, **call) is True
        assert evaluate('fired_armed("restock_alarm")', fired=launched, **call) is False

    def test_an_armed_firing_where_none_can_be_known_is_refused(self) -> None:
        alarm = frozenset({"restock_alarm"})
        with pytest.raises(ValueError, match="cannot be known"):
            Firings(dimensions=alarm, armed=alarm, armed_known=False)

    def test_an_armed_firing_that_is_not_a_firing_is_refused(self) -> None:
        with pytest.raises(ValueError, match="an armed firing is a firing"):
            Firings(armed=frozenset({"restock_alarm"}))

    @pytest.mark.parametrize("predicate", ["fired", "fired_armed"])
    def test_with_no_world_events_recorded_it_raises_rather_than_answering_false(self, predicate: str) -> None:
        with pytest.raises(DSLError, match=rf"{predicate}\('restock_alarm'\) reads the cell's world events, and none"):
            evaluate(f'{predicate}("restock_alarm")', end_state={}, ledger=CallLedger(), world=None, fired=None)

    @pytest.mark.parametrize("predicate", ["fired", "fired_armed"])
    @pytest.mark.parametrize("arguments", ["(variation.alarm)", "()", '("a", "b")', '("")'])
    def test_it_takes_one_dimension_name_as_a_string_literal(self, predicate: str, arguments: str) -> None:
        with pytest.raises(DSLError, match=rf"{predicate}\(\) takes one dimension name as a string literal"):
            parse(predicate + arguments)

    def test_referenced_fires_names_each_dimension_once_in_source_order(self) -> None:
        expression = 'fired("b") and (fired_armed("a") or not fired("b")) and fired_armed("c")'
        assert _grammar.referenced_fires(expression) == ("b", "a", "c")

    def test_a_fired_name_that_is_not_a_triggered_dimension_is_named(self) -> None:
        world = self._triggered_world()
        assert _dsl.undefined_fire_references('fired("restock_alarm")', world) == ()
        assert _dsl.undefined_fire_references('fired("restock_alarn")', world) == (
            "fired('restock_alarn') names no dimension this host's world declares",
        )
        assert "set at t=0" in _dsl.undefined_fire_references('fired("stock")', world)[0]
        assert "declares no world" in _dsl.undefined_fire_references('fired("stock")', None)[0]
        assert "set at t=0" in _dsl.undefined_fire_references('fired_armed("stock")', world)[0]

    def test_a_precondition_reading_fired_is_refused_where_it_is_written(self) -> None:
        with pytest.raises(ValidationError, match="before any trigger could fire"):
            Precondition(expression='fired("restock_alarm")', presumes="the alarm already went off")
        with pytest.raises(ValidationError, match="before any trigger could fire"):
            Precondition(expression='fired_armed("restock_alarm")', presumes="the armed alarm already went off")
        Precondition(expression="state.stock >= 1", presumes="something is on the shelf")
