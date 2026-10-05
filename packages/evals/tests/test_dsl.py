"""Unit tests for the goal-state DSL."""

from __future__ import annotations

from typing import Any

import pytest

import threetears.evals.contracts.dsl as _dsl
from threetears.evals.contracts.dsl import DSLError, Missing, evaluate, evaluate_with_detail, extract_paths, parse
from threetears.evals.contracts.host import WorldDimension, WorldRegistry
from threetears.evals.contracts.call_ledger import CallLedger


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
    assert (
        evaluate(
            "intersects(state.shop.cart, variation.target_categories)",
            **state,
            variation={"target_categories": ["kitchen", "outdoor"]},
        )
        is True
    )
    assert (
        evaluate(
            "intersects(state.shop.cart, variation.target_categories)",
            **state,
            variation={"target_categories": ["toys"]},
        )
        is False
    )


def test_intersects_missing_path_returns_false():
    state = _state_with()
    assert (
        evaluate(
            "intersects(state.shop.cart, variation.target_categories)",
            **state,
            variation={"target_categories": ["kitchen"]},
        )
        is False
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


def test_not_on_missing_path_returns_true():
    """Negating a missing-path comparison yields True.

    Per the DSL spec, ``Missing >= 1`` evaluates to False (Missing
    propagates through the comparison as "unknown / can't satisfy").
    ``not False`` is then True. This matches user intent: "is it NOT the
    case that the cart has >= 1 items?" — a missing cart clearly
    doesn't satisfy the predicate, so the negation is True.
    """
    state = _state_with()
    assert evaluate("not state.shop.cart.length >= 1", **state) is True


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


def test_ordering_predicate_rejects_non_string_spec():
    """Passing a path expression where a 'tool.action' string is required raises."""
    state = _state_with()
    state["ledger"].record("shop", "search", {})
    # state.shop is a dict, not a 'tool.action' string spec.
    with pytest.raises(DSLError, match="string spec"):
        evaluate("called_before(state.shop, state.chat)", **state)


def test_call_count_rejects_non_string_spec():
    state = _state_with()
    state["ledger"].record("shop", "search", {})
    with pytest.raises(DSLError, match="string spec"):
        evaluate("call_count(state.shop) >= 1", **state)


def test_ordering_predicate_rejects_non_string_constant():
    """Passing an int / list constant where 'tool.action' is required raises at run time."""
    state = _state_with()
    state["ledger"].record("shop", "search", {})
    with pytest.raises(DSLError, match="string spec"):
        evaluate('called_before(1, "shop.search")', **state)
    with pytest.raises(DSLError, match="string spec"):
        evaluate("last_call_was([1, 2, 3])", **state)


def test_dunder_attribute_access_blocked():
    """``_``-prefixed attrs resolve to Missing — closes the dunder-escape gap.

    Without this guard, an expression like ``state.shop.__class__`` would
    return the actual class object via Python's getattr fallback.
    Comparisons against Missing return False, so the expression evaluates
    to False rather than leaking implementation details.
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
    assert (
        evaluate(
            "any(intersects(it.tags, variation.target) for it in state.shop.cart)",
            **state,
            variation={"target": ["kitchen", "klezmer"]},
        )
        is True
    )


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
    """When the entire expression evaluates to Missing, detail says so.

    A comparison against a Missing path returns False (per spec), so the
    detail surfaces 'False'. The 'Missing' label only fires when the
    expression itself resolves to Missing without a comparison — e.g.,
    'state.shop.cart' alone, with no operator.
    """
    state = _state_with()
    # Bare path → resolves to Missing → detail labels it.
    result, detail = evaluate_with_detail("state.shop.cart", **state)
    assert result is False
    assert "Missing" in detail


def test_evaluate_with_detail_reports_false_for_compared_missing():
    state = _state_with()
    result, detail = evaluate_with_detail("state.shop.cart.length >= 1", **state)
    assert result is False
    # Comparison against Missing returns False; detail surfaces the value.
    assert detail == "False"


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
    assert _dsl.__doc__, "the dsl module docstring is the corpus these tests read"
    return [
        line.strip()
        for line in _dsl.__doc__.splitlines()
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
