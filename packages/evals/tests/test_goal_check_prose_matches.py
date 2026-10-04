"""A goal check's text comparisons, and which of them read model prose.

``contains()`` keeps membership over structured values; a substring or equality test over text a
model wrote is what an authoring gate refuses. The language reports the comparisons
(:func:`~threetears.evals.contracts.dsl.extract_text_matches`); the host's vocabulary says which read prose
(:func:`~threetears.evals.contracts.dsl.world_prose_matches`). Each refusal below has its accepted sibling on
the SAME registry, so an inverted rule cannot pass.
"""

from __future__ import annotations

import pytest

from threetears.evals.contracts import dsl as _dsl
from threetears.evals.contracts.dsl import DSLError, extract_text_matches, world_prose_matches
from threetears.evals.contracts.host.world import WorldDimension, WorldRegistry
from threetears.evals.contracts.prose import PROSE_SCHEMA_KEY, schema_is_prose, schema_nodes_at


_BINDINGS = {"h.seed": lambda value: None, "h.read": list}

_MESSAGE = {
    "type": "object",
    "properties": {
        "content": {"type": "string", PROSE_SCHEMA_KEY: True},
        "author": {"type": "string"},
    },
}
_ITEM = {"type": "object", "properties": {"title": {"type": "string"}, "uri": {"type": "string"}}}


def _dimension(name: str, schema: dict) -> WorldDimension:
    return WorldDimension(
        name=name,
        schema=schema,
        matters="a scenario reads it back at the end, so a goal check needs a path to it",
        carrier="h",
        seed="h.seed",
        read="h.read",
    )


@pytest.fixture
def registry() -> WorldRegistry:
    return WorldRegistry(
        [
            _dimension("chat.messages", {"type": "array", "items": _MESSAGE}),
            _dimension("chat.replies", {"type": "array", "items": {"type": "string", PROSE_SCHEMA_KEY: True}}),
            _dimension("shop.cart", {"type": "array", "items": _ITEM}),
            _dimension(
                "chat.pinned",
                {"anyOf": [{"type": "object", "properties": {}, "additionalProperties": False}, _MESSAGE]},
            ),
            _dimension(
                "chat.drafts",
                {"type": "array", "items": {"anyOf": [{"type": "null"}, {"type": "string", PROSE_SCHEMA_KEY: True}]}},
            ),
            _dimension("chat.tags", {"type": "array", "items": {"anyOf": [{"type": "null"}, {"type": "string"}]}}),
        ],
        bindings=_BINDINGS,
    )


class TestTheAuthoringRuleOverProse:
    def test_contains_over_message_content_is_refused(self, registry):
        refused = world_prose_matches('contains(state.chat.messages[-1].content, "kitchen")', registry)
        assert [match.operand for match in refused] == [("chat", "messages", "[]", "content")]
        assert refused[0].predicate == "contains"

    def test_contains_over_a_structured_list_is_still_accepted(self, registry):
        assert world_prose_matches('contains(state.shop.cart, "Blue Kettle")', registry) == ()

    def test_equality_against_a_literal_over_prose_is_refused(self, registry):
        refused = world_prose_matches('state.chat.messages[0].content == "hello"', registry)
        assert [match.predicate for match in refused] == ["equality"]

    def test_emptiness_over_prose_is_accepted(self, registry):
        """Whether anything was written is structure, not a reading of what was written."""
        assert world_prose_matches('state.chat.messages[0].content == ""', registry) == ()
        assert world_prose_matches("state.chat.messages.length >= 1", registry) == ()

    def test_a_non_prose_field_of_the_same_item_is_accepted(self, registry):
        assert world_prose_matches('contains(state.chat.messages[-1].author, "bob")', registry) == ()

    def test_it_inside_a_generator_resolves_to_the_iterable_it_binds(self, registry):
        refused = world_prose_matches('any(it.content == "hi" for it in state.chat.messages)', registry)
        assert [match.operand for match in refused] == [("chat", "messages", "[]", "content")]
        assert world_prose_matches('any(it.title == "X" for it in state.shop.cart)', registry) == ()

    def test_membership_in_an_array_of_prose_strings_is_refused(self, registry):
        """Matching a literal against members that are prose is equality over prose."""
        assert len(world_prose_matches('contains(state.chat.replies, "ok")', registry)) == 1

    def test_a_path_naming_no_dimension_is_left_to_the_vocabulary_gate(self, registry):
        assert world_prose_matches('contains(state.nowhere.at_all, "x")', registry) == ()

    def test_a_host_declaring_no_world_refuses_nothing(self):
        assert world_prose_matches('contains(state.chat.messages[-1].content, "kitchen")', None) == ()

    def test_a_malformed_expression_raises(self, registry):
        with pytest.raises(DSLError):
            world_prose_matches("contains(state.chat.messages[-1].content, lambda: 1)", registry)


class TestAShapeListIsReadThroughEveryShape:
    """A value under ``anyOf`` takes one of its shapes, so a field prose in any shape is prose."""

    def test_a_prose_field_inside_a_branch_is_refused(self, registry: WorldRegistry) -> None:
        refused = world_prose_matches('contains(state.chat.pinned.content, "kitchen")', registry)
        assert [match.operand for match in refused] == [("chat", "pinned", "content")]

    def test_a_structural_field_inside_a_branch_is_accepted(self, registry: WorldRegistry) -> None:
        assert world_prose_matches('contains(state.chat.pinned.author, "bob")', registry) == ()

    def test_membership_in_an_array_whose_items_take_a_prose_shape_is_refused(self, registry: WorldRegistry) -> None:
        """An array's items under ``anyOf`` are read through every shape, as a field under one is."""
        refused = world_prose_matches('contains(state.chat.drafts, "ok")', registry)
        assert [match.operand for match in refused] == [("chat", "drafts")]

    def test_membership_in_an_array_whose_items_take_no_prose_shape_is_accepted(self, registry: WorldRegistry) -> None:
        assert world_prose_matches('contains(state.chat.tags, "ok")', registry) == ()

    def test_the_array_and_the_path_addressing_agree_on_one_schema(self, registry: WorldRegistry) -> None:
        """The membership test and an element's own path read the same shapes, so they refuse together."""
        drafts = registry.get("chat.drafts")
        assert drafts is not None
        assert schema_is_prose(drafts.schema)
        assert any(schema_is_prose(node) for node in schema_nodes_at(drafts.schema, ("[]",)))
        tags = registry.get("chat.tags")
        assert tags is not None
        assert not schema_is_prose(tags.schema)


class TestTheExtractor:
    def test_every_documented_example_is_walkable(self):
        """The whole taught surface runs through the text-match walker without raising."""
        assert _dsl.__doc__
        examples = [
            line.strip().split("#", 1)[0].strip()
            for line in _dsl.__doc__.splitlines()
            if line.startswith("    ") and line.strip() and not line.strip().startswith(("*", "-"))
        ]
        walked = 0
        for example in examples:
            try:
                extract_text_matches(example)
            except DSLError:
                continue
            walked += 1
        assert walked >= 10, walked

    def test_a_count_comparison_is_not_a_text_match(self):
        assert extract_text_matches('call_count("shop.add_item") >= 2') == ()
        assert extract_text_matches("state.shop.cart.length == 2") == ()

    def test_a_variation_comparand_is_a_text_match(self):
        (match,) = extract_text_matches("state.shop.cart[0].title == variation.title")
        assert match.operand == ("shop", "cart", "[]", "title")

    def test_intersects_reports_each_state_side(self):
        matches = extract_text_matches("intersects(state.a.tags, state.b.tags)")
        assert [match.operand for match in matches] == [("a", "tags"), ("b", "tags")]


class TestSchemaNodesAt:
    def test_steps_through_items_and_properties(self):
        schema = {"type": "array", "items": _MESSAGE}
        assert schema_nodes_at(schema, ("[]", "content")) == (_MESSAGE["properties"]["content"],)

    def test_an_undescribed_position_addresses_nothing(self):
        assert schema_nodes_at({"type": "array", "items": _MESSAGE}, ("length",)) == ()
        assert schema_nodes_at({"type": "string"}, ("[]",)) == ()
