"""Every schema walker in the contracts agrees on where one schema can sit inside another.

Four readers walk a world or parameter schema: the honoured-subset audit (``honoured_kind``), the registry's
self-contradiction check, the prose gate over world paths and the call-parameter gate over recorded calls.
They once each kept their own list of nesting positions and disagreed — one stopped at ``anyOf``, one never
looked under ``additionalProperties`` — so a defect or a prose field written where one walker did not look
was invisible to it. Each case below writes the SAME leaf at one position and asks every walker about it, so
a walker that stops short of a position fails here beside the walkers that do not.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import pytest

from threetears.evals.kernel.dsl import call_parameter_matches, undefined_call_references, world_prose_matches
from threetears.evals.kernel.host.world import WorldDimension, WorldRegistrationError, WorldRegistry
from threetears.evals.kernel.host.world_schema import UnsupportedSchemaError, honoured_kind
from threetears.evals.schema.prose import PROSE_SCHEMA_KEY, schema_nodes_at
from threetears.evals.schema.schema_nesting import nested_schemas


class _Position:
    """One place a leaf schema can be written, and how each walker names or addresses it."""

    def __init__(self, wrap: Callable[[dict[str, Any]], dict[str, Any]], path: str, at: str) -> None:
        self.wrap = wrap
        self.path = path
        """The DSL path below the root that addresses the leaf."""
        self.at = at
        """The schema path suffix a refusal names it by."""


_POSITIONS = {
    "an anyOf branch": _Position(
        lambda leaf: {"type": "object", "properties": {"f": {"anyOf": [{"type": "boolean"}, leaf]}}}, "f", ".f.anyOf[1]"
    ),
    "an additionalProperties schema": _Position(
        lambda leaf: {"type": "object", "properties": {"meta": {"type": "object", "additionalProperties": leaf}}},
        "meta.anykey",
        ".meta.additionalProperties",
    ),
    "nested items": _Position(
        lambda leaf: {
            "type": "object",
            "properties": {"grid": {"type": "array", "items": {"type": "array", "items": leaf}}},
        },
        "grid[0][0]",
        ".grid.items.items",
    ),
}

_EACH_POSITION = pytest.mark.parametrize("position", list(_POSITIONS.values()), ids=list(_POSITIONS))


def _registry(schema: dict[str, Any]) -> WorldRegistry:
    return WorldRegistry(
        [
            WorldDimension(
                name="d",
                schema=schema,
                matters="a goal check reads it back",
                carrier="c",
                seed="d.seed",
                read="d.read",
            )
        ],
        bindings={"d.seed": lambda value: None, "d.read": dict},
    )


class TestEveryWalkerReachesEveryPosition:
    @_EACH_POSITION
    def test_the_honoured_subset_audit_refuses_a_keyword_there(self, position: _Position) -> None:
        with pytest.raises(UnsupportedSchemaError, match="^" + re.escape(f"d{position.at} ")):
            honoured_kind(position.wrap({"type": "string", "pattern": "^x$"}), at="d")

    @_EACH_POSITION
    def test_the_registry_refuses_a_contradiction_there(self, position: _Position) -> None:
        with pytest.raises(WorldRegistrationError, match=re.escape(f"d's schema{position.at} declares minLength=3")):
            _registry(position.wrap({"type": "string", "minLength": 3, "maxLength": 1}))

    @_EACH_POSITION
    def test_the_prose_gate_refuses_a_prose_field_there_and_accepts_a_structural_one(self, position: _Position) -> None:
        expression = f'state.d.{position.path} == "x"'

        assert world_prose_matches(expression, _registry(position.wrap({"type": "string", PROSE_SCHEMA_KEY: True})))
        assert world_prose_matches(expression, _registry(position.wrap({"type": "string"}))) == ()

    @_EACH_POSITION
    def test_the_call_gate_refuses_free_text_there_and_accepts_a_closed_value(self, position: _Position) -> None:
        expression = f'calls("t.a")[0].{position.path} == "x"'

        def reader(leaf: dict[str, Any]) -> Callable[[str, str], dict[str, Any]]:
            return lambda tool, action: position.wrap(leaf)

        assert [reason for _, reason in call_parameter_matches(expression, reader({"type": "string"}))] == [
            f"t.a's {position.path.replace('[0]', '.[]')} is free text"
        ]
        assert call_parameter_matches(expression, reader({"type": "string", "enum": ["x"]})) == ()

    @_EACH_POSITION
    def test_the_vocabulary_gate_finds_a_parameter_there(self, position: _Position) -> None:
        expression = f'calls("t.a")[0].{position.path}.length > 0'

        assert undefined_call_references(expression, None, lambda tool, action: position.wrap({"type": "string"})) == ()


class TestTheIterator:
    def test_it_names_each_position_in_declaration_order(self) -> None:
        schema = {
            "items": {"type": "string"},
            "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
            "additionalProperties": {"type": "boolean"},
            "anyOf": [{"type": "null"}],
        }

        assert [(nested.keyword, nested.key, nested.at("r")) for nested in nested_schemas(schema)] == [
            ("items", None, "r.items"),
            ("properties", "a", "r.a"),
            ("properties", "b", "r.b"),
            ("additionalProperties", None, "r.additionalProperties"),
            ("anyOf", 0, "r.anyOf[0]"),
        ]

    def test_a_boolean_additional_properties_holds_no_schema(self) -> None:
        assert list(nested_schemas({"type": "object", "additionalProperties": False})) == []

    def test_a_declared_property_wins_over_additional_properties_when_addressed_by_name(self) -> None:
        declared = {"type": "integer"}
        schema = {"type": "object", "properties": {"n": declared}, "additionalProperties": {"type": "string"}}

        assert schema_nodes_at(schema, ("n",)) == (declared,)
        assert schema_nodes_at(schema, ("other",)) == ({"type": "string"},)

    def test_an_undescribed_position_addresses_nothing(self) -> None:
        assert schema_nodes_at({"type": "array", "items": {"type": "object"}}, ("length",)) == ()
        assert schema_nodes_at({"type": "string"}, ("[]",)) == ()
