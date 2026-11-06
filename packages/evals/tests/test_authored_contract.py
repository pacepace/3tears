"""The strict projection every package call sends, and the confidence and reading vocabularies the stored analyses
share: what survives of the authored contract now that the memo writer is retired.

The projection is held over a reply shaped as a writer's structured reply is (:class:`_Pick`): nested models in
lists and alone, so its schema has definitions and references, with titles, defaults and a closed vocabulary.
"""

from __future__ import annotations

from typing import Any, Literal

import pytest
from pydantic import BaseModel, Field

from threetears.evals.kernel import authored


class _Why(BaseModel):
    """One reason a pick is made over a warning, named by its key."""

    key: str = Field(description="The warning's key.")
    why: str = Field(default="", description="Why the pick stands over it.")


class _Pick(BaseModel):
    """A structured reply naming an arm, how sure, and each warning it is made over."""

    arm: str = Field(description="The arm picked, by its label.")
    reasons: list[str] = Field(description="Why, strongest first.")
    confidence: authored.Confidence
    over: list[_Why] = Field(default_factory=list, description="Each warning the pick is made over.")
    first: _Why | None = Field(default=None, description="The warning that weighs most, if any.")
    reading: Literal["measure", "judged"] = "measure"


def _schema() -> dict[str, Any]:
    return authored.strict_schema(_Pick.model_json_schema())


def _walk(node: Any):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


def _schema_nodes(node: Any):
    """Every schema node, stepping over the name maps (`properties`, `$defs`) rather than yielding them."""
    if isinstance(node, list):
        for item in node:
            yield from _schema_nodes(item)
        return
    if not isinstance(node, dict):
        return
    yield node
    for key, value in node.items():
        if key in ("properties", "$defs") and isinstance(value, dict):
            for child in value.values():
                yield from _schema_nodes(child)
        else:
            yield from _schema_nodes(value)


class TestTheStrictProjection:
    def test_every_object_is_closed_and_fully_required(self):
        for node in _walk(_schema()):
            if "properties" in node:
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])

    def test_a_reference_stands_alone(self):
        refs = [node for node in _walk(_schema()) if "$ref" in node]
        assert refs, "the schema has nested definitions, so it has references"
        assert all(set(node) == {"$ref"} for node in refs)

    def test_nothing_the_providers_refuse_survives(self):
        # Checked on schema NODES, never on the name maps under `properties` and `$defs`, where a property may be
        # named like a keyword and must survive.
        offenders = [
            node for node in _schema_nodes(_schema()) if {"default", "discriminator", "title", "oneOf"} & set(node)
        ]
        assert not offenders, offenders[:3]

    def test_one_of_becomes_any_of_and_a_ref_loses_its_siblings(self):
        projected = authored.strict_schema(
            {"type": "object", "properties": {"x": {"oneOf": [{"$ref": "#/$defs/A", "description": "d"}]}}}
        )
        assert projected["properties"]["x"] == {"anyOf": [{"$ref": "#/$defs/A"}]}

    def test_a_map_is_refused_rather_than_sent(self):
        class _Map(BaseModel):
            m: dict[str, str]

        with pytest.raises(authored.MapFieldInSchema):
            authored.strict_schema(_Map.model_json_schema())


class TestOneVocabulary:
    """The contract states its vocabularies standalone, so nothing else must disagree with it."""

    def test_the_confidence_tiers_are_the_analysis_models_tiers(self):
        import typing

        from threetears.evals.kernel.campaign import ConfidenceTier

        assert set(typing.get_args(authored.Confidence)) == set(typing.get_args(ConfidenceTier))

    def test_the_reading_kinds_are_the_analysis_models_kinds(self):
        import typing

        from threetears.evals.kernel.campaign import ReadingKind

        assert set(typing.get_args(authored.Reading)) == set(typing.get_args(ReadingKind))
