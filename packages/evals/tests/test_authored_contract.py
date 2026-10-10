"""The authored contract: a small document shape, strictly projected, sized to what every writer accepts."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from threetears.evals.analysis.viz_refs import REFERENCEABLE_VIZ_TYPES
from threetears.evals.kernel import authored
from threetears.evals.kernel.campaign import ENGINE_CAVEAT_KINDS


KINDS = ENGINE_CAVEAT_KINDS
TYPES = REFERENCEABLE_VIZ_TYPES


def _schema(kinds=KINDS) -> dict[str, Any]:
    return authored.authored_schema(kinds, TYPES)


def _payload(**overrides: Any) -> dict[str, Any]:
    finding = {
        "title": "Latency halves",
        "body": "The fast arm halves p95 at no measured quality cost.",
        "confidence": "high",
        "axes": ["lookup.max_rounds"],
        "evidence": [{"cell": "k1:a1", "measure_id": "total_ms", "reading": "measure"}],
        "chart": {
            "type": "none",
            "cells": [],
            "measures": [],
            "axis": "",
            "note": "",
            "caption": "",
        },
        "caveats": [{"kind": "sampling", "text": "Five cases."}],
        "invalidates": [],
        "durable": "",
    }
    base = {
        "headline": "Two rounds buys latency and costs honesty.",
        "summary": "",
        "findings": [finding],
        "decisions": [
            {
                "proposal": "Keep three rounds.",
                "disposition": "adopted",
                "cells": ["k1:a1"],
                "confidence": "medium",
                "rests_on": [0],
                "revisit_when": "",
            }
        ],
        "questions": [],
        "next": [],
    }
    base.update(overrides)
    return base


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


class TestTheSchemaFitsEveryWriter:
    def test_the_schema_holds_no_more_slots_than_the_budget(self):
        slots = authored.count_slots(_schema())
        assert slots <= authored.SLOT_BUDGET, (
            f"the authored schema holds {slots} free-text and list slots, over the budget of "
            f"{authored.SLOT_BUDGET}; re-probe every eligible writer before raising it"
        )

    def test_a_host_registering_caveat_kinds_stays_inside_the_budget(self):
        # Kinds are enum values, which the grammar barely pays for; a host adding them adds no slot.
        with_kinds = _schema(KINDS | {"host_a", "host_b", "host_c"})
        assert authored.count_slots(with_kinds) == authored.count_slots(_schema())

    def test_nothing_is_a_union(self):
        schema = json.dumps(_schema())
        assert '"anyOf"' not in schema and '"oneOf"' not in schema
        assert '"null"' not in schema, "absence is an empty string or list, never null"

    def test_the_slot_counter_counts_what_the_grammar_pays_for(self):
        class _Probe(BaseModel):
            model_config = ConfigDict(extra="forbid")
            a: str
            b: list[str]
            c: int

        # a string, an array, and the array's string items: three slots; the integer is none.
        assert authored.count_slots(authored.strict_schema(_Probe.model_json_schema())) == 3

    def test_the_budget_guard_fires_on_a_schema_over_it(self):
        padded = create_model(
            "Padded",
            __base__=authored.AuthoredAnalysis,
            **{f"pad{i}": (str, ...) for i in range(authored.SLOT_BUDGET)},
        )
        assert authored.count_slots(authored.strict_schema(padded.model_json_schema())) > authored.SLOT_BUDGET


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
        # Checked on schema NODES, never on the name maps under `properties` and `$defs`: `title` is
        # also a finding's property name, and that one must survive.
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


class TestTheCheckOnReturn:
    def test_a_well_formed_payload_passes(self):
        authored.validate_authored(_payload(), KINDS, TYPES)

    def test_an_extra_key_is_refused(self):
        with pytest.raises(ValidationError):
            authored.validate_authored(_payload(bluf="the old shape"), KINDS, TYPES)

    def test_a_missing_key_is_refused(self):
        payload = _payload()
        del payload["next"]
        with pytest.raises(ValidationError):
            authored.validate_authored(payload, KINDS, TYPES)

    def test_null_is_refused_where_the_contract_says_empty(self):
        with pytest.raises(ValidationError):
            authored.validate_authored(_payload(summary=None), KINDS, TYPES)

    def test_an_unregistered_caveat_kind_is_refused_and_a_registered_one_accepted(self):
        payload = _payload()
        payload["findings"][0]["caveats"] = [{"kind": "host_kind", "text": "x"}]
        with pytest.raises(authored.OffVocabulary):
            authored.validate_authored(payload, KINDS, TYPES)
        authored.validate_authored(payload, KINDS | {"host_kind"}, TYPES)

    def test_a_chart_type_off_the_menu_is_refused(self):
        payload = _payload()
        payload["findings"][0]["chart"]["type"] = "scatter"
        with pytest.raises(authored.OffVocabulary):
            authored.validate_authored(payload, KINDS, TYPES)

    def test_the_directive_sends_the_projected_schema_strictly(self):
        directive = authored.response_format(KINDS, TYPES)
        assert directive["type"] == "json_schema"
        assert directive["json_schema"]["strict"] is True
        assert directive["json_schema"]["schema"] == _schema()

    def test_the_host_vocabularies_are_sent_as_enums(self):
        defs = _schema(KINDS | {"host_kind"})["$defs"]
        assert set(defs["Caveat"]["properties"]["kind"]["enum"]) == KINDS | {"host_kind"}
        assert set(defs["Chart"]["properties"]["type"]["enum"]) == TYPES | {authored.NO_CHART}


class TestOneVocabulary:
    """The contract states its vocabularies standalone, so nothing else must disagree with it."""

    def test_the_confidence_tiers_are_the_analysis_models_tiers(self):
        import typing

        from threetears.evals.kernel.campaign import CONFIDENCE_TIERS

        assert set(typing.get_args(authored.Confidence)) == set(CONFIDENCE_TIERS)

    def test_the_reading_kinds_are_the_analysis_models_kinds(self):
        import typing

        from threetears.evals.kernel.campaign import ReadingKind

        assert set(typing.get_args(authored.Reading)) == set(typing.get_args(ReadingKind))

    def test_every_chart_type_on_offer_has_a_reading_contract(self):
        from threetears.evals.analysis.viz_refs import CHART_READINGS

        # The menu the schema offers, the positional contract the model is told, and the chart table
        # the compiler reads are one set — a type in one and not another is offered and then refused.
        assert set(CHART_READINGS) == set(TYPES)


def test_the_model_is_told_how_each_chart_type_reads_its_lists():
    from threetears.evals.analysis.viz_refs import CHART_READINGS

    description = authored.authored_schema(KINDS, CHART_READINGS)["$defs"]["Chart"]["properties"]["type"]["description"]
    for chart_type, reads in CHART_READINGS.items():
        assert f"{chart_type}: {reads}" in description


class TestTheWriterCannotChooseATier:
    """What a verdict stands on is read off its evidence by code, so the writer is never offered a tier."""

    def test_the_sent_schema_has_no_tier_field_and_no_tier_word(self):
        schema = _schema()
        fields = {name for node in _walk(schema) for name in node.get("properties", {})}
        assert not {name for name in fields if "tier" in name}, "the sent schema offers the writer a tier"
        # No enum anywhere in what is sent carries a tier's name, so no field renamed around the check
        # can still hand the writer "calibrated" to pick.
        words = {value for node in _walk(schema) for value in node.get("enum", [])}
        assert not words & {"mechanical", "directional", "separation", "incidental", "calibrated"}

    def test_a_planted_tier_is_refused_on_return(self):
        payload = _payload()
        payload["findings"][0]["evidence_tier"] = "calibrated"
        with pytest.raises(ValidationError, match="evidence_tier"):
            authored.validate_authored(payload, KINDS, TYPES)
