"""The report plan's models: the planner's contract strictly projected, the stored plan, and the documentation copy.

``docs/eval-report-kit/schema/report-plan.schema.json`` documents the plan and ``references/r5-kairo-round2.plan.json``
is the plan the planner should write for Kairo round 2; both are held to the models here, field for field, enum for
enum and bound for bound, so neither copy drifts from what the package reads.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, get_args

import jsonschema
import pytest
from pydantic import BaseModel, ValidationError

from threetears.evals.analysis.report.charts import KIT_CHARTS
from threetears.evals.analysis.report.charts import ChartReference as KitReference
from threetears.evals.kernel import authored, report_plan
from threetears.evals.kernel.report_plan import (
    ChartQuery,
    ChartReference,
    ChartSpec,
    Decision,
    DecisionSlot,
    DurableClaim,
    Explore,
    ExploreControl,
    Flag,
    Masthead,
    MethodItem,
    OptionRow,
    Options,
    PlanArm,
    PlannedReport,
    ReportPlan,
    Section,
    Signals,
    WarningAnswer,
    WarningRef,
)
from threetears.evals.schema.prose import field_is_prose

from packages.evals.tests.report_kit_support import KIT

DOCS_SCHEMA = KIT / "schema" / "report-plan.schema.json"
R5_PLAN = KIT / "references" / "r5-kairo-round2.plan.json"

#: The fields code fills on the stored plan, top level and on each chart.
CODE_FILLED = frozenset({"schema", "signals", "shape", "masthead", "arms", "show_arms_key"})
CODE_FILLED_ON_A_CHART = frozenset({"level", "unit", "domain"})

#: Each object in the documentation schema, by where it sits, and the model it documents.
DOCUMENTED: dict[tuple[str, ...], type[BaseModel]] = {
    (): ReportPlan,
    ("signals",): Signals,
    ("masthead",): Masthead,
    ("arms", "[]"): PlanArm,
    ("flag", "[]"): Flag,
    ("decision",): Decision,
    ("decision", "slots", "[]"): DecisionSlot,
    ("options",): Options,
    ("options", "rows", "[]"): OptionRow,
    ("sections", "[]"): Section,
    ("explore", "[]"): Explore,
    ("explore", "[]", "controls", "[]"): ExploreControl,
    ("method", "[]"): MethodItem,
    ("durable_claims", "[]"): DurableClaim,
    ("answers", "[]"): WarningAnswer,
    ("answers", "[]", "warning"): WarningRef,
    ("$defs", "chart"): ChartSpec,
    ("$defs", "chart", "query"): ChartQuery,
    ("$defs", "chart", "reference", "[]"): ChartReference,
}

#: The JSON Schema keywords both copies state for a property, compared one for one.
_CONSTRAINTS = ("enum", "const", "pattern", "minItems", "maxItems", "minLength", "maxLength", "minimum")


def _docs() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(DOCS_SCHEMA.read_text(encoding="utf-8"))
    return loaded


def _r5() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(R5_PLAN.read_text(encoding="utf-8"))
    return loaded


def _planned(plan: dict[str, Any]) -> dict[str, Any]:
    """A stored plan without the fields code fills: what the planner wrote."""
    planned = {key: value for key, value in deepcopy(plan).items() if key not in CODE_FILLED}
    for chart in [*planned["summary"], *(section["chart"] for section in planned["sections"])]:
        for key in CODE_FILLED_ON_A_CHART:
            del chart[key]
    return planned


def _contract() -> dict[str, Any]:
    schema: dict[str, Any] = report_plan.planner_response_format()["json_schema"]["schema"]
    return schema


def _walk(node: Any):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


def _docs_node(path: tuple[str, ...]) -> dict[str, Any]:
    """The documentation schema's node at ``path``: property names, ``[]`` for a list's items, references followed."""
    root = _docs()
    node = root
    for step in path:
        if step == "[]":
            node = node["items"]
        elif step == "$defs" and node is root:
            node = root["$defs"]
        elif node is root.get("$defs"):
            node = node[step]
        else:
            node = node["properties"][step]
        if "$ref" in node:
            node = root["$defs"][node["$ref"].rsplit("/", 1)[1]]
    return node


def _constraints(node: dict[str, Any]) -> dict[str, Any]:
    """A property's constraints, and its items' under ``items.``, so a list's bounds and its members' both count."""
    found = {key: node[key] for key in _CONSTRAINTS if key in node}
    items = node.get("items")
    if isinstance(items, dict):
        found |= {f"items.{key}": items[key] for key in _CONSTRAINTS if key in items}
    if "const" in found:
        found["enum"] = [found.pop("const")]
    return found


class TestThePlannersContract:
    def test_the_strict_projection_succeeds_and_is_sent_strictly(self):
        directive = report_plan.planner_response_format()
        assert directive["type"] == "json_schema"
        assert directive["json_schema"]["name"] == report_plan.PLAN_SCHEMA_NAME
        assert directive["json_schema"]["strict"] is True

    def test_every_object_is_closed_and_fully_required(self):
        for node in _walk(_contract()):
            if "properties" in node:
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])

    def test_nothing_is_null_or_a_union(self):
        sent = json.dumps(_contract())
        assert '"anyOf"' not in sent and '"oneOf"' not in sent
        assert '"null"' not in sent, "absence is an empty string or a list of at most one, never null"

    def test_a_reference_stands_alone(self):
        refs = [node for node in _walk(_contract()) if "$ref" in node]
        assert refs and all(set(node) == {"$ref"} for node in refs)

    def test_the_contract_holds_no_field_code_fills(self):
        # A chart query's `arms` is the planner's filter, not the stored plan's arm colours, so each code-filled
        # name is looked for where code fills it: the top level, and a chart.
        contract = _contract()
        assert not set(contract["properties"]) & CODE_FILLED
        assert not set(contract["$defs"]["PlannedChart"]["properties"]) & CODE_FILLED_ON_A_CHART

    def test_the_stored_plan_is_the_contract_plus_what_code_fills(self):
        planned = set(PlannedReport.model_fields)
        stored = {field.alias or name for name, field in ReportPlan.model_fields.items()}
        assert stored == planned | CODE_FILLED
        assert set(ChartSpec.model_fields) == set(report_plan.PlannedChart.model_fields) | CODE_FILLED_ON_A_CHART

    def test_the_planners_part_of_the_kairo_plan_is_a_valid_reply(self):
        PlannedReport.model_validate(_planned(_r5()))

    def test_a_reply_carrying_a_code_filled_field_is_refused(self):
        for key, value in (("shape", "D"), ("show_arms_key", True)):
            with pytest.raises(ValidationError, match="Extra inputs"):
                PlannedReport.model_validate({**_planned(_r5()), key: value})

    def test_the_prose_is_typed_as_prose(self):
        prose = {
            (model.__name__, name)
            for model in (PlannedReport, Flag, DecisionSlot, report_plan.PlannedSection, MethodItem, WarningAnswer)
            for name, field in model.model_fields.items()
            if field_is_prose(field)
        }
        assert {
            ("PlannedReport", "headline"),
            ("PlannedReport", "lede"),
            ("Flag", "text"),
            ("DecisionSlot", "text"),
            ("PlannedSection", "headline"),
            ("MethodItem", "text"),
            ("WarningAnswer", "text"),
        } <= prose


class TestTheStoredPlan:
    def test_the_kairo_plan_loads_and_round_trips(self):
        plan = ReportPlan.model_validate(_r5())
        assert plan.model_dump(mode="json") == _r5()
        assert plan.schema_id == report_plan.REPORT_PLAN_SCHEMA

    def test_the_kairo_plan_decides_and_says_how_sure_it_is(self):
        plan = ReportPlan.model_validate(_r5())
        assert (plan.mode, plan.headline_kind, plan.decision.confidence) == ("decide", "decision", "high")
        assert [disclosure for flag in plan.flag for disclosure in flag.discloses] == [
            "late_criteria",
            "judge_uncalibrated",
        ]


def _kairo(**changes: Any) -> dict[str, Any]:
    plan = _r5()
    plan.update(changes)
    return plan


def _refused(plan: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ReportPlan.model_validate(plan)


class TestFieldConstraints:
    def test_at_most_three_summary_charts(self):
        summary = _r5()["summary"]
        _refused(_kairo(summary=[*summary, *summary]))

    @pytest.mark.parametrize("count", [2, 10])
    def test_method_holds_three_to_nine_items(self, count: int):
        item = _r5()["method"][0]
        _refused(_kairo(method=[item] * count))

    @pytest.mark.parametrize("count", [0, 6])
    def test_the_options_table_holds_one_to_five_rows(self, count: int):
        options = _r5()["options"]
        _refused(_kairo(options={**options, "rows": [options["rows"][0]] * count}))

    @pytest.mark.parametrize("count", [3, 5])
    def test_the_decision_holds_exactly_four_slots(self, count: int):
        decision = _r5()["decision"]
        _refused(_kairo(decision={**decision, "slots": (decision["slots"] * 2)[:count]}))

    @pytest.mark.parametrize("count", [0, 4])
    def test_explore_holds_one_to_three_controls(self, count: int):
        explore = _r5()["explore"][0]
        _refused(_kairo(explore=[{**explore, "controls": (explore["controls"] * 2)[:count]}]))

    @pytest.mark.parametrize("field", ["flag", "explore"])
    def test_an_optional_part_is_at_most_one_and_never_null(self, field: str):
        _refused(_kairo(**{field: _r5()[field] * 2}))
        _refused(_kairo(**{field: None}))
        ReportPlan.model_validate(_kairo(**{field: []}))

    def test_a_flag_carries_no_confidence(self):
        flag = _r5()["flag"][0]
        _refused(_kairo(flag=[{**flag, "confidence": "high"}]))

    @pytest.mark.parametrize("disclosure", ["late", "confound:", "judge_uncalibrated:x", ""])
    def test_a_disclosure_off_the_vocabulary_is_refused(self, disclosure: str):
        flag = _r5()["flag"][0]
        _refused(_kairo(flag=[{**flag, "discloses": [disclosure]}]))

    def test_a_named_disclosure_is_accepted(self):
        flag = _r5()["flag"][0]
        ReportPlan.model_validate(
            _kairo(flag=[{**flag, "discloses": ["confound:observed:reasoning_ratio", "short_run:01a170ab"]}])
        )

    def test_an_empty_id_is_refused(self):
        options = _r5()["options"]
        _refused(_kairo(options={**options, "rows": [{"option": "", "role": "recommended"}]}))

    def test_a_lede_over_its_length_is_refused(self):
        _refused(_kairo(lede="x" * 421))

    def test_a_domain_of_one_number_is_refused(self):
        summary = _r5()["summary"]
        _refused(_kairo(summary=[{**summary[0], "domain": [0.0]}]))

    def test_a_confidence_off_the_memos_tiers_is_refused(self):
        decision = _r5()["decision"]
        _refused(_kairo(decision={**decision, "confidence": "moderate"}))


class TestTheVocabularies:
    def test_the_chart_kinds_are_the_kits_and_a_table(self):
        kinds = get_args(report_plan.PlannedChart.model_fields["kind"].annotation)
        assert set(kinds) == set(KIT_CHARTS) | {"table"}

    def test_the_reference_kinds_are_the_kits(self):
        kinds = get_args(ChartReference.model_fields["kind"].annotation)
        assert set(kinds) == set(get_args(KitReference.model_fields["kind"].annotation))

    def test_the_confidence_tiers_are_the_memos(self):
        assert get_args(Decision.model_fields["confidence"].annotation) == get_args(authored.Confidence)
        assert get_args(DurableClaim.model_fields["confidence"].annotation) == get_args(authored.Confidence)


class TestTheDocumentationCopy:
    def test_it_is_a_valid_schema_and_the_kairo_plan_meets_it(self):
        jsonschema.Draft202012Validator.check_schema(_docs())
        jsonschema.validate(_r5(), _docs())

    @pytest.mark.parametrize("path", list(DOCUMENTED), ids=lambda path: "/".join(path) or "plan")
    def test_each_object_has_the_models_fields_all_required(self, path: tuple[str, ...]):
        node = _docs_node(path)
        fields = {field.alias or name for name, field in DOCUMENTED[path].model_fields.items()}
        assert set(node["properties"]) == fields
        assert set(node["required"]) == fields
        assert node["additionalProperties"] is False

    @pytest.mark.parametrize("path", list(DOCUMENTED), ids=lambda path: "/".join(path) or "plan")
    def test_each_property_states_the_models_enums_and_bounds(self, path: tuple[str, ...]):
        model = DOCUMENTED[path]
        properties = model.model_json_schema(by_alias=True)["properties"]
        node = _docs_node(path)
        for name, documented in node["properties"].items():
            assert _constraints(documented) == _constraints(properties[name]), f"{'/'.join(path)}/{name}"
