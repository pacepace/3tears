"""Every package model call whose reply is read into a typed shape sends that shape as a strict schema.

The proposer sent ``{"type": "json_object"}``, so a provider was free to write any object: a host whose stored
proposer prompt predates ``RubricDim.scale`` becoming required got drafts with no ``scale`` on every dimension, and
every paid draft failed validation. A strict ``json_schema`` derived from the model the reply is read into makes a
provider that enforces it unable to write that draft at all, whatever the prompt says, as the analysis writer's
contract already does. Held here for the proposer (both axes), the variation writer and the judge; the reply is
still validated on return, so a provider that cannot enforce a strict schema is caught there, as before.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from threetears.evals import kernel
from threetears.evals.gen import RUBRIC_PROPOSAL_CONTRACT, plan_draft
from threetears.evals.kernel import plan_variation_calls
from threetears.evals.kernel.provider import withhold_failure_detail
from threetears.evals.kernel.authored import response_format as authored_response_format
from threetears.evals.kernel.authored import strict_schema as authored_strict_schema
from threetears.evals.run.judge import judge_reply_format
from threetears.evals.run.simulator import SIMULATED_USER_RESPONSE_FORMAT, next_speaker_response_format
from threetears.evals.run.judge_service import JudgeContext, JudgeService
from threetears.evals.schema.models import (
    SCALE_LEVELS,
    EvalTemplate,
    JudgedArtifact,
    JudgeEvidence,
    RubricDim,
    RubricProposal,
    VariationAxis,
)

_RESET = "call prompt_reset for the slot"

#: A system prompt written for the draft's output contract, as every stored proposer prompt has to be.
_SYSTEM = f"SYSTEM TEXT\n{RUBRIC_PROPOSAL_CONTRACT.line}"


def _strict(response_format: dict[str, Any] | None) -> dict[str, Any]:
    """The schema a strict directive sends, refusing any other directive."""
    assert response_format is not None and response_format["type"] == "json_schema", response_format
    assert response_format["json_schema"]["strict"] is True
    schema: dict[str, Any] = response_format["json_schema"]["schema"]
    return schema


def _draft(*, scale: str | None) -> dict[str, Any]:
    dim: dict[str, Any] = {
        "name": "boundary.graceful_decline",
        "description": "declines without stonewalling",
        "axis": "boundary",
        "scoring_guide": {level: None for levels in SCALE_LEVELS.values() for level in levels} | {"pass": "declines"},
    }
    if scale is not None:
        dim["scale"] = scale
    return dim


def test_a_proposal_sends_the_draft_model_as_a_strict_schema_that_requires_every_dimension_s_scale() -> None:
    schema = _strict(
        plan_draft(
            reset=_RESET, axis="capability", system_prompt=_SYSTEM, subject_feed="F", catalog_feed="C"
        ).response_format
    )
    rubric_dim = schema["$defs"]["RubricDim"]

    assert "scale" in rubric_dim["required"], "a draft with no scale on a dimension is not a reply in this shape"
    guide = rubric_dim["properties"]["scoring_guide"]
    assert guide["additionalProperties"] is False
    assert set(guide["properties"]) == {level for levels in SCALE_LEVELS.values() for level in levels}
    jsonschema.validate(_draft(scale="pass_fail"), {"$ref": "#/$defs/RubricDim", "$defs": schema["$defs"]})
    with pytest.raises(jsonschema.ValidationError, match="'scale' is a required property"):
        jsonschema.validate(_draft(scale=None), {"$ref": "#/$defs/RubricDim", "$defs": schema["$defs"]})


def test_the_proposal_schema_names_every_field_of_the_draft_model() -> None:
    """Derived from RubricProposal, so a field the model gains is a field the schema asks for."""
    schema = _strict(
        plan_draft(
            reset=_RESET, axis="capability", system_prompt=_SYSTEM, subject_feed="F", catalog_feed="C"
        ).response_format
    )
    assert set(schema["properties"]) == set(RubricProposal.model_fields)
    assert set(schema["$defs"]["RubricDim"]["properties"]) == set(RubricDim.model_fields)


def test_a_variation_call_sends_its_values_envelope_as_a_strict_schema() -> None:
    template = EvalTemplate(
        scope_id="s",
        name="t",
        intent="i",
        candidate_kind="k",
        variation_axes=[VariationAxis(name="x", generator="llm")],
    )
    (call,) = plan_variation_calls(template, 2, []).values()
    schema = _strict(call.response_format)

    jsonschema.validate({"values": ["a", "b"]}, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(["a", "b"], schema)


def _judge_request(scale: str) -> Any:
    service = JudgeService(client_factory=lambda model, temperature: None, failure_describer=withhold_failure_detail)
    context = JudgeContext(
        case_id="tc-1",
        intent="book me a table",
        variation={},
        goal_outcomes=[],
        judged_artifact=JudgedArtifact.TRANSCRIPT,
        judge_evidence=JudgeEvidence(subject="S", case_material="M", artifact="A"),
    )
    return service.dimension_request(RubricDim(name="tone.warm", description="warm", scale=scale), context)


@pytest.mark.parametrize(
    ("scale", "answers", "refused"),
    [
        ("ordinal", [1, 5, "cannot_tell"], [0, 6, "pass", "3"]),
        ("pass_fail", ["pass", "fail", "cannot_tell"], [1, "maybe"]),
    ],
)
def test_a_judge_request_sends_its_dimension_s_answers_as_a_strict_schema(
    scale: str, answers: list[Any], refused: list[Any]
) -> None:
    schema = _strict(_judge_request(scale).response_format)

    for answer in answers:
        jsonschema.validate({"reasoning": "r", "criteria_scores": {"tone.warm": answer}}, schema)
    for answer in refused:
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"reasoning": "r", "criteria_scores": {"tone.warm": answer}}, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"reasoning": "r", "criteria_scores": {"tone.cold": answers[0]}}, schema)


async def test_a_judge_call_sends_the_schema_its_request_was_priced_with() -> None:
    """Built from the same criterion, so the judge repeat's admission (matched by equality) finds the call it priced."""
    sent: list[Any] = []

    class _Client:
        async def generate(self, *, system: str, user: str, response_format: Any = None) -> Any:
            sent.append(response_format)
            raise RuntimeError("no provider here")

    request = _judge_request("pass_fail")
    service = JudgeService(
        client_factory=lambda model, temperature: _Client(), failure_describer=withhold_failure_detail
    )
    await service.score_request(request, case_id="tc-1")

    assert sent == [request.response_format]


def test_a_scorer_offering_no_cannot_tell_is_sent_a_schema_without_it() -> None:
    schema = _strict(
        judge_reply_format([{"name": "a"}, {"name": "b", "scale": "pass_fail"}], cannot_tell_offered=False)
    )
    jsonschema.validate({"reasoning": "r", "criteria_scores": {"a": 3, "b": "fail"}}, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"reasoning": "r", "criteria_scores": {"a": "cannot_tell", "b": "fail"}}, schema)


def test_no_package_call_sends_the_unshaped_json_object_directive() -> None:
    """Named nowhere but where it is defined and exported: every call that reads JSON sends the shape it reads."""
    source = Path(kernel.__file__).resolve().parent.parent
    naming = []
    for path in sorted(source.rglob("*.py")):
        if path.relative_to(source).parts[0] == "schema":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Name) and node.id == "JSON_OBJECT_RESPONSE_FORMAT":
                naming.append(f"{path.relative_to(source)}:{node.lineno}")
    assert naming == []


# =============================================================================
# What is sent is the portable strict subset; what is left out is enforced on return
# =============================================================================


def _keywords(node: Any, found: set[str]) -> set[str]:
    """Every keyword a schema's nodes carry — a property's or a definition's name is a name, not a keyword."""
    if isinstance(node, list):
        for item in node:
            _keywords(item, found)
    elif isinstance(node, dict):
        for key, value in node.items():
            found.add(key)
            if key in ("properties", "$defs"):
                for child in value.values():
                    _keywords(child, found)
            elif key not in ("enum", "required", "description"):
                _keywords(value, found)
    return found


def _every_directive() -> dict[str, dict[str, Any] | None]:
    """Every strict directive the package sends, by the call that sends it."""
    template = EvalTemplate(
        scope_id="s",
        name="t",
        intent="i",
        candidate_kind="k",
        variation_axes=[VariationAxis(name="x", generator="llm")],
    )
    (variation,) = plan_variation_calls(template, 2, []).values()
    return {
        "proposal": plan_draft(
            reset=_RESET, axis="capability", system_prompt=_SYSTEM, subject_feed="F", catalog_feed="C"
        ).response_format,
        "variation": variation.response_format,
        "judge (ordinal)": _judge_request("ordinal").response_format,
        "judge (pass_fail)": _judge_request("pass_fail").response_format,
        "simulated user": SIMULATED_USER_RESPONSE_FORMAT,
        "next speaker": next_speaker_response_format(["alice", "bob"]),
        "analysis": authored_response_format(["scope"], ["delta_table"]),
    }


@pytest.mark.parametrize("call", list(_every_directive()))
def test_every_directive_sends_only_keywords_both_providers_strict_modes_accept(call: str) -> None:
    """``minLength``, ``minimum`` and ``maximum`` rode in the proposal's schema, which a strict mode refuses or demotes."""
    schema = _strict(_every_directive()[call])
    assert _keywords(schema, set()) <= kernel.PORTABLE_STRICT_KEYWORDS, sorted(
        _keywords(schema, set()) - kernel.PORTABLE_STRICT_KEYWORDS
    )


def test_a_constraint_left_out_of_the_schema_is_still_enforced_on_return() -> None:
    """The sent schema no longer says a dimension's description is non-empty; the reply's validation still does."""
    schema = _strict(
        plan_draft(
            reset=_RESET, axis="capability", system_prompt=_SYSTEM, subject_feed="F", catalog_feed="C"
        ).response_format
    )
    empty = _draft(scale="pass_fail") | {"description": ""}
    jsonschema.validate(empty, {"$ref": "#/$defs/RubricDim", "$defs": schema["$defs"]})
    with pytest.raises(ValueError, match="description"):
        RubricDim.model_validate(empty)


def test_a_const_is_sent_as_its_one_value_enum_and_a_structural_keyword_is_refused() -> None:
    assert authored_strict_schema({"type": "string", "const": "x", "minLength": 1}) == {"type": "string", "enum": ["x"]}
    with pytest.raises(kernel.UnportableSchema, match="'allOf'"):
        authored_strict_schema({"type": "object", "properties": {"a": {"allOf": [{"type": "string"}]}}})
