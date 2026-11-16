"""An argument a model sends ill-shaped is a refused call it can retry, never a crash of the loop it runs in (#1249).

Every call's arguments are held to the action's parameters in one place (``MountedTool.call``), for every action:

- **null for an optional parameter is that parameter not given.** The seven pairs that raised a bare ``TypeError`` on
  ``null`` (an arm parameter whose length constraint sat outside its optional) answer exactly as the call that left it
  out;
- **a value that does not fit is refused** ``validation_failed`` (422), naming the parameter and what it takes: a
  wrong type, a list where a string is expected, ``null`` for a required parameter, and a value pydantic raises as a
  bare ``TypeError`` rather than a ``ValidationError`` (a declaration that lets a constraint meet a value it cannot
  measure), which also logs one WARNING naming the action;
- **a ``TypeError`` raised by the package's own code is a defect**, never a refusal: one a parameter model's own
  validator raises, or one the action's handler raises, propagates as itself;
- **in the report writer, the loop goes on**: the model reads the refusal as its tool result, and the report is
  written.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Final

import pytest
from pydantic import BeforeValidator, Field, field_validator

from packages.evals.tests.fixtures.writer_cases.analyst_cli import seeded_host
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER
from packages.evals.tests.report_writer_support import (
    FakeToolTurnClient,
    FakeTurnModel,
    call,
    kairo_id,
    reply,
    review,
    submitted,
    written,
)
from threetears.evals.actions import (
    Action,
    ActionCatalogue,
    Caller,
    MountedTool,
    ToolSpec,
    analyst_tools,
    eval_catalogue,
)
from threetears.evals.ops import OpsHost
from threetears.evals.schema.base import EvalBaseModel

#: Each action and optional parameter that raised a bare TypeError on null, with the call that reads the snapshot
#: without it: arguments every one of them answers on Kairo's snapshot.
NULL_PAIRS: Final[list[tuple[str, str, dict[str, Any]]]] = [
    ("readings_list", "arm_filter", {}),
    ("contrasts_list", "arm_filter", {}),
    ("measure_read", "arm_filter", {"measure_key": "cost_usd"}),
    ("examples_list", "arm_filter", {}),
    ("measure_breakdown", "arm_filter", {"measure_key": "persona.holds_character", "slice_by": "stratum"}),
    ("price_tradeoff", "arm_filter", {"gain_key": "cost_usd"}),
    ("price_tradeoff", "what_if", {"gain_key": "cost_usd"}),
]


def _writers_tool() -> MountedTool:
    """The analyst's tool as the report writer mounts it: read-only."""
    (reading,) = analyst_tools(writes=False)
    return eval_catalogue().mount(reading)


@pytest.mark.parametrize(("action", "parameter", "arguments"), NULL_PAIRS, ids=[f"{a}-{p}" for a, p, _ in NULL_PAIRS])
async def test_null_for_an_optional_arm_parameter_answers_as_the_call_that_left_it_out(
    action: str, parameter: str, arguments: dict[str, Any]
) -> None:
    host, tool = off_loop(seeded_host), _writers_tool()
    left_out = {"action": action, "snapshot_id": kairo_id(), **arguments}
    omitted = await tool.call(left_out, host=host, caller=CALLER)
    nulled = await tool.call({**left_out, parameter: None}, host=host, caller=CALLER)
    assert omitted.refusal is None, omitted.text
    assert (nulled.text, nulled.structured, nulled.refusal) == (omitted.text, omitted.structured, None)


@pytest.mark.parametrize(
    ("given", "parameter"),
    [
        ({"arm_filter": 3}, "arm_filter"),
        ({"arm_filter": ["arm:x"]}, "arm_filter"),
        ({"aspect_filter": {"id": "a"}}, "aspect_filter"),
        ({"snapshot_id": None}, "snapshot_id"),
    ],
    ids=["a-number-for-a-string", "a-list-for-a-string", "an-object-for-a-string", "null-for-a-required-one"],
)
async def test_a_value_that_does_not_fit_is_refused_naming_its_parameter(given: dict[str, Any], parameter: str) -> None:
    host, tool = off_loop(seeded_host), _writers_tool()
    outcome = await tool.call({"action": "readings_list", "snapshot_id": kairo_id(), **given}, host=host, caller=CALLER)
    assert outcome.refusal is not None and (outcome.refusal.kind, outcome.refusal.status) == ("validation_failed", 422)
    assert outcome.recovery is not None and outcome.recovery.parameters == (parameter,)
    assert f"- {parameter}: Input should be a valid string" in outcome.text, outcome.text


# --- the one place every action's arguments are validated ---------------------------------------------------------------


def _measured(value: Any) -> Any:
    return value


class _Unmeasured(EvalBaseModel):
    """A host's parameters declared as the arm parameters were: a length constraint outside the optional."""

    label: Annotated[str | None, BeforeValidator(_measured), Field(min_length=1, description="A label.")]


class _Broken(EvalBaseModel):
    """A host's parameters whose own validator has a bug: a TypeError raised in its own code."""

    label: str = Field(description="A label.")

    @field_validator("label")
    @classmethod
    def _sized(cls, value: str) -> str:
        return value + len(value)  # type: ignore[operator]  # the defect under test


class _Done(EvalBaseModel):
    """What the host's action answers."""

    label: str | None


async def _labelled(host: OpsHost, caller: Caller, params: Any) -> _Done:
    return _Done(label=params.label)


async def _crashes(host: OpsHost, caller: Caller, params: Any) -> _Done:
    return _Done(label=params.label + 1)


def _tool(params: type[EvalBaseModel], handler: Any = _labelled) -> MountedTool:
    action = Action(
        name="label_read",
        summary="Read a label.",
        workflow="Host",
        permission="read",
        params=params,
        result=_Done,
        handler=handler,
        render=lambda result: f"label: {result.label}",
        example={"label": "x"},
    )
    return ActionCatalogue([action]).mount(ToolSpec(name="labels", description="d", permissions=frozenset({"read"})))


async def test_a_value_pydantic_raises_as_a_bare_type_error_is_refused_naming_its_parameter(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with pytest.raises(TypeError, match="min_length"):
        _Unmeasured.model_validate({"label": None})  # the control: pydantic raises it bare
    with caplog.at_level(logging.WARNING, logger="threetears.evals.actions.catalogue"):
        outcome = await _tool(_Unmeasured).call(
            {"action": "label_read", "label": None}, host=off_loop(seeded_host), caller=CALLER
        )
    assert outcome.refusal is not None and outcome.refusal.kind == "validation_failed"
    assert outcome.recovery is not None and outcome.recovery.parameters == ("label",)
    assert "- label: Input given as null does not fit; it takes optional string" in outcome.text, outcome.text
    (warned,) = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert "action label_read" in warned.getMessage() and "pydantic" in warned.getMessage()


async def test_a_type_error_a_parameter_models_own_validator_raises_is_a_defect_and_propagates() -> None:
    with pytest.raises(TypeError):
        await _tool(_Broken).call({"action": "label_read", "label": "x"}, host=off_loop(seeded_host), caller=CALLER)


async def test_a_type_error_the_actions_own_logic_raises_is_a_defect_and_propagates() -> None:
    class Plain(EvalBaseModel):
        label: str = Field(description="A label.")

    with pytest.raises(TypeError):
        await _tool(Plain, _crashes).call(
            {"action": "label_read", "label": "x"}, host=off_loop(seeded_host), caller=CALLER
        )


# --- in the report writer -----------------------------------------------------------------------------------------------


async def test_the_writer_loop_goes_on_past_an_ill_shaped_argument_and_the_model_reads_the_refusal() -> None:
    snap = kairo_id()
    analyst = FakeToolTurnClient(
        reply(
            call(1, "readings_list", snap, arm_filter=None),
            call(2, "price_tradeoff", snap, gain_key="cost_usd", arm_filter=None, what_if=None),
            call(3, "readings_list", snap, arm_filter=["arm:September"]),
        ),
        reply(submitted()),
    )
    model = FakeTurnModel(
        analyst=analyst, editor_1=FakeToolTurnClient(reply(call(9, "readings_list", snap)), reply(review()))
    )

    result = await written(model)

    assert result.ending == "stored", result.outcome.reason
    answered = analyst.requests[1]["messages"][-1]["content"]
    results = {block["tool_use_id"]: block for block in answered if block["type"] == "tool_result"}
    assert not results["toolu_001"].get("is_error") and not results["toolu_002"].get("is_error")
    refused = results["toolu_003"]
    assert refused.get("is_error") is True
    text = refused["content"] if isinstance(refused["content"], str) else refused["content"][0]["text"]
    assert "readings_list was called with values it cannot take" in text and "- arm_filter:" in text, text
