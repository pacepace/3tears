"""An argument a model sends ill-shaped is a refused call it can retry, never a crash of the loop it runs in (#1249).

Every call's arguments are held to the action's parameters in one place (``MountedTool.call``), for every action:

- **null is what each parameter declares it to be, for every action.** The seven pairs that raised a bare
  ``TypeError`` on ``null`` (an arm parameter whose length constraint sat outside its optional) declare ``null`` not
  given (``NullIsOmitted``) and answer exactly as the call that left it out. Swept over every action of the catalogue
  and every parameter it does not require: a parameter that declares so reads ``null`` as left out; one whose schema
  admits ``null`` hands it to the action as sent, told from an omission (``campaign_update``'s
  ``control_from_run_id: null`` clears the control, where leaving it out keeps it); one whose schema does not is
  refused 422 naming it. None raises, and no ``null`` reaches a pydantic constraint unchecked (the #1249 shape);
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
import re
from collections.abc import Mapping
from dataclasses import replace
from typing import Annotated, Any, Final

import pytest
from pydantic import BaseModel, BeforeValidator, Field, field_validator

from packages.evals.tests.fixtures.toyhost.campaign import toyhost_aspect_design
from packages.evals.tests.fixtures.writer_cases.analyst_cli import seeded_host
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, TOYHOST_SUBJECT, ops_fixture
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
    ActionOutcome,
    Caller,
    MountedTool,
    ToolSpec,
    analyst_tools,
    eval_catalogue,
    standard_tools,
)
from threetears.evals.actions.params import NullIsOmitted
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


# --- null, swept over every action and every parameter it does not require ----------------------------------------------

#: The catalogue's actions as a host gets them: the engine's and the analyst's.
CATALOGUE: Final = eval_catalogue()

#: Every action and every parameter it does not require, read from the catalogue itself, so an action or a parameter
#: added later is swept without a line here.
OPTIONAL: Final[list[tuple[Action, str]]] = [
    (action, name)
    for action in CATALOGUE.actions
    for name, info in action.params.model_fields.items()
    if not info.is_required()
]


class _Reached(Exception):
    """The call validated and reached its action, which stops it here: with the parameters it was handed."""

    def __init__(self, params: BaseModel) -> None:
        super().__init__("reached the handler")
        self.params = params


async def _reached(host: OpsHost, caller: Caller, params: Any) -> Any:
    raise _Reached(params)


def _alone(action: Action) -> MountedTool:
    """``action`` alone on a tool, its handler replaced by one that hands back what it was given."""
    reaching = replace(action, handler=_reached)
    spec = ToolSpec(
        name="sweep",
        description="Every optional parameter.",
        permissions=frozenset({action.permission}),
        cut=action.cut,
    )
    return ActionCatalogue([reaching]).mount(spec)


async def _sent(tool: MountedTool, arguments: Mapping[str, Any], host: OpsHost) -> BaseModel | ActionOutcome:
    """The parameters the action was handed, or the outcome of a call refused before it."""
    try:
        return await tool.call(arguments, host=host, caller=CALLER)
    except _Reached as reached:
        return reached.params


def _said(answer: BaseModel | ActionOutcome) -> str:
    return answer.text if isinstance(answer, ActionOutcome) else repr(answer)


def _admits_null(schema: Mapping[str, Any]) -> bool:
    """Whether a parameter's JSON schema, as its declaration generates it, admits ``null``."""
    branches = [*schema.get("anyOf", ()), *schema.get("oneOf", ())]
    if branches:
        return any(_admits_null(branch) for branch in branches)
    kind = schema.get("type")
    if kind is None:
        return not ({"enum", "const", "$ref", "allOf"} & schema.keys())
    return kind == "null" or (isinstance(kind, list) and "null" in kind)


def _null_is_omitted(action: Action, name: str) -> bool:
    return any(isinstance(marker, NullIsOmitted) for marker in action.params.model_fields[name].metadata)


@pytest.fixture(scope="module")
def sweep_host() -> OpsHost:
    """A host no swept call reaches: each is answered at validation or handed back from its handler."""
    return seeded_host()


@pytest.mark.parametrize(("action", "parameter"), OPTIONAL, ids=[f"{a.name}-{p}" for a, p in OPTIONAL])
async def test_null_for_every_optional_parameter_is_its_declared_meaning_or_a_refusal_naming_it(
    action: Action, parameter: str, sweep_host: OpsHost, caplog: pytest.LogCaptureFixture
) -> None:
    tool = _alone(action)
    left_out = {"action": action.name, **{k: v for k, v in action.example.items() if k != parameter}}
    with caplog.at_level(logging.WARNING, logger="threetears.evals.actions.catalogue"):
        omitted = await _sent(tool, left_out, sweep_host)
        nulled = await _sent(tool, {**left_out, parameter: None}, sweep_host)

    assert not caplog.records, f"null met a constraint pydantic could not apply to it (#1249): {caplog.records}"
    if not _admits_null(action.params.model_json_schema()["properties"][parameter]):
        assert isinstance(nulled, ActionOutcome), f"{parameter}'s schema does not admit null, yet it validated"
        assert nulled.refusal is not None and (nulled.refusal.kind, nulled.refusal.status) == ("validation_failed", 422)
        assert nulled.recovery is not None and nulled.recovery.parameters == (parameter,), nulled.text
        assert re.search(rf"^- {parameter}[:.]", nulled.text, re.MULTILINE), nulled.text
    elif isinstance(omitted, ActionOutcome):
        # The action takes this parameter or another (one of campaign_id or analysis_id), so leaving it out of the
        # example is refused: null reads as that omission, refused alike and never raised.
        assert omitted.refusal is not None and omitted.refusal.status == 422, omitted.text
        assert isinstance(nulled, ActionOutcome), f"null validated where leaving {parameter} out does not"
        assert (nulled.text, nulled.refusal, nulled.recovery) == (omitted.text, omitted.refusal, omitted.recovery)
    elif _null_is_omitted(action, parameter):
        assert isinstance(nulled, BaseModel), _said(nulled)
        assert nulled.model_dump() == omitted.model_dump(), "declared not given: read as left out"
        assert nulled.model_fields_set == omitted.model_fields_set, "and its next page's call sends what was left out"
    else:
        assert isinstance(nulled, BaseModel), f"{parameter}'s schema admits null: {_said(nulled)}"
        assert getattr(nulled, parameter) is None and parameter in nulled.model_fields_set, "null, as sent"
        assert nulled.model_dump(exclude={parameter}) == omitted.model_dump(exclude={parameter})
        assert action.params.model_fields[parameter].default is None, "an action that reads no omission reads null so"


def test_the_sweep_holds_every_kind_of_optional_parameter() -> None:
    """The sweep is not vacuous: it meets each branch, and the seven #1249 pairs are the ones declared not given."""
    declared = {(action.name, name) for action, name in OPTIONAL if _null_is_omitted(action, name)}
    admitting = [
        (action.name, name)
        for action, name in OPTIONAL
        if _admits_null(action.params.model_json_schema()["properties"][name])
    ]
    assert declared == {(action, parameter) for action, parameter, _ in NULL_PAIRS}
    assert ("campaign_update", "control_from_run_id") in admitting and len(admitting) > len(declared)
    assert len(OPTIONAL) > len(admitting), "and some optional parameter does not admit null"


async def test_null_for_campaign_updates_control_clears_it_where_leaving_it_out_keeps_it() -> None:
    """The meaning ``campaign_update`` gives ``control_from_run_id: null``: the control cleared, told from an update
    that leaves it out, which keeps it (the declaration open again once its runs are gone)."""
    fixture = off_loop(ops_fixture)
    evals = CATALOGUE.mount_all(standard_tools())[0]
    run, behavior = fixture.campaign.run_ids[0], fixture.campaign.behavior

    async def sent(action: str, **arguments: Any) -> ActionOutcome:
        outcome = await evals.call({"action": action, **arguments}, host=fixture.host, caller=CALLER)
        assert not outcome.is_error, outcome.text
        return outcome

    def control() -> object:
        campaign = off_loop(fixture.host.eval_host.storage.load_campaign, campaign_id, TOYHOST_SCOPE)
        assert campaign is not None
        return campaign.control

    created = await sent(
        "campaign_create",
        name="null clears",
        subject_id=TOYHOST_SUBJECT.subject_id,
        behavior=behavior,
        aspects=[{"label": behavior}],
        exploratory=True,
    )
    assert created.structured is not None
    campaign_id = created.structured["id"]
    design = toyhost_aspect_design().model_dump(mode="json")
    await sent("campaign_update", campaign_id=campaign_id, aspects=[{"label": behavior, "design": design}])
    await sent("campaign_runs_add", campaign_id=campaign_id, run_ids=[run])
    await sent("campaign_update", campaign_id=campaign_id, control_from_run_id=run)
    await sent("campaign_runs_remove", campaign_id=campaign_id, run_ids=[run])

    await sent("campaign_update", campaign_id=campaign_id, name="null clears, renamed")
    kept = control()
    await sent("campaign_update", campaign_id=campaign_id, control_from_run_id=None)

    assert kept is not None, "an update that leaves control_from_run_id out keeps the control"
    assert control() is None, "control_from_run_id: null clears it"


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
