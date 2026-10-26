"""A refusal's recovery fields are the next call's input: a caller builds the corrected call from them alone.

Every case here is refused once, corrected by :func:`_corrected` from the refusal's recovery fields alone (it is
DENIED the text, so a field the code forgot cannot be supplied from the prose by a test author who knew the answer),
and called again, and the corrected call must succeed. A test asserting the keys were present would pass on a block
naming the wrong parameter or offering a value the tool still refuses; a retry cannot.

Held here too: the fields ride the text's last line as compact JSON (a client may forward only the text), and a
refusal with no corrected call to build (a run that does not exist) carries none.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from packages.evals.tests.analyst_support import KAIRO, fixture_snapshots
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, OpsFixture, ops_fixture
from threetears.evals.actions import ActionOutcome, MountedTool, analyst_tools, eval_catalogue, standard_tools

#: Where the snapshot id goes in an analyst call's arguments, filled from the fixture snapshot the case names.
_SNAPSHOT = "<snapshot>"


def _seeded() -> tuple[OpsFixture, dict[str, str]]:
    """The toy host, every fixture snapshot stored in its scope, and each snapshot's id by name."""
    fixture = ops_fixture()
    snapshots = fixture_snapshots(TOYHOST_SCOPE)
    for snapshot in snapshots.values():
        fixture.host.eval_host.storage.save_report_snapshot(snapshot)
    return fixture, {name: snapshot.id for name, snapshot in snapshots.items()}


def _tool(name: str) -> MountedTool:
    specs = {spec.name: spec for spec in (*standard_tools(), *analyst_tools())}
    return eval_catalogue().mount(specs[name])


def _recovery(outcome: ActionOutcome) -> dict[str, Any]:
    """The refusal's recovery fields, asserted to be a refusal carrying some, and mirrored as its text's last line."""
    assert outcome.ending == "refused", f"the call was expected to be refused: {outcome.text}"
    assert outcome.recovery is not None, f"the refusal carries no recovery: {outcome.text}"
    data = outcome.recovery.as_data()
    assert outcome.text.splitlines()[-1] == "recovery: " + json.dumps(data, separators=(",", ":"))
    return data


def _corrected(arguments: dict[str, Any], recovery: dict[str, Any]) -> dict[str, Any]:
    """The caller's arguments, corrected using ONLY the recovery fields.

    In order of specificity: a field of a parameter (``query.arms``) with no values to take is dropped from it, and
    the parameter it sits in is then kept (its fault follows from the field's); a parameter the action does not
    accept is dropped; one with valid values takes the first (inside a list where the call sent a list); one the
    example shows takes the example's form; anything else is dropped, the one correction that changes the question
    rather than fixing the call.
    """
    corrected = dict(arguments)
    named = recovery.get("parameters", [])
    assert named, "the refusal named no parameter to correct"
    fields_of = {parameter.partition(".")[0] for parameter in named if "." in parameter}
    for parameter in named:
        top, _dot, field = parameter.partition(".")
        if not field and top in fields_of:
            continue
        if field:
            corrected[top] = {key: value for key, value in corrected[top].items() if key != field}
        elif "accepted" in recovery and parameter not in recovery["accepted"]:
            corrected.pop(parameter, None)
        elif "valid_values" in recovery:
            values = recovery["valid_values"]
            assert values, "a closed set with no members leaves no corrected call to build"
            corrected[parameter] = [values[0]] if isinstance(arguments.get(parameter), list) else values[0]
        elif "example" in recovery and parameter in recovery["example"]:
            corrected[parameter] = recovery["example"][parameter]
        else:
            corrected.pop(parameter, None)
    return corrected


async def _retry(fixture: OpsFixture, tool: MountedTool, arguments: dict[str, Any]) -> dict[str, Any]:
    """Refuse once, correct from the fields, call again; the recovery the refusal carried."""
    refused = await tool.call(arguments, host=fixture.host, caller=CALLER)
    recovery = _recovery(refused)
    answered = await tool.call(_corrected(arguments, recovery), host=fixture.host, caller=CALLER)
    assert answered.ending == "ok", f"the corrected call was refused too: {answered.text}"
    return recovery


# --------------------------------------------------------------------------
# The catalogue's own refusals, on the evals tool. Each RETRIES.
# --------------------------------------------------------------------------

_EVALS_CASES = [
    pytest.param({}, ["action"], id="no action"),
    pytest.param({"action": "runs_lst"}, ["action"], id="an unknown action"),
    pytest.param({"action": "runs_list", "run_id": "r", "confirm": "r"}, ["confirm", "run_id"], id="undeclared"),
    pytest.param({"action": "runs_list", "status": "done"}, ["status"], id="a value outside its set"),
    pytest.param({"action": "help", "run_id": "r"}, ["run_id"], id="a parameter help does not take"),
    pytest.param({"action": "help", "topic": "runs_lst"}, ["topic"], id="a help topic it does not have"),
]


@pytest.mark.parametrize(("arguments", "at_fault"), _EVALS_CASES)
async def test_a_call_the_catalogue_refuses_is_corrected_from_its_fields_alone(
    arguments: dict[str, Any], at_fault: list[str]
) -> None:
    fixture = off_loop(ops_fixture)
    recovery = await _retry(fixture, _tool("evals"), arguments)
    assert recovery["parameters"] == at_fault
    assert recovery["see"]["action"] == "help"


async def test_an_undeclared_parameter_names_what_the_action_accepts_and_an_example() -> None:
    fixture = off_loop(ops_fixture)
    refused = await _tool("evals").call({"action": "runs_list", "run_id": "r"}, host=fixture.host, caller=CALLER)
    recovery = _recovery(refused)
    action = _tool("evals").action("runs_list")
    assert action is not None
    assert recovery["accepted"] == list(action.params.model_fields)
    assert recovery["example"] == {"action": "runs_list", **action.example}
    assert recovery["see"] == {"action": "help", "topic": "runs_list"}


async def test_a_refusal_with_no_corrected_call_carries_no_recovery() -> None:
    """A run that does not exist is a next step (find one), not a call to correct: no field would build it."""
    fixture = off_loop(ops_fixture)
    refused = await _tool("evals").call(
        {"action": "run_get", "run_id": "no-such-run"}, host=fixture.host, caller=CALLER
    )
    assert refused.ending == "refused" and refused.recovery is None
    assert not refused.text.splitlines()[-1].startswith("recovery: ")


# --------------------------------------------------------------------------
# The analyst's refusals of a value the snapshot does not hold. Each RETRIES.
# --------------------------------------------------------------------------

_ANALYST_CASES = [
    pytest.param(KAIRO, {"action": "capability_gap", "why": "w", "workaround": "x"}, ["need"], id="a missing value"),
    pytest.param(KAIRO, {"action": "readings_list", "arm_filter": "no-such-arm"}, ["arm_filter"], id="an arm"),
    pytest.param(KAIRO, {"action": "contrasts_list", "aspect_filter": "nowhere"}, ["aspect_filter"], id="an aspect"),
    pytest.param(KAIRO, {"action": "measure_read", "measure_key": "no.such"}, ["measure_key"], id="a measure"),
    pytest.param(KAIRO, {"action": "examples_list", "outcome_filter": "none such"}, ["outcome_filter"], id="a class"),
    pytest.param(KAIRO, {"action": "options_dominance", "axes": ["no.such"]}, ["axes"], id="an axis"),
    pytest.param("r2", {"action": "decision_sensitivity", "parameter": "margin:nowhere"}, ["parameter"], id="a sweep"),
    pytest.param(
        KAIRO,
        {"action": "measure_breakdown", "measure_key": "no.such"},
        ["measure_key"],
        id="a breakdown's measure",
    ),
]


@pytest.mark.parametrize(("name", "arguments", "at_fault"), _ANALYST_CASES)
async def test_an_analyst_refusal_names_its_parameter_and_the_values_the_snapshot_holds(
    name: str, arguments: dict[str, Any], at_fault: list[str]
) -> None:
    fixture, snapshots = off_loop(_seeded)
    recovery = await _retry(fixture, _tool("evals_analyst"), {**arguments, "snapshot_id": snapshots[name]})
    assert recovery["parameters"] == at_fault
    assert recovery["see"] == {"action": "help", "topic": arguments["action"]}


async def test_a_block_the_page_cannot_draw_names_each_fault_and_the_catalog_that_lists_what_it_can() -> None:
    fixture, snapshots = off_loop(_seeded)
    tool = _tool("evals_analyst")
    arguments = {
        "action": "block_render",
        "snapshot_id": snapshots[KAIRO],
        "block_kind": "table",
        "query": {"source": "readings", "arms": ["no-such-arm"]},
    }
    recovery = await _retry(fixture, tool, arguments)
    assert recovery["parameters"] == ["query.arms", "query"], "the arm, and the query left selecting nothing"
    assert recovery["valid_values_from"] == "block_catalog" and tool.action("block_catalog") is not None
