"""Actions as tools of their own: the cut a loop running a model in process hands it.

Contracts held here (``threetears.evals.actions.action_tools``, and the ``standalone`` keyword of
``MountedTool.call`` and ``actions.render``):

- **every analyst action is a tool** — the ones present now and every one a later chunk registers, read off the
  analyst's actions and the toolsets — named for its action, described by its summary, detail and example's
  arguments, whose input schema has no titles, admits nothing beyond the parameters and validates its own example;
- **a call is the mounted tool's call**: on every fixture snapshot, and for the freeze on a campaign, the outcome a
  tool returns is the one the analyst's MCP tool returns for the same arguments with ``action`` added;
- **its refusals read as its own tool's**: no ``action=``, no ``help``, an example of the tool's arguments alone and
  another action a remedy names as the tool of that name, where the MCP refusal of the same call shows the
  ``action`` key; a call naming ``action`` or ``topic`` is refused and runs nothing;
- **the cut**: an unknown or repeated name raises, each action is taken from a tool of its own cut, and one parameter
  name meaning two things on a cut raises as it does at mount.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

import jsonschema
import pytest
from pydantic import Field

from packages.evals.tests.analyst_support import action_calls, fixture_snapshots, read_actions
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, OpsFixture, ops_fixture
from threetears.evals.actions import (
    Action,
    ActionCatalogue,
    ActionOutcome,
    ActionTool,
    Caller,
    MountedTool,
    action_tools,
    analyst_actions,
    analyst_tools,
    analyst_toolset,
    editor_toolset,
    eval_catalogue,
    standard_tools,
)
from threetears.evals.actions.render import render_cited
from threetears.evals.analysis.analyst.snapshot import frozen_input
from threetears.evals.kernel import EvalReportSnapshot
from threetears.evals.ops import OpsHost
from threetears.evals.schema.base import EvalBaseModel

#: An id no snapshot has.
_NO_SNAPSHOT = "report-snapshot:none-such"


def _analyst_tool() -> MountedTool:
    (spec,) = analyst_tools()
    return eval_catalogue().mount(spec)


def _tools() -> dict[str, ActionTool]:
    """Every analyst action, the freeze among them, as a tool of its own, by name."""
    names = [action.name for action in analyst_actions()]
    return {tool.name: tool for tool in action_tools(eval_catalogue(), names)}


def _seeded() -> tuple[OpsFixture, dict[str, EvalReportSnapshot]]:
    """The toy host, with every fixture snapshot stored in its scope."""
    fixture = ops_fixture()
    snapshots = fixture_snapshots(TOYHOST_SCOPE)
    for snapshot in snapshots.values():
        fixture.host.eval_host.storage.save_report_snapshot(snapshot)
    return fixture, snapshots


def _seen(outcome: ActionOutcome) -> tuple[Any, ...]:
    """Everything an outcome carries, to compare two calls by."""
    return (outcome.text, outcome.structured, outcome.refusal, outcome.action, outcome.text_field, outcome.ending)


def _without_action(arguments: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in arguments.items() if key != "action"}


def _titled(schema: Any) -> Iterator[str]:
    """Where a schema still carries a ``title``: every key but the names under ``properties``."""
    if isinstance(schema, dict):
        for key, value in schema.items():
            if key == "title":
                yield key
            elif key == "properties":
                for property_schema in value.values():
                    yield from _titled(property_schema)
            else:
                yield from _titled(value)
    elif isinstance(schema, list):
        for item in schema:
            yield from _titled(item)


def _assert_standalone(text: str) -> None:
    """A text of a tool of its own: it shows no ``action`` key, and sends nobody to ``help``."""
    assert "action=" not in text and '"action"' not in text, text
    assert "help" not in text.lower(), text


# =============================================================================
# Every analyst action is a tool
# =============================================================================


def test_every_analyst_action_is_a_tool_named_described_and_schemed_by_its_action() -> None:
    tools = _tools()
    assert list(tools) == [action.name for action in analyst_actions()]
    assert {"report_snapshot_freeze", "capability_gap"} <= set(tools)
    for name, tool in tools.items():
        action = tool.action
        assert action.name == name and tool.mounted.action(name) is action
        assert tool.description.startswith(action.summary) and action.detail in tool.description
        assert tool.description.endswith(f"Example arguments: {json.dumps(dict(action.example))}")
        assert "action=" not in tool.description and '"action"' not in tool.description, name

        schema = tool.input_schema()
        assert schema["additionalProperties"] is False and schema["type"] == "object", name
        assert list(_titled(schema)) == [], name
        assert set(schema["properties"]) == set(action.params.model_fields), name
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.validate(dict(action.example), schema)
        with pytest.raises(jsonschema.ValidationError, match="Additional properties"):
            jsonschema.validate({**action.example, "action": name}, schema)


def test_a_parameter_named_title_keeps_its_name_while_every_title_goes() -> None:
    """The control for the title walk: a parameter called ``title`` is a parameter, not a schema's title."""

    class _Titled(EvalBaseModel):
        """A call with a parameter named title."""

        title: str = Field(default="", description="The block's title.")

    tool = _one_tool_catalogue(_Titled)
    schema = tool.input_schema()
    assert schema["properties"] == {"title": {"default": "", "description": "The block's title.", "type": "string"}}
    assert "title" not in schema and list(_titled(schema)) == []


def test_the_toolsets_are_tools_in_the_order_they_name_them() -> None:
    for toolset in (analyst_toolset(), editor_toolset()):
        tools = action_tools(eval_catalogue(), toolset)
        assert tuple(tool.name for tool in tools) == toolset
        assert all(tool.mounted.spec.cut == "analyst" for tool in tools)


# =============================================================================
# A call is the mounted tool's call
# =============================================================================


async def test_every_read_tool_answers_as_the_mcp_tool_on_every_fixture_snapshot() -> None:
    fixture, snapshots = off_loop(_seeded)
    tools, mcp = _tools(), _analyst_tool()
    actions = read_actions()
    assert {action.name for action in actions} >= {"capability_gap"}
    for name, snapshot in snapshots.items():
        for action in actions:
            for arguments in action_calls(action, frozen_input(snapshot), snapshot.id):
                where = f"{action.name} on {name}"
                standalone = await tools[action.name].call(_without_action(arguments), host=fixture.host, caller=CALLER)
                mounted = await mcp.call(arguments, host=fixture.host, caller=CALLER)
                assert standalone.ending == "ok", f"{where}: {standalone.text}"
                assert _seen(standalone) == _seen(mounted), where
                assert "action=" not in standalone.text, where


async def test_the_freeze_tool_freezes_as_the_mcp_tool_does() -> None:
    fixture = off_loop(ops_fixture)
    freeze, mcp = _tools()["report_snapshot_freeze"], _analyst_tool()
    arguments = {"campaign_id": fixture.campaign.id}

    first = await mcp.call({**arguments, "action": freeze.name}, host=fixture.host, caller=CALLER)
    standalone = await freeze.call(arguments, host=fixture.host, caller=CALLER)
    again = await mcp.call({**arguments, "action": freeze.name}, host=fixture.host, caller=CALLER)

    assert first.structured is not None and first.structured["reused"] is False, first.text
    assert standalone.structured is not None and standalone.structured["reused"] is True
    assert _seen(standalone) == _seen(again)
    assert standalone.text == render_cited(freeze.action.result.model_validate(standalone.structured))


# =============================================================================
# Its refusals read as its own tool's
# =============================================================================

#: Calls each refused before or by the engine, on a tool, with what the refusal says.
_REFUSED = [
    pytest.param(
        "capability_gap",
        {"snapshot_id": _NO_SNAPSHOT, "need": "n", "why": "w", "workaround": "x", "bogus": 1},
        "capability_gap does not take bogus",
        id="an undeclared parameter",
    ),
    pytest.param(
        "capability_gap",
        {"snapshot_id": "", "need": "n", "why": "w", "workaround": "x"},
        "capability_gap was called with values it cannot take",
        id="a value that does not validate",
    ),
    pytest.param(
        "report_snapshot_freeze",
        {"campaign_id": "c", "analysis_id": "a"},
        "name a campaign_id or an analysis_id, exactly one",
        id="both sources",
    ),
    pytest.param(
        "capability_gap",
        {"snapshot_id": _NO_SNAPSHOT, "need": "n", "why": "w", "workaround": "x"},
        "report_snapshot_freeze is a tool of its own: call report_snapshot_freeze with its own parameters",
        id="an engine refusal whose remedy names another action",
    ),
]


@pytest.mark.parametrize(("name", "arguments", "says"), _REFUSED)
async def test_a_refused_call_reads_as_its_own_tools_refusal_where_mcp_shows_the_action_key(
    name: str, arguments: dict[str, Any], says: str
) -> None:
    fixture = off_loop(ops_fixture)
    tool = _tools()[name]
    standalone = await tool.call(arguments, host=fixture.host, caller=CALLER)
    mounted = await _analyst_tool().call({**arguments, "action": name}, host=fixture.host, caller=CALLER)

    assert standalone.ending == "refused" and standalone.refusal == mounted.refusal
    assert standalone.action == mounted.action == name
    assert says in standalone.text
    _assert_standalone(standalone.text)
    if "Example: " in mounted.text:
        assert f'Example: {{"action": "{name}", ' in mounted.text, "the control: MCP shows the action key"
        assert f"Example: {json.dumps(dict(tool.action.example))}" in standalone.text
    assert standalone.text.splitlines()[0] == mounted.text.splitlines()[0]


async def test_a_remedy_naming_a_standard_action_names_it_as_a_tool_not_as_another_mcp_tools() -> None:
    fixture = off_loop(ops_fixture)
    tool = _tools()["report_snapshot_freeze"]
    arguments = {"campaign_id": "no-such-campaign"}
    standalone = await tool.call(arguments, host=fixture.host, caller=CALLER)
    mounted = await _analyst_tool().call({**arguments, "action": tool.name}, host=fixture.host, caller=CALLER)

    assert standalone.refusal is not None and standalone.refusal.kind == "not_found"
    assert "campaigns_list is not an action of tool evals_analyst" in mounted.text, "the control: MCP says where"
    assert "campaigns_list is a tool of its own: call campaigns_list with its own parameters" in standalone.text
    assert "evals_analyst" not in standalone.text and "standard tools" not in standalone.text
    _assert_standalone(standalone.text)


@pytest.mark.parametrize("reserved", ["action", "topic"])
async def test_a_call_naming_action_or_topic_is_refused_and_runs_nothing(
    reserved: str, caplog: pytest.LogCaptureFixture
) -> None:
    fixture, snapshots = off_loop(_seeded)
    tool = _tools()["capability_gap"]
    arguments = {"snapshot_id": snapshots["r1"].id, "need": "n", "why": "w", "workaround": "x"}
    assert (await tool.call(arguments, host=fixture.host, caller=CALLER)).ending == "ok", "the control: it runs"

    with caplog.at_level(logging.INFO, logger="threetears.evals.analyst"):
        refused = await tool.call({**arguments, reserved: "report_snapshot_freeze"}, host=fixture.host, caller=CALLER)
    assert refused.ending == "refused" and refused.structured is None
    assert refused.refusal is not None and (refused.refusal.kind, refused.refusal.status) == ("validation_failed", 422)
    assert refused.action == "capability_gap"
    assert refused.text.startswith(f"refused: capability_gap does not take {reserved}.")
    _assert_standalone(refused.text)
    assert [record for record in caplog.records if record.name == "threetears.evals.analyst"] == []


# =============================================================================
# The cut
# =============================================================================


@pytest.mark.parametrize(
    ("names", "message"),
    [
        (["capability_gap", "no_such_action"], "no action of the catalogue is named 'no_such_action'"),
        (["capability_gap", "capability_gap"], "actions named twice: capability_gap"),
    ],
    ids=["an unknown name", "a repeated name"],
)
def test_an_unknown_or_repeated_name_raises(names: list[str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        action_tools(eval_catalogue(), names)


async def test_each_action_is_taken_from_a_tool_of_its_own_cut() -> None:
    fixture = off_loop(ops_fixture)
    gap, campaigns = action_tools(eval_catalogue(), ["capability_gap", "campaigns_list"])
    assert (gap.mounted.spec.cut, campaigns.mounted.spec.cut) == ("analyst", None)
    standalone = await campaigns.call({}, host=fixture.host, caller=CALLER)
    evals = eval_catalogue().mount(standard_tools()[0])
    assert _seen(standalone) == _seen(await evals.call({"action": "campaigns_list"}, host=fixture.host, caller=CALLER))


class _Note(EvalBaseModel):
    """A note, as one action means it."""

    note: str = Field(default="", description="A note.")


class _OtherNote(EvalBaseModel):
    """A note, as another action means it."""

    note: int = Field(default=0, description="A count.")


class _Noted(EvalBaseModel):
    """What a note action returns."""

    note: str


async def _noted(host: OpsHost, caller: Caller, params: EvalBaseModel) -> _Noted:
    del host, caller
    return _Noted(note=str(getattr(params, "note", "")))


def _note_action(name: str, params: type[EvalBaseModel], *, cut: str | None = "notes") -> Action:
    return Action(
        name=name,
        summary="Note something.",
        workflow="Notes",
        permission="read",
        params=params,
        result=_Noted,
        handler=_noted,
        render=render_cited,
        example={},
        cut=cut,
    )


def _one_tool_catalogue(params: type[EvalBaseModel]) -> ActionTool:
    (tool,) = action_tools(ActionCatalogue([_note_action("note_take", params)]), ["note_take"])
    return tool


def test_one_parameter_name_meaning_two_things_on_a_cut_raises_as_at_mount() -> None:
    catalogue = ActionCatalogue([_note_action("note_take", _Note), _note_action("note_count", _OtherNote)])
    with pytest.raises(ValueError, match="parameter 'note' means two things"):
        action_tools(catalogue, ["note_take"])
    apart = ActionCatalogue([_note_action("note_take", _Note), _note_action("note_count", _OtherNote, cut="counts")])
    assert [tool.name for tool in action_tools(apart, ["note_take", "note_count"])] == ["note_take", "note_count"]


def test_a_tool_whose_mounted_tool_lacks_its_action_is_refused() -> None:
    gap = _tools()["capability_gap"]
    evals = eval_catalogue().mount(standard_tools()[0])
    with pytest.raises(ValueError, match="does not carry action 'capability_gap'"):
        ActionTool(gap.action, evals)
