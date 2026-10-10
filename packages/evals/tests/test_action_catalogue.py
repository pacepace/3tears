"""The action catalogue: the engine's actions, the tools cut from them, help, and every refusal a call can meet.

Driven over the toy host through :meth:`MountedTool.call` — the one path every transport takes — so what
is pinned here holds for FastMCP (``test_fastmcp_transport.py``) and any adapter after it:

- **Every action is classed, and the tools split on class.** ``evals`` carries read, spend and write;
  ``evals_admin`` the destructive ones; a read-only tool only reads. The host supplies the prefix.
- **Help is generated from the catalogue**: an index grouped by workflow, and a page per action whose
  example is a call that validates.
- **Errors teach.** No action, an unknown one, one on another tool, an undeclared parameter, a bad
  value, an engine refusal — each says what was wrong and what would be right.
- **A host contributes actions** that mount and run like the engine's.
- **Every construction refusal fires** — a name that is not ``noun_verb``, an undescribed or nested or
  reserved parameter, a destructive action with no ``confirm``, long work that returns no job, an
  example that is not a valid call, a duplicate name, one parameter name meaning two things on a tool.
"""

from __future__ import annotations

from threetears.evals.schema import DEFAULT_LAUNCH_K_RUNS

from typing import Any

import pytest
from pydantic import BaseModel, Field

from threetears.evals.actions import (
    Action,
    ActionCatalogue,
    Caller,
    ToolSpec,
    engine_actions,
    eval_catalogue,
    read_only_tools,
    standard_tools,
)
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.ops import JobsStarted, OpsHost, RunLine
from packages.evals.tests.ops_support import CALLER, ops_fixture


class _Echo(EvalBaseModel):
    """A host action's parameters."""

    note: str = Field(description="What to echo back.")


class _Echoed(EvalBaseModel):
    """A host action's result."""

    note: str
    scope_id: str


async def _echo(host: OpsHost, caller: Caller, params: _Echo) -> _Echoed:
    return _Echoed(note=params.note, scope_id=caller.scope_id)


def _action(**overrides: Any) -> Action:
    """A valid host action, with ``overrides`` applied — the subject of every construction refusal."""
    fields: dict[str, Any] = {
        "name": "note_echo",
        "summary": "Echo a note back.",
        "workflow": "Host",
        "permission": "read",
        "params": _Echo,
        "result": _Echoed,
        "handler": _echo,
        "render": lambda result: f"echo: {result.note}",
        "example": {"note": "hello"},
    }
    return Action(**(fields | overrides))


@pytest.fixture
def tools() -> dict[str, Any]:
    mounted = eval_catalogue().mount_all(standard_tools())
    return {tool.name: tool for tool in mounted}


# =============================================================================
# The catalogue, and the tools cut from it
# =============================================================================


def test_every_engine_action_is_noun_verb_and_classed() -> None:
    names = {action.name: action.permission for action in engine_actions()}
    assert names == {
        "templates_list": "read",
        "runs_list": "read",
        "campaigns_list": "read",
        "case_sets_list": "read",
        "case_set_mint": "write",
        "run_launch": "spend",
        "sweep_launch": "spend",
        "launch_estimate": "read",
        "job_poll": "read",
        "job_cancel": "write",
        "run_get": "read",
        "results_list": "read",
        "result_get": "read",
        "campaign_create": "write",
        "analysis_generate": "spend",
        "analysis_estimate": "read",
        "judge_second": "spend",
        "judge_second_estimate": "read",
        "judge_drift_check": "spend",
        "judge_temperature": "spend",
        "judge_temperature_estimate": "read",
        "analyses_list": "read",
        "insights_list": "read",
        "insight_get": "read",
        "analyses_undescribable": "read",
        "report_read": "read",
        "bars_propose": "read",
        "reporter_case_freeze": "write",
        "reporter_cases_list": "read",
        "judge_cases_freeze": "write",
        "judge_profiles_record": "write",
        "judge_profiles_list": "read",
        "scope_pivot": "read",
        "runs_compare": "read",
        "runs_bisect": "read",
        "scope_history": "read",
        "scope_frontier": "read",
        "scope_out_of_run_spend": "read",
        "scope_export": "read",
        "run_archive": "write",
        "campaign_archive": "write",
        "analysis_archive": "write",
        "result_rate": "write",
        "reporter_case_archive": "write",
        "run_delete": "destructive",
        "analysis_delete": "destructive",
        "insight_delete": "destructive",
    }
    assert {action.name for action in engine_actions() if action.long_running} == {
        "run_launch",
        "sweep_launch",
        "analysis_generate",
    }


def test_the_standard_tools_split_on_class_and_take_the_hosts_prefix() -> None:
    evals, admin = eval_catalogue().mount_all(standard_tools("lab"))
    assert (evals.name, admin.name) == ("lab", "lab_admin")
    assert {action.permission for action in evals.actions} == {"read", "spend", "write"}
    assert [action.name for action in admin.actions] == ["run_delete", "analysis_delete", "insight_delete"]
    assert evals.hints.open_world and not evals.hints.destructive and not evals.hints.read_only
    assert admin.hints.destructive


def test_a_read_only_tool_carries_only_reads() -> None:
    (tool,) = eval_catalogue().mount_all(read_only_tools())
    assert tool.hints.read_only
    assert {action.permission for action in tool.actions} == {"read"}
    assert tool.action("run_launch") is None and tool.action("job_poll") is not None


def test_the_input_schema_is_the_action_enum_and_the_flat_parameters(tools: dict[str, Any]) -> None:
    schema = tools["evals"].input_schema()
    assert schema["required"] == ["action"] and schema["additionalProperties"] is False
    assert schema["properties"]["action"]["enum"][0] == "help"
    assert set(schema["properties"]["action"]["enum"][1:]) == {a.name for a in tools["evals"].actions}
    flat = {name for name in schema["properties"] if name not in {"action", "topic"}}
    declared = {name for action in tools["evals"].actions for name in action.params.model_fields}
    assert flat == declared
    assert all(schema["properties"][name]["description"] for name in flat)


# =============================================================================
# Help, generated
# =============================================================================


async def test_help_is_the_index_grouped_by_workflow(tools: dict[str, Any]) -> None:
    outcome = await tools["evals"].call({"action": "help"}, host=ops_fixture().host, caller=CALLER)
    assert not outcome.is_error
    text = outcome.text
    for heading in ("## Find what is there", "## Run and watch", "## Analyse and report", "## Curate"):
        assert heading in text
    for action in tools["evals"].actions:
        assert f"- {action.name} ({action.permission}" in text
    assert text.index("## Run and watch") < text.index("- run_launch") < text.index("## Analyse and report")
    assert "run_delete" not in text, "the destructive actions are another tool's"


async def test_a_help_page_carries_the_parameters_and_an_example(tools: dict[str, Any]) -> None:
    outcome = await tools["evals"].call(
        {"action": "help", "topic": "run_launch"}, host=ops_fixture().host, caller=CALLER
    )
    assert "- template_id (string, required): A template's id" in outcome.text
    assert f"- k_runs (integer, default {DEFAULT_LAUNCH_K_RUNS})" in outcome.text
    assert "poll each with action='job_poll'" in outcome.text
    assert 'Example: {"action": "run_launch", "template_id": "tmpl-1"' in outcome.text


def test_every_example_is_a_call_its_action_accepts() -> None:
    """The example help prints is held at construction; this proves the hold covers every engine action."""
    for action in engine_actions():
        action.params.model_validate(dict(action.example))


# =============================================================================
# Errors that teach
# =============================================================================


async def _call(tool: Any, arguments: dict[str, Any]) -> Any:
    return await tool.call(arguments, host=ops_fixture().host, caller=CALLER)


async def test_no_action_is_refused_with_the_valid_actions(tools: dict[str, Any]) -> None:
    outcome = await _call(tools["evals"], {"run_id": "x"})
    assert outcome.is_error and "name the action to run" in outcome.text
    assert "valid actions: help, templates_list" in outcome.text


async def test_an_unknown_action_is_refused_with_the_valid_actions(tools: dict[str, Any]) -> None:
    outcome = await _call(tools["evals"], {"action": "runs_lst"})
    assert outcome.is_error and "there is no action 'runs_lst'" in outcome.text
    assert "runs_list" in outcome.text


async def test_an_action_on_another_tool_is_refused_saying_where_it_lives(tools: dict[str, Any]) -> None:
    outcome = await _call(tools["evals"], {"action": "run_delete", "run_id": "r", "confirm": "r"})
    assert outcome.is_error
    assert "run_delete is a destructive action, and tool evals carries read, spend, write actions only" in outcome.text


async def test_an_undeclared_parameter_is_refused_with_the_valid_set(tools: dict[str, Any]) -> None:
    """``confirm`` is a parameter of the tool next door, and ``run_id`` of a sibling: neither is ``runs_list``'s."""
    outcome = await _call(tools["evals"], {"action": "runs_list", "run_id": "r", "confirm": "r"})
    assert outcome.is_error and outcome.structured is None
    assert outcome.text.startswith("refused: runs_list does not take confirm, run_id.")
    assert "- status (" in outcome.text and "- include_archived (" in outcome.text
    assert 'Example: {"action": "runs_list", "status": "completed"}' in outcome.text


async def test_a_value_that_does_not_validate_is_refused_naming_the_parameter(tools: dict[str, Any]) -> None:
    outcome = await _call(tools["evals"], {"action": "runs_list", "status": "done"})
    assert outcome.is_error and "runs_list was called with values it cannot take" in outcome.text
    assert "- status:" in outcome.text and "runs_list accepts:" in outcome.text


async def test_a_missing_required_parameter_is_refused_naming_it(tools: dict[str, Any]) -> None:
    outcome = await _call(tools["evals"], {"action": "run_get"})
    assert outcome.is_error and "- run_id: Field required" in outcome.text


async def test_an_engine_refusal_is_a_refused_call_with_the_engines_reason(tools: dict[str, Any]) -> None:
    outcome = await _call(tools["evals"], {"action": "run_get", "run_id": "no-such-run"})
    assert outcome.is_error and outcome.structured is None
    assert outcome.text.startswith("refused: run_get: ") and outcome.text.endswith("'no-such-run' not found")


async def test_a_destructive_action_refuses_a_confirm_that_does_not_echo_the_id(tools: dict[str, Any]) -> None:
    fixture = ops_fixture()
    run_id = fixture.campaign.run_ids[0]
    outcome = await tools["evals_admin"].call(
        {"action": "run_delete", "run_id": run_id, "confirm": "yes"}, host=fixture.host, caller=CALLER
    )
    assert outcome.is_error and run_id in outcome.text
    assert fixture.host.eval_host.storage.load_eval_run(run_id, CALLER.scope_id) is not None, "nothing was destroyed"


async def test_help_refuses_a_parameter_it_does_not_take_and_a_topic_it_does_not_have(tools: dict[str, Any]) -> None:
    extra = await _call(tools["evals"], {"action": "help", "run_id": "r"})
    assert extra.is_error and "help takes only `topic`" in extra.text
    unknown = await _call(tools["evals"], {"action": "help", "topic": "run_delete"})
    assert unknown.is_error and "is a destructive action" in unknown.text


async def test_a_result_carries_its_text_and_its_model_as_data(tools: dict[str, Any]) -> None:
    fixture = ops_fixture()
    run_id = fixture.campaign.run_ids[0]
    outcome = await tools["evals"].call(
        {"action": "run_archive", "run_id": run_id, "archived": True}, host=fixture.host, caller=CALLER
    )
    assert not outcome.is_error and outcome.action == "run_archive"
    assert RunLine.model_validate(outcome.structured).archived is True
    assert outcome.text.startswith(f"- {run_id}:") and outcome.text.endswith(", archived")


# =============================================================================
# A host's own actions
# =============================================================================


async def test_a_host_action_mounts_and_runs_like_the_engines() -> None:
    (tool,) = eval_catalogue((_action(),)).mount_all(read_only_tools())
    outcome = await tool.call({"action": "note_echo", "note": "hi"}, host=ops_fixture().host, caller=CALLER)
    assert (outcome.text, outcome.structured) == ("echo: hi", {"note": "hi", "scope_id": CALLER.scope_id})
    assert "- note_echo (read): Echo a note back." in tool.help_index()


def test_a_host_action_cannot_take_an_engine_actions_name() -> None:
    with pytest.raises(ValueError, match="two actions are named 'run_get'"):
        eval_catalogue((_action(name="run_get"),))


def test_help_cannot_be_declared() -> None:
    """``help`` is generated per tool; the noun_verb rule is what keeps an action from taking its name."""
    with pytest.raises(ValueError, match="action 'help': an action is named noun_verb"):
        _action(name="help")


# =============================================================================
# Construction refusals
# =============================================================================


class _Nested(EvalBaseModel):
    inner: _Echo = Field(description="A nested model.")


class _Undescribed(EvalBaseModel):
    note: str


class _Reserved(EvalBaseModel):
    topic: str = Field(description="Collides with help's.")


class _OptionalConfirm(EvalBaseModel):
    confirm: str | None = Field(default=None, description="Optional, which a destructive action may not be.")


class _OtherMeaning(EvalBaseModel):
    note: int = Field(description="A note, but a number.")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"name": "echo"}, "named noun_verb"),
        ({"name": "Note_Echo"}, "named noun_verb"),
        ({"summary": "two\nlines"}, "one non-blank line"),
        ({"summary": "  "}, "one non-blank line"),
        ({"permission": "admin"}, "class 'admin' is none of"),
        ({"params": _Undescribed, "example": {"note": "x"}}, "note is not"),
        ({"params": _Nested, "example": {"inner": {"note": "x"}}}, "parameters are flat"),
        ({"params": _Reserved, "example": {"topic": "x"}}, "topic is reserved"),
        ({"permission": "destructive"}, "declares a required `confirm`"),
        ({"permission": "destructive", "params": _OptionalConfirm, "example": {}}, "declares a required `confirm`"),
        ({"long_running": True}, "long work returns JobsStarted"),
        ({"result": JobsStarted}, "long work returns JobsStarted"),
        ({"example": {"note": "x", "extra": 1}}, "its example is not a valid call"),
        ({"example": {}}, "its example is not a valid call"),
    ],
)
def test_an_action_no_transport_could_mount_honestly_is_refused(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _action(**overrides)


def test_the_valid_action_the_refusals_start_from_is_accepted() -> None:
    """The control: every refusal above differs from this action by its override alone."""
    assert _action().name == "note_echo"


def test_one_parameter_name_meaning_two_things_on_a_tool_is_refused() -> None:
    catalogue = ActionCatalogue([_action(), _action(name="note_count", params=_OtherMeaning, example={"note": 1})])
    with pytest.raises(ValueError, match="parameter 'note' means two things — note_echo declares"):
        catalogue.mount(read_only_tools()[0])


def test_the_same_name_on_two_tools_may_mean_two_things() -> None:
    """One meaning per TOOL: the read tool and the admin tool never share an input schema."""
    other = _action(name="note_count", params=_OtherMeaning, example={"note": 1}, permission="write")
    catalogue = ActionCatalogue([_action(), other])
    catalogue.mount(ToolSpec(name="reads", description="d", permissions=frozenset({"read"})))
    catalogue.mount(ToolSpec(name="writes", description="d", permissions=frozenset({"write"})))


def test_a_tool_with_no_action_of_its_classes_is_refused() -> None:
    with pytest.raises(ValueError, match="no action has those classes"):
        ActionCatalogue([_action()]).mount(ToolSpec(name="t", description="d", permissions=frozenset({"spend"})))


def test_two_tools_with_one_name_are_refused() -> None:
    with pytest.raises(ValueError, match="two tools share a name"):
        eval_catalogue().mount_all([*read_only_tools(), *read_only_tools()])


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"name": "Evals", "description": "d", "permissions": frozenset({"read"})}, "not a lowercase identifier"),
        ({"name": "evals", "description": " ", "permissions": frozenset({"read"})}, "the description is blank"),
        ({"name": "evals", "description": "d", "permissions": frozenset()}, "it mounts one or more of"),
        ({"name": "evals", "description": "d", "permissions": frozenset({"admin"})}, "it mounts one or more of"),
    ],
)
def test_a_malformed_tool_is_refused(fields: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ToolSpec(**fields)


@pytest.mark.parametrize("identity", ["", "  "])
def test_a_caller_names_who_is_calling(identity: str) -> None:
    with pytest.raises(ValueError, match="identity is blank"):
        Caller(scope_id="s", identity=identity)


def test_a_flat_parameter_model_is_any_pydantic_model() -> None:
    """A host's parameters need not use the engine's base: any flat, described model mounts."""

    class Plain(BaseModel):
        note: str = Field(description="What to echo back.")

    assert _action(params=Plain).params is Plain
