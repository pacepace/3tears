"""Refused calls: recorded apart from the calls that succeeded, read by ``refused()`` and ``refused_count()`` only.

A call a tool refused changed nothing, so it is never among ``CallLedger.calls`` and no call builtin counts it.
It is still something the candidate did: a candidate whose every call was refused must not read, to every check,
as one that never acted. So the world records each refusal where it decides it, on the ledger's refusal record,
and two builtins read that record and nothing else. A ledger that keeps no refusal record is never read as
"none refused".
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from threetears.evals.kernel.dsl import evaluate
from threetears.evals.quick import Dimension, ToolRefused, World, WorldTool, WorldTools, callable_host, run_eval
from threetears.evals.run import GoalCheckUnevaluable, get_result_trace, grade_goal_checks, list_results, recheck_result
from threetears.evals.schema import CallLedger, GoalStateOutcome, RecordedCall, RefusedCall
from threetears.evals.schema.goal_grammar import DSLError, parse, reads_call_ledger, reads_refusals, referenced_actions
from packages.evals.tests.factories import make_eval_result
from packages.evals.tests.off_loop import off_loop

SCOPE = "refused-calls-tests"

QUEUE = "player.queue_track"


def _graded(expression: str, ledger: CallLedger) -> bool:
    return evaluate(expression, end_state={}, ledger=ledger, world=None)


# =============================================================================
# The ledger
# =============================================================================


def test_a_refusal_is_recorded_apart_from_the_calls_with_its_reason_and_place_among_them() -> None:
    ledger = CallLedger(refused=[])
    ledger.record_refusal("player", "queue_track", {"track": "t1"}, reason="the queue is locked")
    ledger.record("player", "search", {"query": "jazz"})
    ledger.record_refusal("player", "queue_track", None, reason="no track given")

    assert ledger.calls == [RecordedCall(tool="player", action="search", params={"query": "jazz"})]
    assert ledger.refused == [
        RefusedCall(
            tool="player", action="queue_track", params={"track": "t1"}, reason="the queue is locked", position=0
        ),
        RefusedCall(tool="player", action="queue_track", params={}, reason="no track given", position=1),
    ]


def test_a_ledger_keeps_no_refusal_record_until_one_is_started_and_the_first_refusal_starts_it() -> None:
    ledger = CallLedger()
    assert ledger.refused is None

    ledger.record_refusal("player", "queue_track", {}, reason="locked")
    assert [refusal.reason for refusal in ledger.refused or []] == ["locked"]


def test_refused_parameters_are_insulated_from_the_caller() -> None:
    params = {"track": "t1", "tags": ["a"]}
    ledger = CallLedger(refused=[])
    ledger.record_refusal("player", "queue_track", params, reason="locked")
    params["tags"].append("b")

    assert ledger.refused is not None and ledger.refused[0].params == {"track": "t1", "tags": ["a"]}


def test_a_refusal_with_no_reason_is_refused_rather_than_recorded_blank() -> None:
    with pytest.raises(ValueError, match="reason"):
        CallLedger(refused=[]).record_refusal("player", "queue_track", {}, reason="  ")


def test_a_ledger_with_refusals_round_trips_and_one_stored_before_them_reads_as_keeping_no_record() -> None:
    ledger = CallLedger(refused=[])
    ledger.record_refusal("player", "queue_track", {"track": "t1"}, reason="locked")

    assert CallLedger.model_validate(json.loads(ledger.model_dump_json())) == ledger
    assert CallLedger.model_validate({"calls": []}).refused is None


# =============================================================================
# The goal language
# =============================================================================


def test_refused_and_refused_count_read_only_refusals_and_the_call_builtins_read_only_successes() -> None:
    ledger = CallLedger(refused=[])
    ledger.record_refusal("player", "queue_track", {"track": "t1"}, reason="locked")
    ledger.record_refusal("player", "queue_track", {"track": "t2"}, reason="locked")
    ledger.record("player", "queue_track", {"track": "t3"})

    assert _graded(f'refused_count("{QUEUE}") == 2', ledger) is True
    assert _graded(f'refused("{QUEUE}")', ledger) is True
    assert _graded('refused("player.search")', ledger) is False
    assert _graded(f'call_count("{QUEUE}") == 1', ledger) is True
    assert _graded(f'all(it.track == "t3" for it in calls("{QUEUE}"))', ledger) is True


def test_a_candidate_whose_every_call_was_refused_is_told_apart_from_one_that_never_called() -> None:
    """The production failure: every ``queue_track`` refused read, to every call check, as no attempt at all."""
    refused_every_time = CallLedger(refused=[])
    for track in ("t1", "t2", "t3"):
        refused_every_time.record_refusal("player", "queue_track", {"track": track}, reason="locked")
    never_called = CallLedger(refused=[])

    for ledger in (refused_every_time, never_called):
        assert _graded(f'call_count("{QUEUE}") == 0', ledger) is True
    assert _graded(f'refused_count("{QUEUE}") >= 1', refused_every_time) is True
    assert _graded(f'refused_count("{QUEUE}") >= 1', never_called) is False


@pytest.mark.parametrize("expression", [f'refused("{QUEUE}")', f'refused_count("{QUEUE}") == 0'])
def test_a_ledger_keeping_no_refusal_record_is_never_read_as_none_refused(expression: str) -> None:
    with pytest.raises(DSLError, match="keeps no refusal record"):
        _graded(expression, CallLedger())
    with pytest.raises(GoalCheckUnevaluable):
        grade_goal_checks([expression], ledger=CallLedger(), end_state={}, fired=None, variation={}, world=None)


@pytest.mark.parametrize("expression", ['refused("queue_track")', "refused(variation.tool)", "refused_count()"])
def test_a_refusal_builtin_takes_one_tool_action_literal_as_every_call_builtin_does(expression: str) -> None:
    with pytest.raises(DSLError):
        parse(expression)


def test_the_refusal_builtins_name_their_actions_and_read_the_ledger_for_the_authoring_gate_and_the_recheck() -> None:
    expression = f'refused_count("{QUEUE}") == 0 and call_count("player.search") >= 1'

    assert referenced_actions(expression) == (("player", "queue_track"), ("player", "search"))
    assert reads_call_ledger(expression) and reads_refusals(expression)
    assert not reads_refusals('call_count("player.search") >= 1')


# =============================================================================
# The world records each refusal where it refuses
# =============================================================================


def switch(room: dict[str, Any], to: str) -> str:
    """Turn the lamp on or off."""
    room["lamp"] = to
    return f"lamp {to}"


def _room() -> World:
    return World(
        "room",
        [Dimension("lamp", {"enum": ["on", "off"]}, "What the candidate switches.")],
        tools=[WorldTool(switch, to={"enum": ["on", "off"]})],
    )


def _start(case: Mapping[str, Any]) -> Mapping[str, Any]:
    return {"lamp": case["lamp"]}


async def test_every_call_the_world_refuses_is_recorded_before_it_is_refused_and_a_check_grades_it() -> None:
    async def sloppy(case: Mapping[str, Any], tools: WorldTools) -> str:
        with pytest.raises(ToolRefused, match="dim"):
            await tools.call("switch", to="dim")
        with pytest.raises(ToolRefused, match="no tool 'shout'"):
            await tools.call("shout", loud=True)
        return await tools.call("switch", to="on")

    world = _room()
    host = callable_host(world=world)
    checks = ['refused_count("room.switch") == 1', 'call_count("room.switch") == 1', 'not refused("room.switch")']
    summary = await run_eval(
        [{"lamp": "off"}], sloppy, host=host, world=world, seed=_start, goal_checks=checks, scope_id=SCOPE, k=1
    )

    (result,) = off_loop(list_results, host.storage, summary.run_id, SCOPE)
    trace = off_loop(get_result_trace, host.storage, result)
    assert trace is not None and trace.call_ledger is not None
    assert trace.call_ledger.calls == [RecordedCall(tool="room", action="switch", params={"to": "on"})]
    refused = trace.call_ledger.refused or []
    assert [(r.tool, r.action, r.params, r.position) for r in refused] == [
        ("room", "switch", {"to": "dim"}, 0),
        ("room", "shout", {"loud": True}, 0),
    ]
    assert "dim" in refused[0].reason and "no tool 'shout'" in refused[1].reason
    assert [(o.expression, o.passed) for o in result.goal_state_outcomes] == list(zip(checks, [True, True, False], strict=True))


async def test_a_world_cell_with_no_refusal_keeps_an_empty_record_so_a_refusal_check_grades_it() -> None:
    async def tidy(case: Mapping[str, Any], tools: WorldTools) -> str:
        return await tools.call("switch", to="on")

    world = _room()
    host = callable_host(world=world)
    check = 'refused_count("room.switch") == 0'
    summary = await run_eval(
        [{"lamp": "off"}], tidy, host=host, world=world, seed=_start, goal_checks=[check], scope_id=SCOPE, k=1
    )

    (result,) = off_loop(list_results, host.storage, summary.run_id, SCOPE)
    trace = off_loop(get_result_trace, host.storage, result)
    assert trace is not None and trace.call_ledger is not None and trace.call_ledger.refused == []
    assert [(o.expression, o.passed) for o in result.goal_state_outcomes] == [(check, True)]


# =============================================================================
# The re-check
# =============================================================================


def test_a_refusal_check_on_a_ledger_keeping_no_refusal_record_is_kept_as_stored() -> None:
    check = f'refused_count("{QUEUE}") == 0'
    report, outcomes = recheck_result(
        make_eval_result(goal_state_outcomes=[GoalStateOutcome(expression=check, passed=False)]),
        ledger=CallLedger(),
        end_state=None,
        world=None,
        variation={},
        provenance="commissioned",
    )

    assert outcomes is None and report.not_rechecked is None and report.regraded == 0
    (kept,) = report.kept_as_stored
    assert kept.expression == check and "keeps no refusal record" in kept.reason


def test_a_refusal_check_re_grades_from_a_stored_refusal_record() -> None:
    check = f'refused_count("{QUEUE}") == 0'
    ledger = CallLedger(refused=[])
    ledger.record_refusal("player", "queue_track", {}, reason="locked")
    report, outcomes = recheck_result(
        make_eval_result(goal_state_outcomes=[GoalStateOutcome(expression=check, passed=True)]),
        ledger=ledger,
        end_state=None,
        world=None,
        variation={},
        provenance="commissioned",
    )

    assert [(flip.expression, flip.was, flip.now) for flip in report.flips] == [(check, True, False)]
    assert outcomes is not None and [outcome.passed for outcome in outcomes] == [False]
