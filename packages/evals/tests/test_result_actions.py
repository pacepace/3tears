"""A run's results, listed and read one at a time, through the actions a host mounts.

``run_get`` stops at a run's counts and means, so a cell whose candidate failed every action could be
found in a summary and read nowhere but the database. Driven through :meth:`MountedTool.call`, the one
path every transport takes:

- **``results_list`` pages a run's results** as light rows in a stable order — by case, then repeat, then
  id — each with its coordinates, its condition and its headline measures; ``next_offset`` reads the next
  page, ``condition_filter`` narrows to one condition and ``total`` counts what matched across every page.
- **``result_get`` reads one result whole**: the stored record with its usage rows, the condition every
  surface resolves, and its trace as the kind stored it — so each action the kind recorded as failed reads
  back failed, in the tool's own words, beside a call ledger that holds only what succeeded.
- **Both answer only in the caller's scope**: a run or result in another scope is not found, never an
  empty listing that would read as a run that produced nothing.
- **A page is bounded for an agent**: a limit above the action's ceiling is refused, naming the parameter.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.evals.actions import eval_catalogue, standard_tools
from threetears.evals.contracts import (
    TRANSCRIPT_DIM_ID,
    CallLedger,
    EvalResult,
    EvalTrace,
    GoalStateOutcome,
    ResultOutcome,
    RoleUsage,
    RubricScore,
    eval_trace_doc_id,
)
from threetears.evals.ops import ResultDetail, ResultListing, results_list
from packages.evals.tests.factories import make_eval_result, make_eval_run
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, OpsFixture, ops_fixture

RUN_ID = "run-shakedown"
OTHER_SCOPE = "another-tenant"

#: A turn in which the candidate tried three actions and the world refused every one — the shape a host's
#: kind stores, which the engine never interprets and so must hand back as written.
FAILED_TURN: dict[str, Any] = {
    "turn": 1,
    "candidate": "I'll add that to your notes and remind you tomorrow.",
    "actions": [
        {"tool": "notes", "action": "write", "success": False, "result": "notes is read-only for this account"},
        {"tool": "reminders", "action": "create", "success": False, "result": "unknown parameter 'when'"},
        {"tool": "reminders", "action": "create", "success": False, "result": "due time is in the past"},
    ],
}


@pytest.fixture
def tools() -> dict[str, Any]:
    return {tool.name: tool for tool in eval_catalogue().mount_all(standard_tools())}


def _result(result_id: str, test_case_id: str, k: int, **overrides: Any) -> EvalResult:
    return make_eval_result(
        id=result_id, scope_id=TOYHOST_SCOPE, eval_run_id=RUN_ID, test_case_id=test_case_id, k_iteration=k, **overrides
    )


def _seeded() -> OpsFixture:
    """The toy operations host, with one run of four results stored beside its corpus.

    Saved out of order, so the listing's order is the action's and not the store's. The case ``tc-a`` k=1
    result is the shakedown's cell: delivered, every action refused, a candidate usage row, a trace whose
    call ledger is empty. ``tc-a`` k=2 failed its candidate; ``tc-b`` k=1 was a harness fault and stored no
    trace; ``tc-b`` k=2 delivered cleanly.
    """
    fixture = ops_fixture()
    storage = fixture.host.eval_host.storage
    storage.save_eval_run(make_eval_run(id=RUN_ID, scope_id=TOYHOST_SCOPE, test_case_ids=["tc-a", "tc-b"], k_runs=2))
    usage = [
        RoleUsage(
            role="candidate", model="sonnet", prompt_tokens=1200, completion_tokens=80, cost_usd=0.004, call_count=3
        )
    ]
    every_action_refused = _result(
        "res-all-refused",
        "tc-a",
        1,
        usage=usage,
        cost_usd=0.004,
        host_measures={"actions_taken": 0.0},
        goal_state_outcomes=[GoalStateOutcome(expression='call_count("notes.write") >= 1', passed=False)],
    )
    storage.save_eval_result(
        _result("res-ok", "tc-b", 2),
        EvalTrace(
            id=eval_trace_doc_id("res-ok"),
            scope_id=TOYHOST_SCOPE,
            result_id="res-ok",
            eval_run_id=RUN_ID,
            trace=[{"turn": 1, "candidate": "done"}],
        ),
    )
    storage.save_eval_result(_result("res-harness", "tc-b", 1, infra_error="simulator timed out"))
    storage.save_eval_result(
        every_action_refused,
        EvalTrace(
            id=eval_trace_doc_id(every_action_refused.id),
            scope_id=TOYHOST_SCOPE,
            result_id=every_action_refused.id,
            eval_run_id=RUN_ID,
            trace=[FAILED_TURN],
            call_ledger=CallLedger(),
        ),
    )
    storage.save_eval_result(
        _result("res-candidate", "tc-a", 2, candidate_error="model refused: 400 bad request"),
        EvalTrace(
            id=eval_trace_doc_id("res-candidate"),
            scope_id=TOYHOST_SCOPE,
            result_id="res-candidate",
            eval_run_id=RUN_ID,
            trace=[{"turn": 1, "candidate": ""}],
        ),
    )
    return fixture


async def _call(tool: Any, fixture: OpsFixture, arguments: dict[str, Any]) -> Any:
    return await tool.call(arguments, host=fixture.host, caller=CALLER)


# =============================================================================
# results_list
# =============================================================================


async def test_a_runs_results_are_listed_light_in_case_then_repeat_order(tools: dict[str, Any]) -> None:
    fixture = _seeded()
    outcome = await _call(tools["evals"], fixture, {"action": "results_list", "run_id": RUN_ID})

    assert not outcome.is_error, outcome.text
    listing = ResultListing.model_validate(outcome.structured)
    assert [(line.id, line.test_case_id, line.k_iteration) for line in listing.results] == [
        ("res-all-refused", "tc-a", 1),
        ("res-candidate", "tc-a", 2),
        ("res-harness", "tc-b", 1),
        ("res-ok", "tc-b", 2),
    ]
    assert [line.condition.value for line in listing.results] == ["ok", "candidate_fail", "infra_exclude", "ok"]
    assert (listing.total, listing.offset, listing.next_offset) == (4, 0, None)
    refused, candidate, harness, _ = listing.results
    assert (refused.model, refused.cost_usd, refused.host_measures) == ("sonnet", 0.004, {"actions_taken": 0.0})
    assert (refused.goal_checks_passed, refused.goal_checks) == (0, 1)
    assert candidate.goal_checks_passed == 0, "a candidate failure counts every check failed, as every rate does"
    assert harness.goal_checks_passed is None and not harness.has_trace, "a harness fault's checks enter no rate"
    assert candidate.judge_scores == {"conversation.tone": 4}, "the score as the judge gave it"
    assert "- res-all-refused: case tc-a k=1, sonnet: ok (completed)" in outcome.text
    assert "Read one with action='result_get'" in outcome.text


async def test_a_page_says_where_the_next_one_starts_and_the_filter_narrows_the_total(tools: dict[str, Any]) -> None:
    fixture = _seeded()
    first = await _call(tools["evals"], fixture, {"action": "results_list", "run_id": RUN_ID, "limit": 3})
    page = ResultListing.model_validate(first.structured)
    assert [line.id for line in page.results] == ["res-all-refused", "res-candidate", "res-harness"]
    assert (page.total, page.next_offset) == (4, 3)
    assert f"offset={page.next_offset}" in first.text

    last = await _call(tools["evals"], fixture, {"action": "results_list", "run_id": RUN_ID, "limit": 3, "offset": 3})
    assert [line.id for line in ResultListing.model_validate(last.structured).results] == ["res-ok"]
    assert ResultListing.model_validate(last.structured).next_offset is None

    narrowed = await _call(
        tools["evals"], fixture, {"action": "results_list", "run_id": RUN_ID, "condition_filter": "ok"}
    )
    listing = ResultListing.model_validate(narrowed.structured)
    assert [line.id for line in listing.results] == ["res-all-refused", "res-ok"] and listing.total == 2


async def test_a_page_above_the_ceiling_is_refused_naming_the_parameter(tools: dict[str, Any]) -> None:
    action = tools["evals"].action("results_list")
    ceiling = action.params.model_json_schema()["properties"]["limit"]["maximum"]
    outcome = await _call(tools["evals"], _seeded(), {"action": "results_list", "run_id": RUN_ID, "limit": ceiling + 1})
    assert outcome.is_error and "limit" in outcome.text


def test_the_condition_filter_offers_exactly_the_conditions_a_result_can_be_in(tools: dict[str, Any]) -> None:
    schema = tools["evals"].action("results_list").params.model_json_schema()["properties"]["condition_filter"]
    (offered,) = [option["enum"] for option in schema["anyOf"] if "enum" in option]
    assert sorted(offered) == sorted(outcome.value for outcome in ResultOutcome)


def test_the_operation_lists_every_row_when_no_surface_bounds_it() -> None:
    """The bound is the agent surface's; a host reading through the operation pages as it chooses."""
    fixture = _seeded()
    listing = results_list(fixture.host.eval_host, RUN_ID, TOYHOST_SCOPE, offset=1)
    assert [line.id for line in listing.results] == ["res-candidate", "res-harness", "res-ok"]
    assert listing.limit is None and listing.next_offset is None


# =============================================================================
# result_get
# =============================================================================


async def test_a_result_whose_every_action_failed_reads_back_each_failure_as_stored(tools: dict[str, Any]) -> None:
    """The shakedown's cell: delivered, so a summary counts it scored, and only its trace says why it did nothing."""
    fixture = _seeded()
    outcome = await _call(tools["evals"], fixture, {"action": "result_get", "result_id": "res-all-refused"})

    assert not outcome.is_error, outcome.text
    detail = ResultDetail.model_validate(outcome.structured)
    assert detail.condition.scoring.value == "ok" and detail.condition.disclosure is None
    assert detail.trace is not None
    (turn,) = detail.trace.trace
    assert [(action["success"], action["result"]) for action in turn["actions"]] == [
        (False, "notes is read-only for this account"),
        (False, "unknown parameter 'when'"),
        (False, "due time is in the past"),
    ]
    assert detail.trace.call_ledger == CallLedger(), "the ledger holds only calls that succeeded: none did"
    (row,) = detail.result.usage
    assert (row.role, row.model, row.call_count, row.cost_usd) == ("candidate", "sonnet", 3, 0.004)
    for action in FAILED_TURN["actions"]:
        assert f'"result": "{action["result"]}", "success": false' in outcome.text
    assert "call ledger (0 call(s) that succeeded):" in outcome.text
    assert '"role": "candidate"' in outcome.text and "cost $0.004 over" in outcome.text
    assert 'goal check call_count("notes.write") >= 1: failed' in outcome.text


async def test_a_failed_results_condition_and_errors_read_as_every_surface_reads_them(tools: dict[str, Any]) -> None:
    fixture = _seeded()
    candidate = await _call(tools["evals"], fixture, {"action": "result_get", "result_id": "res-candidate"})
    detail = ResultDetail.model_validate(candidate.structured)
    assert detail.condition.scoring.value == "candidate_fail" and detail.condition.disclosure
    assert detail.condition.disclosure in candidate.text
    assert "candidate error: model refused: 400 bad request" in candidate.text

    harness = await _call(tools["evals"], fixture, {"action": "result_get", "result_id": "res-harness"})
    detail = ResultDetail.model_validate(harness.structured)
    assert detail.condition.scoring.value == "infra_exclude" and detail.trace is None
    assert "infra error: simulator timed out" in harness.text and harness.text.endswith("trace: none stored")


# =============================================================================
# Scope
# =============================================================================


async def test_a_run_or_result_in_another_scope_is_not_found(tools: dict[str, Any]) -> None:
    """Another tenant's run is refused as not found, never listed as empty, and its results are unreadable."""
    fixture = _seeded()
    storage = fixture.host.eval_host.storage
    storage.save_eval_run(make_eval_run(id="run-elsewhere", scope_id=OTHER_SCOPE))
    storage.save_eval_result(
        make_eval_result(id="res-elsewhere", scope_id=OTHER_SCOPE, eval_run_id="run-elsewhere", model="sonnet")
    )

    listed = await _call(tools["evals"], fixture, {"action": "results_list", "run_id": "run-elsewhere"})
    assert listed.is_error and "not found" in listed.text
    read = await _call(tools["evals"], fixture, {"action": "result_get", "result_id": "res-elsewhere"})
    assert read.is_error and "not found" in read.text


async def test_a_result_read_needs_its_id(tools: dict[str, Any]) -> None:
    outcome = await _call(tools["evals"], _seeded(), {"action": "result_get"})
    assert outcome.is_error and "result_id" in outcome.text


def test_a_judged_result_lists_its_reserved_axes_beside_its_rubric() -> None:
    fixture = _seeded()
    fixture.host.eval_host.storage.save_eval_result(
        _result(
            "res-judged",
            "tc-c",
            1,
            transcript_score=RubricScore(dim=TRANSCRIPT_DIM_ID, score=2, scale="ordinal"),
        )
    )
    (line,) = [
        line for line in results_list(fixture.host.eval_host, RUN_ID, TOYHOST_SCOPE).results if line.id == "res-judged"
    ]
    assert line.judge_scores == {"conversation.tone": 4, TRANSCRIPT_DIM_ID: 2}
