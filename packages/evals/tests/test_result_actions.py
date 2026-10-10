"""A run's results, listed and read one at a time, through the actions a host mounts.

``run_get`` stops at a run's counts and means, so a cell whose candidate failed every action could be
found in a summary and read nowhere but the database. Driven through :meth:`MountedTool.call`, the one
path every transport takes:

- **``results_list`` pages a run's results** as light rows in a stable order — by case, then repeat, then
  id — each with its coordinates, its condition and its headline measures; ``next_offset`` reads the next
  page, ``condition_filter`` narrows to one condition and ``total`` counts what matched across every page.
- **``result_get`` reads one result and one part of its trace**: by default the stored record with its usage
  rows, the condition every surface resolves, each goal check as evaluated and as counted, and the output its
  kind stored — so an action the kind recorded as failed reads back failed, in the tool's own words, beside a
  call ledger that holds only what succeeded. The judge's evidence and the spans come back only when their
  part is asked for, in the text and the data alike; a trace the record promises and no document backs reads
  as missing.
- **Both answer only in the caller's scope, and only for an id of their own type**: a run or result in
  another scope, or a run's id handed where a result's belongs, is not found — never an empty listing that
  would read as a run that produced nothing, and never a validation error from inside the store.
- **A page is bounded for an agent**: a limit above the action's ceiling is refused, naming the parameter,
  and an impossible page is refused by the operation itself. Spend reads as money at any size.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.evals.actions import eval_catalogue, standard_tools
from threetears.evals.schema import (
    TRANSCRIPT_DIM_ID,
    CallLedger,
    EvalResult,
    EvalTrace,
    GoalStateOutcome,
    JudgedArtifact,
    JudgeEvidence,
    LatencyMetrics,
    RecordedCall,
    RoleUsage,
    RubricScore,
    eval_trace_doc_id,
)
from threetears.evals.kernel import ResultOutcome
from threetears.evals.analysis.reporting import decompose_total_ms
from threetears.evals.kernel.errors import ValidationFailedError
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


async def test_a_filtered_page_hint_keeps_its_filter_and_its_size(tools: dict[str, Any]) -> None:
    """Following the hint reads the next page of the same rows at the same size, not the first page of all of them."""
    fixture = _seeded()
    outcome = await _call(
        tools["evals"], fixture, {"action": "results_list", "run_id": RUN_ID, "condition_filter": "ok", "limit": 1}
    )
    listing = ResultListing.model_validate(outcome.structured)
    assert [line.id for line in listing.results] == ["res-all-refused"] and listing.next_offset == 1
    assert f"More: action='results_list', run_id='{RUN_ID}', offset=1, condition_filter='ok', limit=1." in outcome.text
    default = await _call(tools["evals"], fixture, {"action": "results_list", "run_id": RUN_ID})
    assert "limit=" not in default.text, "a page of the default size needs no limit to repeat"


async def test_an_offset_past_the_end_is_an_empty_last_page(tools: dict[str, Any]) -> None:
    outcome = await _call(tools["evals"], _seeded(), {"action": "results_list", "run_id": RUN_ID, "offset": 10})
    listing = ResultListing.model_validate(outcome.structured)
    assert (listing.results, listing.total, listing.next_offset) == ([], 4, None)
    assert outcome.text == f"results of run {RUN_ID}: 4, none from row 11"


@pytest.mark.parametrize(
    ("bounds", "message"), [({"offset": -1}, "offset is -1"), ({"limit": 0}, "limit is 0")], ids=["offset", "limit"]
)
def test_the_operation_refuses_a_page_that_cannot_exist(bounds: dict[str, int], message: str) -> None:
    with pytest.raises(ValidationFailedError, match=message):
        results_list(_seeded().host.eval_host, RUN_ID, TOYHOST_SCOPE, **bounds)


async def test_a_spend_below_a_hundredth_of_a_cent_reads_as_money(tools: dict[str, Any]) -> None:
    fixture = _seeded()
    fixture.host.eval_host.storage.save_eval_result(_result("res-cheap", "tc-c", 1, cost_usd=0.00003))
    outcome = await _call(tools["evals"], fixture, {"action": "results_list", "run_id": RUN_ID})
    (line,) = [line for line in outcome.text.splitlines() if line.startswith("- res-cheap:")]
    assert "cost $0.0000300;" in line and "e-" not in line


def test_the_operation_lists_every_row_when_no_surface_bounds_it() -> None:
    """The bound is the agent surface's; a host reading through the operation pages as it chooses."""
    fixture = _seeded()
    listing = results_list(fixture.host.eval_host, RUN_ID, TOYHOST_SCOPE, offset=1)
    assert [line.id for line in listing.results] == ["res-candidate", "res-harness", "res-ok"]
    assert listing.limit is None and listing.next_offset is None


# =============================================================================
# result_get
# =============================================================================


async def _get(tool: Any, fixture: OpsFixture, result_id: str, part: str | None = None) -> tuple[Any, ResultDetail]:
    arguments = {"action": "result_get", "result_id": result_id} | ({} if part is None else {"part": part})
    outcome = await _call(tool, fixture, arguments)
    assert not outcome.is_error, outcome.text
    return outcome, ResultDetail.model_validate(outcome.structured)


async def test_a_result_whose_every_action_failed_reads_back_each_failure_as_stored(tools: dict[str, Any]) -> None:
    """The shakedown's cell: delivered, so a summary counts it scored, and only its output says why it did nothing."""
    outcome, detail = await _get(tools["evals"], _seeded(), "res-all-refused")

    assert detail.part == "record" and detail.trace_state == "stored"
    assert detail.condition.scoring.value == "ok" and detail.condition.disclosure is None
    assert detail.record is not None
    (turn,) = detail.record.output
    assert [(action["success"], action["result"]) for action in turn["actions"]] == [
        (False, "notes is read-only for this account"),
        (False, "unknown parameter 'when'"),
        (False, "due time is in the past"),
    ]
    assert detail.record.call_ledger == CallLedger(), "the ledger holds only calls that succeeded: none did"
    (row,) = detail.result.usage
    assert (row.role, row.model, row.call_count, row.cost_usd) == ("candidate", "sonnet", 3, 0.004)
    for action in FAILED_TURN["actions"]:
        assert f'"result": "{action["result"]}", "success": false' in outcome.text
    assert "call ledger (0 call(s) the kind recorded as succeeded):" in outcome.text
    assert '"role": "candidate"' in outcome.text and "cost $0.00400 over" in outcome.text
    assert 'goal check call_count("notes.write") >= 1: failed\n' in outcome.text


async def test_a_failed_results_checks_read_as_evaluated_and_as_counted(tools: dict[str, Any]) -> None:
    """A check that evaluated True on a candidate failure counts failed, as results_list counts it — both are said."""
    fixture = _seeded()
    candidate, detail = await _get(tools["evals"], fixture, "res-candidate")
    assert detail.condition.scoring.value == "candidate_fail" and detail.condition.disclosure
    assert detail.condition.disclosure in candidate.text
    assert "candidate error: model refused: 400 bad request" in candidate.text
    assert (
        "goal check state.shop.cart.length >= 1: passed as evaluated; counts failed (candidate failure)"
        in candidate.text.splitlines()
    )

    harness, detail = await _get(tools["evals"], fixture, "res-harness")
    assert detail.condition.scoring.value == "infra_exclude" and detail.trace_state == "none" and detail.record is None
    assert "infra error: simulator timed out" in harness.text and harness.text.endswith("trace: none stored")
    assert (
        "goal check state.shop.cart.length >= 1: passed as evaluated; not counted (harness fault)"
        in harness.text.splitlines()
    )


async def test_a_trace_its_record_promises_but_no_document_backs_reads_as_missing(tools: dict[str, Any]) -> None:
    """Never as none stored: the marker is written from the trace write's own outcome, so the two disagreeing is a fault."""
    fixture = _seeded()
    storage = fixture.host.eval_host.storage
    (stored,) = [
        result for result in storage.query_eval_results_by_run(RUN_ID, TOYHOST_SCOPE) if result.id == "res-harness"
    ]
    storage.replace_eval_result(stored.model_copy(update={"has_trace": True}), if_match=None)

    outcome, detail = await _get(tools["evals"], fixture, "res-harness")
    assert detail.trace_state == "missing" and detail.record is None
    assert outcome.text.endswith("trace: recorded but its document is missing")


#: A judge's evidence and a run of spans, each heavier than everything else the record carries together.
_ARTIFACT = "candidate: " + "the same long reply, over and over. " * 400
_SPANS = [{"name": f"span-{n}", "attributes": {"payload": "x" * 200}} for n in range(50)]


def _heavy(fixture: OpsFixture) -> None:
    fixture.host.eval_host.storage.save_eval_result(
        _result("res-heavy", "tc-c", 1),
        EvalTrace(
            id=eval_trace_doc_id("res-heavy"),
            scope_id=TOYHOST_SCOPE,
            result_id="res-heavy",
            eval_run_id=RUN_ID,
            trace=[{"turn": 1, "candidate": "short"}],
            otel_trace=_SPANS,
            judge_evidence=JudgeEvidence(subject="a concierge", case_material="the scenario", artifact=_ARTIFACT),
            judged_artifact=JudgedArtifact.TRANSCRIPT,
            call_ledger=CallLedger(calls=[RecordedCall(tool="notes", action="read", params={"id": 7})]),
            end_state={"notes": {"count": 2}},
        ),
    )


async def test_the_default_read_leaves_the_judges_evidence_and_the_spans_out_and_says_how_to_ask(
    tools: dict[str, Any],
) -> None:
    fixture = _seeded()
    _heavy(fixture)
    outcome, detail = await _get(tools["evals"], fixture, "res-heavy")

    assert detail.judge is None and detail.spans is None and detail.record is not None
    assert (detail.record.judged_artifact, detail.record.span_count) == (JudgedArtifact.TRANSCRIPT, len(_SPANS))
    assert "the same long reply" not in str(outcome.structured) and "span-0" not in str(outcome.structured)
    assert "the same long reply" not in outcome.text and "span-0" not in outcome.text
    assert "part='judge'" in outcome.text and "part='spans'" in outcome.text
    assert len(outcome.text) < len(_ARTIFACT) / 4
    assert 'end state: {"notes": {"count": 2}}' in outcome.text
    assert '- notes.read {"id": 7}' in outcome.text


async def test_the_judge_part_reads_what_the_judge_was_sent(tools: dict[str, Any]) -> None:
    fixture = _seeded()
    _heavy(fixture)
    outcome, detail = await _get(tools["evals"], fixture, "res-heavy", "judge")

    assert detail.part == "judge" and detail.record is None and detail.spans is None
    assert detail.judge is not None and detail.judge.evidence.artifact == _ARTIFACT
    assert "judge evidence (transcript):" in outcome.text and _ARTIFACT.strip() in outcome.text
    assert "subject:\na concierge" in outcome.text and "case material:\nthe scenario" in outcome.text
    assert "span-0" not in outcome.text and "usage (" not in outcome.text

    unjudged, detail = await _get(tools["evals"], fixture, "res-all-refused", "judge")
    assert detail.judge is None and unjudged.text.endswith("nothing was sent to a judge for this cell")


async def test_the_spans_part_reads_the_spans(tools: dict[str, Any]) -> None:
    fixture = _seeded()
    _heavy(fixture)
    outcome, detail = await _get(tools["evals"], fixture, "res-heavy", "spans")

    assert detail.spans == _SPANS and detail.record is None and detail.judge is None
    assert f"spans ({len(_SPANS)}):" in outcome.text and "span-49" in outcome.text
    assert "the same long reply" not in outcome.text


# =============================================================================
# Scope, and an id of another type
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


@pytest.mark.parametrize(
    ("action", "arguments"),
    [
        ("result_get", {"result_id": RUN_ID}),
        ("result_get", {"result_id": eval_trace_doc_id("res-ok")}),
        ("run_get", {"run_id": "res-ok"}),
        ("results_list", {"run_id": "res-ok"}),
    ],
    ids=["result_get-a-run-id", "result_get-a-trace-id", "run_get-a-result-id", "results_list-a-result-id"],
)
async def test_an_id_of_another_type_is_not_found(
    tools: dict[str, Any], action: str, arguments: dict[str, str]
) -> None:
    """Ids of every type share a scope, so a run's id handed where a result's belongs resolves — to the wrong document."""
    outcome = await _call(tools["evals"], _seeded(), {"action": action, **arguments})
    assert outcome.is_error and "not found" in outcome.text, outcome.text


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


# =============================================================================
# latency (#653)
# =============================================================================


def _timed(fixture: OpsFixture, result_id: str, k: int, latency: LatencyMetrics) -> None:
    fixture.host.eval_host.storage.save_eval_result(_result(result_id, "tc-c", k, latency=latency))


async def test_a_full_latency_block_reads_back_with_its_orchestration_remainder(tools: dict[str, Any]) -> None:
    fixture = _seeded()
    _timed(
        fixture,
        "res-timed",
        1,
        LatencyMetrics(total_ms=1500.0, llm_ms=900.0, tool_ms=350.0, async_wait_ms=0.0, judge_ms=420.0),
    )

    outcome, detail = await _get(tools["evals"], fixture, "res-timed")

    assert detail.latency_partition.orchestration_ms == 250.0, "decompose_total_ms's remainder, carried as derived"
    lines = outcome.text.splitlines()
    assert "latency: total_ms 1500ms, llm_ms 900ms, tool_ms 350ms, async_wait_ms 0ms, judge_ms 420ms" in lines
    assert "orchestration_ms 250ms (total_ms less llm_ms and tool_ms)" in lines


async def test_an_unmeasured_component_reads_absent_and_the_remainder_is_withheld(tools: dict[str, Any]) -> None:
    fixture = _seeded()
    _timed(fixture, "res-part-timed", 2, LatencyMetrics(total_ms=1500.0, llm_ms=900.0))

    outcome, detail = await _get(tools["evals"], fixture, "res-part-timed")

    withheld = decompose_total_ms(LatencyMetrics(total_ms=1500.0, llm_ms=900.0)).withheld
    assert withheld is not None and detail.latency_partition.withheld == withheld
    lines = outcome.text.splitlines()
    assert "latency: total_ms 1500ms, llm_ms 900ms, tool_ms absent, async_wait_ms absent, judge_ms absent" in lines
    assert f"orchestration_ms withheld: {withheld}" in lines
    assert "tool_ms 0ms" not in outcome.text, "an absent component is never shown as zero"


async def test_a_result_with_no_latency_says_so_and_why_nothing_is_partitioned(tools: dict[str, Any]) -> None:
    outcome, _detail = await _get(tools["evals"], _seeded(), "res-ok")

    assert "latency: none recorded" in outcome.text.splitlines()
    assert "orchestration_ms withheld: this cell timed nothing" in outcome.text


async def test_a_listing_row_carries_total_ms_so_a_slow_cell_is_found_without_opening_it(
    tools: dict[str, Any],
) -> None:
    fixture = _seeded()
    _timed(fixture, "res-timed", 1, LatencyMetrics(total_ms=1500.0, llm_ms=900.0, tool_ms=350.0))

    outcome = await _call(tools["evals"], fixture, {"action": "results_list", "run_id": RUN_ID})

    listing = ResultListing.model_validate(outcome.structured)
    by_id = {line.id: line.total_ms for line in listing.results}
    assert by_id["res-timed"] == 1500.0 and by_id["res-ok"] is None
    assert "total_ms 1500ms" in outcome.text and "total_ms absent" in outcome.text
