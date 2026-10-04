"""Contracts for the single place a result's condition is resolved.

The defect this module replaced was not that any one predicate was wrong — it was that
there was one per consumer, spread across the runner, the usage resolver, the MCP render
and a React page, each reading a different subset of the same record, and they disagreed.
So these tests pin two kinds of thing: that each axis answers correctly, and that the arms
cannot be extended without someone deciding what the new arm means.
"""

from __future__ import annotations

import ast
import pathlib
from typing import get_args

import pytest

from threetears.evals.contracts.models import (
    OUTCOME_DIM_ID,
    TRANSCRIPT_DIM_ID,
    CellTermination,
    EvalResult,
    GoalStateOutcome,
    RubricScore,
)
from threetears.evals.contracts.result_condition import (
    JudgingState,
    ResultOutcome,
    candidate_failure_cause,
    counted_goal_verdicts,
    resolve_result_condition,
)
from threetears.evals.contracts.usage_capture import resolve_result_usage
from threetears.evals.contracts.identity import IDENTITY_VERSION
from packages.evals.tests.factories import result_capture_defaults


def _result(**overrides: object) -> EvalResult:
    """A minimal well-formed result; overrides carry whatever the case is about."""
    base: dict[str, object] = {
        **result_capture_defaults(),
        "scope_id": "u1",
        "eval_run_id": "run-1",
        "test_case_id": "tc-1",
        "model": "openai/gpt-5-mini",
        "k_iteration": 1,
    }
    base.update(overrides)
    return EvalResult(**base)  # type: ignore[arg-type]


# =============================================================================
# The arms cannot grow silently
# =============================================================================


#: A result in each judging state, so every arm the resolver can classify is driven through it.
_JUDGING_STATES: dict[str, dict[str, object]] = {
    "scored": {"rubric_scores": [RubricScore(dim="conversation.tone", score=4, scale="ordinal")]},
    "partial": {
        "rubric_scores": [RubricScore(dim="conversation.tone", score=4, scale="ordinal")],
        "judge_error": "judge timed out",
    },
    "failed": {"judge_error": "judge timed out"},
    "not_attempted": {},
}

#: A result in each scoring outcome, on the same terms.
_SCORING_OUTCOMES: dict[ResultOutcome, dict[str, object]] = {
    ResultOutcome.OK: {},
    ResultOutcome.CANDIDATE_FAIL: {"candidate_error": "provider returned 500"},
    ResultOutcome.INFRA_EXCLUDE: {"infra_error": "cell timeout: 300s"},
}


def test_every_termination_arm_has_a_disclosure():
    """A new termination arm must not resolve to "nothing to report" by omission.

    The resolver indexes its disclosure map rather than ``.get``-ing it, so a missing arm raises --
    but only for a reader who happens to load such a result. Driving every arm through the resolver
    is what turns that into a build-time failure, which is the point at which the omission is cheap.
    (The map's keys are typed by the arm's ``Literal``, so a stale extra key is the type checker's.)
    """
    for arm in get_args(CellTermination):
        assert resolve_result_condition(_result(termination=arm)).termination == arm


def test_every_judging_arm_has_a_disclosure():
    """Same contract on the judging axis, for the same reason."""
    assert set(_JUDGING_STATES) == set(get_args(JudgingState)), "a judging arm has no case driving it"
    for arm, fields in _JUDGING_STATES.items():
        assert resolve_result_condition(_result(**fields)).judging == arm


def test_every_scoring_arm_has_a_disclosure():
    """Same contract on the scoring axis -- the one an operator acts on.

    This axis decides whether a result counts toward the aggregates, so an arm that resolves to
    "nothing to report" lets a result be dropped from every number while its scores still render.
    """
    assert set(_SCORING_OUTCOMES) == set(ResultOutcome), "a scoring arm has no case driving it"
    for arm, fields in _SCORING_OUTCOMES.items():
        assert resolve_result_condition(_result(**fields)).scoring is arm


# =============================================================================
# Termination — the stored axis
# =============================================================================


def test_a_cancelled_cell_discloses_that_its_total_stops_short_of_the_call_in_flight():
    """The run loop keeps what a cancelled cell spent, so its total is a floor, not a zero."""
    condition = resolve_result_condition(_result(termination="cell_timeout", usage=[], infra_error="cell timeout"))

    assert condition.termination == "cell_timeout"
    assert condition.scoring is ResultOutcome.INFRA_EXCLUDE
    assert "not the call in flight when the deadline struck" in (condition.disclosure or "")


def test_an_excluded_result_discloses_the_exclusion_beside_its_scores():
    """The failure this closes: a real score rendered beside an error badge, unqualified.

    Judging is not skipped when a cell breaks — by then the transcript holds turns, so the
    judge runs and the result persists real scores. Nothing on the judging axis discloses
    anything about a result that WAS scored, so without a scoring arm an operator reads a
    number from a cell that counts toward nothing.
    """
    condition = resolve_result_condition(
        _result(
            termination="completed",
            infra_error="apparatus: cassette miss",
            rubric_scores=[RubricScore(dim="reply.helpfulness", score=4, scale="ordinal")],
        )
    )

    assert condition.scoring is ResultOutcome.INFRA_EXCLUDE
    assert condition.judging == "scored"  # it really does carry a score
    assert "EXCLUDED from the aggregates" in (condition.disclosure or "")


def test_a_candidate_failure_discloses_that_the_score_is_not_about_content():
    condition = resolve_result_condition(
        _result(termination="completed", candidate_error="LLM returned no parseable action")
    )

    assert condition.scoring is ResultOutcome.CANDIDATE_FAIL
    assert "scored as a hard fail" in (condition.disclosure or "")


def test_a_candidate_failure_discloses_that_its_checks_count_as_failed():
    condition = resolve_result_condition(
        _result(termination="completed", candidate_error="LLM returned no parseable action")
    )

    assert "any goal-state check on it counts as failed" in (condition.disclosure or "")


_CHECKS = [
    GoalStateOutcome(expression='not any(calls("shop.add_note"))', passed=True),
    GoalStateOutcome(expression='call_count("shop.add_item") >= 1', passed=False),
]


def test_a_healthy_result_counts_each_check_as_it_evaluated():
    counted = counted_goal_verdicts(_result(goal_state_outcomes=_CHECKS))

    assert counted == [(_CHECKS[0], True), (_CHECKS[1], False)]


def test_a_candidate_failure_counts_every_check_as_failed():
    """The check that EVALUATED True is the one this exists for."""
    counted = counted_goal_verdicts(_result(candidate_error="candidate turn LLM error: x", goal_state_outcomes=_CHECKS))

    assert counted == [(_CHECKS[0], False), (_CHECKS[1], False)]


@pytest.mark.parametrize("fault", [{"infra_error": "slot did not start"}, {"judge_error": "judge timed out"}])
def test_a_harness_faulted_result_is_in_no_rate(fault):
    assert counted_goal_verdicts(_result(goal_state_outcomes=_CHECKS, **fault)) is None


def test_a_candidate_failure_outranks_an_infra_fault_on_the_same_result():
    """The precedence classify_result holds: a broken candidate cannot launder itself into an exclusion."""
    counted = counted_goal_verdicts(
        _result(candidate_error="candidate turn LLM error: x", infra_error="slot", goal_state_outcomes=_CHECKS)
    )

    assert counted == [(_CHECKS[0], False), (_CHECKS[1], False)]


def test_a_healthy_scored_result_discloses_nothing():
    """A result in no notable condition gains no chrome — the disclosure is a warning, not a label."""
    condition = resolve_result_condition(
        _result(
            termination="completed",
            usage=[],
            rubric_scores=[RubricScore(dim="reply.helpfulness", score=4, scale="ordinal")],
        )
    )

    assert condition.disclosure is None
    assert condition.judging == "scored"
    assert condition.scoring is ResultOutcome.OK


# =============================================================================
# Judging — the axis two surfaces disagreed on
# =============================================================================


def test_a_pinned_judge_that_never_ran_is_not_evidence_of_judging():
    """The regression this axis exists for.

    ``judge_model`` is the RUN-level pin, resolved at launch and stamped onto degraded
    exits so a dead cell still says which judge its run pinned. A surface reading it as
    evidence — MCP did, and the detail page's own comment warned about it — says
    "judged by X" over a result whose scores are all absent, which sends an
    investigation at rubric configuration instead of at the timeout that killed the cell.
    """
    condition = resolve_result_condition(_result(termination="cell_timeout", judge_model="openai/gpt-5-mini"))

    assert condition.judging == "not_attempted"


@pytest.mark.parametrize(
    "evidence",
    [
        {"judge_reasoning": "it answered the question"},
        {"rubric_scores": [RubricScore(dim="reply.helpfulness", score=3, scale="ordinal")]},
        {"transcript_score": RubricScore(dim=TRANSCRIPT_DIM_ID, score=5, scale="ordinal")},
        {"outcome_score": RubricScore(dim=OUTCOME_DIM_ID, score=2, scale="ordinal")},
    ],
    ids=["reasoning-only", "rubric-only", "transcript-axis-only", "outcome-axis-only"],
)
def test_any_one_judge_output_counts_as_judged(evidence: dict[str, object]):
    """A judge can score without narrating, and a template can score without either reserved axis.

    Requiring one particular output would report a real judging pass as none — which is
    what makes the four-way OR the contract rather than an implementation detail.
    """
    assert resolve_result_condition(_result(**evidence)).judging == "scored"


def test_a_judge_that_scored_some_dims_and_errored_on_others_is_partial():
    """Collapsing this into either neighbour loses whichever half the reader needed."""
    condition = resolve_result_condition(
        _result(
            rubric_scores=[RubricScore(dim="reply.helpfulness", score=4, scale="ordinal")],
            judge_error="dim 'tone': 429",
        )
    )

    assert condition.judging == "partial"
    assert "some dimensions were scored and some were not" in (condition.disclosure or "")


def test_a_judge_that_produced_nothing_and_errored_is_failed():
    condition = resolve_result_condition(_result(judge_error="judge: connection reset"))

    assert condition.judging == "failed"
    assert condition.scoring is ResultOutcome.INFRA_EXCLUDE


# =============================================================================
# The known false affirmative
# =============================================================================


def test_a_cancelled_cells_usage_is_reported_lost_not_captured():
    """A lost capture, refused at the resolver that once published it as complete.

    A timed-out cell stores ``usage=[]`` — byte-identical to a subject-factory failure's
    honest "capture ran and attributed no roles". Branching on ``usage is not None`` alone
    therefore published ``source="captured", partial=False`` for a cell that had burned
    past its deadline: a claim that something which ran for minutes consumed nothing
    anywhere. The run loop now keeps the rows a cancelled cell had captured, but the
    call in flight at the deadline never reported — here the cell's only one — so the
    capture is cut short rather than complete, and only the recorded termination separates
    the two.
    """
    resolved = resolve_result_usage(_result(termination="cell_timeout", usage=[]))

    assert resolved.source == "lost"
    assert resolved.partial is True
    assert resolved.usage == []


def test_a_factory_failure_still_reports_an_honest_empty_capture():
    """The other side of the same distinction — this one really did attribute no roles.

    Paired with the test above deliberately: the two records differ ONLY in the stored
    termination, so a fix that reported every empty capture as lost would pass that test
    and fail this one.
    """
    resolved = resolve_result_usage(_result(termination="factory_failed", usage=[]))

    assert resolved.source == "captured"
    assert resolved.partial is False


# =============================================================================
# Results come from the runner, which states every capture field at every exit
# =============================================================================


#: The package's source root, resolved from this file rather than the working directory — the sibling
#: source-scanning canaries anchor the same way, and a cwd-relative path silently reads
#: nothing (or the wrong tree) the first time the suite is invoked from elsewhere.
_SOURCE_ROOT = pathlib.Path(__file__).resolve().parents[1] / "src"


def _eval_result_constructions(module: pathlib.Path) -> list[ast.Call]:
    """Every ``EvalResult(...)`` call in one module.

    Matches the bare name *and* the attribute form (``models.EvalResult(...)``). Matching
    only the bare name would let an attribute-form construction slip the caller below
    while its docstring claims to cover every producer — no such call exists today, which
    is exactly why the gap would go unnoticed until one did.
    """
    tree = ast.parse(module.read_text())
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == "EvalResult")
            or (isinstance(node.func, ast.Attribute) and node.func.attr == "EvalResult")
        )
    ]


def test_the_runner_is_the_only_producer_of_eval_results():
    """Results come from the runner, whose every exit states what it captured.

    ``EvalResult`` requires the fields the runner writes at each exit — how the cell ended, the
    cost composition, the per-role rows, the covariates — so a result that lacks one is refused.
    That makes those fields honest only while the runner is what fills them: a second producer
    (a replay path, an importer, an operator-edit surface) would have to state them too, and
    stating a capture it never made is fabrication the model cannot detect.

    Failing here is not "add this module to an allowlist" — it is a prompt to decide what the
    new producer actually observed, and only then to widen the scan.
    """
    producers = sorted(
        path.relative_to(_SOURCE_ROOT).as_posix()
        for path in (_SOURCE_ROOT / "threetears").rglob("*.py")
        if _eval_result_constructions(path)
    )

    assert producers == ["threetears/evals/run/runner.py"], (
        f"EvalResult is constructed outside the runner: {producers}. A result's capture fields are the "
        "runner's observations; a second producer must have observed what it states."
    )


class TestTheJudgeCouldNotTell:
    """A "can't tell" answer is judge output and is disclosed, and only a rubric dim reaches the aggregates."""

    def test_a_result_answered_cannot_tell_on_every_dim_was_judged(self):
        result = EvalResult(
            scope_id="u",
            eval_run_id="r",
            test_case_id="t",
            model="m",
            k_iteration=1,
            judge_cannot_tell={"reply.refusal": "no request to refuse"},
            termination="completed",
            cost_usd=0.0,
            cost_roles=["candidate", "inner_agent", "judge", "simulator"],
            usage=[],
            covariates={},
            phase_timings={},
            host_measures={},
            variant_key="vk-1",
            identity_version=IDENTITY_VERSION,
        )
        condition = resolve_result_condition(result)
        assert condition.judging == "scored"
        assert condition.disclosure is not None and "reply.refusal" in condition.disclosure
        assert "left out of pass^k" in condition.disclosure

    def test_a_reserved_axis_is_disclosed_without_claiming_an_exclusion(self):
        result = EvalResult(
            scope_id="u",
            eval_run_id="r",
            test_case_id="t",
            model="m",
            k_iteration=1,
            rubric_scores=[RubricScore(dim="reply.tone", score=4, scale="ordinal")],
            judge_cannot_tell={"__transcript__": "no turns"},
            termination="completed",
            cost_usd=0.0,
            cost_roles=["candidate", "inner_agent", "judge", "simulator"],
            usage=[],
            covariates={},
            phase_timings={},
            host_measures={},
            variant_key="vk-1",
            identity_version=IDENTITY_VERSION,
        )
        disclosure = resolve_result_condition(result).disclosure
        assert disclosure is not None and "__transcript__" in disclosure
        assert "pass^k" not in disclosure

    def test_nothing_is_said_when_the_judge_could_tell(self):
        result = EvalResult(
            scope_id="u",
            eval_run_id="r",
            test_case_id="t",
            model="m",
            k_iteration=1,
            rubric_scores=[RubricScore(dim="reply.tone", score=4, scale="ordinal")],
            termination="completed",
            cost_usd=0.0,
            cost_roles=["candidate", "inner_agent", "judge", "simulator"],
            usage=[],
            covariates={},
            phase_timings={},
            host_measures={},
            variant_key="vk-1",
            identity_version=IDENTITY_VERSION,
        )
        assert resolve_result_condition(result).disclosure is None


class TestTheOutputCapEndedATurn:
    """A turn the cap cut off is a candidate failure, disclosed as the cap's — not as a failed model."""

    def test_a_cut_off_result_is_a_candidate_failure_named_for_the_cap(self):
        result = _result(covariates={"truncated_rounds": 1.0})
        assert candidate_failure_cause(result) == "output_cap"
        condition = resolve_result_condition(result)
        assert condition.scoring is ResultOutcome.CANDIDATE_FAIL
        assert condition.disclosure is not None and "output cap" in condition.disclosure
        assert "A model this configuration runs" not in condition.disclosure

    def test_its_checks_count_as_failed_like_any_candidate_failure(self):
        result = _result(
            covariates={"truncated_rounds": 1.0},
            goal_state_outcomes=[GoalStateOutcome(expression='call_count("shop.add_item") <= 1', passed=True)],
        )
        assert counted_goal_verdicts(result) == [(result.goal_state_outcomes[0], False)]

    def test_a_model_failure_is_named_first_when_both_are_recorded(self):
        result = _result(candidate_error="candidate turn LLM error: 502", covariates={"truncated_rounds": 1.0})
        assert candidate_failure_cause(result) == "model_failed"

    @pytest.mark.parametrize("covariates", [{}, {"truncated_rounds": 0.0}])
    def test_no_count_or_a_zero_count_is_not_a_failure(self, covariates):
        assert candidate_failure_cause(_result(covariates=covariates)) is None
        assert resolve_result_condition(_result(covariates=covariates)).scoring is ResultOutcome.OK


class TestTheTurnBudgetEndedATurn:
    """A turn the host's turn budget ended is a candidate failure, disclosed as the budget's."""

    def test_an_ended_turn_is_a_candidate_failure_named_for_the_budget(self):
        result = _result(covariates={"turns_ended_by_budget": 1.0})
        assert candidate_failure_cause(result) == "turn_budget"
        condition = resolve_result_condition(result)
        assert condition.scoring is ResultOutcome.CANDIDATE_FAIL
        assert condition.disclosure is not None and "turn budget" in condition.disclosure
        assert "A model this configuration runs" not in condition.disclosure
        assert "output cap" not in condition.disclosure

    def test_its_checks_count_as_failed_like_any_candidate_failure(self):
        result = _result(
            covariates={"turns_ended_by_budget": 1.0},
            goal_state_outcomes=[GoalStateOutcome(expression='call_count("shop.add_note") == 0', passed=True)],
        )
        assert counted_goal_verdicts(result) == [(result.goal_state_outcomes[0], False)]

    def test_a_model_failure_is_named_before_it(self):
        result = _result(candidate_error="candidate turn LLM error: 502", covariates={"turns_ended_by_budget": 1.0})
        assert candidate_failure_cause(result) == "model_failed"

    def test_it_is_named_before_the_output_cap(self):
        """An ended turn left no record of whether its rounds were cut, so the budget is the cause known to hold."""
        result = _result(covariates={"turns_ended_by_budget": 1.0, "truncated_rounds": 1.0})
        assert candidate_failure_cause(result) == "turn_budget"

    @pytest.mark.parametrize("covariates", [{}, {"turns_ended_by_budget": 0.0}])
    def test_no_count_or_a_zero_count_is_not_a_failure(self, covariates):
        assert candidate_failure_cause(_result(covariates=covariates)) is None
        assert resolve_result_condition(_result(covariates=covariates)).scoring is ResultOutcome.OK


def test_every_candidate_failure_cause_has_its_own_disclosure():
    """A cause added without a sentence would borrow another cause's remedy; each names its own lever.

    Driven through the resolver, one result per cause: a cause with no sentence raises there, and two
    causes sharing one would resolve to the same disclosure.
    """
    from threetears.evals.contracts.result_condition import CandidateFailureCause

    causes: dict[str, dict[str, object]] = {
        "model_failed": {"candidate_error": "provider returned 500"},
        "turn_budget": {"covariates": {"turns_ended_by_budget": 1.0}},
        "output_cap": {"covariates": {"truncated_rounds": 1.0}},
    }
    assert set(causes) == set(get_args(CandidateFailureCause)), "a candidate-failure cause has no case driving it"

    disclosures = {}
    for cause, fields in causes.items():
        result = _result(**fields)
        assert candidate_failure_cause(result) == cause
        disclosures[cause] = resolve_result_condition(result).disclosure
    assert len(set(disclosures.values())) == len(disclosures), disclosures


def test_a_cannot_tell_result_with_a_failed_check_is_not_said_to_be_left_out_of_pass_k():
    result = _result(
        judge_cannot_tell={"reply.thread_fit": "nothing was queued"},
        goal_state_outcomes=[GoalStateOutcome(expression='call_count("shop.add_item") >= 2', passed=False)],
    )
    disclosure = resolve_result_condition(result).disclosure
    assert disclosure is not None
    assert "left out of pass^k" not in disclosure
    assert "failed goal-state check fails it" in disclosure
