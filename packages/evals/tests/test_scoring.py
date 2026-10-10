"""Tests for :mod:`threetears.evals.contracts.scoring` — every pure function that turns eval results into
the numbers a run reports.

The module's own thesis is the placement rule, and this file is its mirror: what is asserted
here is what a set of results *scores*, what it *cost*, how long it *took*, and whether the
loop delivered the matrix the run promised — never how the loop got there. A test whose subject
is the runner producing those inputs stays in the runner's suite, and there are a handful:
pass^k over the order ``cell_execution_order`` actually emits is a claim about the two
TOGETHER, and moving it here would leave that coupling asserted by nothing.

An aggregator that speaks a host's own nouns is deliberately absent: it lives in that host's
adapter, with its cases beside it. Reading this file's inventory as "the scoring family" and
finding no host-specific rollup is the question that answer exists for.

**The result factory is shared, not copied.** ``make_scored_result`` lives in
``packages/evals/tests/factories.py`` because this file and the runner's suite both build results
that way — the runner keeps the cases whose subject is a loop and an aggregator together. A
second copy is a silent revert: the day one is fixed, the other keeps proving the old
behaviour.

``percentile``'s own cases open the file because they are the contract the move CREATED rather
than relocated. ``threetears/evals/analysis/bundle.py`` has its own percentile taking its
quantile on the **[0, 1]** scale; this one takes **0-100**. Handed ``0.95`` by someone carrying
that habit across, nearest-rank would return the *minimum* — a plausible number, wrong by the
width of the distribution, with nothing anywhere to notice it. The guard is what makes that
loud, so it needs a test that fires it.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from threetears.evals.analysis import assemble_context_bundle
from threetears.evals.contracts.campaign import EvalCampaign
from threetears.evals.contracts.models import TRANSCRIPT_DIM_ID, EvalResult, LatencyMetrics, RoleUsage, RubricScore
from threetears.evals.contracts.result_condition import ResultOutcome
from threetears.evals.contracts.scoring import (
    NO_PASS_CRITERION_REASON,
    CellSummary,
    compute_composite_summary,
    compute_cost_summary,
    compute_dimension_summary,
    compute_latency_summary,
    compute_pass_hat_k,
    compute_per_case_composites,
    pass_hat_k_at,
    pass_hat_k_cell,
    percentile,
    pool_pass_hat_k,
    result_composite,
    summarize_completeness,
)
from threetears.evals.contracts.identity import IDENTITY_VERSION
from packages.evals.tests.factories import make_eval_result, make_eval_run, make_scored_result
from packages.evals.tests.fixtures.toyhost.corpus import ToyhostStorage
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.factories import result_capture_defaults


# =============================================================================
# percentile — the shared primitive, and the scale its neighbour does not share
# =============================================================================

_SORTED = [1.0, 2.0, 3.0, 4.0, 100.0]


class TestPercentileRefusesTheOtherScale:
    """The failure this guard exists for returns a number, which is why it must raise."""

    @pytest.mark.parametrize("pct", [0.05, 0.5, 0.95, 0.99, 1.0])
    def test_a_zero_to_one_quantile_is_refused_by_name(self, pct):
        with pytest.raises(ValueError, match=r"\[0, 1\] scale"):
            percentile(_SORTED, pct)

    def test_the_closed_end_at_one_is_refused_because_it_is_where_the_scales_invert(self):
        """1.0 is the worst value to let through, not the safest, so the band closes on it.

        On the neighbour's [0, 1] scale it means the MAXIMUM; nearest-rank on 0-100 resolves it
        to rank 1 and returns the MINIMUM. It is also the value a caller reaches for most
        readily after 0.95, so an open end there leaves the guard's own failure mode reachable
        at the point it inverts hardest. The band was open here until a review said so.

        The cost is stated rather than hidden: p1 is not obtainable through this function, and
        the message has to say where the maximum actually is or a refused caller guesses again.
        """
        with pytest.raises(ValueError) as excinfo:
            percentile(_SORTED, 1.0)
        message = str(excinfo.value)
        assert "pass 100" in message, "a caller who meant the maximum must be told how to ask"
        assert "FIRST percentile" in message, "and told what 1.0 means on this scale instead"

    def test_the_refusal_names_the_neighbour_a_caller_confused_it_with(self):
        """A message saying only 'bad input' leaves the caller to find the other convention."""
        with pytest.raises(ValueError) as excinfo:
            percentile(_SORTED, 0.95)
        assert "bundle" in str(excinfo.value), "the message must name where the [0, 1] one lives"

    @pytest.mark.parametrize("pct", [-1.0, 100.5, 1000.0])
    def test_a_quantile_outside_the_scale_entirely_is_refused(self, pct):
        with pytest.raises(ValueError, match=r"\[0, 100\]"):
            percentile(_SORTED, pct)

    @pytest.mark.parametrize("pct", [0.0, 1.5, 50.0, 95.0, 100.0])
    def test_the_legitimate_boundaries_are_not_swept_up_by_the_guard(self, pct):
        """The refused band is ``(0, 1]``, so both ends of the real scale still answer.

        0.0 and 100.0 are the scale's own endpoints and a guard that swallowed either would be
        refusing the thing it exists to serve. 1.5 is the first value above the band — included
        so the band's upper edge is pinned from both sides rather than only from inside.
        """
        assert percentile(_SORTED, pct) in _SORTED


class TestPercentileIsNearestRank:
    """The convention, pinned — an interpolated answer here would be a number nothing observed."""

    def test_it_returns_an_observed_value_never_an_interpolated_one(self):
        assert percentile(_SORTED, 95) == 100.0
        assert percentile(_SORTED, 50) == 3.0
        # The bottom of the scale, taken above the refused (0, 1] band.
        assert percentile(_SORTED, 20) == 1.0

    def test_a_single_sample_is_every_percentile_of_itself(self):
        assert percentile([7.0], 50) == 7.0
        assert percentile([7.0], 95) == 7.0

    def test_nearest_rank_is_not_the_tail_the_bundle_reports(self):
        """Same data, different answers, and why the tail is not read nearest-rank.

        Read on each function's own scale, so this is not the scale confusion above. Nearest-rank's
        95th percentile of five values is their maximum, which is not a 95th percentile; the bundle reads
        its tail median-unbiased and, at five observations, where no estimate is, reports none.
        """
        bundle_p50, bundle_p95 = _bundle_percentiles(_SORTED)

        assert percentile(_SORTED, 50) == 3.0
        assert bundle_p50 == 3.0
        assert percentile(_SORTED, 95) == 100.0
        assert bundle_p95 is None


def _bundle_percentiles(costs: list[float]) -> tuple[float, float]:
    """The p50 and p95 the analysis bundle reports for one run whose results cost ``costs``.

    Read off an assembled bundle's telemetry rollup, which is where the bundle's own percentile
    (interpolated, on the [0, 1] scale) reaches a reader.
    """
    profile = toyhost_profile()
    run = make_eval_run(status="completed")
    results = [
        make_eval_result(
            eval_run_id=run.id,
            test_case_id=f"tc-{i}",
            cost_usd=cost,
            # The priced row the cost is derived from: a cost with no row behind it is no observation of spend.
            usage=[RoleUsage(role="candidate", model=None, cost_usd=cost, call_count=1)],
        )
        for i, cost in enumerate(costs)
    ]
    campaign = EvalCampaign(
        scope_id=run.scope_id,
        name="c",
        subject_id=run.subject_snapshot.subject_id,
        subject_kind="s",
        behavior="b",
        run_ids=[run.id],
        created_by="test:fixture",
    )
    bundle = assemble_context_bundle(campaign, storage=ToyhostStorage([run], {run.id: results}), profile=profile)
    [cost] = [measure for measure in bundle.telemetry.measures.measures if measure.name == "cost_usd"]
    return cost.p50, cost.p95


# =============================================================================
# CellSummary — the row the loop keeps, and what it may not carry
# =============================================================================


def test_cell_summary_holds_no_unbounded_field():
    """A summary cannot carry a trace — enforced over the type, not by review.

    The check is on every field's *value type* rather than on the two field
    names that carry traces today. A summary that regained the whole
    ``EvalResult`` — or gained any list, dict or model — would reintroduce
    precisely the retention this type exists to prevent while passing a
    name-based check, and the regression would be invisible until a sweep
    large enough to matter exhausted the container.
    """
    summary = CellSummary.from_result(
        make_eval_result(),
        persisted=True,
    )

    names = {f.name for f in dataclasses.fields(CellSummary)}
    assert "trace" not in names
    assert "otel_trace" not in names

    bounded = (str, int, float, bool, type(None))
    for name in names:
        value = getattr(summary, name)
        assert isinstance(value, bounded), f"CellSummary.{name} holds an unbounded {type(value).__name__}"


# =============================================================================
# summarize_completeness — what the loop delivered against what the run promised
# =============================================================================


def _cell(*, persisted: bool = True, outcome: ResultOutcome = ResultOutcome.OK) -> CellSummary:
    """One cell summary with only the two fields completeness reads varied."""
    return CellSummary(
        result_id="res-1",
        test_case_id="tc-1",
        model="m1",
        k_iteration=1,
        outcome=outcome,
        termination="completed",
        cost_usd=0.0,
        persisted=persisted,
    )


def test_completeness_of_a_whole_matrix_is_not_degraded():
    run = make_eval_run(candidate_model="m1", k_runs=2, test_case_ids=["tc-1", "tc-2"])

    record = summarize_completeness(run, [_cell() for _ in range(4)])

    assert record.expected_cells == 4  # 2 cases × k=2, at the run's one model
    assert record.produced_cells == 4
    assert record.persisted_cells == 4
    assert record.measured_cells == 4
    assert record.degraded is False


def test_a_lost_write_degrades_the_run_and_is_told_apart_from_a_cell_that_never_ran():
    """The two shortfalls a row count collapses into one number.

    Storage holds three rows either way — whether the loop ran four cells and
    lost one write, or only ever ran three. They are different faults with
    different fixes (a harness dropping data it had, versus a run launched
    against less than it recorded), and the only place the difference survives
    is the summary the loop returned, because the lost cell left no row.
    """
    run = make_eval_run(candidate_model="m1", k_runs=1, test_case_ids=["tc-1", "tc-2", "tc-3", "tc-4"])

    lost_write = summarize_completeness(run, [_cell(), _cell(persisted=False), _cell(), _cell()])
    never_ran = summarize_completeness(run, [_cell(), _cell(), _cell()])

    assert lost_write.persisted_cells == never_ran.persisted_cells == 3
    assert lost_write.degraded is never_ran.degraded is True
    assert lost_write.produced_cells == 4, "a cell that ran and was lost must not read as one that never ran"
    assert never_ran.produced_cells == 3


def test_an_infra_exclusion_degrades_the_run_but_a_candidate_failure_does_not():
    """The distinction the whole record turns on.

    A candidate that failed is a measurement — scored as a hard fail, in the
    denominator, and comparable. A harness failure took the cell out of the
    aggregates, so the run's rates rest on fewer observations than its siblings'
    and nothing else on the document says so.
    """
    run = make_eval_run(candidate_model="m1", k_runs=1, test_case_ids=["tc-1", "tc-2"])

    infra = summarize_completeness(run, [_cell(), _cell(outcome=ResultOutcome.INFRA_EXCLUDE)])
    candidate = summarize_completeness(run, [_cell(), _cell(outcome=ResultOutcome.CANDIDATE_FAIL)])

    assert infra.infra_excluded_cells == 1
    assert infra.measured_cells == 1
    assert infra.degraded is True
    assert candidate.infra_excluded_cells == 0
    assert candidate.measured_cells == 2
    assert candidate.degraded is False


def test_a_cell_that_was_both_excluded_and_lost_is_subtracted_once():
    """``measured_cells`` is a subtraction, so double-counting would go negative.

    Counting exclusions over every cell rather than over the persisted ones
    would charge a lost cell twice and report fewer measurements than the run
    could possibly have made.
    """
    run = make_eval_run(candidate_model="m1", k_runs=1, test_case_ids=["tc-1", "tc-2"])

    record = summarize_completeness(run, [_cell(), _cell(persisted=False, outcome=ResultOutcome.INFRA_EXCLUDE)])

    assert record.persisted_cells == 1
    assert record.infra_excluded_cells == 0
    assert record.measured_cells == 1


def test_the_denominator_comes_from_the_run_not_from_the_cells_it_was_handed():
    """A run launched against fewer cases than it recorded must be visible.

    Counting the matrix off the loop's own cells would make every run complete
    by construction — the shape that lets a short launch read as a clean one.
    """
    run = make_eval_run(candidate_model="m1", k_runs=3, test_case_ids=["tc-1", "tc-2"])

    record = summarize_completeness(run, [_cell(), _cell()])

    assert record.expected_cells == 6
    assert record.produced_cells == 2
    assert record.degraded is True


# =============================================================================
# compute_pass_hat_k — pass^k, the chance that k attempts at a case all pass
# =============================================================================


def _infra_result(test_case_id="tc1", model="m1", run_id="r1", k=1):
    """A result that failed for an INFRA reason (excluded from scoring)."""
    return make_scored_result(
        test_case_id=test_case_id, model=model, run_id=run_id, k=k, infra_error="cell timeout: boom"
    )


def _curve(entry: dict[str, Any]) -> list[tuple[int, float | None, int]]:
    """An entry's pass^k curve as ``(k, value, n_cases)`` triples, for comparing whole."""
    return [(point["k"], point["pass_hat_k"], point["n_cases"]) for point in entry["pass_hat_k_curve"]]


def test_compute_pass_hat_k_all_pass_yields_one():
    results = [make_scored_result(test_case_id=f"tc{i}", k=1) for i in range(3)]
    out = compute_pass_hat_k(results)
    assert out[("m1", "r1")]["pass_hat_k"] == 1.0
    assert out[("m1", "r1")]["n_test_cases"] == 3
    assert out[("m1", "r1")]["n_cases_at_k"] == 3


def test_compute_pass_hat_k_partial_pass():
    results = [
        make_scored_result(test_case_id="tc1", goal_passes=(True,)),
        make_scored_result(test_case_id="tc2", goal_passes=(False,)),
    ]
    out = compute_pass_hat_k(results)
    assert out[("m1", "r1")]["pass_hat_k"] == 0.5
    assert out[("m1", "r1")]["n_cases_at_k"] == 2


def test_compute_pass_hat_k_failure_in_any_k_iter_fails_the_case_at_that_depth():
    """At depth k every one of k attempts must pass; one failure of two zeroes pass^2 for the case."""
    results = [
        make_scored_result(test_case_id="tc1", k=1, goal_passes=(True,)),
        make_scored_result(test_case_id="tc1", k=2, goal_passes=(False,)),
    ]
    entry = compute_pass_hat_k(results)[("m1", "r1")]
    assert entry["k"] == 2
    assert entry["pass_hat_k"] == 0.0
    # ...while one attempt of the two passing is what pass^1 says.
    assert _curve(entry) == [(1, 0.5, 1), (2, 0.0, 1)]


def test_compute_pass_hat_k_is_the_unbiased_per_case_estimate_not_the_all_pass_indicator():
    """Per case, C(c, k) / C(n, k): the share of the case's k-subsets of attempts that all passed.

    Four attempts, three passed. The all-pass indicator over all four reads 0; pass^2 is
    C(3, 2) / C(4, 2) = 3 / 6 and pass^3 is C(3, 3) / C(4, 3) = 1 / 4 — the expectations of a case
    that passes each attempt with probability p are p^2 and p^3, whatever n is.
    """
    passes = (True, True, True, False)
    results = [make_scored_result(k=i + 1, goal_passes=(p,)) for i, p in enumerate(passes)]

    entry = compute_pass_hat_k(results)[("m1", "r1")]

    assert _curve(entry) == [(1, 0.75, 1), (2, 0.5, 1), (3, 0.25, 1), (4, 0.0, 1)]


def test_compute_pass_hat_k_pass_1_is_the_per_case_pass_rate_averaged_over_cases():
    """pass^1 weighs each case once, however many attempts it took: (2/3 + 1/1) / 2, not 3/4."""
    results = [
        make_scored_result(test_case_id="tc-a", k=1, goal_passes=(True,)),
        make_scored_result(test_case_id="tc-a", k=2, goal_passes=(True,)),
        make_scored_result(test_case_id="tc-a", k=3, goal_passes=(False,)),
        make_scored_result(test_case_id="tc-b", k=1, goal_passes=(True,)),
    ]

    point = pass_hat_k_at(compute_pass_hat_k(results)[("m1", "r1")]["pass_hat_k_curve"], 1)

    assert point["pass_hat_k"] == pytest.approx((2 / 3 + 1) / 2)
    assert point["n_cases"] == 2


def test_compute_pass_hat_k_reads_the_headline_at_the_depth_asked():
    results = [make_scored_result(test_case_id="tc1", k=i, goal_passes=(i != 3,)) for i in (1, 2, 3)]
    entry = compute_pass_hat_k(results, k=2)[("m1", "r1")]
    assert (entry["k"], entry["pass_hat_k"], entry["n_cases_at_k"]) == (2, pytest.approx(1 / 3), 1)


def test_compute_pass_hat_k_refuses_a_depth_below_one():
    with pytest.raises(ValueError, match="k >= 1"):
        compute_pass_hat_k([make_scored_result()], k=0)


def test_compute_pass_hat_k_respects_rubric_threshold():
    results = [
        make_scored_result(test_case_id="tc1", rubric_scores=(("x.v", 3),)),
        make_scored_result(test_case_id="tc2", rubric_scores=(("x.v", 2),)),
    ]
    out = compute_pass_hat_k(results, rubric_threshold=3)
    assert out[("m1", "r1")]["pass_hat_k"] == 0.5


def test_compute_pass_hat_k_separates_models():
    results = [
        make_scored_result(test_case_id="tc1", model="m1"),
        make_scored_result(test_case_id="tc1", model="m2", goal_passes=(False,)),
    ]
    out = compute_pass_hat_k(results)
    assert out[("m1", "r1")]["pass_hat_k"] == 1.0
    assert out[("m2", "r1")]["pass_hat_k"] == 0.0


def test_compute_pass_hat_k_empty_results():
    assert compute_pass_hat_k([]) == {}


def test_compute_pass_hat_k_candidate_error_fails_not_passes():
    """A candidate-error result FAILS (counts, 0.0) — it must not slip through the
    vacuous-result guard as a pass, and it is NOT excluded (a broken candidate fails).
    """
    err_result = make_scored_result(test_case_id="tc1", candidate_error="candidate turn LLM error: 402")
    out = compute_pass_hat_k([err_result])
    assert out[("m1", "r1")]["n_test_cases"] == 1
    assert out[("m1", "r1")]["pass_hat_k"] == 0.0
    assert out[("m1", "r1")]["n_cases_at_k"] == 1


def _cut_off(result: EvalResult, rounds: float = 1) -> EvalResult:
    """``result`` with the output cap recorded as having ended ``rounds`` of the candidate's turns."""
    return result.model_copy(update={"covariates": {"truncated_rounds": rounds}})


class TestTheOutputCapsSilenceFails:
    """A turn the output cap ended fails pass^k on every template.

    The hold-template shape is the one that mattered: every check is a restraint check, the cut
    turn did nothing, so every check passed and the judge scored the silence as restraint.
    """

    def test_a_cut_off_hold_trial_whose_checks_and_judge_all_passed_fails(self):
        held = make_scored_result(goal_passes=(True, True), rubric_scores=(("reply.restraint", 5),))
        assert compute_pass_hat_k([held])[("m1", "r1")]["pass_hat_k"] == 1.0  # the control: uncut, it passes
        out = compute_pass_hat_k([_cut_off(held)])
        assert out[("m1", "r1")]["pass_hat_k"] == 0.0
        assert out[("m1", "r1")]["n_test_cases"] == 1  # counted as a fail, not excluded

    def test_a_cut_off_act_trial_whose_checks_passed_on_earlier_rounds_fails(self):
        acted = make_scored_result(goal_passes=(True,), rubric_scores=(("reply.thread_fit", 4),))
        assert compute_pass_hat_k([acted])[("m1", "r1")]["pass_hat_k"] == 1.0  # the control: uncut, it passes
        assert compute_pass_hat_k([_cut_off(acted)])[("m1", "r1")]["pass_hat_k"] == 0.0

    def test_a_cut_off_trial_fails_even_where_the_judge_could_not_tell(self):
        silent = make_scored_result(goal_passes=(True,)).model_copy(
            update={"judge_cannot_tell": {"reply.thread_fit": "nothing was queued"}}
        )
        out = compute_pass_hat_k([_cut_off(silent)])
        assert out[("m1", "r1")]["pass_hat_k"] == 0.0
        assert out[("m1", "r1")]["n_cannot_tell_excluded"] == 0

    def test_a_zero_count_is_an_observation_that_nothing_was_cut(self):
        assert compute_pass_hat_k([_cut_off(make_scored_result(), rounds=0)])[("m1", "r1")]["pass_hat_k"] == 1.0


class TestAFailedTrialIsNeverExcluded:
    """The judge answering it could not tell leaves out only a trial nothing has decided yet."""

    _UNSCORED = {"judge_cannot_tell": {"reply.thread_fit": "no additions to assess"}}

    def test_a_failed_goal_check_fails_the_trial_the_judge_could_not_finish(self):
        failed = make_scored_result(goal_passes=(False, True)).model_copy(update=self._UNSCORED)
        out = compute_pass_hat_k([failed])
        assert out[("m1", "r1")]["n_test_cases"] == 1
        assert out[("m1", "r1")]["pass_hat_k"] == 0.0
        assert out[("m1", "r1")]["n_cannot_tell_excluded"] == 0

    def test_a_scored_dim_below_the_bar_fails_it_too(self):
        failed = make_scored_result(rubric_scores=(("reply.handoff_spoken", 1),)).model_copy(update=self._UNSCORED)
        out = compute_pass_hat_k([failed], rubric_threshold=3)
        assert out[("m1", "r1")]["pass_hat_k"] == 0.0
        assert out[("m1", "r1")]["n_cannot_tell_excluded"] == 0

    def test_a_trial_the_unscored_dim_would_decide_is_still_left_out(self):
        undecided = make_scored_result(rubric_scores=(("reply.handoff_spoken", 4),)).model_copy(update=self._UNSCORED)
        out = compute_pass_hat_k([undecided], rubric_threshold=3)
        assert out[("m1", "r1")]["n_test_cases"] == 0
        assert out[("m1", "r1")]["pass_hat_k"] is None
        assert out[("m1", "r1")]["n_cannot_tell_excluded"] == 1


def test_compute_pass_hat_k_infra_iteration_excluded_case_scored_on_the_rest():
    """An infra-error iteration is EXCLUDED — the case is scored on its other
    iterations, not floored to a fail (the candidate/infra split).
    """
    results = [
        make_scored_result(test_case_id="tc1", k=1, goal_passes=(True,)),  # measured, passes
        _infra_result(test_case_id="tc1", k=2),  # excluded (judge/timeout/factory)
    ]
    entry = compute_pass_hat_k(results)[("m1", "r1")]
    assert entry["n_test_cases"] == 1
    # One scored attempt, and it passed: pass^1 is 1.0, not floored by the excluded one...
    assert _curve(entry) == [(1, 1.0, 1)]
    # ...and pass^2 is unmeasured rather than credited — one attempt cannot speak for two.
    assert (entry["k"], entry["pass_hat_k"], entry["n_cases_at_k"]) == (2, None, 0)


def test_compute_pass_hat_k_all_iterations_excluded_case_drops_from_denominator():
    """A case whose every iteration is infra-excluded drops from n_test_cases."""
    results = [
        make_scored_result(test_case_id="tc1", goal_passes=(True,)),  # a real, measured case
        _infra_result(test_case_id="tc2", k=1),
        _infra_result(test_case_id="tc2", k=2),  # tc2 entirely unmeasured
    ]
    entry = compute_pass_hat_k(results)[("m1", "r1")]
    assert entry["n_test_cases"] == 1  # tc2 gone
    assert _curve(entry) == [(1, 1.0, 1)]


def test_compute_pass_hat_k_all_infra_run_reported_with_zero_measured_cases():
    """A run whose every result is infra-excluded still appears — as n_test_cases
    0 with no pass rate (nothing measured), not absent, not a floored fail, and not
    a pass rate of zero, which would read as every case failing.
    """
    out = compute_pass_hat_k([_infra_result(test_case_id="tc1", k=1)])
    assert ("m1", "r1") in out
    assert out[("m1", "r1")]["n_test_cases"] == 0
    assert out[("m1", "r1")]["n_cases_at_k"] == 0
    assert out[("m1", "r1")]["pass_hat_k"] is None
    assert out[("m1", "r1")]["pass_hat_k_curve"] == []


@pytest.mark.parametrize("judge_model", [None, "judge-model"])
def test_compute_pass_hat_k_empty_score_lists_with_no_error_never_pass(judge_model):
    """A result with no scoreable outcomes cannot pass: unmeasured with no judge (#688), a fail under one.

    With no goal-state check and no judge there is nothing for pass^k to conjoin, so the attempt is left out
    and the group has no pass^k — never the 0.0 that read as failing every criterion. Under a judge the
    criteria were asked and nothing was scored, which is the fail it always was.
    """
    empty = EvalResult(
        scope_id="u",
        eval_run_id="r1",
        test_case_id="tc1",
        model="m1",
        k_iteration=1,
        termination="completed",
        cost_usd=0.0,
        cost_roles=["candidate", "inner_agent", "judge", "simulator"],
        usage=[],
        covariates={},
        phase_timings={},
        host_measures={},
        variant_key="vk-1",
        identity_version=IDENTITY_VERSION,
        judge_model=judge_model,
    )
    row = compute_pass_hat_k([empty])[("m1", "r1")]
    if judge_model is None:
        assert row["pass_hat_k"] is None and row["pass_hat_k_curve"] == []
        assert row["n_no_criterion_excluded"] == 1
        assert row["pass_hat_k_unmeasured_reason"] == NO_PASS_CRITERION_REASON
    else:
        assert row["pass_hat_k"] == 0.0
        assert row["n_no_criterion_excluded"] == 0 and row["pass_hat_k_unmeasured_reason"] is None


# =============================================================================
# compute_pass_hat_k over a partial run — mixed depths, estimated without bias
#
# Cells execute in a per-run shuffled order, so a run that stopped early leaves
# an ARBITRARY SUBSET of its matrix rather than the k-ordered prefix the nested
# loop used to leave. The all-pass indicator held a case the stop left with one
# observation to a weaker bar than its neighbour with three, and flattered the
# run. The estimator leaves a case out of every depth it was not measured to, so
# each point of the curve is an unbiased mean over the cases that reached it.
#
# The case that takes its partial set from the real execution order stays in
# the runner's suite: its subject is the loop and the aggregator together.
# =============================================================================


def test_compute_pass_hat_k_leaves_a_shallow_case_out_of_the_depths_it_never_reached():
    """``tc-shallow`` kept one passed attempt; it informs pass^1 and nothing deeper.

    The all-pass indicator read this run as 0.5 — the shallow case's one pass standing in
    for three — where pass^3 rests on the one case measured three times, which failed once.
    """
    results = [
        make_scored_result(test_case_id="tc-deep", k=1, goal_passes=(True,)),
        make_scored_result(test_case_id="tc-deep", k=2, goal_passes=(True,)),
        make_scored_result(test_case_id="tc-deep", k=3, goal_passes=(False,)),
        make_scored_result(test_case_id="tc-shallow", k=2, goal_passes=(True,)),
    ]

    entry = compute_pass_hat_k(results)[("m1", "r1")]

    assert (entry["k"], entry["pass_hat_k"], entry["n_cases_at_k"]) == (3, 0.0, 1)
    assert _curve(entry) == [(1, pytest.approx((2 / 3 + 1) / 2), 2), (2, pytest.approx(1 / 3), 1), (3, 0.0, 1)]


def test_compute_pass_hat_k_every_point_rests_on_every_case_when_the_run_measured_them_alike():
    """A complete run's cases all reached the planned depth, so no point loses a case."""
    results = [make_scored_result(test_case_id=f"tc{i}", k=k, goal_passes=(True,)) for i in range(3) for k in (1, 2)]

    entry = compute_pass_hat_k(results)[("m1", "r1")]

    assert _curve(entry) == [(1, 1.0, 3), (2, 1.0, 3)]


def test_compute_pass_hat_k_reports_the_depth_reached_by_the_model_it_is_keyed_on():
    """One model's deeper cells must not speak for a model that never got past its first.

    A partial run stops mid-permutation, so the arms are not all equally far
    along. The entry is keyed ``(model, run)``, and a value under that key has to
    describe that key's population — pooling the run's models let an arbitrary
    cell set the reported k for every arm.
    """
    results = [
        make_scored_result(test_case_id="tc1", model="m1", k=1, goal_passes=(True,)),
        make_scored_result(test_case_id="tc1", model="m1", k=2, goal_passes=(True,)),
        make_scored_result(test_case_id="tc1", model="m1", k=3, goal_passes=(True,)),
        make_scored_result(test_case_id="tc1", model="m2", k=1, goal_passes=(True,)),
    ]

    out = compute_pass_hat_k(results)

    assert out[("m1", "r1")]["k"] == 3
    assert out[("m2", "r1")]["k"] == 1


def test_compute_pass_hat_k_depth_counts_attempts_and_the_curve_counts_scored_ones():
    """``k`` and the curve answer different questions and must not be conflated.

    An infra-excluded iteration was attempted — it raises ``k`` — and contributed
    nothing to the estimate, so it must not lengthen the curve. A reader comparing the
    two is reading how much of the attempted work survived.
    """
    results = [
        make_scored_result(test_case_id="tc1", k=1, goal_passes=(True,)),
        _infra_result(test_case_id="tc1", k=2),
    ]

    entry = compute_pass_hat_k(results)[("m1", "r1")]

    assert entry["k"] == 2
    assert len(entry["pass_hat_k_curve"]) == 1


def test_pass_hat_k_at_a_depth_the_curve_never_reached_is_unmeasured():
    assert pass_hat_k_at([], 2) == {"k": 2, "pass_hat_k": None, "n_cases": 0}
    with pytest.raises(ValueError, match="k >= 1"):
        pass_hat_k_at([], 0)


# =============================================================================
# pool_pass_hat_k — a case's attempts across the runs of one cell
#
# A run is one arm, so a per-run grouping can never pool two configurations — and can
# never pool two RUNS of one configuration either, which is what made a repeat run add
# a second copy of each case instead of depth. The pool keys a case by its cell
# (pass_hat_k_cell: the run's context key) as well as its test case.
# =============================================================================


def _pooled_result(run_id: str, k: int, passed: bool, *, test_case_id: str = "tc1", variant_key: str = "vk-1"):
    """A result of ``run_id`` at iteration ``k`` for one variant, passing or failing its one check."""
    result = make_scored_result(test_case_id=test_case_id, run_id=run_id, k=k, goal_passes=(passed,))
    return result.model_copy(update={"variant_key": variant_key})


class TestPoolingAcrossRuns:
    """Repeat runs of one configuration add depth; runs of different configurations never pool."""

    def test_two_runs_of_one_cell_are_more_attempts_at_the_same_case(self):
        results = [_pooled_result("r1", 1, True), _pooled_result("r2", 1, False)]

        entry = pool_pass_hat_k(results, cell_of_run={"r1": "cell-a", "r2": "cell-a"}, k=2)

        assert entry["n_test_cases"] == 1
        assert (entry["pass_hat_k"], entry["n_cases_at_k"]) == (0.0, 1)
        assert _curve(entry) == [(1, 0.5, 1), (2, 0.0, 1)]

    def test_two_cells_keep_one_test_case_as_two_cases(self):
        results = [_pooled_result("r1", 1, True), _pooled_result("r2", 1, False)]

        entry = pool_pass_hat_k(results, cell_of_run={"r1": "cell-a", "r2": "cell-b"}, k=2)

        assert entry["n_test_cases"] == 2
        assert (entry["pass_hat_k"], entry["n_cases_at_k"]) == (None, 0)
        assert _curve(entry) == [(1, 0.5, 2)]

    def test_a_run_the_map_does_not_name_is_its_own_cell(self):
        results = [_pooled_result("r1", 1, True), _pooled_result("r2", 1, True)]

        entry = pool_pass_hat_k(results, cell_of_run={"r1": "cell-a"}, k=1)

        assert entry["n_test_cases"] == 2

    def test_two_variants_never_pool_even_under_one_cell(self):
        results = [_pooled_result("r1", 1, True), _pooled_result("r1", 2, True, variant_key="vk-2")]

        entry = pool_pass_hat_k(results, cell_of_run={"r1": "cell-a"}, k=1)

        assert entry["n_test_cases"] == 2

    def test_the_pool_drops_and_counts_exclusions_as_the_per_run_estimate_does(self):
        unscored = make_scored_result(rubric_scores=(("reply.tone", 4),)).model_copy(
            update={"judge_cannot_tell": {"reply.fit": "nothing to read"}}
        )
        results = [_pooled_result("r1", 1, True), _infra_result(run_id="r1", k=2), unscored]

        entry = pool_pass_hat_k(results, cell_of_run={"r1": "cell-a"}, k=1)

        assert (entry["n_test_cases"], entry["pass_hat_k"], entry["n_cannot_tell_excluded"]) == (1, 1.0, 1)

    def test_a_pool_needs_a_depth(self):
        with pytest.raises(ValueError, match="k >= 1"):
            pool_pass_hat_k([], cell_of_run={}, k=0)


class TestTheCellARunsAttemptsPoolUnder:
    """Runs pool only when a recorded context says they held the same conditions fixed."""

    def test_two_runs_sharing_a_context_key_are_one_cell(self):
        a = make_eval_run(id="r1", context_key="ctx-1", identity_version=IDENTITY_VERSION)
        b = make_eval_run(id="r2", context_key="ctx-1", identity_version=IDENTITY_VERSION, k_runs=3)
        assert pass_hat_k_cell(a) == pass_hat_k_cell(b)

    def test_a_different_context_is_a_different_cell(self):
        a = make_eval_run(id="r1", context_key="ctx-1", identity_version=IDENTITY_VERSION)
        b = make_eval_run(id="r2", context_key="ctx-2", identity_version=IDENTITY_VERSION)
        assert pass_hat_k_cell(a) != pass_hat_k_cell(b)

    def test_one_key_stamped_under_two_predicates_is_two_cells(self):
        a = make_eval_run(id="r1", context_key="ctx-1", identity_version=IDENTITY_VERSION)
        b = make_eval_run(id="r2", context_key="ctx-1", identity_version=IDENTITY_VERSION - 1)
        assert pass_hat_k_cell(a) != pass_hat_k_cell(b)

    def test_a_commissioned_and_a_witnessed_run_are_two_cells(self):
        a = make_eval_run(id="r1", context_key="ctx-1", identity_version=IDENTITY_VERSION)
        b = make_eval_run(
            id="r2", context_key="ctx-1", identity_version=IDENTITY_VERSION, apparatus_provenance="witnessed"
        )
        assert pass_hat_k_cell(a) != pass_hat_k_cell(b)

    def test_a_run_with_no_recorded_context_pools_with_nothing(self):
        a, b = make_eval_run(id="r1"), make_eval_run(id="r2")
        assert a.context_key is None
        assert pass_hat_k_cell(a) != pass_hat_k_cell(b)


# =============================================================================
# compute_latency_summary
# =============================================================================


def _lat_result(model="m1", run_id="r1", k=1, total=100.0, llm=60.0, tool=20.0, latency: bool = True, **kwargs):
    """EvalResult with (or without) LatencyMetrics for summary tests.

    ``kwargs`` carries the error fields ``classify_result`` reads, so a test can
    build a cell whose timings are the harness's rather than the candidate's.
    """
    return EvalResult(
        scope_id="u",
        eval_run_id=run_id,
        test_case_id="tc1",
        model=model,
        k_iteration=k,
        latency=LatencyMetrics(total_ms=total, llm_ms=llm, tool_ms=tool) if latency else None,
        **{**result_capture_defaults(), **kwargs},
    )


def test_compute_latency_summary_aggregates_per_model_run():
    results = [
        _lat_result(k=1, total=100.0, llm=60.0, tool=20.0),
        _lat_result(k=2, total=200.0, llm=120.0, tool=40.0),
    ]
    out = compute_latency_summary(results)
    stats = out[("m1", "r1")]
    assert stats["mean_total_ms"] == 150.0
    assert stats["mean_llm_ms"] == 90.0
    assert stats["mean_tool_ms"] == 30.0  # tool completes the total = llm + tool decomposition
    assert stats["n_results"] == 2


def test_compute_latency_summary_median_and_tail_at_small_n():
    results = [_lat_result(k=i, total=float(t)) for i, t in enumerate([10, 20, 30, 40, 100], start=1)]
    stats = compute_latency_summary(results)[("m1", "r1")]
    assert stats["median_total_ms"] == 30.0  # the middle of 5 values
    # Five totals cannot give a 95th percentile; the slowest is reported under its own name, not as one.
    assert "p95_total_ms" not in stats
    assert stats["max_total_ms"] == 100.0


def test_compute_latency_summary_median_of_an_even_count_is_the_mean_of_the_middle_two():
    """Nearest-rank took the lower middle value (20); the median of 10, 20, 30, 40 is 25."""
    results = [_lat_result(k=i, total=float(t)) for i, t in enumerate([40, 10, 30, 20], start=1)]
    stats = compute_latency_summary(results)[("m1", "r1")]
    assert stats["median_total_ms"] == 25.0


def test_compute_latency_summary_p95_is_type_8_from_thirteen_totals():
    totals = [float(t) for t in range(10, 210, 10)]  # 20 totals, 10..200
    results = [_lat_result(k=i, total=t) for i, t in enumerate(totals, start=1)]
    stats = compute_latency_summary(results)[("m1", "r1")]
    # h = (20 + 1/3) * 0.95 + 1/3 = 19.65: between the 19th (190) and 20th (200) totals.
    assert stats["p95_total_ms"] == pytest.approx(190.0 + 0.65 * 10.0)
    assert stats["max_total_ms"] == 200.0


def test_compute_latency_summary_skips_none_latency():
    results = [
        _lat_result(k=1, total=100.0),
        _lat_result(k=2, latency=False),  # factory-failure slot — no measured time
    ]
    stats = compute_latency_summary(results)[("m1", "r1")]
    assert stats["n_results"] == 1
    assert stats["mean_total_ms"] == 100.0


def test_compute_latency_summary_omits_group_with_all_none():
    out = compute_latency_summary([_lat_result(latency=False)])
    assert out == {}


def test_compute_latency_summary_drops_an_infra_excluded_cells_timings():
    """A harness failure's timings measure the harness, not the configuration.

    The run that surfaced this reported ``mean_llm_ms (n=2)`` beside ``Cases 1``:
    the excluded cell timed a conversation a cassette miss cut short, and that
    time went into the model-attributable axis the comparison views rank on.
    """
    results = [
        _lat_result(k=1, total=100.0, llm=60.0, tool=20.0),
        _lat_result(k=2, total=9000.0, llm=8000.0, tool=500.0, infra_error="apparatus: cassette miss in replay mode"),
    ]

    stats = compute_latency_summary(results)[("m1", "r1")]

    assert stats["n_results"] == 1
    assert stats["n_llm_ms"] == 1
    assert stats["mean_llm_ms"] == 60.0
    assert stats["mean_total_ms"] == 100.0


def test_compute_latency_summary_keeps_a_candidate_failures_timings():
    """A candidate that fails is an outcome of the configuration — it still took time."""
    results = [
        _lat_result(k=1, total=100.0, llm=60.0, tool=20.0),
        _lat_result(k=2, total=300.0, llm=180.0, tool=60.0, candidate_error="candidate LLM returned 400"),
    ]

    stats = compute_latency_summary(results)[("m1", "r1")]

    assert stats["n_results"] == 2
    assert stats["mean_total_ms"] == 200.0


def test_compute_latency_summary_omits_a_group_whose_every_cell_was_excluded():
    """Same shape as the all-``None`` group: omitted rather than reported as zero."""
    results = [
        _lat_result(k=1, judge_error="judge LLM returned 500"),
        _lat_result(k=2, infra_error="apparatus: seed failed"),
    ]

    assert compute_latency_summary(results) == {}


def test_an_unmeasured_component_is_dropped_rather_than_averaged_as_zero():
    """A cell with an llm timing but no turn-root must not pull the total mean down.

    This is the candidate-fails-outside-the-turn-wrapper shape. Averaging its
    unknown total in as 0.0 reports a speed nobody observed, on the axis where
    lower is better.
    """
    results = [
        _lat_result(k=1, total=100.0, llm=60.0, tool=20.0),
        _lat_result(k=2, total=None, llm=80.0, tool=None),
    ]

    stats = compute_latency_summary(results)[("m1", "r1")]

    assert stats["mean_total_ms"] == 100.0, "an unmeasured total was averaged in as zero"
    assert stats["mean_llm_ms"] == 70.0, "the component that WAS measured must still aggregate"
    assert stats["mean_tool_ms"] == 20.0


def test_components_disclose_that_they_rest_on_different_samples():
    """Each mean carries its own denominator; n_results counts results, not measurements.

    Without the per-component counts a mean over one observation prints
    identically to a mean over two, which is the unmeasured-reads-as-measured
    defect moved from the value up to its evidence.
    """
    results = [
        _lat_result(k=1, total=100.0, llm=60.0, tool=20.0),
        _lat_result(k=2, total=None, llm=80.0, tool=None),
    ]

    stats = compute_latency_summary(results)[("m1", "r1")]

    assert stats["n_results"] == 2
    assert stats["n_total_ms"] == 1, "mean_total_ms rests on one observation and must say so"
    assert stats["n_llm_ms"] == 2
    assert stats["n_tool_ms"] == 1


def test_a_denominator_is_absent_exactly_when_its_mean_is():
    """A count without its mean (or the reverse) would be a row nobody can read."""
    stats = compute_latency_summary([_lat_result(total=None, llm=60.0, tool=None)])[("m1", "r1")]

    for mean_key, count_key in (
        ("mean_total_ms", "n_total_ms"),
        ("mean_llm_ms", "n_llm_ms"),
        ("mean_tool_ms", "n_tool_ms"),
    ):
        assert (mean_key in stats) == (count_key in stats), f"{mean_key} and {count_key} disagree about presence"

    assert stats["n_llm_ms"] == 1
    assert "n_total_ms" not in stats


def test_a_component_nothing_measured_is_absent_not_zero():
    results = [_lat_result(k=1, total=None, llm=60.0, tool=None)]

    stats = compute_latency_summary(results)[("m1", "r1")]

    assert "mean_total_ms" not in stats
    assert "median_total_ms" not in stats and "p95_total_ms" not in stats and "max_total_ms" not in stats
    assert "mean_tool_ms" not in stats
    assert stats["mean_llm_ms"] == 60.0


def test_a_harvested_group_that_measured_nothing_is_distinct_from_never_harvested():
    """Empty-but-present says "we tried and got no timings"; omission says nobody ran."""
    tried = compute_latency_summary([_lat_result(total=None, llm=None, tool=None)])
    never = compute_latency_summary([_lat_result(latency=False)])

    assert tried[("m1", "r1")] == {"n_results": 1}
    assert never == {}


def test_compute_latency_summary_separates_models():
    results = [
        _lat_result(model="m1", total=100.0),
        _lat_result(model="m2", total=300.0),
    ]
    out = compute_latency_summary(results)
    assert out[("m1", "r1")]["mean_total_ms"] == 100.0
    assert out[("m2", "r1")]["mean_total_ms"] == 300.0


# =============================================================================
# compute_cost_summary
# =============================================================================


def _cost_result(model="m1", run_id="r1", k=1, cost=0.01, prod_cost=None, **kwargs):
    """EvalResult carrying a cost for the cost-summary tests.

    ``prod_cost`` attaches a candidate usage row — a production-replicating role —
    so the result MEASURES a prod cost. Left None, the result carries no usage
    decomposition at all, which is the unmeasured case.
    """
    if prod_cost is not None:
        kwargs["usage"] = [RoleUsage(role="candidate", model=model, cost_usd=prod_cost)]
    return EvalResult(
        scope_id="u",
        eval_run_id=run_id,
        test_case_id="tc1",
        model=model,
        k_iteration=k,
        **{**result_capture_defaults(), "cost_usd": cost, **kwargs},
    )


def test_compute_cost_summary_groups_by_model_and_run():
    results = [
        _cost_result(model="m1", run_id="r1", k=1, cost=0.01, prod_cost=0.006),
        _cost_result(model="m1", run_id="r1", k=2, cost=0.03, prod_cost=0.014),
        _cost_result(model="m2", run_id="r1", k=1, cost=0.10, prod_cost=0.08),
        _cost_result(model="m1", run_id="r2", k=1, cost=0.50, prod_cost=0.40),
    ]
    out = compute_cost_summary(results)

    assert out[("m1", "r1")] == {
        "total_cost_usd": 0.04,
        "mean_cost_usd": 0.02,
        "n_cost_usd": 2,
        "total_prod_cost_usd": 0.02,
        "mean_prod_cost_usd": 0.01,
        "n_prod_cost_usd": 2,
        "n_results": 2,
    }
    assert out[("m2", "r1")] == {
        "total_cost_usd": 0.10,
        "mean_cost_usd": 0.10,
        "n_cost_usd": 1,
        "total_prod_cost_usd": 0.08,
        "mean_prod_cost_usd": 0.08,
        "n_prod_cost_usd": 1,
        "n_results": 1,
    }
    assert out[("m1", "r2")] == {
        "total_cost_usd": 0.50,
        "mean_cost_usd": 0.50,
        "n_cost_usd": 1,
        "total_prod_cost_usd": 0.40,
        "mean_prod_cost_usd": 0.40,
        "n_prod_cost_usd": 1,
        "n_results": 1,
    }


def test_compute_cost_summary_counts_zero_cost_results():
    """A priced zero is a cost, and is never skipped — zero-cost slots still count."""
    results = [
        _cost_result(model="m1", k=1, cost=0.0, prod_cost=0.0),
        _cost_result(model="m1", k=2, cost=0.02, prod_cost=0.02),
    ]
    out = compute_cost_summary(results)
    assert out[("m1", "r1")] == {
        "total_cost_usd": 0.02,
        "mean_cost_usd": 0.01,
        "n_cost_usd": 2,
        "total_prod_cost_usd": 0.02,
        "mean_prod_cost_usd": 0.01,
        "n_prod_cost_usd": 2,
        "n_results": 2,
    }


def test_compute_cost_summary_omits_unmeasured_prod_cost_from_the_mean():
    """A result with no usage decomposition is absent from the prod mean, not a zero in it.

    The mean is over the measured subset, so it answers "what did a measured
    result cost" rather than being dragged toward an instant nobody observed.
    """
    results = [
        _cost_result(k=1, cost=0.03, prod_cost=0.02),
        _cost_result(k=2, cost=0.03, prod_cost=0.04),
        _cost_result(k=3, cost=0.03),  # no decomposition — unmeasured
    ]
    row = compute_cost_summary(results)[("m1", "r1")]

    assert row["total_prod_cost_usd"] == pytest.approx(0.06)
    assert row["mean_prod_cost_usd"] == pytest.approx(0.03)  # 0.06 / 2, NOT / 3
    assert row["n_prod_cost_usd"] == 2
    # Every result still counts toward the group and the program axis.
    assert row["n_results"] == 3
    assert row["mean_cost_usd"] == pytest.approx(0.03)


def test_compute_cost_summary_omits_prod_keys_when_nothing_measured():
    """Absent, not zero — the group survives carrying only what it measured."""
    row = compute_cost_summary([_cost_result(k=1, cost=0.01), _cost_result(k=2, cost=0.03)])[("m1", "r1")]

    assert "total_prod_cost_usd" not in row
    assert "mean_prod_cost_usd" not in row
    assert "n_prod_cost_usd" not in row
    # The group is still reported, and the program axis is unaffected.
    assert row == {"total_cost_usd": 0.04, "mean_cost_usd": 0.02, "n_cost_usd": 2, "n_results": 2}


def test_an_unpriced_result_is_left_out_of_the_program_cost_and_counted_not_averaged_in_as_zero():
    """Unpriced is unknown, not free: averaged in as zero it would rank the unpriced model cheapest."""
    results = [
        _cost_result(k=1, cost=0.04),
        _cost_result(k=2, cost=None),
        _cost_result(k=3, cost=0.02),
    ]
    row = compute_cost_summary(results)[("m1", "r1")]

    assert row["total_cost_usd"] == pytest.approx(0.06)
    assert row["mean_cost_usd"] == pytest.approx(0.03)  # 0.06 / 2, NOT / 3
    assert row["n_cost_usd"] == 2
    assert row["n_results"] == 3


def test_a_group_with_no_priced_result_reports_no_program_cost_at_all():
    row = compute_cost_summary([_cost_result(k=1, cost=None), _cost_result(k=2, cost=None)])[("m1", "r1")]

    assert row == {"n_cost_usd": 0, "n_results": 2}


def test_compute_cost_summary_cancelled_cell_measures_no_prod_cost():
    """A cell cancelled during its only call reported no usage — that is unmeasured, not free.

    Its ``usage`` is ``[]``, which resolves to the ``lost`` view rather than to a
    captured decomposition, so no production role observed a cost.
    """
    results = [
        _cost_result(k=1, cost=0.05, prod_cost=0.04),
        _cost_result(k=2, cost=0.0, usage=[], termination="cell_timeout"),
    ]
    row = compute_cost_summary(results)[("m1", "r1")]

    assert row["n_prod_cost_usd"] == 1
    assert row["mean_prod_cost_usd"] == pytest.approx(0.04)
    assert row["n_results"] == 2


def test_compute_cost_summary_least_measured_config_does_not_rank_cheapest():
    """The defect this shape exists to prevent, stated as a comparison.

    Two models spend the same per measured result; one simply recorded fewer
    decompositions. Counting the unmeasured results as zeros would make the
    worse-instrumented model look less than half as expensive.
    """
    results = [
        _cost_result(model="thorough", k=1, cost=0.10, prod_cost=0.10),
        _cost_result(model="thorough", k=2, cost=0.10, prod_cost=0.10),
        _cost_result(model="patchy", k=1, cost=0.10, prod_cost=0.10),
        _cost_result(model="patchy", k=2, cost=0.10),  # unmeasured
        _cost_result(model="patchy", k=3, cost=0.10),  # unmeasured
    ]
    out = compute_cost_summary(results)

    assert out[("thorough", "r1")]["mean_prod_cost_usd"] == pytest.approx(0.10)
    assert out[("patchy", "r1")]["mean_prod_cost_usd"] == pytest.approx(0.10)
    # And the evidence gap is readable rather than hidden inside an equal mean.
    assert out[("thorough", "r1")]["n_prod_cost_usd"] == 2
    assert out[("patchy", "r1")]["n_prod_cost_usd"] == 1
    assert out[("patchy", "r1")]["n_results"] == 3


def test_compute_cost_summary_still_counts_an_infra_excluded_cells_spend():
    """Program spend is the family's deliberate exception, and this pins it as a decision.

    pass^k, latency, the dimension means and the composite all drop an
    infra-excluded cell, because none of them can read a harness failure as
    evidence about the candidate. Program spend answers a different question — what
    the program spent — and the tokens a cell burned before its apparatus broke were
    still billed. ``run_summary`` also takes its group set from these keys, so
    dropping them would delete an all-excluded model's headline row from the
    report whose whole job is to disclose it.
    """
    results = [
        _cost_result(k=1, cost=0.01, prod_cost=0.006),
        _cost_result(k=2, cost=0.03, prod_cost=0.014, infra_error="apparatus: cassette miss in replay mode"),
    ]

    row = compute_cost_summary(results)[("m1", "r1")]

    assert row["n_results"] == 2
    assert row["total_cost_usd"] == pytest.approx(0.04)

    # And the group survives even when every one of its cells was excluded.
    all_excluded = compute_cost_summary([_cost_result(k=1, cost=0.02, infra_error="apparatus: seed failed")])
    assert all_excluded[("m1", "r1")]["n_results"] == 1


def test_compute_cost_summary_a_faulted_cell_does_not_lower_the_comparison_cost():
    """#619: the production-replicating axis compares configurations, so a fault-shortened cell is not in it.

    The apparatus broke the second cell after it had spent a fraction of a whole one. Averaged in, the arm's
    mean prod cost would halve on a fault of the rig; program spend still counts those dollars.
    """
    results = [
        _cost_result(k=1, cost=0.02, prod_cost=0.010),
        _cost_result(k=2, cost=0.02, prod_cost=0.010),
        _cost_result(k=3, cost=0.002, prod_cost=0.001, infra_error="apparatus: cassette miss in replay mode"),
    ]

    row = compute_cost_summary(results)[("m1", "r1")]

    assert row["mean_prod_cost_usd"] == pytest.approx(0.010)
    assert row["n_prod_cost_usd"] == 2
    assert row["total_cost_usd"] == pytest.approx(0.042)
    assert row["n_cost_usd"] == 3

    # A group whose every cell faulted keeps its row and its spend, and measures no comparison cost.
    faulted = compute_cost_summary([_cost_result(k=1, cost=0.02, prod_cost=0.01, infra_error="apparatus: seed failed")])
    assert faulted[("m1", "r1")]["total_cost_usd"] == pytest.approx(0.02)
    assert "mean_prod_cost_usd" not in faulted[("m1", "r1")]


def test_compute_cost_summary_empty_results():
    assert compute_cost_summary([]) == {}


# =============================================================================
# compute_dimension_summary (per-rubric-dimension breakdown)
# =============================================================================


def _dim_result(model="m1", run_id="r1", k=1, dims: dict[str, int] | None = None, test_case_id="tc1", **kwargs):
    """EvalResult carrying rubric_scores for the dimension-summary tests.

    ``kwargs`` carries the error fields ``classify_result`` reads
    (``infra_error`` / ``judge_error`` / ``candidate_error``), so a test can
    build the shape whose scores must not reach a mean.
    """
    return EvalResult(
        scope_id="u",
        eval_run_id=run_id,
        test_case_id=test_case_id,
        model=model,
        k_iteration=k,
        rubric_scores=[RubricScore(dim=d, score=s, scale="ordinal") for d, s in (dims or {}).items()],
        **{**result_capture_defaults(), **kwargs},
    )


def test_compute_dimension_summary_aggregates_mean_min_max_per_dim():
    results = [
        _dim_result(k=1, dims={"reply.grounding": 4, "reply.coverage": 2}),
        _dim_result(k=2, dims={"reply.grounding": 2, "reply.coverage": 4}),
    ]
    out = compute_dimension_summary(results)
    assert out[("m1", "r1", "reply.grounding")] == {
        "scale": "ordinal",
        "mean_score": 3.0,
        "min_score": 2,
        "max_score": 4,
        "n": 2,
    }
    assert out[("m1", "r1", "reply.coverage")] == {
        "scale": "ordinal",
        "mean_score": 3.0,
        "min_score": 2,
        "max_score": 4,
        "n": 2,
    }


def test_compute_dimension_summary_separates_models_and_dims():
    results = [
        _dim_result(model="m1", dims={"reply.grounding": 5}),
        _dim_result(model="m2", dims={"reply.grounding": 1}),
    ]
    out = compute_dimension_summary(results)
    assert out[("m1", "r1", "reply.grounding")]["mean_score"] == 5.0
    assert out[("m2", "r1", "reply.grounding")]["mean_score"] == 1.0


def test_compute_dimension_summary_skips_results_with_no_rubric_scores():
    """Goal-only / factory-failure slots carry no rubric_scores → no dimension rows."""
    out = compute_dimension_summary([_dim_result(dims={})])
    assert out == {}


def test_compute_dimension_summary_empty_results():
    assert compute_dimension_summary([]) == {}


def test_compute_dimension_summary_drops_an_infra_excluded_results_score():
    """The judge's reading of a broken transcript must not enter the dimension mean.

    An infra-excluded cell is dropped from pass^k and from the case count, and it
    still carries a rubric score — the runner stores the judge's reading of
    whatever transcript survived, deliberately, for forensics. Pooling it here
    reported N=2 over one usable cell and dragged the mean toward the low score a
    judge gives a broken transcript, which is an apparatus fault scored as
    candidate quality.
    """
    results = [
        _dim_result(k=1, dims={"reply.grounding": 5}),
        _dim_result(k=2, dims={"reply.grounding": 1}, infra_error="apparatus: cassette miss in replay mode"),
    ]

    row = compute_dimension_summary(results)[("m1", "r1", "reply.grounding")]

    assert row["n"] == 1  # the excluded cell is not an observation
    assert row["mean_score"] == 5.0  # ...and its 1 does not drag the mean
    assert row["min_score"] == 5
    assert row["max_score"] == 5


def test_compute_dimension_summary_drops_a_judge_error_score_too():
    """``judge_error`` is infra, always — the same arm, so the same exclusion."""
    results = [
        _dim_result(k=1, dims={"reply.grounding": 4}),
        _dim_result(k=2, dims={"reply.grounding": 1}, judge_error="judge LLM returned 500"),
    ]

    assert compute_dimension_summary(results)[("m1", "r1", "reply.grounding")] == {
        "scale": "ordinal",
        "mean_score": 4.0,
        "min_score": 4,
        "max_score": 4,
        "n": 1,
    }


def test_compute_dimension_summary_keeps_a_candidate_failure_score():
    """A broken candidate is the candidate's own outcome, not the harness's.

    Pinned alongside the exclusion tests because one predicate decides both, and
    widening it to every error field would silently delete the failures pass^k
    exists to count.
    """
    results = [
        _dim_result(k=1, dims={"reply.grounding": 5}),
        _dim_result(k=2, dims={"reply.grounding": 1}, candidate_error="candidate LLM returned 400"),
    ]

    row = compute_dimension_summary(results)[("m1", "r1", "reply.grounding")]

    assert row["n"] == 2
    assert row["mean_score"] == 3.0


def test_compute_dimension_summary_all_excluded_group_reports_no_dimension_row():
    """An entirely infra-excluded group vanishes from this table, as it does from pass^k's denominator.

    A dimension key exists here only because some judge scored it, so there is no
    key under which to report the group unmeasured; the run summary keeps the
    group's headline row via ``compute_cost_summary`` and discloses the excluded
    cells there.
    """
    results = [
        _dim_result(k=1, dims={"reply.grounding": 1}, infra_error="apparatus: seed failed"),
        _dim_result(k=2, dims={"reply.grounding": 2}, judge_error="judge LLM returned 500"),
    ]

    assert compute_dimension_summary(results) == {}


def test_compute_dimension_summary_excludes_the_same_cells_pass_k_does():
    """The two aggregates must agree about which cells were measured.

    The defect this pins is the run-summary row that read ``Cases 1`` beside a
    dimension ``N 2``: one notion of exclusion, so ``n`` and ``n_test_cases``
    cannot disagree about a one-case-per-dim corpus.
    """
    results = [
        _dim_result(k=1, test_case_id="tc-a", dims={"reply.grounding": 5}),
        _dim_result(k=1, test_case_id="tc-b", dims={"reply.grounding": 1}, infra_error="apparatus: cassette miss"),
    ]

    dims = compute_dimension_summary(results)
    passk = compute_pass_hat_k(results)

    assert dims[("m1", "r1", "reply.grounding")]["n"] == passk[("m1", "r1")]["n_test_cases"] == 1


# =============================================================================
# Mean-composite aggregators — the continuous quality sibling of pass^k
# =============================================================================


class TestResultComposite:
    """Per-result normalization of rubric dims to a 0–1 quality score."""

    @pytest.mark.parametrize(("score", "expected"), [(1, 0.0), (3, 0.5), (5, 1.0)])
    def test_normalizes_single_dim_to_unit_interval(self, score: int, expected: float) -> None:
        r = make_eval_result(rubric_scores=[RubricScore(dim="x.d", score=score, scale="ordinal")])
        assert result_composite(r) == pytest.approx(expected)

    def test_averages_across_dims(self) -> None:
        r = make_eval_result(
            rubric_scores=[
                RubricScore(dim="x.a", score=4, scale="ordinal"),
                RubricScore(dim="x.b", score=2, scale="ordinal"),
            ]
        )
        assert result_composite(r) == pytest.approx(0.5)  # (0.75 + 0.25) / 2

    def test_infra_errored_result_has_no_composite(self) -> None:
        # Infra failures (infra_error / judge_error) are unmeasured → None (excluded,
        # not floored).
        assert result_composite(make_eval_result(infra_error="boom")) is None
        assert result_composite(make_eval_result(judge_error="judge boom")) is None

    def test_candidate_errored_result_composites_to_zero(self) -> None:
        # A broken candidate scores 0.0 (depress, don't vanish) even with rubric dims.
        assert result_composite(
            make_eval_result(candidate_error="402", rubric_scores=[RubricScore(dim="x.d", score=5, scale="ordinal")])
        ) == pytest.approx(0.0)

    def test_no_rubric_dims_has_no_composite(self) -> None:
        assert result_composite(make_eval_result(rubric_scores=[])) is None


class TestCompositeSummary:
    """``compute_composite_summary`` — per-(model, run) mean over cases."""

    def _r(self, tc: str, score: int, **kw: Any):
        return make_eval_result(
            eval_run_id="r",
            model="m",
            test_case_id=tc,
            rubric_scores=[RubricScore(dim="x.d", score=score, scale="ordinal")],
            **kw,
        )

    def test_means_over_cases_equally_weighted(self) -> None:
        summary = compute_composite_summary([self._r("tc1", 5), self._r("tc2", 1)])
        assert summary[("m", "r")]["mean_composite"] == pytest.approx(0.5)
        assert summary[("m", "r")]["n_cases"] == 2

    def test_averages_k_iterations_within_a_case_first(self) -> None:
        # Two iterations of ONE case (1.0 and 0.0) → one case at 0.5.
        results = [self._r("tc1", 5, k_iteration=1), self._r("tc1", 1, k_iteration=2)]
        summary = compute_composite_summary(results)
        assert summary[("m", "r")]["mean_composite"] == pytest.approx(0.5)
        assert summary[("m", "r")]["n_cases"] == 1

    def test_candidate_errored_case_depresses_composite_in_a_rubric_group(self) -> None:
        # tc1 scored 1.0; tc2 = candidate error → 0.0 (not hidden). Mean 0.5 over 2 cases.
        results = [
            self._r("tc1", 5),
            make_eval_result(eval_run_id="r", model="m", test_case_id="tc2", candidate_error="boom", rubric_scores=[]),
        ]
        summary = compute_composite_summary(results)
        assert summary[("m", "r")]["mean_composite"] == pytest.approx(0.5)
        assert summary[("m", "r")]["n_cases"] == 2

    def test_infra_errored_case_excluded_from_composite_not_floored(self) -> None:
        # tc1 scored 1.0; tc2 = infra error → EXCLUDED. Mean 1.0 over the 1 measured case.
        results = [
            self._r("tc1", 5),
            make_eval_result(
                eval_run_id="r", model="m", test_case_id="tc2", infra_error="cell timeout", rubric_scores=[]
            ),
        ]
        summary = compute_composite_summary(results)
        assert summary[("m", "r")]["mean_composite"] == pytest.approx(1.0)
        assert summary[("m", "r")]["n_cases"] == 1  # tc2 excluded, not a 0.0 floor

    def test_goal_only_group_has_null_composite(self) -> None:
        results = [
            make_eval_result(eval_run_id="r", model="m", test_case_id="tc1", rubric_scores=[]),
            make_eval_result(eval_run_id="r", model="m", test_case_id="tc2", rubric_scores=[]),
        ]
        summary = compute_composite_summary(results)
        assert summary[("m", "r")]["mean_composite"] is None
        assert summary[("m", "r")]["n_cases"] == 0


class TestPerCaseComposites:
    """``compute_per_case_composites`` — the per-case pairing atom for compare."""

    def test_one_value_per_case_for_rubric_groups(self) -> None:
        results = [
            make_eval_result(
                eval_run_id="r",
                model="m",
                test_case_id="tc1",
                rubric_scores=[RubricScore(dim="x.d", score=5, scale="ordinal")],
            ),
            make_eval_result(
                eval_run_id="r",
                model="m",
                test_case_id="tc2",
                rubric_scores=[RubricScore(dim="x.d", score=3, scale="ordinal")],
            ),
        ]
        per_case = compute_per_case_composites(results)
        assert per_case == {("m", "r", "tc1"): 1.0, ("m", "r", "tc2"): 0.5}

    def test_goal_only_group_contributes_nothing(self) -> None:
        results = [make_eval_result(eval_run_id="r", model="m", test_case_id="tc1", rubric_scores=[])]
        assert compute_per_case_composites(results) == {}


# =============================================================================
# "Can't tell" — excluded from the whole-trial measures, not from the other dims
# =============================================================================


class TestAJudgeThatCouldNotTell:
    """A rubric dim answered "can't tell" leaves pass^k and the composite; a scored one keeps it in."""

    @staticmethod
    def _pair():
        told = make_eval_result(
            test_case_id="tc-told",
            model="m1",
            eval_run_id="r1",
            rubric_scores=[RubricScore(dim="reply.tone", score=5, scale="ordinal")],
            judge_cannot_tell={"reply.refusal": "the user never asked for anything to refuse"},
        )
        scored = make_eval_result(
            test_case_id="tc-scored",
            model="m1",
            eval_run_id="r1",
            rubric_scores=[
                RubricScore(dim="reply.tone", score=5, scale="ordinal"),
                RubricScore(dim="reply.refusal", score=4, scale="ordinal"),
            ],
        )
        return told, scored

    def test_pass_k_counts_the_scored_trial_and_leaves_out_the_other(self):
        told, scored = self._pair()
        entry = compute_pass_hat_k([told, scored])[("m1", "r1")]
        assert entry["n_test_cases"] == 1
        assert entry["pass_hat_k"] == 1.0
        # Left out, and counted as left out for this reason — never a silent shrink.
        assert entry["n_cannot_tell_excluded"] == 1

    def test_the_composite_is_unmeasured_not_the_mean_of_the_rest(self):
        told, scored = self._pair()
        assert result_composite(told) is None
        assert result_composite(scored) is not None

    def test_a_cannot_tell_on_a_reserved_axis_leaves_the_whole_trial_measures_alone(self):
        """The transcript and outcome axes are in neither measure, so they cannot exclude from them."""
        result = make_eval_result(judge_cannot_tell={TRANSCRIPT_DIM_ID: "no turns"})
        assert result_composite(result) is not None
        assert compute_pass_hat_k([result])[("sonnet", "run-1")]["n_test_cases"] == 1

    def test_a_candidate_failure_still_fails_whatever_the_judge_could_not_tell(self):
        failed = make_eval_result(candidate_error="402", judge_cannot_tell={"reply.refusal": "nothing to read"})
        entry = compute_pass_hat_k([failed])[("sonnet", "run-1")]
        assert entry["n_test_cases"] == 1 and entry["pass_hat_k"] == 0.0


class TestWhatAJudgedDimCountsAs:
    """Per-dimension means read one rule: a failed turn's scores at the floor, a faulted one's not at all."""

    @staticmethod
    def _dims(*results: EvalResult) -> dict[str, float]:
        return {dim: row["mean_score"] for (_m, _r, dim), row in compute_dimension_summary(list(results)).items()}

    def test_a_cut_off_hold_turn_the_judge_passed_counts_as_a_fail(self):
        silent = _cut_off(
            make_scored_result(goal_passes=()).model_copy(
                update={"rubric_scores": [RubricScore(dim="reply.treats_as_present", scale="pass_fail", score=1)]}
            )
        )
        delivered = make_scored_result(test_case_id="tc2", goal_passes=()).model_copy(
            update={"rubric_scores": [RubricScore(dim="reply.treats_as_present", scale="pass_fail", score=1)]}
        )
        assert self._dims(delivered)["reply.treats_as_present"] == 1.0  # the control
        assert self._dims(silent, delivered)["reply.treats_as_present"] == 0.5

    def test_a_model_failure_counts_its_judged_level_at_the_bottom_of_the_scale(self):
        failed = make_scored_result(candidate_error="candidate turn LLM error: 502", rubric_scores=(("reply.tone", 4),))
        assert self._dims(failed)["reply.tone"] == 1.0

    def test_a_harness_fault_contributes_nothing(self):
        faulted = make_scored_result(infra_error="delivery never landed", rubric_scores=(("reply.tone", 5),))
        healthy = make_scored_result(test_case_id="tc2", rubric_scores=(("reply.tone", 3),))
        assert self._dims(faulted, healthy) == {"reply.tone": 3.0}
