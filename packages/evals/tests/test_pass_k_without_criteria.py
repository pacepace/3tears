"""An arm with nothing for pass^k to conjoin has no pass^k — unmeasured and said so, never 0.0 (#688); a label is
something to conjoin, so a classifier's arm has a measured one.

An arm whose attempts carry no goal-state check, no label, no check a host declares and no judge has nothing pass^k
can ask. pass^k once failed every such attempt for lacking a criterion, so every frontier point of such a grid read
0.0, a number a reader would act on. A classifier scored only against its expected label is not such an arm: a label
that did not match fails the trial, on the scoreboard as in pass^k, so its pass^k is the share of cases it labelled
right on every repeat, and its pass^1 is its pass rate. Seeded: the classifier's grades are drawn at random and every
expectation is counted from the draws; a judged arm beside them keeps the pass^k it always had.
"""

from __future__ import annotations


import random

from threetears.evals.analysis import assemble_context_bundle
from threetears.evals.analysis.lenses.frontier import compute_frontier
from threetears.evals.schema import RubricScore
from threetears.evals.schema.models import EvalResult, EvalRun, GoalStateOutcome, RoleUsage
from threetears.evals.kernel.scoring import (
    NO_PASS_CRITERION_REASON,
    compute_pass_hat_k,
    has_pass_criterion,
    pass_hat_k_at,
)
from threetears.evals.kernel.run_pages import ResultKey
from packages.evals.tests.factories import (
    fixture_variant_key,
    make_eval_result,
    make_eval_run,
    minimal_declaration,
    make_campaign,
)
from packages.evals.tests.fixtures.toyhost.corpus import ToyhostStorage
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile

CASES = 6
K = 2


def _arms(seed: int) -> tuple[EvalRun, EvalRun, EvalRun, dict[str, list[EvalResult]]]:
    """A classifier graded on its label alone, an arm graded on nothing, and a judged arm, over the same cases."""
    rng = random.Random(seed)
    classifier = make_eval_run(status="completed", candidate_model="classifier-model", k_runs=K)
    bare = make_eval_run(
        status="completed", candidate_model="bare-model", k_runs=K, subject_snapshot=classifier.subject_snapshot
    )
    judged = make_eval_run(
        status="completed", candidate_model="judged-model", k_runs=K, subject_snapshot=classifier.subject_snapshot
    )

    def ungraded(run: EvalRun, prefix: str, measures: dict[str, bool]) -> EvalResult:
        return make_eval_result(
            id=prefix,
            eval_run_id=run.id,
            scope_id=run.scope_id,
            model=run.candidate_model,
            test_case_id=prefix.split("/")[1],
            k_iteration=int(prefix.split("/")[2]),
            goal_state_outcomes=[],
            rubric_scores=[],
            judge_model=None,
            host_measures=measures,
            usage=[RoleUsage(role="candidate", cost_usd=0.01)],
        )

    results = {
        classifier.id: [
            ungraded(classifier, f"c/tc-{case}/{k}", {"match": rng.random() < 0.7})
            for case in range(CASES)
            for k in range(1, K + 1)
        ],
        bare.id: [ungraded(bare, f"b/tc-{case}/{k}", {}) for case in range(CASES) for k in range(1, K + 1)],
        judged.id: [
            make_eval_result(
                id=f"j-{case}-{k}",
                eval_run_id=judged.id,
                scope_id=judged.scope_id,
                model="judged-model",
                test_case_id=f"tc-{case}",
                k_iteration=k,
                goal_state_outcomes=[GoalStateOutcome(expression="saved", passed=True)],
                rubric_scores=[
                    RubricScore(
                        dim="extraction.accuracy", score=4, scale="ordinal", axis="capability", judge_temperature=0.0
                    )
                ],
                usage=[RoleUsage(role="candidate", cost_usd=0.02)],
            )
            for case in range(CASES)
            for k in range(1, K + 1)
        ],
    }
    return classifier, bare, judged, results


def _labelled_right(results: list[EvalResult]) -> tuple[float, float]:
    """The classifier's pass rate, and the share of its cases it labelled right on every repeat: its pass^K."""
    by_case: dict[str, list[bool]] = {}
    for result in results:
        by_case.setdefault(result.test_case_id, []).append(result.host_measures["match"] is True)
    rate = sum(sum(matches) for matches in by_case.values()) / len(results)
    every = sum(all(matches) for matches in by_case.values()) / len(by_case)
    return rate, every


def test_an_arm_with_no_criterion_has_no_pass_k_and_says_why_and_a_label_is_a_criterion() -> None:
    classifier, bare, judged, results = _arms(688)
    assert not any(has_pass_criterion(r, catalog={}) for r in results[bare.id])
    assert all(has_pass_criterion(r, catalog={}) for r in results[classifier.id])

    rows = compute_pass_hat_k([r for rs in results.values() for r in rs], catalog={})
    unmeasured = rows[("bare-model", bare.id)]
    assert unmeasured["pass_hat_k"] is None, "the old 0.0 read as failing criteria the arm never had"
    assert unmeasured["pass_hat_k_curve"] == [] and unmeasured["n_test_cases"] == 0
    assert unmeasured["n_no_criterion_excluded"] == CASES * K
    assert unmeasured["pass_hat_k_unmeasured_reason"] == NO_PASS_CRITERION_REASON

    rate, every = _labelled_right(results[classifier.id])
    assert 0 < every < 1, "the draws must leave some cases right on every repeat and some not"
    labelled = rows[("classifier-model", classifier.id)]
    assert labelled["pass_hat_k"] == every and labelled["n_test_cases"] == CASES
    assert pass_hat_k_at(labelled["pass_hat_k_curve"], 1)["pass_hat_k"] == rate
    assert labelled["n_no_criterion_excluded"] == 0 and labelled["pass_hat_k_unmeasured_reason"] is None

    unchanged = rows[("judged-model", judged.id)]
    assert unchanged["pass_hat_k"] == 1.0 and unchanged["n_no_criterion_excluded"] == 0
    assert unchanged["pass_hat_k_unmeasured_reason"] is None


def test_the_frontier_point_has_no_pass_k_and_no_cost_per_acceptable_outcome() -> None:
    classifier, bare, judged, results = _arms(1688)
    points = {
        p.model: p
        for p in compute_frontier(
            [classifier, bare, judged], [r for rs in results.values() for r in rs], archived_run_ids=None
        )
        .subjects[0]
        .points
    }

    point = points["bare-model"]
    assert point.pass_hat_k is None and point.pass_hat_k_curve == []
    assert point.pass_hat_k_unmeasured_reason == NO_PASS_CRITERION_REASON
    assert point.n_pass_no_criterion == CASES * K
    assert point.cost_per_acceptable_outcome is None
    assert point.production_replicating_cost is not None, "its cost is still measured"

    _, every = _labelled_right(results[classifier.id])
    labelled = points["classifier-model"]
    assert labelled.pass_hat_k == every and labelled.pass_hat_k_unmeasured_reason is None

    judged_point = points["judged-model"]
    assert judged_point.pass_hat_k == 1.0 and judged_point.pass_hat_k_unmeasured_reason is None
    assert judged_point.cost_per_acceptable_outcome is not None


def test_the_campaign_s_bundle_carries_the_reason_and_the_classifier_s_pass_rate() -> None:
    classifier, bare, judged, results = _arms(2688)
    campaign = make_campaign(
        scope_id=classifier.scope_id,
        name="a classifier and an ungraded arm beside a judged arm",
        subject_id=classifier.subject_snapshot.subject_id,
        subject_kind="s",
        behavior="b",
        run_ids=[classifier.id, bare.id, judged.id],
        declared_design=minimal_declaration(control=fixture_variant_key("judged-model")),
        created_by="test:fixture",
    )
    bundle = assemble_context_bundle(
        campaign, storage=ToyhostStorage([classifier, bare, judged], results), profile=toyhost_profile()
    )

    points = {p.model: p for subject in bundle.frontier.subjects for p in subject.points}
    assert points["bare-model"].pass_hat_k is None
    assert points["bare-model"].pass_hat_k_unmeasured_reason == NO_PASS_CRITERION_REASON

    board = next(row for row in bundle.scoreboard if row.variant_key == fixture_variant_key("classifier-model"))
    pass_hat_1 = pass_hat_k_at(points["classifier-model"].pass_hat_k_curve, 1)["pass_hat_k"]
    assert board.no_criterion == 0 and pass_hat_1 == board.passed / board.usable


def test_the_rendered_comparison_says_the_ungraded_arm_has_no_pass_k() -> None:
    from threetears.evals.analysis.reads import compare_two_runs
    from threetears.evals.kernel.errors import NotFoundError
    from threetears.evals.schema.models import EvalTemplate
    from threetears.evals.ops.lenses import RunsCompared, runs_compared_text

    classifier, bare, judged, results = _arms(3688)
    runs = {run.id: run for run in (classifier, bare, judged)}

    class _Store:
        def query_eval_result_keys_by_run(self, run_id: str, scope_id: str) -> list[ResultKey]:
            return [ResultKey(r.id, r.test_case_id, r.k_iteration, r.model, r.scored_at) for r in results[run_id]]

        def load_eval_results(self, result_ids: list[str], scope_id: str) -> list[EvalResult]:
            return [r for rows in results.values() for r in rows if r.id in set(result_ids)]

        def load_eval_run(self, run_id: str, scope_id: str) -> EvalRun:
            return runs[run_id]

    def no_template(template_id: str) -> EvalTemplate:
        raise NotFoundError("template", template_id, remedy="list them first")

    compared = compare_two_runs(
        _Store(),  # type: ignore[arg-type]
        judged.id,
        bare.id,
        judged.scope_id,
        load_template=no_template,
        subject_detail=lambda run: {},
    )
    arm = compared["comparison"]["arm"]
    assert arm["pass_hat_k_b"] is None and arm["pass_hat_k_delta"] is None
    assert arm["pass_hat_k_unmeasured_reason_b"] == NO_PASS_CRITERION_REASON
    assert arm["pass_hat_k_unmeasured_reason_a"] is None
    text = runs_compared_text(
        RunsCompared(
            baseline_run_id=judged.id,
            candidate_run_id=bare.id,
            comparison=compared,
            completeness_disclosures={},
            measurement_window_disclosure=None,
            cassette_mode_disclosure=None,
        )
    )
    assert f"pass^k of run {bare.id} is unmeasured: {NO_PASS_CRITERION_REASON}" in text
