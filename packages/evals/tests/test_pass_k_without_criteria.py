"""An arm with nothing for pass^k to conjoin has no pass^k — unmeasured and said so, never 0.0 (#688).

A classifier scored only against its expected label carries no goal-state check and no judge. pass^k once
failed every one of its attempts for lacking a criterion, so every frontier point of such a grid read 0.0, a
number a reader would act on. Seeded: the classifier's grades are drawn at random and change nothing here,
because pass^k never reads them; a judged arm beside it keeps the pass^k it always had.
"""

from __future__ import annotations

import json
import random

from threetears.evals.analysis import assemble_context_bundle
from threetears.evals.analysis.generator import build_user_message
from threetears.evals.analysis.lenses.frontier import compute_frontier
from threetears.evals.kernel import EvalCampaign
from threetears.evals.schema import RubricScore
from threetears.evals.schema.models import GoalStateOutcome, RoleUsage
from threetears.evals.kernel.scoring import NO_PASS_CRITERION_REASON, compute_pass_hat_k, has_pass_criterion
from packages.evals.tests.factories import fixture_variant_key, make_eval_result, make_eval_run, minimal_declaration
from packages.evals.tests.fixtures.toyhost.corpus import ToyhostStorage
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile

CASES = 6
K = 2


def _arms(seed: int):
    rng = random.Random(seed)
    classifier = make_eval_run(status="completed", candidate_model="classifier-model", k_runs=K)
    judged = make_eval_run(
        status="completed", candidate_model="judged-model", k_runs=K, subject_snapshot=classifier.subject_snapshot
    )
    results = {
        classifier.id: [
            make_eval_result(
                id=f"c-{case}-{k}",
                eval_run_id=classifier.id,
                scope_id=classifier.scope_id,
                model="classifier-model",
                test_case_id=f"tc-{case}",
                k_iteration=k,
                goal_state_outcomes=[],
                rubric_scores=[],
                judge_model=None,
                host_measures={"match": rng.random() < 0.7},
                usage=[RoleUsage(role="candidate", cost_usd=0.01)],
            )
            for case in range(CASES)
            for k in range(1, K + 1)
        ],
        judged.id: [
            make_eval_result(
                id=f"j-{case}-{k}",
                eval_run_id=judged.id,
                scope_id=judged.scope_id,
                model="judged-model",
                test_case_id=f"tc-{case}",
                k_iteration=k,
                goal_state_outcomes=[GoalStateOutcome(expression="saved", passed=True)],
                rubric_scores=[RubricScore(dim="extraction.accuracy", score=4, scale="ordinal")],
                usage=[RoleUsage(role="candidate", cost_usd=0.02)],
            )
            for case in range(CASES)
            for k in range(1, K + 1)
        ],
    }
    return classifier, judged, results


def test_no_attempt_has_a_criterion_so_the_run_has_no_pass_k_and_says_why() -> None:
    classifier, judged, results = _arms(688)
    assert not any(has_pass_criterion(r) for r in results[classifier.id])

    rows = compute_pass_hat_k([*results[classifier.id], *results[judged.id]])
    unmeasured = rows[("classifier-model", classifier.id)]
    assert unmeasured["pass_hat_k"] is None, "the old 0.0 read as failing criteria the arm never had"
    assert unmeasured["pass_hat_k_curve"] == [] and unmeasured["n_test_cases"] == 0
    assert unmeasured["n_no_criterion_excluded"] == CASES * K
    assert unmeasured["pass_hat_k_unmeasured_reason"] == NO_PASS_CRITERION_REASON

    unchanged = rows[("judged-model", judged.id)]
    assert unchanged["pass_hat_k"] == 1.0 and unchanged["n_no_criterion_excluded"] == 0
    assert unchanged["pass_hat_k_unmeasured_reason"] is None


def test_the_frontier_point_has_no_pass_k_and_no_cost_per_acceptable_outcome() -> None:
    classifier, judged, results = _arms(1688)
    points = {
        p.model: p
        for p in compute_frontier(
            [classifier, judged], [r for rs in results.values() for r in rs], archived_run_ids=None
        )
        .subjects[0]
        .points
    }

    point = points["classifier-model"]
    assert point.pass_hat_k is None and point.pass_hat_k_curve == []
    assert point.pass_hat_k_unmeasured_reason == NO_PASS_CRITERION_REASON
    assert point.n_pass_no_criterion == CASES * K
    assert point.cost_per_acceptable_outcome is None
    assert point.production_replicating_cost is not None, "its cost is still measured"

    judged_point = points["judged-model"]
    assert judged_point.pass_hat_k == 1.0 and judged_point.pass_hat_k_unmeasured_reason is None
    assert judged_point.cost_per_acceptable_outcome is not None


def test_the_bundle_the_writer_reads_carries_the_reason() -> None:
    classifier, judged, results = _arms(2688)
    campaign = EvalCampaign(
        scope_id=classifier.scope_id,
        name="classifier beside a judged arm",
        subject_id=classifier.subject_snapshot.subject_id,
        subject_kind="s",
        behavior="b",
        run_ids=[classifier.id, judged.id],
        declared_design=minimal_declaration(control=fixture_variant_key("judged-model")),
        created_by="test:fixture",
    )
    bundle = assemble_context_bundle(
        campaign, storage=ToyhostStorage([classifier, judged], results), profile=toyhost_profile()
    )

    points = {p.model: p for subject in bundle.frontier.subjects for p in subject.points}
    assert points["classifier-model"].pass_hat_k is None
    assert json.dumps(NO_PASS_CRITERION_REASON) in build_user_message(bundle)


def test_the_rendered_comparison_says_the_classifier_has_no_pass_k() -> None:
    from threetears.evals.analysis.reads import compare_two_runs
    from threetears.evals.kernel.errors import NotFoundError
    from threetears.evals.schema.models import EvalTemplate
    from threetears.evals.ops.lenses import RunsCompared, runs_compared_text

    classifier, judged, results = _arms(3688)

    class _Store:
        def query_eval_results_by_run(self, run_id: str, scope_id: str):
            return results[run_id]

        def load_eval_run(self, run_id: str, scope_id: str):
            return classifier if run_id == classifier.id else judged

    def no_template(template_id: str) -> EvalTemplate:
        raise NotFoundError("template", template_id)

    compared = compare_two_runs(
        _Store(),  # type: ignore[arg-type]
        judged.id,
        classifier.id,
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
            candidate_run_id=classifier.id,
            comparison=compared,
            completeness_disclosures={},
            measurement_window_disclosure=None,
            cassette_mode_disclosure=None,
        )
    )
    assert f"pass^k of run {classifier.id} is unmeasured: {NO_PASS_CRITERION_REASON}" in text
