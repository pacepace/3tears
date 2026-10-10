"""pass^k passes a criterion at the behavior's declared threshold, records it, and every label states it (#642).

Every attempt here scores 4 on one capability criterion and passes its goal check. At the default threshold
of 3 each attempt passes and pass^k is 1; at a declared 5 none does and it is 0. The figure records which, and
the surfaces that print pass^k say so.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from threetears.evals.analysis import assemble_context_bundle
from threetears.evals.analysis.bundle import bundle_decision_surface
from threetears.evals.contracts import EvalCampaign, RubricScore
from threetears.evals.contracts.host import (
    DEFAULT_PASS_THRESHOLD,
    BarRegistry,
    PassThreshold,
    pass_threshold_label,
)
from threetears.evals.contracts.host.bars import BarRegistrationError
from threetears.evals.contracts.metrics import METRIC_DESCRIPTORS
from threetears.evals.contracts.models import GoalStateOutcome
from packages.evals.tests.factories import fixture_variant_key, make_eval_result, make_eval_run, minimal_declaration
from packages.evals.tests.fixtures.toyhost.corpus import ToyhostStorage
from packages.evals.tests.fixtures.toyhost.profile import TOYHOST_BARS, toyhost_profile

DECLARED = "extract_invoice_fields"
UNDECLARED = "summarise_invoices"


def _profile(threshold: int):
    return replace(
        toyhost_profile(),
        bars=BarRegistry(
            TOYHOST_BARS,
            pass_thresholds=[
                PassThreshold(DECLARED, threshold, rationale="a field read only acceptably still costs a review")
            ],
        ),
    )


def _bundle(behavior: str, threshold: int, *, score: int = 4):
    runs, results = [], {}
    for model in ("control-model", "contrast-model"):
        run = make_eval_run(status="completed", candidate_model=model, k_runs=1)
        runs.append(run)
        results[run.id] = [
            make_eval_result(
                id=f"{model}-{case}",
                eval_run_id=run.id,
                scope_id=run.scope_id,
                model=model,
                test_case_id=f"tc-{case}",
                goal_state_outcomes=[GoalStateOutcome(expression="saved", passed=True)],
                rubric_scores=[RubricScore(dim="extraction.accuracy", score=score, scale="ordinal")],
            )
            for case in range(4)
        ]
    campaign = EvalCampaign(
        scope_id=runs[0].scope_id,
        name="threshold",
        subject_id=runs[0].subject_snapshot.subject_id,
        subject_kind="s",
        behavior=behavior,
        run_ids=[run.id for run in runs],
        declared_design=minimal_declaration(control=fixture_variant_key("control-model")),
        created_by="test:fixture",
    )
    return assemble_context_bundle(campaign, storage=ToyhostStorage(runs, results), profile=_profile(threshold))


def _pass_hat_k(bundle) -> set[float | None]:
    return {point.pass_hat_k for subject in bundle.frontier.subjects for point in subject.points}


class TestTheDeclaredThreshold:
    def test_a_behavior_declaring_five_computes_and_records_pass_k_at_five(self) -> None:
        bundle = _bundle(DECLARED, 5)
        assert bundle.frontier.rubric_threshold == 5
        assert _pass_hat_k(bundle) == {0.0}, "a 4 does not clear a declared 5"
        assert bundle_decision_surface(bundle).rubric_threshold == 5

    @pytest.mark.parametrize(("score", "expected"), [(3, 0.0), (4, 1.0)])
    def test_a_behavior_declaring_four_computes_and_records_four(self, score: int, expected: float) -> None:
        bundle = _bundle(DECLARED, 4, score=score)
        assert bundle.frontier.rubric_threshold == 4
        assert _pass_hat_k(bundle) == {expected}, "a 3 clears the default 3 and not a declared 4"

    def test_a_behavior_with_no_declaration_records_three(self) -> None:
        bundle = _bundle(UNDECLARED, 5)
        assert bundle.frontier.rubric_threshold == DEFAULT_PASS_THRESHOLD == 3
        assert _pass_hat_k(bundle) == {1.0}
        assert bundle_decision_surface(bundle).rubric_threshold == 3


class TestTheRegistry:
    def test_an_undeclared_behavior_and_no_behavior_read_the_default(self) -> None:
        registry = _profile(4).bars
        assert registry.pass_threshold(DECLARED) == 4
        assert registry.pass_threshold(UNDECLARED) == DEFAULT_PASS_THRESHOLD
        assert registry.pass_threshold(None) == DEFAULT_PASS_THRESHOLD

    @pytest.mark.parametrize(
        ("thresholds", "match"),
        [
            ([PassThreshold("b", 1, "every score clears one, which is nothing")], "outside 2-5"),
            ([PassThreshold("b", 6, "a level the scale does not have")], "outside 2-5"),
            ([PassThreshold("b", 4, "")], "no rationale"),
            ([PassThreshold("b", 4, "one reason"), PassThreshold("b", 3, "another")], "declared twice"),
        ],
    )
    def test_an_unsound_declaration_is_refused(self, thresholds: list[PassThreshold], match: str) -> None:
        with pytest.raises(BarRegistrationError, match=match):
            BarRegistry(pass_thresholds=thresholds)


class TestTheLabels:
    def test_the_label_names_the_depth_and_the_threshold(self) -> None:
        assert pass_threshold_label(3, 4) == "pass^k (k=3, criterion >= 4 of 5)"

    def test_the_descriptions_name_the_threshold_and_where_it_comes_from(self) -> None:
        description = METRIC_DESCRIPTORS["pass_hat_k"].description
        assert "rubric_threshold" in description and "PassThreshold" in description and "3 where" in description
        for name in ("pass_hat_k_a", "pass_hat_k_b", "pass_hat_k_delta"):
            assert "rubric_threshold" in METRIC_DESCRIPTORS[name].description
