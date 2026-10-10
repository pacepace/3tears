"""The prediction-powered estimate: the judge's mean corrected by people's ratings, beside the judge's own (#598).

The estimator's arithmetic is pinned here; whether its interval holds its level is
``test_simulated_prediction_powered.py``'s. The bundle half checks the estimate reaches every surface a judged reading
does (the judged arm, the cell's reading and the code-only report), reads people's ratings through the one pairing
agreement uses, and reads "not available" below the stated minimum.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from threetears.evals.analysis import assemble_context_bundle
from threetears.evals.analysis.agreement import person_scores_by_result
from threetears.evals.analysis.report import TableBlock, build_code_only_report
from threetears.evals.analysis.stats import (
    PPI_MIN_LABELLED_RESULTS,
    clustered_standard_error,
    prediction_powered_mean,
)
from threetears.evals.kernel.surface import PredictionPoweredReading
from threetears.evals.run import rate_result
from threetears.evals.schema.models import EvalCaseStratum, EvalResult
from packages.evals.tests.factories import make_calibration_rating, make_eval_result
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_campaign
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_JUDGED_DIMENSION, TOYHOST_SCOPE, ToyhostStorage
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile


class TestTheEstimator:
    def test_nobody_labelling_anything_gives_nothing_to_combine(self) -> None:
        assert prediction_powered_mean([3.0, 4.0], ["a", "b"], [None, None]) is None

    def test_a_constant_bias_is_measured_exactly_and_subtracted(self) -> None:
        judge = [4.0, 5.0, 4.0, 3.0, 5.0, 4.0]
        human = [3.0, None, 3.0, 2.0, None, 3.0]
        estimate = prediction_powered_mean(judge, ["a", "b", "c", "d", "e", "f"], human)
        assert estimate is not None
        assert estimate.judge_mean == pytest.approx(25.0 / 6.0)
        assert estimate.rectifier == pytest.approx(-1.0)
        assert estimate.mean == pytest.approx(25.0 / 6.0 - 1.0)

    def test_every_observation_labelled_reads_as_the_people_s_own_mean_and_clustered_error(self) -> None:
        judge = [4.0, 5.0, 4.0, 3.0, 5.0, 2.0]
        human = [3.0, 4.0, 4.0, 2.0, 5.0, 1.0]
        cases = ["a", "a", "b", "b", "c", "c"]
        estimate = prediction_powered_mean(judge, cases, human)
        assert estimate is not None
        assert estimate.mean == pytest.approx(sum(human) / len(human))
        assert estimate.sem == pytest.approx(clustered_standard_error(human, cases))

    def test_labels_on_one_case_alone_state_the_estimate_with_no_interval(self) -> None:
        estimate = prediction_powered_mean([4.0, 4.0, 3.0], ["a", "a", "b"], [3.0, 3.0, None])
        assert estimate is not None
        assert (estimate.mean, estimate.sem, estimate.interval) == (pytest.approx(8.0 / 3.0), None, None)

    def test_the_interval_is_clipped_to_the_scale_and_the_estimate_is_not(self) -> None:
        judge = [5.0] * 10
        human = [5.0, 5.0, 5.0, 5.0, 4.0, None, None, None, None, None]
        estimate = prediction_powered_mean(judge, list("abcdefghij"), human, value_range=(1.0, 5.0))
        assert estimate is not None and estimate.interval is not None
        assert estimate.interval[1] == 5.0
        assert estimate.interval[0] < estimate.mean < 5.0

    def test_the_variance_carries_the_covariance_of_two_terms_over_one_set(self) -> None:
        # Unclustered and every result in the judge mean: the residual form equals the two Angelopoulos terms plus
        # 2 cov(f, y - f) / N over the labelled ones, scaled by the labelled cases' small-sample factor.
        judge = [4.0, 2.0, 5.0, 3.0, 4.0, 1.0, 2.0, 5.0]
        human: list[float | None] = [3.0, 2.0, 5.0, 1.0, None, None, None, None]
        estimate = prediction_powered_mean(judge, list("abcdefgh"), human)
        assert estimate is not None and estimate.sem is not None
        n_all, rated = len(judge), [(f, y) for f, y in zip(judge, human) if y is not None]
        f_bar, r_bar = sum(judge) / n_all, sum(y - f for f, y in rated) / len(rated)
        residuals = [
            (f - f_bar) / n_all + ((y - f - r_bar) / len(rated) if y is not None else 0.0) for f, y in zip(judge, human)
        ]
        g = len(rated)
        assert estimate.sem == pytest.approx(math.sqrt(g / (g - 1) * sum(r * r for r in residuals)))


class TestTheLabelsAreAgreementsOwn:
    def test_an_agent_s_rating_and_a_rating_on_another_scale_label_nothing(self) -> None:
        result = make_eval_result(id="r-1")
        dim = result.rubric_scores[0].dim
        scale = result.rubric_scores[0].scale
        person = make_calibration_rating(result_id="r-1", rubric_dim=dim, scale=scale, score=2, rater="p")
        agent = make_calibration_rating(
            result_id="r-1", rubric_dim=dim, scale=scale, score=1, rater="bot", rater_kind="agent"
        )
        assert person_scores_by_result([person, agent], [result]) == {("r-1", dim): [2]}


class _OneStratumToyhost(ToyhostStorage):
    """The toy corpus with every document declaring one stratum, so the code-only report lays out its strata table —
    the table a judged reading's figure is printed in."""

    def load_case_strata(self, test_case_ids: Sequence[str], scope_id: str) -> list[EvalCaseStratum]:
        return [
            stratum.model_copy(update={"stratum": "documents"})
            for stratum in super().load_case_strata(test_case_ids, scope_id)
        ]


def _bundle(*, rated: dict[int, int], shift: int, stratified: bool = False):
    """The toy campaign's bundle with the first ``rated[i]`` results of run ``i`` rated by a person at judge + shift."""
    campaign, toy = toyhost_campaign()
    runs = toy.load_eval_runs(campaign.run_ids, TOYHOST_SCOPE)
    results_by_run: dict[str, list[EvalResult]] = {
        run.id: toy.query_eval_results_by_run(run.id, TOYHOST_SCOPE) for run in runs
    }
    storage = (_OneStratumToyhost if stratified else ToyhostStorage)(runs, results_by_run)
    for index, run in enumerate(runs):
        for result in results_by_run[run.id][: rated.get(index, 0)]:
            score = result.judge_score(TOYHOST_JUDGED_DIMENSION)
            assert score is not None
            rate_result(
                storage,
                result_id=result.id,
                scope_id=TOYHOST_SCOPE,
                rubric_dim=TOYHOST_JUDGED_DIMENSION,
                rater="reviewer-1",
                rater_kind="person",
                score=score.score + shift,
                reason="read it against the source",
            )
    return runs, assemble_context_bundle(campaign, storage=storage, profile=toyhost_profile())


def _arms(bundle):
    (measure,) = [m for m in bundle.judged_measures if m.name == TOYHOST_JUDGED_DIMENSION]
    return {tuple(arm.run_ids): arm for arm in measure.arms}


class TestTheBundleCarriesItBesideTheJudge:
    def test_no_ratings_carry_no_estimate(self) -> None:
        _, bundle = _bundle(rated={}, shift=0)
        assert all(arm.prediction_powered is None for arm in _arms(bundle).values())

    def test_a_fully_rated_arm_is_corrected_by_the_people_s_shift_and_the_judge_s_mean_stands(self) -> None:
        # Run 0's judge scored 3s and 4s; people scored every one a point lower.
        runs, bundle = _bundle(rated={0: 36}, shift=-1)
        arms = _arms(bundle)
        arm = arms[(runs[0].id,)]
        powered = arm.prediction_powered
        assert powered is not None and powered.mean is not None and arm.mean is not None
        assert powered.rectifier == pytest.approx(-1.0)
        assert powered.mean == pytest.approx(arm.mean - 1.0)
        assert (powered.n_labelled, powered.n_labelled_cases) == (36, 12)
        # Every result rated, at a constant shift: the people's scores spread exactly as the judge's do.
        assert powered.sem == pytest.approx(arm.sem, abs=1e-12)
        assert powered.ci_low is not None and powered.ci_high is not None
        assert powered.ci_low <= powered.mean <= powered.ci_high
        assert arms[(runs[1].id,)].prediction_powered is None, "nobody rated the other arm"

    def test_below_the_minimum_it_reads_not_available_and_says_so(self) -> None:
        runs, bundle = _bundle(rated={1: PPI_MIN_LABELLED_RESULTS - 1}, shift=0)
        powered = _arms(bundle)[(runs[1].id,)].prediction_powered
        assert powered is not None
        assert (powered.mean, powered.rectifier, powered.ci_low) == (None, None, None)
        assert powered.n_labelled == PPI_MIN_LABELLED_RESULTS - 1
        assert powered.unavailable_reason is not None and powered.unavailable_reason.startswith("not available")

    def test_at_the_minimum_it_is_stated(self) -> None:
        runs, bundle = _bundle(rated={1: PPI_MIN_LABELLED_RESULTS}, shift=0)
        powered = _arms(bundle)[(runs[1].id,)].prediction_powered
        assert powered is not None and powered.mean is not None

    def test_the_cell_s_reading_is_the_arm_s(self) -> None:
        _, bundle = _bundle(rated={0: 36}, shift=-1)
        arms = sorted((arm.prediction_powered for arm in _arms(bundle).values()), key=repr)
        cells = sorted(
            (
                reading.prediction_powered
                for cell in bundle.cell_measures
                for reading in cell.judged
                if reading.dimension == TOYHOST_JUDGED_DIMENSION
            ),
            key=repr,
        )
        assert cells == arms
        assert any(powered is not None for powered in cells)

    def test_ratings_move_the_fingerprint(self) -> None:
        assert _bundle(rated={}, shift=0)[1].fingerprint() != _bundle(rated={0: 36}, shift=-1)[1].fingerprint()

    def test_the_code_only_report_puts_it_after_the_judge_s_figure(self) -> None:
        _, bundle = _bundle(rated={0: 36, 1: 3}, shift=-1, stratified=True)
        report = build_code_only_report(
            bundle, measures=toyhost_profile().measures, assembled_at="2026-10-10T00:00:00+00:00"
        )
        cells = [
            str(value)
            for block in report.blocks
            if isinstance(block, TableBlock)
            for row in block.rows
            for value in row.values()
            if isinstance(value, str) and "people's ratings" in value
        ]
        assert any(
            " (n=36 over 12 cases); with people's ratings: " in text and "rectifier -1" in text for text in cells
        )
        assert any("with people's ratings: not available (3 rated by people, under 10)" in text for text in cells)


class TestTheReadingSaysWhyItIsMissing:
    def test_a_missing_estimate_without_a_reason_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="exactly one of"):
            PredictionPoweredReading(n_labelled=3, n_labelled_cases=3, min_labelled=10)

    def test_a_stated_estimate_with_a_reason_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="exactly one of"):
            PredictionPoweredReading(
                n_labelled=12, n_labelled_cases=12, min_labelled=10, mean=3.0, unavailable_reason="not available"
            )
