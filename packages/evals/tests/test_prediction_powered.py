"""The prediction-powered estimate: the judge's mean corrected by people's ratings, beside the judge's own (#598).

The estimator's arithmetic is pinned here; whether its interval holds its level is
``test_simulated_prediction_powered.py``'s. The bundle half checks the estimate reaches every surface a judged reading
does (the judged arm, the cell's reading and the code-only report), reads people's ratings through the one pairing
agreement uses, and reads "not available" where the ratings were chosen rather than drawn at random (owner,
2026-11-14). A random sample's ratings, and the minimum they are stated from, are ``test_people_corrected.py``'s: the
bundle reads no blind read yet, so a rating it holds is drawn at random only where it rates the whole cell.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from threetears.evals.analysis import assemble_context_bundle
from threetears.evals.analysis.agreement import person_scores_by_result
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle, evidence_digest
from threetears.evals.analysis.bundle.surface import bundle_decision_surface
from threetears.evals.analysis.report import TableBlock, build_code_only_report
from threetears.evals.analysis.report.build import analysis_campaign_bundle
from threetears.evals.analysis.stats import (
    PPI_MIN_LABELLED_RESULTS,
    clustered_standard_error,
    prediction_powered_mean,
)
from threetears.evals.kernel.surface import PredictionPoweredReading
from threetears.evals.ops.analysis import analyses_list
from threetears.evals.run import rate_result
from threetears.evals.schema.models import EvalCaseStratum, EvalResult
from threetears.evals.schema.reading_states import StateOf
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.campaign_bundle_support import campaign_bundle
from packages.evals.tests.factories import analysed_aspect, make_analysis, make_calibration_rating, make_eval_result
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_campaign
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_JUDGED_DIMENSION, TOYHOST_SCOPE, ToyhostStorage
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import ops_fixture


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

    @pytest.mark.parametrize("rated", [PPI_MIN_LABELLED_RESULTS - 1, PPI_MIN_LABELLED_RESULTS])
    def test_ratings_of_part_of_a_cell_given_outside_a_blind_read_read_not_available_and_say_why(
        self, rated: int
    ) -> None:
        runs, bundle = _bundle(rated={1: rated}, shift=0)
        powered = _arms(bundle)[(runs[1].id,)].prediction_powered
        assert powered is not None
        assert (powered.mean, powered.rectifier, powered.ci_low) == (None, None, None)
        assert powered.n_labelled == rated
        assert powered.state == StateOf(state="not_available", reason="not_drawn_at_random")

    def test_an_agent_s_ratings_never_enter_it(self) -> None:
        campaign, toy = toyhost_campaign()
        runs = toy.load_eval_runs(campaign.run_ids, TOYHOST_SCOPE)
        results_by_run = {run.id: toy.query_eval_results_by_run(run.id, TOYHOST_SCOPE) for run in runs}
        storage = ToyhostStorage(runs, results_by_run)
        for result in results_by_run[runs[0].id]:
            score = result.judge_score(TOYHOST_JUDGED_DIMENSION)
            assert score is not None
            rate_result(
                storage,
                result_id=result.id,
                scope_id=TOYHOST_SCOPE,
                rubric_dim=TOYHOST_JUDGED_DIMENSION,
                rater="sim/reviewer",
                rater_kind="agent",
                score=score.score,
                reason="a model's reading",
            )
        bundle = assemble_context_bundle(campaign, storage=storage, profile=toyhost_profile())
        assert all(arm.prediction_powered is None for arm in _arms(bundle).values())

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
        report = build_code_only_report(bundle, assembled_at="2026-10-10T00:00:00+00:00")
        cells = [
            str(value)
            for block in report.blocks
            if isinstance(block, TableBlock)
            for row in block.values()
            for value in row.values()
            if isinstance(value, str) and "people's ratings" in value
        ]
        assert any(
            " (n=36 over 12 cases); with people's ratings: " in text and "rectifier -1" in text for text in cells
        )
        assert any(
            "with people's ratings: not available (3 rated by people, not drawn at random)" in text for text in cells
        )


class TestTheReadingSaysWhyItIsMissing:
    def test_a_missing_estimate_without_a_reason_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="exactly one of"):
            PredictionPoweredReading(n_labelled=3, n_labelled_cases=3, min_labelled=10)

    def test_a_stated_estimate_with_a_reason_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="exactly one of"):
            PredictionPoweredReading(
                n_labelled=12,
                n_labelled_cases=12,
                min_labelled=10,
                mean=3.0,
                state=StateOf(state="not_available", reason="too_few"),
            )

    def test_a_missing_estimate_is_not_available_never_another_state(self) -> None:
        with pytest.raises(ValidationError, match="not_available"):
            PredictionPoweredReading(
                n_labelled=3,
                n_labelled_cases=3,
                min_labelled=10,
                state=StateOf(state="set_aside", reason="people_disagree"),
            )


def _as_stored_before_state(document: object) -> object:
    """``document`` with every prediction-powered reading in it as a build before ``state`` dumped it: ``state`` gone,
    ``unavailable_reason`` written beside every field, null where the estimate is stated, else that build's sentence."""
    if isinstance(document, list):
        return [_as_stored_before_state(item) for item in document]
    if not isinstance(document, dict):
        return document
    moved = {key: _as_stored_before_state(value) for key, value in document.items()}
    powered = moved.get("prediction_powered")
    if isinstance(powered, dict):
        old = {key: value for key, value in powered.items() if key != "state"}
        old["unavailable_reason"] = (
            None
            if powered["mean"] is not None
            else (
                f"not available: {powered['n_labelled']} of the counted scores here were rated by a person, under "
                f"the {powered['min_labelled']} a prediction-powered estimate is stated from."
            )
        )
        moved["prediction_powered"] = old
    return moved


def _readings(document: object) -> list[dict]:
    """Every prediction-powered reading a document holds, at any depth."""
    if isinstance(document, list):
        return [reading for item in document for reading in _readings(item)]
    if not isinstance(document, dict):
        return []
    held = [document["prediction_powered"]] if isinstance(document.get("prediction_powered"), dict) else []
    return held + [reading for value in document.values() for reading in _readings(value)]


class TestAReadingStoredBeforeStateStillReads:
    """A frozen bundle and a frozen decision surface hold this shape; one stored before ``state`` replaced
    ``unavailable_reason`` reads, lists and dumps again, with no regeneration (the v13 loosening precedent)."""

    def test_the_old_key_is_dropped_and_a_missing_estimate_reads_too_few(self) -> None:
        stated = PredictionPoweredReading.from_dict(
            {
                "n_labelled": 12,
                "n_labelled_cases": 12,
                "min_labelled": 10,
                "mean": 3.0,
                "rectifier": -0.5,
                "sem": None,
                "ci_low": None,
                "ci_high": None,
                "unavailable_reason": None,
            }
        )
        assert (stated.mean, stated.state) == (3.0, None)
        missing = PredictionPoweredReading.from_dict(
            {
                "n_labelled": 3,
                "n_labelled_cases": 3,
                "min_labelled": 10,
                "mean": None,
                "rectifier": None,
                "sem": None,
                "ci_low": None,
                "ci_high": None,
                "unavailable_reason": "not available: 3 of the counted scores here were rated by a person, under the "
                "10 a prediction-powered estimate is stated from.",
            }
        )
        assert missing.state == StateOf(state="not_available", reason="too_few")
        assert "unavailable_reason" not in missing.to_dict()
        assert PredictionPoweredReading.from_dict(missing.to_dict()) == missing

    def test_an_old_reading_that_explained_nothing_is_still_refused(self) -> None:
        # The old shape refused a missing estimate with no sentence; loosening the read accepts nothing it refused.
        with pytest.raises(ValidationError, match="exactly one of"):
            PredictionPoweredReading.from_dict(
                {"n_labelled": 3, "n_labelled_cases": 3, "min_labelled": 10, "mean": None, "unavailable_reason": None}
            )

    def test_an_analysis_stored_before_it_reads_lists_and_dumps_again(self) -> None:
        store = InMemoryDocumentStore()
        fixture = off_loop(ops_fixture, store=store)
        host = fixture.host.eval_host
        _, bundle = _bundle(rated={0: 36, 1: 3}, shift=-1)
        surface = bundle_decision_surface(bundle)
        analysis = make_analysis(
            campaign_id=fixture.campaign.id,
            scope_id=TOYHOST_SCOPE,
            aspects=[analysed_aspect(bundle, decision_surface=surface)],
            campaign_bundle=campaign_bundle(bundle).frame(),
        )
        host.storage.save_analysis(analysis)
        before = analyses_list(host, fixture.campaign.id, TOYHOST_SCOPE)
        raw = store.get(analysis.id, TOYHOST_SCOPE)
        assert raw is not None
        old = _as_stored_before_state(raw)
        assert isinstance(old, dict)
        held = _readings(old["aspects"][0]["bundle"]) + _readings(old["aspects"][0]["decision_surface"])
        # Both kinds the old shape stored are in it, in the bundle and in the surface: a stated one and a missing one.
        assert _readings(old["aspects"][0]["decision_surface"])
        assert any(reading["unavailable_reason"] is None for reading in held)
        assert any(reading["unavailable_reason"] is not None for reading in held)
        assert all("state" not in reading for reading in held)
        store.upsert(old)

        # It lists as it did, line for line: nothing about it needs regenerating.
        assert analyses_list(host, fixture.campaign.id, TOYHOST_SCOPE) == before

        read = host.storage.load_analysis(analysis.id, TOYHOST_SCOPE)
        assert read is not None
        # The surface reads as it was frozen: a stated reading unchanged, a missing one with the old sentence's meaning.
        now, was = _readings(read.aspects[0].decision_surface.to_dict()), _readings(surface.to_dict())
        assert len(now) == len(was)
        for reading, frozen_reading in zip(now, was, strict=True):
            if frozen_reading["mean"] is not None:
                assert reading == frozen_reading
            else:
                assert reading == {**frozen_reading, "state": {"state": "not_available", "reason": "too_few"}}
        frozen = AnalysisContextBundle.from_dict(dict(read.aspects[0].bundle))
        missing = [
            arm.prediction_powered
            for measure in frozen.judged_measures
            for arm in measure.arms
            if arm.prediction_powered is not None and arm.prediction_powered.mean is None
        ]
        assert missing
        assert all(powered.state == StateOf(state="not_available", reason="too_few") for powered in missing)
        dumped = frozen.to_dict()
        assert _readings(dumped) and all("unavailable_reason" not in reading for reading in _readings(dumped))
        assert AnalysisContextBundle.from_dict(dumped) == frozen
        surface_again = read.aspects[0].decision_surface.to_dict()
        assert all("unavailable_reason" not in reading for reading in _readings(surface_again))

    def test_one_frozen_before_the_evidence_manifest_matches_the_fingerprint_taken_over_the_old_key(self) -> None:
        # A shape-66 aspect frozen before the manifest existed carries a v2: fingerprint, the digest of the document
        # itself: taken over `unavailable_reason`, which the re-read bundle no longer dumps. The frame is fingerprinted
        # over that old document, as the build that froze it did, never over today's shape.
        store = InMemoryDocumentStore()
        fixture = off_loop(ops_fixture, store=store)
        host = fixture.host.eval_host
        _, bundle = _bundle(rated={0: 36, 1: 3}, shift=-1)
        analysis = make_analysis(
            campaign_id=fixture.campaign.id,
            scope_id=TOYHOST_SCOPE,
            aspects=[analysed_aspect(bundle, decision_surface=bundle_decision_surface(bundle))],
            campaign_bundle=campaign_bundle(bundle).frame(),
        )
        host.storage.save_analysis(analysis)
        before = analyses_list(host, fixture.campaign.id, TOYHOST_SCOPE)

        old = _as_stored_before_state({key: value for key, value in bundle.to_dict().items() if key != "evidence"})
        assert isinstance(old, dict)
        held = _readings(old)
        assert any(reading["unavailable_reason"] is None for reading in held)
        assert any(reading["unavailable_reason"] is not None for reading in held)
        stored_fingerprint = evidence_digest(old)
        assert stored_fingerprint.startswith("v2:")
        raw = store.get(analysis.id, TOYHOST_SCOPE)
        assert raw is not None
        raw["aspects"][0] = {**raw["aspects"][0], "bundle": old, "bundle_fingerprint": stored_fingerprint}
        raw["campaign_bundle"]["aspects"] = [{"aspect": bundle.aspect, "fingerprint": stored_fingerprint}]
        store.upsert(raw)

        # Read today, the bundle answers the fingerprint it was frozen under, in the version it was frozen in.
        read = host.storage.load_analysis(analysis.id, TOYHOST_SCOPE)
        assert read is not None
        reread = AnalysisContextBundle.from_dict(dict(read.aspects[0].bundle))
        assert all("unavailable_reason" not in reading for reading in _readings(reread.to_dict()))
        assert reread.fingerprint() == stored_fingerprint
        assert analysis_campaign_bundle(read).aspects[0] == reread
        # It lists as it did, line for line: nothing about it needs regenerating.
        assert analyses_list(host, fixture.campaign.id, TOYHOST_SCOPE) == before

        # Saved again, it still reads as frozen.
        host.storage.save_analysis(read)
        again = host.storage.load_analysis(analysis.id, TOYHOST_SCOPE)
        assert again is not None
        assert analysis_campaign_bundle(again).aspects[0] == reread
        assert analyses_list(host, fixture.campaign.id, TOYHOST_SCOPE) == before
