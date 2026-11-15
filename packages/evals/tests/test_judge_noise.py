"""Re-score noise: the judge checked against itself, as figures in the judge's own table and nowhere else.

What is held here, by fixture (the statistics against a planted truth are ``test_simulated_judge_noise.py``):

- an arm's figures exist from :data:`RESCORE_MIN_RESULTS` repeated results over :data:`RESCORE_MIN_CASES` cases, and
  below either the state is ``not available`` with the shared reason ``too_few`` and no figure is given;
- the variance is half the mean squared difference between a reading and its repeat, and a judge that never moved
  reads zero with a range above it, never a certainty;
- ``e`` and the ceiling ``(1 - e)^(k - 1)`` are read on a pass/fail dimension counting toward pass, at the arm's k;
- the share exists at k = 1 and is ``σ² (1/k_control + 1/k_candidate)`` over the per-case differences' variance, its
  range one interval on the sum, inside the share taken at each end of the arms' ranges;
- where the ceiling is read and no first reading passed, it alone is ``not available``, ``too_few``;
- end to end, a repeat by another judge is not read, an arm nobody repeated carries no figure, an arm whose repeats
  were named by id or stored before their draw was recorded is ``not available`` with its reason, and the figures are
  in the judge's table data only: the judged arms and the judged contrasts of the bundle, never the report input.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.evals.analysis import AnalysisContextBundle, assemble_context_bundle
from threetears.evals.analysis.judge_noise import (
    RESCORE_MIN_CASES,
    RESCORE_MIN_RESULTS,
    Repeated,
    SideNoise,
    judge_share,
    rescore_of,
    rescore_variance,
)
from threetears.evals.analysis.report_input_build import report_input
from threetears.evals.kernel import IDENTITY_VERSION, CampaignArm
from threetears.evals.schema.models import EvalResult, JudgeRepeat, RepeatedScore, RubricScore
from threetears.evals.schema.reading_states import SampleKind, StateOf
from packages.evals.tests.campaign_bundle_support import campaign_bundle
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_campaign
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_JUDGED_DIMENSION, TOYHOST_SCOPE, ToyhostStorage
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile

_ORDINAL = (1.0, 5.0)
_PASS_FAIL = (0.0, 1.0)
_TOO_FEW = StateOf(state="not_available", reason="too_few")


def _repeats(results: int, cases: int, *, moved: int = 0) -> list[Repeated]:
    """``results`` repeated results spread over ``cases`` cases, read 3 first; the first ``moved`` re-read 4."""
    return [Repeated(f"c{index % cases}", 3.0, (4.0 if index < moved else 3.0,)) for index in range(results)]


class TestTheFloor:
    @pytest.mark.parametrize(
        ("results", "cases"),
        [(RESCORE_MIN_RESULTS - 1, RESCORE_MIN_CASES), (RESCORE_MIN_RESULTS, RESCORE_MIN_CASES - 1)],
    )
    def test_below_it_the_figures_are_not_available_too_few(self, results: int, cases: int) -> None:
        rescore, variance = rescore_of(_repeats(results, cases, moved=5), value_range=_ORDINAL, k=2, passed=None)
        assert rescore.state == _TOO_FEW
        assert (rescore.repeated_results, rescore.repeated_cases) == (results, cases)
        assert rescore.variance is None and rescore.fail_given_pass is None and rescore.ceiling_low is None
        assert rescore.variance_interval is None and "variance_interval" not in rescore.model_dump()
        assert variance is None

    def test_at_it_they_are_read(self) -> None:
        rescore, variance = rescore_of(
            _repeats(RESCORE_MIN_RESULTS, RESCORE_MIN_CASES, moved=5), value_range=_ORDINAL, k=2, passed=None
        )
        assert rescore.state is None and variance is not None
        assert rescore.variance == pytest.approx(5 * 0.5 / RESCORE_MIN_RESULTS)
        # The variance carries its own range: at twenty results it is wide, and the reader needs it.
        assert rescore.variance_interval == (variance.low, variance.high)
        assert variance.low < variance.point < variance.high


class TestTheVariance:
    def test_it_is_half_the_mean_squared_difference(self) -> None:
        # Ten of twenty results moved by one step, one of them by two: (9 * 1 + 4) / 2 / 20.
        repeated = _repeats(20, 10, moved=9) + [Repeated("c0", 3.0, (5.0,))]
        variance = rescore_variance(repeated[:9] + repeated[10:], value_range=_ORDINAL)
        assert variance is not None
        assert variance.point == pytest.approx((9 * 1 + 4) / 2 / 20)
        assert variance.low <= variance.point <= variance.high <= 4.0

    def test_several_repeats_of_one_result_are_one_reading_of_it(self) -> None:
        # A result re-read twice, once moved by two and once not, reads (4 + 0) / 2 / 2 = 1.
        variance = rescore_variance(
            [Repeated("c0", 3.0, (5.0, 3.0)), Repeated("c1", 3.0, (3.0,))], value_range=_ORDINAL
        )
        assert variance is not None and variance.point == pytest.approx(0.5)

    def test_a_judge_that_never_moved_reads_zero_with_a_range_above_it(self) -> None:
        variance = rescore_variance(_repeats(20, 10), value_range=_ORDINAL)
        assert variance is not None
        assert (variance.point, variance.low) == (0.0, 0.0) and variance.high > 0.0

    def test_none_without_a_repeat(self) -> None:
        assert rescore_variance([], value_range=_ORDINAL) is None


class TestTheCeiling:
    @staticmethod
    def _pass_fail(failed: int, k: int) -> Any:
        repeated = [Repeated(f"c{index % 10}", 1.0, (0.0 if index < failed else 1.0,)) for index in range(20)]
        # Results first read as a fail are not in e's base, whatever their repeat said.
        repeated += [Repeated(f"c{index}", 0.0, (1.0,)) for index in range(5)]
        rescore, _ = rescore_of(repeated, value_range=_PASS_FAIL, k=k, passed=1.0)
        return rescore

    def test_e_is_the_share_of_passes_that_fail_on_a_repeat(self) -> None:
        rescore = self._pass_fail(2, 3)
        assert rescore.fail_given_pass == pytest.approx(0.1)
        assert rescore.fail_given_pass_interval is not None
        low, high = rescore.fail_given_pass_interval
        assert low <= 0.1 <= high

    def test_the_ceiling_is_a_range_about_one_minus_e_to_the_k_minus_one_at_the_arm_s_k(self) -> None:
        rescore = self._pass_fail(2, 3)
        assert rescore.k == 3
        assert rescore.fail_given_pass_interval is not None and rescore.ceiling_low is not None
        assert rescore.ceiling_high is not None
        low_e, high_e = rescore.fail_given_pass_interval
        assert rescore.ceiling_low == pytest.approx((1 - high_e) ** 2)
        assert rescore.ceiling_high == pytest.approx((1 - low_e) ** 2)
        assert rescore.ceiling_low <= 0.81 <= rescore.ceiling_high

    def test_at_one_attempt_there_is_no_ceiling(self) -> None:
        rescore = self._pass_fail(2, 1)
        assert (rescore.ceiling_low, rescore.ceiling_high) == (1.0, 1.0)

    def test_no_ceiling_is_read_where_the_dimension_does_not_count_toward_pass(self) -> None:
        rescore, _ = rescore_of(_repeats(20, 10, moved=4), value_range=_ORDINAL, k=3, passed=None)
        assert rescore.variance is not None
        assert (rescore.fail_given_pass, rescore.ceiling_low, rescore.ceiling_high, rescore.k) == (None,) * 4
        assert rescore.ceiling_state is None

    def test_a_ceiling_read_for_the_dimension_and_not_readable_is_not_available_too_few(self) -> None:
        # Every first reading failed: no pass for a repeat to fail, so no e and no ceiling, and it says why.
        repeated = [Repeated(f"c{index % 10}", 0.0, (1.0 if index < 3 else 0.0,)) for index in range(20)]
        rescore, _ = rescore_of(repeated, value_range=_PASS_FAIL, k=3, passed=1.0)
        assert rescore.state is None and rescore.variance is not None
        assert (rescore.fail_given_pass, rescore.ceiling_low, rescore.ceiling_high, rescore.k) == (None,) * 4
        assert rescore.ceiling_state == _TOO_FEW

    def test_a_ceiling_read_carries_no_state(self) -> None:
        assert self._pass_fail(2, 3).ceiling_state is None


class TestTheShare:
    @staticmethod
    def _side(depth: int) -> SideNoise:
        """A side whose re-score variance is 0.25: ten of twenty results moved by one step."""
        variance = rescore_variance(_repeats(20, 10, moved=10), value_range=_ORDINAL)
        assert variance is not None and variance.point == pytest.approx(0.25)
        return SideNoise(variance, {"a": depth, "b": depth, "c": depth})

    def test_it_exists_at_k_equal_one(self) -> None:
        # Differences -1, 0, 1: variance 1. Each side's re-score variance 0.25 over one score per case: 0.5.
        control, candidate = self._side(1), self._side(1)
        share = judge_share(control, candidate, ["a", "b", "c"], [-1.0, 0.0, 1.0])
        assert share is not None
        assert share.point == pytest.approx(0.5)
        assert share.low < share.point < share.high

    def test_its_range_is_one_interval_on_the_sum_not_the_share_at_each_end_of_the_arms_ranges(self) -> None:
        # Two arms' errors partly cancel: the sum's range sits strictly inside the both-ends bound, and is narrower.
        control, candidate = self._side(1), self._side(1)
        share = judge_share(control, candidate, ["a", "b", "c"], [-1.0, 0.0, 1.0])
        assert share is not None
        both_low = control.variance.low + candidate.variance.low
        both_high = control.variance.high + candidate.variance.high
        assert both_low < share.low and share.high < both_high
        assert (share.high - share.low) < 0.8 * (both_high - both_low)

    def test_each_side_s_variance_is_over_its_own_k(self) -> None:
        share = judge_share(self._side(1), self._side(5), ["a", "b", "c"], [-1.0, 0.0, 1.0])
        assert share is not None and share.point == pytest.approx(0.25 + 0.05)

    def test_none_where_the_differences_have_no_spread(self) -> None:
        assert judge_share(self._side(1), self._side(1), ["a", "b"], [1.0, 1.0]) is None


# --- end to end, through the bundle --------------------------------------------------------------


def _with_repeat(
    result: EvalResult, *, again: int, served: str | None = None, sample: SampleKind | None = "random"
) -> EvalResult:
    """``result`` with one repeat of the toy judged dimension, read ``again``; by ``served`` when it names another
    judge; chosen as ``sample`` says."""
    score = result.judge_score(TOYHOST_JUDGED_DIMENSION)
    assert score is not None
    config = result.judge_config_ids.get(TOYHOST_JUDGED_DIMENSION)
    repeat = JudgeRepeat(
        judge_model="toy-judge",
        sample=sample,
        judge_config_ids={} if config is None else {TOYHOST_JUDGED_DIMENSION: config},
        scores=[
            RepeatedScore(
                dim=TOYHOST_JUDGED_DIMENSION,
                scale=score.scale,
                first_score=score.score,
                first_served_model=score.served_model,
                first_judge_config_id=config,
                first_judge_temperature=score.judge_temperature,
                repeat=RubricScore(
                    dim=TOYHOST_JUDGED_DIMENSION,
                    scale=score.scale,
                    score=again,
                    served_model=served if served is not None else score.served_model,
                    axis="capability",
                    judge_temperature=score.judge_temperature,
                ),
            )
        ],
    )
    return result.model_copy(update={"judge_repeats": [repeat]})


def _rescored(result: EvalResult, score: int) -> EvalResult:
    """``result`` with its judged score on the toy dimension set to ``score``."""
    scores = [
        entry.model_copy(update={"score": score}) if entry.dim == TOYHOST_JUDGED_DIMENSION else entry
        for entry in result.rubric_scores
    ]
    return result.model_copy(update={"rubric_scores": scores})


def _bundle(
    *, repeated: bool = True, served: str | None = None, control: bool = True, sample: SampleKind | None = "random"
) -> AnalysisContextBundle:
    """The toy campaign with its first batch as the control, its scores varied by case on the control's side (so the
    per-case differences have spread), and, when ``repeated``, every result re-read: one in three moved by one step."""
    campaign, toy = toyhost_campaign()
    runs = toy.load_eval_runs(campaign.run_ids, TOYHOST_SCOPE)
    results_by_run: dict[str, list[EvalResult]] = {}
    for position, run in enumerate(runs):
        held = []
        for index, result in enumerate(toy.query_eval_results_by_run(run.id, TOYHOST_SCOPE)):
            if position == 0:
                result = _rescored(result, 2 + sum(map(ord, result.test_case_id)) % 3)
            if repeated:
                score = result.judge_score(TOYHOST_JUDGED_DIMENSION)
                assert score is not None
                moved = score.score + (1 if score.score < 5 else -1) if index % 3 == 0 else score.score
                result = _with_repeat(result, again=moved, served=served, sample=sample)
            held.append(result)
        results_by_run[run.id] = held
    storage = ToyhostStorage(runs, results_by_run)
    first = assemble_context_bundle(campaign, storage=storage, profile=toyhost_profile())
    if not control:
        return first
    (variant,) = {
        arm.variant_key for measure in first.judged_measures for arm in measure.arms if runs[0].id in arm.run_ids
    }
    declared = campaign.model_copy(
        update={
            "arms": [
                CampaignArm(
                    label="control",
                    role="control",
                    variant_key=variant,
                    identity_version=IDENTITY_VERSION,
                    named_at="2026-03-14T09:00:00+00:00",
                )
            ]
        }
    )
    return assemble_context_bundle(declared, storage=storage, profile=toyhost_profile())


def _paths(node: Any, key: str, path: str = "") -> list[str]:
    """The path to every object of a JSON document holding ``key`` set."""
    if isinstance(node, dict):
        here = [path] if node.get(key) is not None else []
        return here + [found for name, value in node.items() for found in _paths(value, key, f"{path}/{name}")]
    if isinstance(node, list):
        return [found for index, value in enumerate(node) for found in _paths(value, key, f"{path}[{index}]")]
    return []


class TestTheBundleFillsTheJudgesTable:
    def test_each_judged_arm_and_the_paired_judged_contrast_carry_their_figures(self) -> None:
        bundle = _bundle()
        (measure,) = bundle.judged_measures
        for arm in measure.arms:
            assert arm.rescore is not None and arm.rescore.state is None
            assert arm.rescore.variance is not None and arm.rescore.variance > 0.0
            assert arm.rescore.repeated_results == arm.n
        (judged,) = [
            comparison
            for family in bundle.multiple_comparisons.families
            for comparison in family.comparisons
            if comparison.reading == "judged"
        ]
        assert judged.test == "paired" and judged.rescore_share is not None
        low, high = judged.rescore_share
        assert 0.0 <= low <= high

    def test_the_figures_are_in_the_judge_s_table_only(self) -> None:
        bundle = _bundle()
        document = bundle.to_dict()
        rescored = _paths(document, "rescore")
        assert rescored and all(path.startswith("/judged_measures[0]/arms[") for path in rescored), rescored
        shared = _paths(document, "rescore_share")
        assert shared and all(path.startswith("/multiple_comparisons/families[") for path in shared), shared
        # The report input may carry them in the judge's own table and nowhere else: no caveat, no first screen.
        built = report_input(campaign_bundle(bundle)).model_dump(mode="json", by_alias=True)
        for key in ("rescore", "rescore_share", "fail_given_pass", "ceiling_low", "ceiling_high"):
            assert all(path.startswith("/judged/") for path in _paths(built, key)), key

    def test_an_arm_nobody_repeated_carries_no_figure(self) -> None:
        bundle = _bundle(repeated=False)
        assert all(arm.rescore is None for measure in bundle.judged_measures for arm in measure.arms)
        assert not _paths(bundle.to_dict(), "rescore_share")

    def test_a_repeat_by_another_judge_is_not_read(self) -> None:
        bundle = _bundle(served="another-judge")
        for measure in bundle.judged_measures:
            for arm in measure.arms:
                assert arm.rescore is not None
                assert (arm.rescore.state, arm.rescore.repeated_results) == (_TOO_FEW, 0)
        assert not _paths(bundle.to_dict(), "rescore_share")

    @pytest.mark.parametrize(("sample", "reason"), [("selected", "not_drawn_at_random"), (None, "not_recorded")])
    def test_repeats_not_drawn_at_random_are_not_read_and_say_why(self, sample: SampleKind | None, reason: str) -> None:
        # Named by id, or stored before the draw was recorded: they may not stand for the arm, so no figure, no share.
        bundle = _bundle(sample=sample)
        for measure in bundle.judged_measures:
            for arm in measure.arms:
                assert arm.rescore is not None
                assert arm.rescore.state == StateOf(state="not_available", reason=reason)
                assert arm.rescore.repeated_results == arm.n and arm.rescore.variance is None
        assert not _paths(bundle.to_dict(), "rescore_share")
