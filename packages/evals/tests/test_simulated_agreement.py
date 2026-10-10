"""Judge agreement — Cohen's kappa, its pooling across raters, and the tier it decides — checked against a known truth (#601).

A judge's evidence tier is decided by kappa: quadratic-weighted on a 1-5 dimension, unweighted on
pass/fail, per rater and pooled by result (:mod:`threetears.evals.analysis.agreement`). The tier compares the
figure with a bar (0.6 for calibration, 0.8 for separation) over at least 20 results, so the figure has to
estimate the population agreement it stands for.

The generator (:func:`~packages.evals.tests.simulation_support.draw_rater_pairs`) has a known population
kappa under ANY disagreement cost — unweighted, quadratic, and with an off-scale "can't tell" category — so
one truth serves every variant the engine computes.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

import pytest

from threetears.evals.analysis import JudgeKey, judge_agreement, judge_evidence_tiers, judge_self_agreement
from threetears.evals.analysis.agreement import agreement_interval
from threetears.evals.analysis.stats import cohen_kappa, kappa_moments
from threetears.evals.schema import CalibrationRating, EvalResult, RubricScore
from threetears.evals.kernel.evidence_tiers import (
    CALIBRATION_MIN_AGREEMENT,
    CALIBRATION_MIN_RESULTS,
    SEPARATION_MIN_AGREEMENT,
    SEPARATION_MIN_RESULTS,
    TierCriterion,
    calibration_criterion,
    separation_criterion,
)
from packages.evals.tests.factories import make_calibration_rating, make_eval_result
from packages.evals.tests.simulation_support import TOLERANCE_Z, at_least, at_most, draw_rater_pairs

_SCALE = [1, 2, 3, 4, 5]
_CANNOT_TELL = -1

#: (label, categories, marginal, weights, unordered, agreement): the variants the engine computes.
_VARIANTS = [
    ("unweighted 1-5, flat marginal", _SCALE, [1, 1, 1, 1, 1], "none", (), 0.6),
    ("quadratic 1-5, peaked marginal", _SCALE, [1, 2, 4, 2, 1], "quadratic", (), 0.6),
    ("pass/fail, 3:1 marginal", [0, 1], [3, 1], "none", (), 0.8),
    (
        "quadratic 1-5 with a can't-tell answer",
        [*_SCALE, _CANNOT_TELL],
        [1, 2, 4, 2, 1, 1],
        "quadratic",
        (_CANNOT_TELL,),
        0.6,
    ),
]


def _kappas(
    seed: str,
    n: int,
    categories: Sequence[int],
    marginal: Sequence[float],
    weights: str,
    unordered: Sequence[int],
    agreement: float,
    replicates: int,
) -> tuple[float, float]:
    """The mean and Monte-Carlo SE of kappa over ``replicates`` draws of ``n`` pairs (undefined draws skipped)."""
    rng = random.Random(seed)
    scale = [category for category in categories if category not in unordered]
    estimates = []
    for _ in range(replicates):
        pairs = draw_rater_pairs(rng, n, categories, marginal=marginal, agreement=agreement)
        kappa = cohen_kappa(
            pairs, scale, weights="quadratic" if weights == "quadratic" else "none", unordered=unordered
        )
        if kappa is not None:
            estimates.append(kappa)
    mean = sum(estimates) / len(estimates)
    spread = math.sqrt(sum((value - mean) ** 2 for value in estimates) / (len(estimates) - 1))
    return mean, spread / math.sqrt(len(estimates))


class TestCohensKappa:
    @pytest.mark.parametrize(
        ("label", "categories", "marginal", "weights", "unordered", "agreement"),
        _VARIANTS,
        ids=[v[0] for v in _VARIANTS],
    )
    def test_it_converges_on_the_population_kappa(
        self,
        label: str,
        categories: Sequence[int],
        marginal: Sequence[float],
        weights: str,
        unordered: Sequence[int],
        agreement: float,
    ) -> None:
        """Over 1,000 items, the mean of 100 kappas sits on the truth within 4 Monte-Carlo SEs (each ~0.002;
        kappa's O(1/n) bias is ~0.0005 here, inside that)."""
        mean, se = _kappas(f"kappa-consistent-{label}", 1000, categories, marginal, weights, unordered, agreement, 100)
        assert abs(mean - agreement) <= TOLERANCE_Z * se + 0.001, f"{label}: mean kappa {mean:.4f} against {agreement}"

    @pytest.mark.parametrize(
        ("label", "categories", "marginal", "weights", "unordered", "agreement"),
        _VARIANTS,
        ids=[v[0] for v in _VARIANTS],
    )
    def test_at_the_calibration_floor_its_bias_is_small_beside_the_bar(
        self,
        label: str,
        categories: Sequence[int],
        marginal: Sequence[float],
        weights: str,
        unordered: Sequence[int],
        agreement: float,
    ) -> None:
        """At the tiers' floor of 20 results kappa's small-sample bias (negative, O(1/n)) stays under 0.03 —
        a twentieth of the 0.6 bar. 2,000 replicates put the mean's SE near 0.003."""
        mean, se = _kappas(
            f"kappa-floor-{label}", CALIBRATION_MIN_RESULTS, categories, marginal, weights, unordered, agreement, 2000
        )
        assert abs(mean - agreement) <= 0.03 + TOLERANCE_Z * se, f"{label}: mean kappa {mean:.4f} against {agreement}"


_JUDGE = "judge-model"
_DIM = "conversation.tone"


def _calibration(
    rng: random.Random, n_results: int, raters: Sequence[str], agreement: float
) -> tuple[list[CalibrationRating], list[EvalResult]]:
    """``n_results`` judged results and people's ratings of them, each person agreeing with the judge at ``agreement``.

    Each result is rated by one or two of ``raters`` (the second with probability one half), so the pooling
    by result is exercised: a result two people rated weighs 1, split between them.
    """
    marginal = [1, 2, 4, 2, 1]
    results, ratings = [], []
    for index in range(n_results):
        judge_score = rng.choices(_SCALE, weights=marginal)[0]
        result = make_eval_result(
            id=f"result-{index}",
            rubric_scores=[RubricScore(dim=_DIM, score=judge_score, scale="ordinal", served_model=_JUDGE)],
        )
        results.append(result)
        who = rng.sample(list(raters), 2 if rng.random() < 0.5 else 1)
        for rater in who:
            score = judge_score if rng.random() < agreement else rng.choices(_SCALE, weights=marginal)[0]
            ratings.append(make_calibration_rating(result_id=result.id, rater=rater, score=score, rubric_dim=_DIM))
    return ratings, results


def test_the_pooled_agreement_with_people_estimates_the_population_kappa() -> None:
    """Three people rate 60 results between them, each agreeing with the judge at a weighted kappa of 0.6.

    The pooled figure :func:`~threetears.evals.analysis.judge_agreement` publishes — each person's quadratic
    kappa, weighted by the results they measured — has mean within 0.03 of 0.6 (each person's kappa rests on
    about 30 results, where kappa's small-sample bias is ~0.015), plus 4 Monte-Carlo SEs over 150 replicates.
    """
    rng = random.Random("judge-agreement-pooled")
    estimates = []
    for _ in range(150):
        ratings, results = _calibration(rng, 60, ("ana", "ben", "cai"), 0.6)
        (dimension,) = judge_agreement(ratings, results).dimensions
        assert dimension.weighted_kappa is not None
        estimates.append(dimension.weighted_kappa)
    mean = sum(estimates) / len(estimates)
    se = math.sqrt(sum((value - mean) ** 2 for value in estimates) / (len(estimates) - 1)) / math.sqrt(len(estimates))
    assert abs(mean - 0.6) <= 0.03 + TOLERANCE_Z * se, f"pooled weighted kappa {mean:.4f} against 0.6"


# ---------------------------------------------------------------------------------------------------
# The tier: decided on the agreement's interval, never its point estimate
# ---------------------------------------------------------------------------------------------------

#: The variants a tier is decided over: 1-5 at three marginals, pass/fail at two, and a repeat that can decline.
_TIER_VARIANTS = [
    ("quadratic 1-5, flat marginal", _SCALE, [1, 1, 1, 1, 1], "quadratic", ()),
    ("quadratic 1-5, peaked marginal", _SCALE, [1, 2, 4, 2, 1], "quadratic", ()),
    ("quadratic 1-5, skewed marginal", _SCALE, [1, 1, 2, 6, 10], "quadratic", ()),
    ("pass/fail, 1:1 marginal", [0, 1], [1, 1], "none", ()),
    ("pass/fail, 3:1 marginal", [0, 1], [3, 1], "none", ()),
    (
        "quadratic 1-5 with a can't-tell answer",
        [*_SCALE, _CANNOT_TELL],
        [1, 2, 4, 2, 1, 1],
        "quadratic",
        (_CANNOT_TELL,),
    ),
]

#: The two criteria, each with its bar and its floor: calibration against people, separation against the judge's
#: own repeats. Each is checked at its floor, the fewest results it decides on.
_CRITERIA = [
    ("calibration", calibration_criterion, CALIBRATION_MIN_AGREEMENT, CALIBRATION_MIN_RESULTS, 1000),
    ("separation", separation_criterion, SEPARATION_MIN_AGREEMENT, SEPARATION_MIN_RESULTS, 400),
]

#: The one-sided error a criterion's bounds admit: how often a judge AT the bar may be shown on either side of it.
_ONE_SIDED = 0.05


def _criterion_over(
    pairs: Sequence[tuple[int, int]], categories: Sequence[int], weights: str, unordered: Sequence[int], build
) -> tuple[TierCriterion, float] | None:
    """The criterion the engine decides over one rater's ``pairs`` (one per result), beside the point estimate."""
    scale = [category for category in categories if category not in unordered]
    cost = "quadratic" if weights == "quadratic" else "none"
    kappa = cohen_kappa(pairs, scale, weights=cost, unordered=unordered)
    moments = kappa_moments(pairs, scale, weights=cost, unordered=unordered)
    if kappa is None or moments is None:
        return None
    interval = agreement_interval(kappa, [(moments, [str(index) for index in range(len(pairs))])])
    return build(len(pairs), len(pairs), kappa, interval), kappa


def _tier_rates(
    seed: str,
    n: int,
    truth: float,
    variant: tuple[str, Sequence[int], Sequence[float], str, Sequence[int]],
    build,
    bar: float,
    replicates: int,
) -> tuple[float, float, float]:
    """Over ``replicates`` draws of ``n`` results at population kappa ``truth``: how often the criterion was met,
    how often it was shown not met, and how often the point estimate alone reached the bar."""
    _, categories, marginal, weights, unordered = variant
    rng = random.Random(seed)
    met = not_met = point = 0
    for _ in range(replicates):
        pairs = draw_rater_pairs(rng, n, categories, marginal=marginal, agreement=truth)
        read = _criterion_over(pairs, categories, weights, unordered, build)
        if read is None:
            continue
        criterion, kappa = read
        met += criterion.state == "met"
        not_met += criterion.state == "not_met"
        point += kappa >= bar
    return met / replicates, not_met / replicates, point / replicates


#: The marginals the power checks hold: every one but the heavily skewed 1-5, whose power is stated, not held.
_POWER_VARIANTS = [_TIER_VARIANTS[0], _TIER_VARIANTS[1], _TIER_VARIANTS[3], _TIER_VARIANTS[4]]


class TestTheTierIsDecidedOnConfidenceBounds:
    """A tier is a claim about the judge, so a judge AT the bar may earn it no more often than the one-sided 5% its
    lower bound admits, on every marginal, for both criteria at their floors. On the point estimate a judge at the
    bar earned it about half the time, and one 0.1 below it about a third of the time."""

    @pytest.mark.parametrize(
        ("criterion", "build", "bar", "floor", "replicates"), _CRITERIA, ids=[c[0] for c in _CRITERIA]
    )
    @pytest.mark.parametrize("variant", _TIER_VARIANTS, ids=[v[0] for v in _TIER_VARIANTS])
    def test_a_judge_at_the_bar_is_shown_on_neither_side_of_it_beyond_the_bounds_error(
        self,
        variant: tuple[str, Sequence[int], Sequence[float], str, Sequence[int]],
        criterion: str,
        build,
        bar: float,
        floor: int,
        replicates: int,
    ) -> None:
        """At the floor (20 results for calibration, 120 for separation), population kappa equal to the bar.

        Met (the tier awarded) at most 5% — measured (seeded) 0.2-1.9% for calibration at 20 results and 2.5-4.5%
        for separation at 120, where the point estimate reached the bar 48-55% of the time. Not met (what
        ``incidental`` rests on), decided on the upper bound's stricter 97.5%: measured 1.4-3.7%.
        """
        met, not_met, point = _tier_rates(
            f"tier-at-bar-{criterion}-{variant[0]}", floor, bar, variant, build, bar, replicates
        )
        assert met <= at_most(_ONE_SIDED, replicates), f"{variant[0]}: {criterion} met {met:.3f} at the bar"
        assert not_met <= at_most(_ONE_SIDED, replicates), f"{variant[0]}: {criterion} not met {not_met:.3f} at the bar"
        assert point >= at_least(0.4, replicates), "the rule this replaces: the point estimate clears the bar often"

    @pytest.mark.parametrize("variant", _POWER_VARIANTS, ids=lambda v: v[0])
    def test_a_judge_a_tenth_below_the_bar_no_longer_calibrates_a_third_of_the_time(
        self, variant: tuple[str, Sequence[int], Sequence[float], str, Sequence[int]]
    ) -> None:
        """True weighted kappa 0.5 against the 0.6 bar, at the floor: the point estimate awarded ``calibrated``
        about a third of the time; the bound 0-0.6% (measured)."""
        replicates = 1000
        met, _, point = _tier_rates(
            f"tier-below-{variant[0]}", CALIBRATION_MIN_RESULTS, 0.5, variant, calibration_criterion, 0.6, replicates
        )
        assert point >= at_least(0.25, replicates)
        assert met <= at_most(_ONE_SIDED, replicates)

    @pytest.mark.parametrize("variant", _POWER_VARIANTS, ids=lambda v: v[0])
    def test_with_sixty_results_a_judge_well_above_the_bar_calibrates_most_of_the_time(
        self, variant: tuple[str, Sequence[int], Sequence[float], str, Sequence[int]]
    ) -> None:
        """Power, stated rather than hidden: a judge at true kappa 0.9 calibrates 14-45% of the time at the
        20-result floor and 71-98% at 60 (measured; 82-98% on these marginals), held here at 70%. About 50 results
        give 80% on most marginals."""
        replicates = 400
        met, _, _ = _tier_rates(f"tier-power-{variant[0]}", 60, 0.9, variant, calibration_criterion, 0.6, replicates)
        assert met >= at_least(0.7, replicates), f"{variant[0]}: calibrated {met:.3f} at true 0.9 over 60 results"

    @pytest.mark.parametrize("variant", _POWER_VARIANTS, ids=lambda v: v[0])
    def test_at_its_floor_a_near_perfect_judge_separates_most_of_the_time(
        self, variant: tuple[str, Sequence[int], Sequence[float], str, Sequence[int]]
    ) -> None:
        """Why separation's floor is 120: there a judge at true self-agreement 0.95 earns it 81-98% of the time on
        these marginals (measured; 70% on a heavily skewed 1-5). At 20 results it could never be earned. Held at
        70%."""
        replicates = 300
        met, _, _ = _tier_rates(
            f"tier-separation-power-{variant[0]}",
            SEPARATION_MIN_RESULTS,
            0.95,
            variant,
            separation_criterion,
            SEPARATION_MIN_AGREEMENT,
            replicates,
        )
        assert met >= at_least(0.7, replicates), f"{variant[0]}: separation {met:.3f} at true 0.95 over the floor"


def _correlated_calibration(
    rng: random.Random, n_results: int, agreement: float
) -> tuple[list[CalibrationRating], list[EvalResult]]:
    """Results rated by one or two people who BOTH give the true score: their disagreements with the judge coincide.

    The judge gives the true score with probability ``agreement`` and a chance score otherwise, so each person's
    kappa with the judge is ``agreement``, and two people rating one result disagree with it together.
    """
    marginal = [1, 2, 4, 2, 1]
    results, ratings = [], []
    for index in range(n_results):
        truth = rng.choices(_SCALE, weights=marginal)[0]
        judge = truth if rng.random() < agreement else rng.choices(_SCALE, weights=marginal)[0]
        result = make_eval_result(
            id=f"result-{index}",
            rubric_scores=[RubricScore(dim=_DIM, score=judge, scale="ordinal", served_model=_JUDGE)],
        )
        results.append(result)
        for rater in rng.sample(["ana", "ben", "cai"], 2 if rng.random() < 0.5 else 1):
            ratings.append(make_calibration_rating(result_id=result.id, rater=rater, score=truth, rubric_dim=_DIM))
    return ratings, results


@pytest.mark.parametrize(
    ("label", "draw"),
    [
        ("independent people", lambda rng: _calibration(rng, 30, ("ana", "ben", "cai"), CALIBRATION_MIN_AGREEMENT)),
        ("people who agree with each other", lambda rng: _correlated_calibration(rng, 30, CALIBRATION_MIN_AGREEMENT)),
    ],
)
def test_the_pooled_tier_from_several_people_holds_its_error_at_the_bar(label: str, draw) -> None:
    """End to end through :func:`judge_agreement` and :func:`judge_evidence_tiers`: three people share 30 results
    (a result rated by one or two of them), each at weighted kappa 0.6 — the bar. The result is the cluster: two
    people's disagreements with the judge on one result are added at the correlation they show. When they agree with
    each other perfectly, adding them as independent awarded the tier 9.5% of the time; measured here 0.5-0.8%."""
    rng = random.Random(f"pooled-tier-at-bar-{label}")
    replicates = 400
    met = 0
    key = JudgeKey(_DIM, "ordinal", _JUDGE, None)
    for _ in range(replicates):
        ratings, results = draw(rng)
        (tier,) = judge_evidence_tiers(judge_agreement(ratings, results), judge_self_agreement([]), {key})
        met += tier.calibration.state == "met"
    assert met / replicates <= at_most(_ONE_SIDED, replicates), f"{label}: met {met / replicates:.3f}"
