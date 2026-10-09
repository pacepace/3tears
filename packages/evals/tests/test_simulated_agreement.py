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
from threetears.evals.contracts import CalibrationRating, EvalResult, RubricScore
from threetears.evals.contracts.evidence_tiers import (
    CALIBRATION_MIN_AGREEMENT,
    CALIBRATION_MIN_RESULTS,
    SEPARATION_MIN_AGREEMENT,
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

#: The two criteria, each with its bar: calibration against people, separation against the judge's own repeats.
_CRITERIA = [
    ("calibration", calibration_criterion, CALIBRATION_MIN_AGREEMENT),
    ("separation", separation_criterion, SEPARATION_MIN_AGREEMENT),
]

#: The one-sided error a 95% interval's lower end admits: how often a judge AT the bar may be shown over it.
_ONE_SIDED = 0.025


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


class TestTheTierIsDecidedOnAnInterval:
    """A tier is a claim about the judge, so a judge AT the bar may earn it no more often than the interval's
    one-sided error (2.5%) — at the 20-result floor, on every marginal, for both criteria. On the point estimate
    a judge at the bar earned it about half the time, and one 0.1 below it about a third of the time."""

    @pytest.mark.parametrize(("criterion", "build", "bar"), _CRITERIA, ids=[c[0] for c in _CRITERIA])
    @pytest.mark.parametrize("variant", _TIER_VARIANTS, ids=[v[0] for v in _TIER_VARIANTS])
    def test_a_judge_at_the_bar_is_shown_on_neither_side_of_it_beyond_the_intervals_error(
        self, variant: tuple[str, Sequence[int], Sequence[float], str, Sequence[int]], criterion: str, build, bar: float
    ) -> None:
        """1,000 replicates of 20 results at a population kappa equal to the bar.

        Met (the tier awarded): at most the one-sided 2.5% — measured (seeded) at most 0.7% for calibration and 0%
        for separation, where the point estimate reached the bar 49-57% of the time. Not met (what ``incidental``
        rests on): the score interval's upper end is a little looser than its lower, measured 1.4-4.6%, so the bound
        held there is 5%.
        """
        replicates = 1000
        met, not_met, point = _tier_rates(
            f"tier-at-bar-{criterion}-{variant[0]}", CALIBRATION_MIN_RESULTS, bar, variant, build, bar, replicates
        )
        assert met <= at_most(_ONE_SIDED, replicates), f"{variant[0]}: {criterion} met {met:.3f} at the bar"
        assert not_met <= at_most(0.05, replicates), f"{variant[0]}: {criterion} not met {not_met:.3f} at the bar"
        assert point >= at_least(0.4, replicates), "the rule this replaces: the point estimate clears the bar often"

    @pytest.mark.parametrize("variant", _TIER_VARIANTS[:2] + _TIER_VARIANTS[3:5], ids=lambda v: v[0])
    def test_a_judge_a_tenth_below_the_bar_no_longer_calibrates_a_third_of_the_time(
        self, variant: tuple[str, Sequence[int], Sequence[float], str, Sequence[int]]
    ) -> None:
        """True weighted kappa 0.5 against the 0.6 bar, at the floor: the point estimate awarded ``calibrated``
        about a third of the time; the interval essentially never (measured 0-0.2%)."""
        replicates = 1000
        met, _, point = _tier_rates(
            f"tier-below-{variant[0]}", CALIBRATION_MIN_RESULTS, 0.5, variant, calibration_criterion, 0.6, replicates
        )
        assert point >= at_least(0.25, replicates)
        assert met <= at_most(_ONE_SIDED, replicates)

    @pytest.mark.parametrize("variant", _TIER_VARIANTS[:2] + _TIER_VARIANTS[3:5], ids=lambda v: v[0])
    def test_with_sixty_results_a_judge_well_above_the_bar_calibrates_most_of_the_time(
        self, variant: tuple[str, Sequence[int], Sequence[float], str, Sequence[int]]
    ) -> None:
        """The price is power, stated rather than hidden. At 20 results a judge at true kappa 0.9 calibrates at most
        a third of the time (1-5: about 1%; pass/fail: 8-33%) and separation is never shown — twenty perfect repeats
        bound kappa below 0.8. At 60 results and true 0.9 it calibrates 74-93% of the time (measured), held here
        at a floor of 60%."""
        replicates = 400
        met, _, _ = _tier_rates(f"tier-power-{variant[0]}", 60, 0.9, variant, calibration_criterion, 0.6, replicates)
        assert met >= at_least(0.6, replicates), f"{variant[0]}: calibrated {met:.3f} at true 0.9 over 60 results"


def test_the_pooled_tier_from_several_people_holds_its_error_at_the_bar() -> None:
    """End to end through :func:`judge_agreement` and :func:`judge_evidence_tiers`: three people share 20 results
    (a result rated by one or two of them), each at weighted kappa 0.6 — the bar. Raters of one result are added
    at full correlation, so the pooled interval is conservative: measured 0% met over 400 replicates."""
    rng = random.Random("pooled-tier-at-bar")
    replicates = 400
    met = 0
    key = JudgeKey(_DIM, "ordinal", _JUDGE, None)
    for _ in range(replicates):
        ratings, results = _calibration(rng, CALIBRATION_MIN_RESULTS, ("ana", "ben", "cai"), CALIBRATION_MIN_AGREEMENT)
        (tier,) = judge_evidence_tiers(judge_agreement(ratings, results), judge_self_agreement([]), {key})
        met += tier.calibration.state == "met"
    assert met / replicates <= at_most(_ONE_SIDED, replicates)
