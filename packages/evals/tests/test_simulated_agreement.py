"""Judge agreement — Cohen's kappa and its pooling across raters — checked against a known truth (#601).

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

from threetears.evals.analysis import judge_agreement
from threetears.evals.analysis.stats import cohen_kappa
from threetears.evals.contracts import CalibrationRating, EvalResult, RubricScore
from threetears.evals.contracts.evidence_tiers import CALIBRATION_MIN_RESULTS
from packages.evals.tests.factories import make_calibration_rating, make_eval_result
from packages.evals.tests.simulation_support import TOLERANCE_Z, draw_rater_pairs

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
