"""Prediction-powered inference — a biased judge's mean corrected by a few human labels — checked against a known truth (#598).

A judge scores every observation and people label a few. The judge-only interval is centred on the judge's mean, so
a judge biased by a constant puts it around the wrong value however many observations it scores. The
prediction-powered estimate (:func:`~threetears.evals.analysis.stats.prediction_powered_mean`) adds the rectifier,
the mean of ``human - judge`` over the labelled observations, and its interval has to cover the mean people would
have given at its nominal level — with repeats of a case correlated, as an eval's are.

The generator draws a clustered arm (:func:`~packages.evals.tests.simulation_support.draw_clustered`) as the
people's scores, whose expectation is the truth, and a judge that adds a known constant bias and its own noise.
"""

from __future__ import annotations

import random

from threetears.evals.analysis.stats import (
    INTERVAL_LEVEL,
    observed_mean_interval,
    prediction_powered_mean,
)
from packages.evals.tests.simulation_support import ClusteredDesign, at_least, draw_clustered

#: 2,000 replicates: the Monte-Carlo SE at 95% coverage is sqrt(0.95 * 0.05 / 2000) = 0.0049, so the coverage bound
#: is 0.930. Each replicate is 30 cases x 3 repeats with 24 labels, which keeps the file to a few seconds.
_REPLICATES = 2_000
_DESIGN = ClusteredDesign(n_cases=30, repeats=3, between_case_sd=0.8, repeat_sd=0.4)
_LABELLED = 24
_TRUTH = 3.0
#: Large beside the judge-only interval's half-width (about 0.3 here), so that interval misses nearly always.
_BIAS = 0.8
_JUDGE_NOISE_SD = 0.3


def _replicate(rng: random.Random) -> tuple[list[float], list[int], list[float | None]]:
    """One arm: the judge's scores, each observation's case, and people's scores on a random ``_LABELLED`` of them."""
    people = draw_clustered(rng, _DESIGN, mean=_TRUTH)
    flat = [(case, score) for case, scores in enumerate(people) for score in scores]
    judge = [score + _BIAS + rng.gauss(0.0, _JUDGE_NOISE_SD) for _, score in flat]
    labelled = set(rng.sample(range(len(flat)), _LABELLED))
    human = [score if index in labelled else None for index, (_, score) in enumerate(flat)]
    return judge, [case for case, _ in flat], human


def _coverage(seed: str) -> tuple[float, float, float, float]:
    """PPI coverage, judge-only coverage, and the mean widths of the PPI and labelled-only intervals."""
    rng = random.Random(seed)
    ppi_hits = judge_hits = 0
    ppi_width = labelled_width = 0.0
    for _ in range(_REPLICATES):
        judge, cases, human = _replicate(rng)
        estimate = prediction_powered_mean(judge, cases, human)
        assert estimate is not None and estimate.interval is not None
        low, high = estimate.interval
        ppi_hits += low <= _TRUTH <= high
        ppi_width += high - low
        judge_only = observed_mean_interval(judge, cases=cases)
        assert judge_only is not None
        judge_hits += judge_only[0] <= _TRUTH <= judge_only[1]
        rated = [(value, case) for value, case in zip(human, cases) if value is not None]
        labelled_only = observed_mean_interval([value for value, _ in rated], cases=[case for _, case in rated])
        assert labelled_only is not None
        labelled_width += labelled_only[1] - labelled_only[0]
    return (
        ppi_hits / _REPLICATES,
        judge_hits / _REPLICATES,
        ppi_width / _REPLICATES,
        labelled_width / _REPLICATES,
    )


class TestABiasedJudgeCorrectedByPeople:
    def test_the_ppi_interval_covers_the_truth_at_nominal_where_the_judge_only_interval_does_not(self) -> None:
        ppi, judge_only, _, _ = _coverage("ppi-coverage")
        assert ppi >= at_least(INTERVAL_LEVEL, _REPLICATES), f"PPI coverage {ppi:.4f}"
        assert judge_only <= 0.05, f"judge-only coverage {judge_only:.4f}: the bias must defeat it"

    def test_the_judge_s_scores_buy_width_over_the_labels_alone(self) -> None:
        # The judge tracks people (its noise is below their spread), so leaning on its scores for the bulk of the
        # mean narrows the interval below what the 24 labels give on their own — PPI's reason to exist.
        _, _, ppi_width, labelled_width = _coverage("ppi-width")
        assert ppi_width < labelled_width, f"PPI width {ppi_width:.4f} against labelled-only {labelled_width:.4f}"
