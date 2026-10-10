"""A classifier's per-label precision, recall and F1, and their intervals, checked against a known truth (#601).

:func:`~threetears.evals.analysis.confusion.label_statistics` counts each label's precision and recall from
a classifier's observations, each tagged with its case, and bounds each with an interval labelled 95%; the
analysis bundle and the run summary both read them from there. With one answer per case that is the Wilson
interval; where repeats of a case recur, it is the cluster-aware one
(:func:`~threetears.evals.analysis.stats.proportion_interval`, #590), which counts cases rather than repeats.

Checked here:

- with one answer per case, the Wilson interval's coverage averaged over the true rate is its nominal 95%,
  computed exactly from the binomial (no simulation error at all);
- with repeats of a case that tend to agree (a classifier that finds a case hard finds it hard every
  time), coverage of the label's true recall and precision at the bank sizes a classifier eval runs
  (3–5 cases a label, 9–15 cases in all, k = 1–5);
- precision, recall and F1 converge on the truth.
"""

from __future__ import annotations

import math
import random

import pytest

from threetears.evals.analysis.confusion import LabelStatistics, label_statistics
from threetears.evals.analysis.stats import INTERVAL_LEVEL
from threetears.evals.kernel.metrics import confusion_cell
from packages.evals.tests.simulation_support import at_least, draw_confusion

_LABELS = ("negative", "neutral", "positive")


def _one_label(observations: list[tuple[str, str]], label: str = "negative") -> LabelStatistics:
    (statistics,) = [entry for entry in label_statistics(observations) if entry.label == label]
    return statistics


class TestOneAnswerPerCase:
    """With independent answers the Wilson interval's coverage, averaged over the true rate, is its nominal."""

    @pytest.mark.parametrize("n", [2, 3, 5, 8, 10, 15, 20])
    def test_mean_coverage_is_nominal_exactly(self, n: int) -> None:
        """``x`` correct of ``n`` answers is binomial, so coverage at a true rate ``p`` is the binomial mass of
        every ``x`` whose interval contains ``p`` — exact, no Monte-Carlo error. A discrete interval's
        coverage oscillates with ``p``, so the criterion is the mean over ``p`` in 0.01..0.99 (Brown, Cai &
        DasGupta 2001), held to within one point of 95%.

        One set of observations serves both statistics: ``x`` answers right and ``n - x`` swapped each way,
        so the label's recall and its precision are both ``x / n``. Every answer is its own case.
        """
        intervals = {}
        for x in range(n + 1):
            pairs = [("negative", "negative")] * x + [("negative", "positive"), ("positive", "negative")] * (n - x)
            observations = [(confusion_cell(*pair), f"case-{index}") for index, pair in enumerate(pairs)]
            statistics = _one_label(observations)
            assert statistics.recall_interval == statistics.precision_interval
            assert statistics.recall_interval is not None
            intervals[x] = statistics.recall_interval
        rates = [index / 100 for index in range(1, 100)]
        coverage = [
            sum(math.comb(n, x) * p**x * (1 - p) ** (n - x) for x, (low, high) in intervals.items() if low <= p <= high)
            for p in rates
        ]
        mean = sum(coverage) / len(coverage)
        assert abs(mean - INTERVAL_LEVEL) <= 0.01, f"n={n}: mean coverage {mean:.4f} against {INTERVAL_LEVEL}"


def _coverage(
    seed: str, *, cases_per_label: int, repeats: int, intra_case_correlation: float, replicates: int
) -> tuple[float, float]:
    """The share of replicates whose recall and precision intervals contain the label's true value.

    Each replicate draws its own true accuracy uniformly from [0.6, 0.95], so the coverage measured is the
    mean over the rates a working classifier has rather than one point of the interval's oscillation, and
    reads one label, so replicates are independent. A label the classifier never gave in a replicate has no
    precision and no precision interval; precision's coverage is over the replicates that state one.
    """
    rng = random.Random(seed)
    recall_hits = precision_hits = precision_stated = 0
    for _ in range(replicates):
        accuracy = rng.uniform(0.6, 0.95)
        statistics = _one_label(
            draw_confusion(
                rng,
                _LABELS,
                cases_per_label=cases_per_label,
                repeats=repeats,
                accuracy=accuracy,
                intra_case_correlation=intra_case_correlation,
            )
        )
        assert statistics.recall_interval is not None, "every label is expected by cases_per_label cases"
        recall_hits += statistics.recall_interval[0] <= accuracy <= statistics.recall_interval[1]
        if statistics.precision_interval is not None:
            precision_stated += 1
            precision_hits += statistics.precision_interval[0] <= accuracy <= statistics.precision_interval[1]
    assert precision_stated >= 0.99 * replicates, "precision must be stated nearly always for its rate to mean much"
    return recall_hits / replicates, precision_hits / precision_stated


#: 3,000 replicates: SE at 95% coverage is sqrt(0.95 * 0.05 / 3000) = 0.0040, so the bound is 0.934 and a
#: 92% interval falls outside it.
_REPLICATES = 3000


class TestRepeatsOfACase:
    """Repeats of one case are correlated draws, not independent trials, whenever cases differ in difficulty."""

    @pytest.mark.parametrize(
        ("cases_per_label", "repeats", "correlation"),
        [(3, 1, 0.5), (5, 1, 0.5)],
    )
    def test_one_answer_a_case_covers_at_nominal(self, cases_per_label: int, repeats: int, correlation: float) -> None:
        """With k = 1 the answers are independent whatever the cases' difficulty, so both intervals cover."""
        recall, precision = _coverage(
            f"classifier-k1-{cases_per_label}",
            cases_per_label=cases_per_label,
            repeats=repeats,
            intra_case_correlation=correlation,
            replicates=_REPLICATES,
        )
        assert recall >= at_least(INTERVAL_LEVEL, _REPLICATES), f"recall coverage {recall:.4f}"
        assert precision >= at_least(INTERVAL_LEVEL, _REPLICATES), f"precision coverage {precision:.4f}"

    @pytest.mark.parametrize(("cases_per_label", "repeats"), [(3, 3), (5, 3)])
    def test_precision_over_repeats_covers_at_nominal(self, cases_per_label: int, repeats: int) -> None:
        """Precision's denominator gathers answers from every label's cases, which dilutes the clustering;
        at k = 3 its interval still covers."""
        _, precision = _coverage(
            f"classifier-precision-{cases_per_label}-{repeats}",
            cases_per_label=cases_per_label,
            repeats=repeats,
            intra_case_correlation=0.5,
            replicates=_REPLICATES,
        )
        assert precision >= at_least(INTERVAL_LEVEL, _REPLICATES), f"precision coverage {precision:.4f}"

    @pytest.mark.parametrize(
        ("cases_per_label", "repeats", "correlation"),
        [(3, 3, 0.5), (5, 3, 0.5), (5, 5, 0.5), (5, 3, 0.2)],
    )
    def test_recall_over_repeats_covers_at_nominal(
        self, cases_per_label: int, repeats: int, correlation: float
    ) -> None:
        """Recall's answers all come from the label's own few cases, so clustering bites hardest here. Counting
        every repeat as a trial covered 0.76-0.91 (#601); the cluster-aware interval (#590) covers 0.99."""
        recall, _ = _coverage(
            f"classifier-recall-{cases_per_label}-{repeats}-{correlation}",
            cases_per_label=cases_per_label,
            repeats=repeats,
            intra_case_correlation=correlation,
            replicates=_REPLICATES,
        )
        assert recall >= at_least(INTERVAL_LEVEL, _REPLICATES), f"recall coverage {recall:.4f}"

    def test_precision_over_many_repeats_covers_at_nominal(self) -> None:
        """At k = 5 counting repeats as trials covered 0.90 (#601); the cluster-aware interval covers 0.996."""
        _, precision = _coverage(
            "classifier-precision-5-5",
            cases_per_label=5,
            repeats=5,
            intra_case_correlation=0.5,
            replicates=_REPLICATES,
        )
        assert precision >= at_least(INTERVAL_LEVEL, _REPLICATES), f"precision coverage {precision:.4f}"


def test_precision_recall_and_f1_converge_on_the_truth() -> None:
    """Over 4,000 cases a label the three statistics sit on the true 0.8 (each SE about 0.006: 4 SE is 0.025)."""
    rng = random.Random("classifier-consistent")
    observations = draw_confusion(
        rng, _LABELS, cases_per_label=4000, repeats=1, accuracy=0.8, intra_case_correlation=0.3
    )
    for statistics in label_statistics(observations):
        assert statistics.precision is not None and statistics.recall is not None and statistics.f1 is not None
        for value in (statistics.precision, statistics.recall, statistics.f1):
            assert abs(value - 0.8) <= 0.025, f"{statistics.label}: {value:.4f} against 0.8"
