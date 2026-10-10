"""The 95th percentiles the engine reports, checked against the population percentile they stand for (#601).

Both tails — ``p95_total_ms`` in a run's latency summary
(:func:`~threetears.evals.kernel.scoring.compute_latency_summary`) and a numeric measure's ``p95`` in the
analysis bundle (``MeasureSummary.p95``) — are one estimator,
:func:`~threetears.evals.kernel.scoring.median_unbiased_quantile` (Hyndman–Fan type 8). The two rules it
replaced understated the tail: the summary's nearest-rank was the sample maximum for every ``n <= 19`` (below
the true p95 0.86 of the time at n=3, 0.77 at n=5), and the bundle's linear interpolation sat below it 0.84,
0.73 and 0.68 of the time at n=5, 15 and 30.

A figure labelled "95th percentile" is read as an estimate of the population's 95th percentile, so the
property asked of each is that it is about as likely to fall above the truth as below: with probability in
[0.40, 0.60] (±0.10 allowing for a rank estimator's discreteness) the estimate is below the true p95.

**Below 13 observations no estimator built from the sample's order statistics has that property** — type 8's
position falls past the largest observation, and the largest of n falls below the true p95 with probability
``0.95^n`` (0.86 at n=3). So there the engine reports no p95 at all, and the maximum under its own name; the
tests below check that refusal rather than a criterion no such estimator can meet. Observations are normal;
for a continuous distribution an order statistic's law is exact (``P(X_(r) < q95) = P(Binomial(n, 0.95) >= r)``),
which the first test checks the engine against.
"""

from __future__ import annotations

import math
import random
from statistics import NormalDist

import pytest

from threetears.evals.schema import LatencyMetrics
from threetears.evals.kernel.analysis_measures import MeasureSummary
from threetears.evals.kernel.scoring import compute_latency_summary, median_unbiased_quantile
from packages.evals.tests.bundle_support import one_batch_bundle
from packages.evals.tests.factories import make_eval_result
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.simulation_support import TOLERANCE_Z, monte_carlo_se, within

#: The observations are normal latencies, mean 1000 ms and SD 100 ms (non-negative to every practical
#: purpose), and this is their true 95th percentile.
_MEAN, _SD = 1000.0, 100.0
_TRUE_P95 = _MEAN + _SD * NormalDist().inv_cdf(0.95)

#: How far from one half the share of estimates below the truth may sit, for a rank estimator's discreteness.
_MEDIAN_SLACK = 0.10

#: 4,000 replicates: SE of a share near 0.5 is 0.0079, a 4-SE band of ±0.032.
_REPLICATES = 4000


def _summaries(n: int, seed: str) -> list[dict[str, float]]:
    """``_REPLICATES`` run summaries of ``n`` normal latencies each."""
    rng = random.Random(seed)
    base = make_eval_result(eval_run_id="run-1", model="m1")
    rows = []
    for _ in range(_REPLICATES):
        results = [
            base.model_copy(
                update={"test_case_id": f"tc-{index}", "latency": LatencyMetrics(total_ms=rng.gauss(_MEAN, _SD))}
            )
            for index in range(n)
        ]
        (row,) = compute_latency_summary(results).values()
        rows.append(row)
    return rows


def _summary_p95_below_share(n: int, seed: str) -> float:
    """The share of runs of ``n`` observations whose summary ``p95_total_ms`` lies below the true p95."""
    return sum(row["p95_total_ms"] < _TRUE_P95 for row in _summaries(n, seed)) / _REPLICATES


def _median_unbiased(share: float) -> bool:
    return abs(share - 0.5) <= _MEDIAN_SLACK + TOLERANCE_Z * monte_carlo_se(0.5, _REPLICATES)


class TestTheRunSummaryP95:
    def test_at_thirteen_observations_it_is_the_largest_with_that_order_statistics_law(self) -> None:
        """The smallest n it reports at: type 8's position is exactly 13, so the estimate is the maximum.

        Its share below the truth is that order statistic's exact law, ``0.95^13 = 0.513``.
        """
        share = _summary_p95_below_share(13, "p95-summary-law-13")
        expected = _binomial_at_least(13, 13, 0.95)
        assert within(share, expected, _REPLICATES), (
            f"n=13: {share:.4f} below, the order statistic gives {expected:.4f}"
        )

    @pytest.mark.parametrize("n", [13, 15, 20, 30])
    def test_where_it_is_reported_it_is_median_unbiased(self, n: int) -> None:
        share = _summary_p95_below_share(n, f"p95-summary-{n}")
        assert _median_unbiased(share), f"n={n}: {share:.4f} of estimates below the true p95"

    @pytest.mark.parametrize("n", [3, 5, 12])
    def test_below_thirteen_it_reports_the_maximum_and_no_p95(self, n: int) -> None:
        """Where no order statistic is median-unbiased the engine says so: the tail is the maximum, by name.

        At n=3 and n=5 the old figure, labelled p95, fell below the true p95 0.86 and 0.77 of the time.
        """
        # The only candidate left, the maximum, falls below the true p95 with probability 0.95^n: more often
        # than not at every n here, so no figure from this sample is median-unbiased for it.
        assert 0.95**n > 0.5
        rng = random.Random(f"p95-summary-refused-{n}")
        totals = [rng.gauss(_MEAN, _SD) for _ in range(n)]
        base = make_eval_result(eval_run_id="run-1", model="m1")
        results = [
            base.model_copy(update={"test_case_id": f"tc-{index}", "latency": LatencyMetrics(total_ms=total)})
            for index, total in enumerate(totals)
        ]
        (row,) = compute_latency_summary(results).values()
        assert "p95_total_ms" not in row
        assert row["max_total_ms"] == max(totals)


def _binomial_at_least(n: int, rank: int, p: float) -> float:
    return sum(math.comb(n, x) * p**x * (1 - p) ** (n - x) for x in range(rank, n + 1))


def _type_8_p95(values: list[float]) -> float | None:
    """Hyndman and Fan's type 8 at 0.95, written out independently: ``h = (n + 1/3) 0.95 + 1/3`` in ``[1, n]``."""
    ordered = sorted(values)
    n = len(ordered)
    h = (n + 1.0 / 3.0) * 0.95 + 1.0 / 3.0
    if not 1.0 <= h <= n:
        return None
    j = int(h)
    if j == n:
        return ordered[-1]
    return ordered[j - 1] + (h - j) * (ordered[j] - ordered[j - 1])


def _bundle_total_ms(values: list[float]) -> MeasureSummary:
    results = [
        make_eval_result(id=f"r-{index}", test_case_id=f"tc-{index}", latency=LatencyMetrics(total_ms=value))
        for index, value in enumerate(values)
    ]
    (summary,) = one_batch_bundle(results, profile=toyhost_profile()).run_summaries
    (total,) = [measure for measure in summary.measures.measures if measure.name == "total_ms"]
    return total


class TestTheBundlePercentileIsType8:
    @pytest.mark.parametrize("n", [13, 15, 30])
    def test_a_measures_p95_is_the_type_8_quantile(self, n: int) -> None:
        rng = random.Random(f"p95-bundle-wiring-{n}")
        values = [round(rng.gauss(2000.0, 300.0), 3) for _ in range(n)]
        assert _bundle_total_ms(values).p95 == pytest.approx(_type_8_p95(values), rel=1e-12)

    @pytest.mark.parametrize("n", [3, 5, 12])
    def test_below_thirteen_a_measure_has_no_p95_and_keeps_its_maximum(self, n: int) -> None:
        rng = random.Random(f"p95-bundle-refused-{n}")
        values = [round(rng.gauss(2000.0, 300.0), 3) for _ in range(n)]
        total = _bundle_total_ms(values)
        assert total.p95 is None and total.p05 is None
        assert total.max == max(values)
        assert total.bad_tail() is None


@pytest.mark.parametrize("n", [15, 30])
def test_the_bundle_p95_is_median_unbiased(n: int) -> None:
    """At the sizes it reports. The linear rule it replaced sat below the truth 0.73 (n=15) and 0.68 (n=30)."""
    rng = random.Random(f"p95-bundle-{n}")
    below = 0
    for _ in range(_REPLICATES):
        estimate = median_unbiased_quantile(sorted(rng.gauss(_MEAN, _SD) for _ in range(n)), 0.95)
        assert estimate is not None
        below += estimate < _TRUE_P95
    share = below / _REPLICATES
    assert _median_unbiased(share), f"n={n}: {share:.4f} of estimates below the true p95"
