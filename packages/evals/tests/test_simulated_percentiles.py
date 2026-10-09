"""The 95th percentiles the engine reports, checked against the population percentile they stand for (#601).

Two percentiles, two rules:

- ``p95_total_ms`` in a run's latency summary
  (:func:`~threetears.evals.contracts.scoring.compute_latency_summary`) is NEAREST-RANK: rank
  ``ceil(0.95 n)``, which is the sample maximum for every ``n <= 19`` — that is, at every run size up to
  19 observations, which covers most of what the engine sees (2–15 cases × 1–5 repeats).
- A numeric measure's ``p95`` in the analysis bundle (``MeasureSummary.p95``) is linearly interpolated, as
  numpy's default; ``statistics.quantiles(method="inclusive")`` is the same rule, independently written,
  and ``TestTheBundlePercentileIsLinear`` pins the bundle to it.

A figure labelled "95th percentile" is read as an estimate of the population's 95th percentile, so the
property asked of each is that it is about as likely to fall above the truth as below: with probability in
[0.40, 0.60] (±0.10 allowing for a rank estimator's discreteness) the estimate is below the true p95.
Observations are normal; for a continuous distribution the nearest-rank answer is exact
(``P(X_(r) < q95) = P(Binomial(n, 0.95) >= r)``), which the first test checks the engine against.
"""

from __future__ import annotations

import math
import random
import statistics
from statistics import NormalDist

import pytest

from threetears.evals.contracts import LatencyMetrics
from threetears.evals.contracts.scoring import compute_latency_summary
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


def _summary_p95_below_share(n: int, seed: str) -> float:
    """The share of runs of ``n`` observations whose summary ``p95_total_ms`` lies below the true p95."""
    rng = random.Random(seed)
    base = make_eval_result(eval_run_id="run-1", model="m1")
    below = 0
    for _ in range(_REPLICATES):
        results = [
            base.model_copy(
                update={"test_case_id": f"tc-{index}", "latency": LatencyMetrics(total_ms=rng.gauss(_MEAN, _SD))}
            )
            for index in range(n)
        ]
        (row,) = compute_latency_summary(results).values()
        below += row["p95_total_ms"] < _TRUE_P95
    return below / _REPLICATES


def _binomial_at_least(n: int, rank: int, p: float) -> float:
    return sum(math.comb(n, x) * p**x * (1 - p) ** (n - x) for x in range(rank, n + 1))


class TestTheRunSummaryP95:
    @pytest.mark.parametrize("n", [5, 15])
    def test_it_is_the_nearest_rank_order_statistic(self, n: int) -> None:
        """The share below the truth is the order statistic's exact law — at n=5, 0.95⁵ = 0.774: the maximum."""
        share = _summary_p95_below_share(n, f"p95-summary-law-{n}")
        expected = _binomial_at_least(n, math.ceil(0.95 * n), 0.95)
        assert within(share, expected, _REPLICATES), (
            f"n={n}: {share:.4f} below, the order statistic gives {expected:.4f}"
        )

    def test_at_fifteen_observations_it_is_median_unbiased(self) -> None:
        share = _summary_p95_below_share(15, "p95-summary-15")
        assert abs(share - 0.5) <= _MEDIAN_SLACK + TOLERANCE_Z * monte_carlo_se(0.5, _REPLICATES)

    @pytest.mark.parametrize("n", [3, 5])
    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason=(
            "#601 finding: p95_total_ms is nearest-rank, which for n <= 19 is the sample MAXIMUM, labelled the 95th "
            "percentile. At the run sizes a run of 3-5 observations has, it falls below the true p95 0.86 (n=3) and "
            "0.77 (n=5) of the time (mean bias -0.79 and -0.48 SD) rather than about half."
        ),
    )
    def test_at_small_n_it_is_median_unbiased(self, n: int) -> None:
        share = _summary_p95_below_share(n, f"p95-summary-small-{n}")
        assert abs(share - 0.5) <= _MEDIAN_SLACK + TOLERANCE_Z * monte_carlo_se(0.5, _REPLICATES), (
            f"n={n}: {share:.4f} of estimates below the true p95"
        )


def _linear_p95(values: list[float]) -> float:
    return statistics.quantiles(values, n=100, method="inclusive")[94]


class TestTheBundlePercentileIsLinear:
    @pytest.mark.parametrize("n", [3, 5, 15])
    def test_a_measures_p95_is_the_inclusive_quantile(self, n: int) -> None:
        rng = random.Random(f"p95-bundle-wiring-{n}")
        values = [round(rng.gauss(2000.0, 300.0), 3) for _ in range(n)]
        results = [
            make_eval_result(id=f"r-{index}", test_case_id=f"tc-{index}", latency=LatencyMetrics(total_ms=value))
            for index, value in enumerate(values)
        ]
        (summary,) = one_batch_bundle(results, profile=toyhost_profile()).run_summaries
        (total,) = [measure for measure in summary.measures.measures if measure.name == "total_ms"]
        assert total.p95 == pytest.approx(_linear_p95(values), rel=1e-12)


@pytest.mark.parametrize("n", [5, 15, 30])
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "#601 finding: a measure's bundle p95 is linearly interpolated, which at the engine's sample sizes sits "
        "below the true 95th percentile 0.84 (n=5), 0.73 (n=15) and 0.68 (n=30) of the time (mean bias -0.61, "
        "-0.25, -0.14 SD): a tail figure that understates the tail."
    ),
)
def test_the_bundle_p95_is_median_unbiased(n: int) -> None:
    rng = random.Random(f"p95-bundle-{n}")
    below = sum(1 for _ in range(_REPLICATES) if _linear_p95([rng.gauss(_MEAN, _SD) for _ in range(n)]) < _TRUE_P95)
    share = below / _REPLICATES
    assert abs(share - 0.5) <= _MEDIAN_SLACK + TOLERANCE_Z * monte_carlo_se(0.5, _REPLICATES), (
        f"n={n}: {share:.4f} of estimates below the true p95"
    )
