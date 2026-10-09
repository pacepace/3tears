"""The cost estimate's prediction band, checked against sweeps whose cost is drawn from a known distribution (#601).

:func:`~threetears.evals.analysis.reporting.compute_estimate_cost` prices a proposed sweep from the history of
per-observation costs and brackets the prediction with a ~95% PREDICTION band for the sweep's total:
``t(0.95, n-1) · s · sqrt(m + m²/n)`` over ``n`` historical observations and ``m`` proposed ones. The
band's claim is about the sweep that will actually run: drawn from the same distribution as the history, its
realised total falls inside the band 95% of the time.

Each replicate draws a history of five priced observations, asks for the estimate of a 5-case, k=3 sweep
(15 observations), then draws that sweep's 15 costs from the same distribution and checks whether their
total landed inside the band.
"""

from __future__ import annotations

import random
from collections.abc import Callable

import pytest

from threetears.evals.analysis.reporting import compute_estimate_cost
from threetears.evals.analysis.stats import INTERVAL_LEVEL
from packages.evals.tests.factories import make_eval_result, make_eval_run
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.simulation_support import at_least

_HISTORY = 5
_CASES, _REPEATS = 5, 3


def _coverage(draw: Callable[[random.Random], float], *, replicates: int, seed: str) -> float:
    """The share of replicates whose realised sweep total lands inside the published band."""
    rng = random.Random(seed)
    profile = toyhost_profile()
    run = make_eval_run(status="completed", candidate_model="priced-model", template_id="template-1")
    base = make_eval_result(eval_run_id=run.id, scope_id=run.scope_id, model="priced-model")
    covered = 0
    for _ in range(replicates):
        history = [base.model_copy(update={"cost_usd": draw(rng)}) for _ in range(_HISTORY)]
        estimate = compute_estimate_cost(
            [run],
            history,
            models=["priced-model"],
            k_runs=_REPEATS,
            n_test_cases=_CASES,
            template_id="template-1",
            profile=profile,
        )
        (cell,) = estimate.cells
        assert cell.predicted is not None
        low, high = cell.predicted.interval_low, cell.predicted.interval_high
        assert low is not None and high is not None
        realised = sum(draw(rng) for _ in range(_CASES * _REPEATS))
        covered += low <= realised <= high
    return covered / replicates


def test_on_normal_costs_the_band_covers_the_sweep_at_its_level() -> None:
    """Normal per-observation costs (mean $1, SD $0.30): the t prediction band is exact here, so it covers
    95%. 1,000 replicates: SE 0.0069, a 4-SE band of ±0.028."""
    replicates = 1000
    coverage = _coverage(lambda rng: rng.gauss(1.0, 0.3), replicates=replicates, seed="cost-band-normal")
    assert abs(coverage - INTERVAL_LEVEL) <= INTERVAL_LEVEL - at_least(INTERVAL_LEVEL, replicates), (
        f"coverage {coverage:.4f} against {INTERVAL_LEVEL}"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "#601 finding: the cost band assumes normal per-observation costs. With right-skewed costs (lognormal, "
        "log-SD 1.0: the 95th percentile ~5x the median, as a few long conversations make it) and five "
        "historical observations, measured coverage of the realised 15-observation sweep total 0.82 against "
        "the stated ~95%; at log-SD 0.5 it is 0.92."
    ),
)
def test_on_skewed_costs_the_band_covers_the_sweep_at_its_level() -> None:
    """1,000 replicates: SE at 95% is 0.0069, so the bound is 0.922."""
    replicates = 1000
    coverage = _coverage(lambda rng: rng.lognormvariate(0.0, 1.0), replicates=replicates, seed="cost-band-skewed")
    assert coverage >= at_least(INTERVAL_LEVEL, replicates), f"coverage {coverage:.4f} against {INTERVAL_LEVEL}"
