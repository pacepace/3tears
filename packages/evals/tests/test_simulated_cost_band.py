"""The cost estimate's prediction band, checked against sweeps whose cost is drawn from a known distribution (#601).

:func:`~threetears.evals.analysis.reporting.compute_estimate_cost` prices a proposed sweep from the history of
per-observation costs and brackets the prediction with a ~95% PREDICTION band for the sweep's total. The
band's claim is about the sweep that will actually run: drawn from the same distribution as the history, its
realised total falls inside the band 95% of the time.

The band is read on the log scale (:func:`~threetears.evals.analysis.stats.lognormal_sum_prediction_band`),
because costs are positive and right-skewed. The normal-theory band it replaced, ``t(0.95, n-1) · s ·
sqrt(m + m²/n)``, covered a lognormal sweep (log-SD 1.0) only 0.82 of the time, and 0.92 at log-SD 0.5. The
log-scale band has to hold on normal costs too, where the old band was exact.

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
    """Normal per-observation costs (mean $1, SD $0.30), where the old t band was exact.

    1,000 replicates: SE 0.0069, a 4-SE band of ±0.028 either side, so a band too WIDE fails it too.
    """
    replicates = 1000
    coverage = _coverage(lambda rng: rng.gauss(1.0, 0.3), replicates=replicates, seed="cost-band-normal")
    assert abs(coverage - INTERVAL_LEVEL) <= INTERVAL_LEVEL - at_least(INTERVAL_LEVEL, replicates), (
        f"coverage {coverage:.4f} against {INTERVAL_LEVEL}"
    )


@pytest.mark.parametrize("log_sd", [0.5, 1.0])
def test_on_skewed_costs_the_band_covers_the_sweep_at_its_level(log_sd: float) -> None:
    """Lognormal costs: at log-SD 1.0 the 95th percentile is ~5x the median, as a few long conversations make it.

    The normal band covered 0.92 (log-SD 0.5) and 0.82 (1.0). 1,000 replicates: SE at 95% is 0.0069, so the
    bound is 0.922.
    """
    replicates = 1000
    coverage = _coverage(
        lambda rng: rng.lognormvariate(0.0, log_sd), replicates=replicates, seed=f"cost-band-skewed-{log_sd}"
    )
    assert coverage >= at_least(INTERVAL_LEVEL, replicates), f"coverage {coverage:.4f} against {INTERVAL_LEVEL}"


def test_the_band_states_what_it_assumes() -> None:
    """A band read on the log scale says so, and says it treats each observation as independent."""
    profile = toyhost_profile()
    run = make_eval_run(status="completed", candidate_model="priced-model", template_id="template-1")
    base = make_eval_result(eval_run_id=run.id, scope_id=run.scope_id, model="priced-model")
    history = [base.model_copy(update={"cost_usd": cost}) for cost in (0.8, 1.1, 1.6, 0.9, 2.4)]
    (cell,) = compute_estimate_cost(
        [run], history, models=["priced-model"], k_runs=3, n_test_cases=5, template_id="template-1", profile=profile
    ).cells
    assert cell.band_basis is not None
    assert "lognormal" in cell.band_basis
    assert "independent" in cell.band_basis
