"""Seeded simulation of the pass^k estimator: unbiased at mixed depths, and pooling adds depth (#591).

Each case passes every attempt independently with a known probability ``p_i``, so the true pass^k
of the set is ``mean_i p_i^k``. A run stopped early leaves its cases at mixed depths, and because
cells execute in a per-run shuffled order WHICH case is left shallow is random — so every replicate
deals the depth multiset out to the cases afresh. The estimator under test is the production one,
fed real ``EvalResult`` rows; the all-pass indicator it replaced is restated here only to show the
simulation can see the bias it had.
"""

from __future__ import annotations

import math
import random
import statistics

import pytest

from threetears.evals.schema.models import EvalResult
from threetears.evals.kernel.scoring import compute_pass_hat_k, pass_hat_k_at, pool_pass_hat_k
from packages.evals.tests.factories import make_scored_result

#: Each case's chance of passing one attempt.
_P = (0.95, 0.9, 0.8, 0.7, 0.5)
#: The depths a stopped run left, from the issue: two cases measured once, one twice, two three times.
_ISSUE_DEPTHS = (1, 3, 2, 3, 1)
_DEEPEST = max(_ISSUE_DEPTHS)
_REPLICATES = 3000
_SEED = 591


def _true_pass_hat_k(k: int) -> float:
    return math.fsum(p**k for p in _P) / len(_P)


#: One result per (run, case, attempt, outcome), built once: the estimator only reads them, and
#: building a pydantic row per draw would cost the file its time budget.
_BANK: dict[tuple[str, int, int, bool], EvalResult] = {
    (run_id, case, attempt, passed): make_scored_result(
        test_case_id=f"tc{case}", run_id=run_id, k=attempt, goal_passes=(passed,)
    )
    for run_id in ("r1", "r2")
    for case in range(len(_P))
    for attempt in range(1, 2 * _DEEPEST + 1)
    for passed in (True, False)
}


def _attempts(rng: random.Random, depths: tuple[int, ...]) -> list[list[bool]]:
    """Each case's attempts, its depth dealt at random from ``depths``, as a shuffled stop leaves them."""
    dealt = list(depths)
    rng.shuffle(dealt)
    return [[rng.random() < p for _ in range(depth)] for p, depth in zip(_P, dealt, strict=True)]


def _rows(attempts: list[list[bool]], *, run_id: str = "r1", first: int = 1) -> list[EvalResult]:
    return [
        _BANK[(run_id, case, first + i, passed)]
        for case, case_attempts in enumerate(attempts)
        for i, passed in enumerate(case_attempts)
    ]


def _all_pass_indicator(attempts: list[list[bool]]) -> float:
    """The estimator #591 replaced: the share of cases that passed every attempt they happened to get."""
    return sum(all(case) for case in attempts) / len(attempts)


def test_the_estimator_is_unbiased_at_the_issues_mixed_depths_and_the_old_one_was_not():
    """Every point of the curve lands within Monte-Carlo error of mean_i p_i^k; the old figure does not.

    Tolerance is four Monte-Carlo standard errors of the replicate mean, the SE taken from the
    replicates' own spread: SE = sd / sqrt(R). At R = 3000 the per-replicate sd of the pass^3
    estimate is about 0.33 (two qualifying cases, each a 0/1 all-pass over three attempts), so
    SE ≈ 0.33 / sqrt(3000) ≈ 0.0060 and the band is ±0.024 — while the all-pass indicator's
    expectation here is mean_i(0.4 p_i + 0.2 p_i^2 + 0.4 p_i^3) = 0.637 against a true pass^3 of
    0.513, a bias of +0.124, about twenty of those SEs. Four SEs leaves a correct estimator a
    two-sided false-alarm chance near 6e-5 per check, and the seed fixes the draw anyway.
    """
    rng = random.Random(_SEED)
    estimates: dict[int, list[float]] = {k: [] for k in range(1, _DEEPEST + 1)}
    old: list[float] = []
    for _ in range(_REPLICATES):
        attempts = _attempts(rng, _ISSUE_DEPTHS)
        curve = compute_pass_hat_k(_rows(attempts), k=_DEEPEST)[("m1", "r1")]["pass_hat_k_curve"]
        for k, values in estimates.items():
            point = pass_hat_k_at(curve, k)
            # The depth multiset is fixed, so the number of cases measured at least k deep is too.
            assert point["n_cases"] == sum(depth >= k for depth in _ISSUE_DEPTHS)
            assert point["pass_hat_k"] is not None
            values.append(point["pass_hat_k"])
        old.append(_all_pass_indicator(attempts))

    for k, values in estimates.items():
        mean = statistics.fmean(values)
        tolerance = 4 * statistics.stdev(values) / math.sqrt(_REPLICATES)
        assert mean == pytest.approx(_true_pass_hat_k(k), abs=tolerance), f"pass^{k} is biased"

    old_mean = statistics.fmean(old)
    old_tolerance = 4 * statistics.stdev(old) / math.sqrt(_REPLICATES)
    assert old_mean - _true_pass_hat_k(_DEEPEST) > old_tolerance, "the simulation must be able to see the old bias"


def test_at_uniform_depth_the_estimator_is_the_all_pass_indicator():
    """C(c, n) / C(n, n) is 1 exactly when every attempt passed, so the two agree draw for draw."""
    rng = random.Random(_SEED)
    for _ in range(200):
        attempts = _attempts(rng, (_DEEPEST,) * len(_P))
        entry = compute_pass_hat_k(_rows(attempts), k=_DEEPEST)[("m1", "r1")]
        assert entry["pass_hat_k"] == pytest.approx(_all_pass_indicator(attempts))


def test_pooling_two_runs_of_one_cell_matches_one_run_of_the_combined_depth():
    """A repeat run is more attempts at the same cases: its pool reads as one run measured that deep."""
    rng = random.Random(_SEED)
    for _ in range(300):
        first, second = _attempts(rng, _ISSUE_DEPTHS), _attempts(rng, _ISSUE_DEPTHS)
        combined = [a + b for a, b in zip(first, second, strict=True)]

        pooled = pool_pass_hat_k(
            _rows(first, run_id="r1") + _rows(second, run_id="r2"),
            cell_of_run={"r1": "cell", "r2": "cell"},
            k=_DEEPEST,
        )
        one_run = compute_pass_hat_k(_rows(combined), k=_DEEPEST)[("m1", "r1")]

        assert pooled["pass_hat_k_curve"] == one_run["pass_hat_k_curve"]
        assert pooled["n_test_cases"] == one_run["n_test_cases"] == len(_P)
