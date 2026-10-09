"""Seeded simulations with a known truth for the two decisions read against a declared margin.

A bar's verdict (:func:`~threetears.evals.analysis.stats.interval_clears`, seeded by
:func:`~threetears.evals.analysis.stats.bar_seed`) and the run-history change read
(:func:`~threetears.evals.analysis.stats.paired_change` with its equivalence test) are checked
against data drawn from a distribution whose truth the test knows, at the case counts an eval
actually runs (n = 3, 6, 15). Each case is observed once, so a reading's interval is not
affected by repeats of one case.

Every rate is a Monte-Carlo estimate, so every bound is set from its standard error,
``sqrt(p (1 - p) / N)``, and stated where it is asserted. The generators are seeded, so a run
is deterministic; the tolerances say how far the bound sits from what the truth predicts, so a
change of seed would not flip an assertion by luck.
"""

from __future__ import annotations

import math
import random
from collections import Counter

import pytest

from threetears.evals.analysis.stats import (
    SIGNIFICANCE_ALPHA,
    bar_seed,
    interval_clears,
    observed_mean_interval,
    paired_change,
)

#: Simulated repetitions per rate. At p = 0.05 the standard error is sqrt(0.05 * 0.95 / 4000) ≈ 0.0034.
REPS = 4000

#: The measure's declared margin, in units of the case-level standard deviation (σ = 1).
MARGIN = 0.1


def _sample(rng: random.Random, n: int, mean: float, *, higher_is_better: bool) -> list[float]:
    """``n`` case values around ``mean`` with σ = 1, mirrored for a lower-is-better measure."""
    sign = 1.0 if higher_is_better else -1.0
    return [sign * rng.gauss(mean, 1.0) for _ in range(n)]


def _bar_rates(n: int, shortfall: float, *, higher_is_better: bool, seed: int) -> tuple[float, float]:
    """``(clear rate, mean-rule miss rate)`` for a candidate ``shortfall`` σ worse than the incumbent.

    Each repetition measures the incumbent, seeds its bar from that measurement, then measures a
    candidate whose true mean sits ``shortfall`` below (above, when lower is better) the
    incumbent's. The second rate is the rule this replaced, computed here only to show what it
    did: a bar at the incumbent's mean, read on the candidate's mean.
    """
    rng = random.Random(seed)
    cleared = mean_rule_misses = 0
    for _ in range(REPS):
        incumbent = _sample(rng, n, 0.0, higher_is_better=higher_is_better)
        candidate = _sample(rng, n, -shortfall, higher_is_better=higher_is_better)
        incumbent_interval = observed_mean_interval(incumbent)
        assert incumbent_interval is not None
        threshold = bar_seed(incumbent_interval, higher_is_better=higher_is_better)
        verdict = interval_clears(
            observed_mean_interval(candidate), threshold, margin=MARGIN, higher_is_better=higher_is_better
        )
        assert verdict is not None, "every sample here has n >= 2, so every verdict is decided"
        cleared += verdict
        incumbent_mean, candidate_mean = sum(incumbent) / n, sum(candidate) / n
        mean_rule_misses += candidate_mean < incumbent_mean if higher_is_better else candidate_mean > incumbent_mean
    return cleared / REPS, mean_rule_misses / REPS


@pytest.mark.parametrize("higher_is_better", [True, False], ids=["higher is better", "lower is better"])
@pytest.mark.parametrize("n", [3, 6, 15])
def test_an_unchanged_incumbent_clears_its_own_bar_at_the_nominal_rate(n: int, higher_is_better: bool) -> None:
    """Re-measured unchanged, the incumbent clears the bar its own baseline proposed at least 95% of the time.

    Simulated: 99.7–99.85% at every n (conservative: the seed sits at the far end of one interval
    and the verdict reads the near end of the other). The bound is the interval's nominal 95%;
    at p = 0.997 the standard error is sqrt(0.997 * 0.003 / 4000) ≈ 0.0009, so the bound sits
    dozens of standard errors below the expected rate.

    The rule this replaced — a bar at the incumbent's mean, read on the candidate's mean — misses
    an unchanged incumbent half the time by symmetry; at p = 0.5 the standard error is
    sqrt(0.25 / 4000) ≈ 0.0079, so 0.5 ± 0.04 is five standard errors either side.
    """
    clear_rate, mean_rule_miss_rate = _bar_rates(n, 0.0, higher_is_better=higher_is_better, seed=593 + n)

    assert clear_rate >= 0.95
    assert abs(mean_rule_miss_rate - 0.5) <= 0.04


@pytest.mark.parametrize("higher_is_better", [True, False], ids=["higher is better", "lower is better"])
def test_a_candidate_clearly_worse_than_the_margin_misses_its_bar_most_of_the_time_at_fifteen_cases(
    higher_is_better: bool,
) -> None:
    """A candidate 1.6σ worse than the incumbent — sixteen margins — misses its bar in most repetitions at n = 15.

    Simulated power: about 86% at n = 15 (and about 26% at n = 6, 4% at n = 3: a bar on a handful
    of cases catches only a gross regression, which its interval shows). The bound asserts
    "most of the time"; at p = 0.86 the standard error is sqrt(0.86 * 0.14 / 4000) ≈ 0.0055, so
    0.75 sits about twenty standard errors below the expected rate.
    """
    clear_rate, _ = _bar_rates(15, 1.6, higher_is_better=higher_is_better, seed=1593)

    assert 1.0 - clear_rate >= 0.75


def test_a_single_observation_decides_nothing() -> None:
    """One value has no interval, so its verdict is undecided rather than a point compared with the threshold."""
    assert observed_mean_interval([0.9]) is None
    assert interval_clears(None, 0.5, margin=MARGIN, higher_is_better=True) is None


def _paired(
    rng: random.Random, n: int, true_change: float, spread: float, margin: float | None
) -> tuple[str, float | None]:
    """One simulated pair of runs over ``n`` shared cases: the change label and its TOST p."""
    baseline = [rng.gauss(0.0, 1.0) for _ in range(n)]
    current = [value + true_change + rng.gauss(0.0, spread) for value in baseline]
    verdict = paired_change(
        baseline,
        current,
        min_absolute_change=0.0,
        min_relative_change=0.0,
        higher_is_better=True,
        equivalence_margin=margin,
    )
    return verdict.label, verdict.equivalence_p


#: The upper bound on a rate whose truth is at most α: α plus three Monte-Carlo standard errors,
#: sqrt(0.05 * 0.95 / 4000) ≈ 0.0034, so ≈ 0.060.
_AT_MOST_ALPHA = SIGNIFICANCE_ALPHA + 3 * math.sqrt(SIGNIFICANCE_ALPHA * (1 - SIGNIFICANCE_ALPHA) / REPS)


@pytest.mark.parametrize("n", [6, 15])
def test_equivalence_is_falsely_claimed_at_most_alpha_when_the_true_change_sits_on_the_margin(n: int) -> None:
    """With the true change exactly at the margin, TOST shows equivalence at most α of the time.

    The boundary is where the test's error rate is largest, so this is its size. Simulated:
    about 4.7% at n = 6 and 5.1% at n = 15. Measured on the TOST p, the test itself, and on the
    label, which a directional reading may pre-empt and so can only be rarer.
    """
    rng = random.Random(592 + n)
    readings = [_paired(rng, n, 0.5, 0.5, 0.5) for _ in range(REPS)]

    tost_rejections = sum(1 for _, p in readings if p is not None and p < SIGNIFICANCE_ALPHA)
    labelled_equivalent = sum(1 for label, _ in readings if label == "equivalent")
    assert tost_rejections / REPS <= _AT_MOST_ALPHA
    assert labelled_equivalent <= tost_rejections


def test_an_underpowered_null_reads_not_separated_never_no_change() -> None:
    """No true change, three cases, a margin far inside the noise: the read is not_separated, not equivalent.

    The case the old ``flat`` label got wrong: almost nothing is significant at three cases, and
    the read claimed stability anyway. Simulated: about 95% not_separated, 5% a directional label
    (the paired test's own α), and 0.15% equivalent. Bounds: not_separated at least 0.90, about
    thirteen standard errors (sqrt(0.95 * 0.05 / 4000) ≈ 0.0034) below the expected rate;
    equivalent at most α plus three standard errors, since TOST's error rate is at most α at
    every n.
    """
    rng = random.Random(3592)
    labels = Counter(_paired(rng, 3, 0.0, 1.0, MARGIN * 2)[0] for _ in range(REPS))

    assert labels["not_separated"] / REPS >= 0.90
    assert labels["equivalent"] / REPS <= _AT_MOST_ALPHA
    assert set(labels) <= {"not_separated", "equivalent", "improved", "regressed"}


def test_without_a_declared_margin_no_read_claims_equivalence() -> None:
    """The same precise null that earns `equivalent` with a margin never does without one."""
    rng = random.Random(4592)
    readings = [_paired(rng, 15, 0.0, 0.2, None) for _ in range(500)]

    assert all(label != "equivalent" and p is None for label, p in readings)


def test_a_precisely_measured_null_inside_the_margin_reads_equivalent() -> None:
    """Equivalence is reachable: no true change, fifteen cases, spread at the margin — mostly `equivalent`.

    Simulated: about 92%. At p = 0.92 the standard error is sqrt(0.92 * 0.08 / 4000) ≈ 0.0043, so
    0.85 is about sixteen standard errors below.
    """
    rng = random.Random(5592)
    labels = Counter(_paired(rng, 15, 0.0, 0.2, 0.2)[0] for _ in range(REPS))

    assert labels["equivalent"] / REPS >= 0.85
