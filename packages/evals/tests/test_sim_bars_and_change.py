"""Seeded simulations with a known truth for the two decisions read against a declared margin.

A bar's three-valued verdict (:func:`~threetears.evals.analysis.stats.interval_clears`, seeded by
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
from typing import NamedTuple

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


class _BarRates(NamedTuple):
    """How often one candidate's verdict came out each way, and how often the replaced rule missed it."""

    cleared: float
    missed: float
    undecided: float
    mean_rule_missed: float


def _bar_rates(n: int, shortfall: float, *, higher_is_better: bool, margin: float | None, seed: int) -> _BarRates:
    """The verdict rates for a candidate ``shortfall`` σ worse than the incumbent, measured on as many cases.

    Each repetition measures the incumbent, seeds its bar from that measurement, then measures a
    candidate whose true mean sits ``shortfall`` below (above, when lower is better) the
    incumbent's. ``mean_rule_missed`` is the rule this replaced, computed here only to show what it
    did: a bar at the incumbent's mean, read on the candidate's mean.
    """
    rng = random.Random(seed)
    counts: Counter[bool | None] = Counter()
    mean_rule_misses = 0
    for _ in range(REPS):
        incumbent = _sample(rng, n, 0.0, higher_is_better=higher_is_better)
        candidate = _sample(rng, n, -shortfall, higher_is_better=higher_is_better)
        incumbent_interval = observed_mean_interval(incumbent)
        candidate_interval = observed_mean_interval(candidate)
        assert incumbent_interval is not None and candidate_interval is not None, "every sample here has n >= 2"
        incumbent_mean, candidate_mean = sum(incumbent) / n, sum(candidate) / n
        threshold = bar_seed(incumbent_mean, incumbent_interval, higher_is_better=higher_is_better)
        counts[interval_clears(candidate_interval, threshold, margin=margin, higher_is_better=higher_is_better)] += 1
        mean_rule_misses += candidate_mean < incumbent_mean if higher_is_better else candidate_mean > incumbent_mean
    return _BarRates(counts[True] / REPS, counts[False] / REPS, counts[None] / REPS, mean_rule_misses / REPS)


def _upper(rate: float) -> float:
    """``rate`` plus three Monte-Carlo standard errors at that rate: the bound on a rate whose truth is ``rate``."""
    return rate + 3 * math.sqrt(rate * (1 - rate) / REPS)


_DIRECTIONS = pytest.mark.parametrize("higher_is_better", [True, False], ids=["higher is better", "lower is better"])
_MARGINS = pytest.mark.parametrize("margin", [None, MARGIN], ids=["no margin", "margin 0.1σ"])


@_DIRECTIONS
@_MARGINS
@pytest.mark.parametrize("n", [3, 6, 15])
def test_an_unchanged_incumbent_misses_its_own_bar_at_most_at_the_nominal_rate(
    n: int, margin: float | None, higher_is_better: bool
) -> None:
    """Re-measured unchanged on as many cases, the incumbent misses the bar its own baseline proposed at most 2.5% of the time.

    2.5% is the one-sided rate the 95% interval promises, and the seed is derived to meet it
    (:data:`~threetears.evals.analysis.stats.BAR_SEED_HALF_WIDTH_FRACTION`). Simulated with no
    margin: about 1.3%, 2.0% and 2.2% at n = 3, 6, 15 (under nominal at small n, since t on n − 1
    degrees of freedom is wider than the two-sample test needs). The bound is 2.5% plus three
    standard errors, sqrt(0.025 * 0.975 / 4000) ≈ 0.0025, so ≈ 0.032.

    Most re-measurements are undecided, not cleared: a small bank cannot show the incumbent meets
    its own bar, and saying so is the point. Asserted: undecided at least half the time
    (simulated 70–80%; at p = 0.7 the standard error is ≈ 0.0072, so 0.5 is about 28 below).

    The rule this replaced — a bar at the incumbent's mean, read on the candidate's mean — misses
    an unchanged incumbent half the time by symmetry; at p = 0.5 the standard error is
    sqrt(0.25 / 4000) ≈ 0.0079, so 0.5 ± 0.04 is five standard errors either side.
    """
    rates = _bar_rates(n, 0.0, higher_is_better=higher_is_better, margin=margin, seed=593 + n)

    assert rates.missed <= _upper(0.025)
    assert rates.undecided >= 0.5
    assert abs(rates.mean_rule_missed - 0.5) <= 0.04


@_DIRECTIONS
@_MARGINS
@pytest.mark.parametrize("n", [3, 6, 15])
def test_a_candidate_clearly_worse_than_the_incumbent_is_almost_never_shown_to_clear_its_bar(
    n: int, margin: float | None, higher_is_better: bool
) -> None:
    """A candidate 1.6σ worse than the incumbent — sixteen margins — is shown to clear its bar at most 5% of the time.

    Simulated: about 2% at n = 3 and under 0.2% from n = 6. The bound is 5%, about thirteen standard
    errors (sqrt(0.02 * 0.98 / 4000) ≈ 0.0022) above the worst simulated rate. A small bank
    instead leaves it undecided — clearing is never what absence of evidence reads as.
    """
    rates = _bar_rates(n, 1.6, higher_is_better=higher_is_better, margin=margin, seed=1593 + n)

    assert rates.cleared <= 0.05


@_DIRECTIONS
@_MARGINS
def test_a_candidate_clearly_worse_than_the_incumbent_misses_its_bar_most_of_the_time_at_fifteen_cases(
    margin: float | None, higher_is_better: bool
) -> None:
    """At n = 15 the 1.6σ-worse candidate is shown to miss its bar in almost every repetition.

    Simulated power: about 97–98% at n = 15 (about 55–60% at n = 6, 14–15% at n = 3: a bar on a
    handful of cases leaves most regressions undecided, which its interval shows). The bound is
    0.9; at p = 0.97 the standard error is sqrt(0.97 * 0.03 / 4000) ≈ 0.0027, so 0.9 is about
    26 standard errors below.
    """
    rates = _bar_rates(15, 1.6, higher_is_better=higher_is_better, margin=margin, seed=1593 + 15)

    assert rates.missed >= 0.9


@pytest.mark.parametrize(
    ("interval", "decision"),
    [((0.81, 0.95), True), ((0.70, 0.79), False), ((0.75, 0.85), None), ((0.80, 0.90), True)],
    ids=["wholly above", "wholly below", "straddling", "touching the line"],
)
def test_each_side_of_the_line_and_the_straddle(interval: tuple[float, float], decision: bool | None) -> None:
    """A bar at 0.8: cleared needs the whole interval at or above it, missed the whole interval below it."""
    assert interval_clears(interval, 0.8, margin=None, higher_is_better=True) is decision
    mirrored = (-interval[1], -interval[0])
    assert interval_clears(mirrored, -0.8, margin=None, higher_is_better=False) is decision


def test_the_margin_moves_the_line_toward_the_bad_side() -> None:
    """With 0.05 of declared margin an interval wholly above 0.75 clears a bar at 0.8, and one wholly under 0.75 misses."""
    assert interval_clears((0.76, 0.79), 0.8, margin=0.05, higher_is_better=True) is True
    assert interval_clears((0.70, 0.74), 0.8, margin=0.05, higher_is_better=True) is False


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
