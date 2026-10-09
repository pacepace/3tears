"""A reading's interval covers the truth at its stated level when the observations repeat cases (#590).

Seeded simulation with a known truth. Each replicate draws ``G`` cases from a population, observes each
``k`` times, computes the interval the engine reports, and asks whether it holds the population value.
The configurations are the ones the engine meets: 2 to 15 cases at 1 to 5 repeats, with the spread
between cases dominating the spread between repeats, and comparable to it.

The interval it replaced was a t interval over every observation, which counted ``k`` repeats of a case
as ``k`` draws. At 5 cases x 3 repeats with between-case σ 1.0 and repeat σ 0.3 it covered the truth about
72% of the time, and 51% at 2 cases x 5; a Wilson interval over the observations of a proportion fell to 67%.
:func:`test_the_coverage_check_would_catch_an_interval_over_observations` keeps that failure visible, so
this file is known to be able to fail.

**The tolerance is the Monte-Carlo error, not a fudge.** A coverage estimated from ``R`` independent
replicates of an interval whose true coverage is ``p`` has standard error ``sqrt(p(1-p)/R)``. At
``p = 0.95`` and ``R = 1000`` that is ``sqrt(0.95 x 0.05 / 1000) = 0.0069``, so a configuration fails only
when its coverage is more than four standard errors short: below ``0.95 - 4 x 0.0069 = 0.922``. Four rather
than two because roughly a hundred configurations are checked, and at two a correct interval would fail
one of them by chance about once in a run; at four the chance per configuration is about 3 in 100,000.
The pooled coverage over every balanced mean configuration (40,000 replicates, standard error
``sqrt(0.95 x 0.05 / 40000) = 0.0011``) is held two-sided to ``0.95 ± 0.0044``: over equal repeats the
interval is exact, so it must neither undercover nor quietly widen.

The seed only fixes which draws are made; nothing here is tuned to it.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable

import pytest

from threetears.evals.analysis.stats import (
    INTERVAL_LEVEL,
    clustered_standard_error,
    mean_interval,
    observed_mean_interval,
    proportion_interval,
    standard_error_of_mean,
    wilson_interval,
)

REPLICATES = 1000
MC_SE = math.sqrt(INTERVAL_LEVEL * (1 - INTERVAL_LEVEL) / REPLICATES)
FLOOR = INTERVAL_LEVEL - 4 * MC_SE

CASE_COUNTS = (2, 3, 5, 8, 15)
#: (between-case σ, repeat σ): the case dominating the repeat, then the two comparable.
SPREADS = ((1.0, 0.3), (1.0, 1.0))
#: (population rate, Beta concentration): concentration 1 makes cases nearly all-or-nothing (ICC 0.5),
#: concentration 5 leaves the repeats as variable as the cases (ICC 1/6).
RATES = tuple((rate, concentration) for rate in (0.5, 0.8, 0.95) for concentration in (1.0, 5.0))

Interval = tuple[float, float] | None
Sample = tuple[list[float], list[int]]


def _normal(rng: random.Random, cases: int, repeats: int, between: float, within: float, drop: float = 0.0) -> Sample:
    """``cases`` draws of a case effect around a true mean of 0, each observed ``repeats`` times."""
    values: list[float] = []
    owners: list[int] = []
    for case in range(cases):
        effect = rng.gauss(0.0, between)
        for _ in range(repeats):
            if drop and rng.random() < drop:
                continue
            values.append(effect + rng.gauss(0.0, within))
            owners.append(case)
    return values, owners


def _binary(rng: random.Random, cases: int, repeats: int, rate: float, concentration: float) -> Sample:
    """``cases`` case rates from a Beta with mean ``rate``, each case observed ``repeats`` times as 0 or 1."""
    values: list[float] = []
    owners: list[int] = []
    for case in range(cases):
        held = rng.betavariate(rate * concentration, (1 - rate) * concentration)
        for _ in range(repeats):
            values.append(1.0 if rng.random() < held else 0.0)
            owners.append(case)
    return values, owners


def _coverage(draw: Callable[[], Sample], interval: Callable[[Sample], Interval], truth: float) -> float:
    """The share of replicates whose interval holds ``truth``; a replicate with no interval counts as a miss."""
    held = 0
    for _ in range(REPLICATES):
        bounds = interval(draw())
        if bounds is not None and bounds[0] <= truth <= bounds[1]:
            held += 1
    return held / REPLICATES


def _reported_mean(sample: Sample) -> Interval:
    values, cases = sample
    return observed_mean_interval(values, cases=cases)


def _reported_rate(sample: Sample) -> Interval:
    values, cases = sample
    return observed_mean_interval(values, cases=cases, value_range=(0.0, 1.0))


def _over_observations_mean(sample: Sample) -> Interval:
    """The interval #590 replaced: every observation its own draw."""
    values, _ = sample
    sem = standard_error_of_mean(values)
    return None if sem is None else mean_interval(sum(values) / len(values), sem, len(values))


def _over_observations_rate(sample: Sample) -> Interval:
    values, _ = sample
    return wilson_interval(sum(1 for value in values if value == 1.0), len(values))


# =============================================================================
# Coverage
# =============================================================================


def test_a_mean_interval_covers_the_truth_at_every_case_count_and_depth() -> None:
    rng = random.Random(590)
    short: list[str] = []
    covered = 0.0
    configurations = 0
    for between, within in SPREADS:
        for cases in CASE_COUNTS:
            for repeats in (1, 2, 3, 5):
                coverage = _coverage(lambda: _normal(rng, cases, repeats, between, within), _reported_mean, truth=0.0)
                covered += coverage
                configurations += 1
                if coverage < FLOOR:
                    short.append(f"{cases} cases x {repeats}, σ {between}/{within}: {coverage:.3f}")
    assert not short, f"below {FLOOR:.3f}: {short}"
    pooled = covered / configurations
    pooled_se = math.sqrt(INTERVAL_LEVEL * (1 - INTERVAL_LEVEL) / (REPLICATES * configurations))
    assert abs(pooled - INTERVAL_LEVEL) <= 4 * pooled_se, f"pooled coverage {pooled:.4f}"


def test_a_mean_over_unevenly_repeated_cases_still_covers_the_truth() -> None:
    """Some repeats lost — a faulted result is left out — so the cases carry unequal depth.

    The cluster-robust standard error is approximate at unequal depth (it measured about 0.94 at its
    worst here, against 0.95 over equal depth), so only the floor is held.
    """
    rng = random.Random(5901)
    short: list[str] = []
    for between, within in SPREADS:
        for cases in (5, 8, 15):
            for repeats in (3, 5):
                coverage = _coverage(
                    lambda: _normal(rng, cases, repeats, between, within, drop=0.15), _reported_mean, truth=0.0
                )
                if coverage < FLOOR:
                    short.append(f"{cases} cases x {repeats}, σ {between}/{within}: {coverage:.3f}")
    assert not short, f"below {FLOOR:.3f}: {short}"


def test_a_rate_interval_covers_the_truth_at_every_case_count_and_depth() -> None:
    """Proportions, repeated: the interval is approximate and leans wide at few cases, so only the floor is held.

    A single observation per case is the Wilson interval unchanged
    (:func:`test_one_observation_per_case_is_the_unclustered_interval_exactly`), and its own small-n behaviour
    is not what this file tests.
    """
    rng = random.Random(5902)
    short: list[str] = []
    for rate, concentration in RATES:
        for cases in CASE_COUNTS:
            for repeats in (2, 3, 5):
                coverage = _coverage(
                    lambda: _binary(rng, cases, repeats, rate, concentration), _reported_rate, truth=rate
                )
                if coverage < FLOOR:
                    short.append(
                        f"rate {rate}, concentration {concentration}, {cases} cases x {repeats}: {coverage:.3f}"
                    )
    assert not short, f"below {FLOOR:.3f}: {short}"


@pytest.mark.parametrize(
    ("draw", "interval", "truth"),
    [
        (lambda rng: _normal(rng, 5, 3, 1.0, 0.3), _over_observations_mean, 0.0),
        (lambda rng: _binary(rng, 5, 5, 0.8, 1.0), _over_observations_rate, 0.8),
    ],
    ids=["mean", "rate"],
)
def test_the_coverage_check_would_catch_an_interval_over_observations(
    draw: Callable[[random.Random], Sample], interval: Callable[[Sample], Interval], truth: float
) -> None:
    """The replaced interval fails the same floor by a wide margin, so a pass above is a measured one."""
    rng = random.Random(5903)
    coverage = _coverage(lambda: draw(rng), interval, truth)
    assert coverage < FLOOR - 0.1, coverage


# =============================================================================
# The clustered forms reduce to the ones a reader knows
# =============================================================================


def test_one_observation_per_case_is_the_unclustered_interval_exactly() -> None:
    values = [3.0, 4.5, 2.0, 5.0, 4.0, 3.5]
    one_each = list(range(len(values)))
    sem = standard_error_of_mean(values)
    assert sem is not None

    assert clustered_standard_error(values, one_each) == sem
    assert observed_mean_interval(values, cases=one_each) == mean_interval(sum(values) / len(values), sem, len(values))
    for n_true in range(6):
        outcomes = [index < n_true for index in range(6)]
        assert proportion_interval(outcomes, one_each) == wilson_interval(n_true, 6)


def test_equal_repeats_take_the_standard_error_of_the_case_means() -> None:
    """The statistic a comparison between arms already runs on, so a reading and a comparison count alike."""
    by_case = [[1.0, 1.5, 0.5], [3.0, 2.0, 2.5], [4.0, 4.5, 5.0], [2.0, 2.0, 3.5]]
    values = [value for case in by_case for value in case]
    cases = [index for index, case in enumerate(by_case) for _ in case]
    case_means = [sum(case) / len(case) for case in by_case]

    assert clustered_standard_error(values, cases) == pytest.approx(standard_error_of_mean(case_means))


@pytest.mark.parametrize("values", [[2.0, 3.0, 4.0], [1.0, 0.0, 1.0]], ids=["numeric", "proportion"])
def test_repeats_of_one_case_have_no_interval(values: list[float]) -> None:
    """One case has no between-case spread to estimate, however often it was repeated."""
    one_case = [0] * len(values)
    assert clustered_standard_error(values, one_case) is None
    assert observed_mean_interval(values, cases=one_case, value_range=(0.0, 4.0)) is None
    assert observed_mean_interval(values, cases=one_case, value_range=(0.0, 1.0)) is None


@pytest.mark.parametrize("repeats", [2, 3, 5])
@pytest.mark.parametrize("cases", [2, 4, 9])
def test_a_repeated_rate_keeps_a_width_and_contains_itself_at_both_ends(cases: int, repeats: int) -> None:
    owners = [case for case in range(cases) for _ in range(repeats)]
    perfect = proportion_interval([True] * len(owners), owners)
    none = proportion_interval([False] * len(owners), owners)
    assert perfect is not None and perfect[1] == 1.0 and perfect[0] < 1.0
    assert none is not None and none[0] == 0.0 and none[1] > 0.0


def test_observations_and_cases_must_align() -> None:
    with pytest.raises(ValueError, match="every observation needs its case"):
        observed_mean_interval([1.0, 2.0], cases=[0])
