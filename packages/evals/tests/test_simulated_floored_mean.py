"""The interval on a mean cost or latency covers the truth at its stated level on skewed values.

Seeded simulation with a known truth. A spend and a time are bounded below only and right-skewed: a few long
conversations cost several times the median. Each replicate draws cases from a distribution whose mean
is known, asks the engine for the interval it reports
(:func:`~threetears.evals.analysis.stats.observed_mean_interval` with a floor of zero and ``skewed``, the call the
bundle makes for a measure on the cost or latency axis, or in dollars or a unit of time) and checks whether it holds the true mean.

**What is held.** At 5, 10, 20 and 50 cases the interval covers the true mean at least 95% of the time, less the
Monte-Carlo error, on:

- lognormal values at log-SD 0.5, 1.0 and 1.5 (at 1.5 the 95th percentile is about twelve times the median);
- gamma (shape 2), exponential and normal (SD 0.3 of the mean) values, which are not lognormal;
- cases repeated three times, where the repeats of a case share its level, with and without lost repeats;
- cases of which a fifth cost nothing (from 10 cases: at 5, five above the floor are rare, and under five there
  is no interval).

**What it replaced.** The t interval clipped at zero, labelled 95%, covered 83% at 10 lognormal cases (log-SD 1.0),
91% at 50, and 72% at 10 cases at log-SD 1.5.
:func:`test_the_clipped_t_interval_would_fail_this_check` keeps that failure visible, so this file is known to be
able to fail.

**The tolerance is the Monte-Carlo error.** 1,000 replicates: the standard error of a 95% coverage is 0.0069, and a
configuration fails below ``0.95 − 4 × 0.0069 = 0.922`` (``simulation_support.at_least``). The interval is a union
of two and runs wide, so it is held from below only; how wide is stated in ``docs/measuring-soundly.md``.

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
)
from packages.evals.tests.simulation_support import at_least

REPLICATES = 1000
CASE_COUNTS = (5, 10, 20, 50)

#: One case's observations: ``(rng, repeats) -> values``.
DrawCase = Callable[[random.Random, int], list[float]]
Interval = tuple[float, float] | None


def _independent(draw: Callable[[random.Random], float]) -> DrawCase:
    return lambda rng, repeats: [draw(rng) for _ in range(repeats)]


def _lognormal(log_sd: float) -> tuple[DrawCase, float]:
    """Lognormal values with a median of 1; their mean is ``exp(log_sd² / 2)``."""
    return _independent(lambda rng: rng.lognormvariate(0.0, log_sd)), math.exp(0.5 * log_sd * log_sd)


def _repeated_case(rng: random.Random, repeats: int) -> list[float]:
    """A case's level is lognormal (log-SD 1.0) and each repeat varies about it (log-SD 0.5)."""
    level = rng.lognormvariate(0.0, 1.0)
    return [level * rng.lognormvariate(0.0, 0.5) for _ in range(repeats)]


_REPEATED_MEAN = math.exp(0.5 + 0.125)


def _sometimes_free(rng: random.Random, repeats: int) -> list[float]:
    """A fifth of cases cost nothing on every repeat; the rest are lognormal at log-SD 1.0."""
    if rng.random() < 0.2:
        return [0.0] * repeats
    return [rng.lognormvariate(0.0, 1.0) for _ in range(repeats)]


#: name -> (one case's draw, the true mean, repeats per case, the chance a repeat after the first is lost).
SHAPES: dict[str, tuple[DrawCase, float, int, float]] = {
    "lognormal-0.5": (*_lognormal(0.5), 1, 0.0),
    "lognormal-1.0": (*_lognormal(1.0), 1, 0.0),
    "lognormal-1.5": (*_lognormal(1.5), 1, 0.0),
    "gamma-2": (_independent(lambda rng: rng.gammavariate(2.0, 1.0)), 2.0, 1, 0.0),
    "exponential": (_independent(lambda rng: rng.expovariate(1.0)), 1.0, 1, 0.0),
    # Folded at zero so every draw is a spend; at SD 0.3 a draw below zero is 4 in 10,000 and the mean moves by
    # under 1e-4 of itself.
    "normal": (_independent(lambda rng: abs(rng.gauss(1.0, 0.3))), 1.0, 1, 0.0),
    "repeated-3": (_repeated_case, _REPEATED_MEAN, 3, 0.0),
    "repeated-3-some-lost": (_repeated_case, _REPEATED_MEAN, 3, 0.3),
    "a-fifth-free": (_sometimes_free, 0.8 * math.exp(0.5), 1, 0.0),
}


def _engine(values: list[float], cases: list[int]) -> Interval:
    return observed_mean_interval(values, cases=cases, floor=0.0, skewed=True)


def _clipped_t(values: list[float], cases: list[int]) -> Interval:
    sem = clustered_standard_error(values, cases)
    assert sem is not None
    return mean_interval(sum(values) / len(values), sem, len(set(cases)), floor=0.0)


def _coverage(shape: str, n_cases: int, interval: Callable[[list[float], list[int]], Interval]) -> float:
    """The share of replicates whose interval holds the true mean, over the replicates that state one."""
    draw, truth, repeats, lost = SHAPES[shape]
    rng = random.Random(f"floored-mean-{shape}-{n_cases}")
    covered = stated = 0
    for _ in range(REPLICATES):
        values: list[float] = []
        cases: list[int] = []
        for case in range(n_cases):
            for index, value in enumerate(draw(rng, repeats)):
                if index and rng.random() < lost:
                    continue
                values.append(value)
                cases.append(case)
        found = interval(values, cases)
        if found is None:
            continue
        stated += 1
        covered += found[0] <= truth <= found[1]
    assert stated >= 0.95 * REPLICATES, f"{shape} at {n_cases} cases stated an interval {stated} times"
    return covered / stated


#: Every shape at every case count, less the one that rarely states an interval: five cases of which a fifth are
#: free hold five above the floor a third of the time.
CONFIGURATIONS = [
    (shape, n_cases) for shape in sorted(SHAPES) for n_cases in CASE_COUNTS if (shape, n_cases) != ("a-fifth-free", 5)
]


@pytest.mark.parametrize(("shape", "n_cases"), CONFIGURATIONS)
def test_the_interval_on_a_skewed_mean_covers_at_its_level(shape: str, n_cases: int) -> None:
    coverage = _coverage(shape, n_cases, _engine)
    assert coverage >= at_least(INTERVAL_LEVEL, REPLICATES), f"{shape} at {n_cases} cases covered {coverage:.3f}"


@pytest.mark.parametrize(("shape", "n_cases", "ceiling"), [("lognormal-1.0", 10, 0.88), ("lognormal-1.5", 10, 0.78)])
def test_the_clipped_t_interval_would_fail_this_check(shape: str, n_cases: int, ceiling: float) -> None:
    """The interval this one replaced, on the same draws: 83% and 72% in the review, far under the 92.2% bound."""
    coverage = _coverage(shape, n_cases, _clipped_t)
    assert coverage < ceiling < at_least(INTERVAL_LEVEL, REPLICATES)


def test_the_interval_holds_the_mean_and_stays_above_the_floor() -> None:
    rng = random.Random("floored-mean-shape")
    for _ in range(200):
        n_cases = rng.randint(5, 12)
        cases = [case for case in range(n_cases) for _ in range(rng.randint(1, 3))]
        values = [0.0 if rng.random() < 0.15 else rng.lognormvariate(0.0, 1.0) for _ in cases]
        found = _engine(values, cases)
        if found is None:
            continue
        low, high = found
        assert 0.0 <= low <= sum(values) / len(values) <= high
        assert math.isfinite(high)


def test_it_is_never_narrower_than_the_clipped_t_interval() -> None:
    """A union with the t interval: right as the cases grow whatever their distribution."""
    rng = random.Random("floored-mean-union")
    for _ in range(100):
        values = [rng.expovariate(1.0) for _ in range(12)]
        cases = list(range(12))
        found, symmetric = _engine(values, cases), _clipped_t(values, cases)
        assert found is not None and symmetric is not None
        assert found[0] <= symmetric[0] and found[1] >= symmetric[1]


@pytest.mark.parametrize("n_cases", [2, 3, 4])
def test_under_five_differing_cases_state_no_interval(n_cases: int) -> None:
    """Bounds that hold their level on two to four skewed values span orders of magnitude; none is stated."""
    assert _engine([1.0 + case for case in range(n_cases)], list(range(n_cases))) is None


def test_under_five_cases_above_the_floor_state_no_interval() -> None:
    assert _engine([0.0, 0.0, 0.0, 9.0], [0, 1, 2, 3]) is None
    assert _engine([0.0, 0.0, 1.0, 2.0, 3.0, 4.0], list(range(6))) is None
    assert _engine([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0], list(range(7))) is not None


def test_equal_values_keep_an_interval_of_no_width() -> None:
    assert _engine([0.25] * 3, list(range(3))) == (0.25, 0.25)
    assert _engine([0.0] * 6, list(range(6))) == (0.0, 0.0)


def test_a_value_below_the_floor_takes_the_clipped_t_interval() -> None:
    """A host's defect, named by the bundle; the log scale cannot read it."""
    values, cases = [-3.0, 1.0, 2.0, 4.0, 5.0, 6.0], list(range(6))
    assert _engine(values, cases) == _clipped_t(values, cases)


def test_a_spread_past_a_float_stops_at_the_largest_one() -> None:
    found = _engine([1e-300, 1e-250, 1e-200, 1e-100, 1.0], list(range(5)))
    assert found is not None and math.isfinite(found[1])


def test_a_count_keeps_the_clipped_t_interval() -> None:
    """Only a spend or a time takes the wider interval; a count is not checked on skewed values (open problems)."""
    values, cases = [0.0, 0.0, 0.0, 9.0], [0, 1, 2, 3]
    assert observed_mean_interval(values, cases=cases, floor=0.0) == _clipped_t(values, cases)
