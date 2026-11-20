"""The interval on a mean cost or latency covers the truth at its stated level on skewed values.

Seeded simulation with a known truth. A spend and a time are bounded below only and right-skewed: a few long
conversations cost several times the median. Each replicate draws cases from a distribution whose mean
is known, asks the engine for the interval it reports
(:func:`~threetears.evals.analysis.stats.observed_mean_interval` with a floor of zero and ``skewed``, the call the
bundle makes for a measure on the cost or latency axis, or in dollars or a unit of time) and checks whether it holds the true mean.

**What is held, and on which shapes only.** At 5, 10, 20 and 50 cases the interval covers the true mean at least
95% of the time, less the Monte-Carlo error (a replicate that states no interval counts as not covering), on:

- lognormal values at log-SD 0.5, 1.0 and 1.5 (at 1.5 the 95th percentile is about twelve times the median);
- gamma (shape 2), exponential and normal (SD 0.3 of the mean) values, which are not lognormal;
- cases repeated three times, where the repeats of a case share its level, with and without lost repeats;
- cases of which a fifth cost nothing (from 10 cases: at 5, five above the floor are rare, and under five there
  is no interval).

Pooled over those configurations it is held to 95% itself.

**Where it does not hold** (:class:`TestWhereItDoesNotHold`, a strict xfail): a cost where one case in ten is 30
times the rest. There the interval is the t interval and covers as it does, about 40%, 66% and 89% at 5, 10 and 30
cases. No interval on a mean holds its level where rare large cases dominate.

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

import functools
import math
import random
from collections.abc import Callable
from typing import NamedTuple

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


def _sometimes_free(share: float) -> DrawCase:
    """``share`` of cases cost nothing on every repeat; the rest are lognormal at log-SD 1.0."""

    def draw(rng: random.Random, repeats: int) -> list[float]:
        if rng.random() < share:
            return [0.0] * repeats
        return [rng.lognormvariate(0.0, 1.0) for _ in range(repeats)]

    return draw


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
    "a-fifth-free": (_sometimes_free(0.2), 0.8 * math.exp(0.5), 1, 0.0),
    "four-fifths-free": (_sometimes_free(0.8), 0.2 * math.exp(0.5), 1, 0.0),
    # Not claimed: one case in ten costs 30 times the rest (the rest vary by a tenth). See TestWhereItDoesNotHold.
    "one-in-ten-30x": (
        _independent(lambda rng: rng.lognormvariate(0.0, 0.1) * (30.0 if rng.random() < 0.1 else 1.0)),
        math.exp(0.005) * (0.9 + 0.1 * 30.0),
        1,
        0.0,
    ),
}


def _engine(values: list[float], cases: list[int]) -> Interval:
    return observed_mean_interval(values, cases=cases, floor=0.0, skewed=True)


def _clipped_t(values: list[float], cases: list[int]) -> Interval:
    sem = clustered_standard_error(values, cases)
    assert sem is not None
    return mean_interval(sum(values) / len(values), sem, len(set(cases)), floor=0.0)


class Tally(NamedTuple):
    """What the replicates of one configuration did: how many stated an interval, held the truth, had no width."""

    stated: int
    covered: int
    zero_width: int


@functools.cache
def _tally(shape: str, n_cases: int, interval: Callable[[list[float], list[int]], Interval]) -> Tally:
    draw, _, repeats, lost = SHAPES[shape]
    truth = SHAPES[shape][1]
    rng = random.Random(f"floored-mean-{shape}-{n_cases}")
    stated = covered = zero_width = 0
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
        zero_width += found[0] == found[1]
    return Tally(stated, covered, zero_width)


def _coverage(shape: str, n_cases: int, interval: Callable[[list[float], list[int]], Interval]) -> float:
    """The share of ALL replicates whose interval holds the true mean: a replicate that states none holds nothing."""
    return _tally(shape, n_cases, interval).covered / REPLICATES


#: The shapes the level is claimed on, at every case count, less one: five cases of which a fifth are free hold
#: five above the floor a third of the time, and state no interval otherwise.
HELD = sorted(set(SHAPES) - {"four-fifths-free", "one-in-ten-30x"})
CONFIGURATIONS = [
    (shape, n_cases) for shape in HELD for n_cases in CASE_COUNTS if (shape, n_cases) != ("a-fifth-free", 5)
]


@pytest.mark.parametrize(("shape", "n_cases"), CONFIGURATIONS)
def test_the_interval_on_a_skewed_mean_covers_at_its_level(shape: str, n_cases: int) -> None:
    coverage = _coverage(shape, n_cases, _engine)
    assert coverage >= at_least(INTERVAL_LEVEL, REPLICATES), f"{shape} at {n_cases} cases covered {coverage:.3f}"


def test_pooled_over_the_shapes_it_is_claimed_on_it_covers_at_least_its_level() -> None:
    """Each configuration's bound (92.2%) would pass a method that truly held 93%. Pooled, 35,000 replicates have a
    standard error of 0.0012, and the union runs wide, so the pooled coverage is held to 95% itself."""
    covered = sum(_tally(shape, n_cases, _engine).covered for shape, n_cases in CONFIGURATIONS)
    assert covered / (REPLICATES * len(CONFIGURATIONS)) >= INTERVAL_LEVEL


@pytest.mark.parametrize(("shape", "n_cases", "ceiling"), [("lognormal-1.0", 10, 0.88), ("lognormal-1.5", 10, 0.78)])
def test_the_clipped_t_interval_would_fail_this_check(shape: str, n_cases: int, ceiling: float) -> None:
    """The interval this one replaced, on the same draws: 83% and 72% in the review, far under the 92.2% bound."""
    coverage = _coverage(shape, n_cases, _clipped_t)
    assert coverage < ceiling < at_least(INTERVAL_LEVEL, REPLICATES)


class TestWhereItDoesNotHold:
    """One case in ten costs 30 times the rest. The log-scale interval bounds ``exp(mean of logs + variance of logs
    / 2)``, the mean of a lognormal only; here it sits below the mean, the union is the t interval, and the coverage
    is the t interval's: about 40%, 66% and 89% at 5, 10 and 30 cases, 94% at 100. No interval on a mean holds its
    level where rare large cases the sample may not contain dominate the cost (``docs/open-problems.md``)."""

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="no interval on a mean holds its level when rare large cases dominate the cost (open-problems.md)",
    )
    @pytest.mark.parametrize("n_cases", [5, 10, 20])
    def test_a_cost_dominated_by_rare_large_cases_is_covered_at_the_level(self, n_cases: int) -> None:
        assert _coverage("one-in-ten-30x", n_cases, _engine) >= at_least(INTERVAL_LEVEL, REPLICATES)

    @pytest.mark.parametrize("n_cases", [5, 10, 20])
    def test_there_it_covers_exactly_as_the_clipped_t_interval_does(self, n_cases: int) -> None:
        """The limit is stated as "the t interval's coverage", so that is pinned, not only that it falls short."""
        assert (
            _tally("one-in-ten-30x", n_cases, _engine).covered == _tally("one-in-ten-30x", n_cases, _clipped_t).covered
        )


class TestMostCasesFree:
    """Four cases in five cost nothing. Five cases are all free a third of the time, and a ``[0, 0]`` there covered
    nothing; under five cases above zero no interval is stated, all free included."""

    @pytest.mark.parametrize("n_cases", [5, 10, 20, 50])
    def test_no_interval_of_no_width_is_ever_stated(self, n_cases: int) -> None:
        assert _tally("four-fifths-free", n_cases, _engine).zero_width == 0

    def test_the_intervals_it_does_state_cover_at_the_level(self) -> None:
        tally = _tally("four-fifths-free", 50, _engine)
        assert tally.stated >= REPLICATES // 2
        assert tally.covered / tally.stated >= at_least(INTERVAL_LEVEL, tally.stated)

    def test_the_clipped_t_interval_states_a_zero_that_never_covers(self) -> None:
        """The failure this guards, on the same draws: about a third of 5-case samples."""
        tally = _tally("four-fifths-free", 5, _clipped_t)
        assert tally.zero_width >= REPLICATES // 4


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


def test_values_all_equal_above_the_floor_keep_an_interval_of_no_width() -> None:
    """As any equal values do; a limit, recorded in open-problems."""
    assert _engine([0.25] * 3, list(range(3))) == (0.25, 0.25)


def test_every_case_at_the_floor_states_no_interval() -> None:
    """A ``[0, 0]`` at 95% says a free call is never followed by a paid one, which six free calls cannot show."""
    assert _engine([0.0] * 6, list(range(6))) is None


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
