"""Looks, checked against data with no true difference: reading again after every batch errs past α, a plan holds it.

A look is one more reading of a contrast after its data changed (judges design §6). Each row below draws two arms with
no true difference (or, for the floor, a true drop of exactly the margin), reads the contrast at every look and stops
at the first separation (or the first ``held``), and counts how often that happens. Every row is seeded and runs 4,000
replicates; each bound is the nominal rate plus four Monte-Carlo standard errors (:func:`at_most`).

| Set-up | Undeclared (α at every look) | Declared (α/L) |
|---|---|---|
| Cases grow 20 to 60 over 1, 2, 3, 4, 5 looks, mean on t | 5.3%, 9.7%, 11.2%, 12.5%, 12.9% | 5.0%, 3.9%, 2.9% (1, 3, 5 looks): at most 5% |
| Repeats grow on 20 fixed cases, 3 looks | 11.1% | 4.0%: at most 5% |
| A pass rate on the bounded interval, 3 looks | 1.9% | 0.4%: at most 5% |
| Unequal batches (20, +5, +35) | 11.4% | 3.6%: at most 5% |
| A floor read ``held`` at any look, true drop equal to the margin, 3 looks | 1.8% | 0.45%: at most 2.5% |
| Looks run to 2L (planned 3, taken 6) | n/a | 6.1%, against 4.2% within the plan: the guarantee is gone |

The bounded interval a rate and a floor read is conservative at these counts, so even read at every look they stay
under their nominal rate here; the plan still holds them, and the mean on t, which is exact, shows what looking costs.

The rule each row reads is the engine's: a mean's verdict is the family rule
(:func:`~packages.evals.tests.simulation_support.family_verdicts`, which ``test_simulated_multiple_comparisons.py`` pins
to the bundle) at the level :func:`~threetears.evals.analysis.significance.looks_alpha` gives; a rate's separation
needs its adjusted p under that level and its bounded interval at ``1 − level`` clear of zero, the bundle's rule for a
rate; a floor's ``held`` is :func:`~threetears.evals.analysis.floors.read_floor` over a rate carrying the plan. That
the bundle reads its family at α/L, and its floors under the plan, is held in ``test_looks.py``. The undeclared rows
pin :data:`~threetears.evals.analysis.significance.UNPLANNED_FALSE_SEPARATION`, the rate a caveat on unplanned looks
can state, and the reading the plan replaced is shown to fail its bar.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence

import pytest

from threetears.evals.analysis.floors import FloorReadings, RateReading, read_floor
from threetears.evals.analysis.significance import UNPLANNED_FALSE_SEPARATION, looks_alpha
from threetears.evals.analysis.stats import (
    INTERVAL_LEVEL,
    SIGNIFICANCE_ALPHA,
    bounded_difference_interval,
    interval_permits_separation,
    separation_p,
)
from threetears.evals.kernel.declaration import Floor, RateDrop
from packages.evals.tests.simulation_support import at_most, family_verdicts, within

REPLICATES = 4000
#: A floor's one tail at the nominal level: how often a floor at its margin may be read held, at most.
ONE_TAIL = (1 - INTERVAL_LEVEL) / 2

#: A batch: each arm's per-case values at one look, keyed by case.
Look = tuple[dict[str, float], dict[str, float]]
#: Draws the looks of one replicate.
Draw = Callable[[random.Random], list[Look]]


def _sizes(looks: int, first: int = 20, last: int = 60) -> list[int]:
    """Cases read at each of ``looks`` looks, ``first`` to ``last`` in equal batches."""
    if looks == 1:
        return [last]
    return [first + round((last - first) * index / (looks - 1)) for index in range(looks)]


def _growing_cases(sizes: Sequence[int]) -> Draw:
    """Paired normal per-case values, no true difference, read on the first ``size`` cases at each look."""

    def draw(rng: random.Random) -> list[Look]:
        n = sizes[-1]
        shared = [rng.gauss(0.0, 1.0) for _ in range(n)]
        control = {f"c{i}": shared[i] + rng.gauss(0.0, 1.0) for i in range(n)}
        candidate = {f"c{i}": shared[i] + rng.gauss(0.0, 1.0) for i in range(n)}
        return [
            ({f"c{i}": control[f"c{i}"] for i in range(size)}, {f"c{i}": candidate[f"c{i}"] for i in range(size)})
            for size in sizes
        ]

    return draw


def _growing_repeats(cases: int = 20, looks: int = 3) -> Draw:
    """Fixed cases, one more repeat per arm at each look; each case's value is the mean of its repeats so far."""

    def draw(rng: random.Random) -> list[Look]:
        levels = [rng.gauss(0.0, 1.0) for _ in range(cases)]
        sides = [[[level + rng.gauss(0.0, 1.0) for _ in range(looks)] for level in levels] for _ in range(2)]
        return [
            tuple(  # type: ignore[misc]  # a pair, one per side
                {f"c{i}": sum(repeats[: look + 1]) / (look + 1) for i, repeats in enumerate(side)} for side in sides
            )
            for look in range(looks)
        ]

    return draw


def _rate_shares(rng: random.Random, n: int, *, broken: float = 0.0, trials: int = 3, p: float = 0.8) -> list[float]:
    """Per-case shares of ``trials`` trials passed at ``p``, a case broken outright with chance ``broken``."""
    return [0.0 if rng.random() < broken else sum(rng.random() < p for _ in range(trials)) / trials for _ in range(n)]


def _mean_separates(look: Look, alpha: float) -> bool:
    (verdict,) = family_verdicts([(look[0], look[1], True)], alpha=alpha)
    return verdict.verdict in ("improved", "regressed")


def _rate_separates(look: Look, alpha: float) -> bool:
    """A rate's separation as the bundle reads one: its p under the level, and its bounded interval clear of zero."""
    control, candidate = list(look[0].values()), list(look[1].values())
    p = separation_p(control, candidate, paired=True, value_range=(0.0, 1.0))
    if p is None or p >= alpha or sum(candidate) == sum(control):
        return False
    interval = bounded_difference_interval(
        control, candidate, paired=True, value_range=(0.0, 1.0), confidence=1.0 - alpha
    )
    return interval is not None and interval_permits_separation(interval)


def _first_separation_rate(draw: Draw, separates: Callable[[Look, float], bool], alpha: float, seed: str) -> float:
    """How often a replicate separates at some look, read at ``alpha`` at every one and stopped at the first."""
    rng = random.Random(seed)
    hits = sum(any(separates(look, alpha) for look in draw(rng)) for _ in range(REPLICATES))
    return hits / REPLICATES


# --- the mean on t, cases growing ----------------------------------------------------------------------------------


@pytest.mark.parametrize("looks", sorted(UNPLANNED_FALSE_SEPARATION))
def test_reading_at_alpha_at_every_look_errs_at_the_rate_the_caveat_states(looks: int) -> None:
    """The undeclared rate, measured: it is what :data:`UNPLANNED_FALSE_SEPARATION` states for that count."""
    rate = _first_separation_rate(
        _growing_cases(_sizes(looks)), _mean_separates, SIGNIFICANCE_ALPHA, seed=f"undeclared-{looks}"
    )
    assert within(rate, UNPLANNED_FALSE_SEPARATION[looks], REPLICATES), (looks, rate)


@pytest.mark.parametrize("looks", [1, 3, 5])
def test_a_declared_plan_holds_the_error_at_alpha_as_cases_grow(looks: int) -> None:
    rate = _first_separation_rate(
        _growing_cases(_sizes(looks)), _mean_separates, looks_alpha(looks), seed=f"declared-{looks}"
    )
    assert rate <= at_most(SIGNIFICANCE_ALPHA, REPLICATES), (looks, rate)


def test_the_simulation_catches_the_reading_a_plan_replaced() -> None:
    """Five looks each read at α, the reading before a plan could be declared: well past the 5% it states."""
    rate = _first_separation_rate(_growing_cases(_sizes(5)), _mean_separates, SIGNIFICANCE_ALPHA, seed="declared-5")
    assert rate > at_most(SIGNIFICANCE_ALPHA, REPLICATES)


def test_unequal_batches_under_a_plan_hold_alpha_and_without_one_do_not() -> None:
    sizes = [20, 25, 60]
    declared = _first_separation_rate(_growing_cases(sizes), _mean_separates, looks_alpha(3), seed="unequal")
    undeclared = _first_separation_rate(_growing_cases(sizes), _mean_separates, SIGNIFICANCE_ALPHA, seed="unequal")
    assert declared <= at_most(SIGNIFICANCE_ALPHA, REPLICATES)
    assert undeclared > at_most(SIGNIFICANCE_ALPHA, REPLICATES)


def test_repeats_growing_on_fixed_cases_under_a_plan_hold_alpha() -> None:
    declared = _first_separation_rate(_growing_repeats(), _mean_separates, looks_alpha(3), seed="repeats")
    undeclared = _first_separation_rate(_growing_repeats(), _mean_separates, SIGNIFICANCE_ALPHA, seed="repeats")
    assert declared <= at_most(SIGNIFICANCE_ALPHA, REPLICATES)
    assert undeclared > declared


def test_a_pass_rate_on_its_bounded_interval_under_a_plan_holds_alpha() -> None:
    def draw(rng: random.Random) -> list[Look]:
        control, candidate = _rate_shares(rng, 60), _rate_shares(rng, 60)
        return [({f"c{i}": control[i] for i in range(n)}, {f"c{i}": candidate[i] for i in range(n)}) for n in _sizes(3)]

    declared = _first_separation_rate(draw, _rate_separates, looks_alpha(3), seed="rate")
    undeclared = _first_separation_rate(draw, _rate_separates, SIGNIFICANCE_ALPHA, seed="rate")
    assert declared <= at_most(SIGNIFICANCE_ALPHA, REPLICATES)
    assert undeclared > declared


def test_looks_run_past_the_plan_lose_its_guarantee() -> None:
    """Planned three, taken six: each still read at α/3, and the error is past what the plan stated."""
    sizes = [20, 40, 60, 80, 100, 120]
    rate = _first_separation_rate(_growing_cases(sizes), _mean_separates, looks_alpha(3), seed="past-plan")
    within_plan = _first_separation_rate(_growing_cases(sizes[:3]), _mean_separates, looks_alpha(3), seed="past-plan")
    assert rate > within_plan


# --- a floor read held, its true drop at the margin ------------------------------------------------------------------

#: The floor: no pass rate drops by more than 9.5 points. The candidate breaks one case in ten outright and passes the
#: rest as the control does (each trial at 0.95), so its true rate is 9.5 points below the control's.
_FLOOR = Floor(aspect="b", clause=RateDrop(max_points=9.5))


def _held_rate(looks: int | None, seed: str) -> float:
    """How often the floor reads held at some look, cases growing 20 to 60 over three looks, the rate carrying
    ``looks`` as the plan the floor is read under."""
    rng = random.Random(seed)
    held = 0
    for _ in range(REPLICATES):
        control = _rate_shares(rng, 60, p=0.95)
        candidate = _rate_shares(rng, 60, p=0.95, broken=0.1)
        for n in _sizes(3):
            a, b = tuple(control[:n]), tuple(candidate[:n])
            rate = RateReading(
                "the pass rate",
                round(sum(a) * 3),
                n * 3,
                round(sum(b) * 3),
                n * 3,
                interval=bounded_difference_interval(
                    a, b, paired=True, value_range=(0.0, 1.0), confidence=INTERVAL_LEVEL
                ),
                samples=(a, b),
                confidence=INTERVAL_LEVEL,
                looks=looks,
            )
            readings = FloorReadings(rates=(rate,), control_cases=n, candidate_cases=n)
            if read_floor(_FLOOR, readings, control="control", candidate="candidate").outcome == "held":
                held += 1
                break
    return held / REPLICATES


def test_a_floor_at_its_margin_under_a_plan_is_read_held_no_more_often_than_its_tail() -> None:
    assert _held_rate(3, seed="floor") <= at_most(ONE_TAIL, REPLICATES)


def test_a_floor_read_held_at_every_look_with_no_plan_is_held_more_often_than_under_one() -> None:
    """The reading the plan tightens: read at the nominal level at every look, held more often than under the plan
    (1.8% against 0.45% at this seed; the bounded interval's conservatism keeps both under the 2.5% tail here)."""
    assert _held_rate(None, seed="floor") > _held_rate(3, seed="floor")
