"""Looks, checked against data with no true difference: reading again after every batch errs past α, a plan holds it.

A look is an occasion on which a contrast's test could have been read again on more or different evidence (judges
design §6). Each row below draws two arms with no true difference (or, for a floor, a true drop of exactly its margin),
reads the contrast at every look and stops at the first separation (or the first ``held``), and counts how often that
happens. Every row is seeded; each bound is the nominal rate plus four Monte-Carlo standard errors (:func:`at_most`),
and each measured rate is pinned within four of its own (:func:`within`), so a wrong correction moves it off its pin.

**Batches fixed in advance** (4,000 replicates a row):

| Set-up | Undeclared (α at every look) | Declared (α/L) | Bar |
|---|---|---|---|
| Cases grow 20 to 60 over 1, 2, 3, 4, 5 looks, mean on t | 5.3, 9.7, 11.2, 12.5, 12.9% | 5.0, 3.9, 2.9% (1, 3, 5 looks) | 5% |
| Repeats grow on 20 fixed cases, 3 looks | 11.1% | 4.0% | 5% |
| Unequal batches (20, +5, +35) | 11.4% | 3.6% | 5% |
| A pass rate on the bounded interval, 3 looks | 1.9% | 0.4% | 5% (pinned, see below) |
| A rate floor read held, true drop at the margin, 3 looks | 1.8% | 0.45% | 2.5% (pinned, see below) |
| A cost floor read held, equal true cost, 3 looks, t interval | 5.65% | 2.2% | 2.5% |
| Looks past the plan, cases 20 to 120 in batches of 20, 1 to 6 looks, planned 2 | | 2.6, 4.7, 6.1, 7.2, 8.0, 9.0% | 5% within it |
| The same, planned 3 | | 1.5, 2.7, 3.5, 4.2, 4.9, 5.4% | 5% within it |

The bounded interval a rate and a rate floor read is conservative at these counts: read at the nominal level at every
look, they stay under their bars, so those two rows cannot fail on their bars for a missing correction. Each is held
by its pins instead (the uncorrected rate and the corrected one each within four standard errors of what was measured,
four standard errors apart), and the cost floor, on the exact t interval, is the row whose bar the uncorrected reading
fails and the plan holds.

**Batches sized by the planner from the reading before** (10,000 replicates a row; :data:`_PLANNER_EXTENDS`): a
per-case pass rate of three trials at 0.8 or 0.5, paired, launched at 20 or 40 cases. At each look the campaign row's
rule reads the change (one aspect, so its step-down is the bounded interval at ``1 − level`` clear of zero); where it
does not separate, the planner (a parameter: today's, :func:`todays_planner`) sizes the cases to add, and they are
added (at most :data:`EXTEND_CAP` a step: a host's corpus is finite); where it sizes none, the campaign stops. Each
cell is how often a replicate has separated after 0, 1, 2 and 3 extends:

| Pass rate, launched | Undeclared | Declared L = 2 | Declared L = 3 |
|---|---|---|---|
| 0.8, 20 cases | 0.05, 2.1, 4.4, 6.5% | 0, 1.0, 2.1, 3.0% | 0, 0.5, 1.4, 2.1% |
| 0.8, 40 cases | 0.7, 3.3, 5.7, 7.9% | 0.2, 1.5, 2.8, 3.8% | 0.1, 0.8, 1.7, 2.4% |
| 0.5, 20 cases | 0.2, 3.2, 5.9, 7.9% | 0.1, 1.3, 2.9, 4.1% | 0, 0.7, 1.6, 2.3% |
| 0.5, 40 cases | 1.7, 4.8, 7.3, 9.2% | 0.6, 2.0, 3.5, 4.5% | 0.3, 1.25, 2.1, 2.9% |
| Cost floor held, equal true cost, 20 cases (bar 2.5%) | 2.6, 5.0, 6.4, 7.8% | 1.2, 2.4, 3.4, 4.0% | 0.9, 1.9, 2.4, 2.9% |

Every declared row holds its bar within its plan (L − 1 extends): data-sized extends, counted as looks and read at α/L,
err no more than fixed batches do here, so a caveat raises nothing for looks within a plan. Without a plan the error
grows with each extend; past the plan it grows too. The rates the caveat quotes, the largest over these settings, are
:data:`~threetears.evals.analysis.significance.FALSE_SEPARATION`, and
:func:`test_the_table_the_caveat_quotes_is_the_measured_one` holds them to the rows here.

The rule each row reads is the engine's: a mean's verdict is the family rule
(:func:`~packages.evals.tests.simulation_support.family_verdicts`, which ``test_simulated_multiple_comparisons.py`` pins
to the bundle) at the level :func:`~threetears.evals.analysis.significance.looks_alpha` gives; a rate's separation
needs its adjusted p under that level and its bounded interval at ``1 − level`` clear of zero, the bundle's rule for a
rate; a floor's ``held`` is :func:`~threetears.evals.analysis.floors.read_floor` over a rate carrying the plan, or over
cost intervals built as :func:`~threetears.evals.analysis.floors.floor_readings` builds them. That the bundle reads
its family at α/L, and its floors under the plan, is held in ``test_looks.py``.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from statistics import fmean, stdev

import pytest

from threetears.evals.analysis.floors import FloorReadings, RateReading, read_floor
from threetears.evals.analysis.separation import POWER, NoSpread, cases_to_separate
from threetears.evals.analysis.significance import FALSE_SEPARATION, looks_alpha, looks_confidence
from threetears.evals.analysis.stats import (
    INTERVAL_LEVEL,
    SIGNIFICANCE_ALPHA,
    bounded_difference_interval,
    bounded_mean_p,
    difference_interval,
    interval_permits_separation,
    separation_p,
)
from threetears.evals.kernel.declaration import CostLower, Floor, RateDrop
from packages.evals.tests.simulation_support import at_most, family_verdicts, within

REPLICATES = 4000
#: Replicates of a planner-sized row: the rate the caveat quotes is read off these.
EXTEND_REPLICATES = 10_000

#: How often a test with no true difference separates at some look, read at α at every look and stopped at the first
#: separation, by the looks taken: the mean of 20 to 60 cases on the t test, the cases added in equal batches, measured
#: here at each row's seed. It only grows past the last count.
UNPLANNED_FALSE_SEPARATION = {1: 0.05, 2: 0.097, 3: 0.112, 4: 0.125, 5: 0.129}
#: A floor's one tail at the nominal level: how often a floor at its margin may be read held, at most.
ONE_TAIL = (1 - INTERVAL_LEVEL) / 2
#: The most cases one extend adds: a host's corpus is finite, and the planner's size with no true difference is often
#: thousands (the estimate it sizes from sits near zero).
EXTEND_CAP = 100

#: A batch: each arm's per-case values at one look, keyed by case.
Look = tuple[dict[str, float], dict[str, float]]
#: Draws the looks of one replicate.
Draw = Callable[[random.Random], list[Look]]
#: Sizes the cases an unclear look adds: from the paired per-case differences read so far and the two-sided level the
#: next look reads at, the cases to add; None where it can size none (the campaign then stops).
Planner = Callable[[Sequence[float], float], int | None]


def todays_planner(differences: Sequence[float], confidence: float) -> int | None:
    """The planner on this base: the caveat step's sizing (``caveats._settling``), the cases for a two-sided interval at
    ``confidence`` centred on today's estimate to clear zero four times in five
    (:func:`~threetears.evals.analysis.separation.cases_to_separate`, ``sided=2``, at
    :data:`~threetears.evals.analysis.separation.POWER`), less the cases held, at least one. Its spread is the paired
    differences' own; the step reads a rate's spread off its bounded interval, wider at small counts, so it asks for
    more cases than this, which a cap reaches sooner. Chunk M0's exact-sizing planner replaces it here once built."""
    if len(differences) < 2:
        return None
    try:
        needed = cases_to_separate(stdev(differences), fmean(differences), confidence, sided=2, power=POWER)
    except NoSpread:
        return None
    return max(needed - len(differences), 1)


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
    """The undeclared rate, measured: it is what :data:`UNPLANNED_FALSE_SEPARATION` states for that count, and it grows
    with every look."""
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
    assert undeclared > at_most(SIGNIFICANCE_ALPHA, REPLICATES)


#: The pass-rate row's measured rates, undeclared then declared at three looks: four standard errors apart at least,
#: so a correction that does nothing reads the undeclared rate where the declared one is pinned.
_RATE_ROW = (0.019, 0.004)


def test_a_pass_rate_on_its_bounded_interval_under_a_plan_holds_alpha_and_is_moved_by_it() -> None:
    """The bounded interval is conservative at 20 to 60 cases, so even read at α at every look this row stays under
    5%: it is held by its pins, each side within four standard errors of what was measured."""

    def draw(rng: random.Random) -> list[Look]:
        control, candidate = _rate_shares(rng, 60), _rate_shares(rng, 60)
        return [({f"c{i}": control[i] for i in range(n)}, {f"c{i}": candidate[i] for i in range(n)}) for n in _sizes(3)]

    declared = _first_separation_rate(draw, _rate_separates, looks_alpha(3), seed="rate")
    undeclared = _first_separation_rate(draw, _rate_separates, SIGNIFICANCE_ALPHA, seed="rate")
    assert declared <= at_most(SIGNIFICANCE_ALPHA, REPLICATES)
    assert within(undeclared, _RATE_ROW[0], REPLICATES), undeclared
    assert within(declared, _RATE_ROW[1], REPLICATES), declared
    assert not within(_RATE_ROW[0], _RATE_ROW[1], REPLICATES), "the pins cannot tell a no-op correction apart"


#: Looks run past the plan, cases 20 to 120 in batches of 20, each read at α/L: how often a replicate has separated by
#: each look, by the looks planned. Within the plan at most α; past it the guarantee is gone.
_PAST_PLAN: Mapping[int, Mapping[int, float]] = {
    2: {1: 0.0255, 2: 0.047, 3: 0.0605, 4: 0.07225, 5: 0.07975, 6: 0.08975},
    3: {1: 0.01525, 2: 0.02675, 3: 0.035, 4: 0.042, 5: 0.04925, 6: 0.05375},
}


def _cumulative(
    draw: Draw, separates: Callable[[Look, float], bool], alpha: float, seed: str, looks: int
) -> list[float]:
    """How often a replicate has separated by each look, read at ``alpha`` at every one."""
    rng = random.Random(seed)
    hits = [0] * looks
    for _ in range(REPLICATES):
        for index, look in enumerate(draw(rng)):
            if separates(look, alpha):
                for later in range(index, looks):
                    hits[later] += 1
                break
    return [count / REPLICATES for count in hits]


@pytest.mark.parametrize("planned", [2, 3])
def test_looks_run_past_the_plan_lose_its_guarantee(planned: int) -> None:
    """Within the plan the error holds at α; past it, it grows, read at each count."""
    sizes = [20, 40, 60, 80, 100, 120]
    rates = _cumulative(
        _growing_cases(sizes), _mean_separates, looks_alpha(planned), seed=f"past-plan-{planned}", looks=len(sizes)
    )
    assert rates[planned - 1] <= at_most(SIGNIFICANCE_ALPHA, REPLICATES), rates
    assert rates[-1] > rates[planned - 1], rates
    measured = _PAST_PLAN.get(planned, {})
    assert [round(rate, 4) for rate in rates] == pytest.approx(
        [measured.get(taken, -1.0) for taken in range(1, len(sizes) + 1)], abs=4e-3
    ), ("measured", planned, rates)


# --- floors read held, the true drop at the margin -------------------------------------------------------------------

#: The floor: no pass rate drops by more than 9.5 points. The candidate breaks one case in ten outright and passes the
#: rest as the control does (each trial at 0.95), so its true rate is 9.5 points below the control's.
_FLOOR = Floor(aspect="b", clause=RateDrop(max_points=9.5))
#: The rate floor row's measured rates, undeclared then declared at three looks.
_FLOOR_ROW = (0.018, 0.0045)


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


def test_a_rate_floor_at_its_margin_under_a_plan_is_read_held_no_more_often_than_its_tail_and_is_moved_by_it() -> None:
    """The bounded interval keeps even the uncorrected reading under the 2.5% tail here, so the row is held by its
    pins: uncorrected and corrected each within four standard errors of what was measured."""
    declared = _held_rate(3, seed="floor")
    undeclared = _held_rate(None, seed="floor")
    assert declared <= at_most(ONE_TAIL, REPLICATES)
    assert within(undeclared, _FLOOR_ROW[0], REPLICATES), undeclared
    assert within(declared, _FLOOR_ROW[1], REPLICATES), declared
    assert not within(_FLOOR_ROW[0], _FLOOR_ROW[1], REPLICATES), "the pins cannot tell a no-op correction apart"


#: A cost floor (cheaper than the control), the two arms' true cost equal: per-case costs share a case level, each
#: reply's cost around it. ``held`` at a look is a false claim of cheaper.
_COST_FLOOR = Floor(aspect="b", clause=CostLower())


def _costs(rng: random.Random, levels: Sequence[float]) -> list[float]:
    return [level + rng.gauss(0.0, 0.002) for level in levels]


def _cost_held(control: Sequence[float], candidate: Sequence[float], planned: int | None) -> bool:
    """Whether the cost floor reads ``held`` on these per-case costs, its intervals built as ``floor_readings``
    builds them: the t interval at the nominal level, and under a plan of more than one look the held one at
    :func:`~threetears.evals.analysis.significance.looks_confidence`."""
    a, b = list(control), list(candidate)
    readings = FloorReadings(
        rates=(),
        control_cost=fmean(a),
        candidate_cost=fmean(b),
        control_cases=len(a),
        candidate_cases=len(b),
        cost_interval=difference_interval(a, b, paired=True, confidence=INTERVAL_LEVEL),
        cost_compared=True,
        cost_held_interval=(
            None
            if planned is None or planned <= 1
            else difference_interval(a, b, paired=True, confidence=looks_confidence(INTERVAL_LEVEL, planned))
        ),
    )
    return read_floor(_COST_FLOOR, readings, control="control", candidate="candidate").outcome == "held"


#: The fixed-batch cost row, 20, 40 and 60 cases: read held at some look, undeclared then declared at three looks.
_COST_FIXED = (0.0565, 0.02175)


def _cost_fixed_rate(planned: int | None, seed: str) -> float:
    rng = random.Random(seed)
    held = 0
    for _ in range(REPLICATES):
        levels = [0.01 + rng.gauss(0.0, 0.003) for _ in range(60)]
        control, candidate = _costs(rng, levels), _costs(rng, levels)
        if any(_cost_held(control[:n], candidate[:n], planned) for n in _sizes(3)):
            held += 1
    return held / REPLICATES


def test_a_cost_floor_read_held_at_every_look_with_no_plan_fails_its_tail_and_a_plan_holds_it() -> None:
    """The row whose bar the missing correction fails: on the exact t interval, three looks at the nominal level read
    a cost no lower as cheaper past the 2.5% tail; the plan holds it there."""
    undeclared = _cost_fixed_rate(None, seed="cost-fixed")
    declared = _cost_fixed_rate(3, seed="cost-fixed")
    assert undeclared > at_most(ONE_TAIL, REPLICATES), undeclared
    assert declared <= at_most(ONE_TAIL, REPLICATES), declared
    assert (round(undeclared, 4), round(declared, 4)) == pytest.approx(_COST_FIXED, abs=4e-3), (undeclared, declared)


# --- batches sized by the planner from the reading before ----------------------------------------------------------

#: The extends measured: one, two, three.
EXTENDS = (1, 2, 3)


def _rate_row_separates(control: Sequence[float], candidate: Sequence[float], level: float) -> bool:
    """The campaign row's rule on one aspect: the bounded interval on the change at ``1 − level`` clear of zero.

    The interval is computed only where it can clear: its lower end is the largest null the bet refutes at the tail
    ``level / 2``, and the bet's p rises with the null, so a lower end above zero needs the bet against no change to
    have refuted it already (:func:`~threetears.evals.analysis.stats.bounded_mean_p` at zero, on the same differences,
    range and stakes); likewise the upper end on the differences negated. Where neither side refutes no change, the
    interval reaches zero and is not computed (:func:`test_the_shortcut_never_changes_a_row_s_reading` holds it)."""
    if not _may_clear(control, candidate, level):
        return False
    interval = bounded_difference_interval(
        control, candidate, paired=True, value_range=(0.0, 1.0), confidence=1.0 - level
    )
    return interval is not None and (interval[0] > 0 or interval[1] < 0)


def _may_clear(control: Sequence[float], candidate: Sequence[float], level: float) -> bool:
    """Whether the bet against no change refutes it at the tail on either side: a necessary condition for the bounded
    interval at ``1 − level`` to clear zero (with a hair of room for the float rounding of the differences)."""
    tail = level / 2
    differences = [b - a for a, b in zip(control, candidate, strict=True)]
    return any(
        bounded_mean_p(side, 0.0, (-1.0, 1.0), alpha=tail) < tail * (1 + 1e-6)
        for side in (differences, [-difference for difference in differences])
    )


def test_the_shortcut_never_changes_a_row_s_reading() -> None:
    """Wherever the shortcut skips the interval, the interval reaches zero: drawn near a separation, so both outcomes
    occur, at the sizes and levels the planner-sized rows read."""
    rng = random.Random("shortcut")
    skipped = cleared = 0
    for _ in range(600):
        n = rng.choice([20, 40, 80, 140])
        level = rng.choice([SIGNIFICANCE_ALPHA, looks_alpha(2), looks_alpha(3)])
        control = _rate_shares(rng, n, p=0.7)
        candidate = _rate_shares(rng, n, p=rng.choice([0.7, 0.8, 0.9]))
        interval = bounded_difference_interval(
            control, candidate, paired=True, value_range=(0.0, 1.0), confidence=1.0 - level
        )
        clears = interval is not None and (interval[0] > 0 or interval[1] < 0)
        cleared += clears
        if not _may_clear(control, candidate, level):
            skipped += 1
            assert not clears, (n, level)
    assert skipped > 100 and cleared > 100, (skipped, cleared)


def planner_extend_rates(
    p: float, launched: int, planned: int | None, *, seed: str, planner: Planner = todays_planner
) -> list[float]:
    """How often a pass rate with no true difference has separated after zero, one, two and three planner-sized
    extends: read at :func:`~threetears.evals.analysis.significance.looks_alpha` of ``planned`` at every look, and
    stopped at the first separation."""
    rng = random.Random(seed)
    level = looks_alpha(planned)
    hits = [0] * (len(EXTENDS) + 1)
    for _ in range(EXTEND_REPLICATES):
        control, candidate = _rate_shares(rng, launched, p=p), _rate_shares(rng, launched, p=p)
        for look in range(len(EXTENDS) + 1):
            if _rate_row_separates(control, candidate, level):
                for later in range(look, len(hits)):
                    hits[later] += 1
                break
            if look == len(EXTENDS):
                break
            added = planner([b - a for a, b in zip(control, candidate, strict=True)], 1.0 - level)
            if added is None:
                break
            added = min(added, EXTEND_CAP)
            control += _rate_shares(rng, added, p=p)
            candidate += _rate_shares(rng, added, p=p)
    return [count / EXTEND_REPLICATES for count in hits]


#: The planner-sized rows, by (per-trial pass rate, cases launched, looks planned): how often a replicate has separated
#: after 0, 1, 2 and 3 extends, measured at each row's seed.
_PLANNER_EXTENDS: Mapping[tuple[float, int, int | None], tuple[float, float, float, float]] = {
    (0.8, 20, None): (0.0005, 0.0208, 0.0444, 0.0654),
    (0.8, 20, 2): (0.0, 0.0102, 0.0205, 0.0302),
    (0.8, 20, 3): (0.0, 0.0052, 0.0139, 0.0206),
    (0.8, 40, None): (0.0069, 0.0326, 0.0573, 0.0791),
    (0.8, 40, 2): (0.0023, 0.015, 0.0275, 0.0381),
    (0.8, 40, 3): (0.0008, 0.0078, 0.0172, 0.0239),
    (0.5, 20, None): (0.002, 0.0321, 0.0591, 0.079),
    (0.5, 20, 2): (0.0007, 0.0132, 0.0294, 0.0406),
    (0.5, 20, 3): (0.0, 0.0074, 0.0159, 0.0232),
    (0.5, 40, None): (0.0166, 0.0479, 0.0731, 0.0923),
    (0.5, 40, 2): (0.0059, 0.0203, 0.0348, 0.0448),
    (0.5, 40, 3): (0.0026, 0.0125, 0.0212, 0.0288),
}


@pytest.mark.parametrize("planned", [None, 2, 3])
@pytest.mark.parametrize(("p", "launched"), [(0.8, 20), (0.8, 40), (0.5, 20), (0.5, 40)])
def test_planner_sized_extends_err_at_the_rate_the_table_states(p: float, launched: int, planned: int | None) -> None:
    """Each row pinned within four standard errors of what was measured; a declared row within its plan at most α
    (where one is not, the caveat must say so for looks within the plan)."""
    rates = planner_extend_rates(p, launched, planned, seed=f"extend-{p}-{launched}-{planned}")
    measured = _PLANNER_EXTENDS.get((p, launched, planned))
    assert measured is not None and all(
        within(rate, pinned, EXTEND_REPLICATES) for rate, pinned in zip(rates, measured, strict=True)
    ), ("measured", p, launched, planned, rates)
    assert rates[0] <= at_most(SIGNIFICANCE_ALPHA if planned is None else looks_alpha(planned), EXTEND_REPLICATES)
    if planned is not None:
        within_plan = rates[planned - 1]
        assert within_plan <= at_most(SIGNIFICANCE_ALPHA, EXTEND_REPLICATES), ("past alpha within the plan", rates)


def cost_planner_rates(
    planned: int | None, *, seed: str, launched: int = 20, planner: Planner = todays_planner
) -> list[float]:
    """How often a cost floor with equal true cost reads held after zero, one, two and three planner-sized extends,
    each sized to show the candidate cheaper at the level ``held`` is read at."""
    rng = random.Random(seed)
    confidence = looks_confidence(INTERVAL_LEVEL, planned)
    hits = [0] * (len(EXTENDS) + 1)
    for _ in range(EXTEND_REPLICATES):
        levels = [0.01 + rng.gauss(0.0, 0.003) for _ in range(launched)]
        control, candidate = _costs(rng, levels), _costs(rng, levels)
        for look in range(len(EXTENDS) + 1):
            if _cost_held(control, candidate, planned):
                for later in range(look, len(hits)):
                    hits[later] += 1
                break
            if look == len(EXTENDS):
                break
            added = planner([b - a for a, b in zip(control, candidate, strict=True)], confidence)
            if added is None:
                break
            more = [0.01 + rng.gauss(0.0, 0.003) for _ in range(min(added, EXTEND_CAP))]
            control += _costs(rng, more)
            candidate += _costs(rng, more)
    return [count / EXTEND_REPLICATES for count in hits]


#: The planner-sized cost floor row, by the looks planned: how often it has read held after 0 to 3 extends.
_COST_PLANNER: Mapping[int | None, tuple[float, float, float, float]] = {
    None: (0.0256, 0.0495, 0.0642, 0.0784),
    2: (0.0122, 0.0238, 0.0335, 0.0401),
    3: (0.0089, 0.0189, 0.0244, 0.0293),
}


@pytest.mark.parametrize("planned", [None, 2, 3])
def test_a_cost_floor_at_its_margin_under_planner_sized_extends(planned: int | None) -> None:
    rates = cost_planner_rates(planned, seed=f"cost-extend-{planned}")
    measured = _COST_PLANNER.get(planned)
    assert measured is not None and all(
        within(rate, pinned, EXTEND_REPLICATES) for rate, pinned in zip(rates, measured, strict=True)
    ), ("measured", planned, rates)
    if planned is not None:
        assert rates[planned - 1] <= at_most(ONE_TAIL, EXTEND_REPLICATES), ("past the tail within the plan", rates)
    else:
        assert rates[-1] > at_most(ONE_TAIL, EXTEND_REPLICATES), ("the uncorrected reading should fail its tail", rates)


# --- the table the caveat quotes -------------------------------------------------------------------------------------


def test_the_table_the_caveat_quotes_is_the_measured_one() -> None:
    """``FALSE_SEPARATION`` states, for each procedure, plan and count, the largest rate measured here: the fixed
    batches' undeclared rates, the past-plan rows, and the planner-sized rows over every setting."""
    fixed = FALSE_SEPARATION["fixed_batches"]
    assert {(None, looks): rate for looks, rate in UNPLANNED_FALSE_SEPARATION.items()} | {
        (planned, taken): rate for planned, row in _PAST_PLAN.items() for taken, rate in row.items()
    } == dict(fixed)
    extended = FALSE_SEPARATION["planner_extends"]
    expected: dict[tuple[int | None, int], float] = {}
    for (_, _, planned), rates in _PLANNER_EXTENDS.items():
        for extends, rate in enumerate(rates):
            key = (planned, extends + 1)
            expected[key] = max(expected.get(key, 0.0), rate)
    assert expected == dict(extended)
