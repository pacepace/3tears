"""A decision rule's switch on a rate, checked against data whose true drop is exactly the margin.

A rate clause (``RateDrop(max_points=9.5)``: no pass rate drops by more than 9.5 points) settles ``switch`` when the interval on the candidate
minus the control keeps the drop within the margin (:mod:`~threetears.evals.analysis.report.decision`). That is a
claim of no worse by more than the margin, so at a true drop of exactly the margin it must be made no more often
than the interval's one tail. A pass rate's per-case values are coarse shares of a few repeats, and a candidate that
breaks one case in ten leaves most samples of a dozen cases agreeing with the control.

On that shape the bounded interval (:func:`~threetears.evals.analysis.stats.bounded_difference_interval`, the one a
switch reads) holds its tail; the family's t interval, which a switch read until ``.30``, does not
(``test_the_simulation_catches_the_reading_it_replaced``). Each case runs three times; the control passes each trial
with chance 0.95 and the candidate as often, except on the one case in ten it breaks entirely, which puts its true
rate 0.095 below the control's: exactly the margin. Every test owns its seed; tolerances come
from the Monte-Carlo standard error.
"""

from __future__ import annotations

import random
from collections.abc import Callable

from threetears.evals.analysis.report.decision import RateReading, RuleReadings, evaluate_rule
from threetears.evals.kernel.declaration import DecisionRule, RateDrop
from threetears.evals.analysis.stats import (
    INTERVAL_LEVEL,
    bounded_difference_interval,
    difference_interval,
    separation_test,
)
from packages.evals.tests.simulation_support import at_most

REPLICATES = 1000
CASES = 12
REPEATS = 3
#: The interval's one tail, at a family of one: what a switch at the margin may be read at most.
ONE_TAIL = (1 - INTERVAL_LEVEL) / 2
#: G2's rule, its margin set to the true drop.
RULE = DecisionRule(clauses=(RateDrop(max_points=9.5),))

Interval = Callable[[list[float], list[float]], tuple[float, float] | None]


def _bounded(control: list[float], candidate: list[float]) -> tuple[float, float] | None:
    return bounded_difference_interval(control, candidate, paired=True, value_range=(0.0, 1.0))


def _family_t(control: list[float], candidate: list[float]) -> tuple[float, float] | None:
    """The interval a family carried for a rate until ``.30``: the t interval, the bounded one only with no spread."""
    if separation_test(control, candidate, paired=True, value_range=(0.0, 1.0)).basis in ("bounded", "identical"):
        return _bounded(control, candidate)
    return difference_interval(control, candidate, paired=True, value_range=(0.0, 1.0))


def _switch_rate(interval: Interval, seed: int) -> float:
    """How often the rule settles switch on the rate clause, at a true drop of 9.5 points against a 9.5-point margin."""
    rng = random.Random(seed)
    switched = 0
    for _ in range(REPLICATES):
        control: list[float] = []
        candidate: list[float] = []
        for _case in range(CASES):
            broken = rng.random() < 0.1
            control.append(sum(rng.random() < 0.95 for _ in range(REPEATS)) / REPEATS)
            candidate.append(0.0 if broken else sum(rng.random() < 0.95 for _ in range(REPEATS)) / REPEATS)
        reading = RateReading(
            "the pass rate",
            round(sum(control) * REPEATS),
            CASES * REPEATS,
            round(sum(candidate) * REPEATS),
            CASES * REPEATS,
            interval=interval(control, candidate),
        )
        readings = RuleReadings(rates=(reading,), control_cases=CASES, candidate_cases=CASES)
        switched += evaluate_rule(RULE, readings, control="v1", candidate="v2").clauses[0].outcome == "switch"
    return switched / REPLICATES


def test_at_the_margin_a_switch_on_the_bounded_interval_holds_its_tail() -> None:
    assert _switch_rate(_bounded, seed=3001) <= at_most(ONE_TAIL, REPLICATES)


def test_the_simulation_catches_the_reading_it_replaced() -> None:
    """The family's t interval, what a rate clause's switch read until ``.30``: about 14% against the 2.5% it states."""
    assert _switch_rate(_family_t, seed=3001) > at_most(ONE_TAIL, REPLICATES)
