"""The paired equivalence test (TOST) on coarse scores, against a known truth at the margin (#693).

Equivalence is the one claim of "no meaningful change", so its error rate is the rate at which a reader is told
a move sits inside the margin when it sits on it. The engine's scores are coarse — pass/fail per case, 1-5 rubric
points — and on coarse values the one-sided t-test does not hold its rate: on a two-point lattice of differences
at the margin over 12 pairs it claimed ``equivalent`` 7-8% of the time, and where a regression broke a few cases
outright it was far worse. A pass/fail difference that is 0 on most cases and -1 on one case in ten (a true
mean change of -0.1, exactly at a 0.1 margin) leaves twelve agreeing cases 28% of the time; the old reading took
a sample with no spread as the exact sign-flip test's ``2 ** -n`` and called it equivalent, and the t-test
called the rest equivalent whenever one or two -1s sat among zeros. Measured over 4,000 seeded replicates per
cell, the old reading's false-equivalence rate reached 0.77 (pass/fail, margin 0.05, five pairs).

With the measure's declared range, each one-sided test is the bounded test by betting
(:func:`~threetears.evals.analysis.stats.bounded_mean_p`), which holds α for every distribution on the range at
every n. The simulation below draws differences from the shapes coarse scores take at the margin — a two-point
lattice, pass/fail differences with and without a rare full drop, 1-5 differences with a rare one- two- or
four-point drop or symmetric noise — and checks the false-equivalence rate is at most α within Monte-Carlo
error in every cell. If the t-test or the sign-flip reading of a sample with no spread came back on a declared
range, the rare-drop cells would fail at once (``test_the_simulation_catches_the_readings_it_replaced``).
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

import pytest

from threetears.evals.analysis.stats import (
    SIGNIFICANCE_ALPHA,
    bounded_mean_p,
    paired_equivalence,
    t_critical_two_sided,
)
from packages.evals.tests.simulation_support import at_least, at_most

#: Replicates per cell: at α the Monte-Carlo SE is 0.0049, so the bound is α + 4 SE ≈ 0.070.
REPLICATES = 2000

PASS_FAIL = (0.0, 1.0)
RUBRIC = (1.0, 5.0)


def _support(name: str, margin: Fraction) -> tuple[list[Fraction], list[float]]:
    """A distribution of paired differences whose mean is exactly ``-margin``: the lower edge of equivalence."""
    m = margin
    shapes: dict[str, tuple[list[Fraction], list[Fraction]]] = {
        # Pass/fail: a regression that breaks one case in 1/m, nothing else moving.
        "pass/fail rare drop": ([Fraction(-1), Fraction(0)], [m, 1 - m]),
        # Pass/fail: the same net change among cases flipping both ways.
        "pass/fail flips": (
            [Fraction(-1), Fraction(0), Fraction(1)],
            [Fraction(1, 20) + m, Fraction(9, 10) - m, Fraction(1, 20)],
        ),
        # The issue's lattice: every difference a quarter-point either side of the margin.
        "two-point lattice": ([-m - Fraction(1, 4), -m + Fraction(1, 4)], [Fraction(1, 2), Fraction(1, 2)]),
        # 1-5 rubric differences: a rare drop of 2 or 4 points, or symmetric one-point noise shifted to the margin.
        "rubric 2-point drop": ([Fraction(-2), Fraction(0)], [m / 2, 1 - m / 2]),
        "rubric 4-point drop": ([Fraction(-4), Fraction(0)], [m / 4, 1 - m / 4]),
        "rubric noise": (
            [Fraction(-1), Fraction(0), Fraction(1)],
            [Fraction(1, 4) + m / 2, Fraction(1, 2), Fraction(1, 4) - m / 2],
        ),
    }
    values, weights = shapes[name]
    assert sum(v * w for v, w in zip(values, weights)) == -m and min(weights) >= 0
    return values, [float(w) for w in weights]


def _false_equivalence_rate(
    shape: str, margin: Fraction, n: int, value_range: tuple[float, float] | None, replicates: int = REPLICATES
) -> float:
    values, weights = _support(shape, margin)
    rng = random.Random(f"tost-{shape}-{margin}-{n}")
    claimed = 0
    for _ in range(replicates):
        diffs = rng.choices(values, weights, k=n)
        claimed += paired_equivalence(diffs, float(margin), value_range=value_range)[0] is True
    return claimed / replicates


#: (shape, margin, range): every shape the engine's coarse scores take, at margins where equivalence is reachable
#: inside 30 pairs and where it is not. A difference wider than the range's width cannot occur, so a shape whose
#: drop exceeds it is left out of that range.
CELLS = [
    *[
        (shape, Fraction(margin), PASS_FAIL)
        for shape in ("pass/fail rare drop", "pass/fail flips", "two-point lattice")
        for margin in ("1/10", "1/4", "1/2")
    ],
    *[
        (shape, Fraction(margin), RUBRIC)
        for shape in ("pass/fail rare drop", "rubric 2-point drop", "rubric 4-point drop")
        for margin in ("1/2", "1")
    ],
    ("rubric noise", Fraction(1, 2), RUBRIC),
]


@pytest.mark.parametrize("n", [5, 12, 30])
@pytest.mark.parametrize(("shape", "margin", "value_range"), CELLS, ids=lambda v: str(v))
def test_a_false_equivalence_is_claimed_at_most_alpha_on_every_coarse_support(
    shape: str, margin: Fraction, value_range: tuple[float, float], n: int
) -> None:
    rate = _false_equivalence_rate(shape, margin, n, value_range)
    assert rate <= at_most(SIGNIFICANCE_ALPHA, REPLICATES), (
        f"{shape} at margin {margin} on {value_range}, {n} pairs: claimed equivalent {rate:.4f} of the time"
    )


@pytest.mark.parametrize(
    ("shape", "margin", "n", "old_rate"),
    [
        ("two-point lattice", Fraction(1, 2), 12, 0.079),
        ("pass/fail rare drop", Fraction(1, 4), 12, 0.16),
        ("rubric 2-point drop", Fraction(1, 2), 12, 0.16),
    ],
)
def test_the_simulation_catches_the_readings_it_replaced(shape: str, margin: Fraction, n: int, old_rate: float) -> None:
    """With no declared range the engine still reads a spread by the t-test and these cells go over α — what the
    old reading did on every range. ``old_rate`` is what the old reading measured (the t-test where the
    differences had spread, ``2 ** -n`` where they had none); the no-range reading, untested where there is no
    spread, still exceeds α, which is the proof the cells above would catch either reading coming back."""
    replicates = 4000
    rate = _false_equivalence_rate(shape, margin, n, None, replicates)
    assert rate > at_most(SIGNIFICANCE_ALPHA, replicates) and old_rate > SIGNIFICANCE_ALPHA


class TestKnownAnswers:
    @pytest.mark.parametrize(("margin", "value_range", "n_needed"), [(0.25, PASS_FAIL, 12), (0.5, RUBRIC, 26)])
    def test_agreeing_cases_show_equivalence_once_they_rule_out_a_hidden_drop(
        self, margin: float, value_range: tuple[float, float], n_needed: int
    ) -> None:
        """``n`` agreeing cases still leave a drop of the full width in one case of ``width / margin`` unseen
        ``(1 − margin / width) ** n`` of the time, so no valid test's p is smaller, and no valid test can show
        equivalence before that reaches α (11 and 23 pairs here); the bounded test needs a few pairs more."""
        width = value_range[1] - value_range[0]
        for n in (n_needed - 1, n_needed):
            equivalent, p = paired_equivalence([0.0] * n, margin, value_range=value_range)
            assert p is not None and p >= (1 - margin / width) ** n
            assert equivalent is (n == n_needed)
        assert (1 - margin / width) ** (n_needed - 1) < SIGNIFICANCE_ALPHA

    def test_with_no_range_a_difference_with_no_spread_is_untested(self) -> None:
        """No t, and no bound on a case that moved unseen: no test decides, so neither label nor p."""
        assert paired_equivalence([0.0] * 30, 0.5) == (None, None)

    def test_with_no_range_a_spread_is_read_by_the_t_test(self) -> None:
        """Equivalent exactly when the 90% t interval on the mean difference sits inside ± the margin."""
        diffs = [0.1, -0.1, 0.05, 0.0, -0.05, 0.02]
        n, mean = len(diffs), sum(diffs) / len(diffs)
        sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1))
        half = t_critical_two_sided(1 - 2 * SIGNIFICANCE_ALPHA, n - 1) * sd / math.sqrt(n)
        for margin in (abs(mean) + half * 1.01, abs(mean) + half * 0.99):
            equivalent, p = paired_equivalence(diffs, margin)
            assert p is not None and equivalent is (margin > abs(mean) + half)

    def test_a_difference_outside_the_declared_range_is_untested(self) -> None:
        """Values that contradict their declared range leave its bound no bound: no test, rather than a wrong one."""
        assert paired_equivalence([0.0] * 20 + [1.5], 0.5, value_range=PASS_FAIL) == (None, None)
        with pytest.raises(ValueError, match="outside"):
            bounded_mean_p([2.0], 0.0, (-1.0, 1.0))

    def test_the_bounded_p_refutes_a_null_at_the_bottom_of_the_range_outright(self) -> None:
        assert bounded_mean_p([-1.0, -0.5], -1.0, (-1.0, 1.0)) == 0.0
        assert bounded_mean_p([-1.0, -1.0], -1.0, (-1.0, 1.0)) == 1.0


def test_equivalence_is_reachable_on_a_coarse_lattice_with_no_true_change() -> None:
    """The power the guarantee leaves: a two-point lattice a quarter-point either side of no change, margin 0.5 on
    a 0-1 range, is shown equivalent from eight pairs — simulated at about 98% — where the t-test claimed it from
    five, at the price of the 7-8% false-equivalence rate on the same lattice at the margin."""
    rng = random.Random("tost-power")
    replicates = 1000
    shown = sum(
        paired_equivalence(rng.choices([Fraction(-1, 4), Fraction(1, 4)], k=8), 0.5, value_range=PASS_FAIL)[0] is True
        for _ in range(replicates)
    )
    assert shown / replicates >= at_least(0.95, replicates)
