"""The guardrail decision, checked against data with a known truth.

A guardrail is decided for an arm against the control by the bar rule read on the difference
(:func:`~threetears.evals.analysis.stats.guardrail_decision`): ``held`` when the 95% interval on
``arm − control`` lies wholly on the good side of the margin, ``breached`` when wholly beyond it,
``undecided`` otherwise. Each one-sided claim is one tail of a 95% interval, so:

- under no change, a guardrail reads ``breached`` at most 2.5% of the time (and less with a margin);
- under a regression exactly at the margin, it reads ``held`` at most 2.5% of the time;
- under a regression beyond the margin, it reads ``breached`` as often as the one-sided t-test's power says,
  which at fifteen cases and a regression one standard deviation of the differences past the margin is
  about 0.94;
- on pass/fail data at its ceiling, where whole draws have no spread and the bounded interval steps in, the
  no-change ``breached`` rate still holds.

**``held`` is a claim of safety, so its error rate is held on coarse, skewed values too.** On a declared range
the interval is the bounded test's (:func:`~threetears.evals.analysis.stats.bounded_difference_interval`), and at
the margin it reads ``held`` at most 2.5% of the time on every shape coarse scores take — a rare full drop on
pass/fail, a rare two- or four-point drop on 1-5, flips both ways — where the t interval it replaced read ``held``
up to 9.5% of the time (``test_the_simulation_catches_the_reading_it_replaced``). With no declared range ``held`` is
never read at all, whatever the data, and the reason names the remedy. A breach on a declared range is the bounded
test's as well, so a rare large gain hidden in a few cases does not read as a breach above its rate. With no range
the breach is still the t interval's, and on the rare-gain shape it reads ``breached`` above 2.5%: recorded as a
strict xfail, since no test of a mean holds its rate there (``docs/open-problems.md``).

And end to end, through the assembled bundle: an arm with a real capability gain and a real guardrail
regression is never read as an arm whose guardrails all held, so it is either refused (breached) or
recommended only with its undecided guardrail stated — and the bundle's decision is exactly the rule's on
the per-case values. Every test owns its seed; tolerances come from the Monte-Carlo standard error.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

import pytest

from threetears.evals.analysis.stats import (
    GUARDRAIL_HELD_NEEDS_RANGE,
    INTERVAL_LEVEL,
    difference_interval,
    guardrail_decision,
    interval_clears,
)
from packages.evals.tests.guardrail_support import BOUNDARY, CAPABILITY, two_arm_bundle
from packages.evals.tests.simulation_support import (
    ClusteredDesign,
    at_most,
    case_means,
    draw_binary_paired_arms,
    draw_paired_arms,
    paired_t_power,
    within,
)

REPLICATES = 2000

#: The variance of one paired per-case difference under :func:`draw_paired_arms` at the default design, k=3:
#: ``2 between² (1 − ρ) + 2 repeat² / k`` with between 1, ρ 0.5, repeat 0.5.
_DIFF_SD = math.sqrt(2 * 1.0 * 0.5 + 2 * 0.25 / 3)


def _rate(decision: str, *, n: int, effect: float, margin: float | None, seed: str) -> float:
    """The share of normal paired draws, ``n`` cases × 3 repeats, the rule decides ``decision``."""
    rng = random.Random(seed)
    hits = 0
    for _ in range(REPLICATES):
        control, contrast = draw_paired_arms(rng, ClusteredDesign(n_cases=n, repeats=3), effect=effect)
        verdict = guardrail_decision(
            case_means(control), case_means(contrast), paired=True, margin=margin, higher_is_better=True
        )
        hits += verdict.decision == decision
    return hits / REPLICATES


class TestTheDecisionHoldsItsErrorRates:
    @pytest.mark.parametrize("n", [5, 10, 15])
    @pytest.mark.parametrize("margin", [None, 0.5])
    def test_no_change_reads_breached_at_most_one_tail(self, n: int, margin: float | None) -> None:
        rate = _rate("breached", n=n, effect=0.0, margin=margin, seed=f"null-{n}-{margin}")
        assert rate <= at_most(0.025, REPLICATES)

    @pytest.mark.parametrize("n", [5, 15])
    @pytest.mark.parametrize("margin", [0.0, 0.5])
    def test_a_regression_at_the_margin_reads_held_at_most_one_tail(self, n: int, margin: float) -> None:
        rate = _rate("held", n=n, effect=-margin, margin=margin, seed=f"edge-{n}-{margin}")
        assert rate <= at_most(0.025, REPLICATES)

    @pytest.mark.parametrize("margin", [0.0, 0.5])
    def test_a_regression_beyond_the_margin_is_caught_with_the_t_tests_power_at_fifteen_cases(
        self, margin: float
    ) -> None:
        """One standard deviation of the differences past the margin: the one-sided test at 2.5% has power ~0.94."""
        reference = paired_t_power(15, 1.0, 0.05)
        rate = _rate("breached", n=15, effect=-(margin + _DIFF_SD), margin=margin, seed=f"power-{margin}")
        assert reference > 0.9
        assert within(rate, reference, REPLICATES)

    @pytest.mark.parametrize("base_rate", [0.7, 0.9, 0.97])
    @pytest.mark.parametrize("n", [5, 15])
    def test_a_pass_fail_guardrail_near_its_ceiling_reads_breached_at_most_one_tail(
        self, base_rate: float, n: int
    ) -> None:
        rng = random.Random(f"binary-null-{base_rate}-{n}")
        breached = 0
        for _ in range(REPLICATES):
            control, contrast = draw_binary_paired_arms(rng, n, 3, base_rate=base_rate)
            verdict = guardrail_decision(
                case_means(control),
                case_means(contrast),
                paired=True,
                margin=None,
                higher_is_better=True,
                value_range=(0.0, 1.0),
            )
            breached += verdict.decision == "breached"
        assert breached / REPLICATES <= at_most(0.025, REPLICATES)


#: Bundles assembled per end-to-end check; each takes a few tens of milliseconds.
BUNDLES = 100


class TestACapabilityGainCannotCarryAGuardrailLoss:
    def test_the_arm_never_reads_as_held_and_the_bundle_decides_as_the_rule_does(self) -> None:
        """Capability: 40% → about 93% of cases done. Guardrail: about 95% → about 51% of unsafe asks declined."""
        rng = random.Random("capability-gain-guardrail-loss")
        all_held = 0
        for _ in range(BUNDLES):
            capability = draw_binary_paired_arms(rng, 15, 1, base_rate=0.4, effect=3.0)
            boundary = draw_binary_paired_arms(rng, 15, 1, base_rate=0.95, between_case_sd=0.5, effect=-3.0)
            scores = {
                name: tuple([int(case[0]) for case in side] for side in sides)
                for name, sides in ((CAPABILITY, capability), (BOUNDARY, boundary))
            }
            bundle = two_arm_bundle(
                [(CAPABILITY, "capability", scores[CAPABILITY]), (BOUNDARY, "boundary", scores[BOUNDARY])]
            )
            assert not any(c.name == BOUNDARY for f in bundle.multiple_comparisons.families for c in f.comparisons)
            assert any(c.name == CAPABILITY for f in bundle.multiple_comparisons.families for c in f.comparisons)
            (check,) = bundle.guardrails.checks
            expected = guardrail_decision(
                [float(v) for v in scores[BOUNDARY][0]],
                [float(v) for v in scores[BOUNDARY][1]],
                paired=True,
                margin=None,
                higher_is_better=True,
                value_range=(0.0, 1.0),
            )
            assert (check.decision, check.interval) == (expected.decision, expected.interval)
            standing = bundle.guardrails.of_arm(check.contrast.variant_key)
            all_held += not standing.breached and not standing.undecided
        assert all_held / BUNDLES <= at_most(0.025, BUNDLES)


# --- held on coarse, skewed values: the bounded test, and never with no range -----------------------------------

PASS_FAIL = (0.0, 1.0)
RUBRIC = (1.0, 5.0)

#: The one-sided rate each claim read off a 95% interval may err at.
ONE_TAIL = (1 - INTERVAL_LEVEL) / 2

#: Each shape of paired difference coarse scores take, as (differences, their weights) with mean exactly ``-m``.
SHAPES = {
    # Pass/fail at its ceiling: a regression that breaks one case in 1/m, nothing else moving.
    "pass/fail rare drop": (PASS_FAIL, lambda m: ([-1, 0], [m, 1 - m])),
    # Pass/fail: the same net change among cases flipping both ways.
    "pass/fail flips": (
        PASS_FAIL,
        lambda m: ([-1, 0, 1], [Fraction(1, 20) + m, Fraction(9, 10) - m, Fraction(1, 20)]),
    ),
    # 1-5 at its ceiling: a rare drop of two or four points.
    "rubric 2-point drop": (RUBRIC, lambda m: ([-2, 0], [m / 2, 1 - m / 2])),
    "rubric 4-point drop": (RUBRIC, lambda m: ([-4, 0], [m / 4, 1 - m / 4])),
    # 1-5: most cases a point worse, a rare one four points better — a difference that looks worse than it is.
    "rubric rare gain": (RUBRIC, lambda m: ([-1, 4], [1 - (1 - m) / 5, (1 - m) / 5])),
}


def _paired_arms(rng: random.Random, shape: str, m: Fraction, n: int) -> tuple[list[float], list[float]]:
    """``n`` cases of (control, arm) values on the shape's range whose differences are drawn from the shape at ``-m``."""
    (low, high), support = SHAPES[shape]
    values, weights = support(m)
    assert sum(Fraction(v) * Fraction(w) for v, w in zip(values, weights)) == -m and min(weights) >= 0
    control, arm = [], []
    for diff in rng.choices(values, [float(w) for w in weights], k=n):
        # A drop from the top of the range, a gain from the bottom, no change at the top: the guardrail's usual state.
        start = high if diff <= 0 else low
        control.append(start)
        arm.append(start + diff)
    return control, arm


def _claim_rate(
    decision: str,
    shape: str,
    m: Fraction,
    n: int,
    *,
    declared: bool,
    higher_is_better: bool = True,
    reading: str = "engine",
    replicates: int = REPLICATES,
) -> float:
    """How often ``decision`` is read on the shape at a true worsening of exactly ``m``, against a margin of ``m``."""
    (low, high), _ = SHAPES[shape]
    rng = random.Random(f"coarse-{decision}-{shape}-{m}-{n}-{declared}-{higher_is_better}")
    hits = 0
    for _ in range(replicates):
        control, arm = _paired_arms(rng, shape, m, n)
        if not higher_is_better:  # the mirror image: a rise is the worsening
            control, arm = [low + high - v for v in control], [low + high - v for v in arm]
        margin = float(m) or None
        if reading == "t":
            interval = difference_interval(control, arm, paired=True)
            cleared = (
                None
                if interval is None
                else interval_clears(interval, 0.0, margin=margin, higher_is_better=higher_is_better)
            )
            said = {True: "held", False: "breached", None: "undecided"}[cleared]
        else:
            said = guardrail_decision(
                control,
                arm,
                paired=True,
                margin=margin,
                higher_is_better=higher_is_better,
                value_range=(low, high) if declared else None,
            ).decision
        hits += said == decision
    return hits / replicates


#: (shape, margin, n): the cells where the t interval read ``held`` most often at the margin, 3.6% to 9.5%.
HELD_CELLS = [
    ("rubric 4-point drop", Fraction(1, 2), 30),
    ("pass/fail rare drop", Fraction(1, 4), 30),
    ("pass/fail flips", Fraction(1, 10), 30),
    ("rubric 2-point drop", Fraction(1, 2), 30),
]


class TestHeldIsNeverClaimedAboveItsRate:
    @pytest.mark.parametrize(
        ("shape", "m", "n", "higher_is_better"),
        # Lower-is-better on the cell where the t interval was worst that way round (4.9%).
        [*((*cell, True) for cell in HELD_CELLS), ("pass/fail flips", Fraction(1, 10), 30, False)],
        ids=str,
    )
    def test_at_the_margin_a_declared_range_reads_held_at_most_one_tail(
        self, shape: str, m: Fraction, n: int, higher_is_better: bool
    ) -> None:
        rate = _claim_rate("held", shape, m, n, declared=True, higher_is_better=higher_is_better)
        assert rate <= at_most(ONE_TAIL, REPLICATES), f"{shape} at margin {m}, {n} cases: held {rate:.4f}"

    def test_the_simulation_catches_the_reading_it_replaced(self) -> None:
        """The t interval — what decided ``held`` on every guardrail until this rule — goes over 2.5% on the worst
        cell, about 9.5%: the proof the cells above would catch it coming back."""
        rate = _claim_rate("held", "rubric 4-point drop", Fraction(1, 2), 30, declared=True, reading="t")
        assert rate > at_most(ONE_TAIL, REPLICATES)

    @pytest.mark.parametrize(("shape", "m", "n"), [*HELD_CELLS, ("pass/fail flips", Fraction(1, 4), 12)], ids=str)
    def test_with_no_declared_range_held_is_never_read(self, shape: str, m: Fraction, n: int) -> None:
        assert _claim_rate("held", shape, m, n, declared=False, replicates=200) == 0.0

    def test_with_no_declared_range_an_arm_alike_with_the_control_is_undecided_and_names_the_remedy(self) -> None:
        """A t interval well inside the margin read ``held`` until now; it reads undecided and says what to declare."""
        control = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 3
        arm = [value + delta for value, delta in zip(control, [0.01, -0.01, 0.0] * 6)]
        verdict = guardrail_decision(control, arm, paired=True, margin=0.5, higher_is_better=True)
        assert verdict.interval is not None and verdict.interval[0] > -0.5, "the t interval would have read held"
        assert (verdict.decision, verdict.basis, verdict.refusal) == ("undecided", "t", GUARDRAIL_HELD_NEEDS_RANGE)
        ranged = guardrail_decision(control, arm, paired=True, margin=0.5, higher_is_better=True, value_range=(0, 7))
        assert ranged.basis == "bounded" and ranged.refusal is None


class TestABreachIsNotClaimedAboveItsRateOnADeclaredRange:
    @pytest.mark.parametrize("m", [Fraction(0), Fraction(1, 2)])
    def test_a_rare_large_gain_unseen_does_not_read_as_a_breach(self, m: Fraction) -> None:
        """At the margin (no worse than it), the arm looks worse in most samples; it is breached at most 2.5%."""
        rate = _claim_rate("breached", "rubric rare gain", m, 30, declared=True)
        assert rate <= at_most(ONE_TAIL, REPLICATES)

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="with no declared range a breach is still read off the t interval, which no test can fix (open-problems.md)",
    )
    def test_with_no_declared_range_the_t_breach_holds_its_rate_on_a_rare_large_gain(self) -> None:
        """About 4.1% at 30 cases and no margin, against the nominal 2.5%."""
        replicates = 4000
        rate = _claim_rate("breached", "rubric rare gain", Fraction(0), 30, declared=False, replicates=replicates)
        assert rate <= at_most(ONE_TAIL, replicates)
