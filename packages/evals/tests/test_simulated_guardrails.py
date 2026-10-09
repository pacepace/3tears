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

And end to end, through the assembled bundle: an arm with a real capability gain and a real guardrail
regression is never read as an arm whose guardrails all held, so it is either refused (breached) or
recommended only with its undecided guardrail stated — and the bundle's decision is exactly the rule's on
the per-case values. Every test owns its seed; tolerances come from the Monte-Carlo standard error.
"""

from __future__ import annotations

import math
import random

import pytest

from threetears.evals.analysis.stats import guardrail_decision
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
