"""The paired equivalence test (TOST) where every paired difference is one amount, against a known truth.

A difference with no spread has no t statistic, and :func:`~threetears.evals.analysis.stats.paired_equivalence`
once answered it with no p at all. A campaign contrast puts every equivalence p into the Holm family beside its
separation p, so a pattern with no p dropped out of the family silently: two arms that agreed on every one of
twelve cases could never read ``equivalent``, however wide the declared margin. The exact test decides it — each
one-sided test shifted by the margin is all one sign, and of the ``2 ** n`` equally likely sign flips only the
observed one is that extreme in the tested direction, so the TOST p is ``2 ** -n``.

The known answers pin the value; the simulation checks what it is FOR: on a two-point lattice centred on the
margin (the boundary, where a false equivalence claim is as likely as it gets), the all-one-amount samples that
the exact path claims equivalent arrive at the rate ``2 ** -n``, which is at most α wherever the claim is made.
"""

from __future__ import annotations

import random
import pytest

from threetears.evals.analysis.stats import SIGNIFICANCE_ALPHA, exact_decimal, paired_equivalence
from packages.evals.tests.simulation_support import at_most, within

#: The declared margin every case here tests against.
MARGIN = 0.5


class TestKnownAnswers:
    @pytest.mark.parametrize("n", [5, 6, 12])
    @pytest.mark.parametrize("amount", [0.0, 0.25, -0.4])
    def test_a_constant_difference_inside_the_margin_has_the_exact_one_sided_p(self, n: int, amount: float) -> None:
        assert paired_equivalence([amount] * n, MARGIN) == (True, 2.0**-n)

    @pytest.mark.parametrize("amount", [0.5, -0.5, 0.75])
    def test_a_constant_difference_on_or_beyond_the_margin_has_p_one(self, amount: float) -> None:
        assert paired_equivalence([amount] * 6, MARGIN) == (False, 1.0)

    def test_below_the_exact_tests_reach_no_test_decides(self) -> None:
        """At four pairs the smallest one-sided p is 1/16, above α: untested, never a p that could not be small."""
        assert 2.0**-4 > SIGNIFICANCE_ALPHA
        assert paired_equivalence([0.0] * 4, MARGIN) == (None, None)

    def test_a_constant_shift_written_in_decimals_is_read_exactly(self) -> None:
        """0.1..0.6 against 0.4..0.9: over floats the differences carry a residue a t-test reads as a tiny spread."""
        baseline = [i / 10 for i in range(1, 7)]
        current = [(i + 3) / 10 for i in range(1, 7)]
        assert len({c - b for b, c in zip(baseline, current)}) > 1, "the float differences are not one amount"
        exact = [exact_decimal(c) - exact_decimal(b) for b, c in zip(baseline, current)]
        assert paired_equivalence(exact, MARGIN) == (True, 2.0**-6)


@pytest.mark.parametrize("n", [5, 6, 8])
def test_the_exact_path_claims_a_false_equivalence_at_its_stated_rate(n: int) -> None:
    """True difference on the lower margin, differences on a two-point lattice: all-one-sign has probability 2^-n.

    Each paired difference is ``-MARGIN ± 0.25`` with equal chance, so the true mean difference sits exactly on
    the margin and equivalence is false. Every all-``+0.25`` sample is a constant difference inside the margin,
    which the exact path claims equivalent; that happens with probability ``2 ** -n``, which the simulated rate
    matches within Monte-Carlo error and which is at most α.
    """
    replicates = 20_000
    rng = random.Random(f"tost-exact-{n}")
    claimed = 0
    for _ in range(replicates):
        diffs = [-MARGIN + rng.choice((-0.25, 0.25)) for _ in range(n)]
        equivalent, p = paired_equivalence(diffs, MARGIN)
        if equivalent and len(set(diffs)) == 1:
            assert p == 2.0**-n
            claimed += 1
    rate = claimed / replicates
    assert within(rate, 2.0**-n, replicates)
    assert rate <= at_most(SIGNIFICANCE_ALPHA, replicates)
