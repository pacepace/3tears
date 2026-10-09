"""The separation test between two arms, checked against data with a known truth (#601).

Every between-arm verdict the engine publishes — a campaign family's comparisons, a mechanism check,
``compare_two_runs`` — rests on :func:`~threetears.evals.analysis.stats.composite_significance` over
per-case means: paired over the cases both arms ran, Welch's test otherwise. ``test_stats.py`` pins its
outputs on fixed inputs. This file checks what those outputs are FOR, by simulation at the sample sizes
the engine sees (2–15 cases, 1–5 repeats):

- under no difference, it calls one at most α of the time (exactly α for normal data, where the paired
  t-test is exact);
- under a stated difference, it calls one as often as the noncentral-t power says it should;
- its effect size estimates the population effect.

Replicate counts and tolerance bands are derived in each test from the Monte-Carlo standard error
(:mod:`packages.evals.tests.simulation_support`). Every test owns its seed.
"""

from __future__ import annotations

import math
import random

import pytest

from threetears.evals.analysis.stats import SIGNIFICANCE_ALPHA, composite_significance, t_critical_two_sided
from packages.evals.tests.simulation_support import (
    TOLERANCE_Z,
    ClusteredDesign,
    at_least,
    at_most,
    case_means,
    draw_binary_paired_arms,
    draw_clustered,
    draw_paired_arms,
    hedges_correction,
    paired_t_power,
    student_t_critical,
    within,
)


def _significant_share(draws: list[tuple[list[float], list[float]]], *, paired: bool) -> float:
    """The share of draws the separation test calls significant (an untested draw calls nothing)."""
    return sum(1 for a, b in draws if composite_significance(a, b, paired=paired).significant) / len(draws)


def _welch_null_rate(n_a: int, n_b: int, sd_a: float, sd_b: float, *, replicates: int) -> float:
    """The unpaired test's significant share over two arms of independent cases with equal means.

    Each side's case means come from ``n_a`` (``n_b``) cases × 3 repeats, case levels spread ``sd_a``
    (``sd_b``), so the two sides share nothing and differ only in size and spread.
    """
    rng = random.Random(f"welch-null-{n_a}-{n_b}-{sd_a}-{sd_b}")
    draws = []
    for _ in range(replicates):
        a = case_means(draw_clustered(rng, ClusteredDesign(n_cases=n_a, repeats=3, between_case_sd=sd_a)))
        b = case_means(draw_clustered(rng, ClusteredDesign(n_cases=n_b, repeats=3, between_case_sd=sd_b)))
        draws.append((a, b))
    return _significant_share(draws, paired=False)


class TestTheReferenceAnswers:
    """The references the simulations are held to reproduce published values, so a miss is the engine's."""

    @pytest.mark.parametrize(
        ("df", "published"),
        [(1, 12.706), (2, 4.303), (4, 2.776), (9, 2.262), (14, 2.145), (30, 2.042)],
    )
    def test_the_t_critical_values_are_the_tabled_ones(self, df: int, published: float) -> None:
        assert student_t_critical(0.95, df) == pytest.approx(published, abs=5e-4)

    @pytest.mark.parametrize(
        ("n", "effect", "published"),
        # Cohen (1988) / G*Power: one-sample two-sided t at α=0.05.
        [(10, 1.0, 0.803), (16, 0.74, 0.790), (32, 0.5, 0.782)],
    )
    def test_the_paired_power_is_the_tabled_one(self, n: int, effect: float, published: float) -> None:
        assert paired_t_power(n, effect, 0.05) == pytest.approx(published, abs=2e-3)

    @pytest.mark.parametrize("df", [1, 2, 3, 4, 5, 9, 14, 29])
    def test_the_engines_t_multiplier_is_the_closed_forms(self, df: int) -> None:
        """The engine's t multiplier (incomplete beta, bisected) against the independent trigonometric series."""
        assert t_critical_two_sided(0.95, df) == pytest.approx(student_t_critical(0.95, df), rel=1e-9)

    def test_the_hedges_factor_is_the_published_one(self) -> None:
        # Hedges (1981), J(df) = 1 - 3 / (4 df - 1) to first order; exact J(4) = 0.7979, J(9) = 0.9139.
        assert 1 / hedges_correction(4) == pytest.approx(0.7979, abs=1e-4)
        assert 1 / hedges_correction(9) == pytest.approx(0.9139, abs=1e-4)


class TestNoDifferenceIsCalledAtAlpha:
    """Under no difference between the arms, the test separates them α of the time."""

    #: 4,000 replicates: SE at α=0.05 is sqrt(0.05 * 0.95 / 4000) = 0.0034, so the 4-SE band is
    #: [0.036, 0.064] — a test running at 6.5% or 3.5% lands outside it.
    REPLICATES = 4000

    @pytest.mark.parametrize(
        ("n_cases", "repeats"),
        [(2, 1), (3, 3), (5, 3), (10, 1), (15, 5)],
    )
    def test_paired_over_case_means_is_exact_on_normal_data(self, n_cases: int, repeats: int) -> None:
        """The paired t-test over case means is exact for normal data, so its false-positive rate IS α.

        Arms correlate 0.6 per case, a case's level varies with SD 1 and a repeat with SD 0.5: the
        between-case variance that dominates a small bank cancels in the per-case difference.
        """
        rng = random.Random(f"paired-null-{n_cases}-{repeats}")
        design = ClusteredDesign(n_cases=n_cases, repeats=repeats, between_case_sd=1.0, repeat_sd=0.5)
        draws = []
        for _ in range(self.REPLICATES):
            control, contrast = draw_paired_arms(rng, design, effect=0.0, correlation=0.6)
            draws.append((case_means(control), case_means(contrast)))
        rate = _significant_share(draws, paired=True)
        assert within(rate, SIGNIFICANCE_ALPHA, self.REPLICATES), (
            f"{n_cases} cases x {repeats}: false-positive rate {rate:.4f} against α={SIGNIFICANCE_ALPHA}"
        )

    @pytest.mark.parametrize(
        ("n_a", "n_b", "sd_a", "sd_b"),
        [(2, 2, 1.0, 1.0), (3, 3, 1.0, 1.0), (5, 15, 1.0, 3.0), (10, 10, 1.0, 2.0)],
    )
    def test_welch_on_unshared_cases_holds_alpha(self, n_a: int, n_b: int, sd_a: float, sd_b: float) -> None:
        """Arms that share fewer than two cases are tested unpaired, by Welch's test, at most at α.

        Welch's degrees of freedom are an approximation, so this asserts an upper bound rather than
        exactness, over unequal sizes and unequal spreads.
        """
        rate = _welch_null_rate(n_a, n_b, sd_a, sd_b, replicates=self.REPLICATES)
        assert rate <= at_most(SIGNIFICANCE_ALPHA, self.REPLICATES), (
            f"{n_a} vs {n_b} cases: false-positive rate {rate:.4f} against α={SIGNIFICANCE_ALPHA}"
        )

    #: 8,000 replicates: SE at α is 0.0024, so the bound is 0.0597 and a measured 0.070 clears it by 3.6
    #: of its own SEs.
    LOPSIDED_REPLICATES = 8000

    @pytest.mark.parametrize(
        ("n_a", "n_b", "sd_a", "sd_b"),
        [(2, 10, 1.0, 1.0), (2, 10, 3.0, 1.0), (3, 10, 3.0, 1.0)],
    )
    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason=(
            "#601 finding: Welch's test (the unpaired fallback when arms share < 2 cases) runs above alpha when one "
            "side has 2-3 cases and the other many: false-positive rate 0.097 (2 vs 10 cases, equal SD), 0.117 (2 vs 10, small "
            "side 3x the SD) and 0.074 (3 vs 10, 3x) against nominal 0.05. "
            "Satterthwaite's df overstates the information in a 2-3 case side."
        ),
    )
    def test_welch_with_a_two_or_three_case_side_holds_alpha(
        self, n_a: int, n_b: int, sd_a: float, sd_b: float
    ) -> None:
        rate = _welch_null_rate(n_a, n_b, sd_a, sd_b, replicates=self.LOPSIDED_REPLICATES)
        assert rate <= at_most(SIGNIFICANCE_ALPHA, self.LOPSIDED_REPLICATES), (
            f"{n_a} vs {n_b} cases (SD {sd_a} vs {sd_b}): false-positive rate {rate:.4f} against α={SIGNIFICANCE_ALPHA}"
        )

    @pytest.mark.parametrize(
        ("n_cases", "repeats"),
        [(3, 3), (5, 1), (5, 3), (8, 3), (10, 5), (15, 3)],
    )
    def test_paired_over_pass_rates_stays_within_bradleys_liberal_bound(self, n_cases: int, repeats: int) -> None:
        """A pass/fail measure's case means sit on a lattice of k+1 values; the t-test is not exact there.

        The criterion for a t-test on discrete data is robustness, not exactness: Bradley's (1978)
        liberal bound puts the false-positive rate at no more than 1.5α. Cases differ in difficulty
        (between-case SD 1 on the logit scale around a 0.7 pass rate), and every case is equally hard on
        both arms, so no difference exists.
        """
        rng = random.Random(f"binary-null-{n_cases}-{repeats}")
        draws = []
        for _ in range(self.REPLICATES):
            control, contrast = draw_binary_paired_arms(rng, n_cases, repeats, base_rate=0.7, between_case_sd=1.0)
            draws.append((case_means(control), case_means(contrast)))
        rate = _significant_share(draws, paired=True)
        bound = 1.5 * SIGNIFICANCE_ALPHA + TOLERANCE_Z * math.sqrt(0.075 * 0.925 / self.REPLICATES)
        assert rate <= bound, f"{n_cases} cases x {repeats}: false-positive rate {rate:.4f} against 1.5α=0.075"


class TestAStatedDifferenceIsFoundAtItsPower:
    """Under a stated difference, the paired test finds it as often as the noncentral t says it should."""

    #: 3,000 replicates: SE at power 0.8 is sqrt(0.8 * 0.2 / 3000) = 0.0073, a 4-SE band of ±0.029.
    REPLICATES = 3000

    @pytest.mark.parametrize(
        ("n_cases", "repeats", "effect_size"),
        [(5, 3, 1.5), (10, 3, 1.0), (15, 1, 0.8)],
    )
    def test_power_matches_the_noncentral_t(self, n_cases: int, repeats: int, effect_size: float) -> None:
        """``effect_size`` is d_z: the true mean per-case difference over the true SD of that difference.

        The difference of two case means has variance ``2·between²·(1 - correlation) + 2·repeat²/k``
        (:func:`~packages.evals.tests.simulation_support.draw_paired_arms`), so the effect drawn is
        ``effect_size`` times its square root.
        """
        rng = random.Random(f"paired-power-{n_cases}-{repeats}-{effect_size}")
        design = ClusteredDesign(n_cases=n_cases, repeats=repeats, between_case_sd=1.0, repeat_sd=0.5)
        difference_sd = math.sqrt(2 * 1.0 * (1 - 0.6) + 2 * 0.25 / repeats)
        draws = []
        for _ in range(self.REPLICATES):
            control, contrast = draw_paired_arms(rng, design, effect=effect_size * difference_sd, correlation=0.6)
            draws.append((case_means(control), case_means(contrast)))
        power = _significant_share(draws, paired=True)
        expected = paired_t_power(n_cases, effect_size, SIGNIFICANCE_ALPHA)
        assert within(power, expected, self.REPLICATES), (
            f"{n_cases} cases x {repeats}, d_z={effect_size}: power {power:.4f} against the noncentral t's {expected:.4f}"
        )


def _mean_effect_size(rng: random.Random, n_a: int, n_b: int, *, paired: bool, replicates: int) -> tuple[float, float]:
    """The mean and Monte-Carlo SE of the reported Cohen's d over draws whose population effect is 0.5.

    Paired: ``n_a`` per-case differences, normal with mean 0.5 and SD 1, so d_z = 0.5. Unpaired: two
    normal samples of SD 1 whose means differ by 0.5, so d = 0.5.
    """
    estimates = []
    for _ in range(replicates):
        if paired:
            a = [0.0] * n_a
            b = [rng.gauss(0.5, 1.0) for _ in range(n_a)]
        else:
            a = [rng.gauss(0.0, 1.0) for _ in range(n_a)]
            b = [rng.gauss(0.5, 1.0) for _ in range(n_b)]
        estimate = composite_significance(a, b, paired=paired).cohens_d
        assert estimate is not None
        estimates.append(estimate)
    mean = sum(estimates) / replicates
    spread = math.sqrt(sum((value - mean) ** 2 for value in estimates) / (replicates - 1))
    return mean, spread / math.sqrt(replicates)


#: The effect-size designs: (label, n_a, n_b, paired). The paired one's SD has n - 1 df, the unpaired
#: one's pooled SD n_a + n_b - 2.
_EFFECT_DESIGNS = [
    ("paired, 3 cases", 3, 3, True),
    ("paired, 5 cases", 5, 5, True),
    ("paired, 10 cases", 10, 10, True),
    ("unpaired, 3 vs 3 cases", 3, 3, False),
    ("unpaired, 5 vs 5 cases", 5, 5, False),
]

#: The designs whose bias clears the Monte-Carlo band by a wide margin. Unpaired 5 vs 5 is biased too
#: (+10%, 0.05 on a true 0.5) but its mean's 4-SE band at this replicate count is ±0.04, too close to call.
_BIASED_DESIGNS = [design for design in _EFFECT_DESIGNS if design[0] != "unpaired, 5 vs 5 cases"]


class TestTheEffectSize:
    """The Cohen's d a comparison reports (``compare_two_runs``' ``cohens_d``), against the population effect."""

    #: 5,000 replicates: the reported d's SD at n=3 is about 1.0, so its mean's SE is ~0.014 and the
    #: 4-SE band ±0.057 on a true 0.5; at n=10, SD ~0.36, SE 0.005, band ±0.02.
    REPLICATES = 5000

    @pytest.mark.parametrize(("label", "n_a", "n_b", "paired"), _EFFECT_DESIGNS, ids=[d[0] for d in _EFFECT_DESIGNS])
    def test_its_bias_is_exactly_hedges_factor(self, label: str, n_a: int, n_b: int, paired: bool) -> None:
        """The reported d is the textbook sample estimator, whose mean is the true effect times J(df)⁻¹."""
        rng = random.Random(f"effect-size-shape-{label}")
        mean, se = _mean_effect_size(rng, n_a, n_b, paired=paired, replicates=self.REPLICATES)
        df = n_a - 1 if paired else n_a + n_b - 2
        expected = 0.5 * hedges_correction(df)
        assert abs(mean - expected) <= TOLERANCE_Z * se, f"{label}: mean d {mean:.4f} against 0.5·J⁻¹ = {expected:.4f}"

    def test_it_is_consistent(self) -> None:
        """With many cases the reported d converges on the population effect."""
        rng = random.Random("effect-size-consistent")
        mean, se = _mean_effect_size(rng, 400, 400, paired=True, replicates=200)
        assert abs(mean - 0.5) <= TOLERANCE_Z * se + 0.5 * (hedges_correction(399) - 1)

    @pytest.mark.parametrize(("label", "n_a", "n_b", "paired"), _BIASED_DESIGNS, ids=[d[0] for d in _BIASED_DESIGNS])
    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason=(
            "#601 finding: the reported Cohen's d is the uncorrected sample estimator, biased upward at the engine's "
            "sample sizes. Measured mean d for a true 0.5: paired 3 cases 0.88 (+77%), 5 cases 0.65 (+30%), "
            "10 cases 0.55 (+10%); unpaired 3 vs 3 0.61 (+22%); theory (Hedges' J) +77%, +25%, +9%, +25%. Hedges' g (d x J(df)) is unbiased."
        ),
    )
    def test_it_estimates_the_population_effect_without_bias(
        self, label: str, n_a: int, n_b: int, paired: bool
    ) -> None:
        rng = random.Random(f"effect-size-bias-{label}")
        mean, se = _mean_effect_size(rng, n_a, n_b, paired=paired, replicates=self.REPLICATES)
        assert abs(mean - 0.5) <= TOLERANCE_Z * se, f"{label}: mean d {mean:.4f} against a true 0.5"


def test_at_least_and_at_most_bracket_the_nominal() -> None:
    """The band helpers are symmetric 4-SE bounds — the arithmetic every comment above quotes."""
    assert at_most(0.05, 4000) == pytest.approx(0.05 + 4 * math.sqrt(0.05 * 0.95 / 4000))
    assert at_least(0.05, 4000) == pytest.approx(0.05 - 4 * math.sqrt(0.05 * 0.95 / 4000))
