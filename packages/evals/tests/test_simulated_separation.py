"""The separation test between two arms, checked against data with a known truth (#601).

Every between-arm verdict the engine publishes — a campaign family's comparisons, a mechanism check,
``compare_two_runs`` — rests on :func:`~threetears.evals.analysis.stats.composite_significance` over
per-case means: paired over the cases both arms ran, Welch's statistic on Hsu's conservative degrees of
freedom otherwise. ``test_stats.py`` pins its
outputs on fixed inputs. This file checks what those outputs are FOR, by simulation at the sample sizes
the engine sees (2–15 cases, 1–5 repeats):

- under no difference, it calls one at most α of the time (exactly α for normal data, where the paired
  t-test is exact);
- under a stated difference, it calls one as often as the noncentral-t power says it should;
- its effect size estimates the population effect;
- the interval on the difference covers the true difference at its stated level.

Replicate counts and tolerance bands are derived in each test from the Monte-Carlo standard error
(:mod:`packages.evals.tests.simulation_support`). Every test owns its seed.
"""

from __future__ import annotations

import math
import random

import pytest

from threetears.evals.analysis.stats import (
    INTERVAL_LEVEL,
    SIGNIFICANCE_ALPHA,
    composite_significance,
    difference_interval,
    level_difference,
    paired_change,
    separation_p,
    t_critical_two_sided,
)
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
    def test_the_unpaired_test_holds_alpha(self, n_a: int, n_b: int, sd_a: float, sd_b: float) -> None:
        """Arms that share fewer than two cases are tested unpaired, at most at α.

        Welch's statistic on Hsu's ``min(n) − 1`` degrees of freedom is conservative, not exact, so this
        asserts an upper bound, over unequal sizes and unequal spreads.
        """
        rate = _welch_null_rate(n_a, n_b, sd_a, sd_b, replicates=self.REPLICATES)
        assert rate <= at_most(SIGNIFICANCE_ALPHA, self.REPLICATES), (
            f"{n_a} vs {n_b} cases: false-positive rate {rate:.4f} against α={SIGNIFICANCE_ALPHA}"
        )

    #: 8,000 replicates: SE at α is 0.0024, so the bound is 0.0597. On Welch–Satterthwaite's df the first
    #: four designs here measured 0.0995, 0.109, 0.064 and 0.070, each outside it, and the next two 0.056 and
    #: 0.054, above α but inside the band. On Hsu's df the largest is 0.042.
    LOPSIDED_REPLICATES = 8000

    @pytest.mark.parametrize(
        ("n_a", "n_b", "sd_a", "sd_b"),
        [
            (2, 10, 1.0, 1.0),
            (2, 10, 3.0, 1.0),
            (3, 10, 1.0, 1.0),
            (3, 10, 3.0, 1.0),
            (5, 10, 3.0, 1.0),
            (2, 10, 1.0, 3.0),
        ],
    )
    def test_with_a_two_or_three_case_side_it_holds_alpha(self, n_a: int, n_b: int, sd_a: float, sd_b: float) -> None:
        """The #601 finding: on Satterthwaite's df the test ran at up to 11% when one side had 2–3 cases.

        Satterthwaite's df overstates what a two-case side's variance is known to; Hsu's ``min(n) − 1`` does
        not, and holds α at every variance ratio (Mickey & Brown 1966) — the small side with the larger
        spread included, which is where a permutation test would not.
        """
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
    """The mean and Monte-Carlo SE of the reported Hedges' g over draws whose population effect is 0.5.

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
        estimate = composite_significance(a, b, paired=paired).hedges_g
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


class TestTheEffectSize:
    """The Hedges' g a comparison reports (``compare_two_runs``' ``hedges_g``), against the population effect."""

    #: 5,000 replicates: the reported g's SD at n=3 is about 0.6, so its mean's SE is ~0.008 and the
    #: 4-SE band ±0.03 on a true 0.5; at n=10, SD ~0.33, SE 0.005, band ±0.02.
    REPLICATES = 5000

    @pytest.mark.parametrize(("label", "n_a", "n_b", "paired"), _EFFECT_DESIGNS, ids=[d[0] for d in _EFFECT_DESIGNS])
    def test_it_estimates_the_population_effect_without_bias(
        self, label: str, n_a: int, n_b: int, paired: bool
    ) -> None:
        """The #601 finding: Cohen's d, reported before, read 0.88 for a true 0.5 at three paired cases (+77%),
        0.65 at five, 0.55 at ten and 0.61 unpaired at 3 vs 3 — exactly Hedges' factor ``1/J``. Hedges' g is d
        times J, and its mean is the population effect."""
        rng = random.Random(f"effect-size-bias-{label}")
        mean, se = _mean_effect_size(rng, n_a, n_b, paired=paired, replicates=self.REPLICATES)
        assert abs(mean - 0.5) <= TOLERANCE_Z * se, f"{label}: mean g {mean:.4f} against a true 0.5"

    def test_it_is_cohens_d_times_hedges_factor(self) -> None:
        """g is the textbook d corrected by J, the factor the simulation support computes independently."""
        a, b = [0.2, 0.4, 0.6, 0.5, 0.3], [0.5, 0.75, 0.95, 0.82, 0.58]
        diffs = [y - x for x, y in zip(a, b)]
        mean = sum(diffs) / len(diffs)
        sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1))
        assert composite_significance(a, b, paired=True).hedges_g == pytest.approx(mean / sd / hedges_correction(4))

    def test_two_pairs_report_no_effect_size(self) -> None:
        """At one degree of freedom E[1/s] diverges: no factor unbiases d, so none is reported — the p still is."""
        result = composite_significance([0.0, 0.0], [0.4, 0.7], paired=True)
        assert result.hedges_g is None and result.p_value is not None

    def test_it_is_consistent(self) -> None:
        """With many cases the reported g converges on the population effect."""
        rng = random.Random("effect-size-consistent")
        mean, se = _mean_effect_size(rng, 400, 400, paired=True, replicates=200)
        assert abs(mean - 0.5) <= TOLERANCE_Z * se


class TestTheIntervalOnTheDifference:
    """The interval a contrast states on its difference covers the true difference at its level."""

    #: 4,000 replicates: SE at 95% coverage is 0.0034, a 4-SE band of ±0.014.
    REPLICATES = 4000

    @pytest.mark.parametrize(("n_cases", "repeats"), [(2, 1), (3, 3), (5, 3), (10, 1), (15, 5)])
    def test_the_paired_interval_covers_exactly(self, n_cases: int, repeats: int) -> None:
        """The paired t interval over case means is exact for normal data: it covers 95% of the time."""
        rng = random.Random(f"paired-interval-{n_cases}-{repeats}")
        design = ClusteredDesign(n_cases=n_cases, repeats=repeats, between_case_sd=1.0, repeat_sd=0.5)
        hits = 0
        for _ in range(self.REPLICATES):
            control, contrast = draw_paired_arms(rng, design, effect=0.7, correlation=0.6)
            interval = difference_interval(case_means(control), case_means(contrast), paired=True)
            assert interval is not None
            hits += interval[0] <= 0.7 <= interval[1]
        rate = hits / self.REPLICATES
        assert within(rate, INTERVAL_LEVEL, self.REPLICATES), f"{n_cases} x {repeats}: coverage {rate:.4f}"

    @pytest.mark.parametrize(
        ("n_a", "n_b", "sd_a", "sd_b"), [(2, 10, 3.0, 1.0), (3, 3, 1.0, 1.0), (5, 15, 1.0, 3.0), (10, 10, 1.0, 2.0)]
    )
    def test_the_unpaired_interval_covers_at_least_nominally(
        self, n_a: int, n_b: int, sd_a: float, sd_b: float
    ) -> None:
        """The unpaired interval inverts the conservative test, so it covers at least 95%, at any variance ratio."""
        rng = random.Random(f"unpaired-interval-{n_a}-{n_b}-{sd_a}-{sd_b}")
        hits = 0
        for _ in range(self.REPLICATES):
            a = case_means(draw_clustered(rng, ClusteredDesign(n_cases=n_a, repeats=3, between_case_sd=sd_a)))
            b = case_means(draw_clustered(rng, ClusteredDesign(n_cases=n_b, repeats=3, between_case_sd=sd_b), mean=0.4))
            interval = difference_interval(a, b, paired=False)
            assert interval is not None
            hits += interval[0] <= 0.4 <= interval[1]
        rate = hits / self.REPLICATES
        assert rate >= at_least(INTERVAL_LEVEL, self.REPLICATES), f"{n_a} vs {n_b}: coverage {rate:.4f}"

    def test_it_excludes_zero_exactly_when_the_test_rejects(self) -> None:
        """One statistic, two readings: the 95% interval and the test at α=0.05 never disagree."""
        rng = random.Random("interval-test-duality")
        for paired in (True, False):
            for _ in range(500):
                a = [rng.gauss(0.0, 1.0) for _ in range(4)]
                b = [rng.gauss(1.0, 1.0) for _ in range(4)]
                interval = difference_interval(a, b, paired=paired)
                result = composite_significance(a, b, paired=paired)
                assert interval is not None and result.significant is not None
                assert result.significant is not (interval[0] <= 0.0 <= interval[1])


def test_at_least_and_at_most_bracket_the_nominal() -> None:
    """The band helpers are symmetric 4-SE bounds — the arithmetic every comment above quotes."""
    assert at_most(0.05, 4000) == pytest.approx(0.05 + 4 * math.sqrt(0.05 * 0.95 / 4000))
    assert at_least(0.05, 4000) == pytest.approx(0.05 - 4 * math.sqrt(0.05 * 0.95 / 4000))


def _counterexample_null(rng: random.Random, n_cases: int) -> tuple[list[float], list[float]]:
    """Paired 1-5 scores from a judge whose mean did not move: +1 four times in five, −4 the fifth (mean 0).

    A case moving +1 starts at 1-4; the fifth starts at 5 and lands on 1, so every score stays on the scale.
    """
    before: list[float] = []
    after: list[float] = []
    for _ in range(n_cases):
        if rng.random() < 0.8:
            start = float(rng.randint(1, 4))
            before.append(start)
            after.append(start + 1.0)
        else:
            before.append(5.0)
            after.append(1.0)
    return before, after


def _separates_by_contrast(a: list[float], b: list[float], value_range: tuple[float, float] | None) -> bool:
    """A campaign contrast's and the frontier's reading: the separation test's p below α."""
    p = separation_p(a, b, paired=True, value_range=value_range)
    return p is not None and p < SIGNIFICANCE_ALPHA


def _separates_by_history(a: list[float], b: list[float], value_range: tuple[float, float] | None) -> bool:
    """The history lens's reading: a directional label from the paired change."""
    verdict = paired_change(
        a, b, min_absolute_change=0.0, min_relative_change=0.0, higher_is_better=True, value_range=value_range
    )
    return verdict.label in ("improved", "regressed")


def _separates_by_levels(a: list[float], b: list[float], value_range: tuple[float, float] | None) -> bool:
    """The mechanism and scope lenses' reading: the between-level test separates."""
    tested = level_difference(dict(enumerate(a)), dict(enumerate(b)), value_range=value_range)
    return tested.separated is True


class TestAUniformMoveIsNotReadByTheSignFlip:
    """#597: every case moving by one amount is not evidence the mean moved, and no path reads it as such.

    The exact sign-flip p tests symmetry, not the mean. Under the counterexample null (a mean that did not move,
    +1 four times in five and −4 the fifth) ten cases all at +1 arise about one time in nine, and the sign flip
    states 2^-9 for them. Every path read that pattern as separated before #597: about 12% false separation at
    ten cases against a nominal 5%. On a declared range the bounded test reads it now, and with no range nothing
    does, so each path holds α.
    """

    @pytest.mark.parametrize("value_range", [(1.0, 5.0), None], ids=["declared-range", "no-range"])
    @pytest.mark.parametrize(
        "separates",
        [_separates_by_contrast, _separates_by_history, _separates_by_levels],
        ids=["contrast-and-frontier", "history", "mechanism-and-scope"],
    )
    def test_the_counterexample_null_is_called_separated_at_most_alpha(self, separates, value_range) -> None:
        replicates = 3000
        rng = random.Random(f"sign-flip-counterexample-{separates.__name__}-{value_range}")
        hits = sum(separates(*_counterexample_null(rng, 10), value_range) for _ in range(replicates))
        assert hits / replicates <= at_most(SIGNIFICANCE_ALPHA, replicates), f"false separation {hits / replicates:.4f}"
