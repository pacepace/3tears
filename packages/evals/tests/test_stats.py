"""Unit tests for :mod:`threetears.evals.analysis.stats`.

The two-sided Student-t p-value is a dependency-free reimplementation (the
project carries no scipy), so the accuracy tests pin it against known reference
values computed with scipy: ``2 * scipy.stats.t.sf(abs(t), df)``.
"""

from __future__ import annotations

import math
from fractions import Fraction

import pytest

from threetears.evals.analysis.stats import (
    EQUIVALENCE_NEEDS_RANGE,
    PAIRED_TEST_NAME,
    SIGNIFICANCE_ALPHA,
    UNIFORM_MOVE_NEEDS_RANGE,
    UNPAIRED_TEST_NAME,
    bounded_separation_p,
    composite_significance,
    difference_interval,
    level_difference,
    paired_change,
    separation_p,
    separation_test,
    standard_error_of_mean,
    t_critical_two_sided,
)


class TestStandardErrorOfMean:
    """The dispersion every pivot cell reports, including its None-vs-0.0 contract."""

    @pytest.mark.parametrize("values", [[], [0.7]])
    def test_fewer_than_two_observations_is_none_not_zero(self, values: list[float]) -> None:
        """Unknown precision must not render as zero spread.

        Zero would rank the least certain cell in a pivot as the most certain
        one, which is the absence/zero conflation the reporting tier exists to
        prevent.
        """
        assert standard_error_of_mean(values) is None

    def test_constant_sample_of_two_or_more_is_zero_not_none(self) -> None:
        """A genuinely constant sample measured zero spread — that is a result, not an absence."""
        result = standard_error_of_mean([0.5, 0.5, 0.5])
        assert result == 0.0
        assert result is not None

    def test_matches_hand_computed_sd_over_sqrt_n(self) -> None:
        """sd(ddof=1)/sqrt(n) for [1, 2, 3, 4]: sd = sqrt(5/3), n = 4."""
        expected = math.sqrt(5.0 / 3.0) / math.sqrt(4)
        assert standard_error_of_mean([1.0, 2.0, 3.0, 4.0]) == pytest.approx(expected)

    def test_uses_ddof_one_not_population_sd(self) -> None:
        """The population form would divide by n; ddof=1 divides by n-1 and is larger."""
        values = [1.0, 2.0, 3.0, 4.0]
        population_sem = math.sqrt(sum((v - 2.5) ** 2 for v in values) / len(values)) / math.sqrt(len(values))
        assert standard_error_of_mean(values) > population_sem


class TestStudentTPValue:
    """The incomplete-beta t-distribution p-value matches scipy reference values.

    Read through the module's public surface: :func:`t_critical_two_sided` is the p-value's
    inverse, found by bisection over the same function, so it lands on the reference ``t`` exactly
    when the p-value at that ``t`` is the reference p. The p-value's sign and zero behaviour are read
    off :func:`composite_significance`, the one public caller that reports it.
    """

    @pytest.mark.parametrize(
        ("t", "df", "expected"),
        [
            (2.5, 4.0, 0.0667665448),
            (1.1, 3.0, 0.3516831949),
            (5.0, 8.0, 0.0010528258),
            (0.3, 10.0, 0.7703206076),
            (3.2, 6.7, 0.0159869094),
        ],
    )
    def test_matches_scipy_reference(self, t: float, df: float, expected: float) -> None:
        assert t_critical_two_sided(1 - expected, df) == pytest.approx(t, rel=1e-6)

    def test_zero_t_gives_p_one(self) -> None:
        """No difference (t=0) → p=1: never significant."""
        _d, sig, p = composite_significance([0.0] * 4, [1.0, -1.0, 0.5, -0.5], paired=True)
        assert p == pytest.approx(1.0)
        assert sig is False

    def test_symmetric_in_sign_of_t(self) -> None:
        a, b = [0.2, 0.4, 0.6, 0.5, 0.3], [0.5, 0.45, 0.95, 0.52, 0.58]
        _d, _sig, forward = composite_significance(a, b, paired=True)
        _d, _sig, backward = composite_significance(b, a, paired=True)
        assert forward is not None
        assert forward == pytest.approx(backward)

    def test_nonpositive_df_is_refused(self) -> None:
        """No t-distribution exists on zero degrees of freedom, so no tail is quoted on one."""
        with pytest.raises(ValueError, match="df must be positive"):
            t_critical_two_sided(0.95, 0.0)


class TestCompositeSignificance:
    """Effect size + significance flag over two composite-score samples."""

    def test_paired_clear_improvement_is_significant(self) -> None:
        d, sig, _p = composite_significance([0.2, 0.4, 0.6, 0.5, 0.3], [0.5, 0.75, 0.95, 0.82, 0.58], paired=True)
        assert sig is True
        assert d is not None and d > 0  # B beats A → positive effect

    def test_paired_noise_is_not_significant(self) -> None:
        d, sig, _p = composite_significance([0.50, 0.60, 0.40, 0.55], [0.52, 0.58, 0.45, 0.50], paired=True)
        assert sig is False
        assert d is not None

    def test_unpaired_welch_clear_separation_is_significant(self) -> None:
        d, sig, _p = composite_significance(
            [0.20, 0.30, 0.25, 0.35, 0.28], [0.70, 0.80, 0.75, 0.72, 0.90], paired=False
        )
        assert sig is True
        assert d is not None and d > 0

    def test_unpaired_overlap_is_not_significant(self) -> None:
        _d, sig, _p = composite_significance([0.40, 0.60, 0.50, 0.55], [0.45, 0.65, 0.52, 0.60], paired=False)
        assert sig is False

    def test_identical_samples_are_zero_effect_not_significant(self) -> None:
        """Deterministically identical runs → d=0, definitively not significant, and no t to quote."""
        assert composite_significance([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], paired=True) == (0.0, False, None)

    def test_constant_nonzero_gap_is_undefined(self) -> None:
        """A perfectly constant gap has no finite effect size (sd=0) → nothing measured."""
        assert composite_significance([0.5, 0.5], [0.7, 0.7], paired=True) == (None, None, None)

    def test_too_few_observations_returns_none(self) -> None:
        assert composite_significance([0.5], [0.6], paired=True) == (None, None, None)
        assert composite_significance([0.5], [0.6, 0.7], paired=False) == (None, None, None)

    def test_paired_length_mismatch_returns_none(self) -> None:
        assert composite_significance([0.5, 0.6], [0.6], paired=True) == (None, None, None)

    def test_empty_samples_return_none(self) -> None:
        assert composite_significance([], [], paired=True) == (None, None, None)
        assert composite_significance([], [], paired=False) == (None, None, None)

    def test_reports_the_p_value_it_tested_against(self) -> None:
        """The p travels with the verdict, and it is the number the verdict came from.

        Recomputed here from the samples rather than pinned as a literal: the
        contract is that the flag and the p are the SAME test, so the assertion
        that matters is ``sig is (p < alpha)`` beside a p that matches the
        t-statistic those samples produce.
        """
        a = [0.2, 0.4, 0.6, 0.5, 0.3]
        b = [0.5, 0.75, 0.95, 0.82, 0.58]
        d, sig, p = composite_significance(a, b, paired=True)

        diffs = [y - x for x, y in zip(a, b)]
        n = len(diffs)
        mean_diff = sum(diffs) / n
        sd = math.sqrt(sum((v - mean_diff) ** 2 for v in diffs) / (n - 1))
        t_stat = mean_diff / (sd / math.sqrt(n))

        # The p is the two-sided tail of the paired t-statistic on n - 1 degrees of freedom: the
        # critical value at that p's confidence is the statistic itself.
        assert p is not None
        assert t_critical_two_sided(1 - p, n - 1) == pytest.approx(t_stat, rel=1e-6)
        assert sig is (p < SIGNIFICANCE_ALPHA)
        # Hedges' g_z: d_z times J(n - 1), the exact small-sample factor.
        j = math.exp(math.lgamma((n - 1) / 2) - math.lgamma((n - 2) / 2)) / math.sqrt((n - 1) / 2)
        assert d == pytest.approx(j * mean_diff / sd)

    def test_the_unpaired_path_reports_its_p_too(self) -> None:
        """Welch's arm must not be the one that quietly drops the statistic."""
        _d, sig, p = composite_significance(
            [0.20, 0.30, 0.25, 0.35, 0.28], [0.70, 0.80, 0.75, 0.72, 0.90], paired=False
        )
        assert p is not None
        assert sig is (p < SIGNIFICANCE_ALPHA)

    def test_a_negative_verdict_carries_the_p_that_produced_it(self) -> None:
        """A negative verdict is only checkable when the p behind it is shown."""
        _d, sig, p = composite_significance([0.50, 0.60, 0.40, 0.55], [0.52, 0.58, 0.45, 0.50], paired=True)
        assert sig is False
        assert p is not None and p >= SIGNIFICANCE_ALPHA

    @pytest.mark.parametrize(
        ("sample_a", "sample_b", "paired"),
        [
            ([0.5], [0.6], True),  # fewer than two pairs
            ([0.5], [0.6, 0.7], False),  # fewer than two on one side
            ([0.5, 0.6], [0.6], True),  # length mismatch
            ([], [], True),
            ([], [], False),
            ([0.5, 0.5], [0.7, 0.7], True),  # deterministic gap: zero difference variance
            ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], True),  # identical: no t-statistic exists
            ([0.4, 0.4], [0.4, 0.4], False),  # unpaired, zero pooled variance
        ],
    )
    def test_returns_no_p_when_the_test_is_undefined(
        self, sample_a: list[float], sample_b: list[float], paired: bool
    ) -> None:
        """Absence is None, never 1.0.

        1.0 is a real result — it is what t=0 produces when two samples coincide
        — so using it for "no test ran" would make an untested comparison read
        as one that was tested and found identical.
        """
        assert composite_significance(sample_a, sample_b, paired=paired).p_value is None

    def test_effect_sign_follows_direction(self) -> None:
        """B worse than A → negative Hedges' g."""
        d, _sig, _p = composite_significance([0.70, 0.80, 0.75, 0.90], [0.20, 0.30, 0.25, 0.28], paired=False)
        assert d is not None and d < 0


class TestPairedChange:
    """The regression classifier: significance AND magnitude, with direction and disclosure."""

    def test_significant_decline_over_threshold_is_a_regression(self) -> None:
        verdict = paired_change(
            [0.80, 0.82, 0.78, 0.81, 0.79],
            [0.60, 0.61, 0.59, 0.62, 0.58],
            min_absolute_change=0.05,
            min_relative_change=0.0,
            higher_is_better=True,
        )
        assert verdict.label == "regressed"
        assert verdict.significant is True
        assert verdict.exceeds_threshold is True
        assert verdict.delta is not None and verdict.delta < 0

    def test_significant_gain_over_threshold_is_an_improvement(self) -> None:
        verdict = paired_change(
            [0.40, 0.42, 0.38, 0.41, 0.39],
            [0.80, 0.81, 0.79, 0.82, 0.78],
            min_absolute_change=0.05,
            min_relative_change=0.0,
            higher_is_better=True,
        )
        assert verdict.label == "improved"

    def test_lower_is_better_makes_an_increase_the_regression(self) -> None:
        """Cost/latency invert: a significant rise is the decline, not the drop.

        The rise is not one constant amount: a constant +0.2 over five pairs has the exact sign-flip p of
        1/16 and is untested, which the float residue in ``0.30 - 0.10`` once hid behind a t-test.
        """
        verdict = paired_change(
            [0.10, 0.11, 0.09, 0.10, 0.10],
            [0.30, 0.32, 0.29, 0.31, 0.30],
            min_absolute_change=0.05,
            min_relative_change=0.0,
            higher_is_better=False,
        )
        assert verdict.label == "regressed"

    def test_a_significant_but_tiny_change_is_below_threshold_not_flagged(self) -> None:
        """The joint gate: significance alone must not flag when the move is below threshold.

        Nor is it "no change": the move is separated from zero, so it reads ``below_threshold`` (#592).
        """
        verdict = paired_change(
            [0.800] * 6,
            [0.809, 0.811, 0.809, 0.811, 0.809, 0.811],
            min_absolute_change=0.05,
            min_relative_change=0.10,
            higher_is_better=True,
        )
        assert verdict.significant is True
        assert verdict.exceeds_threshold is False
        assert verdict.label == "below_threshold"

    def test_relative_threshold_can_flag_when_absolute_would_not(self) -> None:
        """A small absolute move on a small baseline is a large relative one."""
        verdict = paired_change(
            [0.020] * 6,
            [0.039, 0.041, 0.039, 0.041, 0.039, 0.041],
            min_absolute_change=1.0,  # absolute gate never clears
            min_relative_change=0.50,  # but +100% relative does
            higher_is_better=True,
        )
        assert verdict.exceeds_threshold is True
        assert verdict.label == "improved"

    def test_both_thresholds_off_lets_significance_alone_flag(self) -> None:
        """0.0/0.0 imposes no magnitude floor — a significant move is flagged on its own."""
        verdict = paired_change(
            [0.80] * 6,
            [0.789, 0.791, 0.789, 0.791, 0.789, 0.791],
            min_absolute_change=0.0,
            min_relative_change=0.0,
            higher_is_better=True,
        )
        assert verdict.exceeds_threshold is True
        assert verdict.label == "regressed"

    def test_an_off_relative_gate_does_not_nullify_an_active_absolute_one(self) -> None:
        """A 0.0 relative gate is OFF, not a gate that passes everything.

        The tell of the bug this pins: with OR-combined gates, ``|relative| >= 0``
        is always true, so an unset relative threshold would silently pass every
        change and defeat the absolute floor the caller did set.
        """
        verdict = paired_change(
            [0.800] * 6,
            [0.8009, 0.8011, 0.8009, 0.8011, 0.8009, 0.8011],
            min_absolute_change=0.05,  # the real floor: +0.001 does not clear it
            min_relative_change=0.0,  # OFF, must not rescue the change
            higher_is_better=True,
        )
        assert verdict.significant is True
        assert verdict.exceeds_threshold is False
        assert verdict.label == "below_threshold"

    def test_a_uniform_decline_on_a_declared_range_is_read_by_the_bounded_test(self) -> None:
        """Every case dropping by the same amount leaves no t, and the bounded test on the range decides (#597).

        The exact sign-flip p (``2 ** (1 - n)``) this once carried tests symmetry, not the mean. The bounded test
        holds α for every distribution on the range, and its p is the one carried, so the label can be checked.
        """
        verdict = paired_change(
            [1.0] * 6,
            [0.0] * 6,
            min_absolute_change=0.05,
            min_relative_change=0.0,
            higher_is_better=True,
            value_range=(0.0, 1.0),
        )
        expected = bounded_separation_p([1.0] * 6, [0.0] * 6, paired=True, value_range=(0.0, 1.0))
        assert expected is not None and expected < SIGNIFICANCE_ALPHA
        assert verdict.p_value == expected
        assert verdict.significant is True
        assert verdict.label == "regressed"
        assert verdict.hedges_g is None, "no spread, so no finite effect size"
        assert verdict.not_separated_reason is None

    @pytest.mark.parametrize("n_pairs", [2, 3, 6, 40])
    def test_a_uniform_move_on_a_measure_with_no_range_is_not_separated_and_says_why(self, n_pairs) -> None:
        """With no range no test of the mean can call a uniform move, at any n: not separated, naming the remedy.

        Never ``regressed`` (the sign-flip reading called six pairs that), and never ``untested`` either: the rule
        refused the claim, and the reason says what to declare for a test to run.
        """
        verdict = paired_change(
            [1.0] * n_pairs,
            [0.0] * n_pairs,
            min_absolute_change=0.05,
            min_relative_change=0.0,
            higher_is_better=True,
        )
        assert verdict.n_pairs == n_pairs
        assert (verdict.label, verdict.significant, verdict.p_value) == ("not_separated", False, None)
        assert verdict.not_separated_reason == UNIFORM_MOVE_NEEDS_RANGE
        assert verdict.hedges_g is None

    def test_a_uniform_move_over_two_pairs_on_a_range_states_its_p_and_is_not_separated(self) -> None:
        """The bounded test runs at every n; over two pairs its p cannot reach α, and it is stated."""
        verdict = paired_change(
            [1.0, 1.0],
            [0.0, 0.0],
            min_absolute_change=0.05,
            min_relative_change=0.0,
            higher_is_better=True,
            value_range=(0.0, 1.0),
        )
        assert verdict.label == "not_separated"
        assert verdict.p_value is not None and verdict.p_value > SIGNIFICANCE_ALPHA

    def test_the_counterexample_null_cannot_read_twenty_uniform_cases_below_their_chance(self) -> None:
        """A judge whose mean did not move (+1 four times in five, −4 the fifth) gives twenty cases at +1 one time
        in 87. A valid p for that outcome is at least that; the sign-flip p said one in 524,288 (#597)."""
        baseline = [float(1 + i % 4) for i in range(20)]
        current = [x + 1.0 for x in baseline]
        verdict = paired_change(
            baseline,
            current,
            min_absolute_change=0.0,
            min_relative_change=0.0,
            higher_is_better=True,
            value_range=(1.0, 5.0),
        )
        assert verdict.p_value is not None and verdict.p_value >= 0.8**20 > 2.0**-19

    def test_a_deterministic_no_change_is_not_separated_not_untested(self) -> None:
        """Two identical paired samples (>= 2 pairs) are measured, and read 'not_separated' without a margin.

        Zero difference-variance makes the paired t-test undefined, but a perfect
        no-change is a definite result, not too-few-data: ``composite_significance``
        reports it as ``(0.0, not-significant)``. 'untested' is reserved for
        what no test can decide. Three identical pairs with no declared margin still
        claim no stability: only an equivalence test against a margin may (#592).
        """
        verdict = paired_change(
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            min_absolute_change=0.05,
            min_relative_change=0.10,
            higher_is_better=True,
        )
        assert verdict.label == "not_separated"
        assert verdict.significant is False
        assert verdict.delta == 0.0
        assert verdict.n_pairs == 3

    def test_fewer_than_two_pairs_is_untested_never_a_measured_reading(self) -> None:
        """One pair cannot be tested; it must not masquerade as a measured 'no change'."""
        verdict = paired_change([0.8], [0.4], min_absolute_change=0.0, min_relative_change=0.0, higher_is_better=True)
        assert verdict.label == "untested"
        assert verdict.significant is None
        assert verdict.n_pairs == 1

    def test_no_pairs_is_untested_with_null_delta(self) -> None:
        verdict = paired_change([], [], min_absolute_change=0.0, min_relative_change=0.0, higher_is_better=True)
        assert verdict.label == "untested"
        assert verdict.delta is None
        assert verdict.n_pairs == 0

    def test_a_verdict_carries_the_p_it_was_thresholded_against(self) -> None:
        """A label a reader cannot check against its own number is an assertion, not a report."""
        verdict = paired_change(
            [0.80, 0.82, 0.78, 0.81, 0.79],
            [0.60, 0.61, 0.59, 0.62, 0.58],
            min_absolute_change=0.05,
            min_relative_change=0.0,
            higher_is_better=True,
        )
        assert verdict.p_value is not None
        assert verdict.significant is (verdict.p_value < SIGNIFICANCE_ALPHA)

    @pytest.mark.parametrize(
        ("baseline", "current", "expected_label"),
        [
            ([0.8], [0.4], "untested"),  # one pair: the test is undefined
            ([], [], "untested"),  # no pairs at all
            # Every case moved by the same amount on a measure with no range: no test of the mean
            # ran, so the move is not separated and no p is stated (#597).
            ([1.0] * 4, [0.0] * 4, "not_separated"),
        ],
    )
    def test_a_label_no_t_test_produced_reports_no_p(self, baseline, current, expected_label) -> None:
        """The absence is the disclosure: it says the label did not come from a test."""
        verdict = paired_change(
            baseline, current, min_absolute_change=0.05, min_relative_change=0.0, higher_is_better=True
        )
        assert verdict.label == expected_label
        assert verdict.p_value is None

    def test_a_constant_shift_written_in_decimals_is_read_exactly_not_as_a_float_residue(self) -> None:
        """``i/10`` against ``i/10 + 0.5``: every case moved by exactly 0.5, though the floats differ by a hair.

        Over floats the differences carry a residue a t-test reads as a tiny, perfectly consistent spread,
        with a p near 1e-113; read exactly, there is no spread, and on the declared range the bounded test
        reads it — the same p :func:`separation_p` states for the same values.
        """
        baseline = [i / 10 for i in range(8)]
        current = [i / 10 + 0.5 for i in range(8)]
        verdict = paired_change(
            baseline,
            current,
            min_absolute_change=0.05,
            min_relative_change=0.0,
            higher_is_better=True,
            value_range=(0.0, 1.5),
        )

        assert verdict.p_value == separation_p(baseline, current, paired=True, value_range=(0.0, 1.5))
        assert verdict.p_value is not None and verdict.p_value > 1e-6, "never the residue's p"
        assert verdict.hedges_g is None, "no spread, so no finite effect size"
        assert verdict.delta == 0.5

    def test_the_same_shift_with_no_range_is_not_separated_not_a_residue_p(self) -> None:
        baseline = [i / 10 for i in range(4)]
        current = [i / 10 + 0.5 for i in range(4)]
        verdict = paired_change(
            baseline, current, min_absolute_change=0.05, min_relative_change=0.0, higher_is_better=True
        )

        assert (verdict.label, verdict.p_value, verdict.significant) == ("not_separated", None, False)
        assert verdict.not_separated_reason == UNIFORM_MOVE_NEEDS_RANGE

    def test_misaligned_samples_are_a_pairing_bug_not_missing_data(self) -> None:
        with pytest.raises(ValueError, match="aligned"):
            paired_change([0.8, 0.7], [0.6], min_absolute_change=0.0, min_relative_change=0.0, higher_is_better=True)

    def test_carries_the_numbers_a_surface_discloses(self) -> None:
        verdict = paired_change(
            [0.80, 0.80, 0.80, 0.80],
            [0.40, 0.40, 0.40, 0.40],
            min_absolute_change=0.05,
            min_relative_change=0.0,
            higher_is_better=True,
        )
        assert verdict.delta == pytest.approx(-0.40)
        assert verdict.relative_delta == pytest.approx(-0.50)
        assert verdict.n_pairs == 4


class TestEquivalence:
    """The TOST against the measure's declared margin — the only route to a claim of no meaningful change (#592)."""

    @staticmethod
    def _change(baseline: list[float], current: list[float], *, margin: float | None, gate: float = 0.0):
        return paired_change(
            baseline,
            current,
            min_absolute_change=gate,
            min_relative_change=0.0,
            higher_is_better=True,
            equivalence_margin=margin,
        )

    @pytest.mark.parametrize(
        "diffs",
        [
            [0.01, -0.02, 0.0, 0.015, -0.005],
            [0.05, 0.09, 0.02, 0.07, 0.04, 0.06],
            [0.2, -0.3, 0.1, 0.4, -0.2],
            [0.08, 0.11, 0.09],
            [0.0001, -0.0001] * 20,
        ],
        ids=["centred", "near the edge", "noisy", "three pairs", "forty tight pairs"],
    )
    def test_a_margin_with_no_declared_range_is_never_equivalent_and_names_the_remedy(self, diffs) -> None:
        """No test of a mean holds α without a range, so the margin is not tested at all (#695) — even forty pairs
        a hair apart, which the paired t TOST this replaced called equivalent, read no closer than not separated."""
        n = len(diffs)

        verdict = self._change([0.5] * n, [0.5 + d for d in diffs], margin=0.1)

        assert verdict.label != "equivalent" and verdict.equivalence_p is None
        assert verdict.equivalence_untested_reason == EQUIVALENCE_NEEDS_RANGE
        assert verdict.equivalence_untested_reason.startswith("declare value_range")

    def test_a_significant_move_under_the_gate_and_inside_the_margin_reads_equivalent(self) -> None:
        """A real but immaterial move, precisely measured on a declared range: the one shape that earns 'no
        meaningful change'."""
        verdict = paired_change(
            [0.80] * 12,
            [0.81, 0.812, 0.808, 0.811, 0.809, 0.81] * 2,
            min_absolute_change=0.05,
            min_relative_change=0.0,
            higher_is_better=True,
            equivalence_margin=0.05,
            value_range=(0.75, 0.85),
        )

        assert verdict.significant is True and verdict.exceeds_threshold is False
        assert verdict.label == "equivalent" and verdict.equivalence_untested_reason is None

    def test_without_a_margin_no_test_runs_and_no_label_claims_equivalence(self) -> None:
        verdict = self._change([0.80] * 6, [0.81, 0.812, 0.808, 0.811, 0.809, 0.81], margin=None, gate=0.05)

        assert verdict.label == "below_threshold"
        assert verdict.equivalence_p is None and verdict.equivalence_margin is None

    def test_a_directional_label_the_gate_earns_is_kept_and_its_equivalence_p_still_carried(self) -> None:
        """With the gate off, a significant move is flagged even inside the margin — the caller asked for it — and the TOST p rides along."""
        verdict = paired_change(
            [0.80] * 12,
            [0.81, 0.812, 0.808, 0.811, 0.809, 0.81] * 2,
            min_absolute_change=0.0,
            min_relative_change=0.0,
            higher_is_better=True,
            equivalence_margin=0.05,
            value_range=(0.75, 0.85),
        )

        assert verdict.label == "improved"
        assert verdict.equivalence_p is not None and verdict.equivalence_p < SIGNIFICANCE_ALPHA

    def test_identical_samples_with_no_declared_range_are_not_tested_for_equivalence(self) -> None:
        """Zero spread has no t, and with no range nothing bounds a case that could have moved unseen (#693)."""
        verdict = self._change([0.5] * 30, [0.5] * 30, margin=0.05)

        assert verdict.label == "not_separated"
        assert verdict.equivalence_p is None

    @pytest.mark.parametrize(("n_pairs", "label"), [(11, "not_separated"), (12, "equivalent")])
    def test_identical_samples_on_a_declared_range_are_equivalent_once_the_bounded_test_rejects(
        self, n_pairs, label
    ) -> None:
        """On a 0-1 range at margin 0.25, a hidden full drop in one case of four survives n agreeing cases 0.75^n of
        the time; the bounded test needs twelve (any valid test, eleven)."""
        verdict = paired_change(
            [0.5] * n_pairs,
            [0.5] * n_pairs,
            min_absolute_change=0.0,
            min_relative_change=0.0,
            higher_is_better=True,
            equivalence_margin=0.25,
            value_range=(0.0, 1.0),
        )

        assert verdict.label == label
        assert verdict.equivalence_p is not None and verdict.equivalence_p >= 0.75**n_pairs

    def test_fewer_than_two_pairs_runs_no_equivalence_test(self) -> None:
        verdict = self._change([0.5], [0.5], margin=0.05)

        assert verdict.label == "untested"
        assert verdict.equivalence_p is None


class TestTCriticalTwoSided:
    """The interval multiplier — the reason a small-n interval is not 1.96 wide."""

    @pytest.mark.parametrize(
        ("df", "expected"),
        [(1, 12.706), (2, 4.303), (5, 2.571), (10, 2.228), (30, 2.042), (1000, 1.962)],
        ids=["df1", "df2", "df5", "df10", "df30", "df1000"],
    )
    def test_matches_the_published_95_percent_table(self, df, expected):
        assert t_critical_two_sided(0.95, df) == pytest.approx(expected, abs=0.005)

    def test_approaches_the_normal_multiplier_only_at_large_df(self):
        """The whole point: 1.96 is the limit, not the value at the n an eval arm produces."""
        assert t_critical_two_sided(0.95, 2) > 4.0
        assert t_critical_two_sided(0.95, 100000) == pytest.approx(1.96, abs=0.01)

    def test_a_wider_interval_needs_a_larger_multiplier(self):
        assert t_critical_two_sided(0.99, 10) > t_critical_two_sided(0.95, 10)

    @pytest.mark.parametrize(("confidence", "df"), [(0.95, 0), (0.95, -1), (0.0, 5), (1.0, 5)])
    def test_rejects_inputs_with_no_answer(self, confidence, df):
        with pytest.raises(ValueError):
            t_critical_two_sided(confidence, df)


class TestDisclosedTestNames:
    """Every verdict rides beside the name of the test that produced it."""

    @pytest.mark.parametrize("name", [PAIRED_TEST_NAME, UNPAIRED_TEST_NAME])
    def test_each_name_states_the_threshold_it_applied(self, name: str) -> None:
        """A test name without its α leaves the reader unable to place the verdict."""
        assert f"α={SIGNIFICANCE_ALPHA}" in name

    def test_the_paired_and_unpaired_names_are_distinguishable(self) -> None:
        """A paired test over shared cases and an unpaired one answer different questions."""
        assert PAIRED_TEST_NAME != UNPAIRED_TEST_NAME
        assert "paired" in PAIRED_TEST_NAME
        assert "Welch" in UNPAIRED_TEST_NAME


class TestLevelDifference:
    """The between-level test the divergence lens and the mechanism check read: t where it exists, exact where not."""

    def test_shared_cases_are_paired_and_match_the_paired_t_test(self) -> None:
        a = {"c1": 1.0, "c2": 2.0, "c3": 4.0, "c4": 3.5}
        b = {"c1": 1.6, "c2": 2.1, "c3": 5.0, "c4": 3.9}
        tested = level_difference(a, b)
        reference = composite_significance(list(a.values()), list(b.values()), paired=True)
        assert (tested.test, tested.n_a, tested.n_b) == ("paired", 4, 4)
        assert tested.p_value == pytest.approx(reference.p_value)
        assert tested.separated == reference.significant
        diffs = [b[case] - a[case] for case in a]
        mean = sum(diffs) / 4
        assert tested.se == pytest.approx(math.sqrt(sum((d - mean) ** 2 for d in diffs) / 3) / 2)

    def test_disjoint_cases_are_welch_s(self) -> None:
        a = {"c1": 1.0, "c2": 2.0, "c3": 4.0}
        b = {"d1": 3.0, "d2": 5.5, "d3": 4.0, "d4": 6.0}
        tested = level_difference(a, b)
        reference = composite_significance(list(a.values()), list(b.values()), paired=False)
        assert tested.test == "unpaired"
        assert tested.p_value == pytest.approx(reference.p_value)
        assert tested.delta == pytest.approx(sum(b.values()) / 4 - sum(a.values()) / 3)

    def test_one_case_a_side_is_untested(self) -> None:
        tested = level_difference({"c1": 1.0}, {"c1": 2.0, "c2": 3.0})
        assert (tested.separated, tested.p_value, tested.test) == (None, None, None)
        assert tested.untested_reason == "fewer than two cases on a side"

    @pytest.mark.parametrize("n", [2, 6, 40])
    def test_an_alike_shift_with_no_range_is_not_separated_and_says_why(self, n: int) -> None:
        """Every case moving by one amount leaves no t, and with no range no test of the mean can call it (#597).

        The sign-flip p ``2^(1-n)`` this once read called six cases separated; it tests symmetry, not the mean.
        """
        a = {f"c{i}": Fraction(i, 10) for i in range(n)}
        b = {case: value + Fraction(1, 10) for case, value in a.items()}
        tested = level_difference(a, b)
        assert (tested.separated, tested.p_value, tested.untested_reason) == (False, None, None)
        assert tested.not_separated_reason == UNIFORM_MOVE_NEEDS_RANGE

    @pytest.mark.parametrize("n", [2, 6, 12])
    def test_an_alike_shift_on_a_declared_range_is_read_by_the_bounded_test(self, n: int) -> None:
        a = {f"c{i}": Fraction(i, 10) for i in range(n)}
        b = {case: value + Fraction(1) for case, value in a.items()}
        tested = level_difference(a, b, value_range=(0.0, 3.0))
        expected = bounded_separation_p(list(a.values()), list(b.values()), paired=True, value_range=(0.0, 3.0))
        assert expected is not None
        assert tested.p_value == expected
        assert tested.separated is (expected < SIGNIFICANCE_ALPHA)
        assert tested.not_separated_reason is None

    def test_the_counterexample_null_cannot_read_twenty_alike_cases_below_their_chance(self) -> None:
        """Twenty cases each up one point on 1-5 arise one time in 87 from a mean that did not move (+1 four times
        in five, −4 the fifth); a valid p is at least that, where the sign flip said one in 524,288 (#597)."""
        a = {f"c{i}": float(1 + i % 4) for i in range(20)}
        b = {case: value + 1.0 for case, value in a.items()}
        tested = level_difference(a, b, value_range=(1.0, 5.0))
        assert tested.p_value is not None and tested.p_value >= 0.8**20 > 2.0**-19

    def test_an_alike_shift_of_decimals_is_decided_exactly(self) -> None:
        """0.1→0.3 and 0.2→0.4 are one shift; in floats they differ in the last place, which a t would read."""
        a = {"c1": Fraction("0.1"), "c2": Fraction("0.2")}
        b = {"c1": Fraction("0.3"), "c2": Fraction("0.4")}
        assert level_difference(a, b).not_separated_reason == UNIFORM_MOVE_NEEDS_RANGE

    @pytest.mark.parametrize(("n_a", "n_b"), [(2, 2), (4, 4), (3, 5), (12, 12)])
    def test_two_constants_are_read_by_the_bounded_test_or_not_called(self, n_a: int, n_b: int) -> None:
        """Two unpaired constants: the bounded test on a declared range, and with none not separated."""
        a = {f"a{i}": 0.0 for i in range(n_a)}
        b = {f"b{i}": 1.0 for i in range(n_b)}
        unranged = level_difference(a, b)
        assert (unranged.separated, unranged.p_value) == (False, None)
        assert unranged.not_separated_reason == UNIFORM_MOVE_NEEDS_RANGE
        ranged = level_difference(a, b, value_range=(0.0, 1.0))
        expected = bounded_separation_p(list(a.values()), list(b.values()), paired=False, value_range=(0.0, 1.0))
        assert expected is not None and ranged.p_value == expected
        assert ranged.separated is (expected < SIGNIFICANCE_ALPHA)

    def test_identical_values_are_tested_and_not_separated(self) -> None:
        tested = level_difference({"c1": 3.0, "c2": 3.0}, {"c1": 3.0, "c2": 3.0})
        assert (tested.separated, tested.p_value, tested.delta) == (False, 1.0, 0.0)

    def test_equivalence_needs_a_margin_a_declared_range_and_shared_cases(self) -> None:
        a = {f"c{i}": 0.05 + 0.09 * i for i in range(10)}
        b = {case: value + (0.01 if int(case[1:]) % 2 else -0.01) for case, value in a.items()}
        on_range = {"equivalence_margin": 0.5, "value_range": (0.0, 1.0)}
        assert level_difference(a, b).equivalent is None
        assert level_difference(a, b, **on_range).equivalent is True
        assert level_difference(a, b, equivalence_margin=0.5).equivalent is None, "no range, no test (#695)"
        disjoint = {f"d{i}": value for i, value in enumerate(b.values())}
        assert level_difference(a, disjoint, **on_range).equivalent is None


class TestSeparationPAgreesWithLevelDifference:
    """One concept, one answer: the frontier's separation p and the between-level test's p for one pattern."""

    @pytest.mark.parametrize(("n_a", "n_b"), [(2, 2), (3, 3), (4, 4), (3, 5), (2, 9)])
    def test_two_unpaired_constants_read_one_answer(self, n_a: int, n_b: int) -> None:
        """Each side constant, the two different: the bounded p on a range, and no p and the same refusal without."""
        a, b = [1.0] * n_a, [2.0] * n_b
        tested = level_difference({f"a{i}": 1.0 for i in range(n_a)}, {f"b{i}": 2.0 for i in range(n_b)})
        assert separation_p(a, b, paired=False) is None and tested.p_value is None
        assert separation_test(a, b, paired=False).refusal == tested.not_separated_reason == UNIFORM_MOVE_NEEDS_RANGE
        ranged = level_difference(
            {f"a{i}": 1.0 for i in range(n_a)}, {f"b{i}": 2.0 for i in range(n_b)}, value_range=(0.0, 3.0)
        )
        assert separation_p(a, b, paired=False, value_range=(0.0, 3.0)) == ranged.p_value is not None

    @pytest.mark.parametrize("n", [2, 5, 6, 8])
    def test_an_alike_paired_shift_reads_one_answer(self, n: int) -> None:
        # Exactly representable, so the differences carry no float residue for a t statistic to read.
        a = [float(i) for i in range(n)]
        b = [value + 0.5 for value in a]
        tested = level_difference(dict(enumerate(a)), dict(enumerate(b)))
        assert separation_p(a, b, paired=True) is None and tested.p_value is None and tested.separated is False
        ranged = level_difference(dict(enumerate(a)), dict(enumerate(b)), value_range=(0.0, 10.0))
        assert separation_p(a, b, paired=True, value_range=(0.0, 10.0)) == ranged.p_value is not None

    @pytest.mark.parametrize("n", [3, 6, 8, 12])
    def test_a_constant_shift_with_float_residue_is_read_exactly(self, n: int) -> None:
        """``i/10`` against ``i/10 + 0.5``: the float differences are not all 0.5, the decimal ones are.

        Over floats the t-test read the residue as a tiny, perfectly consistent spread and returned a p near
        1e-113. Read exactly there is no spread: the bounded test on a range, and no test without one, on both paths.
        """
        a = [i / 10 for i in range(n)]
        b = [i / 10 + 0.5 for i in range(n)]
        assert len({y - x for x, y in zip(a, b)}) > 1, "the fixture must carry float residue to test anything"
        tested = level_difference(dict(enumerate(a)), dict(enumerate(b)), value_range=(0.0, 2.0))
        p = separation_p(a, b, paired=True, value_range=(0.0, 2.0))
        assert p is not None and p == tested.p_value and p > 1e-6
        assert separation_p(a, b, paired=True) is None

    @pytest.mark.parametrize("paired", [True, False])
    def test_identical_values_read_one(self, paired: bool) -> None:
        """No gap and no spread: the exact p is 1, as level_difference states it."""
        assert separation_p([3.0, 3.0, 3.0], [3.0, 3.0, 3.0], paired=paired) == 1.0
        assert level_difference({"c1": 3.0, "c2": 3.0}, {"c1": 3.0, "c2": 3.0}).p_value == 1.0

    def test_a_t_test_p_is_the_level_difference_p(self) -> None:
        a = {"c1": 1.0, "c2": 2.0, "c3": 4.0}
        b = {"d1": 3.0, "d2": 5.5, "d3": 4.0, "d4": 6.0}
        assert separation_p(list(a.values()), list(b.values()), paired=False) == pytest.approx(
            level_difference(a, b).p_value
        )

    def test_one_value_a_side_has_no_p(self) -> None:
        assert separation_p([1.0], [2.0, 2.0], paired=False) is None
        assert separation_p([1.0], [2.0], paired=True) is None


def test_a_difference_interval_stays_inside_the_differences_a_declared_range_allows() -> None:
    """Three of six pass/fail cases flip: the t interval runs to 1.075, past the largest difference two rates can have.

    Clipped to ± the range's width, it still covers whatever the unclipped one covered (the true difference is
    inside the clip), and zero is excluded exactly when it was.
    """
    control, contrast = [0.0] * 6, [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    low, high = difference_interval(control, contrast, paired=True) or (0.0, 0.0)
    assert high > 1.0, "the unclipped t interval runs past the range"
    assert difference_interval(control, contrast, paired=True, value_range=(0.0, 1.0)) == (low, 1.0)
    assert difference_interval([1.0] * 6, [0.0, 0.0, 0.0, 1.0, 1.0, 1.0], paired=True, value_range=(0.0, 1.0)) == (
        -1.0,
        -low,
    )
    on_a_wider_scale = difference_interval(control, contrast, paired=True, value_range=(1.0, 5.0))
    assert on_a_wider_scale == (low, high), "a 1-5 difference can reach 4"
