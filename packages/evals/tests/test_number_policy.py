"""The number policy's primitives: each rule a person can hold in their head, held to its own cases.

The policy decides a figure's form from what was measured, once per measure:

1. A measure reads at one precision on the whole page: its levels, changes and interval ends share one count of
   decimals.
2. That precision comes from the measure's uncertainty: the narrowest 95% half-width among its headline intervals,
   printed to its first significant digit, or its second when that digit is a 1 or a 2. A rate or a change in points
   never goes past one decimal, except a figure that would round onto a boundary (zero, a rate's 0% or 100%), which
   takes just enough digits to stay off it. The exception holds for those capped figures and for interval ends alone:
   any other level or change reads at its measure's decimals.
3. An interval prints one way: ``low to high``, a change's ends signed, its unit written once. One with an absent end
   reads as an absent figure.
4. Figures that are no measurement keep fixed forms: a p-value as APA writes it, an effect at two decimals, a declared
   value as declared, an exact zero whole with its unit, a rate always with its ``%``.

Every figure rounds half to even, and an absent or non-finite one reads as an em dash. Rule 5 (every renderer prints the decided text) is the page's, not a primitive's,
and so is the interval level's reading (the joint guarantee in words): neither is held here.

The before-and-after rows are the real pages' figures that the primitives alone can state: Kairo's written page, the
scaled campaign's code page, Discodon's code-only scorecard and rung 5. A row's inputs are representative of the page's
figure (a level printed ``91.5%`` today from a value under it, ``91.47``), not read off a fixture: nothing calls the
primitives yet, so every page still prints as it does today.
"""

from __future__ import annotations

import math

import pytest

from threetears.evals.analysis.report.forms import (
    MeasureForm,
    policy_bounded_places,
    policy_declared,
    policy_effect,
    policy_figure,
    policy_interval,
    policy_p,
    policy_percent,
    policy_places,
    policy_rate,
    policy_zero,
)
from threetears.evals.kernel.numbers import ABSENT, set_places

#: A rate's floor and ceiling, in percent, as a caller names them.
RATE_BOUNDS = (0.0, 100.0)


class TestRuleOneOnePrecisionPerMeasure:
    def test_the_narrowest_interval_of_any_role_sets_every_figure_s_decimals(self):
        """A measure's levels and changes share one precision, so its narrowest headline interval, whichever role it
        belongs to, decides for all of them: a change's wide interval does not coarsen a level read tightly."""
        assert policy_places([5.55, 3.7, 0.6]) == 1
        assert policy_places([0.6, 5.55, 3.7]) == 1


class TestRuleTwoTheDigitsComeFromTheUncertainty:
    @pytest.mark.parametrize(
        ("half_width", "places"),
        [
            (3.7, 0),
            (4.05, 0),
            (0.6, 1),
            (0.05, 2),
            (0.0047, 3),
            (0.35, 1),
            (37.0, 0),
        ],
    )
    def test_a_half_width_reads_to_its_first_significant_digit(self, half_width, places):
        assert policy_places([half_width]) == places

    @pytest.mark.parametrize(("half_width", "places"), [(0.16, 2), (0.25, 2), (1.9, 1), (0.0012, 4)])
    def test_a_half_width_that_leads_with_a_one_or_a_two_reads_to_its_second(self, half_width, places):
        assert policy_places([half_width]) == places

    @pytest.mark.parametrize(("half_width", "places"), [(0.0996, 2), (0.296, 1), (0.996, 1)])
    def test_the_leading_digit_is_read_at_two_significant_figures(self, half_width, places):
        """0.0996 leads with the 1 of 0.10, and 0.296 with the 3 of 0.30."""
        assert policy_places([half_width]) == places

    def test_an_absent_zero_or_non_finite_half_width_says_nothing(self):
        assert policy_places([None, 0.0, math.nan, math.inf, 0.6]) == 1

    def test_a_rate_or_a_change_in_points_reads_to_at_most_one_decimal(self):
        assert policy_places([0.05], capped=True) == 1
        assert policy_places([0.6], capped=True) == 1
        assert policy_places([3.7], capped=True) == 0

    def test_a_measure_with_no_interval_reads_at_two_significant_figures_of_its_least_figure(self):
        figures = [0.0022, 1.3, 0.0]
        assert policy_places([], figures) == set_places(figures) == 4
        assert policy_places([None], figures, capped=True) == 1
        assert policy_places([]) == 0

    def test_the_figures_are_not_read_where_an_interval_is(self):
        assert policy_places([3.7], [0.0022]) == 0

    def test_the_decimals_stop_at_six(self):
        """A half-width finer than a millionth would ask for an eighth decimal; a figure that fine reads as the bound it
        is under (``< 0.000001``), so the decimals stop at six."""
        assert policy_places([3e-8]) == 6
        assert policy_places([3e-6]) == 6
        assert policy_places([3e-5]) == 5


class TestTheBoundaryException:
    """A capped figure or an interval's end that would round onto a boundary takes just enough digits to stay off it
    (owner ruling): zero for a change in points and every end, and a rate's floor and ceiling."""

    def test_a_change_in_points_near_zero_keeps_its_digits_past_the_cap(self):
        assert policy_figure(-0.03, 0, signed=True, bounds=(0.0,)) == "-0.03"
        assert policy_figure(-0.03, 1, signed=True, bounds=(0.0,)) == "-0.03"
        assert policy_figure(0.004, 1, signed=True, bounds=(0.0,)) == "+0.004"

    def test_an_uncapped_figure_reads_at_its_measure_s_decimals(self):
        """``+0.4 ms`` on a whole-millisecond measure reads as rule 2 sets it: the exception is not its."""
        assert policy_figure(0.4, 0, signed=True, bounds=()) == "0"
        assert policy_figure(0.0022, 2, bounds=()) == "0.00"

    def test_a_rate_under_its_ceiling_never_reads_100(self):
        assert policy_rate(99.7, 0) == "99.7%"
        assert policy_rate(99.96, 0) == "99.96%"
        assert policy_rate(100.0, 0) == "100%"

    def test_a_rate_over_its_floor_never_reads_0(self):
        assert policy_rate(0.3, 0) == "0.3%"
        assert policy_rate(0.04, 0) == "0.04%"
        assert policy_rate(0.0, 0) == "0%"

    def test_a_figure_on_a_boundary_reads_at_its_measure_s_decimals(self):
        assert policy_rate(100.0, 1) == "100.0%"
        assert policy_bounded_places(100.0, 1, bounds=RATE_BOUNDS) == 1

    def test_a_figure_off_every_boundary_takes_no_more(self):
        assert policy_bounded_places(85.3, 0, bounds=RATE_BOUNDS) == 0
        assert policy_rate(85.3, 0) == "85%"

    def test_only_a_rate_has_a_ceiling(self):
        """A measured figure (a time, a count per trial) has no 100 to stay off."""
        assert policy_figure(99.7, 0, bounds=()) == "100"

    def test_float_residue_of_a_boundary_reads_as_the_boundary(self):
        """A figure within a millionth of a boundary is that boundary as computed, so no digits chase it."""
        assert policy_rate(99.99999999999999, 0) == "100%"
        assert policy_rate(1e-15, 0) == "0%"
        assert policy_bounded_places(1e-15, 0, bounds=(0.0,)) == 0
        assert policy_figure(1e-15, 1, signed=True, bounds=(0.0,)) == "0"

    def test_a_rate_under_its_floor_reads_as_given(self):
        assert policy_rate(-0.3, 0) == "-0.3%"


class TestRuleThreeOneIntervalForm:
    def test_a_change_s_ends_carry_their_signs_and_its_unit_is_written_once(self):
        assert policy_interval(-12.9, -1.8, 0, change=True, unit="pts", bounds=()) == ("-13", "-2", "-13 to -2 pts")
        assert policy_interval(-3.46, 7.9, 0, change=True, unit="pts", bounds=())[2] == "-3 to +8 pts"

    def test_a_level_s_ends_carry_no_plus(self):
        assert policy_interval(2.37, 2.47, 2, change=False, bounds=()) == ("2.37", "2.47", "2.37 to 2.47")

    def test_a_unit_that_sits_on_the_number_is_written_on_the_high_end_alone(self):
        assert policy_interval(73.2, 91.5, 0, change=False, unit="%", bounds=RATE_BOUNDS)[2] == "73 to 92%"

    def test_under_a_head_that_names_the_unit_none_is_written(self):
        assert policy_interval(-12.9, -1.8, 0, change=True, bounds=())[2] == "-13 to -2"

    def test_an_end_is_kept_off_zero_alone(self):
        """Only the end that would print as zero takes more digits; the other reads at its measure's."""
        assert policy_interval(-15.09, -0.03, 1, change=True, unit="pts", bounds=())[2] == "-15.1 to -0.03 pts"
        assert policy_interval(-15.09, -0.03, 0, change=True, unit="pts", bounds=())[2] == "-15 to -0.03 pts"

    def test_ends_that_would_print_alike_take_more_decimals_together(self):
        assert policy_interval(2.41, 2.44, 1, change=False, bounds=())[2] == "2.41 to 2.44"
        assert policy_interval(2.401, 2.404, 1, change=False, bounds=())[2] == "2.401 to 2.404"

    def test_an_exact_zero_end_reads_whole(self):
        assert policy_interval(0.0, 0.92, 2, change=False, bounds=())[2] == "0 to 0.92"
        assert policy_interval(0.0, 2.5, 1, change=True, bounds=())[2] == "0 to +2.5"
        assert policy_interval(-0.0, 2.5, 1, change=True, bounds=())[2] == "0 to +2.5"

    def test_an_end_near_zero_is_kept_off_it_where_the_interval_spans_zero(self):
        """Every end is kept off zero, whatever its measure: ``0 to +2`` would claim the interval stops at none."""
        assert policy_interval(-0.04, 2.0, 0, change=True, bounds=())[2] == "-0.04 to +2"

    def test_a_rate_s_interval_is_kept_off_its_floor_and_ceiling(self):
        assert policy_interval(95.2, 99.8, 0, change=False, unit="%", bounds=RATE_BOUNDS)[2] == "95 to 99.8%"
        assert policy_interval(0.2, 3.1, 0, change=False, unit="%", bounds=RATE_BOUNDS)[2] == "0.2 to 3%"
        assert policy_interval(97.0, 100.0, 0, change=False, unit="%", bounds=RATE_BOUNDS)[2] == "97 to 100%"

    def test_without_a_rate_s_bounds_only_zero_is_kept(self):
        """Why ``bounds`` is required: a rate's interval passed none would read its ceiling."""
        assert policy_interval(95.2, 99.8, 0, change=False, unit="%", bounds=())[2] == "95 to 100%"

    def test_equal_ends_read_alike(self):
        assert policy_interval(2.0, 2.0, 1, change=False, bounds=())[2] == "2.0 to 2.0"

    def test_ends_apart_by_float_residue_alone_read_alike(self):
        """Ends within a millionth of each other are one figure as computed, so no digits chase their difference."""
        assert policy_interval(2.0, 2.0 + 1e-12, 1, change=False, bounds=())[2] == "2.0 to 2.0"

    def test_an_interval_with_an_absent_end_reads_as_one_absent_figure(self):
        """An em dash with no unit, never ``— pts``: there are no ends to state."""
        assert policy_interval(math.nan, 1.6, 1, change=True, unit="pts", bounds=()) == (ABSENT, ABSENT, ABSENT)
        assert policy_interval(-3.6, math.inf, 1, change=True, bounds=())[2] == ABSENT


class TestRuleFourFixedForms:
    @pytest.mark.parametrize(
        ("p", "text"),
        [(1e-7, "< 0.001"), (0.00008342, "< 0.001"), (0.0034, "0.003"), (0.4996, "0.50"), (1.0, "1.00")],
    )
    def test_a_p_value_reads_as_apa_writes_it(self, p, text):
        assert policy_p(p) == text

    @pytest.mark.parametrize(("effect", "text"), [(0.08475, "0.08"), (0.1228, "0.12"), (-0.19, "-0.19")])
    def test_an_effect_reads_at_two_decimals(self, effect, text):
        assert policy_effect(effect) == text

    def test_a_declared_value_reads_as_declared(self):
        assert policy_declared(-15.0, "pts") == "-15 pts"
        assert policy_declared(0.5) == "0.5"
        assert policy_declared(None, "pts") == ABSENT

    @pytest.mark.parametrize(("value", "text"), [(1234.5, "1234.5"), (0.1, "0.1"), (1e-05, "0.00001"), (-0.0, "0")])
    def test_a_declared_value_is_never_rounded_nor_written_in_an_exponent(self, value, text):
        assert policy_declared(value) == text

    def test_an_exact_zero_reads_whole_with_its_unit(self):
        assert policy_zero("notices") == "0 notices"
        assert policy_zero("%") == "0%"
        assert policy_zero() == "0"
        assert policy_figure(0.0, 2, signed=True, bounds=(0.0,)) == "0"

    def test_a_rate_always_carries_its_percent(self):
        assert policy_rate(0.0, 1) == "0%"
        assert policy_rate(14.29, 0) == "14%"
        assert policy_rate(None, 0) == ABSENT


class TestAnAbsentOrNegativeZeroFigure:
    """An absent or non-finite figure reads as the em dash, never a zero; ``-0.0`` is a zero, with no sign."""

    @pytest.mark.parametrize("value", [None, math.nan, math.inf])
    def test_an_absent_figure_reads_as_an_em_dash(self, value):
        assert policy_figure(value, 1, signed=True, bounds=(0.0,)) == ABSENT
        assert policy_rate(value, 0) == ABSENT
        assert policy_percent(value, 0) == ABSENT

    def test_negative_zero_reads_as_zero(self):
        assert policy_figure(-0.0, 1, signed=True, bounds=(0.0,)) == "0"
        assert policy_rate(-0.0, 1) == "0%"
        assert policy_percent(-0.0, 1) == "0%"


class TestHalfToEven:
    @pytest.mark.parametrize(
        ("value", "places", "text"), [(62.5, 0, "62%"), (63.5, 0, "64%"), (0.125, 2, "0.12%"), (-2.5, 0, "-2%")]
    )
    def test_a_percent_rounds_half_to_even(self, value, places, text):
        assert policy_percent(value, places) == text

    def test_an_exact_zero_percent_states_no_precision(self):
        assert policy_percent(0.0, 2) == "0%"

    def test_a_figure_rounds_half_to_even(self):
        assert policy_figure(2.5, 0, signed=True, bounds=(0.0,)) == "+2"
        assert policy_figure(-3.5, 0, signed=True, bounds=(0.0,)) == "-4"


class TestKairoBeforeAndAfter:
    def test_the_options_table_s_pass_rate(self):
        assert policy_rate(765 / 897 * 100, 0) == "85%"

    def test_the_change_chart_s_head_and_qwen3_8_flash_s_change(self):
        places = policy_places([3.7, 5.55], capped=True)
        assert places == 0
        assert policy_rate(85.3, places) == "85%"
        assert policy_figure(-7.6, places, signed=True, bounds=(0.0,)) == "-8"
        assert policy_interval(-12.9, -1.8, places, change=True, bounds=())[2] == "-13 to -2"

    def test_the_prose_change_reads_as_its_chart_does(self):
        assert policy_figure(-3.0, 0, signed=True, bounds=(0.0,)) == "-3"
        assert policy_interval(-7.2, 2.8, 0, change=True, bounds=())[2] == "-7 to +3"

    def test_the_margin_line(self):
        assert policy_declared(-15.0, "pts") == "-15 pts"

    def test_holds_character_by_arm_and_by_stratum(self):
        places = policy_places([4.05], capped=True)
        assert [policy_rate(rate, places) for rate in (91.47, 85.4, 86.6)] == ["91%", "85%", "87%"]
        assert [policy_rate(rate, places) for rate in (65.8, 42.9, 45.6)] == ["66%", "43%", "46%"]

    def test_the_send_message_check_is_unchanged(self):
        places = policy_places([0.6], capped=True)
        assert places == 1
        assert [policy_rate(rate, places) for rate in (99.7, 98.7, 100.0)] == ["99.7%", "98.7%", "100.0%"]
        assert policy_figure(-1.0, places, signed=True, bounds=(0.0,)) == "-1.0"
        assert policy_interval(-3.6, 1.6, places, change=True, unit="pts", bounds=())[2] == "-3.6 to +1.6 pts"

    def test_failed_reply_notices(self):
        low, high = -0.0069, 0.00248
        places = policy_places([(high - low) / 2])
        assert places == 3
        assert policy_figure(0.0022, places, bounds=()) == "0.002"
        assert policy_figure(-0.0022, places, signed=True, bounds=()) == "-0.002"
        assert policy_interval(low, high, places, change=True, bounds=())[2] == "-0.007 to +0.002"

    def test_adjusted_p_values(self):
        assert [policy_p(p) for p in (1e-7, 0.00008342, 0.4996, 1.0)] == ["< 0.001", "< 0.001", "0.50", "1.00"]

    def test_the_corrected_comparison_interval_writes_its_unit_once(self):
        assert policy_interval(-12.9, -1.8, 0, change=True, unit="pts", bounds=())[2] == "-13 to -2 pts"


class TestScaledCampaignBeforeAndAfter:
    def test_p95_extraction_time(self):
        places = policy_places([0.05, 0.159])
        assert places == 2
        assert policy_figure(2.42, places, bounds=()) == "2.42"
        assert policy_interval(2.37, 2.47, places, change=False, bounds=())[2] == "2.37 to 2.47"
        assert policy_figure(0.028, places, signed=True, bounds=()) == "+0.03"
        assert policy_interval(-0.131, 0.187, places, change=True, bounds=())[2] == "-0.13 to +0.19"

    def test_fields_stripped(self):
        places = policy_places([0.07, 0.187])
        assert places == 2
        assert policy_figure(1.52, places, bounds=()) == "1.52"
        assert policy_interval(1.45, 1.59, places, change=False, bounds=())[2] == "1.45 to 1.59"
        assert policy_figure(0.095, places, signed=True, bounds=()) == "+0.10"
        assert policy_interval(-0.092, 0.282, places, change=True, bounds=())[2] == "-0.09 to +0.28"

    def test_the_scorecard_s_pass_rate(self):
        """Its level and change round apart, so 21% and 23% sit beside +2 pts: the closure the policy accepts."""
        places = policy_places([5.7], capped=True)
        assert places == 0
        assert [policy_rate(rate, places) for rate in (20.53, 23.0)] == ["21%", "23%"]
        assert policy_figure(2.5, places, signed=True, bounds=(0.0,)) == "+2"
        assert policy_interval(-3.46, 7.9, places, change=True, unit="pts", bounds=())[2] == "-3 to +8 pts"

    def test_the_pass_rate_s_contrasts(self):
        """The end that excludes zero takes its digits alone (owner ruling on the boundary exception), where the
        plan's earlier row gave both ends two decimals."""
        assert policy_figure(-7.4, 0, signed=True, bounds=(0.0,)) == "-7"
        assert policy_interval(-15.09, -0.03, 0, change=True, unit="pts", bounds=())[2] == "-15 to -0.03 pts"

    def test_the_statistics(self):
        assert [policy_effect(value) for value in (0.08475, 0.1228)] == ["0.08", "0.12"]


class TestCodeOnlyAndRungFiveBeforeAndAfter:
    def test_discodon_s_code_only_scorecard_reads_its_interval_in_its_level_s_form(self):
        assert policy_rate(84.4, 0) == "84%"
        assert policy_interval(73.2, 91.5, 0, change=False, unit="%", bounds=RATE_BOUNDS)[2] == "73 to 92%"

    def test_a_lost_call_rate_of_none_carries_its_percent(self):
        assert policy_rate(0.0, 0) == "0%"

    def test_rung_five_s_report_finished_rate_reads_in_percent(self):
        assert [policy_rate(100 * passed / 7, 0) for passed in (1, 3)] == ["14%", "43%"]


class TestMeasureFormPolicyFields:
    def test_a_form_built_as_today_holds_no_policy_decision(self):
        """The policy's fields default to none, so every form built today decides as it did."""
        form = MeasureForm(
            key="pass_rate",
            kind="trial",
            reads_as="rate",
            unit="%",
            change_unit="pts",
            factor=100.0,
            change_factor=100.0,
            level_decimals=1,
            change_decimals=1,
        )
        assert form.places is None
        assert form.half_width is None
