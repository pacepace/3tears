"""The one number rule — what it writes at both ends of a value's range, and that it is the only one.

What is here is the rule's two ENDS stated as properties rather than cases, the signed form, and the canary that keeps a
second implementation from growing back: two formatters in one engine is how one analysis came to
print a latency as ``1.235e+04`` beside an interval that spelled the same magnitude ``12,346``. The walk that finds
every site deciding a figure's form outside the rule, and the list of those right for good, are
``test_one_owner_of_number_forms.py``.
"""

from __future__ import annotations

import ast
import math
import re
from pathlib import Path

import pytest

from threetears.evals.analysis import latency_partition, references, significance
from threetears.evals.kernel import numbers
from threetears.evals.kernel.numbers import (
    ABSENT,
    WHOLE_FROM,
    WRITTEN_FROM,
    format_number,
    format_signed,
    round_half_even,
    spell_effect,
    spell_p,
)
from threetears.evals.analysis.report import forms as report_forms
from threetears.evals.analysis import quantities
from threetears.evals.schema.models import LatencyMetrics


_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"


class TestTheLargeEnd:
    @pytest.mark.parametrize("value", [1000.5, 12345.6, 99999.99, 1234567.89, 9.87654321e12, -54321.9])
    def test_a_large_value_never_takes_an_exponent(self, value):
        assert "e" not in format_number(value)

    @pytest.mark.parametrize(("value", "text"), [(12345.6, "12346"), (1000.4, "1000"), (-54321.9, "-54322")])
    def test_a_large_value_is_rounded_to_the_integer(self, value, text):
        assert format_number(value) == text

    def test_the_line_is_where_four_figures_reach_the_units_digit(self):
        """Just under it, four significant figures; at it, the integer — the two agree on the digits."""
        assert format_number(WHOLE_FROM - 0.5) == "999.5"
        assert format_number(WHOLE_FROM + 0.25) == "1000"

    @pytest.mark.parametrize("value", [50300.0, 50300.5, 1234567.0, 1e21])
    def test_no_thousands_separator(self, value):
        """These numbers sit inside `CI [low, high]` and `a; b` lists, where a comma reads as an item."""
        assert "," not in format_number(value)


class TestTheSmallEnd:
    @pytest.mark.parametrize("value", [0.000042, 1e-9, -3.3e-7, 5e-324, 3.6e-169])
    def test_a_small_non_zero_value_never_reads_as_zero(self, value):
        text = format_number(value)
        assert text not in {"0", "-0"} and float(text.lstrip("<> ")) != 0.0, text

    @pytest.mark.parametrize(
        ("value", "text"), [(3.6e-169, "< 0.000001"), (-3.3e-7, "> -0.000001"), (9.9999e-7, "< 0.000001")]
    )
    def test_below_a_millionth_a_value_reads_as_the_bound_it_is_under(self, value, text):
        """A p of 3.6e-169 written out is a line of zeros no reader counts; the bound is what a reader acts on."""
        assert format_number(value) == text
        assert WRITTEN_FROM == 1e-6

    def test_it_is_four_significant_figures_in_the_middle(self):
        assert format_number(0.0123456) == "0.01235"
        assert format_number(12.3456) == "12.35"

    @pytest.mark.parametrize("value", [0.000042, 5.165e-05, 1.2345e-05, 1e-9, -3.3e-7, 9.99996e-5, 3.6e-169])
    def test_a_small_value_never_takes_an_exponent(self, value):
        """A mean beside its sem read ``0.0005306 ± 5.165e-05``: two spellings of one magnitude on one line."""
        assert "e" not in format_number(value)

    @pytest.mark.parametrize(
        ("value", "text"),
        [
            (5.165e-05, "0.00005165"),
            (0.0005306, "0.0005306"),
            (1.2345e-05, "0.00001234"),
            (-0.000042, "-0.000042"),
            (9.99996e-5, "0.0001"),
        ],
    )
    def test_a_small_value_is_four_significant_figures_written_out(self, value, text):
        assert format_number(value) == text


class TestWholeAndAbsent:
    @pytest.mark.parametrize(
        ("value", "text"), [(0.0, "0"), (-0.0, "0"), (42.0, "42"), (1e21, "1000000000000000000000")]
    )
    def test_a_whole_number_is_written_whole_at_any_magnitude(self, value, text):
        assert format_number(value) == text

    @pytest.mark.parametrize("value", [None, math.nan, math.inf, -math.inf])
    def test_an_absent_or_non_finite_value_is_a_dash_never_a_zero(self, value):
        assert format_number(value) == ABSENT == "—"


class TestTheSignedForm:
    @pytest.mark.parametrize(
        ("value", "text"),
        [
            (1.5, "+1.5"),
            (-1.5, "-1.5"),
            (0.0, "0"),
            (12345.6, "+12346"),
            (-0.000042, "-0.000042"),
            (3.6e-169, "< 0.000001"),
            (None, "—"),
            (math.nan, "—"),
        ],
    )
    def test_a_change_carries_its_direction_and_the_one_spelling(self, value, text):
        assert format_signed(value) == text


class TestTheSignedFormWithAUnit:
    """A change joined to its unit carries its sign by the one signed spelling, bound and all."""

    @pytest.mark.parametrize(
        ("value", "unit", "text"),
        [
            (1.5, "ms", "+1.5 ms"),
            (-1.5, "ms", "-1.5 ms"),
            (0.0, "ms", "0 ms"),
            (3.6e-7, "ms", "< 0.000001 ms"),
            (-3.6e-7, "ms", "> -0.000001 ms"),
            (3.6e-7, "%", "< 0.000001%"),
            (None, "ms", "—"),
        ],
    )
    def test_a_change_in_a_unit_reads_as_format_signed_joined_to_it(self, value, unit, text):
        assert quantities.signed_with_unit(value, unit) == text

    @pytest.mark.parametrize(
        ("value", "signed", "text"),
        [
            (0.003092, False, "$0.003092"),
            (-0.0038, True, "-$0.0038"),
            (0.0002, True, "+$0.0002"),
            (-3.6e-7, True, "> -$0.000001"),
            (None, False, "—"),
        ],
    )
    def test_money_is_written_with_its_currency_sign_as_the_scoreboard_writes_it(self, value, signed, text):
        said = quantities.signed_with_unit(value, "usd") if signed else quantities.with_unit(value, "usd")
        assert said == text and "usd" not in said

    def test_no_signed_spelling_signs_a_bound(self):
        """``+< 0.000001 ms`` is no spelling: a bound already says which side of zero it is on."""
        for value in (3.6e-7, -3.6e-7):
            for unit in ("ms", "%", ""):
                text = quantities.signed_with_unit(value, unit)
                assert not re.search(r"[+-][<>]", text), text


class TestOneRulerPerQuantity:
    """A quantity is stated in the unit a reader thinks in, chosen once over every value it is stated with."""

    @pytest.mark.parametrize(
        ("values", "unit", "scaled"),
        [([900.0, 1200.0], "ms", (0.001, "s")), ([450.0], "ms", (1.0, "ms")), ([5.0], None, (1.0, ""))],
    )
    def test_display_scale_picks_one_unit_for_every_value(self, values, unit, scaled):
        assert quantities.display_scale(values, unit) == scaled


class TestThePrefixedForm:
    """A prefix — a currency symbol — goes on the number: after its sign, inside a bound."""

    @pytest.mark.parametrize(
        ("value", "text"),
        [
            (1.23, "$1.23"),
            (-1.23, "-$1.23"),
            (0.0, "$0"),
            (-0.0, "$0"),
            (12345.6, "$12346"),
            (-12345.6, "-$12346"),
            (3.6e-169, "< $0.000001"),
            (-3.3e-7, "> -$0.000001"),
            (1e-6, "$0.000001"),
            (None, ABSENT),
            (math.inf, ABSENT),
        ],
    )
    def test_the_prefix_is_on_the_number(self, value, text):
        assert format_number(value, prefix="$") == text

    @pytest.mark.parametrize(
        "value", [1.23, -1.23, 0.0, 42.0, -42.0, 12345.6, -12345.6, 3.6e-169, -3.3e-7, 5.165e-5, None, math.nan]
    )
    def test_an_empty_prefix_is_the_text_without_one(self, value):
        assert format_number(value, prefix="") == format_number(value)

    @pytest.mark.parametrize("value", [1.23, -1.23, 3.6e-169, -3.3e-7, 12345.6])
    def test_a_prefix_never_lands_on_a_sign_or_a_comparison(self, value):
        assert not re.search(r"\$[-<>]", format_number(value, prefix="$"))


class TestHalfToEven:
    """The one rounding every decided figure goes through: half to even, on the value as stored."""

    @pytest.mark.parametrize(
        ("value", "places", "text"), [(0.125, 2, "0.12"), (0.375, 2, "0.38"), (2.5, 0, "2"), (3.5, 0, "4")]
    )
    def test_a_tie_rounds_to_the_even_digit(self, value, places, text):
        assert round_half_even(value, places) == text

    def test_it_rounds_the_stored_value_not_its_shortest_spelling(self):
        """2.675 is stored a hair under its spelling, so it rounds down, as every other figure on the page does."""
        assert round_half_even(2.675, 2) == "2.67"

    @pytest.mark.parametrize(("value", "text"), [(-0.04, "0.0"), (-0.0, "0.0"), (0.04, "0.0")])
    def test_a_value_that_rounds_to_zero_carries_no_sign(self, value, text):
        assert round_half_even(value, 1) == text

    def test_a_negative_value_keeps_its_sign(self):
        assert round_half_even(-1.25, 1) == "-1.2"

    def test_no_places_reads_whole(self):
        assert round_half_even(84.6, 0) == "85"


class TestAPValue:
    """A p-value as APA spells it: under a thousandth as the bound, three decimals below 0.1, two from it up."""

    @pytest.mark.parametrize(
        ("p", "text"),
        [
            (3.6e-169, "< 0.001"),
            (0.00008342, "< 0.001"),
            (0.0009996, "< 0.001"),
            (0.001, "0.001"),
            (0.0031, "0.003"),
            (0.0994, "0.099"),
            (0.1, "0.10"),
            (0.4996, "0.50"),
            (1.0, "1.00"),
            (0.0, "< 0.001"),
        ],
    )
    def test_it_reads_as_apa_reads_it(self, p, text):
        assert spell_p(p) == text

    @pytest.mark.parametrize("p", [None, math.nan, math.inf])
    def test_an_absent_p_reads_as_absent(self, p):
        assert spell_p(p) == ABSENT


class TestAnEffectSize:
    """An effect size or a correlation reads at two decimals, its sign kept."""

    @pytest.mark.parametrize(
        ("value", "text"), [(0.08475, "0.08"), (0.1228, "0.12"), (-0.194, "-0.19"), (-0.001, "0.00")]
    )
    def test_it_reads_at_two_decimals(self, value, text):
        assert spell_effect(value) == text

    @pytest.mark.parametrize("value", [None, math.nan, -math.inf])
    def test_an_absent_effect_reads_as_absent(self, value):
        assert spell_effect(value) == ABSENT


class TestThereIsOneImplementation:
    def test_every_surface_that_spells_numbers_holds_the_same_function(self):
        """The quantities and a page's forms, which spell every figure the report's builder states, spell their values
        with it; nothing holds a copy. The references spell an evidence row's figures with it too: they are numbers,
        which reach no page's forms (``test_report_layer_direction.py``), so they hold the rule's own function and no
        page's."""
        for module in (quantities, report_forms, references):
            assert module.format_number is numbers.format_number, module.__name__
        assert not hasattr(references, "PageForms")

    def test_no_other_module_defines_a_number_formatter(self):
        """A second ``def format_number`` anywhere in the engine or its surfaces is the defect itself.

        Walks every module of the package rather than a list of the ones that once had a copy,
        because the copy that matters is the next one, in a module nobody thought to list.
        """
        defining = []
        for path in sorted((_SOURCE_ROOT / "threetears").rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            if any(isinstance(node, ast.FunctionDef) and node.name == "format_number" for node in ast.walk(tree)):
                defining.append(path.relative_to(_SOURCE_ROOT).as_posix())
        assert defining == ["threetears/evals/kernel/numbers.py"]


class TestTheWidenedModulesRenderByTheRule:
    """The significance and latency-partition modules' reader-facing spellings, driven where their old specs disagree.

    ``.4g`` and the rule agree on every finite p-value and on any effect size below 1e4, so each
    case sits where they part: the large end, where ``.4g`` and a bare ``:g`` turn to an exponent,
    and a non-finite value, which ``.4g`` spells ``nan`` and the rule spells absent.
    """

    def test_a_large_effect_size_is_written_whole(self):
        assert f"{12345.6:.4g}" == "1.235e+04"
        cell = significance.format_significance(significant=True, paired=True, p=3e-7, effect=12345.6, n=4)
        assert cell == f"{significance.SIGNIFICANT_LABEL} (p={format_number(3e-7)}, d_z=12346, n=4)"

    def test_a_p_value_that_is_not_a_number_reads_as_absent(self):
        """The one input where ``.4g`` and the rule part on a p-value: ``.4g`` prints ``nan``.

        Every finite p lies in [0, 1], where the two spellings agree digit for digit, so this is the
        only render a p routed off the rule can be seen from.
        """
        assert f"{math.nan:.4g}" == "nan"
        cell = significance.format_significance(significant=False, paired=True, p=math.nan, effect=0.5, n=4)
        assert f"p={ABSENT}, " in cell

    def test_a_large_partition_overrun_is_written_whole(self):
        overrun = 1234467.5
        assert f"{overrun:g}" == "1.23447e+06"
        partition = latency_partition.decompose_total_ms(
            LatencyMetrics(total_ms=100.0, llm_ms=overrun + 100.0, tool_ms=0.0)
        )
        assert f"total_ms by {format_number(overrun)}ms, " in partition.withheld
        assert format_number(overrun) == "1234468"


@pytest.mark.parametrize(
    ("value", "text"),
    [(30.0, "30"), (0.25, "0.25"), (1e-07, "1e-07"), (1234567.0, "1.23457e+06"), (-2.5, "-2.5"), (-0.0, "-0")],
)
def test_a_declared_value_reads_as_its_declarer_wrote_it(value: float, text: str) -> None:
    assert numbers.spell_declared(value) == text
