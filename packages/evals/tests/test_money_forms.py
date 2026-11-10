"""Money reads one form per set (#993): a key family, a table's column and a chart each spell every amount alike.

Owner ruling of 2026-10-27: within a figure type every value is spelled in one form, so successive rows never alternate
between ``$0.008`` and ``$0.11``. Owner defaults of 2026-11-07 (Q3, Q4), ruled 2026-11-07: two significant figures, one
more only where two would print an interval's ends alike, totals at or over $1 to cents whatever their size; the
per-1,000 mark (``$8/1000``) is retired, the count goes in the header or the axis instead. The form is one rule for the
set, never padding: ``$0.008`` and ``$0.11`` read ``$0.0080`` and ``$0.11``, never ``$0.1100``. Contracts held here:

- :func:`~threetears.evals.kernel.numbers.money_form` chooses the form from the whole set: plain dollars, each member to
  two significant figures, cents at least, and every member to three where two would print an interval's ends alike;
- every figure-key money family, the options table's cost column and a chart's notes and ticks print one form, on r1
  to r5 and Kairo round 2, the expectation computed from each input's own figures; a values table's column keeps its
  amounts' digits, never moved per 1,000;
- a change reads in the form of the amounts it compares, to no more decimals than its interval's width supports (the
  ``+$0.492`` on an interval tens of cents wide reads ``+$0.49``);
- a cost per 1,000 times the declared monthly volume reaches the monthly figure beside it, within the form's rounding;
- the options table states each arm's cost per run (or per 1,000) beside its pass rate, with its price source, the
  provider that served it and its priced share; a partly priced arm says ``n of m calls priced`` in place of a
  smaller total, an arm none of whose calls were priced says ``unpriced``, never ``$0``, and the unanswered mark stays;
- the single-value helpers: ``spell_dollars`` and ``dollars_text`` spell a total in the money form, and
  ``format_number`` keeps the number rule: the number format cases, and Discodon's two call shapes
  (``dollars_text(cost)``, ``format_number(cost, prefix="$")``);
- every chart and table kind that draws a money set under a cent moves its ruler and states it, never stacks its own
  unit's count beside the moved one, and reads its ticks and notes in one form; a total is never scaled, and a chart of
  totals under a cent reads plain dollars on its ticks (the kit's ``usd``), its notes and its values table alike, as a
  key and a table write it; a change chart scales by the amounts it compares;
- the kit's money formats and the package's money form spell each kit money case alike (``kit_money`` in the number
  format cases, which the kit's own reading in headless Chromium pins too; ``usdpk`` stays readable for a chart
  stored with it).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Iterator
from typing import Any, get_args
from pathlib import Path

import pytest

from packages.evals.tests.analyst_support import KAIRO, fixture_inputs, fixture_snapshots
from threetears.evals.analysis import ABSENT, UNPRICED, dollars_text, format_number
from threetears.evals.analysis.analyst.snapshot import frozen_input
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.quantities import MONEY
from threetears.evals.analysis.report.blocks import ChartBlock, TableBlock, TableCell
from threetears.evals.analysis.report.charts import BarRow, BarsChart, GateMatrixChart, LatticeChart
from threetears.evals.analysis.report.block_spec import block_look, complete_chart
from threetears.evals.analysis.report.figure_keys import (
    Figure,
    cost_basis,
    criterion_ruler,
    format_figure,
    render,
    resolve,
)
from threetears.evals.analysis.report_rows import PriceStep, query_rows
from threetears.evals.analysis.report.parts.full_results import every_table
from threetears.evals.analysis.report.parts.look import figure_text, quantity
from threetears.evals.analysis.report.parts.model import PageChart
from threetears.evals.analysis.report.parts.resolve import resolve_chart
from threetears.evals.analysis.report.parts.tables import arms_table
from threetears.evals.kernel.chart_query import ChartQuery, PlanChartKind, PlannedChart
from threetears.evals.analysis.report.cost_unit import MoneyRuler, money_ruler
from threetears.evals.analysis.report_gates import gate_check
from threetears.evals.analysis.report_input import Criterion, Pricing, ReportInput, Unanswered
from threetears.evals.analysis.report_arms import fact_arms
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.kernel.numbers import CENT, MoneyForm, money_form, spell_dollars

#: r1 to r5, and Kairo round 2 frozen from its bundle.
INPUTS: dict[str, ReportInput] = {
    **{name: held for name, held in fixture_inputs().items() if name in ("r1", "r2", "r3", "r4", "r5")},
    KAIRO: frozen_input(fixture_snapshots("s")[KAIRO]),
}

#: An amount of money as a page prints it: its sign, its digits and the per-1,000 mark where it has one.
_AMOUNT = re.compile(r"([+\-]?)\$(\d[\d,]*)(?:\.(\d+))?(/1000)?")

#: The number format cases every renderer reads.
_CASES = Path(__file__).resolve().parent / "fixtures" / "number-format-cases.json"


def _places(texts: Iterable[str]) -> set[tuple[bool, int]]:
    """The spellings the amounts in some texts are written in: per 1,000 or not, and how many decimals."""
    return {
        (bool(found.group(4)), len(found.group(3) or ""))
        for text in texts
        for found in _AMOUNT.finditer(text)
        if found.group(2) != "0" or found.group(3)
    }


#: What an amount from a dollar up says of its set's form: to cents (``$3.48``, ``$1,208.00``), or, a rate's from $10,
#: in whole dollars (``$75``); never one decimal (owner ruling Q3, 2026-11-07).
_CENTS, _WHOLE, _TEN_CENTS = "cents", "whole", "cents from $10"


def _forms(texts: Iterable[str]) -> set[tuple[bool, int | str]]:
    """The forms the amounts in some texts are written in: per 1,000 or not, and the significant figures an amount
    under a dollar states (``$0.0080`` and ``$0.11`` both two); an amount from a dollar up reads :data:`_CENTS`, or
    :data:`_WHOLE` or :data:`_TEN_CENTS` from $10, and any other decimals as a form of its own."""
    forms: set[tuple[bool, int | str]] = set()
    for text in texts:
        for found in _AMOUNT.finditer(text):
            whole, fraction = found.group(2).replace(",", ""), found.group(3) or ""
            if whole == "0" and not fraction.strip("0"):
                continue
            per = bool(found.group(4))
            if int(whole) >= 10 and not fraction:
                forms.add((per, _WHOLE))
            elif int(whole) >= 1:
                ten = _TEN_CENTS if int(whole) >= 10 else _CENTS
                forms.add((per, ten if len(fraction) == 2 else f"{len(fraction)} decimals"))
            else:
                forms.add((per, len(fraction) - (len(fraction) - len(fraction.lstrip("0")))))
    return forms


def _one_form(texts: Iterable[str]) -> bool:
    """Whether the amounts in some texts read by one rule: one scale; every amount under a dollar to two significant
    figures, or three where it reconciles with a month beside it; every amount from a dollar to cents, and from $10
    either all in whole dollars (a rate) or all to cents (a total)."""
    forms = _forms(texts)
    marks = {mark for _, mark in forms}
    figures = {mark for mark in marks if isinstance(mark, int)}
    spelled = {mark for mark in marks if isinstance(mark, str)}
    return (
        len({per for per, _ in forms}) <= 1
        and figures <= {2, 3}
        and spelled <= {_CENTS, _WHOLE, _TEN_CENTS}
        and not {_WHOLE, _TEN_CENTS} <= spelled
    )


def _written_places(found: re.Match[str]) -> int:
    """The place of the last figure a spelled amount states, as decimals: its decimals where it writes some; else, for
    whole dollars, minus the zeros past its first two figures (owner ruling Q3: a rate reads two significant figures at
    every magnitude, so ``$160`` states the tens and ``$1,200`` the hundreds, ``$75`` and ``$123`` the dollars)."""
    if found.group(3):
        return len(found.group(3))
    digits = found.group(2).replace(",", "")
    return -(len(digits) - max(2, len(digits.rstrip("0"))))


def _amount(text: str) -> float:
    """The dollars one spelled amount reads, as a reader takes it off the page."""
    found = _AMOUNT.search(text)
    assert found is not None, text
    value = float(found.group(2).replace(",", "") + "." + (found.group(3) or "0"))
    value /= 1000 if found.group(4) else 1
    return -value if found.group(1) == "-" else value


def _arms(held: ReportInput, family: str) -> dict[str, Figure]:
    """Each arm's figure in one key family, where the input states one."""
    figures: dict[str, Figure] = {}
    for arm in held.arms:
        try:
            figures[arm.id] = resolve(family.format(arm=arm.id), held)
        except UnresolvableReference:
            # NOSILENT: an arm the input states no such figure for is no member of the family's set.
            continue
    return figures


# --- the form is chosen per set ----------------------------------------------------------------------------------


def test_a_set_with_a_member_under_a_cent_reads_in_plain_dollars_each_member_to_its_own_two_figures() -> None:
    """Q3 and Q4: the per-1,000 mark is retired; a set under a cent reads in plain dollars, every member to its own two
    significant figures, never padded to its least member's decimals (owner ruling 2026-11-07: no ``$0.1100``)."""
    form = money_form([0.008, 0.11])
    assert [form.spell(amount) for amount in (0.008, 0.11)] == ["$0.0080", "$0.11"]


def test_each_member_reads_two_figures_and_only_a_tight_interval_takes_a_third() -> None:
    """The set chooses its form: each member reads to two significant figures; only an interval whose two ends would
    read alike at two takes a third, for itself (a cell's value and ends together), never the whole set: ``$0.11``
    beside it never reads ``$0.110``."""
    planted = [0.008, 0.11]
    together = [money_form(planted).spell(amount) for amount in planted]
    assert together == ["$0.0080", "$0.11"] and _forms(together) == {(False, 2)}
    form = money_form([*planted, 0.0451, 0.0454])
    # The tight interval alone takes its third figure (owner ruling Q3: "one more only where"), its neighbours keep two.
    assert form.spell_ends(0.0451, 0.0454) == ("$0.0451", "$0.0454")
    assert [form.spell(amount) for amount in planted] == ["$0.0080", "$0.11"]
    assert form.places_of([0.0452, 0.0451, 0.0454], ends=(0.0451, 0.0454)) == 4
    assert form.places_of([0.12, 0.10, 0.14], ends=(0.10, 0.14)) == 2


def test_each_member_reads_to_two_figures_and_never_fewer_than_cents() -> None:
    form = money_form([0.03, 3.477])
    assert form == MoneyForm(per_thousand=False, decimals=2, figures=2), "the largest member's decimals: the ticks'"
    assert [form.spell(amount) for amount in (0.03, 3.477)] == ["$0.030", "$3.48"]
    assert money_form([1.892, 2.427]) == MoneyForm(per_thousand=False, decimals=2, figures=2)
    assert money_form([75.05, 6.014]) == MoneyForm(per_thousand=False, decimals=2, figures=2)
    form = money_form([0.0013, 0.61])
    assert form == MoneyForm(per_thousand=False, decimals=2, figures=2)
    assert [form.spell(amount) for amount in (0.0013, 0.61)] == ["$0.0013", "$0.61"]


def test_a_rate_beside_its_month_takes_a_third_figure_only_where_two_would_leave_the_month_off() -> None:
    """Owner ruling Q3, "one more figure only where needed": a cost per 1,000 printed beside the month computed from it
    reads a third figure where two would round it by more than half a percent ($0.302 to $0.30 is 0.7%), and two where
    they do not ($0.633 to $0.63 is 0.5%); with no month beside it, two."""
    beside = money_form([0.302, 0.633, 0.048], rate=True, reconciles=True)
    assert [beside.spell(amount) for amount in (0.302, 0.633, 0.048)] == ["$0.302", "$0.63", "$0.048"]
    alone = money_form([0.302, 0.633, 0.048], rate=True)
    assert [alone.spell(amount) for amount in (0.302, 0.633, 0.048)] == ["$0.30", "$0.63", "$0.048"]
    assert not money_form([0.302], reconciles=True).reconciles, "a total is the month itself, never reconciled"


def test_a_set_whose_least_amount_is_ten_dollars_or_more_reads_cents_on_every_surface() -> None:
    """r4's monthly costs read alike on a chart, in a key and in a table: one rule, money_form's, and a total of ten
    dollars or more reads to cents (owner ruling Q3, 2026-11-07), never whole dollars."""
    form = money_form([14.5, 120.0])
    assert form == MoneyForm(per_thousand=False, decimals=2, figures=2)
    assert [form.spell(amount) for amount in (14.5, 120.0, 1208.0)] == ["$14.50", "$120.00", "$1,208.00"]
    assert money_form([9.5, 120.0]) == MoneyForm(per_thousand=False, decimals=2, figures=2)
    # From $1 to under $10 a rate reads in cents, as a total does: dollars and cents is how a reader expects money of
    # that size, never "$1.9" (coordinator ruling on Discodon's host pass, 2026-11-07).
    one_to_ten = money_form([1.9, 5.12, 0.02], rate=True)
    assert [one_to_ten.spell(amount) for amount in (1.9, 5.12, 0.02, -1.43)] == ["$1.90", "$5.12", "$0.020", "-$1.43"]
    # A rate of $10 or more reads in whole dollars at two figures (Q3): "$75 [$73, $77]", never "$74.89".
    rate = money_form([74.89, 6.014], rate=True)
    assert [rate.spell(amount) for amount in (74.89, 73.13, 76.65, 6.014, 140.0)] == [
        "$75",
        "$73",
        "$77",
        "$6.01",
        "$140",
    ]
    held = INPUTS["r4"]
    months = _arms(held, "cost_per_month.{arm}")
    values = [float(figure.value) for figure in months.values() if isinstance(figure.value, float)]
    drawn = quantity("dollars", values)
    for figure in months.values():
        assert isinstance(figure.value, float)
        key = format_figure(figure, ("usd",), forms=PageForms.of(held))
        assert key == figure_text(figure.value * drawn.factor, drawn) == money_form(values).spell(figure.value)
        assert re.fullmatch(r"\$[\d,]+\.\d\d", key), key


def test_a_rate_reads_two_significant_figures_at_every_magnitude_and_a_total_keeps_cents() -> None:
    """Owner ruling Q3, two significant figures per amount: a rate from $100 up rounds into its dollars ($123 reads
    ``$120``, $1,234 reads ``$1,200``), as one from $10 reads whole dollars; a total of the same size keeps its cents.
    A table's column of rates reads alike, and a tight interval takes its third figure there too."""
    rate = money_form([123.4, 1234.0, 74.89], rate=True)
    assert [rate.spell(amount) for amount in (123.4, 1234.0, 74.89, 140.0)] == ["$120", "$1,200", "$75", "$140"]
    assert rate.spell_ends(121.0, 124.0) == ("$121", "$124")
    assert rate.decimals == 0, "a chart's ticks read whole dollars at the fewest"
    total = money_form([123.4, 1234.0])
    assert [total.spell(amount) for amount in (123.4, 1234.0)] == ["$123.40", "$1,234.00"]
    rows = PageForms.off_page().money_columns(
        [{"cost": TableCell(value=123.4, unit=MONEY)}, {"cost": TableCell(value=7.5, unit=MONEY)}]
    )
    assert [row["cost"].text for row in rows] == ["$120", "$7.50"]


def test_unknown_reads_unpriced_never_a_zero_and_is_no_member() -> None:
    form = money_form([None, 0.11, math.nan])
    assert form == money_form([0.11])
    assert form.spell(None) == UNPRICED
    assert form.spell(math.nan) == ABSENT
    assert form.spell(0) == "$0"


def test_a_planted_family_under_a_cent_reads_in_one_plain_form_in_every_key() -> None:
    """A cost per 1,000 of $0.008 beside one of $0.11: both keys of ``cost_per_1k.*`` read in one form, plain dollars,
    never with the retired per-1,000 mark (Q4)."""
    held = INPUTS["r4"]
    measure = held.measures["cost_per_1k"]
    arms = list(measure.per_arm)
    planted = {arm: (0.008 if index == 0 else 0.11) for index, arm in enumerate(arms)}
    per_arm = {arm: stat.model_copy(update={"mean": planted[arm]}) for arm, stat in measure.per_arm.items()}
    moved = held.model_copy(
        update={"measures": {**held.measures, "cost_per_1k": measure.model_copy(update={"per_arm": per_arm})}}
    )
    texts = [format_figure(resolve(f"cost_per_1k.{arm}", moved), ("usd",), forms=PageForms.of(moved)) for arm in arms]
    assert not any(text.endswith("/1000") for text in texts), texts
    # A rate: $8.00 to cents, $110 in whole dollars from $10 (Q3).
    assert _one_form(texts) and {text.split(" ")[0] for text in texts} == {"$8.00", "$110"}, texts


# --- one form per key family, column and chart, on every fixture ----------------------------------------------------

#: The money key families held per arm, each with the filter it is cited with.
_FAMILIES = (
    ("cost_per_1k.{arm}", ("usd",)),
    ("cost_per_1k.{arm}", ("usd_per_1k",)),
    ("scoreboard.{arm}.reply_cost_usd", ()),
    ("cost_per_month.{arm}", ("usd",)),
)


@pytest.mark.parametrize("name", sorted(INPUTS))
@pytest.mark.parametrize(("family", "filters"), _FAMILIES)
def test_every_key_of_a_money_family_reads_in_its_sets_one_form(
    name: str, family: str, filters: tuple[str, ...]
) -> None:
    held = INPUTS[name]
    figures = _arms(held, family)
    if not figures:
        pytest.skip(f"{name} states no {family}")
    first = next(iter(figures.values()))
    under = min((abs(float(f.value)) for f in figures.values() if isinstance(f.value, float) and f.value), default=1)
    # A family that counts something and holds an amount under a cent reads per 1,000 of it with ``usd`` too, its count
    # said with the amount (Q4): as ``usd_per_1k`` reads it.
    counted = filters == ("usd_per_1k",) or (bool(first.per) and under < CENT and filters in ((), ("usd",)))
    factor = 1000.0 if counted and first.scale == "usd" else 1.0
    values = {arm: float(figure.value) * factor for arm, figure in figures.items() if isinstance(figure.value, float)}
    # A month's cost is a total, to cents; every other family is money per something, whole dollars from $10; the
    # page's cost per 1,000 reconciles with the month printed beside it where a volume is declared (Q3).
    rate = not family.startswith("cost_per_month")
    reconciles = rate and family.startswith("cost_per_1k") and held.meta.volume_per_month is not None
    form = money_form(values.values(), rate=rate, reconciles=reconciles)
    if counted and min((abs(value) for value in values.values() if value), default=1) < CENT:
        # Per 1,000 a family under a cent moves its ruler to per 1,000,000, the mark never stacked on it.
        values = {arm: value * 1000 for arm, value in values.items()}
        form = money_form(values.values(), rate=rate, reconciles=reconciles)
    forms = PageForms.of(held)
    texts = {arm: format_figure(figure, filters, forms=forms) for arm, figure in figures.items()}
    for arm, text in texts.items():
        assert text.startswith(form.spell(values[arm])), (arm, text)
    assert _one_form(texts.values()), texts


@pytest.mark.parametrize("name", sorted(INPUTS))
def test_the_options_table_cost_column_reads_in_one_form(name: str) -> None:
    held = INPUTS[name]
    table = arms_table(held, block_look(held, fact_arms(held)), (), forms=PageForms.of(held))
    column = next((column for column in table.columns if column.key == "cost_per_1k"), None)
    if column is None:
        pytest.skip(f"{name} states no cost")
    costs = _arms(held, "cost_per_1k.{arm}")
    per_run = [float(figure.value) / 1000 for figure in costs.values() if isinstance(figure.value, float)]
    count = 1
    while min((cost * count for cost in per_run if cost), default=1) < CENT and count < 1_000_000:
        count *= 1000
    form = money_form((cost * count for cost in per_run), rate=True, reconciles=held.meta.volume_per_month is not None)
    cells = [str(row["cost_per_1k"].value) for row in table.rows if row["cost_per_1k"].value is not None]
    assert cells == [form.spell(cost * count) for cost in per_run]
    assert ("per 1,000" in column.header) == (count > 1)
    assert _one_form(cells), cells


def test_a_values_tables_column_reads_in_one_form_and_keeps_every_digit() -> None:
    """A gate matrix's costs under a cent keep dollars, never moved per 1,000 with the retired mark (Q4), each
    amount's digits kept and its thousands grouped; a cost too small to state reads as a bound."""
    costs = [0.008, 0.11, 1239, 1e-9]
    matrix = GateMatrixChart.model_validate(
        {
            "columns": [{"label": "A"}],
            "rows": [
                {"label": str(at), "cells": [{"status": "neutral", "label": "", "cost": cost}]}
                for at, cost in enumerate(costs)
            ],
            "cost_fmt": "usd",
        }
    )
    spelled = [row[1] for row in matrix.values_table()[1]]
    # Money is never padded: each amount keeps its own figures (owner ruling 2026-11-07), never $0.110 beside $0.008.
    assert spelled == ["$0.008", "$0.11", "$1,239", "< $0.000001"]
    for cost, text in zip(costs[:3], spelled[:3], strict=True):
        assert _amount(text) == pytest.approx(cost)


def test_a_tables_money_column_reads_one_form_chosen_from_its_values() -> None:
    """A column holding $0.008 and $0.11 reads both to two significant figures, in plain dollars: never $0.008 beside
    $0.11, nor $0.1100 beside $0.0080 (owner ruling 2026-11-07); each cell is its own value at its own figures, so it
    states no digit the value does not hold."""
    rows = PageForms.off_page().money_columns(
        [
            {"arm": TableCell(value="A"), "cost": TableCell(value=0.008, unit="usd")},
            {"arm": TableCell(value="B"), "cost": TableCell(value=0.11, unit="usd")},
        ]
    )
    assert [row["cost"].text for row in rows] == ["$0.0080", "$0.11"]
    assert [row["arm"].text for row in rows] == ["A", "B"]


def _money_cells(table: TableBlock) -> dict[str, list[tuple[TableCell, str]]]:
    """Each figure type's money cells, with the text a reader is shown: a column's, by its key; in a table of arms
    across (a slice per row, each row one measure across the arms), every row's on one measure, by the measure."""
    across = any(column.key.startswith("arm:") for column in table.columns)
    cells: dict[str, list[tuple[TableCell, str]]] = {}
    for row in table.rows:
        measure = row["measure"].text if "measure" in row else ""
        for key, cell in row.items():
            if cell.is_number and cell.unit == "usd":
                cells.setdefault(f"measure:{measure}" if across else key, []).append((cell, cell.text))
    return cells


def _query(source: str, measures: list[str]) -> ChartQuery:
    """A chart's query over some measures, every other field at its plain value."""
    return ChartQuery.model_validate(
        {
            "source": source,
            "aspect": "",
            "measures": measures,
            "arms": [],
            "strata": [],
            "limit": 0,
            "transform": "none",
            "small_multiples_by": "none",
        }
    )


def _planned(kind: PlanChartKind, query: ChartQuery) -> PlannedChart:
    """A block as the doc page plans one: a kind and a query, no title, line or emphasis."""
    return PlannedChart(kind=kind, title="", query=query, reference=[], emphasis="")


def _engine_tables(held: ReportInput) -> list[TableBlock]:
    look = block_look(held, fact_arms(held))
    tables = list(every_table(held, look, forms=PageForms.of(held)))
    for key, measure in held.measures.items():
        if measure.unit in ("usd", "usd_per_1k", "usd_per_1m") and measure.per_arm:
            spec, faults = complete_chart(
                _planned("table", _query("measures", [key])), held, "block", forms=PageForms.of(held)
            )
            assert not faults, faults
            drawn = resolve_chart(spec, look, title="", fallback_title="t", forms=PageForms.of(held))
            assert isinstance(drawn.chart, TableBlock)
            tables.append(drawn.chart)
    return tables


@pytest.mark.parametrize("name", sorted(INPUTS))
def test_every_engine_tables_money_column_reads_one_form_from_its_values(name: str) -> None:
    held = INPUTS[name]
    source = cost_basis(held).source
    beside = {"cost_per_1k", f"measure:{source}", source} if held.meta.volume_per_month is not None else set()
    checked = 0
    for table in _engine_tables(held):
        for key, cells in _money_cells(table).items():
            values = [float(cell.value) for cell, _ in cells if isinstance(cell.value, int | float)]
            ends = [float(end) for cell, _ in cells for end in (cell.low, cell.high) if end is not None]
            # A table's money is drawn per the count its head names: a rate; the page's cost per 1,000 reconciles
            # with its month where a volume is declared.
            form = money_form([*values, *ends], rate=True, reconciles=key in beside)
            assert _one_form(text for _, text in cells), (table.name, key)
            for cell, text in cells:
                assert isinstance(cell.value, int | float)
                # A cell reads its value and its ends at one precision, a figure finer only where its own interval's
                # ends would read alike (P19).
                places = form.places_of([float(cell.value), cell.low, cell.high], ends=(cell.low, cell.high))
                assert text.split(" (")[0] == MoneyForm(decimals=places).spell(float(cell.value)), (table.name, key)
            checked += 1
    if not checked:
        pytest.skip(f"{name}'s tables state no money")


def test_r1s_tasks_table_reads_each_tasks_cost_across_the_arms_in_one_form() -> None:
    """One task's cost across the arms is one figure type with every other task's, a rate: each to two significant
    figures, whole dollars from $10 and cents under it (owner ruling Q3, 2026-11-07), so $58 | $24 | $4.60 | $8.90; its
    count is written as the page writes every count, "per 1,000 conversations", never the input label's "1k"."""
    held = INPUTS["r1"]
    (tasks,) = [
        table
        for table in every_table(held, block_look(held, fact_arms(held)), forms=PageForms.of(held))
        if table.name == "tasks"
    ]
    rows = [[cell.text for cell in row.values()] for row in tasks.rows]
    assert ["Billing & account Q&A", "400", "Cost per 1,000 conversations", "$58", "$24", "$4.60", "$8.90"] in rows


def test_a_cost_still_under_a_cent_at_the_rulers_cap_reads_there_with_no_mark() -> None:
    """r1 at a millionth of its cost stays under a cent even per 1,000,000 conversations: its Full results column reads
    at that count, each cost to its own two significant figures, and never with the per-1,000 mark beside it."""
    held = _measures_shrunk(INPUTS["r1"], frozenset({"cost_per_1k"}), 1e6)
    (arms,) = [
        table
        for table in every_table(held, block_look(held, fact_arms(held)), forms=PageForms.of(held))
        if table.name == "arms_by_measure"
    ]
    (column,) = [column for column in arms.columns if column.key == "cost_per_1k"]
    assert column.header == "Cost per 1,000,000 conversations"
    assert [row["cost_per_1k"].text for row in arms.rows] == ["$0.075", "$0.031", "$0.0060", "$0.012"]


def test_r2s_cost_reads_in_its_tables_as_its_key_reads_in_prose() -> None:
    """r2's brisk-s.low.r1 costs $0.302 per 1,000 calls; at 4,000,000 calls a month $0.30 would give $1,200 beside a
    monthly $1,208, so its key and every table keep the third figure its month needs (owner ruling Q3: one more figure
    only where needed)."""
    held = INPUTS["r2"]
    said = format_figure(resolve("cost_per_1k.brisk-s.low.r1", held), ("usd",), forms=PageForms.of(held))
    assert said == "$0.302"
    found = [
        text
        for table in _engine_tables(held)
        for cells in _money_cells(table).values()
        for cell, text in cells
        if cell.value == pytest.approx(0.302)
    ]
    assert found and set(found) == {said}


@pytest.mark.parametrize("name", sorted(INPUTS))
def test_a_charts_notes_and_ticks_read_in_one_form(name: str) -> None:
    held = INPUTS[name]
    costs = [float(figure.value) for figure in _arms(held, "cost_per_1k.{arm}").values()]
    if not costs:
        pytest.skip(f"{name} states no cost")
    drawn = quantity("usd_per_1k", costs)
    notes = [figure_text(cost * drawn.factor, drawn) for cost in costs]
    assert _one_form(notes), notes
    assert drawn.money is not None and not drawn.money.per_thousand
    assert drawn.fmt == f"usd:{drawn.money.decimals}"
    # The ticks are written as the notes beside them are: to the largest note's decimals ("$1.00", never "$1.000"),
    # and no note carries fewer decimals than the ticks.
    largest = notes[max(range(len(costs)), key=lambda at: abs(costs[at]))]
    assert _places([largest]) == {(False, drawn.money.decimals)}, (largest, drawn.fmt)
    assert all(decimals >= drawn.money.decimals for _, decimals in _places(notes)), (notes, drawn.fmt)


def test_a_planted_chart_under_a_cent_draws_a_thousand_times_over_its_ticks_and_notes_alike() -> None:
    """A chart of $0.008 and $0.11 per 1,000 draws them per 1,000,000, as $8.00 and $110, a rate in whole dollars from
    $10 (Q3): its ticks (the kit's usd:0, the largest note's) and its notes in one form, and no per-1,000 mark the
    ticks could not draw."""
    drawn = quantity("usd_per_1k", [0.008, 0.11])
    assert (drawn.factor, drawn.counted, drawn.fmt) == (1000.0, 1_000_000, "usd:0")
    notes = [figure_text(value * drawn.factor, drawn) for value in (0.008, 0.11)]
    assert notes == ["$8.00", "$110"]
    assert _one_form(notes)


def test_a_cost_scatter_labels_its_money_in_its_axis_form() -> None:
    """A cost-against-quality scatter over r5's cost per reply: drawn per 1,000 replies, its money reads $1.89-class,
    never 1,000 times that."""
    held = INPUTS["r5"]
    spec, faults = complete_chart(
        _planned("scatter_frontier", _query("measures", ["production_replicating_cost", "delivered_items"])),
        held,
        "block",
        forms=PageForms.of(held),
    )
    assert not faults, faults
    drawn = resolve_chart(
        spec, block_look(held, fact_arms(held)), title="", fallback_title="t", forms=PageForms.of(held)
    )
    assert isinstance(drawn.chart, ChartBlock)
    _, rows = drawn.chart.values_table()
    costs = {
        arm: float(stat.mean) * 1000
        for arm, stat in held.measures["production_replicating_cost"].per_arm.items()
        if stat.mean is not None
    }
    form = money_form(costs.values())
    spelled = {form.spell(cost) for cost in costs.values()}
    texts = [cell for row in rows for cell in row if "$" in cell]
    assert texts and all(text.split(" ")[0] in spelled for text in texts), texts
    assert all(float(text.split(" ")[0].lstrip("$")) < 10 for text in texts), texts


# --- deltas follow their amounts ---------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(INPUTS))
def test_a_money_change_reads_in_its_amounts_form_to_its_intervals_precision(name: str) -> None:
    """Kairo's production-cost contrast of 1017 read ``+$0.492`` on an interval tens of cents wide (per 1,000)."""
    held = INPUTS[name]
    checked = 0
    for contrast in held.contrasts:
        measure = held.measures.get(contrast.measure)
        if measure is None or measure.unit not in ("usd", "usd_per_1k", "usd_per_1m") or contrast.interval is None:
            continue
        means = [float(stat.mean) for stat in measure.per_arm.values() if stat.mean is not None]
        low, high = (float(end) for end in contrast.interval)
        figure = resolve(f"contrast.{contrast.arm}.{contrast.measure}.delta", held)
        text = format_figure(figure, forms=PageForms.of(held))
        if figure.per and min(abs(mean) for mean in means if mean) < CENT:
            # Money per something under a cent reads per 1,000 of it with its count (Q4), on its amounts' ruler.
            own = {"usd": 1, "usd_per_1k": 1000, "usd_per_1m": 1_000_000}[measure.unit]
            ruler = money_ruler(means, per=own, start=1000).for_changes([float(contrast.delta)], [high - low])
            assert text == f"{ruler.spell(float(contrast.delta), signed=True)} {ruler.words(figure.per)}"
            form = ruler.form
            width = (high - low) * ruler.scale
        else:
            form = money_form(means).for_changes([float(contrast.delta)], [high - low])
            assert text == form.spell(float(contrast.delta), signed=True)
            width = high - low
        assert form.decimals <= max(2, 1 - math.floor(math.log10(width)))
        checked += 1
    if not checked:
        pytest.skip(f"{name} holds no money contrast with an interval")


def test_the_kairo_cost_contrast_reads_two_figures_of_its_interval() -> None:
    held = INPUTS[KAIRO]
    text = format_figure(
        resolve("contrast.1017.production_replicating_cost.delta", held), ("usd_per_1k",), forms=PageForms.of(held)
    )
    assert text.startswith("+$0.49 "), text


@pytest.mark.parametrize("name", sorted(INPUTS))
def test_a_cost_gap_reads_in_the_form_of_the_costs_it_compares(name: str) -> None:
    held = INPUTS[name]
    costs = {arm: float(figure.value) for arm, figure in _arms(held, "cost_per_1k.{arm}").items()}
    if len(costs) < 2:
        pytest.skip(f"{name} states fewer than two costs")
    form = money_form(costs.values(), rate=True)
    first, second = list(costs)[:2]
    text = format_figure(resolve(f"cost.gap.{first}.{second}.per_1k", held), ("usd",), forms=PageForms.of(held))
    # A gap takes its precision from its own size, in the form (the scale) of the costs it compares (#993).
    gap = costs[first] - costs[second]
    figure = resolve(f"cost.gap.{first}.{second}.per_1k", held)
    if figure.per and min(abs(cost) for cost in costs.values() if cost) < CENT:
        # Money per something under a cent reads per 1,000 more of it with its count (Q4), on its amounts' ruler.
        ruler = money_ruler(costs.values(), per=1000, start=1000).for_changes([gap])
        assert text == f"{ruler.spell(gap, signed=True)} {ruler.words(figure.per)}"
    else:
        assert text == form.for_changes([gap]).spell(gap, signed=True)


# --- the monthly figure reconciles -------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(name for name, held in INPUTS.items() if held.meta.volume_per_month))
def test_a_cost_per_1000_times_the_volume_reaches_the_monthly_figure_within_the_forms_rounding(name: str) -> None:
    held = INPUTS[name]
    volume = float(held.meta.volume_per_month or 0)
    checked = 0
    for arm in _arms(held, "cost_per_1k.{arm}"):
        price_text = render(f"{{cost_per_1k.{arm}|usd_per_1k}}", {}, forms=PageForms.of(held))
        month_text = render(f"{{cost_per_month.{arm}|usd}}", {}, forms=PageForms.of(held))
        price, month = _amount(price_text) / (1000 if "per 1,000,000" in price_text else 1), _amount(month_text)
        (per_thousand, decimals), *_ = _places([price_text])
        rounding = 10.0**-decimals / 2 / (1000 if per_thousand or "per 1,000,000" in price_text else 1)
        (_, month_decimals), *_ = _places([month_text])
        assert abs(price * volume / 1000 - month) <= rounding * volume / 1000 + 10.0**-month_decimals / 2, arm
        checked += 1
    assert checked


# --- the options table: cost beside quality, with its pricing ------------------------------------------------------


def _priced(held: ReportInput, pricing: dict[str, Pricing], unanswered: str = "") -> ReportInput:
    """The input with each named arm's scoreboard row carrying the pricing given, and one arm's lost calls counted."""
    rows = dict(held.scoreboard)
    for arm, priced in pricing.items():
        rows[arm] = rows[arm].model_copy(update={"pricing": priced})
    if unanswered:
        counted = {"candidate": Unanswered(retried=0, lost=3, calls=894, share=3 / 897)}
        rows[unanswered] = rows[unanswered].model_copy(update={"unanswered": counted})
    return held.model_copy(update={"scoreboard": rows})


def _row(table: TableBlock, held: ReportInput, arm: str) -> dict[str, TableCell]:
    return table.rows[[one.id for one in held.arms].index(arm)]


def test_the_options_table_shows_the_price_source_provider_and_priced_share_beside_the_pass_rate() -> None:
    held = INPUTS["r5"]
    arms = [arm.id for arm in held.arms]
    pricing = {
        arm: Pricing(sources=["openrouter:reported"], providers=["DeepInfra"], calls=897, priced_calls=897)
        for arm in arms
    }
    moved = _priced(held, pricing, unanswered=arms[1])
    table = arms_table(moved, block_look(moved, fact_arms(moved)), (), forms=PageForms.of(moved))
    keys = [column.key for column in table.columns]
    assert keys.index("pass_rate") < keys.index("cost_per_1k") < keys.index("priced_by")
    for arm in arms:
        row = _row(table, moved, arm)
        assert row["priced_by"].value == "openrouter:reported"
        assert row["priced_by"].flag == "served by DeepInfra"
        assert "all 897 calls priced" in (row["cost_per_1k"].flag or "")
        assert row["pass_rate"].value is not None
    assert "3 calls unanswered" in (_row(table, moved, arms[1])["cost_per_1k"].flag or "")


def test_a_partly_priced_arm_says_how_many_calls_were_priced_and_shows_no_smaller_total() -> None:
    held = INPUTS["r5"]
    arms = [arm.id for arm in held.arms]
    pricing = {arm: Pricing(sources=["openrouter:reported"], calls=897, priced_calls=897) for arm in arms}
    pricing[arms[0]] = Pricing(sources=["openrouter:reported"], calls=897, priced_calls=850)
    moved = _priced(held, pricing, unanswered=arms[0])
    table = arms_table(moved, block_look(moved, fact_arms(moved)), (), forms=PageForms.of(moved))
    cell = _row(table, moved, arms[0])["cost_per_1k"]
    assert cell.value == "850 of 897 calls priced"
    assert "$" not in cell.text
    assert "unanswered" in (cell.flag or "")


def test_an_arm_none_of_whose_calls_were_priced_reads_unpriced() -> None:
    held = INPUTS["r5"]
    arms = [arm.id for arm in held.arms]
    key = "production_replicating_cost"
    measure = held.measures[key]
    per_arm = {arm: stat for arm, stat in measure.per_arm.items() if arm != arms[0]}
    rows = dict(held.scoreboard)
    rows[arms[0]] = rows[arms[0]].model_copy(
        update={"reply_cost_usd": None, "pricing": Pricing(sources=[], calls=897, priced_calls=0)}
    )
    moved = held.model_copy(
        update={
            "measures": {**held.measures, key: measure.model_copy(update={"per_arm": per_arm})},
            "scoreboard": rows,
        }
    )
    table = arms_table(moved, block_look(moved, fact_arms(moved)), (), forms=PageForms.of(moved))
    assert _row(table, moved, arms[0])["cost_per_1k"].value == UNPRICED


# --- the single-value helpers are byte-identical -----------------------------------------------------------------


def test_format_number_reads_every_number_format_case_as_it_did() -> None:
    cases = json.loads(_CASES.read_text(encoding="utf-8"))
    for case in cases["cases"]:
        assert format_number(case["value"]) == case["expected"], case["why"]
    for case in cases["prefixed"]["cases"]:
        assert format_number(case["value"], prefix=case["prefix"]) == case["expected"], case


#: Discodon's two money call shapes and what each read before #993, which they read still.
_DISCODON_SHAPES = (
    (0.0, "$0", "$0"),
    (0.000160, "$0.00016", "$0.00016"),
    (0.0123, "$0.012", "$0.0123"),
    (0.302, "$0.30", "$0.302"),
    (0.5, "$0.50", "$0.5"),
    (1.6, "$1.60", "$1.6"),
    (7560.0, "$7,560.00", "$7560"),
    (-1.2, "-$1.20", "-$1.2"),
)


@pytest.mark.parametrize(("amount", "dollars", "prefixed"), _DISCODON_SHAPES)
def test_discodons_two_call_shapes_read_as_the_money_form_spells_one_amount(
    amount: float, dollars: str, prefixed: str
) -> None:
    """Discodon's ``dollars_text(cost)`` reads as the money form spells a total alone (owner ruling Q3, finding C-5):
    one amount, one spelling on a page; ``format_number(cost, prefix="$")`` keeps the number rule."""
    assert dollars_text(amount) == dollars == spell_dollars(amount)
    assert format_number(amount, prefix="$") == prefixed
    assert dollars_text(None) == UNPRICED


def _costing(held: ReportInput, per_1k: dict[str, float]) -> ReportInput:
    """The input with each arm's cost per 1,000 (its cost_per_1k measure's mean) as given."""
    measure = held.measures["cost_per_1k"]
    per_arm = {arm: stat.model_copy(update={"mean": per_1k[arm]}) for arm, stat in measure.per_arm.items()}
    return held.model_copy(
        update={"measures": {**held.measures, "cost_per_1k": measure.model_copy(update={"per_arm": per_arm})}}
    )


def test_an_options_table_whose_costs_are_a_cent_or_more_a_run_reads_per_run() -> None:
    """r4 with each arm costing $0.05 to $0.09 a reply: the column reads per reply, in dollars, one form."""
    held = INPUTS["r4"]
    arms = [arm.id for arm in held.arms]
    planted = _costing(held, {arm: 50.0 + 10 * index for index, arm in enumerate(arms)})
    table = arms_table(planted, block_look(planted, fact_arms(planted)), (), forms=PageForms.of(planted))
    (column,) = [column for column in table.columns if column.key == "cost_per_1k"]
    assert column.header == "Cost per reply"
    cells = [str(row["cost_per_1k"].value) for row in table.rows]
    assert cells == [money_form([0.05, 0.06, 0.07, 0.08]).spell(0.05 + 0.01 * index) for index in range(len(arms))]
    assert cells[0] == "$0.050"


def test_an_options_table_under_a_cent_per_1000_runs_reads_per_1000000() -> None:
    held = INPUTS["r3"]
    table = arms_table(held, block_look(held, fact_arms(held)), (), forms=PageForms.of(held))
    (column,) = [column for column in table.columns if column.key == "cost_per_1k"]
    assert column.header == "Cost per 1,000,000 turns"
    assert all("/1000" not in str(row["cost_per_1k"].value) for row in table.rows)


def test_r3s_price_per_1000_moves_its_ruler_and_stacks_no_mark() -> None:
    """r3's least cost is $0.0013 per 1,000 turns: its usd_per_1k keys read per 1,000,000 turns, never
    $180.00/1000 per 1,000; a rate of $10 or more reads in whole dollars (Q3)."""
    held = INPUTS["r3"]
    text = format_figure(
        resolve("measure.cost_per_1m.atlas-m.zero.mean", held), ("usd_per_1k",), forms=PageForms.of(held)
    )
    assert text.startswith("$180 per 1,000,000 "), text
    assert "/1000" not in text


# --- a partly priced arm ---------------------------------------------------------------------------------------------


def _part_priced_control() -> tuple[ReportInput, str, str]:
    held = INPUTS["r5"]
    control, candidate = (arm.id for arm in held.arms[:2])
    pricing = {arm.id: Pricing(sources=["openrouter:reported"], calls=897, priced_calls=897) for arm in held.arms}
    pricing[control] = Pricing(sources=["openrouter:reported"], calls=897, priced_calls=850)
    return _priced(held, pricing), control, candidate


def test_no_uplift_is_read_against_a_partly_priced_control() -> None:
    moved, control, candidate = _part_priced_control()
    assert moved.arms[0].role == "control"
    table = arms_table(moved, block_look(moved, fact_arms(moved)), (), forms=PageForms.of(moved))
    flag = _row(table, moved, candidate)["cost_per_1k"].flag or ""
    assert "no uplift: the control has 850 of 897 calls priced" in flag
    assert "%" not in flag


def test_a_partly_priced_arms_cost_keys_say_so_beside_the_amount() -> None:
    moved, control, candidate = _part_priced_control()
    assert format_figure(resolve(f"cost_per_1k.{control}", moved), ("usd",), forms=PageForms.of(moved)).endswith(
        " (850 of 897 calls priced)"
    )
    assert "calls priced" not in format_figure(
        resolve(f"cost_per_1k.{candidate}", moved), ("usd",), forms=PageForms.of(moved)
    )
    assert "850 of 897 calls priced" in format_figure(
        resolve(f"cost.uplift.{candidate}", moved), forms=PageForms.of(moved)
    )


def test_a_partly_priced_arms_bar_says_so_beside_its_cost() -> None:
    moved, control, _ = _part_priced_control()
    spec, faults = complete_chart(
        _planned("bars", _query("measures", ["production_replicating_cost"])), moved, "block", forms=PageForms.of(moved)
    )
    assert not faults, faults
    drawn = resolve_chart(
        spec, block_look(moved, fact_arms(moved)), title="", fallback_title="t", forms=PageForms.of(moved)
    )
    assert isinstance(drawn.chart, ChartBlock)
    data = drawn.chart.data
    assert isinstance(data, BarsChart) and data.rows is not None
    notes = {row.id: row.note or "" for row in data.rows}
    assert notes[control].endswith(" (850 of 897 calls priced)")
    assert all("calls priced" not in note for arm, note in notes.items() if arm != control)


# --- every chart and table kind that draws money, on sets under a cent -------------------------------------------------

#: The stat fields a planted set under a cent scales, as stored.
_STAT_NUMBERS = ("mean", "half_width", "sem", "p50", "p95", "change")


def _shrunk(node: Any, keys: frozenset[str], by: float) -> Any:
    """A report input's document with every figure of the money measures ``keys`` (each arm's stats wherever they sit,
    and each contrast on them) divided by ``by``."""
    if isinstance(node, list):
        return [_shrunk(item, keys, by) for item in node]
    if not isinstance(node, dict):
        return node
    shrunk = {}
    for name, value in node.items():
        if name in keys and isinstance(value, dict) and "per_arm" not in value and "mean" in value:
            shrunk[name] = _stat_shrunk(value, by)
        else:
            shrunk[name] = _shrunk(value, keys, by)
    if shrunk.get("measure") in keys:
        for field in ("delta", "threshold"):
            if isinstance(shrunk.get(field), int | float):
                shrunk[field] = shrunk[field] / by
        if isinstance(shrunk.get("interval"), list):
            shrunk["interval"] = [end / by for end in shrunk["interval"]]
    return shrunk


def _stat_shrunk(stat: dict[str, Any], by: float) -> dict[str, Any]:
    out = dict(stat)
    for field in _STAT_NUMBERS:
        if isinstance(out.get(field), int | float):
            out[field] = out[field] / by
    if isinstance(out.get("ci"), list):
        out["ci"] = [end / by for end in out["ci"]]
    return out


def _measures_shrunk(held: ReportInput, keys: frozenset[str], by: float) -> ReportInput:
    data = held.model_dump(mode="json", by_alias=True)
    for key in keys:
        measure = data["measures"][key]
        measure["per_arm"] = {arm: _stat_shrunk(stat, by) for arm, stat in measure["per_arm"].items()}
        measure["by_aspect"] = {
            aspect: {arm: _stat_shrunk(stat, by) for arm, stat in figures.items()}
            for aspect, figures in measure.get("by_aspect", {}).items()
        }
    return ReportInput.model_validate(_shrunk(data, keys, by))


def _sliced(held: ReportInput, key: str, *, gate: bool) -> ReportInput:
    """The input with every arm's overall figure on a money measure copied into each of its slices (tasks and strata),
    so a heatmap over them draws money; with ``gate``, a budget gate on it too, so a gate scatter draws it."""
    data = held.model_dump(mode="json", by_alias=True)
    overall = data["measures"][key]["per_arm"]
    for part in ("tasks", "strata"):
        for item in data.get(part) or []:
            for arm, figures in item["per_arm"].items():
                if arm in overall:
                    figures[key] = dict(overall[arm])
    if gate:
        means = [stat["mean"] for stat in overall.values() if stat.get("mean") is not None]
        data["criteria"].append(
            {
                "id": "cost_budget",
                "kind": "gate",
                "measure": key,
                "statistic": "mean",
                "direction": "lower_is_better",
                "threshold": sorted(means)[len(means) // 2],
                "scope": "overall",
                "label": "Cost budget",
            }
        )
    return ReportInput.model_validate(data)


def _under_a_cent() -> dict[str, tuple[ReportInput, str, str, str]]:
    """Each planted input whose money is under a cent where its unit states it, with the money measure, what it counts,
    the words its own unit's ruler reads (which no scaled chart may print) and the words it must read instead."""
    r2 = _measures_shrunk(_sliced(INPUTS["r2"], "cost_per_1k", gate=True), frozenset({"cost_per_1k"}), 1000.0)
    r5 = _measures_shrunk(
        _sliced(INPUTS["r5"], "production_replicating_cost", gate=False),
        frozenset({"production_replicating_cost", "cost_usd"}),
        1000.0,
    )
    rows = {
        arm: row.model_copy(
            update={
                "reply_cost_usd": row.reply_cost_usd / 1000 if row.reply_cost_usd is not None else None,
                "test_cost_usd": row.test_cost_usd / 10_000 if row.test_cost_usd is not None else None,
            }
        )
        for arm, row in r5.scoreboard.items()
    }
    r5 = r5.model_copy(update={"scoreboard": rows})
    # Two money measures on different rulers: the cost per reply under a cent per 1,000, the blended cost not.
    # Both money measures in every slice, so a slice's change chart draws two money measures on two rulers.
    both = _sliced(_sliced(INPUTS["r5"], "production_replicating_cost", gate=False), "cost_usd", gate=False)
    two_rulers = _measures_shrunk(both, frozenset({"production_replicating_cost"}), 1000.0)
    per_reply = {
        arm: row.model_copy(update={"reply_cost_usd": row.reply_cost_usd / 1000 if row.reply_cost_usd else None})
        for arm, row in two_rulers.scoreboard.items()
    }
    two_rulers = two_rulers.model_copy(update={"scoreboard": per_reply})
    r1 = _measures_shrunk(INPUTS["r1"], frozenset({"cost_per_1k"}), 1000.0)
    # Under a cent even per 1,000,000: the ruler's cap, where the set reads with no mark beside the count.
    capped = _measures_shrunk(INPUTS["r1"], frozenset({"cost_per_1k"}), 1e6)
    return {
        "r1 under a cent even per 1,000,000 conversations": (
            capped,
            "cost_per_1k",
            "per 1k conversations",
            "per 1,000,000 conversations",
        ),
        "r1 under a cent per 1,000 conversations": (
            r1,
            "cost_per_1k",
            "per 1k conversations",
            "per 1,000,000 conversations",
        ),
        # Its blended cost reads per 1,000 replies, rightly; only the cost per reply's own words would be stale.
        "r5 with two money measures on different rulers": (
            two_rulers,
            "production_replicating_cost",
            "production replicating cost, per 1,000 replies",
            "per 1,000,000 replies",
        ),
        "r2 under a cent per 1,000 calls": (r2, "cost_per_1k", "per 1k calls", "per 1,000,000 calls"),
        "r5 under a cent per 1,000 replies": (
            r5,
            "production_replicating_cost",
            "per 1,000 replies",
            "per 1,000,000 replies",
        ),
    }


UNDER_A_CENT = _under_a_cent()


def _queries(held: ReportInput, key: str) -> list[ChartQuery]:
    """Every query a money chart can be asked on an input: each source over every measure, over the money one alone,
    and over the money one against each other measure either way round (a scatter's and a gate scatter's two axes),
    each also over two arms (two arms compared); the scoreboard over its own money, and prices over each gain axis."""
    sources = get_args(ChartQuery.model_fields["source"].annotation)
    arms = [arm.id for arm in held.arms]
    others = [other for other in [*held.measures, *held.judged] if other != key]
    queries = []
    for source in sources:
        if source == "scoreboard":
            # The scoreboard's own money: its cost per reply and its trials' total, which counts nothing.
            named = [["reply_cost_usd"], ["test_cost_usd"]]
        elif source == "prices":
            named = [[other] for other in others]
        else:
            named = [[key], *([key, other] for other in others), *([other, key] for other in others)]
        for measures in ([], *named):
            queries.append(_query(source, measures))
            queries.append(ChartQuery.model_validate({**_query(source, measures).model_dump(), "arms": arms[:2]}))
    return queries


def _drawn_money(held: ReportInput, key: str) -> list[tuple[str, PageChart]]:
    """Every chart and table of every kind the page draws money on, on the input, as the page draws it."""
    look = block_look(held, fact_arms(held))
    drawn = []
    seen: set[str] = set()
    for kind in get_args(PlanChartKind):
        for query in _queries(held, key):
            spec, faults = complete_chart(_planned(kind, query), held, "block", forms=PageForms.of(held))
            if faults:
                continue
            try:
                page = resolve_chart(spec, look, title="", fallback_title="t", forms=PageForms.of(held))
            except UnresolvableReference:
                # NOSILENT: a kind the input cannot draw over this query is no chart of money; the sweep reads on.
                continue
            label = f"{kind}/{query.source}/{','.join(query.measures) or 'all'}/{len(query.arms)}"
            if _draws_money(page) and label not in seen:
                seen.add(label)
                drawn.append((label, page))
    # The Full results tables, every one the input holds, with the cost column of the page's one cost rule.
    for table in every_table(held, look, forms=PageForms.of(held)):
        page = PageChart(title="", chart=table, notes=[])
        if _draws_money(page):
            drawn.append((f"full/{table.name}/all/0", page))
    return drawn


def _draws_money(page: PageChart) -> bool:
    """Whether a drawn chart or table states some amount of money among the values it draws."""
    if isinstance(page.chart, TableBlock):
        return any("$" in cell.text for row in page.chart.rows for cell in row.values())
    _, rows = page.chart.values_table()
    return any("$" in cell for row in rows for cell in row)


def _texts(node: Any, skip: frozenset[str] = frozenset()) -> Iterator[str]:
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for key, value in node.items():
            if key not in skip:
                yield from _texts(value, skip)
    elif isinstance(node, list):
        for value in node:
            yield from _texts(value, skip)


#: The chart fields carrying a drawn value's own text (the kit contract's text fields).
_VALUE_TEXTS = frozenset(
    {
        "value_text",
        "lo_text",
        "hi_text",
        "values_text",
        "x_text",
        "y_text",
        "from_text",
        "to_text",
        "cost_text",
        "latency_text",
        "low_text",
        "high_text",
        "detail_text",
        "matrix_text",
    }
)


def _value_texts(node: Any) -> Iterator[tuple[str, str]]:
    """Every value's own text in a chart's data, with the field carrying it."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in _VALUE_TEXTS:
                yield from ((key, text) for text in _texts(value))
            else:
                yield from _value_texts(value)
    elif isinstance(node, list):
        for value in node:
            yield from _value_texts(value)


def _page_texts(page: PageChart) -> list[str]:
    if isinstance(page.chart, TableBlock):
        headers = [column.header for column in page.chart.columns]
        return [*headers, *(cell.text for row in page.chart.rows for cell in row.values()), *page.notes]
    data = page.chart.data.model_dump(mode="json") if page.chart.data is not None else {}
    _, rows = page.chart.values_table()
    return [*_texts(data), *(cell for row in rows for cell in row), *page.notes]


#: A kit money format, its sign and its decimals.
_MONEY_FMT = re.compile(r"\+?usd(?:pk)?:(\d)")

#: The fields of a chart's data that hold the format of the numbers beside them, by the numbers' field names.
_FORMATTED = {
    "fmt": ("value",),
    "x_fmt": ("x",),
    "y_fmt": ("y",),
    "cost_fmt": ("cost",),
    "detail_fmt": ("detail",),
}

#: The words a stated count of what money counts reads in: ``per 1,000,000 replies``.
_COUNT = re.compile(r"per ([\d,]+) ")

#: The fields that name a chart's parts in words: where a code key must never show.
_NAMING = ("title", "subtitle", "x_title", "y_title", "label", "group", "row_header", "cost_unit")


def _formats(node: Any) -> Iterator[tuple[str, int]]:
    """Every money format in a chart's data, with the field it sits in, wherever a panel overrides it."""
    if isinstance(node, dict):
        for name, value in node.items():
            if name in _FORMATTED and isinstance(value, str) and (found := _MONEY_FMT.fullmatch(value)):
                yield name, int(found.group(1))
            else:
                yield from _formats(value)
    elif isinstance(node, list):
        for value in node:
            yield from _formats(value)


def _numbers(node: Any, fields: tuple[str, ...]) -> Iterator[float]:
    """Every number a chart's data holds in the named fields, wherever they sit, but in a panel drawn in another
    format (a pass rate's beside a cost's), whose numbers are no money."""
    if isinstance(node, dict):
        own = node.get("fmt")
        if isinstance(own, str) and not _MONEY_FMT.fullmatch(own) and "value" in fields:
            return
        for name, value in node.items():
            if name in fields and isinstance(value, int | float) and not isinstance(value, bool):
                yield float(value)
            elif name in fields and isinstance(value, list):
                yield from (float(one) for one in value if isinstance(one, int | float))
            else:
                yield from _numbers(value, fields)
    elif isinstance(node, list):
        for value in node:
            yield from _numbers(value, fields)


def _naming(node: Any) -> Iterator[str]:
    if isinstance(node, dict):
        for name, value in node.items():
            if name in _NAMING and isinstance(value, str):
                yield value
            else:
                yield from _naming(value)
    elif isinstance(node, list):
        for value in node:
            yield from _naming(value)


def _stored(held: ReportInput, keys: Iterable[str]) -> list[float]:
    """Every money figure the input states of some keys, as stored: each statistic wherever it sits, each contrast,
    each two arms' difference within a place, and each step's price along every frontier."""
    return [*_stored_of(held, keys), *(value for value, _ in _tails(held))]


def _stored_of(held: ReportInput, keys: Iterable[str]) -> list[float]:
    """Every money figure the input states of some keys, as stored: each statistic wherever it sits, each contrast,
    and each two arms' difference within a place."""
    data = held.model_dump(mode="json", by_alias=True)
    found: list[float] = []
    places: list[dict[str, float]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            means: dict[str, float] = {}
            for name, value in node.items():
                if name in keys and isinstance(value, dict) and "mean" in value:
                    found.extend(float(value[f]) for f in _STAT_NUMBERS if isinstance(value.get(f), int | float))
                    found.extend(float(end) for end in value.get("ci") or [])
                elif isinstance(value, dict):
                    for key in keys:
                        stat = value.get(key)
                        if isinstance(stat, dict) and isinstance(stat.get("mean"), int | float):
                            means[name] = float(stat["mean"])
                if node.get("measure") in keys and name in ("delta", "threshold") and isinstance(value, int | float):
                    found.append(float(value))
                if node.get("measure") in keys and name == "interval" and isinstance(value, list):
                    found.extend(float(end) for end in value if isinstance(end, int | float))
                walk(value)
            if len(means) > 1:
                places.append(means)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(data)
    for key in keys:
        measure = held.measures.get(key)
        if measure is not None:
            stats = [*measure.per_arm.values(), *(stat for by in measure.by_aspect.values() for stat in by.values())]
            for stat in stats:
                found.extend(float(getattr(stat, f)) for f in _STAT_NUMBERS if getattr(stat, f) is not None)
                found.extend(float(end) for end in stat.ci or ())
                if stat.mean is not None and stat.half_width is not None:
                    found.extend((float(stat.mean - stat.half_width), float(stat.mean + stat.half_width)))
            places.append({arm: float(stat.mean) for arm, stat in measure.per_arm.items() if stat.mean is not None})
    for means in places:
        found.extend(
            first - second for at, first in enumerate(means.values()) for second in list(means.values())[at + 1 :]
        )
        found.extend(
            second - first for at, first in enumerate(means.values()) for second in list(means.values())[at + 1 :]
        )
    return found


def _tails(held: ReportInput) -> list[tuple[float, int]]:
    """The money figures the input states of no one measure, each with the count it is stated per (0 for a total):
    each step's price along every frontier, each arm's cost per reply and its run's total, and each arm's cost by the
    package's one cost rule."""
    found: list[tuple[float, int]] = []
    for gain in [*held.measures, *held.judged]:
        try:
            rows = query_rows(_query("prices", [gain]), held)
        except UnresolvableReference:
            # NOSILENT: a measure no frontier is priced on states no price.
            continue
        for row in rows:
            if isinstance(row.item, PriceStep):
                opened = (row.item.open_from,) if row.item.open_from is not None else ()
                # A price per point is per 1,000 of what the cost counts, whatever the measure's own unit.
                ends = (*(row.interval or ()), *opened)
                found.extend((value, 1000) for value in (float(row.item.value), *(float(end) for end in ends)))
    for arm in held.arms:
        row = held.scoreboard.get(arm.id)
        if row is not None:
            # A reply's cost is per reply; a run's total counts nothing per anything (0: never scaled).
            found.extend(
                (float(v), own) for v, own in ((row.reply_cost_usd, 1), (row.test_cost_usd, 0)) if v is not None
            )
        try:
            # The arm's cost by the package's one cost rule, a figure the Full results table states.
            figure = resolve(f"cost_per_1k.{arm.id}", held)
        except UnresolvableReference:
            # NOSILENT: an input that holds no cost for the arm states none.
            continue
        if isinstance(figure.value, float):
            found.append((figure.value, 1000))
    return found


#: The count each money unit states its figures per.
_OWN = {"usd": 1, "usd_per_1k": 1000, "usd_per_1m": 1_000_000}


def _owned(held: ReportInput) -> list[tuple[float, int]]:
    """Every money figure and line the input states, each with the count it is stated per (0 for a total, which is
    never scaled): a text's amount reads one of these at the count it names over the figure's own."""
    found = [
        (value, _OWN[measure.unit])
        for key, measure in held.measures.items()
        if measure.unit in _OWN
        for value in _stored_of(held, {key})
    ]
    lines = [
        (float(criterion.threshold), _OWN[str(figure.scale)])
        for criterion in held.criteria
        for figure in (resolve(f"criteria.{criterion.id}.threshold", held),)
        if figure.scale in _OWN
    ]
    return [*found, *lines, *_tails(held)]


def _reason_ruler(held: ReportInput, reason: str) -> tuple[Criterion, MoneyRuler] | None:
    """The money criterion a gate's reason is about, by its reader name, and the ruler it states money on; None for a
    reason on no money criterion."""
    for criterion in held.criteria:
        line = resolve(f"criteria.{criterion.id}.threshold", held)
        if line.scale in ("usd", "usd_per_1k", "usd_per_1m") and (criterion.label or criterion.id) in reason:
            return criterion, criterion_ruler(held, criterion, line)[0]
    return None


#: A tip's line comparing the control's level with an arm's: ``September $1.89 · qwen3.8-flash $2.43``.
_LEVELS = re.compile(r"^.+? ([+\-]?\$[\d,.]+) · .+? ([+\-]?\$[\d,.]+)$")


def _descriptive_rows(node: Any) -> Iterator[tuple[float, float, float]]:
    """Each change row whose tip compares two levels: the control's level, the arm's, and the row's value."""
    if isinstance(node, dict):
        tip = node.get("tip")
        value = node.get("value")
        if isinstance(tip, list) and isinstance(value, int | float):
            for line in tip:
                if isinstance(line, str) and (found := _LEVELS.match(line)):
                    yield _amount(found.group(1)), _amount(found.group(2)), float(value)
        for child in node.values():
            yield from _descriptive_rows(child)
    elif isinstance(node, list):
        for child in node:
            yield from _descriptive_rows(child)


def _reconciles(value: float, stored: list[float], scale: float, places: int | None = None) -> bool:
    """Whether a drawn value is a stored figure times the scale: exactly for a number the chart's data holds, and to
    half the last decimal written for an amount a text spells (``places``), which reads to its own figures."""
    if value == 0:
        return True
    if places is None:
        return any(math.isclose(value / scale, one, rel_tol=1e-6, abs_tol=1e-12) for one in stored)
    return any(abs(value - one * scale) <= 10.0**-places / 2 + abs(value) * 1e-9 for one in stored)


@pytest.mark.parametrize("name", sorted(UNDER_A_CENT))
def test_every_kind_that_draws_money_under_a_cent_moves_its_ruler_and_states_it(name: str) -> None:
    """Every chart and table kind that draws a set under a cent, scatter, gate scatter, both heatmaps and prices
    included: it moves the ruler (no per-1,000 mark beside a chart's count; only a chart of totals carries one), never
    states two counts
    nor its own unit's count beside the moved one, states the drawn count, and draws each value at that count times
    the figure stored; its ticks, notes and tips read to one form's decimals; its words name no code key."""
    held, key, stale, moved = UNDER_A_CENT[name]
    own = {"usd": 1, "usd_per_1k": 1000, "usd_per_1m": 1_000_000}[held.measures[key].unit or "usd"]
    money = {name for name, measure in held.measures.items() if measure.unit in ("usd", "usd_per_1k", "usd_per_1m")}
    stored = _stored(held, {*money, "reply_cost_usd"})
    drawn = _drawn_money(held, key)
    reasons = {
        reason
        for arm in held.arms
        for task in (None, *(one.id for one in held.tasks))
        for reason in (lambda check: (*check.fails, *check.borders))(gate_check(held, arm.id, task=task))
    }
    kinds = {label.split("/")[0] for label, _ in drawn}
    expected = {"bars", "table", "dot_interval", "diverging_bars", "scatter_frontier", "heatmap", "diff_heatmap"}
    expected |= {"gate_scatter", "gate_matrix", "lattice"} if name.startswith("r2") else set()
    expected = {"full", "gate_matrix"} if name.startswith("r1") else expected
    assert expected <= kinds, sorted(expected - kinds)
    checked = 0
    for label, page in drawn:
        texts = _page_texts(page)
        joined = " | ".join(texts)
        totals = "test_cost_usd" in label
        if isinstance(page.chart, ChartBlock) and not totals:
            # A counted set moves its ruler, so the mark never sits beside its count; only a total, which counts
            # nothing, reads per 1,000 with its mark (test_a_chart_of_totals_under_a_cent_reads_per_1000_with_its_mark).
            assert "/1000" not in joined, (label, joined[:400])
        spelled = [abs(_amount(found.group(0))) for text in texts for found in _AMOUNT.finditer(text)]
        if "/prices/" not in label and any(0 < amount < 0.01 for amount in spelled):
            # A price per point is per 1,000 of what the cost counts: its own count, which is no stale one. A chart of
            # fewer points may hold no amount under a cent, so it keeps its own count, rightly.
            assert stale not in joined, (label, joined[:600])
        # No count stacked on another: one text states one count (said again for each item it lists, at most), and
        # never the same count twice in a row.
        stacked = [
            text
            for text in texts
            if len(set(_COUNT.findall(text))) > 1 or re.search(r"(per [\d,]+ [\w ]+?), \1(?!\w)", text)
        ]
        assert not stacked, (label, stacked[:3])
        words = [
            *page.notes,
            *(
                _naming(page.chart.data.model_dump(mode="json"))
                if isinstance(page.chart, ChartBlock) and page.chart.data is not None
                else []
            ),
        ]
        assert not [text for text in words if re.search(r"\busd\b|\b\w+_usd\b", text, re.I)], (label, words)
        under = any(0 < amount < 0.01 for amount in spelled) or moved in joined
        if under and not totals and (key in label or label.split("/")[2] == "all" or "reply_cost_usd" in label):
            # A chart states the count it moved its ruler to; a table either does, or writes each amount as a figure
            # key spells it alone (a computed table prints it as prose citing it does): plain dollars, the per-1,000
            # mark retired (Q4).
            amounts = [text for text in texts if "$" in text]
            stated = [text for text in amounts if any(_amount(found.group(0)) for found in _AMOUNT.finditer(text))]
            keyed = bool(stated) and not any("/1000" in text for text in stated)
            assert moved in joined or (isinstance(page.chart, TableBlock) and keyed), (label, joined[:600])
        if not isinstance(page.chart, ChartBlock) or page.chart.data is None:
            continue
        dump = page.chart.data.model_dump(mode="json")
        formats = dict(_formats(dump))
        digits = set(formats.values())
        if len(digits) == 1:
            # Ticks, notes and tips in one form: every amount the chart writes to its own figures, never finer than its
            # ticks' decimals, but a gate's reason, which a tip quotes as the gate check gives it on its criterion's
            # own ruler.
            (most,) = digits
            amounts = [text for text in _texts(dump) if "$" in text and text not in reasons]
            assert _one_form(amounts), (label, amounts[:6])
            # Every amount to its own figures, the ticks to the largest's decimals: no tick carries more decimals than
            # an amount beside it.
            assert all(decimals >= most for _, decimals in _places(amounts)), (label, most, amounts[:6])
            for text in (text for text in _texts(dump) if text in reasons):
                # A gate's reason, which a tip quotes, on its criterion's own ruler: its form's decimals, and each
                # amount the criterion's ruler times a figure stored or its line.
                found = _reason_ruler(held, text)
                if found is None:
                    continue
                criterion, ruler = found
                assert all(decimals <= ruler.form.decimals for _, decimals in _places([text])), (label, text)
                line = [float(criterion.threshold)]
                off = [
                    one
                    for one in _AMOUNT.finditer(text)
                    if not _reconciles(
                        _amount(one.group(0)), [*stored, *line], ruler.scale, places=_written_places(one)
                    )
                ]
                assert not off, (label, text, ruler.scale)
            for row in _descriptive_rows(dump):
                # A tip comparing two levels states levels that differ by the bar, on the chart's one ruler.
                low, high, value = row
                assert math.isclose(high - low, value, abs_tol=10.0**-most), (label, row)
            checked += 1
        counts = {int(count.replace(",", "")) for text in texts for count in _COUNT.findall(text)}
        # A price per point is per 1,000 of what the cost counts, whatever the measure's own unit.
        here = 1000 if "/prices/" in label else own
        scale = (max(counts) / here) if counts and not totals else 1.0
        if any(source in label for source in ("/levers/", "/settle/", "/configs/")) and label.startswith(
            ("diverging", "forest", "dot_interval")
        ):
            # A lever's effect is a mean over the configs it moves between, which no stored figure states one by one.
            continue
        for field, _digits in formats.items():
            values = list(_numbers(dump, _FORMATTED[field]))
            # An axis that names its own count (a scatter's two axes may sit on two rulers) is read at it.
            axis = dump.get(f"{field[0]}_title") if field in ("x_fmt", "y_fmt") else None
            named = _COUNT.findall(axis) if isinstance(axis, str) else []
            here_scale = int(named[0].replace(",", "")) / here if named and not totals else scale
            off = [value for value in values if not _reconciles(value, stored, here_scale)]
            assert not off, (label, field, here_scale, off[:4])
    assert checked, "the control: the one-form check reads some chart"
    read = 0
    owned = _owned(held)
    for label, page in drawn:
        levers = any(source in label for source in ("/levers/", "/settle/"))
        if levers or ("/configs/" in label and label.startswith(("diverging", "forest", "dot_interval"))):
            # A lever's effect is a mean over the configs it moves between, which no stored figure states one by one.
            continue
        read += _amounts_reconcile(label, page, owned, "test_cost_usd" in label)
    assert read, "the control: the text check reads some amount"


def _column_texts(page: PageChart) -> list[tuple[str, str, str]]:
    """Every text a chart or table shows a reader, each with the words naming its count (its row's or its column's
    head, empty for none) and the figure type it is in a table (empty off a table): a column's, its interval's column
    with it; where the arms are the columns, each row's measure across them. A table's cells and heads, a chart's data,
    its values table and its lines."""
    if isinstance(page.chart, TableBlock):
        heads: dict[str, str] = {}
        before = ""
        for column in page.chart.columns:
            # An interval's column, headed by its level alone, reads on the figure's column beside it.
            heads[column.key] = column.header if _COUNT.findall(column.header) or not before else before
            before = heads[column.key]
        # A row naming its measure in words (a slices table's measure cell) reads at the count those words name.
        cells = []
        for row in page.chart.rows:
            words = " ".join(cell.text for cell in row.values() if "$" not in cell.text)
            measure = row["measure"].text if "measure" in row else ""
            for key, cell in row.items():
                kind = f"measure:{measure}" if key.startswith("arm:") else key.removesuffix(":interval")
                cells.append((cell.text, words if _COUNT.findall(words) else heads[key], kind))
        named = [(column.header, "", "") for column in page.chart.columns]
        return [*cells, *named, *((note, "", "") for note in page.notes)]
    data = page.chart.data.model_dump(mode="json") if page.chart.data is not None else {}
    headers, rows = page.chart.values_table()
    table = [(cell, headers[at] if at < len(headers) else "", "") for row in rows for at, cell in enumerate(row)]
    # A value's own text (a point's ``x_text``) names no count, so it reads at its axis title's, never the chart's largest.
    heads = {"x_text": str(data.get("x_title") or ""), "y_text": str(data.get("y_title") or "")}
    valued = [(text, heads.get(field, ""), "") for field, text in _value_texts(data)]
    return [
        *((text, "", "") for text in _texts(data, skip=_VALUE_TEXTS)),
        *valued,
        *table,
        *((n, "", "") for n in page.notes),
    ]


def _amounts_reconcile(label: str, page: PageChart, owned: list[tuple[float, int]], totals: bool) -> int:
    """Every amount of money a chart or table shows, in any of its texts, read back by value: a figure the input
    stores or a criterion's line, times the count its own text names (else its column's head or row, else the one the
    chart or table names) over the count the figure is stated per, to the rounding of the decimals it is written to.
    A text naming no count reads figures as stored. Returns how many it read."""
    texts = _column_texts(page)
    counts = [int(count.replace(",", "")) for text, _, _ in texts for count in _COUNT.findall(text)]
    default = max(counts) if counts and not totals else None
    read = 0
    columns: dict[str, list[str]] = {}
    for text, head, column in texts:
        if "$" not in text:
            continue
        named = _COUNT.findall(text) or _COUNT.findall(head)
        count = int(named[0].replace(",", "")) if named and not totals else default
        for found in _AMOUNT.finditer(text):
            value = _amount(found.group(0))
            # Never two scales on one amount: a count named over it, and the per-1,000 mark on it.
            assert not (found.group(4) and (named or default)), (label, text, "stacked")
            places = _written_places(found) + (3 if found.group(4) else 0)
            rounding = 10.0**-places / 2 + abs(value) * 1e-9
            # A figure too small to show at the decimals written is no reading of a zero: a zero reads a zero.
            near = [
                abs(one) * (count / per if count and per else 1.0) for one, per in owned if (one == 0) == (value == 0)
            ]
            assert any(abs(abs(value) - one) <= rounding and (value == 0 or one >= rounding) for one in near), (
                label,
                text,
                count,
            )
            if column and value != 0:
                columns.setdefault(column, []).append(found.group(0))
            read += 1
    # One figure type in a table (a column, an interval's ends with its figures, a measure across the arms) reads in
    # one form, each amount to the form's figures; a zero ($0) says nothing of a form.
    assert all(_one_form(amounts) for amounts in columns.values()), (
        label,
        str({column: amounts for column, amounts in columns.items() if not _one_form(amounts)}),
    )
    return read


def test_a_chart_of_totals_under_a_cent_reads_plain_dollars() -> None:
    """A run's trials' cost counts nothing per anything: drawn as stored, never a thousand times over, so its ruler
    cannot move. Its set is under a cent, so it reads in plain dollars to two significant figures as a key and a table
    do (Q3, Q4): its ticks in the kit's ``usd`` to the form's decimals, its bars' notes and its values table in the
    same spelling, and no count named, since the set states none."""
    held, *_ = UNDER_A_CENT["r5 under a cent per 1,000 replies"]
    spec, faults = complete_chart(
        _planned("bars", _query("scoreboard", ["test_cost_usd"])), held, "block", forms=PageForms.of(held)
    )
    assert not faults, faults
    page = resolve_chart(
        spec, block_look(held, fact_arms(held)), title="", fallback_title="t", forms=PageForms.of(held)
    )
    assert isinstance(page.chart, ChartBlock) and isinstance(page.chart.data, BarsChart)
    rows = page.chart.data.rows or []
    stored = {arm: row.test_cost_usd for arm, row in held.scoreboard.items()}
    assert rows and all(row.value == pytest.approx(stored[row.id]) for row in rows)
    form = money_form(stored.values())
    assert min(stored.values()) < CENT, "the control: the planted set is under a cent"
    assert page.chart.data.fmt == f"usd:{form.decimals}"
    _, table = page.chart.values_table()
    assert [line[1] for line in table] == [form.spell(stored[row.id]) for row in rows]
    assert not any("/1000" in text for text in [*(row.note or "" for row in rows), *(line[1] for line in table)])
    assert not any("per 1,000" in note for note in page.notes)
    assert not any("/1000" in note for note in page.notes)


def test_a_counted_set_at_the_cap_never_carries_the_mark_and_a_total_carries_one_scale() -> None:
    """The MONEY set ruling: a set still under a cent at 1,000,000 of what it counts reads at that count, never with
    the mark beside it; a total reads plain dollars and no count (the mark retired, Q4). Neither amount carries two
    scales."""
    counted = money_ruler([0.0000000012, 0.0000000030], per=1, start=1000)
    assert (counted.counted, counted.form.per_thousand) == (1_000_000, False)
    assert "/1000" not in counted.spell(0.0000000012)
    total = money_ruler([0.0004, 0.008], per=0)
    assert (total.counted, total.form.per_thousand, total.words("replies")) == (0, False, "")
    assert total.spell(0.008) == "$0.0080"


def test_the_kit_money_cases_read_as_the_package_spells_them() -> None:
    """Each ``kit_money`` case of the number format cases: the package writes in a chart's notes and values table
    what the kit draws on its ticks (the same cases, read by the kit in headless Chromium)."""
    cases = json.loads(_CASES.read_text(encoding="utf-8"))["kit_money"]["cases"]
    assert {case["fmt"].lstrip("+").split(":")[0] for case in cases} >= {"usdpk", "usd"}
    for case in cases:
        name, _, digits = case["fmt"].lstrip("+").partition(":")
        form = MoneyForm(per_thousand=name == "usdpk", decimals=int(digits))
        assert form.spell(case["value"], signed=case["fmt"].startswith("+")) == case["expected"], case["why"]
        drawn = BarsChart(rows=[BarRow(label="a", value=case["value"])], domain=(0, 1), fmt=case["fmt"])
        assert drawn.table([])[1][0][1] == case["expected"], case["why"]


def test_a_change_chart_scales_by_its_amounts_never_by_an_interval_end_near_zero() -> None:
    """r5's cost contrasts with one interval end planted within a cent of zero: the chart is drawn on the measure's
    own ruler (per 1,000 replies), as its amounts are, never per 1,000,000 because of the end."""
    held = INPUTS["r5"]
    contrasts = [
        c.model_copy(update={"interval": (0.0000001, c.interval[1])})
        if c.measure == "production_replicating_cost" and c.interval
        else c
        for c in held.contrasts
    ]
    planted = held.model_copy(update={"contrasts": contrasts})
    spec, faults = complete_chart(
        _planned("diverging_bars", _query("contrasts", ["production_replicating_cost"])),
        planted,
        "block",
        forms=PageForms.of(planted),
    )
    assert not faults, faults
    page = resolve_chart(
        spec, block_look(planted, fact_arms(planted)), title="", fallback_title="t", forms=PageForms.of(planted)
    )
    joined = " | ".join(_page_texts(page))
    assert "per 1,000,000" not in joined
    assert "per 1,000 replies" in joined


def test_a_change_under_50_cents_between_large_totals_never_reads_zero() -> None:
    form = money_form([1208.0, 1208.4])
    assert form.decimals == 2 and form.spell(1208.4) == "$1,208.40", "a total of $10 or more reads to cents (Q3)"
    assert form.for_changes([0.4]).spell(0.4, signed=True) == "+$0.40"
    assert form.for_changes([0.4], [30.0]).decimals == 0, "an interval $30 wide supports no cents"


def test_a_change_columns_precision_is_capped_by_its_widest_interval() -> None:
    assert money_form([1.892, 2.427]).for_changes([0.4924, 0.616], [0.636, 0.474]).decimals == 2
    assert money_form([1.892, 2.427]).for_changes([0.4924], [6.36]).decimals == 2
    assert money_form([0.0012, 0.0015]).for_changes([0.000004], [0.002]).decimals == 4


def _part_priced(held: ReportInput) -> tuple[ReportInput, str]:
    """The input with its first arm's cost counting only part of its calls."""
    arm = held.arms[0].id
    return _priced(held, {arm: Pricing(sources=["openrouter:reported"], calls=897, priced_calls=850)}), arm


def test_a_lattice_tile_says_a_partly_priced_cost_beside_it() -> None:
    moved, arm = _part_priced(INPUTS["r3"])
    spec, faults = complete_chart(_planned("lattice", _query("configs", [])), moved, "block", forms=PageForms.of(moved))
    assert not faults, faults
    page = resolve_chart(
        spec, block_look(moved, fact_arms(moved)), title="", fallback_title="t", forms=PageForms.of(moved)
    )
    assert isinstance(page.chart, ChartBlock) and isinstance(page.chart.data, LatticeChart)
    tips = {cell.id: " ".join(cell.tip or []) for row in page.chart.data.rows for cell in row.cells}
    assert "850 of 897 calls priced" in tips[arm], tips[arm]
    assert all("calls priced" not in tip for other, tip in tips.items() if other != arm)


def test_a_task_matrix_cell_says_a_partly_priced_cost_beside_it() -> None:
    moved, arm = _part_priced(INPUTS["r1"])
    spec, faults = complete_chart(
        _planned("gate_matrix", _query("tasks", [])), moved, "block", forms=PageForms.of(moved)
    )
    assert not faults, faults
    page = resolve_chart(
        spec, block_look(moved, fact_arms(moved)), title="", fallback_title="t", forms=PageForms.of(moved)
    )
    assert isinstance(page.chart, ChartBlock) and isinstance(page.chart.data, GateMatrixChart)
    column = [one.label for one in page.chart.data.columns].index(block_look(moved, fact_arms(moved)).arm(arm).name)
    costed = [row.cells[column] for row in page.chart.data.rows if row.cells[column].cost is not None]
    assert costed and all("850 of 897 calls priced" in (cell.reasons or []) for cell in costed)
    others = [cell for row in page.chart.data.rows for at, cell in enumerate(row.cells) if at != column]
    assert all("850 of 897 calls priced" not in (cell.reasons or []) for cell in others)


def test_r2s_cost_gate_under_a_cent_reads_its_reason_and_line_on_the_criterions_ruler() -> None:
    """The gate's reason and its row's line: both figures on the criterion's ruler, moved per 1,000,000 calls, the
    count named once."""
    held, *_ = UNDER_A_CENT["r2 under a cent per 1,000 calls"]
    assert "Cost budget, per 1,000,000 calls $1.00 > $0.30" in gate_check(held, "atlas-m.off.r0").fails
    spec, faults = complete_chart(
        _planned("gate_matrix", _query("measures", [])), held, "block", forms=PageForms.of(held)
    )
    assert not faults, faults
    page = resolve_chart(
        spec, block_look(held, fact_arms(held)), title="", fallback_title="t", forms=PageForms.of(held)
    )
    assert isinstance(page.chart, ChartBlock) and isinstance(page.chart.data, GateMatrixChart)
    details = [row.detail or "" for row in page.chart.data.rows]
    assert any(detail.endswith("$0.30 per 1,000,000 calls") for detail in details), details


def test_a_partly_priced_arms_cost_in_the_full_results_says_so_beside_it() -> None:
    moved, arm = _part_priced(INPUTS["r1"])
    (table,) = [
        one
        for one in every_table(moved, block_look(moved, fact_arms(moved)), forms=PageForms.of(moved))
        if one.name == "arms_by_measure"
    ]
    cost = next(
        column.key for column in table.columns if column.key.startswith("measure:cost") or column.key == "cost_per_1k"
    )
    flags = {row["arm"].value: row[cost].flag for row in table.rows}
    look = block_look(moved, fact_arms(moved))
    assert flags[look.arm(arm).name] == "850 of 897 calls priced", flags
    assert all(flag is None or "calls priced" not in flag for name, flag in flags.items() if name != look.arm(arm).name)


def test_a_task_matrix_cell_without_a_gate_never_repeats_its_cost_among_its_reasons() -> None:
    held = fixture_inputs()["r1 without its gates"]
    spec, faults = complete_chart(
        _planned("gate_matrix", _query("tasks", ["cost_per_1k", "latency_s"])), held, "block", forms=PageForms.of(held)
    )
    assert not faults, faults
    page = resolve_chart(
        spec, block_look(held, fact_arms(held)), title="", fallback_title="t", forms=PageForms.of(held)
    )
    assert isinstance(page.chart, ChartBlock) and isinstance(page.chart.data, GateMatrixChart)
    ungated = [cell for row in page.chart.data.rows for cell in row.cells if cell.status == "neutral" and cell.cost]
    assert ungated, "the control: r1's matrix has ungated cells with a cost"
    for cell in ungated:
        # Its other measure's words, but never the cost it prints in its cost slot.
        assert cell.reasons and not any("$" in reason for reason in cell.reasons), cell.reasons
