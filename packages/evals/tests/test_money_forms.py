"""Money reads one form per set (#993): a key family, a table's column and a chart each spell every amount alike.

Owner ruling of 2026-10-27: tiny amounts read per 1,000 (``$8/1000``), and within a figure type every value is spelled
in one form, so successive rows never alternate between ``$0.008`` and ``$0.11``. Contracts held here:

- :func:`~threetears.evals.kernel.numbers.money_form` chooses the form from the whole set: per 1,000 when any member is
  under a cent, then the decimals that give the least member two significant figures, cents at least; a set chosen per
  value fails (a planted ``$0.008`` and ``$0.11`` read per 1,000 throughout);
- every figure-key money family, the options table's cost column and a chart's notes and ticks print one form, on r1
  to r5 and Kairo round 2, the expectation computed from each input's own figures; a values table's column reads per
  1,000 throughout where any amount in it is under a cent, each amount moved by its digits, never rounded;
- a change reads in the form of the amounts it compares, to no more decimals than its interval's width supports (the
  ``+$0.492`` on an interval tens of cents wide reads ``+$0.49``);
- a cost per 1,000 times the declared monthly volume reaches the monthly figure beside it, within the form's rounding;
- the options table states each arm's cost per run (or per 1,000) beside its pass rate, with its price source, the
  provider that served it and its priced share; a partly priced arm says ``n of m calls priced`` in place of a
  smaller total, an arm none of whose calls were priced says ``unpriced``, never ``$0``, and the unanswered mark stays;
- the single-value helpers (``format_number``, ``spell_dollars``, ``dollars_text``) are byte-identical: the number
  format cases, and Discodon's two call shapes (``dollars_text(cost)``, ``format_number(cost, prefix="$")``);
- the bundle's census carries an arm's pricing into the report input, and an arm assembled without it leaves no key,
  so a stored input digests as it did.
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
from packages.evals.tests.fixtures.report_plans.dataset import kairo_bundle
from packages.evals.tests.run_facts_support import CASES, aspect_bundle, campaign_over, result_of, run_of
from threetears.evals.analysis import ABSENT, UNPRICED, dollars_text, format_number
from threetears.evals.analysis.analyst.snapshot import analyst_input, frozen_input
from threetears.evals.analysis.bundle.schema import ArmPricing
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.blocks import ChartBlock, TableBlock, TableCell, money_columns
from threetears.evals.analysis.report.charts import BarsChart
from threetears.evals.analysis.report.block_spec import block_look, complete_chart
from threetears.evals.analysis.report.charts import even_precision
from threetears.evals.analysis.report.figure_keys import Figure, format_figure, render, resolve
from threetears.evals.analysis.report.plan_page.full_results import every_table
from threetears.evals.analysis.report.plan_page.look import figure_text, quantity
from threetears.evals.analysis.report.plan_page.model import PageChart
from threetears.evals.analysis.report.plan_page.resolve import resolve_chart
from threetears.evals.analysis.report.plan_page.tables import arms_table
from threetears.evals.kernel.report_plan import ChartQuery, PlanChartKind, PlannedChart
from threetears.evals.analysis.report_input import Pricing, ReportInput, Unanswered
from threetears.evals.analysis.report_input_build import report_input
from threetears.evals.schema.models import RoleUsage
from threetears.evals.analysis.report_plan_facts import fact_arms
from threetears.evals.kernel.numbers import MoneyForm, money_form, spell_dollars

#: r1 to r5, and Kairo round 2 frozen from its bundle.
INPUTS: dict[str, ReportInput] = {
    **{name: held for name, held in fixture_inputs().items() if name in ("r1", "r2", "r3", "r4", "r5")},
    KAIRO: frozen_input(fixture_snapshots("s")[KAIRO]),
}

#: The arms of the census's planted campaign: the control and the candidate.
BASE, FAST = "base", "fast"

#: An amount of money as a page prints it: its sign, its digits and the per-1,000 mark where it has one.
_AMOUNT = re.compile(r"([+\-]?)\$(\d[\d,]*)(?:\.(\d+))?(/1000)?")

#: The number format cases every renderer reads.
_CASES = Path(__file__).resolve().parent / "fixtures" / "number-format-cases.json"


def _forms(texts: Iterable[str]) -> set[tuple[bool, int]]:
    """The forms the amounts in some texts are written in: per 1,000 or not, and how many decimals."""
    return {
        (bool(found.group(4)), len(found.group(3) or ""))
        for text in texts
        for found in _AMOUNT.finditer(text)
        if found.group(2) != "0" or found.group(3)
    }


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


def test_a_set_with_a_member_under_a_cent_reads_per_1000_throughout() -> None:
    form = money_form([0.008, 0.11])
    assert [form.spell(amount) for amount in (0.008, 0.11)] == ["$8.00/1000", "$110.00/1000"]


def test_the_form_is_the_sets_never_one_values() -> None:
    """The mutation this guards against: spelling each member by its own form reads ``$0.11`` beside ``$8.00/1000``."""
    planted = [0.008, 0.11]
    alone = [money_form([amount]).spell(amount) for amount in planted]
    together = [money_form(planted).spell(amount) for amount in planted]
    assert len(_forms(alone)) == 2
    assert len(_forms(together)) == 1


def test_the_decimals_give_the_least_member_two_figures_and_never_fewer_than_cents() -> None:
    assert money_form([0.03, 3.477]) == MoneyForm(per_thousand=False, decimals=3)
    assert money_form([1.892, 2.427]) == MoneyForm(per_thousand=False, decimals=2)
    assert money_form([75.05, 6.014]) == MoneyForm(per_thousand=False, decimals=2)
    assert money_form([0.0013, 0.61]) == MoneyForm(per_thousand=True, decimals=2)


def test_a_set_whose_least_amount_is_ten_dollars_or_more_reads_whole_dollars_on_every_surface() -> None:
    """r4's monthly costs read alike on a chart, in a key and in a table: one rule, money_form's."""
    assert money_form([14.5, 120.0]) == MoneyForm(per_thousand=False, decimals=0)
    assert money_form([9.5, 120.0]) == MoneyForm(per_thousand=False, decimals=2)
    held = INPUTS["r4"]
    months = _arms(held, "cost_per_month.{arm}")
    values = [float(figure.value) for figure in months.values() if isinstance(figure.value, float)]
    drawn = quantity("dollars", values)
    for figure in months.values():
        assert isinstance(figure.value, float)
        key = format_figure(figure, ("usd",))
        assert key == figure_text(figure.value * drawn.factor, drawn) == money_form(values).spell(figure.value)
        assert "." not in key


def test_unknown_reads_unpriced_never_a_zero_and_is_no_member() -> None:
    form = money_form([None, 0.11, math.nan])
    assert form == money_form([0.11])
    assert form.spell(None) == UNPRICED
    assert form.spell(math.nan) == ABSENT
    assert form.spell(0) == "$0"


def test_a_planted_family_under_a_cent_reads_per_1000_in_every_key() -> None:
    """A cost per 1,000 of $0.008 beside one of $0.11: both keys of ``cost_per_1k.*`` read per 1,000."""
    held = INPUTS["r4"]
    measure = held.measures["cost_per_1k"]
    arms = list(measure.per_arm)
    planted = {arm: (0.008 if index == 0 else 0.11) for index, arm in enumerate(arms)}
    per_arm = {arm: stat.model_copy(update={"mean": planted[arm]}) for arm, stat in measure.per_arm.items()}
    moved = held.model_copy(
        update={"measures": {**held.measures, "cost_per_1k": measure.model_copy(update={"per_arm": per_arm})}}
    )
    texts = [format_figure(resolve(f"cost_per_1k.{arm}", moved), ("usd",)) for arm in arms]
    assert all(text.endswith("/1000") for text in texts), texts
    assert len(_forms(texts)) == 1


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
    factor = 1000.0 if filters == ("usd_per_1k",) and next(iter(figures.values())).scale == "usd" else 1.0
    values = {arm: float(figure.value) * factor for arm, figure in figures.items() if isinstance(figure.value, float)}
    form = money_form(values.values())
    if filters == ("usd_per_1k",) and form.per_thousand:
        # Per 1,000 a family under a cent moves its ruler to per 1,000,000, the mark never stacked on it.
        values = {arm: value * 1000 for arm, value in values.items()}
        form = money_form(values.values())
    texts = {arm: format_figure(figure, filters) for arm, figure in figures.items()}
    for arm, text in texts.items():
        assert text.startswith(form.spell(values[arm])), (arm, text)
    assert len(_forms(texts.values())) == 1, texts


@pytest.mark.parametrize("name", sorted(INPUTS))
def test_the_options_table_cost_column_reads_in_one_form(name: str) -> None:
    held = INPUTS[name]
    table = arms_table(held, block_look(held, fact_arms(held)), ())
    column = next((column for column in table.columns if column.key == "cost_per_1k"), None)
    if column is None:
        pytest.skip(f"{name} states no cost")
    costs = _arms(held, "cost_per_1k.{arm}")
    per_run = [float(figure.value) / 1000 for figure in costs.values() if isinstance(figure.value, float)]
    count = 1
    while money_form(cost * count for cost in per_run).per_thousand and count < 1_000_000:
        count *= 1000
    form = money_form(cost * count for cost in per_run)
    cells = [str(row["cost_per_1k"].value) for row in table.rows if row["cost_per_1k"].value is not None]
    assert cells == [form.spell(cost * count) for cost in per_run]
    assert ("per 1,000" in column.header) == (count > 1)
    assert len(_forms(cells)) == 1


def test_a_values_tables_column_reads_in_one_form_and_keeps_every_digit() -> None:
    """A column holding an amount under a cent reads per 1,000 throughout, each amount moved by its digits exactly; a
    column with none keeps dollars; a bound is left as it is."""
    rows = [["A", "$0.008", "+$0.616"], ["B", "$0.11", "+$0.4924"], ["C", "$1239", "< $0.000001"]]
    spelled = even_precision(rows)
    assert [row[1] for row in spelled] == ["$8/1000", "$110/1000", "$1,239,000/1000"]
    assert [row[2] for row in spelled] == ["+$0.616", "+$0.4924", "< $0.000001"]
    for row in spelled:
        assert _amount(row[1]) == pytest.approx(_amount(rows[spelled.index(row)][1]))


def test_a_tables_money_column_reads_one_form_chosen_from_its_values() -> None:
    """A column holding $0.008 and $0.11 reads both per 1,000, to the same decimals: never $8.00/1000 beside
    $110/1000; each cell is its own value at those decimals, so it states no digit the value does not hold."""
    rows = money_columns(
        [
            {"arm": TableCell(value="A"), "cost": TableCell(value=0.008, unit="usd")},
            {"arm": TableCell(value="B"), "cost": TableCell(value=0.11, unit="usd")},
        ]
    )
    assert [row["cost"].text for row in rows] == ["$8.00/1000", "$110.00/1000"]
    assert [row["arm"].text for row in rows] == ["A", "B"]


def _money_cells(table: TableBlock) -> dict[str, list[tuple[TableCell, str]]]:
    """Each column's money cells, with the text a reader is shown, by column key."""
    cells: dict[str, list[tuple[TableCell, str]]] = {}
    for row in table.rows:
        for key, cell in row.items():
            if cell.is_number and cell.unit == "usd":
                cells.setdefault(key, []).append((cell, cell.text))
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
    tables = list(every_table(held, look))
    for key, measure in held.measures.items():
        if measure.unit in ("usd", "usd_per_1k", "usd_per_1m") and measure.per_arm:
            spec, faults = complete_chart(_planned("table", _query("measures", [key])), held, "block")
            assert not faults, faults
            drawn = resolve_chart(spec, held, look, title="", fallback_title="t")
            assert isinstance(drawn.chart, TableBlock)
            tables.append(drawn.chart)
    return tables


@pytest.mark.parametrize("name", sorted(INPUTS))
def test_every_engine_tables_money_column_reads_one_form_from_its_values(name: str) -> None:
    held = INPUTS[name]
    checked = 0
    for table in _engine_tables(held):
        for key, cells in _money_cells(table).items():
            values = [float(cell.value) for cell, _ in cells if isinstance(cell.value, int | float)]
            ends = [float(end) for cell, _ in cells for end in (cell.low, cell.high) if end is not None]
            form = money_form([*values, *ends])
            assert len(_forms(text for _, text in cells)) == 1, (table.name, key)
            for cell, text in cells:
                assert isinstance(cell.value, int | float)
                assert text.split(" (")[0] == form.spell(float(cell.value)), (table.name, key, text)
            checked += 1
    if not checked:
        pytest.skip(f"{name}'s tables state no money")


def test_r2s_cost_reads_in_its_tables_as_its_key_reads_in_prose() -> None:
    """r2's brisk-s.low.r1 costs $0.302 per 1,000 calls; at 4,000,000 calls a month $0.30 would give $1,200 beside a
    monthly $1,208, so the table keeps the third figure its whole column needs."""
    held = INPUTS["r2"]
    said = format_figure(resolve("cost_per_1k.brisk-s.low.r1", held), ("usd",))
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
    assert len(_forms(notes)) == 1, notes
    assert drawn.money is not None and not drawn.money.per_thousand
    assert drawn.fmt == f"usd:{drawn.money.decimals}"
    assert _forms(notes) == {(False, drawn.money.decimals)}, "the ticks' decimals are the notes'"


def test_a_planted_chart_under_a_cent_draws_a_thousand_times_over_its_ticks_and_notes_alike() -> None:
    """A chart of $0.008 and $0.11 per 1,000 draws them per 1,000,000, as $8.00 and $110.00: its ticks (the kit's
    usd:2) and its notes in one form, and no per-1,000 mark the ticks could not draw."""
    drawn = quantity("usd_per_1k", [0.008, 0.11])
    assert (drawn.factor, drawn.counted, drawn.fmt) == (1000.0, 1_000_000, "usd:2")
    notes = [figure_text(value * drawn.factor, drawn) for value in (0.008, 0.11)]
    assert notes == ["$8.00", "$110.00"]
    assert _forms(notes) == {(False, 2)}


def test_a_cost_scatter_labels_its_money_in_its_axis_form() -> None:
    """A cost-against-quality scatter over r5's cost per reply: drawn per 1,000 replies, its money reads $1.89-class,
    never 1,000 times that."""
    held = INPUTS["r5"]
    spec, faults = complete_chart(
        _planned("scatter_frontier", _query("measures", ["production_replicating_cost", "delivered_items"])),
        held,
        "block",
    )
    assert not faults, faults
    drawn = resolve_chart(spec, held, block_look(held, fact_arms(held)), title="", fallback_title="t")
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
        form = money_form(means).for_changes([float(contrast.delta)], [high - low])
        text = format_figure(resolve(f"contrast.{contrast.arm}.{contrast.measure}.delta", held))
        assert text == form.spell(float(contrast.delta), signed=True)
        assert form.decimals <= max(2, 1 - math.floor(math.log10((high - low) * (1000 if form.per_thousand else 1))))
        checked += 1
    if not checked:
        pytest.skip(f"{name} holds no money contrast with an interval")


def test_the_kairo_cost_contrast_reads_two_figures_of_its_interval() -> None:
    held = INPUTS[KAIRO]
    text = format_figure(resolve("contrast.1017.production_replicating_cost.delta", held), ("usd_per_1k",))
    assert text.startswith("+$0.49 "), text


@pytest.mark.parametrize("name", sorted(INPUTS))
def test_a_cost_gap_reads_in_the_form_of_the_costs_it_compares(name: str) -> None:
    held = INPUTS[name]
    costs = {arm: float(figure.value) for arm, figure in _arms(held, "cost_per_1k.{arm}").items()}
    if len(costs) < 2:
        pytest.skip(f"{name} states fewer than two costs")
    form = money_form(costs.values())
    first, second = list(costs)[:2]
    text = format_figure(resolve(f"cost.gap.{first}.{second}.per_1k", held), ("usd",))
    # A gap takes its precision from its own size, in the form (the scale) of the costs it compares (#993).
    gap = costs[first] - costs[second]
    assert text == form.for_changes([gap]).spell(gap, signed=True)


# --- the monthly figure reconciles -------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(name for name, held in INPUTS.items() if held.meta.volume_per_month))
def test_a_cost_per_1000_times_the_volume_reaches_the_monthly_figure_within_the_forms_rounding(name: str) -> None:
    held = INPUTS[name]
    volume = float(held.meta.volume_per_month or 0)
    checked = 0
    for arm in _arms(held, "cost_per_1k.{arm}"):
        price_text = render(f"{{cost_per_1k.{arm}|usd_per_1k}}", held, {})
        month_text = render(f"{{cost_per_month.{arm}|usd}}", held, {})
        price, month = _amount(price_text) / (1000 if "per 1,000,000" in price_text else 1), _amount(month_text)
        (per_thousand, decimals), *_ = _forms([price_text])
        rounding = 10.0**-decimals / 2 / (1000 if per_thousand or "per 1,000,000" in price_text else 1)
        (_, month_decimals), *_ = _forms([month_text])
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
    table = arms_table(moved, block_look(moved, fact_arms(moved)), ())
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
    table = arms_table(moved, block_look(moved, fact_arms(moved)), ())
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
    table = arms_table(moved, block_look(moved, fact_arms(moved)), ())
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
    (0.000160, "$0.000160", "$0.00016"),
    (0.0123, "$0.0123", "$0.0123"),
    (0.302, "$0.30", "$0.302"),
    (0.5, "$0.50", "$0.5"),
    (1.6, "$1.60", "$1.6"),
    (7560.0, "$7,560.00", "$7560"),
    (-1.2, "-$1.20", "-$1.2"),
)


@pytest.mark.parametrize(("amount", "dollars", "prefixed"), _DISCODON_SHAPES)
def test_discodons_two_call_shapes_read_as_they_did(amount: float, dollars: str, prefixed: str) -> None:
    assert dollars_text(amount) == dollars == spell_dollars(amount)
    assert format_number(amount, prefix="$") == prefixed
    assert dollars_text(None) == UNPRICED


# --- the census carries an arm's pricing ---------------------------------------------------------------------------


def test_an_arms_pricing_reaches_the_report_input_and_an_arm_without_it_carries_no_key() -> None:
    bundle = kairo_bundle()
    plain = analyst_input(bundle)
    assert all(row.pricing is None for row in plain.scoreboard.values())
    assert '"pricing"' not in plain.model_dump_json()
    pricing = ArmPricing(sources=["openrouter:reported"], providers=["DeepInfra"], calls=10, priced_calls=9)
    aspects = [
        aspect.model_copy(
            update={"scoreboard": [row.model_copy(update={"pricing": pricing}) for row in aspect.scoreboard]}
        )
        for aspect in bundle.aspects
    ]
    priced = analyst_input(bundle.model_copy(update={"aspects": aspects}))
    for row in priced.scoreboard.values():
        assert row.pricing == Pricing(
            sources=["openrouter:reported"], providers=["DeepInfra"], calls=10, priced_calls=9
        )


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
    table = arms_table(planted, block_look(planted, fact_arms(planted)), ())
    (column,) = [column for column in table.columns if column.key == "cost_per_1k"]
    assert column.header == "Cost per reply"
    cells = [str(row["cost_per_1k"].value) for row in table.rows]
    assert cells == [money_form([0.05, 0.06, 0.07, 0.08]).spell(0.05 + 0.01 * index) for index in range(len(arms))]
    assert cells[0] == "$0.050"


def test_an_options_table_under_a_cent_per_1000_runs_reads_per_1000000() -> None:
    held = INPUTS["r3"]
    table = arms_table(held, block_look(held, fact_arms(held)), ())
    (column,) = [column for column in table.columns if column.key == "cost_per_1k"]
    assert column.header == "Cost per 1,000,000 turns"
    assert all("/1000" not in str(row["cost_per_1k"].value) for row in table.rows)


def test_r3s_price_per_1000_moves_its_ruler_and_stacks_no_mark() -> None:
    """r3's least cost is $0.0013 per 1,000 turns: its usd_per_1k keys read per 1,000,000 turns, never
    $180.00/1000 per 1,000."""
    held = INPUTS["r3"]
    text = format_figure(resolve("measure.cost_per_1m.atlas-m.zero.mean", held), ("usd_per_1k",))
    assert text.startswith("$180.00 per 1,000,000 "), text
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
    table = arms_table(moved, block_look(moved, fact_arms(moved)), ())
    flag = _row(table, moved, candidate)["cost_per_1k"].flag or ""
    assert "no uplift: the control has 850 of 897 calls priced" in flag
    assert "%" not in flag


def test_a_partly_priced_arms_cost_keys_say_so_beside_the_amount() -> None:
    moved, control, candidate = _part_priced_control()
    assert format_figure(resolve(f"cost_per_1k.{control}", moved), ("usd",)).endswith(" (850 of 897 calls priced)")
    assert "calls priced" not in format_figure(resolve(f"cost_per_1k.{candidate}", moved), ("usd",))
    assert "850 of 897 calls priced" in format_figure(resolve(f"cost.uplift.{candidate}", moved))


def test_a_partly_priced_arms_bar_says_so_beside_its_cost() -> None:
    moved, control, _ = _part_priced_control()
    spec, faults = complete_chart(_planned("bars", _query("measures", ["production_replicating_cost"])), moved, "block")
    assert not faults, faults
    drawn = resolve_chart(spec, moved, block_look(moved, fact_arms(moved)), title="", fallback_title="t")
    assert isinstance(drawn.chart, ChartBlock)
    data = drawn.chart.data
    assert isinstance(data, BarsChart) and data.rows is not None
    notes = {row.id: row.note or "" for row in data.rows}
    assert notes[control].endswith(" (850 of 897 calls priced)")
    assert all("calls priced" not in note for arm, note in notes.items() if arm != control)


# --- the census reads an arm's pricing off its stored usage rows -------------------------------------------------------


def _row_of(role: str, *, cost: float | None, calls: int | None, source: str = "", provider: str = "") -> RoleUsage:
    return RoleUsage(
        role=role,
        model=FAST,
        cost_usd=cost,
        price_source=source or None,
        call_count=calls,
        provider=provider or None,
    )


def test_the_census_reads_an_arms_pricing_off_its_production_usage_rows() -> None:
    """Sources from the priced rows only, providers from every row, the calls and priced calls the rows count; the
    judge's rows, which no production deployment pays for, are no part of it."""
    usage = [
        [
            _row_of("candidate", cost=0.002, calls=3, source="openrouter:reported"),
            _row_of("external", cost=None, calls=1, provider="searx"),
            _row_of("judge", cost=0.5, calls=1, source="anthropic:published"),
        ]
        for _ in range(CASES)
    ]
    base, fast = run_of(BASE), run_of(FAST)
    trials = [
        (base, [result_of(base, index) for index in range(CASES)]),
        (fast, [result_of(fast, index, usage=rows) for index, rows in enumerate(usage)]),
    ]
    _, bundle, _ = aspect_bundle(trials)
    (pricing,) = [row.pricing for row in bundle.scoreboard if row.pricing is not None and row.pricing.providers]
    assert pricing == ArmPricing(
        sources=["openrouter:reported"], providers=["searx"], calls=4 * CASES, priced_calls=3 * CASES
    )
    built = report_input(campaign_over(trials))
    (row,) = [row for row in built.scoreboard.values() if row.pricing is not None and row.pricing.providers]
    assert row.pricing == Pricing(
        sources=["openrouter:reported"], providers=["searx"], calls=4 * CASES, priced_calls=3 * CASES
    )
    table = arms_table(built, block_look(built, fact_arms(built)), ())
    assert any(
        cell == f"{3 * CASES} of {4 * CASES} calls priced" for cell in (r["cost_per_1k"].value for r in table.rows)
    )


def test_a_row_that_did_not_count_its_calls_leaves_the_priced_share_uncounted() -> None:
    usage = [[_row_of("candidate", cost=0.002, calls=None, source="openrouter:reported")] for _ in range(CASES)]
    base, fast = run_of(BASE), run_of(FAST)
    trials = [
        (base, [result_of(base, index) for index in range(CASES)]),
        (fast, [result_of(fast, index, usage=rows) for index, rows in enumerate(usage)]),
    ]
    _, bundle, _ = aspect_bundle(trials)
    priced = [row.pricing for row in bundle.scoreboard if row.pricing is not None and row.pricing.calls is None]
    assert priced and priced[0].priced_calls is None and priced[0].sources == ["openrouter:reported"]


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


def _under_a_cent() -> dict[str, tuple[ReportInput, str, str, str]]:
    """Each planted input whose money is under a cent where its unit states it, with the money measure, what it counts,
    the words its own unit's ruler reads (which no scaled chart may print) and the words it must read instead."""
    r2 = _measures_shrunk(INPUTS["r2"], frozenset({"cost_per_1k"}), 1000.0)
    r5 = _measures_shrunk(INPUTS["r5"], frozenset({"production_replicating_cost", "cost_usd"}), 1000.0)
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
    return {
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
    """Every query a money chart can be asked on an input: each source, over every measure and over the money one,
    and two arms compared."""
    sources = get_args(ChartQuery.model_fields["source"].annotation)
    arms = [arm.id for arm in held.arms]
    queries = []
    for source in sources:
        # The scoreboard's own money: its cost per reply and its trials' total, which counts nothing.
        named = [["reply_cost_usd"], ["test_cost_usd"]] if source == "scoreboard" else [[key]]
        for measures in ([], *named):
            queries.append(_query(source, measures))
            queries.append(ChartQuery.model_validate({**_query(source, measures).model_dump(), "arms": arms[:2]}))
    return queries


def _drawn_money(held: ReportInput, key: str) -> list[tuple[str, PageChart]]:
    """Every chart and table of every kind the page draws money on, on the input, as the page draws it."""
    look = block_look(held, fact_arms(held))
    drawn = []
    for kind in get_args(PlanChartKind):
        for query in _queries(held, key):
            spec, faults = complete_chart(_planned(kind, query), held, "block")
            if faults:
                continue
            try:
                page = resolve_chart(spec, held, look, title="", fallback_title="t")
            except UnresolvableReference:
                # NOSILENT: a kind the input cannot draw over this query is no chart of money; the sweep reads on.
                continue
            if _draws_money(page):
                drawn.append((f"{kind}/{query.source}/{','.join(query.measures) or 'all'}", page))
    return drawn


def _draws_money(page: PageChart) -> bool:
    """Whether a drawn chart or table states some amount of money among the values it draws."""
    if isinstance(page.chart, TableBlock):
        return any("$" in cell.text for row in page.chart.rows for cell in row.values())
    _, rows = page.chart.values_table()
    return any("$" in cell for row in rows for cell in row)


def _texts(node: Any) -> Iterator[str]:
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _texts(value)
    elif isinstance(node, list):
        for value in node:
            yield from _texts(value)


def _page_texts(page: PageChart) -> list[str]:
    if isinstance(page.chart, TableBlock):
        headers = [column.header for column in page.chart.columns]
        return [*headers, *(cell.text for row in page.chart.rows for cell in row.values()), *page.notes]
    data = page.chart.data.model_dump(mode="json") if page.chart.data is not None else {}
    _, rows = page.chart.values_table()
    return [*_texts(data), *(cell for row in rows for cell in row), *page.notes]


@pytest.mark.parametrize("name", sorted(UNDER_A_CENT))
def test_every_kind_that_draws_money_under_a_cent_moves_its_ruler_and_states_it(name: str) -> None:
    """Every chart and table kind that draws a set under a cent: it moves the ruler (no per-1,000 mark on a chart,
    whose ticks cannot write one), never prints its own unit's ruler beside the moved one, and states the drawn count;
    its ticks and its notes read in one form."""
    held, key, stale, moved = UNDER_A_CENT[name]
    drawn = _drawn_money(held, key)
    kinds = {label.split("/")[0] for label, _ in drawn}
    assert {"bars", "table", "dot_interval", "diverging_bars"} <= kinds or name.startswith("r2"), kinds
    assert len(kinds) >= 3, kinds
    for label, page in drawn:
        texts = _page_texts(page)
        joined = " | ".join(texts)
        if isinstance(page.chart, ChartBlock):
            assert "/1000" not in joined, (label, joined[:400])
        assert stale not in joined, (label, joined[:600])
        if key in label or label.endswith("/all") or "reply_cost_usd" in label:
            # A chart states the count it moved its ruler to; a table either does, or writes each amount with the
            # per-1,000 mark (a figure key's own spelling, which a computed table prints as prose citing it does).
            amounts = [text for text in texts if "$" in text]
            stated = [
                text
                for text in amounts
                if any(
                    float(f"{found.group(2).replace(',', '')}.{found.group(3) or 0}")
                    for found in _AMOUNT.finditer(text)
                )
            ]
            marked = bool(stated) and all("/1000" in text for text in stated)
            assert moved in joined or (isinstance(page.chart, TableBlock) and marked), (label, joined[:600])
        if isinstance(page.chart, ChartBlock) and page.chart.data is not None:
            dump = page.chart.data.model_dump(mode="json")
            digits = {int(fmt.rsplit(":", 1)[1]) for fmt in _texts(dump) if re.fullmatch(r"\\+?usd:\\d", fmt)}
            notes = [text for text in _texts({k: v for k, v in dump.items()}) if "$" in text]
            if len(digits) == 1 and notes:
                (places,) = digits
                spelled = {decimals for per_thousand, decimals in _forms(notes) if not per_thousand}
                assert spelled <= {places} | ({0} if places == 0 else set()), (label, places, notes[:6])


def test_a_total_under_a_cent_is_never_scaled() -> None:
    """A run's trials' cost counts nothing per anything: drawn as stored, never a thousand times over."""
    held, *_ = UNDER_A_CENT["r5 under a cent per 1,000 replies"]
    spec, faults = complete_chart(_planned("bars", _query("scoreboard", ["test_cost_usd"])), held, "block")
    assert not faults, faults
    page = resolve_chart(spec, held, block_look(held, fact_arms(held)), title="", fallback_title="t")
    assert isinstance(page.chart, ChartBlock) and isinstance(page.chart.data, BarsChart)
    rows = page.chart.data.rows or []
    stored = {arm: row.test_cost_usd for arm, row in held.scoreboard.items()}
    assert rows and all(row.value == pytest.approx(stored[row.id]) for row in rows)
    assert not any("per 1,000" in note for note in page.notes)


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
        _planned("diverging_bars", _query("contrasts", ["production_replicating_cost"])), planted, "block"
    )
    assert not faults, faults
    page = resolve_chart(spec, planted, block_look(planted, fact_arms(planted)), title="", fallback_title="t")
    joined = " | ".join(_page_texts(page))
    assert "per 1,000,000" not in joined
    assert "per 1,000 replies" in joined


def test_a_change_under_50_cents_between_whole_dollar_amounts_never_reads_zero() -> None:
    form = money_form([1208.0, 1208.4])
    assert form.decimals == 0
    assert form.for_changes([0.4]).spell(0.4, signed=True) == "+$0.40"
    assert form.for_changes([0.4], [30.0]).decimals == 0, "an interval $30 wide supports no cents"


def test_a_change_columns_precision_is_capped_by_its_widest_interval() -> None:
    assert money_form([1.892, 2.427]).for_changes([0.4924, 0.616], [0.636, 0.474]).decimals == 2
    assert money_form([1.892, 2.427]).for_changes([0.4924], [6.36]).decimals == 2
    assert money_form([0.0012, 0.0015]).for_changes([0.000004], [0.002]).decimals == 2
