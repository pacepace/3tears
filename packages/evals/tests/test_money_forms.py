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
from collections.abc import Iterable
from pathlib import Path

import pytest

from packages.evals.tests.analyst_support import KAIRO, fixture_inputs, fixture_snapshots
from packages.evals.tests.fixtures.report_plans.dataset import kairo_bundle
from threetears.evals.analysis import ABSENT, UNPRICED, dollars_text, format_number
from threetears.evals.analysis.analyst.snapshot import analyst_input, frozen_input
from threetears.evals.analysis.bundle.schema import ArmPricing
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.blocks import TableBlock, TableCell, TableColumn
from threetears.evals.analysis.report.block_spec import block_look
from threetears.evals.analysis.report.charts import even_precision
from threetears.evals.analysis.report.figure_keys import Figure, format_figure, render, resolve
from threetears.evals.analysis.report.plan_page.look import figure_text, quantity
from threetears.evals.analysis.report.plan_page.model import PLANNED_SECTION
from threetears.evals.analysis.report.plan_page.tables import arms_table
from threetears.evals.analysis.report_input import Pricing, ReportInput, Unanswered
from threetears.evals.analysis.report_plan_facts import fact_arms
from threetears.evals.kernel.numbers import MoneyForm, money_form, spell_dollars

#: r1 to r5, and Kairo round 2 frozen from its bundle.
INPUTS: dict[str, ReportInput] = {
    **{name: held for name, held in fixture_inputs().items() if name in ("r1", "r2", "r3", "r4", "r5")},
    KAIRO: frozen_input(fixture_snapshots("s")[KAIRO]),
}

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


def test_a_table_block_reads_each_columns_money_in_one_form() -> None:
    table = TableBlock(
        section=PLANNED_SECTION,
        name="planted",
        title="Planted",
        columns=[TableColumn(key="arm", header="Arm"), TableColumn(key="cost", header="Cost")],
        rows=[
            {"arm": TableCell(value="A"), "cost": TableCell(value="$0.008")},
            {"arm": TableCell(value="B"), "cost": TableCell(value="$0.11")},
        ],
        order="as planted",
        total_rows=2,
    )
    cells = [str(row["cost"].value) for row in table.one_money_form().rows]
    assert cells == ["$8/1000", "$110/1000"]


@pytest.mark.parametrize("name", sorted(INPUTS))
def test_a_charts_notes_and_ticks_read_in_one_form(name: str) -> None:
    held = INPUTS[name]
    costs = [float(figure.value) for figure in _arms(held, "cost_per_1k.{arm}").values()]
    if not costs:
        pytest.skip(f"{name} states no cost")
    drawn = quantity("usd_per_1k", costs)
    notes = [figure_text(cost * drawn.factor, drawn) for cost in costs]
    assert len(_forms(notes)) == 1, notes
    assert drawn.money is not None
    if not drawn.money.per_thousand:
        assert drawn.fmt == f"usd:{drawn.money.decimals}"


def test_a_planted_chart_under_a_cent_notes_per_1000() -> None:
    drawn = quantity("usd_per_1k", [0.008, 0.11])
    assert [figure_text(value, drawn) for value in (0.008, 0.11)] == ["$8.00/1000", "$110.00/1000"]


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
        form = money_form(means).for_change(high - low)
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
    assert text == form.spell(costs[first] - costs[second], signed=True)


# --- the monthly figure reconciles -------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(name for name, held in INPUTS.items() if held.meta.volume_per_month))
def test_a_cost_per_1000_times_the_volume_reaches_the_monthly_figure_within_the_forms_rounding(name: str) -> None:
    held = INPUTS[name]
    volume = float(held.meta.volume_per_month or 0)
    checked = 0
    for arm in _arms(held, "cost_per_1k.{arm}"):
        price_text = render(f"{{cost_per_1k.{arm}|usd_per_1k}}", held, {})
        month_text = render(f"{{cost_per_month.{arm}|usd}}", held, {})
        price, month = _amount(price_text), _amount(month_text)
        (per_thousand, decimals), *_ = _forms([price_text])
        rounding = 10.0**-decimals / 2 / (1000 if per_thousand else 1)
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
