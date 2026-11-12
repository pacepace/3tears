"""One arm states one cost on a page, whichever table or key states it.

An arm the page's cost measure holds no figure for, but whose scoreboard row holds a cost per reply, is a case the
cost rule has to answer once: the options table, the full results' arms-by-measure table and the writer's
``cost_per_1k`` key read the same figure for it, and an arm the measure does hold keeps the measure's own figure.
"""

from __future__ import annotations

from threetears.evals.analysis.report.block_spec import block_look
from threetears.evals.analysis.report.blocks import TableBlock, TableCell
from threetears.evals.analysis.report.figure_keys import aspect_costs_per_thousand, cost_basis, resolve
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.analysis.report.parts.full_results import every_table
from threetears.evals.analysis.report.parts.tables import arms_table
from threetears.evals.analysis.report_arms import fact_arms
from threetears.evals.analysis.report_input import Aspect, ReportInput, ScoreboardRow, Stat

from packages.evals.tests.analyst_support import fixture_inputs

#: r5's cost measure (dollars per reply) and the arm the input below leaves out of it, whose scoreboard row holds
#: ``0.002406`` dollars a reply.
_COST = "production_replicating_cost"
_LEFT_OUT = "1017"


def _without_a_cost_figure() -> ReportInput:
    held = fixture_inputs()["r5"]
    measure = held.measures[_COST]
    bare = measure.model_copy(update={"per_arm": {a: s for a, s in measure.per_arm.items() if a != _LEFT_OUT}})
    return held.model_copy(update={"measures": {**held.measures, _COST: bare}})


def _tables(held: ReportInput) -> tuple[TableBlock, TableBlock]:
    forms = PageForms.of(held)
    look = block_look(held, fact_arms(held))
    full = next(
        table
        for table in every_table(held, look, forms=forms)
        if any(column.key == f"measure:{_COST}" for column in table.columns)
    )
    return arms_table(held, look, (), forms=forms), full


def _cell(table: TableBlock, held: ReportInput, key: str, arm: str) -> TableCell:
    """The arm's cell in a column of the options table (a row per arm, in the input's order) or of the full results
    table (found by the arm's chip)."""
    if key.startswith("cost_per_1k"):
        row = table.rows[[held_arm.id for held_arm in held.arms].index(arm)]
    else:
        row = next(row for row in table.rows if str(row["arm"].value) == arm)
    cell = row[key]
    assert isinstance(cell, TableCell)
    return cell


def test_an_arm_the_cost_measure_lacks_states_its_cost_per_reply_in_both_tables_and_the_key() -> None:
    held = _without_a_cost_figure()
    assert _LEFT_OUT not in held.measures[_COST].per_arm
    assert held.scoreboard[_LEFT_OUT].reply_cost_usd == 0.002406
    options, full = _tables(held)
    assert _cell(options, held, "cost_per_1k", _LEFT_OUT).value == "$2.41"
    assert _cell(full, held, f"measure:{_COST}", _LEFT_OUT).value_text == "$2.41"
    key = resolve(f"cost_per_1k.{_LEFT_OUT}", held)
    assert key.value == 2.406 and key.per == "replies"
    assert aspect_costs_per_thousand(held, None)[_LEFT_OUT] == 2.406


def test_the_arms_the_measure_holds_keep_its_figures() -> None:
    held = _without_a_cost_figure()
    options, full = _tables(held)
    for arm, text in (("September", "$1.89"), ("qwen3.8-flash", "$2.43")):
        assert _cell(options, held, "cost_per_1k", arm).value == text
        assert _cell(full, held, f"measure:{_COST}", arm).value_text == text


def test_every_arm_the_measure_holds_is_unchanged_by_the_fallback() -> None:
    held = fixture_inputs()["r5"]
    options, full = _tables(held)
    assert cost_basis(held).source == _COST
    for arm in ("September", "qwen3.8-flash", _LEFT_OUT):
        assert _cell(options, held, "cost_per_1k", arm).value == _cell(full, held, f"measure:{_COST}", arm).value_text


def test_an_aspect_the_cost_measure_lacks_an_arm_on_reads_that_aspects_cost_per_reply() -> None:
    held = _without_a_cost_figure()
    arms = {"September": 0.0011, "qwen3.8-flash": 0.0026, _LEFT_OUT: 0.0021}
    aspect = Aspect(
        id="arrival",
        name="arrival",
        asked=False,
        readings=[],
        scoreboard={arm: ScoreboardRow(ran=10, usable=10, passed=5, reply_cost_usd=cost) for arm, cost in arms.items()},
    )
    measure = held.measures[_COST].model_copy(
        update={"by_aspect": {"arrival": {arm: Stat(mean=cost) for arm, cost in arms.items() if arm != _LEFT_OUT}}}
    )
    held = held.model_copy(update={"aspects": [*held.aspects, aspect], "measures": {**held.measures, _COST: measure}})
    assert aspect_costs_per_thousand(held, "arrival")[_LEFT_OUT] == 2.1
    options, full = _tables(held)
    column = "cost_per_1k@arrival"
    assert _cell(options, held, column, _LEFT_OUT).value == "$2.10"
    rows = [row for row in full.rows if row["aspect"].value == "arrival" and row["arm"].value == _LEFT_OUT]
    assert [row[f"measure:{_COST}"].value_text for row in rows] == ["$2.10"]
