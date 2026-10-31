"""The page's tables lay out every aspect's figures where the input holds more than one aspect.

The input's by-arm maps hold the lead aspect's figures alone (``report_input``'s More than one aspect), so a table that
read them would show one aspect of three. Here a three-aspect input, built off the r5 fixture with two more aspects
beside its own (each its own scoreboard, cost figures and a measure of its own), is held to:

- the arms table and the options table: one column group per aspect, in the input's aspect order, of its pass rate
  (its passes over its usable trials beside it), its cost and, in the arms table, its usable trials, each head naming
  its aspect, every cost column in one form;
- every arm by measure: a row per aspect and arm, the aspect named, each figure the aspect's own;
- the arms key: each aspect's usable trials.

Every figure a cell states is the one its aspect's figure key resolves to. An input of one aspect, and one whose other
aspects hold no figures of their own, lays its tables out as for one aspect.
"""

from __future__ import annotations

import pytest

from packages.evals.tests.analyst_support import fixture_inputs
from threetears.evals.analysis.report.block_spec import block_look
from threetears.evals.analysis.report.blocks import TableBlock, TableCell
from threetears.evals.analysis.report.figure_keys import aspect_costs_per_thousand, render, resolve
from threetears.evals.analysis.report.plan_page.full_results import every_table
from threetears.evals.analysis.report.plan_page.look import figure_text, quantity
from threetears.evals.analysis.report.plan_page.tables import arms_key, arms_table, options_table
from threetears.evals.analysis.report_input import Aspect, Measure, ReportInput, ScoreboardRow, Stat
from threetears.evals.analysis.report_plan_facts import fact_arms
from threetears.evals.kernel.report_plan import OptionRow, Options, ReportPlan

#: The cost measure the r5 input's cost is read from, and the measure only the second aspect reads.
_COST = "production_replicating_cost"
_ARRIVAL_ONLY = "goal_state:arrival_greets"

#: The two aspects added beside r5's own, each arm's (passed, usable, ran, cost per reply in dollars) there.
_ADDED: dict[str, dict[str, tuple[int, int, int, float]]] = {
    "arrival": {
        "September": (127, 180, 180, 0.0011),
        "qwen3.8-flash": (125, 180, 180, 0.0026),
        "1017": (120, 179, 180, 0.0021),
    },
    "queue discipline": {
        "September": (174, 240, 240, 0.0009),
        "qwen3.8-flash": (154, 240, 240, 0.0027),
        "1017": (160, 238, 240, 0.0018),
    },
}


def _three_aspects() -> ReportInput:
    """r5 with two aspects beside its own, the lead (its own, asked): each its own scoreboard and figures."""
    held = fixture_inputs()["r5"]
    lead = held.aspects[0].model_copy(update={"name": "character"})
    added = [
        Aspect(
            id=aspect,
            name=aspect,
            asked=False,
            readings=[],
            scoreboard={
                arm: ScoreboardRow(ran=ran, usable=usable, passed=passed, reply_cost_usd=cost)
                for arm, (passed, usable, ran, cost) in rows.items()
            },
        )
        for aspect, rows in _ADDED.items()
    ]
    cost = held.measures[_COST]
    measures = dict(held.measures)
    measures[_COST] = cost.model_copy(
        update={
            "by_aspect": {
                aspect: {arm: Stat(mean=row[3]) for arm, row in rows.items()} for aspect, rows in _ADDED.items()
            }
        }
    )
    measures[_ARRIVAL_ONLY] = Measure(
        label="arrival greeting",
        unit="share",
        higher_is_better=True,
        by_aspect={"arrival": {arm: Stat(mean=0.5 + index / 10) for index, arm in enumerate(_ADDED["arrival"])}},
    )
    return held.model_copy(update={"aspects": [lead, *added], "measures": measures})


def _row(table: TableBlock, report_input: ReportInput, arm: str) -> dict[str, TableCell]:
    index = [held.id for held in report_input.arms].index(arm)
    return table.rows[index]


def _pass_rate_text(passed: float, usable: float) -> str:
    drawn = quantity("share")
    return figure_text(passed / usable * drawn.factor, drawn)


def _key_count(report_input: ReportInput, aspect: str, arm: str, statistic: str) -> int:
    figure = resolve(f"aspects.{aspect}.scoreboard.{arm}.{statistic}", report_input)
    assert isinstance(figure.value, float)
    return int(figure.value)


def test_the_arms_table_lays_out_a_column_group_per_aspect_in_the_inputs_order() -> None:
    held = _three_aspects()
    table = arms_table(held, block_look(held, fact_arms(held)), ())
    keys = [column.key for column in table.columns]
    aspects = [aspect.id for aspect in held.aspects]
    grouped = [key for key in keys if "@" in key]
    assert grouped == [
        f"{family}@{aspect}" for aspect in aspects for family in ("pass_rate", "cost_per_1k", "usable")
    ], "one group per aspect, in the input's order: pass rate, cost, trials"
    assert "pass_rate" not in keys and "cost_per_1k" not in keys, "no column reads the lead aspect alone"
    headers = {column.key: column.header for column in table.columns}
    assert headers["pass_rate@character adherence (v2)"] == "Character pass rate"
    assert headers["pass_rate@queue discipline"] == "Queue discipline pass rate"
    assert headers["usable@arrival"] == "Arrival trials usable"
    # Every aspect's cost reads in one form, its head naming the count.
    costs = {headers[key] for key in grouped if key.startswith("cost_per_1k@")}
    assert costs == {f"{name} cost per 1,000 replies" for name in ("Character", "Arrival", "Queue discipline")}


def test_each_aspects_pass_rate_trials_and_cost_are_what_its_keys_resolve_to() -> None:
    held = _three_aspects()
    table = arms_table(held, block_look(held, fact_arms(held)), ())
    for aspect in (aspect.id for aspect in held.aspects):
        costs = aspect_costs_per_thousand(held, aspect)
        for arm in (arm.id for arm in held.arms):
            row = _row(table, held, arm)
            passed, usable, ran = (_key_count(held, aspect, arm, field) for field in ("passed", "usable", "ran"))
            assert row[f"pass_rate@{aspect}"].value == _pass_rate_text(passed, usable)
            assert row[f"pass_rate@{aspect}"].flag == f"{passed} of {usable}"
            assert row[f"usable@{aspect}"].value == f"{usable} of {ran}"
            said = row[f"cost_per_1k@{aspect}"].value
            assert isinstance(said, str) and said.startswith("$"), (aspect, arm, said)
            assert costs[arm] > 0
    # The added aspects' costs are their own figures, per 1,000 replies.
    assert aspect_costs_per_thousand(held, "arrival")["qwen3.8-flash"] == pytest.approx(2.6)
    assert _row(table, held, "qwen3.8-flash")["cost_per_1k@arrival"].value == "$2.60"
    # The cells' figures are ones prose can cite: their keys render.
    assert render("{aspects.arrival.scoreboard.September.passed}", held, {}) == "127"


def test_the_options_table_lays_out_the_plans_pass_rate_and_cost_per_aspect() -> None:
    held = _three_aspects()
    plan = ReportPlan.model_construct(
        options=Options(
            rows=[OptionRow(option="September", role="recommended"), OptionRow(option="1017", role="alternative")],
            columns=["pass_rate", "cost_per_1k"],
        )
    )
    table = options_table(plan, held, block_look(held, fact_arms(held))).table
    aspects = [aspect.id for aspect in held.aspects]
    keys = [column.key for column in table.columns]
    assert [key for key in keys if "@" in key] == [
        *(f"pass_rate@{aspect}" for aspect in aspects),
        *(f"cost_per_1k@{aspect}" for aspect in aspects),
    ], "each family the plan names, a column per aspect, where the plan places it"
    first = table.rows[0]
    assert first["pass_rate@arrival"].value == _pass_rate_text(127, 180)
    assert first["pass_rate@arrival"].flag == "127 of 180"
    assert first["cost_per_1k@queue discipline"].value == "$0.90"


def test_every_arm_by_measure_is_a_row_per_aspect_and_arm_with_each_aspects_own_figures() -> None:
    held = _three_aspects()
    (table,) = [
        table for table in every_table(held, block_look(held, fact_arms(held))) if table.name == "arms_by_measure"
    ]
    keys = [column.key for column in table.columns]
    assert keys[:3] == ["arm", "aspect", "pass_rate"]
    arms = [arm.id for arm in held.arms]
    names = [aspect.name for aspect in held.aspects]
    assert [row["aspect"].value for row in table.rows] == [name for name in names for _ in arms]
    by_place = {(row["aspect"].value, index % len(arms)): row for index, row in enumerate(table.rows)}
    # Each aspect's pass rate is its own scoreboard's.
    for aspect in held.aspects:
        for index, arm in enumerate(arms):
            passed, usable = (_key_count(held, aspect.id, arm, field) for field in ("passed", "usable"))
            assert by_place[aspect.name, index]["pass_rate"].value == _pass_rate_text(passed, usable)
    # A measure read on one aspect only is that aspect's, a dash on the others.
    column = f"measure:{_ARRIVAL_ONLY}"
    assert column in keys
    for (aspect, _index), row in by_place.items():
        assert (row[column].value is not None) == (aspect == "arrival"), aspect
    # The cost measure is each aspect's own figure, never the lead aspect's on every row.
    cost = f"measure:{_COST}"
    stated = {aspect: by_place[aspect, 1][cost].value for aspect in names}
    assert len(set(stated.values())) == len(names), stated


def test_the_arms_key_counts_each_aspects_usable_trials() -> None:
    held = _three_aspects()
    table = arms_key(held, block_look(held, fact_arms(held))).table
    keys = [column.key for column in table.columns]
    assert "usable" not in keys
    assert [key for key in keys if "@" in key] == [f"usable@{aspect.id}" for aspect in held.aspects]
    assert _row(table, held, "1017")["usable@queue discipline"].value == "238 of 240"


def test_an_input_whose_other_aspects_hold_no_figures_lays_its_tables_out_for_one() -> None:
    held = fixture_inputs()["r5"]
    bare = held.model_copy(update={"aspects": [*held.aspects, Aspect(id="arrival", name="arrival", readings=[])]})
    look = block_look(held, fact_arms(held))
    assert arms_table(bare, look, ()) == arms_table(held, look, ())
    assert arms_key(bare, look) == arms_key(held, look)
    assert every_table(bare, look) == every_table(held, look)
