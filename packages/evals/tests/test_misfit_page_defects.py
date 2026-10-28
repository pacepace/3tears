"""The code-side defects the prototype-4 misfit reviews found on the document's page, each pinned where it was made.

The editor and the owner's blind scoring of the W3 acceptance runs (r1, r2) named five things code printed wrong; four
are fixed and pinned here:

- **A values table's Note column repeated its Value** at another rounding (``+2`` beside ``+2``, ``$15.00`` beside
  ``$14.76``): a note that only restates the row's value, the label a chart draws at a bar's end, is not printed again
  beside it; a note that says more (``worse``, ``-7.6 pts worse``) is kept.
- *(Not here: an interval read at mixed precision, ``+2`` beside ``+0.4444 to +3.556``. A values table reads each
  figure as its key reads it, which is how a cited figure is found in its block, and the key spells a value and its
  interval's ends by the number rule one by one; one precision for the three is a figure-key rule, filed apart.)*
- **An ungrouped volume** (``the declared volume of 4000000 a month``): a count key groups its thousands.
- **One interaction printed twice** (``reasoning effect, across retry`` and ``retry effect, across reasoning``, both
  -0.5): four arms crossing two settings are one interaction, listed once.
- **"Fails (no figure)" in the gate grid** for a cell the page reads as unknown: the full gate grid says ``Unknown``
  where the input holds no figure, never a fail.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import pytest

from packages.evals.tests.analyst_support import fixture_inputs, fixture_snapshots
from packages.evals.tests.fixtures.report_plans.analyst_cli import REPLAY, snapshot_ids
from packages.evals.tests.ops_support import CALLER
from threetears.evals.analysis.analyst.levers import lever_effects
from threetears.evals.analysis.report.charts import BarRow, BarsChart, ForestChart, ForestRow
from threetears.evals.analysis.report.doc_page.markdown import doc_page_markdown
from threetears.evals.analysis.report.figure_keys import render
from threetears.evals.analysis.report.plan_page.full_results import every_table
from threetears.evals.analysis.report.block_spec import block_look
from threetears.evals.analysis.report_doc_store import doc_verdicts_and_page
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_plan_facts import fact_arms
from threetears.evals.kernel.report_doc import ReportDoc

#: The acceptance runs the reviews read.
_RUNS = Path(__file__).parent / "fixtures" / "report_docs" / "prototype4"


@cache
def _inputs() -> dict[str, ReportInput]:
    return {
        snapshot.id: ReportInput.model_validate(snapshot.report_input)
        for snapshot in fixture_snapshots(CALLER.scope_id).values()
    }


def _page_markdown(run: str) -> str:
    """The stored document of one acceptance run, laid out over the fixture snapshot it was written on."""
    replay = json.loads((_RUNS / run / REPLAY).read_text())
    record = json.loads((_RUNS / run / "record.json").read_text())
    held = _inputs()[snapshot_ids()[replay["snapshot_of"]]]
    verdicts, page = doc_verdicts_and_page(ReportDoc.model_validate(record["doc"]), held)
    assert verdicts == [] and page is not None, verdicts
    return doc_page_markdown(page)


# --- a values table's notes and precision ----------------------------------------------------------------------------


def test_a_note_restating_its_rows_value_is_not_printed_beside_it_and_one_that_says_more_is() -> None:
    restating = BarsChart(
        rows=[
            BarRow(label="Billing", value=2.0, lo=0.4444, hi=3.556, note="+2"),
            BarRow(label="Troubleshooting", value=5.0, lo=2.737, hi=7.263, note="+5"),
        ],
        domain=(-1, 8),
        fmt="+num",
        level="95%",
    )
    headers, rows = restating.table([])
    assert headers == ["Row", "Value", "95% interval"]
    assert rows == [["Billing", "+2", "+0.4444 to +3.556"], ["Troubleshooting", "+5", "+2.737 to +7.263"]]

    with_unit = BarsChart(rows=[BarRow(label="Overall", value=0.8, note="+0.8 pts")], domain=(-1, 2), fmt="+num")
    assert with_unit.table([], read_at="pts") == (["Row", "Value"], [["Overall", "+0.8"]])

    saying_more = ForestChart(
        rows=[
            ForestRow(label="A", value=-7.581, lo=-10.79, hi=-3.381, note="worse"),
            ForestRow(label="B", value=-3.010, lo=-5.815, hi=1.136, note="not separated"),
        ],
        domain=(-15, 5),
        fmt="+num",
        level="95%",
    )
    headers, rows = saying_more.table([])
    assert headers[-1] == "Note"
    assert [row[-1] for row in rows] == ["worse", "not separated"]


def test_a_money_note_restating_its_amount_is_not_printed_beside_it() -> None:
    chart = BarsChart(
        rows=[BarRow(label="Brisk-S → Atlas-M", value=4.943, lo=4.15, hi=6.109, note="$4.94")],
        domain=(0, 25),
        fmt="usd:2",
        level="95%",
    )
    assert chart.table([])[1] == [["Brisk-S → Atlas-M", "$4.94", "$4.15 to $6.11"]]


@pytest.mark.parametrize("run", ["r1-reading-only-bars", "r1-without-its-gates"])
def test_the_r1_pages_print_no_note_restating_its_value(run: str) -> None:
    page = _page_markdown(run)
    assert "| +2 | +0.4444 to +3.556 | +2 |" not in page
    assert "| +2 | +0.4444 to +3.556 |" in page
    assert "| $4.94 | $4.15 to $6.11 | $4.94 |" not in page


# --- a count key, an interaction, a gate cell ------------------------------------------------------------------------


def test_a_count_key_groups_its_thousands() -> None:
    held = fixture_inputs()["r2"]
    assert held.meta.volume_per_month == 4_000_000
    assert render("At {meta.volume_per_month} a month.", held, {}) == "At 4,000,000 a month."
    assert "the declared volume of 4,000,000 a month" in _page_markdown("r2-with-one-gate")


def test_four_arms_crossing_two_settings_are_one_interaction() -> None:
    """r2's arms cross three settings (3 models, 3 reasoning levels, 2 retries): 36 squares of four arms, each one
    interaction, never one per setting moved across the other."""
    effects = lever_effects(fixture_inputs()["r2"], "pass_rate", limit=1000)
    squares = [
        frozenset(
            part
            for figure in (row.first, row.second)
            for part in figure.key.removeprefix(f"lever.{row.setting}.").removesuffix(".pass_rate.effect").split(".")
        )
        for row in effects.interactions
    ]
    assert len(effects.interactions) == 36
    assert len(set(squares)) == len(squares)
    page = _page_markdown("r2-with-its-gates-read-in-each-task")
    assert "reasoning effect, across retry" in page and "retry effect, across reasoning" not in page


def test_the_full_gate_grid_reads_a_cell_with_no_figure_as_unknown_never_a_fail() -> None:
    page = _page_markdown("r2-with-its-gates-read-in-each-task")
    assert "Fails (no figure)" not in page
    assert "Unknown (no figure)" in page
    held = fixture_inputs()["r2 with its gates read in each task"]
    (grid,) = [
        table for table in every_table(held, block_look(held, fact_arms(held))) if table.name == "gate_matrix_full"
    ]
    unknown = [cell for row in grid.rows for cell in row.values() if cell.value == "Unknown"]
    assert unknown and all(cell.status == "neutral" and cell.flag == "no figure" for cell in unknown)
