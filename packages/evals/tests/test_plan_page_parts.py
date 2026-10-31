"""The planned page's own tables, held to what the input and the gate check read: the options table's column
families, the Full results tables and the explore part's flip table.

Held over every r1–r5 anchor plan and the Kairo reference plan, each against its own input. The expected side is read
off the input itself (a figure's raw value, its scoreboard counts, the readings' intervals), the figure keys spelled
here (:func:`~threetears.evals.analysis.report.figure_keys.resolve`), the gate check and the margin sensitivity, never
typed in.

- **Options columns.** Every family in the plan checks' vocabulary
  (:data:`~threetears.evals.analysis.report_plan_checks.OPTION_COLUMN_FAMILIES`) has a renderer, so a column the
  checks accept is a column the page fills. Each family's cell states what its input reads: a measure's figure (its
  number the raw figure, on its scale), the pass rate's passes over usable trials, the gate check's first reason, the
  cost's uplift on the control, a reading's direction and margin verdict in words. A cell is a dash only where the
  figure key this test spells for it does not resolve, so a family reading a misspelled key cannot pass for one the
  input does not hold.
- **A pill says its status in words**, on every table of every page.
- **Full results.** Each table the plan lists is drawn, in its order (the strata table followed by a tasks table where
  the input holds tasks): every arm by measure; every corrected comparison, its change and interval the input's on
  the measure's scale; every slice's figures; every gate at every place it is read, an arm failing (or borderline on)
  a row exactly where the gate check fails it (or holds it borderline) somewhere; every configuration; every cell.
- **Explore.** A plan whose control moves the margin draws a flip table of each reading the margin sensitivity reads,
  its flip point, its margin verdict and whether the call holds; a plan with no such control, or an input with no
  margin, draws none; the plan's ``recomputes`` is not printed.
"""

from __future__ import annotations

import functools
import math
import re
from pathlib import Path
from typing import Any

import pytest

from packages.evals.tests.fixtures.report_plans import dataset
from packages.evals.tests.report_kit_support import KIT
from threetears.evals.analysis import PlannedPage, TableBlock, planned_page_html, planned_page_markdown
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.figure_keys import format_figure, render, resolve
from threetears.evals.analysis.report.plan_page.build import plan_page
from threetears.evals.analysis.report.plan_page.words import EXPLORE_HOLDS
from threetears.evals.analysis.report_decision import margin_sensitivity
from threetears.evals.analysis.report_gates import gate_check
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_plan_checks import OPTION_COLUMN_FAMILIES
from threetears.evals.analysis.report_rows import query_aspect
from threetears.evals.kernel.report_plan import ReportPlan

#: The Kairo reference plan.
REFERENCE_PLAN = KIT / "references" / "r5-kairo-round2.plan.json"

#: The kinds whose resolvers are other chunks' and may not be in this tree: a page here leaves their charts out, which
#: none of these tables reads.
_OTHER_KINDS = frozenset(
    {
        "gate_matrix",
        "gate_scatter",
        "lattice",
        "dot_interval",
        "range_bars",
        "scatter_frontier",
        "heatmap",
        "diff_heatmap",
        "confusion",
        "table",
    }
)

_PLANS = [
    *((name, dataset.HERE / name / "anchor.plan.json") for name in ("r1", "r2", "r3", "r4", "r5")),
    ("r5", REFERENCE_PLAN),
]
_IDS = ["r1", "r2", "r3", "r4", "r5", "r5-reference"]

#: A number as a cell prints it: its sign, digits and decimal point, its thousands unseparated.
_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")


@functools.cache
def _input(name: str) -> ReportInput:
    return dataset.report_input_of(name)


@functools.cache
def _plan(path: Path) -> ReportPlan:
    plan = dataset.load_plan(path)
    return plan.model_copy(
        update={
            "summary": [chart for chart in plan.summary if chart.kind not in _OTHER_KINDS],
            "sections": [section for section in plan.sections if section.chart.kind not in _OTHER_KINDS],
        }
    )


@functools.cache
def _page(name: str, path: Path) -> PlannedPage:
    return plan_page(_plan(path), _input(name))


def _number(text: Any) -> float:
    found = _NUMBER.search(str(text).replace(",", ""))
    assert found is not None, f"no number in {text!r}"
    return float(found.group(0))


def _resolves(key: str, report_input: ReportInput) -> bool:
    try:
        resolve(key, report_input)
    except UnresolvableReference:
        return False
    return True


# --- the options table's column families ----------------------------------------------------------------------------


def _column_of(family: str, report_input: ReportInput) -> str:
    """A column of the family over the input: a keyed family names the input's first key of its kind."""
    if family == "judged:<dim>":
        return f"judged:{report_input.primary_judged or next(iter(report_input.judged))}"
    if family == "measure:<key>":
        return f"measure:{next(iter(report_input.measures))}"
    return family


@pytest.mark.parametrize("family", OPTION_COLUMN_FAMILIES)
def test_every_column_family_the_plan_checks_accept_has_a_renderer(family: str) -> None:
    plan, report_input = _plan(REFERENCE_PLAN), _input("r5")
    column = _column_of(family, report_input)
    asked = plan.model_copy(update={"options": plan.options.model_copy(update={"columns": [column]})})
    table = plan_page(asked, report_input).options.table
    arms = {arm.id: arm for arm in report_input.arms}
    described = any(
        (arm := arms.get(row.option)) is not None and (arm.settings or arm.served_model) for row in asked.options.rows
    )
    assert [one.key for one in table.columns] == ["option", "role", *(["runs"] if described else []), column]
    assert any(row[column].value is not None for row in table.rows), f"{family} filled no cell over r5"


def test_a_measure_column_states_each_arm_s_figure_on_its_scale() -> None:
    report_input, path = _input("r4"), dataset.HERE / "r4" / "anchor.plan.json"
    plan, page = _plan(path), _page("r4", path)
    columns = [column for column in plan.options.columns if column.startswith("measure:")]
    assert columns, "r4's plan must hold a measure column for this to test it"
    checked = 0
    for option, row in zip(plan.options.rows, page.options.table.rows, strict=True):
        for column in columns:
            key = column.removeprefix("measure:")
            measure = report_input.measures[key]
            stat = measure.per_arm.get(option.option)
            if stat is None or stat.mean is None:
                assert row[column].value is None
                continue
            scale = 100.0 if measure.unit == "share" else 1.0
            assert math.isclose(_number(row[column].value), float(stat.mean) * scale, abs_tol=0.051)
            assert row[column].value == format_figure(resolve(f"measure.{key}.{option.option}.mean", report_input))
            checked += 1
    assert checked, "no measure cell was checked"


def _canonical_keys(column: str, arm: str, report_input: ReportInput) -> list[str]:
    """The figure keys a column's cell reads for an arm, spelled here: a dash is right only where none resolves."""
    if column == "pass_rate":
        return [f"pass_rate.{arm}"]
    if column == "reading.change":
        return [f"reading.{arm}.change"]
    if column == "cost_per_1k":
        return [f"cost_per_1k.{arm}"]
    if column.startswith("judged:"):
        return [f"judged.{column.removeprefix('judged:')}.{arm}"]
    if column.startswith("measure:"):
        key = column.removeprefix("measure:")
        return [f"measure.{key}.{arm}.{statistic}" for statistic in ("mean", "p50", "p95")]
    return []


@pytest.mark.parametrize(("name", "path"), _PLANS, ids=_IDS)
def test_each_options_cell_says_what_its_input_reads_and_is_a_dash_only_where_it_holds_nothing(
    name: str, path: Path
) -> None:
    report_input, plan, page = _input(name), _plan(path), _page(name, path)
    aspect = query_aspect("", report_input)
    control = next((arm.id for arm in report_input.arms if arm.role == "control"), None)
    for option, row in zip(plan.options.rows, page.options.table.rows, strict=True):
        arm = option.option
        for column in plan.options.columns:
            cell = row[column]
            keys = _canonical_keys(column, arm, report_input)
            if cell.value is None:
                held = [key for key in keys if _resolves(key, report_input)]
                assert not held, f"{column} on {arm} is a dash while {held} reads a figure"
            if column == "pass_rate" and cell.value is not None:
                board = report_input.scoreboard.get(arm)
                if board is None or board.passed is None or not board.usable:
                    assert cell.flag is None, "no counts are stated where the scoreboard holds none"
                    continue
                assert cell.flag == f"{board.passed} of {board.usable}"
                assert math.isclose(_number(cell.value), 100 * board.passed / board.usable, abs_tol=0.051)
            elif column == "gate":
                checked = gate_check(report_input, arm)
                reasons = (*checked.fails, *checked.borders)
                assert cell.flag == (reasons[0] if reasons else None)
            elif column == "cost_per_1k" and cell.value is not None and arm != control and control is not None:
                uplift = resolve(f"cost.uplift.{arm}", report_input)
                assert cell.flag is not None and cell.flag.endswith(format_figure(uplift).lstrip("+-"))
            elif column == "reading.verdict" and arm != control:
                reading = next((one for one in aspect.readings if one.arm == arm), None) if aspect else None
                if reading is None or reading.within_interval is None:
                    assert cell.value is None
                    continue
                low, high = (float(end) for end in reading.within_interval)
                lead = "Worse" if high < 0 else "Better" if low > 0 else "Not separated"
                assert str(cell.value).startswith(lead)
                if aspect is not None and aspect.margin is not None:
                    assert str(cell.value).endswith(
                        "within margin" if -low < float(aspect.margin) else "outside margin"
                    )


def _tables(page: PlannedPage) -> list[TableBlock]:
    tables = [page.options.table, *page.full_results]
    if page.arms_key is not None:
        tables.append(page.arms_key.table)
    if page.explore is not None:
        tables.append(page.explore.flips)
    return tables


@pytest.mark.parametrize(("name", "path"), _PLANS, ids=_IDS)
def test_every_pill_says_its_status_in_words(name: str, path: Path) -> None:
    pills = [cell for table in _tables(_page(name, path)) for row in table.rows for cell in row.values() if cell.status]
    assert pills, "a page with no pill holds this to nothing"
    for cell in pills:
        assert cell.value is not None and re.search(r"[A-Za-z]{3}", cell.text), (
            f"a {cell.status} pill reads {cell.text!r}"
        )


# --- full results ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "path"), _PLANS, ids=_IDS)
def test_full_results_draws_each_table_the_plan_lists_in_its_order(name: str, path: Path) -> None:
    report_input, plan, page = _input(name), _plan(path), _page(name, path)
    expected: list[str] = []
    for listed in plan.full_results:
        if listed == "strata":
            expected += [*(["strata"] if report_input.strata else []), *(["tasks"] if report_input.tasks else [])]
        else:
            expected.append(listed)
    assert [table.name for table in page.full_results] == expected
    html = planned_page_html(page)
    assert html.index('data-part="full_results"') < html.index('class="method"')
    markdown = planned_page_markdown(page)
    assert all(f"**{table.title}**" in markdown for table in page.full_results)


def _full(page: PlannedPage, table: str) -> TableBlock:
    return next(one for one in page.full_results if one.name == table)


@pytest.mark.parametrize(("name", "path"), _PLANS, ids=_IDS)
def test_arms_by_measure_is_every_arm_and_each_figure_it_holds(name: str, path: Path) -> None:
    report_input, page = _input(name), _page(name, path)
    table = _full(page, "arms_by_measure")
    names = {arm.id: (arm.short_name or arm.name or arm.id) for arm in report_input.arms}
    assert [row["arm"].value for row in table.rows] == [names[arm.id] for arm in report_input.arms]
    for arm, row in zip(report_input.arms, table.rows, strict=True):
        for key, judged in report_input.judged.items():
            stat = judged.per_arm.get(arm.id)
            if f"judged:{key}" in row and stat is not None and stat.mean is not None:
                scale = 100.0 if judged.scale == "pass_fail" and float(stat.mean) <= 1 else 1.0
                assert math.isclose(_number(row[f"judged:{key}"].value), float(stat.mean) * scale, rel_tol=0.01)


@pytest.mark.parametrize(("name", "path"), _PLANS, ids=_IDS)
def test_arms_by_measure_states_each_figure_in_one_column(name: str, path: Path) -> None:
    headers = [column.header for column in _full(_page(name, path), "arms_by_measure").columns]
    assert len(headers) == len(set(headers)), headers


def test_r2_arms_by_measure_reads_its_pass_rate_off_its_measure_once() -> None:
    """r2's arms count no passes and its input holds a ``pass_rate`` measure: the measure's column is the one."""
    (path,) = [path for name, path in _PLANS if name == "r2"]
    keys = [column.key for column in _full(_page("r2", path), "arms_by_measure").columns]
    assert "pass_rate" not in keys and keys.count("measure:pass_rate") == 1, keys


@pytest.mark.parametrize(("name", "path"), [(n, p) for n, p in _PLANS if "contrasts" in _plan(p).full_results])
def test_every_corrected_comparison_is_a_row_its_change_the_input_s_on_its_scale(name: str, path: Path) -> None:
    report_input, page = _input(name), _page(name, path)
    table = _full(page, "contrasts")
    assert len(table.rows) == len(report_input.contrasts)
    for contrast, row in zip(report_input.contrasts, table.rows, strict=True):
        if contrast.delta is None:
            continue
        drawn = _number(row["change"].value)
        # The scale is the drawn one: a rate in points, money per reply per 1,000 replies, any other figure as it is.
        assert any(
            math.isclose(drawn, float(contrast.delta) * factor, rel_tol=0.01, abs_tol=0.051)
            for factor in (1.0, 100.0, 1000.0)
        ), f"{contrast.arm} on {contrast.measure}: {row['change'].value} is not {contrast.delta} on any scale"
        assert (row["verdict"].status == "bad") == (contrast.verdict == "regressed")


@pytest.mark.parametrize(("name", "path"), [(n, p) for n, p in _PLANS if "gate_matrix_full" in _plan(p).full_results])
def test_the_gate_matrix_fails_an_arm_on_a_row_exactly_where_the_gate_check_fails_it(name: str, path: Path) -> None:
    report_input, page = _input(name), _page(name, path)
    table = _full(page, "gate_matrix_full")
    assert table.rows, "every listed gate matrix has a row"
    worst = {"good": 0, "warn": 1, "bad": 2}
    for arm in report_input.arms:
        statuses = [row[f"arm:{arm.id}"].status for row in table.rows]
        drawn = max(worst[status or "good"] for status in statuses)
        assert drawn == {"pass": 0, "borderline": 1, "fail": 2}[gate_check(report_input, arm.id).status], arm.id
        figures = [row[f"arm:{arm.id}"].flag for row in table.rows]
        assert all(figures), "every cell states its figure, or that there is none"


@pytest.mark.parametrize(("name", "path"), _PLANS, ids=_IDS)
def test_each_slice_table_is_a_row_per_slice_and_measure_it_holds(name: str, path: Path) -> None:
    report_input, page = _input(name), _page(name, path)
    held = {
        "strata": report_input.strata,
        "tasks": report_input.tasks,
        "cells": list(report_input.cells.values()),
    }
    for table in page.full_results:
        if table.name not in held:
            continue
        slices = held[table.name]
        expected = sum(len({key for figures in one.per_arm.values() for key in figures}) for one in slices)
        assert len(table.rows) == expected
        cases = [one.n_cases for one in slices for _ in {key for figures in one.per_arm.values() for key in figures}]
        assert [row["cases"].value for row in table.rows] == cases


@pytest.mark.parametrize(("name", "path"), [(n, p) for n, p in _PLANS if "configs" in _plan(p).full_results])
def test_every_configuration_is_its_arm_s_level_of_each_factor(name: str, path: Path) -> None:
    report_input, page = _input(name), _page(name, path)
    table = _full(page, "configs")
    assert len(table.rows) == len(report_input.configs)
    for config, row in zip(report_input.configs, table.rows, strict=True):
        for factor in report_input.factors:
            level = next(one for one in factor.levels if one.id == config.levels[factor.id])
            assert row[f"factor:{factor.id}"].value == (level.name or level.id)


# --- explore ---------------------------------------------------------------------------------------------------------


def test_a_control_moving_the_margin_draws_each_reading_s_flip_and_whether_the_call_holds() -> None:
    report_input, plan, page = _input("r5"), _plan(REFERENCE_PLAN), _page("r5", REFERENCE_PLAN)
    assert any(control.binds == "margin" for control in plan.explore[0].controls)
    sensitivity = margin_sensitivity(report_input)
    assert page.explore is not None and sensitivity.candidates
    assert page.explore.title == render(plan.explore[0].title, report_input, {})
    rows = page.explore.flips.rows
    assert len(rows) == len(sensitivity.candidates)
    for candidate, row in zip(sensitivity.candidates, rows, strict=True):
        assert candidate.flip_margin is not None
        assert math.isclose(_number(row["flip"].value), candidate.flip_margin, abs_tol=0.051)
        assert row["declared"].status == ("good" if candidate.within else "bad")
    assert page.explore.holds == EXPLORE_HOLDS[sensitivity.holds], "the line says whether the call holds"
    html = planned_page_html(page)
    assert 'data-part="explore"' in html
    assert render(plan.explore[0].recomputes, report_input, {}) not in html, "recomputes is not printed"


def test_where_the_margin_moves_the_call_the_explore_line_says_it_does_not_hold() -> None:
    """r5 with qwen3.8-flash the cheapest arm: shown no worse at the declared margin it dominates, and below its flip
    margin it no longer is, so the call does not hold at every margin, and the line says so."""
    data = dataset.report_input_of("r5").model_dump(mode="json", by_alias=True, exclude_none=True)
    data["measures"]["production_replicating_cost"]["per_arm"]["qwen3.8-flash"]["mean"] = 0.0015
    report_input = ReportInput.model_validate(data)
    sensitivity = margin_sensitivity(report_input)
    assert sensitivity.holds is False and margin_sensitivity(_input("r5")).holds is True

    page = plan_page(_plan(REFERENCE_PLAN), report_input)

    assert page.explore is not None and page.explore.holds == EXPLORE_HOLDS[False] != EXPLORE_HOLDS[True]


def test_a_plan_whose_controls_move_anything_but_the_margin_draws_no_explore_part() -> None:
    path = dataset.HERE / "r1" / "anchor.plan.json"
    plan = _plan(path)
    assert plan.explore and not any(control.binds == "margin" for control in plan.explore[0].controls)
    assert _page("r1", path).explore is None
    reference = _plan(REFERENCE_PLAN)
    controls = [control.model_copy(update={"binds": "p95"}) for control in reference.explore[0].controls]
    rebound = reference.model_copy(update={"explore": [reference.explore[0].model_copy(update={"controls": controls})]})
    assert plan_page(rebound, _input("r5")).explore is None


def test_an_input_no_aspect_of_which_declares_a_margin_draws_no_explore_part() -> None:
    path = dataset.HERE / "r4" / "anchor.plan.json"
    plan, report_input = _plan(path), _input("r4")
    assert plan.explore and not margin_sensitivity(report_input).candidates
    controls = [control.model_copy(update={"binds": "margin"}) for control in plan.explore[0].controls]
    rebound = plan.model_copy(update={"explore": [plan.explore[0].model_copy(update={"controls": controls})]})
    assert plan_page(rebound, report_input).explore is None
