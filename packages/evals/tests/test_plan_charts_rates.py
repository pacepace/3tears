"""The rate, cell and table charts a planned page draws: dot strips, range bars, the frontier scatter, heatmaps,
difference heatmaps, confusion matrices and the plain table.

Held over every r1–r5 anchor and current run, and the Kairo reference plan, each against its own input; kinds no
stored plan draws are held on charts of the same inputs. The expected side is always read from the input (the
rows a query selects, :func:`~threetears.evals.analysis.report_rows.query_rows`, the criteria, the target matrix,
the gate check), never typed in.

- **The pin.** Every (kind, source) pair the plan checks accept (read through ``validate_plan``, which refuses a kind
  over a source it cannot draw) resolves on some r1–r5 input: a chart of that kind over that source draws.
- **Each kind** draws what its query selects, every value the input's, validated as its kit chart model; small
  multiples a panel per arm, per axis or per slice; every line the stored one, else its criterion's; a dot strip's
  intervals its figures' own, else, where its stored level says ``computed``, the binomial one their ``spread`` key
  reads, named as code's.
- **Checks and renderer agree** on a chart drawn as each arm less the control: both refuse it on an input with no
  control, and both take it on one with a control, or a difference heatmap naming its two arms.
- **Markup** in input names is text on the page.
"""

from __future__ import annotations

import functools
import itertools
import math
from pathlib import Path
from typing import Any, Literal, get_args

import pytest

from packages.evals.tests.fixtures.report_plans import dataset
from packages.evals.tests.hostile_page_support import HOSTILE_LINE
from packages.evals.tests.page_contract_support import what_runs
from packages.evals.tests.report_kit_support import KIT
from packages.evals.tests.static_page_support import assert_static
from threetears.evals.analysis import ChartBlock, TableBlock, planned_page_html, planned_page_markdown
from threetears.evals.analysis.errors import PlanRefusal, UnresolvableReference
from threetears.evals.kernel.report_plan import plan_recommended_arm
from threetears.evals.analysis.report.charts import (
    KIT_CHARTS,
    ChartReference,
    ConfusionChart,
    DiffHeatmapChart,
    DotIntervalChart,
    HeatmapChart,
    RangeBarsChart,
    ScatterChart,
)
from threetears.evals.analysis.report.plan_page.build import plan_page
from threetears.evals.analysis.report.figure_keys import resolve
from threetears.evals.analysis.report.plan_page.charting import COMPUTED, figure_unit, misses_line
from threetears.evals.analysis.report.plan_page.look import PageLook, page_look, quantity
from threetears.evals.analysis.report.plan_page.model import PageChart
from threetears.evals.analysis.report.plan_page.resolve import resolve_chart
from threetears.evals.analysis.report.plan_page.words import COMPUTED_NOTE
from threetears.evals.analysis.report_decision import control_arm, surviving_options
from threetears.evals.analysis.report_gates import gate_check
from threetears.evals.analysis.report_input import Cell, ReportInput, ScoreboardRow, Stat, Stratum, Task
from threetears.evals.analysis.report_plan_checks import validate_plan
from threetears.evals.analysis.report_plan_facts import plan_facts
from threetears.evals.analysis.report.block_spec import block_look
from threetears.evals.analysis.report_plan_facts import fact_arms
from threetears.evals.analysis.analyst.snapshot import frozen_input
from packages.evals.tests.analyst_support import SHAPE_NAMES, shape_snapshots
from threetears.evals.analysis.report_rows import query_rows
from threetears.evals.kernel.report_plan import ChartQuery, ChartSpec, ReportPlan

INPUTS = ("r1", "r2", "r3", "r4", "r5")
REFERENCE_PLAN = KIT / "references" / "r5-kairo-round2.plan.json"

#: This module's kinds.
_MINE = frozenset({"dot_interval", "range_bars", "scatter_frontier", "heatmap", "diff_heatmap", "confusion", "table"})


@functools.cache
def _input(name: str) -> ReportInput:
    return dataset.report_input_of(name)


@functools.cache
def _plan(path: Path) -> ReportPlan:
    return dataset.load_plan(path)


def _anchor(name: str) -> ReportPlan:
    return _plan(dataset.HERE / name / "anchor.plan.json")


def _query(source: str, **fields: Any) -> ChartQuery:
    base: dict[str, Any] = {
        "source": source,
        "aspect": "",
        "measures": [],
        "arms": [],
        "strata": [],
        "limit": 0,
        "transform": "none",
        "small_multiples_by": "none",
    }
    return ChartQuery.model_validate(base | fields)


def _spec(kind: str, query: ChartQuery, **fields: Any) -> ChartSpec:
    base: dict[str, Any] = {
        "kind": kind,
        "title": "",
        "query": query.model_dump(),
        "reference": [],
        "emphasis": "",
        "level": "",
        "unit": "",
        "domain": [],
    }
    return ChartSpec.model_validate(base | fields)


def _with_roles(name: str, control: str | None) -> ReportInput:
    """Input ``name`` with ``control`` its control and every other arm a candidate (no control for None)."""
    data = _input(name).model_dump(by_alias=True, exclude_none=True)
    for arm in data["arms"]:
        arm["role"] = "control" if arm["id"] == control else "challenger"
    return ReportInput.model_validate(data)


@functools.cache
def _shape(name: str) -> ReportInput:
    """A shape campaign's input as its snapshot froze it: the pin searches them for data no reference input holds
    (an arm against an absolute bar with its value)."""
    return frozen_input(shape_snapshots("s")[name])


@functools.cache
def _variant(key: str) -> ReportInput:
    """An input by key: ``rN`` as it is, ``rN+control`` with its first arm made the control, ``rN-control`` with its
    control made a candidate, a shape campaign (``s1`` to ``s3``) as it is."""
    if key in SHAPE_NAMES:
        return _shape(key)
    name, _, change = key.partition("+") if "+" in key else key.partition("-")
    if not change:
        return _input(name)
    return _with_roles(name, _input(name).arms[0].id if "+" in key else None)


#: The inputs the pin searches: each fixture input, and each with a control where it has none.
_PIN_INPUTS = [
    *INPUTS,
    *(f"{name}+control" for name in INPUTS if control_arm(_input(name)) is None),
    *SHAPE_NAMES,
]


@functools.cache
def _look(name: str) -> PageLook:
    """The look the page builder gives the input's anchor plan's page: its arms, order, survivors and names; a shape
    campaign, which has no anchor plan, the look one block is drawn with."""
    if name in SHAPE_NAMES:
        return block_look(_variant(name), fact_arms(_variant(name)))
    plan, report_input = _anchor(name.partition("+")[0].partition("-")[0]), _variant(name)
    facts = plan_facts(report_input)
    return page_look(
        facts.arms,
        {arm.id: arm.color_slot for arm in plan.arms},
        order=[row.option for row in plan.options.rows],
        survivors=surviving_options(report_input),
        recommended=plan_recommended_arm(plan),
        names=facts.names,
    )


def _draw(name: str, spec: ChartSpec) -> PageChart:
    return resolve_chart(spec, _variant(name), _look(name), title="chart")


def _one_section(plan: ReportPlan, spec: ChartSpec) -> ReportPlan:
    section = plan.sections[0].model_copy(update={"chart": spec})
    return plan.model_copy(update={"summary": [], "sections": [section]})


def _data(drawn: PageChart) -> Any:
    block = drawn.chart
    assert isinstance(block, ChartBlock) and block.data is not None
    kind, data = block.kit_chart() or ("", {})
    model = KIT_CHARTS[kind]
    assert model.model_validate(data).kit_data() == data, "the drawn data validates as its kit chart model"
    assert block.values_table()[1], "the chart has values to read"
    return block.data


# --- the pin -------------------------------------------------------------------------------------------------------


_KINDS: tuple[str, ...] = get_args(ChartSpec.model_fields["kind"].annotation)
_SOURCES: tuple[str, ...] = get_args(ChartQuery.model_fields["source"].annotation)


def _accepted_pairs() -> list[tuple[str, str]]:
    """Every (kind, source) pair the plan checks accept: those for which ``validate_plan`` names no kind fault."""
    plan, report_input = _anchor("r5"), _input("r5")
    index = len(plan.sections)
    accepted = []
    for kind, source in itertools.product(_KINDS, _SOURCES):
        section = plan.sections[-1].model_copy(update={"id": "pin", "chart": _spec(kind, _query(source))})
        try:
            validate_plan(plan.model_copy(update={"sections": [*plan.sections, section]}), report_input)
        except PlanRefusal as refused:
            if any(path == f"sections[{index}].chart.kind" and "cannot draw" in why for path, why in refused.faults):
                continue
        accepted.append((kind, source))
    return accepted


_PAIRS = _accepted_pairs()


def _vocabulary(name: str, source: str) -> list[str]:
    """Every key a query over ``source`` on input ``name`` could name: what its rows read, and what slices, cells and
    scoreboards hold."""
    report_input = _variant(name)
    keys: dict[str, None] = {}
    rows = query_rows(_query(source), report_input)
    if not rows:
        # A source an analysis computes reads the arms and the one measure a query names: any measure may be named.
        keys |= dict.fromkeys([*report_input.measures, *report_input.judged])
    for row in rows:
        if row.key:
            keys[row.key] = None
        if isinstance(row.item, Stratum | Task | Cell):
            keys |= dict.fromkeys(key for figures in row.item.per_arm.values() for key in figures)
        if isinstance(row.item, ScoreboardRow):
            keys |= dict.fromkeys(field for field, value in row.item if value not in (None, {}))
    return list(keys)


def _candidates(name: str, kind: str, source: str) -> list[ChartSpec]:
    keys = _vocabulary(name, source)
    sets: list[list[str]] = [[], *([key] for key in keys)]
    if kind == "scatter_frontier":
        sets += [list(pair) for pair in itertools.combinations(keys, 2)]
    arms = [arm.id for arm in _variant(name).arms]
    return [
        _spec(kind, _query(source, measures=measures, arms=named), level=level)
        for measures in sets
        for named in ([], arms[:2])
        for level in ("", "95%")
    ]


def _drew(kind: str, drawn: PageChart) -> bool:
    block = drawn.chart
    if kind == "table":
        return isinstance(block, TableBlock) and bool(block.rows)
    return isinstance(block, ChartBlock) and block.chart == kind and bool(block.values_table()[1])


def _resolves_somewhere(kind: str, source: str) -> bool:
    for name in _PIN_INPUTS:
        held = query_rows(_query(source), _variant(name))
        if not held and not any(query_rows(spec.query, _variant(name)) for spec in _candidates(name, kind, source)):
            continue
        for spec in _candidates(name, kind, source):
            try:
                if _drew(kind, _draw(name, spec)):
                    return True
            except UnresolvableReference:
                continue
    return False


@pytest.mark.parametrize(("kind", "source"), _PAIRS, ids=[f"{kind}-{source}" for kind, source in _PAIRS])
def test_every_kind_source_pair_has_a_resolver(kind: str, source: str) -> None:
    assert _resolves_somewhere(kind, source), f"no r1–r5 input draws a {kind} over {source}"


def test_the_pin_reads_every_kind_and_the_table_draws_every_source() -> None:
    kinds = {kind for kind, _ in _PAIRS}
    assert kinds == set(_KINDS), "every kind draws some source"
    assert {source for kind, source in _PAIRS if kind == "table"} == set(_SOURCES)
    assert ("forest", "cells") not in _PAIRS, "a pair the checks refuse is not pinned"


# --- each stored chart of this module's kinds ----------------------------------------------------------------------


def _stored() -> list[Any]:
    found = []
    for name in INPUTS:
        paths = [("anchor", dataset.HERE / name / "anchor.plan.json"), ("run", dataset.current_run(name))]
        if name == "r5":
            paths.append(("reference", REFERENCE_PLAN))
        for label, path in paths:
            plan = _plan(path)
            for spec in [*plan.summary, *(section.chart for section in plan.sections)]:
                if spec.kind in _MINE:
                    found.append(pytest.param(name, path, spec, id=f"{name}-{label}-{spec.kind}-{spec.query.source}"))
    return found


_STORED = _stored()


def test_the_stored_plans_draw_each_of_this_modules_kinds_they_use() -> None:
    assert {spec.kind for _, path, spec in (param.values for param in _STORED)} >= {
        "dot_interval",
        "heatmap",
        "confusion",
    }


@pytest.mark.parametrize(("name", "path", "spec"), _STORED)
def test_every_stored_chart_of_this_modules_kinds_resolves_on_its_page(name: str, path: Path, spec: ChartSpec) -> None:
    (section,) = plan_page(_one_section(_plan(path), spec), _input(name)).sections
    data = _data(section.chart)
    assert section.chart.chart.chart == spec.kind  # type: ignore[union-attr]
    assert isinstance(data, KIT_CHARTS[spec.kind])


def _criterion(report_input: ReportInput, key: str, scope: str, task: str | None = None) -> float | None:
    found = {
        float(criterion.threshold)
        for criterion in report_input.criteria
        if criterion.measure == key
        and criterion.statistic == "mean"
        and (
            criterion.scope == scope and criterion.task is None
            if task is None
            else criterion.task == task or (criterion.scope == "each_task" and criterion.task is None)
        )
    }
    return found.pop() if len(found) == 1 else None


def _stored_of(kind: str) -> list[Any]:
    found = [param for param in _STORED if param.values[2].kind == kind]
    assert found, f"no stored plan draws a {kind}: the test would hold nothing"
    return found


@pytest.mark.parametrize(("name", "path", "spec"), _stored_of("heatmap"))
def test_a_heatmap_of_cells_draws_the_target_matrix_its_counts_and_the_cell_floor(
    name: str, path: Path, spec: ChartSpec
) -> None:
    report_input = _input(name)
    matrix = report_input.target_matrix
    assert matrix is not None
    data = _data(_draw(name, spec))
    assert isinstance(data, HeatmapChart)
    (key,) = spec.query.measures
    cells = {(cell.row, cell.column): cell for cell in report_input.cells.values()}
    arms = list(spec.query.arms)
    panels = data.panels if data.panels is not None else [data]
    assert len(panels) == len(arms)
    floor = _criterion(report_input, key, "each_cell")
    for arm, panel in zip(arms, panels, strict=True):
        if data.panels is not None:
            assert panel.title == _look(name).arm(arm).name  # type: ignore[union-attr]
        assert [row.label for row in panel.rows or []] == [level.name or level.id for level in matrix.rows.levels]
        columns = data.columns if panel.columns is None else panel.columns
        assert [column.label for column in columns or []] == [level.name or level.id for level in matrix.columns.levels]
        for row_level, row in zip(matrix.rows.levels, panel.rows or [], strict=True):
            for column_level, drawn in zip(matrix.columns.levels, row.cells, strict=True):
                cell = cells[row_level.id, column_level.id]
                assert drawn.value == cell.per_arm[arm][key].mean
                assert drawn.n == cell.n_cases
    line = data.reference or panels[0].reference
    assert (line.value if line is not None else None) == floor, "the floor is the criterion read in every cell"


@pytest.mark.parametrize(("name", "path", "spec"), _stored_of("confusion"))
def test_a_confusion_matrix_is_the_axis_table_of_each_arm_labelled_by_level_names(
    name: str, path: Path, spec: ChartSpec
) -> None:
    report_input = _input(name)
    data = _data(_draw(name, spec))
    assert isinstance(data, ConfusionChart)
    tables = [(row.key, row.item) for row in query_rows(spec.query, report_input)]
    arms = list(spec.query.arms)
    expected = [(axis, arm) for axis, _ in tables for arm in arms]
    drawn = data.panels if data.panels is not None else [data]
    assert len(drawn) == len(expected)
    names = {level.id: level.name for factor in report_input.factors for level in factor.levels}
    if report_input.target_matrix is not None:
        for factor in (report_input.target_matrix.rows, report_input.target_matrix.columns):
            names |= {level.id: level.name for level in factor.levels}
    for (axis, arm), matrix in zip(expected, drawn, strict=True):
        table = report_input.confusion[axis]
        assert matrix.matrix == [[float(value) for value in row] for row in table.per_arm[arm]]
        assert matrix.labels == [names.get(label) or label for label in table.labels]
        assert matrix.normalized is (table.unit == "row_percent")


@pytest.mark.parametrize(("name", "path", "spec"), _stored_of("dot_interval"))
def test_a_dot_strip_marks_each_figure_against_its_criterions_line_hollow_where_it_misses(
    name: str, path: Path, spec: ChartSpec
) -> None:
    """A strip per task the query selects (one strip drawn alone, several as panels titled by task), or one strip of
    the arms' overall figures; each figure's interval is its own, else, where the stored level says ``computed``, the
    binomial interval its ``spread`` figure key reads, and none where neither holds."""
    report_input = _input(name)
    drawn = _draw(name, spec)
    data = _data(drawn)
    assert isinstance(data, DotIntervalChart)
    (key,) = spec.query.measures
    strips: dict[str, tuple[dict[str, Stat], float | None]] = {}
    if spec.query.source == "tasks":
        assert spec.query.small_multiples_by != "arm", "a strip per arm is not read here: hold it before storing one"
        for row in query_rows(spec.query, report_input):
            task = row.item
            assert isinstance(task, Task)
            stats = {arm: task.per_arm[arm][key] for arm in spec.query.arms if key in task.per_arm.get(arm, {})}
            if stats:
                strips[task.name or _look(name).reader("task", task.id)] = (
                    stats,
                    _criterion(report_input, key, "each_task", task.id),
                )
    else:
        table = report_input.measures if spec.query.source == "measures" else report_input.judged
        stats = {arm: stat for arm, stat in table[key].per_arm.items() if arm in spec.query.arms}
        strips[""] = (stats, _criterion(report_input, key, "overall"))
    if data.panels is None:
        assert len(strips) == 1, "one strip is drawn alone"
        drawn_strips = [(next(iter(strips)), data.rows or [], data.reference)]
    else:
        drawn_strips = [(panel.title or "", panel.rows or [], panel.reference) for panel in data.panels]
        assert [title for title, *_ in drawn_strips] == list(strips), "a panel per task, titled by it, in query order"
    declared = {table[key].level for table in (report_input.measures, report_input.judged) if key in table}
    level = spec.level or next(iter(declared)) or ""
    assert data.level == level, "the stored level, else the level its figures declare"
    stored = _stored_line(report_input, spec)
    # A rate's interval is drawn within 0 to 100, where a rate lies.
    rate = quantity(figure_unit(report_input, key)).rate
    computed_here = level.endswith(COMPUTED) and spec.query.source in ("measures", "judged")
    computed_rows = 0
    for title, rows, reference in drawn_strips:
        stats, line = strips[title]
        assert {row.id for row in rows} == set(stats)
        expected_line = stored if spec.reference else line
        assert (reference.value if reference is not None else None) == expected_line, (
            "the stored line, else its criterion's"
        )
        for row in rows:
            arm = row.id or ""
            stat = stats[arm]
            assert row.value == stat.mean
            own = (
                stat.ci
                if stat.ci is not None
                else (stat.mean - stat.half_width, stat.mean + stat.half_width)
                if stat.mean is not None and stat.half_width is not None
                else None
            )
            expected = own if level else None
            if expected is None and computed_here and stat.sem is None:
                expected = _binomial_interval(report_input, arm, key)
                computed_rows += expected is not None
            if expected is None:
                assert row.lo is None and row.hi is None, f"{title or 'overall'}, {arm}: no interval to draw"
            else:
                low, high = (max(expected[0], 0.0), min(expected[1], 100.0)) if rate else expected
                assert (row.lo, row.hi) == pytest.approx((low, high)), f"{title or 'overall'}, {arm}"
            assert bool(row.hollow) == (expected_line is not None and row.value < expected_line)
    assert (COMPUTED_NOTE in drawn.notes) == (computed_rows > 0), "a computed interval is named as code's"


def _stored_line(report_input: ReportInput, spec: ChartSpec) -> float | None:
    """The stored reference line's value, read off its figure key: a margin below zero, any other line at its figure;
    None where the plan stores none."""
    if not spec.reference:
        return None
    stored = spec.reference[0]
    value = resolve(stored.key, report_input).value
    assert isinstance(value, float)
    return -value if stored.kind == "margin" else value


def _binomial_interval(report_input: ReportInput, arm: str, key: str) -> tuple[float, float] | None:
    """The interval code computes on an arm's overall rate that states none: its ``spread`` key's, unless the overall
    is the plain mean of its target-matrix cells (its ``overall_interval`` key resolves), whose spread is theirs."""

    def read(figure_key: str) -> object | None:
        try:
            return resolve(figure_key, report_input).value
        except UnresolvableReference:
            return None  # NOSILENT: the key reads nothing on this figure, which is what None tells the caller

    if read(f"overall_interval.{key}.{arm}.interval") is not None:
        return None
    value = read(f"spread.{arm}.{key}.overall.interval")
    if value is None:
        return None
    assert isinstance(value, tuple)
    return float(value[0]), float(value[1])


# --- kinds no stored plan draws ------------------------------------------------------------------------------------


def test_range_bars_of_tasks_run_from_p50_to_p95_against_the_p95_budget() -> None:
    report_input = _input("r1")
    (budget,) = [c for c in report_input.criteria if c.statistic == "p95" and c.scope == "each_task"]
    data = _data(_draw("r1", _spec("range_bars", _query("tasks", measures=[budget.measure]))))
    assert isinstance(data, RangeBarsChart)
    assert [group.label for group in data.groups] == [task.name or task.id for task in report_input.tasks]
    assert data.reference is not None and data.reference.value == budget.threshold
    assert data.reference.kind == "budget"
    for task, group in zip(report_input.tasks, data.groups, strict=True):
        assert len(group.rows) == len(task.per_arm)
        for row in group.rows:
            (arm,) = [arm for arm in task.per_arm if row.label.endswith(_look("r1").arm(arm).name)]
            stat = task.per_arm[arm][budget.measure]
            assert (row.from_, row.to) == (stat.p50, stat.p95)
            assert bool(row.hollow) == (float(stat.p95 or 0) > budget.threshold)


def test_range_bars_of_a_figure_with_only_its_p95_draw_its_end_mark_and_say_so() -> None:
    report_input = _input("r3")
    (key,) = [
        key
        for key, measure in report_input.measures.items()
        if any(s.p95 is not None for s in measure.per_arm.values())
    ]
    drawn = _draw("r3", _spec("range_bars", _query("measures", measures=[key])))
    data = _data(drawn)
    assert isinstance(data, RangeBarsChart)
    assert all(row.from_ == row.to for group in data.groups for row in group.rows)
    assert drawn.notes and "no p50" in drawn.notes[0]


def test_a_frontier_scatter_puts_the_better_higher_measure_up_and_hollows_what_fails_a_gate() -> None:
    report_input = _input("r3")
    (gate,) = [c for c in report_input.criteria if c.scope == "overall" and c.statistic == "mean"]
    cost = next(key for key, measure in report_input.measures.items() if (measure.unit or "").startswith("usd"))
    data = _data(_draw("r3", _spec("scatter_frontier", _query("measures", measures=[cost, gate.measure]))))
    assert isinstance(data, ScatterChart) and data.frontier is True
    points = data.points or []
    assert {point.id for point in points} == set(report_input.measures[gate.measure].per_arm)
    for point in points:
        assert point.y == report_input.measures[gate.measure].per_arm[point.id or ""].mean
        assert point.x == report_input.measures[cost].per_arm[point.id or ""].mean
        assert bool(point.hollow) == (gate_check(report_input, point.id or "").status == "fail")
    costs = [point.x for point in points]
    assert data.x_log is (True if max(costs) / min(costs) > 10 else None)
    assert data.reference is not None and data.reference.value == gate.threshold


def test_a_frontier_scatter_of_tasks_is_a_panel_per_task_each_point_checked_in_its_task() -> None:
    report_input = _input("r1")
    gate = next(c for c in report_input.criteria if c.scope == "each_task" and c.statistic == "mean")
    cost = next(key for key, measure in report_input.measures.items() if (measure.unit or "").startswith("usd"))
    data = _data(_draw("r1", _spec("scatter_frontier", _query("tasks", measures=[gate.measure, cost]))))
    assert isinstance(data, ScatterChart) and data.panels is not None
    assert [panel.title for panel in data.panels] == [task.name or task.id for task in report_input.tasks]
    for task, panel in zip(report_input.tasks, data.panels, strict=True):
        for point in panel.points or []:
            failed = gate_check(report_input, point.id or "", task=task.id).status == "fail"
            assert bool(point.hollow) == failed
        assert panel.reference is not None and panel.reference.value == gate.threshold


def test_a_difference_heatmap_draws_the_second_arm_less_the_first_with_the_compared_rate_under_it() -> None:
    report_input = _input("r3")
    first, second = (arm.id for arm in report_input.arms[:2])
    (key,) = {key for cell in report_input.cells.values() for figures in cell.per_arm.values() for key in figures}
    drawn = _draw("r3", _spec("diff_heatmap", _query("cells", measures=[key], arms=[first, second])))
    data = _data(drawn)
    assert isinstance(data, DiffHeatmapChart)
    matrix = report_input.target_matrix
    assert matrix is not None
    cells = {(cell.row, cell.column): cell for cell in report_input.cells.values()}
    for row_level, row in zip(matrix.rows.levels, data.rows, strict=True):
        for column_level, cell in zip(matrix.columns.levels, row.cells, strict=True):
            held = cells[row_level.id, column_level.id].per_arm
            ours, theirs = held[second][key].mean, held[first][key].mean
            assert ours is not None and theirs is not None
            assert cell.value == pytest.approx(ours - theirs) and cell.detail == ours
    names = _look("r3")
    assert drawn.notes == [
        f"Each cell is {names.arm(second).name} less {names.arm(first).name}, in points; "
        f"the small figure is {names.arm(second).name}'s own."
    ]


def test_a_difference_heatmap_of_one_arm_on_an_input_with_a_control_draws_it_less_the_control() -> None:
    report_input = _input("r5")
    control = control_arm(report_input)
    candidate = next(arm.id for arm in report_input.arms if arm.id != control)
    key = next(iter(report_input.strata[0].per_arm[candidate]))
    data = _data(_draw("r5", _spec("diff_heatmap", _query("strata", measures=[key], arms=[candidate]))))
    assert isinstance(data, DiffHeatmapChart)
    slices = query_rows(_query("strata", measures=[key], arms=[candidate]), report_input)
    for row, drawn in zip(slices, data.rows, strict=True):
        assert isinstance(row.item, Stratum)
        ours, theirs = (row.item.per_arm[arm][key].mean for arm in (candidate, control or ""))
        assert ours is not None and theirs is not None
        assert drawn.cells[0].value == pytest.approx((ours - theirs) * 100 if abs(ours) <= 1 else ours - theirs)


def test_a_heatmap_of_strata_is_the_slices_by_the_arms() -> None:
    report_input = _input("r4")
    key = next(iter(report_input.strata[0].per_arm[report_input.arms[0].id]))
    drawn = _draw("r4", _spec("heatmap", _query("strata", measures=[key])))
    data = _data(drawn)
    assert isinstance(data, HeatmapChart)

    def held(arm: str, item: Stratum) -> bool:
        stat = item.per_arm.get(arm, {}).get(key)
        return stat is not None and stat.mean is not None

    arms = [arm.id for arm in report_input.arms]
    whole = [arm for arm in arms if all(held(arm, item) for item in report_input.strata)]
    kept = report_input.strata
    if not whole:
        whole = [arm for arm in arms if any(held(arm, item) for item in report_input.strata)]
        kept = [item for item in report_input.strata if all(held(arm, item) for arm in whole)]
    assert kept and len(kept) < len(report_input.strata), "the fixture must reach the slice rule"
    assert [row.label for row in data.rows or []] == [item.name or item.id for item in kept]
    assert len(data.columns or []) == len(whole)
    for item, row in zip(kept, data.rows or [], strict=True):
        assert [cell.value for cell in row.cells] == [item.per_arm[arm][key].mean for arm in _look("r4").ordered(whole)]
    assert drawn.notes and drawn.notes[0].startswith("Not in the evidence")


@pytest.mark.parametrize(
    ("name", "source"),
    [("r5", "readings"), ("r4", "contrasts"), ("r5", "floors"), ("r4", "strata")],
)
def test_a_dot_strip_draws_changes_floors_and_slices(name: str, source: str) -> None:
    report_input = _input(name)
    drawn = _draw(name, _spec("dot_interval", _query(source), level="95%"))
    data = _data(drawn)
    assert isinstance(data, DotIntervalChart)
    rows = [row for panel in (data.panels or [data]) for row in panel.rows or []]
    selected = query_rows(_query(source), report_input)
    if source == "floors":
        for row in rows:
            (read,) = [one.item for one in selected if one.arm == row.id]
            assert row.value == -read.worst_drop  # type: ignore[union-attr,operator]
            assert bool(row.hollow) == (read.status == "broken")  # type: ignore[union-attr]
    elif source in ("readings", "contrasts"):
        changes = {one.label: one.item for one in selected}
        assert len(rows) == len([item for item in changes.values() if getattr(item, "delta", 0) is not None])
    else:
        assert rows


# --- small multiples -----------------------------------------------------------------------------------------------


def test_a_heatmap_of_cells_is_a_panel_per_arm_each_with_its_chip() -> None:
    arms = [arm.id for arm in _input("r3").arms[:3]]
    data = _data(_draw("r3", _spec("heatmap", _query("cells", arms=arms, measures=["accuracy"]))))
    assert isinstance(data, HeatmapChart) and data.panels is not None
    look = _look("r3")
    assert [(panel.title, panel.tag) for panel in data.panels] == [
        (look.arm(arm).name, look.arm(arm).tag) for arm in arms
    ]


def test_confusion_is_a_panel_per_axis_for_one_arm_and_per_arm_for_several() -> None:
    report_input = _input("r3")
    axes = list(report_input.confusion)
    one = report_input.arms[0].id
    data = _data(_draw("r3", _spec("confusion", _query("confusion", arms=[one]))))
    assert isinstance(data, ConfusionChart) and data.panels is not None
    assert len(data.panels) == len(axes) and all(panel.tag is None for panel in data.panels)
    two = [arm.id for arm in report_input.arms[:2]]
    data = _data(_draw("r3", _spec("confusion", _query("confusion", arms=two, measures=[axes[0]]))))
    assert isinstance(data, ConfusionChart) and data.panels is not None
    assert [panel.tag for panel in data.panels] == [_look("r3").arm(arm).tag for arm in two]


def test_a_dot_strip_of_tasks_by_arm_is_a_panel_per_arm_a_row_per_task() -> None:
    report_input = _input("r1")
    key = next(c.measure for c in report_input.criteria if c.scope == "each_task" and c.statistic == "mean")
    data = _data(_draw("r1", _spec("dot_interval", _query("tasks", measures=[key], small_multiples_by="arm"))))
    assert isinstance(data, DotIntervalChart) and data.panels is not None
    look = _look("r1")
    assert [panel.title for panel in data.panels] == [look.arm(arm).name for arm in look.order]
    for panel in data.panels:
        assert len(panel.rows or []) == len(report_input.tasks)


# --- the plain table -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("source", _SOURCES)
def test_a_table_lays_out_the_rows_its_query_selects_for_any_source(source: str) -> None:
    # The first input whose rows a query of the source selects: every row, else (a source an analysis computes reads
    # the arms and the measure a query names) the first candidate query that selects any.
    name, query = next(
        (name, spec.query)
        for name in _PIN_INPUTS
        for spec in [_spec("table", _query(source)), *_candidates(name, "table", source)]
        if query_rows(spec.query, _variant(name))
    )
    drawn = _draw(name, _spec("table", query))
    block = drawn.chart
    assert isinstance(block, TableBlock) and block.rows
    assert block.total_rows == len(block.rows)
    selected = query_rows(query, _variant(name))
    # A wide table lays a figure out as a cell: its arm's (its place's) row, its measure's column; any other, a row.
    measured = {
        column.key.removeprefix("measure:")
        for column in block.columns
        if column.key.startswith("measure:") and not column.key.endswith(":interval")
    }
    if measured and source in _SLICE_PLACES:
        _assert_a_row_per_slice_and_arm(name, source, block, selected)
    elif measured:
        assert measured >= {row.key for row in selected if row.key}, "every selected measure has its column"
        assert len(block.rows) == len({row.arm for row in selected}), "a row per selected arm"
        for row in block.rows:
            arm = next(arm for arm in _look(name).arms if _look(name).arm(arm).name == row["arm"].value)
            for key in measured:
                held = any(one.arm == arm and one.key == key and _holds_figure(one.item) for one in selected)
                assert (row[f"measure:{key}"].value is not None) == held, (source, arm, key)
    else:
        assert len(block.rows) >= len(selected), "every selected row is laid out"


#: The slice sources a wide table lays out a row per (slice, arm) of, by the column naming the slice.
_SLICE_PLACES = {"strata": "slice", "tasks": "slice", "cells": "cell"}


def _place_name(report_input: ReportInput, look: PageLook, item: Stratum | Task | Cell) -> str:
    """A slice as its row names it, read from the input: a stratum's or task's name (else the look's reader name), a
    cell's two levels by their names."""
    if isinstance(item, Cell):
        matrix = report_input.target_matrix
        factors = [*report_input.factors, *((matrix.rows, matrix.columns) if matrix is not None else ())]
        names = {level.id: level.name for factor in factors for level in factor.levels if level.name}
        return f"{names.get(item.row, item.row)} × {names.get(item.column, item.column)}"
    return item.name or look.reader("stratum" if isinstance(item, Stratum) else "task", item.id)


def _holds_figure(stat: Any) -> bool:
    return stat is not None and any(getattr(stat, field) is not None for field in ("mean", "p50", "p95"))


def _assert_a_row_per_slice_and_arm(name: str, source: str, block: TableBlock, selected: list[Any]) -> None:
    """A slice table's contract: exactly one row per (slice, arm) the query selects that holds figures, each
    measure's figure in its own column, a dash where the input holds none."""
    report_input, look = _input(name), _look(name)
    expected: dict[tuple[str, str], dict[str, Any]] = {}
    for row in selected:
        for arm, figures in row.item.per_arm.items():
            if arm in look.arms and figures:
                expected[(_place_name(report_input, look, row.item), look.arm(arm).name)] = figures
    laid = [(row[_SLICE_PLACES[source]].value, row["arm"].value) for row in block.rows]
    assert len(laid) == len(set(laid)), f"{source}: a (slice, arm) laid out twice"
    assert set(laid) == set(expected), f"{source}: rows {sorted(set(laid) ^ set(expected))} differ from the input's"
    for row, place in zip(block.rows, laid, strict=True):
        for column in block.columns:
            key = column.key.removeprefix("measure:")
            if key == column.key or column.key.endswith(":interval"):
                continue
            held = _holds_figure(expected[place].get(key))
            assert (row[column.key].value is not None) == held, f"{source}: {place} {key} is in the wrong column"


# --- checks and renderer agree on a chart less the control ---------------------------------------------------------


def _section_faults(name: str, spec: ChartSpec) -> list[tuple[str, str]]:
    plan = _anchor(name.partition("+")[0].partition("-")[0])
    section = plan.sections[-1].model_copy(update={"id": "probe", "chart": spec})
    index = len(plan.sections)
    try:
        validate_plan(plan.model_copy(update={"sections": [*plan.sections, section]}), _variant(name))
    except PlanRefusal as refused:
        return [(path, why) for path, why in refused.faults if path.startswith(f"sections[{index}].chart")]
    return []


@pytest.mark.parametrize("source", ["measures", "judged", "strata", "tasks"])
def test_diverging_bars_less_the_control_are_refused_by_checks_and_renderer_alike_without_a_control_only(
    source: str,
) -> None:
    name = next(name for name in INPUTS if query_rows(_query(source), _input(name)))
    no_control = name if control_arm(_input(name)) is None else f"{name}-control"
    assert control_arm(_variant(no_control)) is None
    key = _vocabulary(name, source)[0]
    spec = _spec("diverging_bars", _query(source, measures=[key]))
    assert any(path.endswith(".kind") and "no control" in why for path, why in _section_faults(no_control, spec))
    with pytest.raises(UnresolvableReference):
        _draw(no_control, spec)
    with_control = name if control_arm(_input(name)) is not None else f"{name}+control"
    assert not [fault for fault in _section_faults(with_control, spec) if "no control" in fault[1]]
    assert _data(_draw(with_control, spec)), "with a control, both take it"


def test_a_difference_heatmap_without_a_control_needs_its_two_arms_named_by_checks_and_renderer_alike() -> None:
    assert control_arm(_input("r3")) is None
    one = [_input("r3").arms[0].id]
    two = [arm.id for arm in _input("r3").arms[:2]]
    lone = _spec("diff_heatmap", _query("cells", arms=one))
    assert any(path.endswith(".query.arms") for path, _ in _section_faults("r3", lone))
    with pytest.raises(UnresolvableReference):
        _draw("r3", lone)
    pair = _spec("diff_heatmap", _query("cells", arms=two))
    assert not [fault for fault in _section_faults("r3", pair) if fault[0].endswith(".query.arms")]
    assert _data(_draw("r3", pair))


# --- markup in input names ------------------------------------------------------------------------------------------


def _hostile_r3() -> ReportInput:
    data = _input("r3").model_dump(by_alias=True, exclude_none=True)
    for index, arm in enumerate(data["arms"]):
        arm.update(name=f"{HOSTILE_LINE} {index}")
        arm.pop("short_name", None)
    for axis in ("rows", "columns"):
        factor = data["target_matrix"][axis]
        factor["name"] = HOSTILE_LINE
        for level in factor["levels"]:
            level["name"] = f"{HOSTILE_LINE} {level['id']}"
    for figures in data["measures"].values():
        figures["label"] = HOSTILE_LINE
    return ReportInput.model_validate(data)


def test_markup_in_input_names_is_text_on_a_page_of_this_modules_kinds() -> None:
    plan = _anchor("r3")
    report_input = _hostile_r3()
    arms = [arm.id for arm in report_input.arms[:2]]
    specs = [
        _spec("heatmap", _query("cells", arms=arms, measures=["accuracy"])),
        _spec("diff_heatmap", _query("cells", arms=arms, measures=["accuracy"])),
        _spec("confusion", _query("confusion", arms=arms)),
        _spec("dot_interval", _query("measures", measures=["balanced_accuracy"])),
        _spec("table", _query("cells", arms=arms)),
    ]
    sections = [
        plan.sections[0].model_copy(update={"id": f"s{index}", "chart": spec}) for index, spec in enumerate(specs)
    ]
    page = plan_page(plan.model_copy(update={"summary": [], "sections": sections}), report_input)
    html = planned_page_html(page)
    assert what_runs(html) == []
    assert HOSTILE_LINE not in html, "no name is written into the page as markup"
    assert_static(planned_page_html(page, static=True))
    assert planned_page_markdown(page)


# --- the legend, the clipped interval, a config's colour and the labels -------------------------------------------


def test_range_bars_carry_a_legend_of_the_arms_drawn_and_name_a_bar_over_its_budget() -> None:
    report_input = _input("r1")
    data = _data(_draw("r1", _spec("range_bars", _query("tasks", measures=["latency_s"]))))
    assert isinstance(data, RangeBarsChart) and data.legend is not None
    look = _look("r1")
    drawn = {arm.id for arm in report_input.arms}
    assert [(item.label, item.color, item.mark) for item in data.legend] == [
        (look.arm(arm).name, look.arm(arm).token, look.arm(arm).shape) for arm in look.ordered(drawn)
    ]
    assert data.reference is not None and data.reference.kind == "budget"
    over = [row for group in data.groups for row in group.rows if row.to > data.reference.value]
    assert over, "some bar ends over the budget, so its tooltip line is reached"
    for row in over:
        assert row.hollow and row.tip and any((data.reference.label or "") in line for line in row.tip[2:])


def test_a_rate_strips_interval_never_runs_past_0_or_100() -> None:
    """r1's speculation task: Atlas-M declines 98% ±2.2, a half-width that runs past 100."""
    report_input = _input("r1")
    spec = _spec("dot_interval", _query("tasks", measures=["declines"], strata=["specul"]), level="95%")
    data = _data(_draw("r1", spec))
    assert isinstance(data, DotIntervalChart) and data.rows
    task = next(task for task in report_input.tasks if task.id == "specul")
    past = 0
    for row in data.rows:
        stat = task.per_arm[row.id or ""]["declines"]
        assert stat.mean is not None and stat.half_width is not None
        assert row.lo == max(stat.mean - stat.half_width, 0) and row.hi == min(stat.mean + stat.half_width, 100)
        past += stat.mean + stat.half_width > 100
    assert past, "some interval runs past 100, so the clip is reached"


def test_a_config_frontier_colours_each_config_by_its_first_factors_level_and_names_the_levels() -> None:
    report_input = _input("r2")
    data = _data(_draw("r2", _spec("scatter_frontier", _query("configs", measures=["pass_rate", "cost_per_1k"]))))
    assert isinstance(data, ScatterChart) and data.points and data.legend
    first = report_input.factors[0]
    levels = {config.arm: config.levels[first.id] for config in report_input.configs}
    colours: dict[str, str] = {}
    for point in data.points:
        assert point.color is not None
        assert colours.setdefault(levels[point.id or ""], point.color) == point.color
    assert len(set(colours.values())) == len(colours) == len(first.levels)
    assert [item.label for item in data.legend] == [level.name or level.id for level in first.levels]
    assert [item.color for item in data.legend] == [colours[level.id] for level in first.levels]


@pytest.mark.parametrize(
    ("name", "source", "measures"),
    [("r2", "configs", ["pass_rate", "cost_per_1k"]), ("r4", "judged", ["persona", "facts"])],
    ids=["many-points", "few-points"],
)
def test_a_frontier_labels_every_point_up_to_eight_else_only_the_stressed_and_recommended(
    name: str, source: str, measures: list[str]
) -> None:
    plan = _anchor(name)
    data = _data(
        _draw(
            name, _spec("scatter_frontier", _query(source, measures=measures), emphasis=plan.sections[0].chart.emphasis)
        )
    )
    assert isinstance(data, ScatterChart) and data.points
    stressed = {data.emphasis, plan_recommended_arm(plan)} - {None, ""}
    labelled = {point.id for point in data.points if point.label}
    if len(data.points) <= 8:
        assert labelled == {point.id for point in data.points}
    else:
        assert stressed and labelled == stressed & {point.id for point in data.points}


@pytest.mark.parametrize("kind", ["floor", "budget"])
def test_a_value_that_is_not_a_number_misses_its_line(kind: Literal["floor", "budget"]) -> None:
    """A dot or bar end no comparison holds for is not drawn as meeting its floor or budget."""
    line = ChartReference(value=90.0, kind=kind)
    assert misses_line(math.nan, line)
