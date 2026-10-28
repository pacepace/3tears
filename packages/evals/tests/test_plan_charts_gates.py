"""The gate family's charts on a planned page: gate matrices, gate scatters and lattices, each drawn from the plan's
query, the criteria and the configs.

Held over every r1–r5 anchor and current run, and over charts these tests build on the same inputs for the sources no
committed plan draws yet. The expected side is always computed from the input, never typed in: the gate check
(:func:`~threetears.evals.analysis.report_gates.gate_check`), routing
(:func:`~threetears.evals.analysis.report_gates.routing_picks`), the criteria's thresholds through their figure keys
(:func:`~threetears.evals.analysis.report.figure_keys.resolve`), the rows a query selects
(:func:`~threetears.evals.analysis.report_rows.query_rows`) and the factors and configs.

- **Every chart of these kinds resolves** to its own kind, the kit's model validates it, and its values table holds
  every row; every (kind, source) pair the plan checks accept for these kinds draws on some input.
- **Gate matrix over tasks** (r1): a cell per task and arm, its pill GENERATOR §3's colour for the task-narrowed
  check and its word, its reasons the failures else the borderline, its cost the task's own per 1,000 replies, the
  routing pick marked; a cross-cutting task flagged. A task no gate is read in states no status.
- **Gate matrix over readings, floors, measures and configs**: §3's direction rule with the margin verdict in the
  words; a floor's status; each criterion checked alone.
- **Gate scatter** (r2): each point at the figures the two gates read, both lines at the gates' thresholds, hollow
  where the check fails; the paths join configs that differ in one factor only, in that factor's order, coloured by
  the first factor's level, and cover the grid.
- **Lattice** (r3): the first factor's levels by the rest's, each tile the config's figure, cost per 1,000 replies,
  p95 latency and gate status.
- **Names are text**: markup in every name these charts print is escaped on the page.
"""

from __future__ import annotations

import functools
import math
from itertools import product
from pathlib import Path
from typing import Any

import pytest

from packages.evals.tests.fixtures.report_plans import dataset
from packages.evals.tests.hostile_page_support import HOSTILE_LINE
from packages.evals.tests.page_contract_support import what_runs
from threetears.evals.analysis import (
    ChartBlock,
    GateMatrixChart,
    GateScatterChart,
    LatticeChart,
    PlannedPage,
    planned_page_html,
    planned_page_markdown,
)
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.cost_unit import REPLIES, per_count, per_noun
from threetears.evals.analysis.report.figure_keys import cost_basis, format_figure, resolve
from threetears.evals.analysis.report.plan_page.build import plan_page
from threetears.evals.analysis.report.plan_page.drawable import plan_look
from threetears.evals.analysis.report.plan_page.resolve import resolve_chart
from threetears.evals.analysis.report.plan_page.words import GATE_WORDS, NOT_GATED
from threetears.evals.analysis.report_decision import margin_sensitivity
from threetears.evals.analysis.report_plan_facts import plan_facts
from threetears.evals.analysis.report_gates import gate_check, routing_picks, task_costs
from threetears.evals.analysis.report_input import Criterion, FloorRead, Reading, ReportInput, Task
from threetears.evals.analysis.report_margins import criterion_place, flip_margin
from threetears.evals.analysis.report_rows import query_rows
from threetears.evals.kernel.report_plan import ChartSpec, ReportPlan

#: The kinds this file holds.
_KINDS = frozenset({"gate_matrix", "gate_scatter", "lattice"})

#: GENERATOR §3: each gate status's colour.
_STATUS_COLOUR = {"pass": "good", "borderline": "warn", "fail": "bad"}


def _plan_paths() -> list[tuple[str, str, Path]]:
    found = []
    for name in ("r1", "r2", "r3", "r4", "r5"):
        found += [(name, "anchor", dataset.HERE / name / "anchor.plan.json"), (name, "run", dataset.current_run(name))]
    return found


_PLANS = _plan_paths()


@functools.cache
def _input(name: str) -> ReportInput:
    return dataset.report_input_of(name)


@functools.cache
def _plan(path: Path) -> ReportPlan:
    return dataset.load_plan(path)


@functools.cache
def _page(name: str, path: Path) -> PlannedPage:
    return plan_page(_plan(path), _input(name))


def _stored_charts() -> list[Any]:
    """Every chart of these kinds a committed plan draws, beside its input, as parameters."""
    found = []
    for name, label, path in _PLANS:
        plan = _plan(path)
        specs = [(chart, True) for chart in plan.summary] + [(section.chart, False) for section in plan.sections]
        page = _page(name, path)
        blocks = [chart.chart for chart in page.summary] + [section.chart.chart for section in page.sections]
        for (spec, summary), block in zip(specs, blocks, strict=True):
            if spec.kind in _KINDS:
                assert isinstance(block, ChartBlock)
                found.append(
                    pytest.param(name, spec, block, summary, id=f"{name}-{label}-{spec.kind}-{spec.query.source}")
                )
    return found


_STORED = _stored_charts()


def _stored(kind: str, source: str) -> list[Any]:
    found = [param for param in _STORED if param.values[1].kind == kind and param.values[1].query.source == source]
    assert found, f"no committed plan draws a {kind} over {source}: the test would hold nothing"
    return found


def _spec(kind: str, source: str, **query: Any) -> ChartSpec:
    """A chart over the input, as a plan stores one: empty where the query names nothing."""
    reference = query.pop("reference", [])
    emphasis = query.pop("emphasis", "")
    level = query.pop("level", "")
    return ChartSpec.model_validate(
        {
            "kind": kind,
            "title": "",
            "query": {
                "source": source,
                "aspect": "",
                "measures": [],
                "arms": [],
                "strata": [],
                "limit": 0,
                "transform": "none",
                "small_multiples_by": "none",
                **query,
            },
            "reference": reference,
            "emphasis": emphasis,
            "level": level,
            "unit": "",
            "domain": [],
        }
    )


def _draw(name: str, spec: ChartSpec) -> tuple[ChartBlock, list[str]]:
    """The chart as the input's own anchor plan's page draws it in its first section."""
    plan = _plan(dataset.HERE / name / "anchor.plan.json")
    section = plan.sections[0].model_copy(update={"chart": spec})
    page = plan_page(plan.model_copy(update={"summary": [], "sections": [section]}), _input(name))
    block = page.sections[0].chart.chart
    assert isinstance(block, ChartBlock)
    return block, page.sections[0].chart.notes


def _gating(report_input: ReportInput) -> list[Criterion]:
    return [criterion for criterion in report_input.criteria if criterion.kind in ("gate", "guardrail")]


def _threshold(report_input: ReportInput, criterion: Criterion) -> float:
    value = resolve(f"criteria.{criterion.id}.threshold", report_input).value
    assert isinstance(value, float)
    return value


# --- every chart --------------------------------------------------------------------------------------------------

#: Every (kind, source) pair the plan checks accept for these kinds, each drawn on an input that holds it.
_PAIRS = [
    ("gate_matrix", "tasks", "r1", {}),
    ("gate_matrix", "readings", "r5", {}),
    ("gate_matrix", "floors", "r5", {}),
    ("gate_matrix", "measures", "r4", {}),
    ("gate_matrix", "configs", "r3", {}),
    ("gate_scatter", "measures", "r2", {"measures": ["pass_rate", "latency_ms"]}),
    ("gate_scatter", "judged", "r4", {"measures": ["facts", "specul"]}),
    ("gate_scatter", "configs", "r2", {}),
    ("gate_scatter", "tasks", "r1", {"measures": ["overall", "latency_s"]}),
    ("lattice", "configs", "r3", {"measures": ["balanced_accuracy"]}),
]


@pytest.mark.parametrize(("kind", "source", "name", "query"), _PAIRS, ids=[f"{k}-{s}" for k, s, _, _ in _PAIRS])
def test_every_kind_and_source_draws_its_own_kind_with_values_to_read(
    kind: str, source: str, name: str, query: dict[str, Any]
) -> None:
    block, _ = _draw(name, _spec(kind, source, **query))
    drawn = block.kit_chart()
    assert drawn is not None and drawn[0] == kind
    _, rows = block.values_table()
    assert rows, f"{kind} over {source} drew nothing"


@pytest.mark.parametrize(("name", "spec", "block", "summary"), _STORED)
def test_every_stored_chart_of_these_kinds_resolves_to_its_kind_and_validates(
    name: str, spec: ChartSpec, block: ChartBlock, summary: bool
) -> None:
    assert block.chart == spec.kind
    drawn = block.kit_chart()
    assert drawn is not None and drawn[0] == spec.kind
    assert type(block.data).model_validate(block.data.kit_data()) == block.data
    _, rows = block.values_table()
    assert rows


def test_the_plans_hold_every_spine_chart_of_these_kinds() -> None:
    """The anchors' spines that no chart query could express are drawn: r1's gate matrix, r2's gate scatter, r3's
    lattice."""
    kinds = {(param.values[0], param.values[1].kind) for param in _STORED}
    assert {("r1", "gate_matrix"), ("r2", "gate_scatter"), ("r3", "lattice")} <= kinds


# --- gate matrix --------------------------------------------------------------------------------------------------


def _reads_in_task(criterion: Criterion, task: str) -> bool:
    return criterion_place(criterion) == "task" and criterion.task in (None, task)


@pytest.mark.parametrize(("name", "spec", "block", "summary"), _stored("gate_matrix", "tasks"))
def test_a_task_matrix_is_each_tasks_gate_check_with_its_cost_and_the_routing_pick(
    name: str, spec: ChartSpec, block: ChartBlock, summary: bool
) -> None:
    report_input = _input(name)
    data = block.data
    assert isinstance(data, GateMatrixChart)
    tasks = [row.item for row in query_rows(spec.query, report_input) if isinstance(row.item, Task)]
    arms = list(spec.query.arms) or [arm.id for arm in report_input.arms]
    names = {arm.id: arm.short_name or arm.name or arm.id for arm in report_input.arms}
    assert [column.label for column in data.columns] == [names[arm] for arm in arms]
    picks = routing_picks(report_input, arms)
    assert [row.label for row in data.rows] == [task.name or task.id for task in tasks]
    for task, row in zip(tasks, data.rows, strict=True):
        assert bool(row.flag) == task.cross_cutting
        costs = task_costs(report_input, task.id)
        gated = any(_reads_in_task(criterion, task.id) for criterion in _gating(report_input))
        for arm, cell in zip(arms, row.cells, strict=True):
            assert cell.cost == costs.get(arm)
            if not gated:
                assert cell.status == "neutral"
                continue
            check = gate_check(report_input, arm, task=task.id)
            assert (cell.status, cell.label) == (_STATUS_COLOUR[check.status], GATE_WORDS[check.status])
            assert (cell.reasons or []) == list(check.fails or check.borders)
            assert bool(cell.pick) == (picks.get(task.id) == arm)


def test_a_task_matrix_prices_its_costs_per_what_the_task_cost_measure_counts() -> None:
    """A task matrix's costs are each task's own figures on a cost measure, so they are per what that measure counts,
    even where the page's cost is the scoreboard's per reply. r1 states its task costs on a per-1k measure of
    conversations; given scoreboard reply costs, the page's cost basis turns to replies and the matrix's must not."""
    data = _input("r1").model_dump(by_alias=True, exclude_none=True)
    for row in data["scoreboard"].values():
        row["reply_cost_usd"] = 0.001
    held = ReportInput.model_validate(data)
    assert cost_basis(held).noun == REPLIES, "the control: the page's cost is the scoreboard's, per reply"
    (key,) = [key for key, measure in held.measures.items() if measure.unit == "usd_per_1k"]
    noun = per_noun(held.measures[key])
    assert noun and noun != REPLIES, "the control: the task cost measure counts something else"
    costed = [task.id for task in held.tasks if task_costs(held, task.id)]
    assert costed, "the control: some task states a cost"
    assert all(task_costs(held, task).measure == key for task in costed)
    plan = _plan(dataset.HERE / "r1" / "anchor.plan.json")
    section = plan.sections[0].model_copy(update={"chart": _spec("gate_matrix", "tasks")})
    page = plan_page(plan.model_copy(update={"summary": [], "sections": [section]}), held)
    block = page.sections[0].chart.chart
    assert isinstance(block, ChartBlock) and isinstance(block.data, GateMatrixChart)
    assert block.data.cost_unit == per_count(noun)
    assert any(f" {per_count(noun)}" in cell for row in block.values_table()[1] for cell in row)


def test_a_task_matrix_whose_tasks_costs_count_different_things_is_refused_naming_each_cost_measure() -> None:
    """A matrix prints one cost unit, so where tasks' costs are read on measures counting different things it is
    refused, naming each. r1 states every task's cost on a per-1k measure; given a cost measure per reply too, a task
    that states only that one is costed on it, per reply, beside tasks costed per what the per-1k measure counts."""
    held = _input("r1")
    (per_1k,) = [key for key, measure in held.measures.items() if measure.unit == "usd_per_1k"]
    spec = _spec("gate_matrix", "tasks")
    drawn = _resolved(held, spec, "r1")
    assert isinstance(drawn.data, GateMatrixChart) and drawn.data.cost_unit, "the control: r1's matrix prices its tasks"
    data = held.model_dump(mode="json", by_alias=True, exclude_none=True)
    reply = "reply_cost"
    assert reply not in data["measures"]
    data["measures"][reply] = {
        "label": "Cost per reply",
        "unit": "usd",
        "higher_is_better": False,
        "merit_axis": "cost",
        "per_arm": {arm: {"mean": stat["mean"] / 1000} for arm, stat in data["measures"][per_1k]["per_arm"].items()},
    }
    moved = data["tasks"][0]
    for figures in moved["per_arm"].values():
        figures[reply] = {"mean": figures.pop(per_1k)["mean"] / 1000}
    mixed = ReportInput.model_validate(data)
    measures = {task.id: task_costs(mixed, task.id).measure for task in mixed.tasks}
    assert measures[moved["id"]] == reply, "the control: the task stating only a reply cost is costed on it"
    assert set(measures.values()) == {reply, per_1k}, "the control: the other tasks stay on the per-1k measure"
    nouns = {per_noun(mixed.measures[key]) for key in (reply, per_1k)}
    assert len(nouns) == 2, "the control: the two cost measures count different things"
    with pytest.raises(UnresolvableReference) as refused:
        _resolved(mixed, spec, "r1")
    said = str(refused.value)
    assert "counting different things" in said and "one cost unit" in said, said
    assert all(key in said for key in (reply, per_1k)), said


def test_r1s_matrix_marks_a_pick_on_every_routable_task_some_arm_can_take() -> None:
    """The fixture reaches the pick: r1's routable tasks each have one, the cross-cutting one none."""
    (param,) = [param for param in _stored("gate_matrix", "tasks") if param.id.startswith("r1-anchor")]
    data = param.values[2].data
    assert isinstance(data, GateMatrixChart)
    report_input = _input("r1")
    for task, row in zip(report_input.tasks, data.rows, strict=True):
        assert any(cell.pick for cell in row.cells) == bool(task.routable)
    assert data.pick_label and data.cost_fmt


def test_a_task_no_gate_reads_states_no_status_and_shows_its_figures() -> None:
    """r2 gates the whole comparison only: its tasks carry figures, and no cell claims a pass."""
    (param,) = _stored("gate_matrix", "tasks")[-1:]
    name, spec, block, _ = param.values
    report_input = _input(name)
    assert not any(criterion_place(criterion) == "task" for criterion in _gating(report_input))
    data = block.data
    assert isinstance(data, GateMatrixChart)
    tasks = [row.item for row in query_rows(spec.query, report_input) if isinstance(row.item, Task)]
    for task, row in zip(tasks, data.rows, strict=True):
        for arm, cell in zip(spec.query.arms, row.cells, strict=True):
            assert cell.status == "neutral" and cell.label not in GATE_WORDS.values()
            figure = resolve(f"task.{task.id}.{arm}.{spec.query.measures[0]}.mean", report_input)
            assert cell.reasons and format_figure(figure) in cell.reasons[0]


def _direction(reading: Reading) -> str:
    assert reading.within_interval is not None
    low, high = reading.within_interval
    return "worse" if high < 0 else "better" if low > 0 else ""


def test_a_readings_matrix_reads_each_candidate_by_the_direction_of_its_change() -> None:
    report_input = _input("r5")
    block, _ = _draw("r5", _spec("gate_matrix", "readings"))
    data = block.data
    assert isinstance(data, GateMatrixChart)
    aspects = [aspect for aspect in report_input.aspects if aspect.readings]
    assert len(data.rows) == len(aspects)
    for aspect, row in zip(aspects, data.rows, strict=True):
        readings = {reading.arm: reading for reading in aspect.readings}
        assert len(row.cells) == len(readings)
        for reading in aspect.readings:
            cell = row.cells[[column.label for column in data.columns].index(_arm_name(report_input, reading.arm))]
            direction = _direction(reading)
            assert cell.status == {"worse": "bad", "better": "good", "": "neutral"}[direction]
            margin = reading.margin if reading.margin is not None else aspect.margin
            assert reading.within_interval is not None and margin is not None
            within = flip_margin(reading) < margin  # the decision's rule, so the pill and the explore part agree
            assert ("within margin" in cell.label) == within and ("outside margin" in cell.label) == (not within)
            floors = [f for f in report_input.floors if f.arm == reading.arm and f.aspect == aspect.id and f.reading]
            assert len(cell.reasons or []) == 1 + len(floors), "each floor reading another rate is shown apart"


def _arm_name(report_input: ReportInput, arm_id: str) -> str:
    arm = next(arm for arm in report_input.arms if arm.id == arm_id)
    return arm.short_name or arm.name or arm.id


def test_a_floors_matrix_reads_each_floors_status() -> None:
    report_input = _input("r5")
    block, _ = _draw("r5", _spec("gate_matrix", "floors"))
    data = block.data
    assert isinstance(data, GateMatrixChart)
    floors = [row.item for row in query_rows(_spec("gate_matrix", "floors").query, report_input)]
    assert all(isinstance(floor, FloorRead) for floor in floors)
    keys = list(dict.fromkeys(floor.floor for floor in floors if isinstance(floor, FloorRead)))
    assert len(data.rows) == len(keys)
    columns = [column.label for column in data.columns]
    for floor in floors:
        assert isinstance(floor, FloorRead)
        cell = data.rows[keys.index(floor.floor)].cells[columns.index(_arm_name(report_input, floor.arm))]
        assert cell.status == {"held": "good", "unclear": "warn", "broken": "bad"}[floor.status]
        assert cell.label.lower() == floor.status
        assert floor.worst_drop is not None and cell.reasons
        assert f"{floor.worst_drop:.1f}".rstrip("0").rstrip(".") in cell.reasons[0]


@pytest.mark.parametrize(("name", "source"), [("r4", "measures"), ("r3", "configs"), ("r2", "configs")])
def test_a_criteria_matrix_checks_each_arm_against_each_gate_alone(name: str, source: str) -> None:
    report_input = _input(name)
    block, _ = _draw(name, _spec("gate_matrix", source))
    data = block.data
    assert isinstance(data, GateMatrixChart)
    criteria = _gating(report_input)
    assert [row.label for row in data.rows] == [criterion.label or criterion.id for criterion in criteria]
    arms = list(dict.fromkeys(row.arm for row in query_rows(_spec("gate_matrix", source).query, report_input)))
    if source == "configs":
        arms = [config.arm for config in report_input.configs if config.arm in arms]
    statuses = set()
    for criterion, row in zip(criteria, data.rows, strict=True):
        assert len(row.cells) == len(arms)
        for arm, cell in zip(arms, row.cells, strict=True):
            check = gate_check(report_input, arm, criterion=criterion.id)
            assert (cell.status, cell.label) == (_STATUS_COLOUR[check.status], GATE_WORDS[check.status])
            assert (cell.reasons or []) == list(check.fails or check.borders)
            statuses.add(check.status)
    assert "fail" in statuses and statuses - {"fail"}, "the input reaches both a failing and a passing cell"


# --- gate scatter -------------------------------------------------------------------------------------------------


def _axis_criteria(report_input: ReportInput) -> tuple[Criterion, Criterion]:
    """The two gates read over the whole comparison, the higher-is-better one first (up)."""
    overall = [criterion for criterion in _gating(report_input) if criterion_place(criterion) == "overall"]
    (up,) = [criterion for criterion in overall if criterion.direction == "higher_is_better"]
    (across,) = [criterion for criterion in overall if criterion.direction == "lower_is_better"]
    return up, across


def _at(report_input: ReportInput, criterion: Criterion, arm: str) -> float:
    stat = report_input.measures[criterion.measure].per_arm[arm]
    value = getattr(stat, criterion.statistic)
    assert value is not None
    return float(value)


@pytest.mark.parametrize(("name", "spec", "block", "summary"), _stored("gate_scatter", "configs"))
def test_a_gate_scatter_draws_each_config_at_the_gates_own_figures_between_the_gates_lines(
    name: str, spec: ChartSpec, block: ChartBlock, summary: bool
) -> None:
    report_input = _input(name)
    data = block.data
    assert isinstance(data, GateScatterChart) and data.points is not None
    up, across = _axis_criteria(report_input)
    assert data.reference is not None and data.x_reference is not None
    assert (data.reference.value, data.reference.kind) == (_threshold(report_input, up), "floor")
    assert (data.x_reference.value, data.x_reference.kind) == (_threshold(report_input, across), "budget")
    assert {point.id for point in data.points} == {config.arm for config in report_input.configs}
    for point in data.points:
        assert point.id is not None
        assert (point.x, point.y) == (_at(report_input, across, point.id), _at(report_input, up, point.id))
        assert bool(point.hollow) == (gate_check(report_input, point.id).status == "fail")
    assert data.region_label


@pytest.mark.parametrize(("name", "spec", "block", "summary"), _stored("gate_scatter", "configs"))
def test_its_paths_join_configs_along_one_factor_coloured_by_the_first_and_cover_the_grid(
    name: str, spec: ChartSpec, block: ChartBlock, summary: bool
) -> None:
    report_input = _input(name)
    data = block.data
    assert isinstance(data, GateScatterChart) and data.paths
    levels = {config.arm: config.levels for config in report_input.configs}
    first = report_input.factors[0]
    colours: dict[str, str] = {}
    along_factors = set()
    for path in data.paths:
        rows = [levels[arm] for arm in path.ids]
        varying = [factor for factor in report_input.factors if len({row[factor.id] for row in rows}) > 1]
        assert len(varying) == 1, "a path holds every other factor at one level"
        (along,) = varying
        along_factors.add(along.id)
        order = [level.id for level in along.levels]
        assert [order.index(row[along.id]) for row in rows] == sorted(order.index(row[along.id]) for row in rows)
        assert len({row[first.id] for row in rows}) == 1
        assert path.color is not None
        assert colours.setdefault(rows[0][first.id], path.color) == path.color
    assert len(set(colours.values())) == len(colours), "each level of the first factor its own colour"
    (along_id,) = along_factors
    assert along_id != first.id
    combinations = math.prod(len(factor.levels) for factor in report_input.factors if factor.id != along_id)
    assert len(data.paths) == combinations
    assert {arm for path in data.paths for arm in path.ids} == set(levels)


@pytest.mark.parametrize("measures", [["overall", "latency_s"], []], ids=["named", "the-gates-every-task-reads"])
def test_a_gate_scatter_over_tasks_panels_each_task_at_its_own_gates(measures: list[str]) -> None:
    report_input = _input("r1")
    block, _ = _draw("r1", _spec("gate_scatter", "tasks", measures=measures))
    data = block.data
    assert isinstance(data, GateScatterChart) and data.panels is not None
    assert [panel.title for panel in data.panels] == [task.name or task.id for task in report_input.tasks]
    for task, panel in zip(report_input.tasks, data.panels, strict=True):
        assert panel.points is not None and panel.reference is not None and panel.x_reference is not None
        for point in panel.points:
            assert point.id is not None
            check = gate_check(report_input, point.id, task=task.id)
            assert bool(point.hollow) == (check.status == "fail")
            assert point.y == task.per_arm[point.id]["overall"].mean


# --- lattice ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "spec", "block", "summary"), _stored("lattice", "configs"))
def test_a_lattices_costs_are_the_pages_cost_and_per_what_it_counts(
    name: str, spec: ChartSpec, block: ChartBlock, summary: bool
) -> None:
    """A lattice tile's cost is the arm's cost by the page's one rule (``cost_per_1k.<arm>``), so its unit is what that
    key's figure is per: the page's cost basis, read from the measure the cost is read on."""
    report_input = _input(name)
    data = block.data
    assert isinstance(data, LatticeChart)
    costs = [cell for row in data.rows for cell in row.cells if cell.cost is not None]
    assert costs, "the control: the lattice draws costs"
    # Costs under a cent per 1,000 are drawn per 1,000,000 (#993: one form, in dollars), every tile alike.
    scale = 1000 if data.cost_unit == per_count(cost_basis(report_input).noun, 1_000_000) else 1
    for cell in costs:
        figure = resolve(f"cost_per_1k.{cell.id}", report_input)
        assert isinstance(figure.value, float)
        assert cell.cost == pytest.approx(figure.value * scale)
        assert data.cost_unit == per_count(figure.per, 1000 * scale)
    assert per_noun(report_input.measures[cost_basis(report_input).source]) == cost_basis(report_input).noun


@pytest.mark.parametrize(("name", "spec", "block", "summary"), _stored("lattice", "configs"))
def test_a_lattice_tiles_the_first_factor_by_the_rest_with_each_configs_figures_and_status(
    name: str, spec: ChartSpec, block: ChartBlock, summary: bool
) -> None:
    report_input = _input(name)
    data = block.data
    assert isinstance(data, LatticeChart)
    first, *rest = report_input.factors
    assert [row.label for row in data.rows] == [level.name or level.id for level in first.levels]
    columns = list(product(*(factor.levels for factor in rest)))
    assert [column.label for column in data.columns] == [
        " · ".join(level.name or level.id for level in combination) for combination in columns
    ]
    by_levels = {
        tuple(config.levels[f.id] for f in report_input.factors): config.arm for config in report_input.configs
    }
    key = spec.query.measures[0]
    (latency,) = {criterion.measure for criterion in report_input.criteria if criterion.statistic == "p95"}
    for level, row in zip(first.levels, data.rows, strict=True):
        for combination, cell in zip(columns, row.cells, strict=True):
            arm = by_levels[(level.id, *(one.id for one in combination))]
            assert cell.id == arm
            assert cell.value == report_input.measures[key].per_arm[arm].mean
            # Drawn per 1,000,000 where a cost is under a cent per 1,000 (#993), every tile alike.
            scale = 1000 if data.cost_unit and "1,000,000" in data.cost_unit else 1
            assert cell.cost == pytest.approx(float(resolve(f"cost_per_1k.{arm}", report_input).value) * scale)
            assert cell.latency == report_input.measures[latency].per_arm[arm].p95
            check = gate_check(report_input, arm)
            assert (cell.status, cell.status_label) == (_STATUS_COLOUR[check.status], GATE_WORDS[check.status])
    assert data.emphasis == spec.emphasis


def _resolved(report_input: ReportInput, spec: ChartSpec, anchor: str) -> ChartBlock:
    """The chart as the page's resolver draws it on ``report_input``, under the look the input's anchor plan gives."""
    plan = _plan(dataset.HERE / anchor / "anchor.plan.json")
    drawn = resolve_chart(spec, report_input, plan_look(plan, report_input, plan_facts(report_input)), title="")
    assert isinstance(drawn.chart, ChartBlock)
    return drawn.chart


def test_at_a_zero_margin_a_reading_is_outside_it_by_the_decisions_rule() -> None:
    """A reading shown better (its interval above zero) flips at a margin of zero: at a declared margin of zero it is
    not within it, as the decision and the explore part read it."""
    data = _input("r5").model_dump(mode="json", by_alias=True, exclude_none=True)
    aspect = data["aspects"][0]
    aspect["margin"] = 0.0
    for reading in aspect["readings"]:
        reading["margin"] = None
    aspect["readings"][0]["within_interval"] = [0.5, 3.0]
    report_input = ReportInput.model_validate(data)
    reading = report_input.aspects[0].readings[0]
    assert flip_margin(reading) == 0.0 and -float(reading.within_interval[0]) < 0.0  # the two rules differ here

    data_block = _resolved(report_input, _spec("gate_matrix", "readings"), "r5").data
    assert isinstance(data_block, GateMatrixChart)
    columns = [column.label for column in data_block.columns]
    cell = data_block.rows[0].cells[columns.index(_arm_name(report_input, reading.arm))]
    assert "outside margin" in cell.label and "within margin" not in cell.label
    assert not any(c.within for c in margin_sensitivity(report_input).candidates if c.arm == reading.arm)


def test_a_lattice_tile_no_gate_reads_has_no_status_never_a_pass() -> None:
    data = _input("r3").model_dump(mode="json", by_alias=True, exclude_none=True)
    data["criteria"] = []
    report_input = ReportInput.model_validate(data)
    arms = [config.arm for config in report_input.configs]
    key = next(key for key, measure in report_input.measures.items() if all(arm in measure.per_arm for arm in arms))

    lattice = _resolved(report_input, _spec("lattice", "configs", measures=[key]), "r3").data

    assert isinstance(lattice, LatticeChart)
    tiles = [cell for row in lattice.rows for cell in row.cells]
    assert len(tiles) == len(report_input.configs)
    assert all((cell.status, cell.status_label) == (None, None) for cell in tiles), "no pill where nothing is gated"
    assert not any(label in (cell.tip or []) for cell in tiles for label in GATE_WORDS.values())


# --- names are text -----------------------------------------------------------------------------------------------


def _hostile(report_input: ReportInput) -> ReportInput:
    """The input with markup in every name these charts print: arms, tasks, factor levels, criteria, measures."""
    data = report_input.model_dump(by_alias=True, exclude_none=True)
    for index, arm in enumerate(data["arms"]):
        arm.update(name=f"{HOSTILE_LINE} {index}")
        arm.pop("short_name", None)
    for item in data.get("tasks", []):
        item["name"] = HOSTILE_LINE
    for factor in data.get("factors", []):
        for index, level in enumerate(factor["levels"]):
            level["name"] = f"{HOSTILE_LINE} {index}"
    for criterion in data.get("criteria", []):
        criterion["label"] = HOSTILE_LINE
    for table in ("measures", "judged"):
        for figures in data[table].values():
            figures["label"] = HOSTILE_LINE
    return ReportInput.model_validate(data)


@pytest.mark.parametrize("name", ["r1", "r2", "r3"])
def test_markup_in_the_names_these_charts_print_is_text_on_the_page(name: str) -> None:
    plan = _plan(dataset.HERE / name / "anchor.plan.json")
    page = plan_page(plan, _hostile(_input(name)))
    drawn = {chart.chart.chart for chart in [*page.summary, *(section.chart for section in page.sections)]}
    assert drawn & _KINDS
    html = planned_page_html(page)
    assert what_runs(html) == []
    assert "<script>alert" not in html and "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert HOSTILE_LINE.split("<")[1] in planned_page_markdown(page)


@pytest.mark.parametrize(
    ("name", "measures"),
    [("r5", ["cost_usd", "delivered_items"]), ("r2", ["pass_rate", "calls_per_task"]), ("r5", [])],
    ids=["no-gate-declared", "one-gated-one-not", "no-gate-none-named"],
)
def test_a_measure_no_gate_reads_is_a_row_without_a_status_showing_each_arms_figure(
    name: str, measures: list[str]
) -> None:
    """An unordered set with no gate (a candidate selection the classifier calls A) still draws its gate matrix: a
    measure no gate reads states no status, only the figures, as a task no gate reads does."""
    report_input = _input(name)
    spec = _spec("gate_matrix", "measures", measures=measures)
    block, _ = _draw(name, spec)
    data = block.data
    assert isinstance(data, GateMatrixChart)
    gated = {criterion.measure for criterion in _gating(report_input)}
    criteria = [c for c in _gating(report_input) if not measures or c.measure in measures]
    shown = measures or (
        [] if criteria else list(dict.fromkeys(row.key for row in query_rows(spec.query, report_input)))
    )
    ungated = [key for key in shown if key not in gated]
    assert ungated, "the case reaches a measure no gate reads"
    assert len(data.rows) == len(criteria) + len(ungated)
    for row in data.rows[len(criteria) :]:
        assert all(cell.status == "neutral" and cell.label not in GATE_WORDS.values() for cell in row.cells)
        assert any(cell.reasons for cell in row.cells), "the row shows the arms' figures"
    for criterion, row in zip(criteria, data.rows, strict=False):
        assert row.label == (criterion.label or criterion.id)


@pytest.mark.parametrize(("name", "measures"), [("r5", []), ("r2", ["pass_rate", "calls_per_task"])])
def test_an_ungated_cell_says_not_gated_only_where_some_gate_is_declared(name: str, measures: list[str]) -> None:
    """A word on every cell saying nothing is gated is noise: an ungated cell carries none (and the kit draws no pill)
    where the input declares no gate or guardrail, and says it is not gated here where one is declared elsewhere."""
    report_input = _input(name)
    block, _ = _draw(name, _spec("gate_matrix", "measures", measures=measures))
    data = block.data
    assert isinstance(data, GateMatrixChart)
    ungated = [cell for row in data.rows for cell in row.cells if cell.status == "neutral"]
    assert ungated, "the case reaches a cell no gate reads"
    said = NOT_GATED if _gating(report_input) else ""
    assert all(cell.label == said for cell in ungated)
    _headers, rows = data.values_table()
    assert all(not text.startswith((":", ",")) for row in rows for text in row[1:])
