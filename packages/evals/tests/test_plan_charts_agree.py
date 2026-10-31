"""The plan checks and the planned page agree on what a chart draws: one unit per axis, and every chart the checks
accept draws, each figure its key's in that key's unit.

- **One unit per axis.** On r1's tasks, which hold rates, a score, seconds and money per 1,000 replies, a chart over
  measures in several units draws each measure in its own panel, in its own unit, where its kind panels by measure (a
  heatmap's matrix, a dot strip's strip, bars in panels by task); every other one (a strip per arm, bars on one axis)
  is refused by the plan checks and by the page alike, and so is a chart of tasks naming no measure where its kind
  draws only one. A chart in one unit shares one format and span across its panels.
- **The sweep.** On every r1–r5 input, and each with a control where it has none (its recommended arm, so its anchor
  plan holds on it) or without the one it has, and on r1 with a control and one of its per-task rates stated as a
  share beside the others' percents, every chart over the vocabulary (each kind, each source it reads; no
  measure, each measure, and a pair of measures for each pair of units; unpanelled, or in panels by arm or by task)
  that the plan checks accept on that input draws on that input. Each figure a chart of figures draws is its key's,
  read through the figure keys, in that key's unit, and every axis's format is that key's. The checks include the
  page's own drawing (:mod:`~threetears.evals.analysis.report.plan_page.drawable`), so none is refused at render.
- **The floor.** That inclusion makes "nothing accepted is refused" hold by construction, so the (kind, source) pairs
  each input's charts were accepted and drawn on are pinned (``fixtures/report_plans/sweep_floor.json``): a pair the
  page stops drawing fails here rather than quietly leaving the accepted set.
- **Over figures by measure, at both ends.** A dot strip over measures in two units is accepted and drawn a strip per
  unit; bars and diverging bars over them are refused by the checks and the page alike; a share and a percent are one
  unit, drawn on one axis, each figure at its own scale, on every kind that draws several measures on one axis (bars,
  a difference heatmap, a strip per arm, range bars).
- **A contrast's unit** is one reading at both ends: a rate's change in points, an ordinal dimension's a score
  change, so a forest mixing the two is refused by the checks and the page alike. A stored axis span on a chart drawn a panel per unit, and bars in two units that fall into
  one panel, are refused by the page.

The expected side is read from the input: the figure keys (:func:`~threetears.evals.analysis.report.figure_keys.resolve`)
and each figure's unit as the page states it, never typed in.
"""

from __future__ import annotations

import functools
import itertools
import json
import re
from collections.abc import Iterator, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, get_args

import pytest

from packages.evals.tests.fixtures.report_plans import dataset
from packages.evals.tests.report_plan_support import planned_reply
from threetears.evals.analysis import ChartBlock, TableBlock
from threetears.evals.analysis.errors import PlanRefusal, UnresolvableReference
from threetears.evals.analysis.report.charts import (
    BarsChart,
    DiffHeatmapChart,
    DotIntervalChart,
    ForestChart,
    HeatmapChart,
    LinesChart,
    RangeBarsChart,
    ScatterChart,
)
from threetears.evals.analysis.report.figure_keys import resolve
from threetears.evals.analysis.report.plan_page.charting import contrast_unit, figure_unit
from threetears.evals.analysis.report.plan_page.drawable import plan_look
from threetears.evals.analysis.report.plan_page.look import PageLook, quantity
from threetears.evals.analysis.report.plan_page.resolve import resolve_chart
from threetears.evals.analysis.report_decision import control_arm
from threetears.evals.analysis.report_input import Cell, ReportInput, ScoreboardRow, Stat, Stratum, Task
from threetears.evals.analysis.report_plan_checks import raised_warnings
from threetears.evals.analysis.report_plan_facts import complete_plan, plan_facts
from threetears.evals.analysis.report_rows import query_rows
from threetears.evals.kernel.report_plan import ChartQuery, ChartSpec, PlannedReport, ReportPlan

INPUTS = ("r1", "r2", "r3", "r4", "r5")

_KINDS: tuple[str, ...] = get_args(ChartSpec.model_fields["kind"].annotation)
_SOURCES: tuple[str, ...] = get_args(ChartQuery.model_fields["source"].annotation)
_PANELS = ("none", "arm", "task")


# --- inputs, and the anchor plan each is probed beside -------------------------------------------------------------


def _anchor_json(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((dataset.HERE / name / "anchor.plan.json").read_text(encoding="utf-8"))
    return loaded


def _recommended(name: str) -> str | None:
    """The arm the anchor recommends; None where it recommends none (a trade-off)."""
    rows = _anchor_json(name)["options"]["rows"]
    return next((row["option"] for row in rows if row["role"] == "recommended"), None)


def _with_control(report_input: ReportInput, control: str | None) -> ReportInput:
    """The input with ``control`` its control and every other arm a candidate (no control for None)."""
    data = report_input.model_dump(by_alias=True, exclude_none=True)
    for arm in data["arms"]:
        arm["role"] = "control" if arm["id"] == control else "challenger"
    return ReportInput.model_validate(data)


def _base(key: str) -> str:
    return key.partition("~")[0].partition("+")[0].partition("-")[0]


#: Each input's rate restated in another scale (:func:`_restated`), by the input: on r1, a judged dimension held per
#: task beside others in percent; on r4, a measure beside another in percent; on r5, a judged dimension held per
#: stratum beside another as a share.
_RESTATED = {"r1": ("resolves", 100.0), "r4": ("specul", 100.0), "r5": ("persona.holds_character", 0.01)}
#: Each input's rate held beside the restated one, in the scale the restated one left.
_BESIDE = {"r1": "steps", "r4": "task", "r5": "persona.booth_stays_offstage"}


def _restated(data: dict[str, Any], key: str, by: float) -> dict[str, Any]:
    """The input with ``key``, a rate, stated on another scale: every figure held for it (the measure's or judged
    dimension's, overall and by aspect, and each task's, stratum's and cell's) divided by ``by``, its criteria's
    thresholds too, and a measure's unit left to be read off them."""

    def scaled(stat: dict[str, Any]) -> dict[str, Any]:
        out = dict(stat)
        for name in ("mean", "p50", "p95", "sem", "half_width"):
            if isinstance(out.get(name), int | float):
                out[name] = out[name] / by
        if isinstance(out.get("ci"), list):
            out["ci"] = [end / by for end in out["ci"]]
        return out

    tables = [table[key] for table in (data["measures"], data.get("judged", {})) if key in table]

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if any(node is table for table in tables):
                held = dict(node)
                held["per_arm"] = {arm: scaled(stat) for arm, stat in node["per_arm"].items()}
                if "by_aspect" in node:
                    held["by_aspect"] = {
                        aspect: {arm: scaled(stat) for arm, stat in figures.items()}
                        for aspect, figures in node["by_aspect"].items()
                    }
                return held
            return {
                name: scaled(value) if name == key and isinstance(value, dict) and "mean" in value else walk(value)
                for name, value in node.items()
            }
        if isinstance(node, list):
            return [walk(item) for item in node]
        return node

    shared = walk(data)
    if key in shared["measures"]:
        shared["measures"][key]["unit"] = None
    for criterion in shared["criteria"]:
        if criterion.get("measure") == key and isinstance(criterion.get("threshold"), int | float):
            criterion["threshold"] /= by
    return shared


def _spread(data: dict[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    """The input with each of ``keys``' figures holding a p50 at its mean and a p95 above it, within its rate's
    scale (as range bars read them): each figure a mean holds, the measure's and each task's."""

    def spread(stat: dict[str, Any]) -> dict[str, Any]:
        mean = stat.get("mean")
        if not isinstance(mean, int | float):
            return stat
        return stat | {"p50": mean, "p95": min(mean * 1.05, 100.0)}

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            return {
                name: {arm: spread(stat) for arm, stat in value.items()}
                if name == "per_arm" and any(node is table.get(key) for table in tables for key in keys)
                else (spread(value) if name in keys and isinstance(value, dict) and "mean" in value else walk(value))
                for name, value in node.items()
            }
        if isinstance(node, list):
            return [walk(item) for item in node]
        return node

    tables = [data["measures"], data.get("judged", {})]
    spread_out: dict[str, Any] = walk(data)
    return spread_out


@functools.cache
def _input(key: str) -> ReportInput:
    """An input by key: ``rN`` as it is; ``rN+control`` with the arm its anchor recommends made the control (so the
    anchor's plan still holds), ``rN-control`` with its control made a candidate; then each change named after a
    ``~``, in order: ``~spread`` gives its two rates (:data:`_RESTATED`, :data:`_BESIDE`) a p50 and a p95, ``~restated``
    states the first on another scale (:func:`_restated`)."""
    base, *changes = key.split("~")
    name = _base(base)
    report_input = dataset.report_input_of(name)
    if base.endswith("+control"):
        recommended = _recommended(name)
        assert recommended is not None, f"{name}'s anchor recommends no arm to make the control"
        report_input = _with_control(report_input, recommended)
    elif base.endswith("-control"):
        report_input = _with_control(report_input, None)
    for change in changes:
        data = report_input.model_dump(by_alias=True, exclude_none=True)
        restated, by = _RESTATED[name]
        if change == "spread":
            data = _spread(data, [restated, _BESIDE[name]])
        else:
            assert change == "restated", change
            data = _restated(data, restated, by)
        report_input = ReportInput.model_validate(data)
    return report_input


def _variants() -> list[str]:
    found = []
    for name in INPUTS:
        found.append(name)
        found.append(f"{name}-control" if control_arm(dataset.report_input_of(name)) is not None else f"{name}+control")
    # A rate stated as a share beside rates in percent, per task, with a control: charts of tasks reading both draw
    # each at its own scale on one axis (a difference heatmap, a strip per arm, bars), or the sweep sees it misdrawn.
    found.append("r1+control~restated")
    return found


_VARIANTS = _variants()


def _completes_on_itself(key: str) -> bool:
    """Whether the anchor's plan completes on the input: every input but one whose control was taken away, which its
    anchor's prose names. Such an input's probes are completed on the input itself (code's fill reads no arm's role)
    and judged and drawn on the variant."""
    return not key.endswith("-control")


@functools.cache
def _look(key: str) -> PageLook:
    """The look the anchor's page draws with on the input: the page's own (:func:`plan_look`)."""
    plan = dataset.load_plan(dataset.HERE / _base(key) / "anchor.plan.json")
    report_input = _input(key)
    return plan_look(plan, report_input, plan_facts(report_input))


# --- probing the checks ----------------------------------------------------------------------------------------------


def _query(source: str, **fields: Any) -> dict[str, Any]:
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
    return base | fields


def _chart(kind: str, source: str, **query: Any) -> dict[str, Any]:
    """A chart as the planner writes it: no line, nothing stressed; code fills its level, unit and axis."""
    return {"kind": kind, "title": "", "query": _query(source, **query), "reference": [], "emphasis": ""}


def _reply(name: str, charts: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], int]:
    """The anchor's reply with a probe section per chart after its own, in plain words that cite no figure; and how
    many sections the anchor has, where the probes start."""
    plan = _anchor_json(name)
    recommended = _recommended(name)
    raised = raised_warnings(dataset.report_input_of(name), recommended) if recommended is not None else []
    warnings = [warning.model_dump() for warning in raised]
    reply = planned_reply(plan, warnings)
    first = len(reply["sections"])
    template = deepcopy(reply["sections"][-1])
    template.update(eyebrow="Probe", headline="A probe chart", takeaway="", caption="", caveats=[])
    reply["sections"] += [template | {"id": f"probe{index}", "chart": chart} for index, chart in enumerate(charts)]
    return reply, first


def _probe_faults(key: str, charts: Sequence[dict[str, Any]]) -> dict[int, list[str]]:
    """What completing each chart on input ``key`` refuses, by the chart's index: the faults at its own chart."""
    reply, first = _reply(_base(key), charts)
    try:
        complete_plan(PlannedReport.model_validate(reply), _input(key))
    except PlanRefusal as refused:
        faults: dict[int, list[str]] = {}
        for path, why in refused.faults:
            for index in range(len(charts)):
                if path.startswith(f"sections[{first + index}].chart"):
                    faults.setdefault(index, []).append(why)
        return faults
    return {}


def _completed(key: str, charts: Sequence[dict[str, Any]]) -> list[ChartSpec]:
    """Each chart as code completes it, on an input its anchor's plan completes on."""
    name = key if _completes_on_itself(key) else _base(key)
    reply, first = _reply(_base(key), charts)
    plan: ReportPlan = complete_plan(PlannedReport.model_validate(reply), _input(name))
    return [section.chart for section in plan.sections[first:]]


def _draw(key: str, spec: ChartSpec) -> Any:
    return resolve_chart(spec, _input(key), _look(key), title="chart").chart


# --- what a chart draws, and what its figures are -------------------------------------------------------------------


@dataclass(frozen=True)
class _Axis:
    """One axis a chart draws figures along: its format, and the values on it."""

    fmt: str
    values: list[float]


def _axes(data: Any) -> list[_Axis]:
    """Every axis a chart's figures lie on, a panel's own where it overrides the chart's format."""
    if isinstance(data, HeatmapChart):
        parts: Sequence[Any] = data.panels or [data]
        return [
            _Axis(part.fmt or data.fmt or "", [cell.value for row in part.rows or [] for cell in row.cells])
            for part in parts
        ]
    if isinstance(data, DotIntervalChart | BarsChart):
        parts = data.panels or [data]
        return [
            _Axis(part.fmt or data.fmt or "", [row.value for row in part.rows or [] if row.value is not None])
            for part in parts
        ]
    if isinstance(data, RangeBarsChart):
        return [
            _Axis(data.fmt or "", [end for group in data.groups for row in group.rows for end in (row.from_, row.to)])
        ]
    if isinstance(data, LinesChart):
        return [_Axis(data.fmt, [value for series in data.series for value in series.values])]
    if isinstance(data, DiffHeatmapChart):
        return [
            _Axis(
                data.detail_fmt or "",
                [cell.detail for row in data.rows for cell in row.cells if cell.detail is not None],
            )
        ]
    if isinstance(data, ScatterChart):
        parts = data.panels or [data]
        axes = []
        for part in parts:
            points = part.points or []
            axes.append(_Axis(part.x_fmt or data.x_fmt or "", [point.x for point in points]))
            axes.append(_Axis(part.y_fmt or data.y_fmt or "", [point.y for point in points]))
        return axes
    return []


#: Each kind whose values are figures as the input holds them, over the sources it draws them from; every other kind
#: draws changes, statuses or examples, and is held to drawing alone.
_FIGURES_DRAWN: dict[str, frozenset[str]] = {
    "bars": frozenset({"measures", "judged", "configs", "strata", "tasks"}),
    "dot_interval": frozenset({"measures", "judged", "strata", "tasks"}),
    "heatmap": frozenset({"cells", "strata", "tasks"}),
    "diff_heatmap": frozenset({"cells", "strata", "tasks"}),
    "range_bars": frozenset({"measures", "tasks"}),
    "lines": frozenset({"strata", "tasks"}),
    "scatter_frontier": frozenset({"measures", "judged", "configs", "tasks"}),
    "gate_scatter": frozenset({"measures", "judged", "configs", "tasks"}),
}
#: The kinds whose every axis is formatted as its measure is (a scatter's axes may be logarithmic or a cost axis).
_FORMATTED_AS_ITS_MEASURE = frozenset({"bars", "dot_interval", "heatmap", "diff_heatmap", "range_bars", "lines"})

#: The statistics a figure is drawn at.
_STATISTICS = ("mean", "p50", "p95")


def _scale_factor(scale: str) -> float:
    """What a figure on a figure key's scale is multiplied by to be drawn: a share as a percent, money per reply per
    1,000 replies."""
    return {"share": 100.0, "usd": 1000.0}.get(scale, 1.0)


def _figure_keys(report_input: ReportInput, key: str) -> Iterator[tuple[str, Stat]]:
    """Every figure key that reads one of ``key``'s figures, with the figure: an arm's on the measure or dimension,
    and within each task."""
    if key in report_input.measures:
        for arm, stat in report_input.measures[key].per_arm.items():
            yield from (
                (f"measure.{key}.{arm}.{name}", stat) for name in _STATISTICS if getattr(stat, name) is not None
            )
    elif key in report_input.judged:
        for arm, stat in report_input.judged[key].per_arm.items():
            if stat.mean is not None:
                yield f"judged.{key}.{arm}", stat
    for task in report_input.tasks:
        for arm, figures in task.per_arm.items():
            if key in figures:
                stat = figures[key]
                yield from (
                    (f"task.{task.id}.{arm}.{key}.{name}", stat)
                    for name in _STATISTICS
                    if getattr(stat, name) is not None
                )


#: The places a figure is compared at: a drawn value is the figure it rounds to here.
_PLACES = 6


@functools.cache
def _figures(input_key: str, key: str) -> frozenset[float]:
    """Every figure the input holds on ``key``, as a chart draws it, rounded to :data:`_PLACES`: read through the
    figure keys and scaled by the figure's own scale; a stratum's or a cell's at the scale the figure keys read the key
    on, or, for a measure only cells hold (which no figure key reads), at its unit's as the page states it."""
    report_input = _input(input_key)
    drawn: set[float] = set()
    scales: set[str] = set()
    for path, _ in _figure_keys(report_input, key):
        figure = resolve(path, report_input)
        assert isinstance(figure.value, float), path
        scales.add(figure.scale)
        drawn.add(figure.value * _scale_factor(figure.scale))
    slices: list[Stratum | Cell] = [*report_input.strata, *report_input.cells.values()]
    held = [stat for item in slices for figures in item.per_arm.values() if (stat := figures.get(key)) is not None]
    if held:
        assert len(scales) <= 1, f"{key!r} is held in slices, and the figure keys read it on {sorted(scales)}"
        factor = _scale_factor(scales.pop()) if scales else quantity(figure_unit(report_input, key)).factor
        drawn |= {
            float(value) * factor for stat in held for name in _STATISTICS if (value := getattr(stat, name)) is not None
        }
    return frozenset(round(figure, _PLACES) for figure in drawn)


def _is_figure(value: float, figures: frozenset[float]) -> bool:
    return round(value, _PLACES) in figures


def _figure_faults(input_key: str, spec: ChartSpec, data: Any, keys: Sequence[str]) -> list[str]:
    """Where a chart of figures draws a value that is no figure of the measures it reads in that measure's unit, or
    an axis formatted otherwise than the measure whose figures lie on it."""
    report_input = _input(input_key)
    faults = []
    axes = _axes(data)
    for axis in axes:
        for value in axis.values:
            whose = [key for key in keys if _is_figure(value, _figures(input_key, key))]
            if not whose:
                faults.append(f"{value} is no figure of {list(keys)} in its unit")
            elif spec.kind in _FORMATTED_AS_ITS_MEASURE:
                # Money's format follows every amount the chart draws of the measure, on every panel (#993: one form
                # per chart), so the expectation reads them all.
                formats = {
                    quantity(figure_unit(report_input, key), _drawn_of(input_key, key, axes)).fmt for key in whose
                }
                if axis.fmt not in formats:
                    faults.append(f"{value} is a figure of {whose}, drawn as {axis.fmt!r}, not {sorted(formats)}")
    return faults


def _drawn_of(input_key: str, key: str, axes: Sequence[_Axis]) -> list[float]:
    """Every value a chart draws, on any of its axes, that is a figure of ``key``."""
    return [value for axis in axes for value in axis.values if _is_figure(value, _figures(input_key, key))]


def _drew(spec: ChartSpec, block: Any) -> bool:
    if spec.kind == "table":
        return isinstance(block, TableBlock) and bool(block.rows)
    return isinstance(block, ChartBlock) and block.chart == spec.kind and bool(block.values_table()[1])


# --- one unit per axis, on r1's tasks -------------------------------------------------------------------------------


_TWO_UNITS = ["latency_s", "cost_per_1k"]


def test_r1s_tasks_hold_measures_in_several_units_and_the_two_named_differ() -> None:
    report_input = _input("r1")
    held = {key for task in report_input.tasks for figures in task.per_arm.values() for key in figures}
    assert len({figure_unit(report_input, key) for key in held}) > 2
    assert len({figure_unit(report_input, key) for key in _TWO_UNITS}) == 2


def _accepted_and_drawn(kind: str, **query: Any) -> Any:
    chart = _chart(kind, "tasks", **query)
    assert _probe_faults("r1", [chart]) == {}, "the plan checks accept it"
    (spec,) = _completed("r1", [chart])
    block = _draw("r1", spec)
    assert _drew(spec, block)
    assert isinstance(block, ChartBlock)
    return spec, block.data


def _keys_of_each_panel(data: Any, keys: Sequence[str], input_key: str = "r1") -> list[str]:
    """The one measure whose figures each panel draws, every value its figure in its unit, formatted as it is."""
    report_input = _input(input_key)
    found = []
    for axis in _axes(data):
        assert axis.values, "each panel draws values"
        whose = [key for key in keys if all(_is_figure(value, _figures(input_key, key)) for value in axis.values)]
        assert len(whose) == 1, f"one measure's figures in a panel, and {whose} hold them"
        assert (
            axis.fmt == quantity(figure_unit(report_input, whose[0]), _drawn_of(input_key, whose[0], _axes(data))).fmt
        )
        found.append(whose[0])
    return found


@pytest.mark.parametrize("measures", [[], _TWO_UNITS], ids=["every-measure", "two-units"])
def test_a_heatmap_of_tasks_in_several_units_draws_each_measure_in_its_own_panel_and_unit(measures: list[str]) -> None:
    report_input = _input("r1")
    held = list(
        dict.fromkeys(key for task in report_input.tasks for figures in task.per_arm.values() for key in figures)
    )
    keys = measures or held
    spec, data = _accepted_and_drawn("heatmap", measures=measures)
    assert isinstance(data, HeatmapChart) and data.panels is not None
    drawn = _keys_of_each_panel(data, keys)
    assert len({figure_unit(report_input, key) for key in drawn}) >= 2, "panels in more than one unit"
    assert all(panel.fmt is not None and panel.domain is not None for panel in data.panels)
    assert not _figure_faults("r1", spec, data, keys)


def test_a_dot_strip_of_tasks_in_two_units_draws_each_task_and_measure_in_its_own_panel_and_unit() -> None:
    spec, data = _accepted_and_drawn("dot_interval", measures=_TWO_UNITS)
    assert isinstance(data, DotIntervalChart) and data.panels is not None
    assert set(_keys_of_each_panel(data, _TWO_UNITS)) == set(_TWO_UNITS)
    assert not _figure_faults("r1", spec, data, _TWO_UNITS)


def test_bars_of_tasks_in_two_units_in_panels_by_task_draw_each_panel_in_its_measures_unit() -> None:
    spec, data = _accepted_and_drawn("bars", measures=_TWO_UNITS, small_multiples_by="task")
    assert isinstance(data, BarsChart) and data.panels is not None
    assert set(_keys_of_each_panel(data, _TWO_UNITS)) == set(_TWO_UNITS)
    assert not _figure_faults("r1", spec, data, _TWO_UNITS)


@pytest.mark.parametrize(
    ("kind", "query", "why"),
    [
        ("dot_interval", {"measures": []}, "names no measure"),
        ("bars", {"measures": []}, "names no measure"),
        ("dot_interval", {"measures": _TWO_UNITS, "small_multiples_by": "arm"}, "more than one unit"),
        ("bars", {"measures": _TWO_UNITS}, "more than one unit"),
        ("bars", {"measures": _TWO_UNITS, "small_multiples_by": "arm"}, "more than one unit"),
    ],
    ids=["strip-every-measure", "bars-every-measure", "strip-by-arm", "bars-one-axis", "bars-by-arm"],
)
def test_a_chart_of_tasks_the_page_cannot_draw_in_its_units_is_refused_by_checks_and_page_alike(
    kind: str, query: dict[str, Any], why: str
) -> None:
    chart = _chart(kind, "tasks", **query)
    faults = _probe_faults("r1", [chart]).get(0, [])
    assert any(why in fault for fault in faults), faults
    # The page refuses the same chart as the checks would have stored it: the one-unit rule reads no code-filled field.
    planned = ChartSpec.model_validate(chart | {"level": "", "unit": "", "domain": []})
    with pytest.raises(UnresolvableReference, match=why):
        _draw("r1", planned)


def test_a_chart_of_tasks_in_one_unit_shares_one_format_and_span_across_its_panels() -> None:
    report_input = _input("r1")
    held = dict.fromkeys(key for task in report_input.tasks for figures in task.per_arm.values() for key in figures)
    rates = [key for key in held if figure_unit(report_input, key) == "percent"][:2]
    assert len(rates) == 2
    spec, data = _accepted_and_drawn("heatmap", measures=rates)
    assert isinstance(data, HeatmapChart) and data.panels is not None
    assert all(panel.fmt is None and panel.domain is None for panel in data.panels)
    assert data.fmt == quantity("percent").fmt
    assert not _figure_faults("r1", spec, data, rates)


# --- one rule at both ends, over figures by measure -----------------------------------------------------------------


def _accepted_on(name: str, report_input: ReportInput, chart: dict[str, Any]) -> ChartSpec:
    """The chart as completion stores it on ``report_input``, beside ``name``'s anchor; refused if completion refuses
    it anywhere."""
    reply, first = _reply(name, [chart])
    return complete_plan(PlannedReport.model_validate(reply), report_input).sections[first].chart


def _refused_on(name: str, report_input: ReportInput, chart: dict[str, Any]) -> list[str]:
    """Why completion refuses the chart on ``report_input``, at the chart's own field."""
    reply, first = _reply(name, [chart])
    with pytest.raises(PlanRefusal) as refused:
        complete_plan(PlannedReport.model_validate(reply), report_input)
    return [why for path, why in refused.value.faults if path.startswith(f"sections[{first}].chart")]


def _drawn_on(name: str, report_input: ReportInput, spec: ChartSpec) -> Any:
    plan = dataset.load_plan(dataset.HERE / name / "anchor.plan.json")
    look = plan_look(plan, report_input, plan_facts(report_input))
    return resolve_chart(spec, report_input, look, title="chart").chart


def _stored(chart: dict[str, Any], **code: Any) -> ChartSpec:
    """A chart as a stored plan could hold it, with code's fields as given (none by default)."""
    return ChartSpec.model_validate(chart | {"level": "", "unit": "", "domain": []} | code)


def test_r4s_overall_measures_hold_figures_in_two_units() -> None:
    report_input = _input("r4")
    assert len({figure_unit(report_input, key) for key in _TWO_UNITS}) == 2
    assert all(report_input.measures[key].per_arm for key in _TWO_UNITS)


def test_a_dot_strip_over_measures_in_two_units_is_accepted_and_drawn_a_panel_per_unit() -> None:
    """The checks once refused any chart over measures in two units; the page draws a dot strip's strips each in its
    measure's unit, and the checks now hold it by the page's rule alone."""
    chart = _chart("dot_interval", "measures", measures=_TWO_UNITS)
    assert _probe_faults("r4", [chart]) == {}
    (spec,) = _completed("r4", [chart])
    assert spec.unit == "", "rows in two units state none"
    block = _draw("r4", spec)
    assert isinstance(block, ChartBlock) and isinstance(block.data, DotIntervalChart)
    assert block.data.panels is not None
    assert set(_keys_of_each_panel(block.data, _TWO_UNITS, "r4")) == set(_TWO_UNITS)
    assert not _figure_faults("r4", spec, block.data, _TWO_UNITS)


@pytest.mark.parametrize("kind", ["bars", "diverging_bars"])
def test_a_single_scale_chart_over_measures_in_two_units_is_refused_by_checks_and_page_alike(kind: str) -> None:
    chart = _chart(kind, "measures", measures=_TWO_UNITS)
    faults = _probe_faults("r4", [chart]).get(0, [])
    assert any("more than one unit" in fault for fault in faults), faults
    with pytest.raises(UnresolvableReference, match="more than one unit"):
        _draw("r4", _stored(chart))


def test_a_share_and_a_percent_are_one_unit_drawn_on_one_axis_in_percent() -> None:
    """r4's two rates, one stated as a share: both draw as a percent, so bars over them share one axis, each figure
    at its own scale."""
    report_input = _input("r4~restated")
    assert (figure_unit(report_input, "task"), figure_unit(report_input, "specul")) == ("percent", "share")
    chart = _chart("bars", "measures", measures=["task", "specul"])
    block = _drawn_on("r4", report_input, _accepted_on("r4", report_input, chart))
    assert isinstance(block, ChartBlock) and isinstance(block.data, BarsChart)
    data_drawn = block.data
    assert data_drawn.panels is None and data_drawn.fmt == quantity("percent").fmt
    drawn = {round(row.value, _PLACES) for row in data_drawn.rows or [] if row.value is not None}
    held = _input("r4").measures
    expected = {round(float(stat.mean), _PLACES) for key in ("task", "specul") for stat in held[key].per_arm.values()}
    assert drawn == expected, "each rate drawn in percent, the share at a hundred times its figure"


def test_bars_over_scoreboard_figures_in_two_units_are_refused_by_checks_and_page_alike() -> None:
    """A scoreboard figure's unit is the one rule's too: a reply's cost beside what the trials cost in all is two
    units on one axis, and two counts are one."""
    chart = _chart("bars", "scoreboard", measures=["reply_cost_usd", "test_cost_usd"])
    faults = _probe_faults("r5", [chart]).get(0, [])
    assert any("more than one unit" in fault for fault in faults), faults
    with pytest.raises(UnresolvableReference, match="more than one unit"):
        _draw("r5", _stored(chart))
    counts = _chart("bars", "scoreboard", measures=["ran", "usable"])
    assert _probe_faults("r5", [counts]) == {}


# --- a share beside a percent, on one axis: each figure at its own scale ---------------------------------------------


def _alike_pair(input_key: str) -> list[str]:
    """The restated rate and the one beside it (:data:`_RESTATED`, :data:`_BESIDE`), in units drawn alike that are not
    one unit: a share and a percent."""
    name = _base(input_key)
    keys = [_RESTATED[name][0], _BESIDE[name]]
    report_input = _input(input_key)
    assert {figure_unit(report_input, key) for key in keys} == {"share", "percent"}
    return keys


def _accepted_spec(input_key: str, chart: dict[str, Any]) -> ChartSpec:
    assert _probe_faults(input_key, [chart]) == {}, "the plan checks accept it"
    (spec,) = _completed(input_key, [chart])
    return spec


def _holding(input_key: str, source: str, keys: Sequence[str]) -> list[str]:
    """The slices some arm holds every one of ``keys`` in (on r1, the one task judging both), by id."""
    rows = query_rows(ChartQuery.model_validate(_query(source)), _input(input_key))
    return [
        row.item.id
        for row in rows
        if isinstance(row.item, Stratum | Task)
        and all(any(key in figures for figures in row.item.per_arm.values()) for key in keys)
    ]


def _own_figures(input_key: str, key: str, values: Sequence[float]) -> list[float]:
    """The values that are ``key``'s figures in its own unit as the page draws it."""
    return [value for value in values if _is_figure(value, _figures(input_key, key))]


@pytest.mark.parametrize(
    ("input_key", "source"), [("r1+control~restated", "tasks"), ("r5~restated", "strata")], ids=["tasks", "strata"]
)
def test_a_difference_heatmap_of_a_share_beside_a_percent_draws_each_at_its_own_scale(
    input_key: str, source: str
) -> None:
    keys = _alike_pair(input_key)
    spec = _accepted_spec(
        input_key, _chart("diff_heatmap", source, measures=keys, strata=_holding(input_key, source, keys))
    )
    block = _draw(input_key, spec)
    assert isinstance(block, ChartBlock) and isinstance(block.data, DiffHeatmapChart)
    data = block.data
    assert data.detail_fmt == quantity("percent").fmt
    for row in data.rows:
        for key, cell in zip(keys, row.cells, strict=True):
            assert cell.detail is not None
            assert _own_figures(input_key, key, [cell.detail]), f"{cell.detail} is no figure of {key} in percent"
            base = cell.detail - cell.value
            assert _own_figures(input_key, key, [base]), f"{cell.detail} less {cell.value} is no figure of {key}"


@pytest.mark.parametrize(
    ("input_key", "source"), [("r1+control~restated", "tasks"), ("r5~restated", "strata")], ids=["tasks", "strata"]
)
def test_a_strip_per_arm_of_a_share_beside_a_percent_draws_each_at_its_own_scale(input_key: str, source: str) -> None:
    keys = _alike_pair(input_key)
    spec = _accepted_spec(input_key, _chart("dot_interval", source, measures=keys, small_multiples_by="arm"))
    block = _draw(input_key, spec)
    assert isinstance(block, ChartBlock) and isinstance(block.data, DotIntervalChart)
    data = block.data
    assert data.panels is not None and data.fmt == quantity("percent").fmt
    assert all(panel.fmt is None for panel in data.panels), "one quantity: every strip shares the chart's axis"
    drawn: dict[str, list[float]] = {key: [] for key in keys}
    for panel in data.panels:
        for row in panel.rows:
            assert row.id is not None and row.value is not None
            drawn[row.id.rpartition("/")[2]].append(row.value)
    for key, values in drawn.items():
        assert values, f"{key} is drawn"
        assert _own_figures(input_key, key, values) == values, f"{key}'s dots are its figures in percent"


@pytest.mark.parametrize(
    ("input_key", "source"),
    [("r4~spread~restated", "measures"), ("r1+control~spread~restated", "tasks")],
    ids=["measures", "tasks"],
)
def test_range_bars_of_a_share_beside_a_percent_draw_each_at_its_own_scale(input_key: str, source: str) -> None:
    keys = _alike_pair(input_key)
    spec = _accepted_spec(input_key, _chart("range_bars", source, measures=keys))
    block = _draw(input_key, spec)
    assert isinstance(block, ChartBlock) and isinstance(block.data, RangeBarsChart)
    data = block.data
    assert data.fmt == quantity("percent").fmt
    ends = [end for group in data.groups for row in group.rows for end in (row.from_, row.to)]
    owned = {key: _own_figures(input_key, key, ends) for key in keys}
    assert all(owned.values()), f"each measure is drawn: {owned}"
    assert all(any(end in values for values in owned.values()) for end in ends), "every end is a figure"
    assert not _figure_faults(input_key, spec, data, keys)


# --- a contrast's unit: one reading, at both ends ---------------------------------------------------------------------


def test_a_contrasts_unit_is_points_for_a_rates_change_and_a_score_for_an_ordinal_ones() -> None:
    report_input = _input("r5")
    units = {(contrast.measure, contrast_unit(report_input, contrast)) for contrast in report_input.contrasts}
    assert ("persona.holds_character", "points") in units, "a pass-fail judged dimension's change"
    assert ("passed", "points") in units, "how each trial ended"
    assert ("__outcome__", "score") in units, "an ordinal judged dimension's change is a score change"
    assert ("production_replicating_cost", "usd") in units


def test_a_forest_of_rates_contrasts_is_one_unit_stored_and_drawn_in_points() -> None:
    keys = ["persona.holds_character", "passed"]
    spec = _accepted_spec("r5", _chart("forest", "contrasts", measures=keys))
    assert spec.unit == "points"
    block = _draw("r5", spec)
    assert isinstance(block, ChartBlock) and isinstance(block.data, ForestChart)
    drawn = [row.value for row in block.data.rows]
    report_input = _input("r5")
    expected = sorted(
        round(float(contrast.delta) * 100, _PLACES)
        for contrast in report_input.contrasts
        if contrast.measure in keys and contrast.delta is not None
    )
    assert sorted(round(value, _PLACES) for value in drawn) == expected, "each change in points"


def test_a_forest_mixing_a_score_change_with_point_changes_is_refused_by_checks_and_page_alike() -> None:
    """An ordinal dimension's change (a score on its own scale) and a rate's (in points) are two quantities, and a
    forest's groups share one axis."""
    chart = _chart("forest", "contrasts", measures=["persona.holds_character", "__outcome__"])
    why = "more than one unit (['points', 'score'])"
    assert any(why in fault for fault in _probe_faults("r5", [chart]).get(0, []))
    with pytest.raises(UnresolvableReference, match=re.escape(why)):
        _draw("r5", _stored(chart))


# --- the refusals no generated plan reaches -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "by"), [("heatmap", "none"), ("dot_interval", "none"), ("bars", "task")], ids=["heatmap", "strip", "bars"]
)
def test_a_stored_axis_span_over_measures_in_several_units_is_refused_by_the_page(kind: str, by: str) -> None:
    """Code stores a span only for a forest of readings; a stored plan carrying one on a chart drawn a panel per unit
    cannot give every panel that one span."""
    chart = _chart(kind, "tasks", measures=_TWO_UNITS, small_multiples_by=by)
    assert _probe_faults("r1", [chart]) == {}, "drawn without a stored span"
    with pytest.raises(UnresolvableReference, match="its stored axis spans one unit"):
        _draw("r1", _stored(chart, domain=[0.0, 10.0]))


def test_bars_in_several_units_that_fall_into_one_panel_are_refused_at_both_ends() -> None:
    """Bars in panels by measure where two measures in different units share a reader name: one group, so one
    panel, so one axis for two units."""
    data = _input("r4").model_dump(by_alias=True, exclude_none=True)
    for key in _TWO_UNITS:
        data["measures"][key]["label"] = "Overall"
    report_input = ReportInput.model_validate(data)
    chart = _chart("bars", "measures", measures=_TWO_UNITS, small_multiples_by="aspect")
    why = "its measures are in more than one unit, and it draws them on one axis"
    with pytest.raises(UnresolvableReference, match=why):
        _drawn_on("r4", report_input, _stored(chart))
    assert any(why in fault for fault in _refused_on("r4", report_input, chart))


# --- the sweep ------------------------------------------------------------------------------------------------------


def _vocabulary(input_key: str, source: str) -> list[str]:
    """Every key a query over ``source`` could name on the input: what its rows read, and what slices, cells and
    scoreboards hold."""
    report_input = _input(input_key)
    keys: dict[str, None] = {}
    for row in query_rows(ChartQuery.model_validate(_query(source)), report_input):
        if row.key:
            keys[row.key] = None
        if isinstance(row.item, Stratum | Task | Cell):
            keys |= dict.fromkeys(key for figures in row.item.per_arm.values() for key in figures)
        if isinstance(row.item, ScoreboardRow):
            keys |= dict.fromkeys(field for field, value in row.item if value not in (None, {}))
    return list(keys)


def _measure_sets(input_key: str, keys: Sequence[str]) -> list[list[str]]:
    """No measure, each measure, and the first pair of measures for each pair of units."""
    report_input = _input(input_key)
    sets: list[list[str]] = [[], *([key] for key in keys)]
    pairs: set[frozenset[str]] = set()
    for one, other in itertools.combinations(keys, 2):
        units = frozenset({figure_unit(report_input, one), figure_unit(report_input, other)})
        if units not in pairs:
            pairs.add(units)
            sets.append([one, other])
    return sets


@dataclass(frozen=True)
class _Swept:
    """What the sweep found on one input: each accepted chart's completed spec, and what drawing it found wrong."""

    accepted: list[tuple[str, str, ChartSpec]]
    refused_by_page: dict[tuple[str, str], list[str]]
    misdrawn: dict[tuple[str, str], list[str]]


@functools.cache
def _sweep(input_key: str) -> _Swept:
    report_input = _input(input_key)
    probes: list[tuple[str, str, dict[str, Any], list[str]]] = []
    for kind, source in itertools.product(_KINDS, _SOURCES):
        if not query_rows(ChartQuery.model_validate(_query(source)), report_input):
            continue
        vocabulary = _vocabulary(input_key, source)
        for measures in _measure_sets(input_key, vocabulary):
            for by in _PANELS:
                probes.append(
                    (kind, source, _chart(kind, source, measures=measures, small_multiples_by=by), vocabulary)
                )
    refused = _probe_faults(input_key, [chart for _, _, chart, _ in probes])
    kept = [probe for index, probe in enumerate(probes) if index not in refused]
    if not _completes_on_itself(input_key):
        # Completed on the input with its control (code's fill reads no arm's role), so each must be accepted there.
        stranded = _probe_faults(_base(input_key), [chart for _, _, chart, _ in kept])
        assert not stranded, f"accepted on {input_key}, refused on {_base(input_key)}: {stranded}"
    specs = _completed(input_key, [chart for _, _, chart, _ in kept])
    accepted = []
    refused_by_page: dict[tuple[str, str], list[str]] = {}
    misdrawn: dict[tuple[str, str], list[str]] = {}
    for (kind, source, _, vocabulary), spec in zip(kept, specs, strict=True):
        accepted.append((kind, source, spec))
        try:
            block = _draw(input_key, spec)
        except UnresolvableReference as refusal:
            refused_by_page.setdefault((kind, source), []).append(str(refusal).partition("cannot be drawn: ")[2])
            continue
        if not _drew(spec, block):
            misdrawn.setdefault((kind, source), []).append(f"{spec.query.measures} draws nothing to read")
            continue
        if source in _FIGURES_DRAWN.get(kind, frozenset()) and isinstance(block, ChartBlock):
            keys = list(spec.query.measures) or vocabulary
            faults = _figure_faults(input_key, spec, block.data, keys)
            if faults:
                misdrawn.setdefault((kind, source), []).append(f"{spec.query.model_dump()}: {faults[:3]}")
    return _Swept(accepted=accepted, refused_by_page=refused_by_page, misdrawn=misdrawn)


# One case per input, each over every kind and source: the sweep is one probe of the plan checks over every chart on
# the input, so a case per (kind, source) made every process its cases landed in sweep the input again, and spread
# over the pool's batches they swept it hundreds of times. Every pair is still held, and a failure names each pair as
# its own case's id did (``<input>-<kind>-<source>``), all of them at once.


@pytest.mark.parametrize("input_key", _VARIANTS)
def test_every_chart_the_plan_checks_accept_draws_each_figure_in_its_keys_unit(input_key: str) -> None:
    swept = _sweep(input_key)
    faults: list[str] = []
    for kind, source in itertools.product(_KINDS, _SOURCES):
        where = f"{input_key}-{kind}-{source}"
        misdrawn = swept.misdrawn.get((kind, source), [])
        if misdrawn:
            faults.append(f"{where}: misdrawn: {misdrawn}")
        refused = swept.refused_by_page.get((kind, source), [])
        if refused:
            faults.append(f"{where}: refused by the page: {refused}")
    assert not faults, "\n".join(faults)


@pytest.mark.parametrize("input_key", _VARIANTS)
def test_the_sweep_reaches_charts_in_several_units_on_every_input_and_the_cases_it_names(input_key: str) -> None:
    assert _sweep(input_key).accepted, f"the checks accept some probe on {input_key}"
    if input_key != "r1":
        return
    mixed = [
        spec
        for kind, source, spec in _sweep("r1").accepted
        if (kind, source) == ("heatmap", "tasks") and not spec.query.measures
    ]
    assert mixed, "r1's every-measure heatmap of tasks, in several units, is among the charts drawn"


#: The floor under the sweep: each input's (kind, source) pairs, as ``kind/source``, that the checks accepted and the
#: page drew when it was pinned. Without it the sweep's "nothing accepted is refused" holds by construction (the checks
#: draw each chart through the page), and a page that came to refuse a chart it drew would only leave the accepted set.
_FLOOR = dataset.HERE / "sweep_floor.json"


@pytest.mark.parametrize("input_key", _VARIANTS)
def test_every_pair_the_floor_pins_is_still_accepted_and_drawn_on_its_input(input_key: str) -> None:
    pinned: dict[str, list[str]] = json.loads(_FLOOR.read_text(encoding="utf-8"))["pairs"]
    assert set(pinned) == set(_VARIANTS), "the floor pins every input the sweep probes, and only those"
    swept = _sweep(input_key)
    broken = set(swept.refused_by_page) | set(swept.misdrawn)
    drawn = {f"{kind}/{source}" for kind, source, _ in swept.accepted if (kind, source) not in broken}
    lost = {input_key: missing} if (missing := sorted(set(pinned[input_key]) - drawn)) else {}
    assert not lost, (
        f"pairs the checks accepted and the page drew are no longer accepted and drawn: {lost}. A deliberate "
        f"narrowing removes them from {_FLOOR.name} in the same change"
    )
