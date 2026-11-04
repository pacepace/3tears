"""One chart's rules, apart from any plan (``threetears.evals.analysis.report.block_spec``): what code completes a
chart with, and what refuses it on its input. ``block_render`` and the document page's evidence blocks run both.

Each case takes a chart a stored plan holds (``fixtures/plan_charts/``: the Kairo reference plan, and the r1 to r5
anchors), edits it as the case says, and reads it against its own input through :func:`complete_chart` (a chart as chosen, completed) or :func:`chart_faults` and
:func:`small_multiple_faults` (a completed chart, refused), each fault at the chart's own field path.

- **A reference line** sits at the asked aspect's margin, or at the one criterion that reads the chart's one measure
  where the chart reads (a task's own criterion where the chart keeps that task, the each-cell criterion on a chart
  of cells, the p95 one on range bars); a chart keeping every task, a criterion reading another statistic, a floor
  at a budget, two criteria, several measures, a chart of readings and a no-preference line are each refused, naming
  why. Every stored chart completes again with the lines it stores.
- **The interval level**: diverging bars over strata whose figures carry a standard error and no interval are drawn at
  the descriptive level, ``95% uncorrected``; bars over the same strata, and strata carrying an interval, draw none.
- **What refuses a chart**: a kind that cannot draw its source, a forest over rows with no interval, a measure, arm,
  aspect or stratum the input does not hold, a measure on a source that names none, a query selecting nothing, a
  reference key that does not resolve, an emphasis on nothing present; and small multiples of a kind the kit does
  not panel. A kind draws each source its resolver reads.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from packages.evals.tests.fixtures.report_plans import dataset
from packages.evals.tests.plan_charts_support import PLAN_CHARTS
from threetears.evals.analysis.report.block_spec import chart_faults, complete_chart, small_multiple_faults
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel.report_plan import ChartSpec, PlannedChart

ASPECT = "character adherence (v2)"

_STEP = re.compile(r"([^.\[\]]+)|\[(\d+)\]")

#: A field path's chart: ``summary[1]`` or ``sections[2].chart``, the rest the chart's own field.
_CHART = re.compile(r"^(summary\[\d+\]|sections\[\d+\]\.chart)")


def _plan(name: str = "r5-reference") -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((PLAN_CHARTS / f"{name}.json").read_text(encoding="utf-8"))
    return loaded


def _anchor(name: str) -> dict[str, Any]:
    return _plan(f"{name}-anchor")


def _data(name: str = "r5") -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((dataset.HERE / name / "input.json").read_text(encoding="utf-8"))
    return loaded


def _input(name: str = "r5", data: dict[str, Any] | None = None) -> ReportInput:
    return ReportInput.model_validate(data if data is not None else _data(name))


def _at(node: Any, path: str) -> tuple[Any, str | int]:
    """The container holding ``path``'s last step, and that step."""
    steps: list[str | int] = [name if name else int(index) for name, index in _STEP.findall(path)]
    for step in steps[:-1]:
        node = node[step]
    return node, steps[-1]


def _get(plan: dict[str, Any], path: str) -> Any:
    holder, step = _at(plan, path)
    return holder[step]


def _set(plan: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    holder, step = _at(plan, path)
    holder[step] = value
    return plan


# --- a completed chart's faults --------------------------------------------------------------------------------------


def _faults(
    plan: dict[str, Any], chart: str, name: str = "r5", data: dict[str, Any] | None = None
) -> list[tuple[str, str]]:
    """What refuses the stored chart at ``chart`` on its input: its own faults (an option of the plan's table
    stressable, as the page reads it), then its panels'."""
    spec = ChartSpec.model_validate(_get(plan, chart))
    options = [row["option"] for row in plan["options"]["rows"]]
    report_input = _input(name, data)
    return [*chart_faults(chart, spec, report_input, options=options), *small_multiple_faults(chart, spec)]


def _paths(plan: dict[str, Any], chart: str, name: str = "r5", data: dict[str, Any] | None = None) -> list[str]:
    return [path for path, _ in _faults(plan, chart, name, data)]


def _why(plan: dict[str, Any], path: str, data: dict[str, Any] | None = None) -> str:
    """Every fault at ``path`` on the Kairo input (or ``data``); the chart must be refused there."""
    match = _CHART.match(path)
    assert match is not None, path
    found = [why for at, why in _faults(plan, match.group(1), data=data) if at == path]
    assert found, f"no fault at {path}"
    return " ".join(found)


class TestCharts:
    def test_a_kind_that_cannot_draw_its_source_is_refused(self):
        assert "cannot draw 'measures'" in _why(_set(_plan(), "summary[1].kind", "forest"), "summary[1].kind")
        plan = _set(_plan(), "sections[1].chart.kind", "lattice")
        assert "sections[1].chart.kind" in _paths(plan, "sections[1].chart")

    def test_a_forest_over_rows_without_intervals_is_refused(self):
        data = _data()
        for contrast in data["contrasts"]:
            if contrast["measure"] == "passed":
                contrast.update(interval=None, verdict="untested")
        assert "carries no interval" in _why(_plan(), "summary[0].kind", data)

    def test_an_absent_measure_is_refused(self):
        plan = _plan()
        plan["summary"][0]["query"]["measures"].append("persona.booth_stays_offstage")
        assert "persona.booth_stays_offstage" in _why(plan, "summary[0].query.measures")

    def test_a_measure_on_a_source_that_names_none_is_refused(self):
        plan = _set(_plan(), "sections[0].chart.query.measures", ["passed"])
        assert "names none" in _why(plan, "sections[0].chart.query.measures")

    def test_an_absent_arm_is_refused(self):
        plan = _set(_plan(), "sections[2].chart.query.arms", ["qwen3.8-flash", "1018"])
        assert "1018" in _why(plan, "sections[2].chart.query.arms")

    def test_an_absent_aspect_is_refused(self):
        plan = _set(_plan(), "sections[0].chart.query.aspect", "character adherence")
        assert "sections[0].chart.query.aspect" in _paths(plan, "sections[0].chart")

    def test_an_absent_stratum_is_refused(self):
        plan = _set(_plan(), "sections[2].chart.query.strata", ["off_lane", "offlane"])
        assert "offlane" in _why(plan, "sections[2].chart.query.strata")

    def test_a_query_selecting_nothing_is_refused(self):
        plan = _set(_plan(), "sections[2].chart.query.arms", ["September"])
        assert "selects nothing" in _why(plan, "sections[2].chart.query")

    def test_an_unresolvable_reference_key_is_refused(self):
        plan = _set(_plan(), "sections[0].chart.reference[0].key", "aspect.floor")
        assert "sections[0].chart.reference[0].key" in _paths(plan, "sections[0].chart")

    def test_an_emphasis_on_nothing_present_is_refused(self):
        plan = _set(_plan(), "sections[0].chart.emphasis", "qwen")
        assert "sections[0].chart.emphasis" in _paths(plan, "sections[0].chart")

    @pytest.mark.parametrize(
        ("name", "path", "source", "kind"),
        [
            ("r1", "sections[2].chart", "tasks", "range_bars"),
            ("r5", "sections[1].chart", "strata", "dot_interval"),
            ("r2", "sections[1].chart", "configs", "diverging_bars"),
        ],
    )
    def test_a_kind_draws_each_source_its_resolver_reads(self, name: str, path: str, source: str, kind: str):
        plan = _anchor(name)
        assert _get(plan, f"{path}.query.source") == source
        assert f"{path}.kind" not in _paths(_set(plan, f"{path}.kind", kind), path, name)

    def test_a_gate_matrix_draws_readings(self):
        plan = _set(_plan(), "summary[0].kind", "gate_matrix")
        plan["summary"][0]["query"] |= {"source": "readings", "aspect": ASPECT, "measures": []}
        assert "summary[0].kind" not in _paths(plan, "summary[0]")

    def test_range_bars_still_draw_no_strata(self):
        plan = _set(_plan(), "sections[1].chart.kind", "range_bars")
        assert "cannot draw 'strata'" in _why(plan, "sections[1].chart.kind")


#: The kinds whose kit chart model carries ``panels`` (``kit/API.md``: bars, and diverging bars as bars, the two
#: scatters, dot intervals, the heatmap and confusion matrices).
PANELLED = {"bars", "diverging_bars", "scatter_frontier", "gate_scatter", "dot_interval", "heatmap", "confusion"}

#: Every other kind the cases repeat a panel of: the gate family's, the forest, lines, range bars, examples, a table.
_UNPANELLED = {"gate_matrix", "lattice", "forest", "lines", "range_bars", "examples", "table"}


class TestSmallMultiples:
    def test_small_multiples_of_a_forest_are_refused(self):
        plan = _set(_plan(), "summary[0].query.small_multiples_by", "arm")
        why = _why(plan, "summary[0].query.small_multiples_by")
        assert "no small multiples of a 'forest'" in why and "'none'" in why

    @pytest.mark.parametrize("kind", sorted(PANELLED | _UNPANELLED))
    def test_only_a_kind_the_kit_panels_repeats_its_panel(self, kind: str):
        plan = _set(_plan(), "summary[1].kind", kind)
        _set(plan, "summary[1].query.small_multiples_by", "arm")
        refused = "summary[1].query.small_multiples_by" in _paths(plan, "summary[1]")
        assert refused == (kind not in PANELLED), kind

    def test_no_small_multiples_is_never_refused(self):
        plan = _set(_plan(), "summary[0].query.small_multiples_by", "none")
        assert "summary[0].query.small_multiples_by" not in _paths(plan, "summary[0]")


# --- completing a chart: its reference line and its level --------------------------------------------------------


def _planned(stored: dict[str, Any]) -> PlannedChart:
    """A stored chart as it was chosen: what code fills (each line's key, the level, unit and axis) left out."""
    return PlannedChart.model_validate(
        {
            "kind": stored["kind"],
            "title": stored["title"],
            "query": stored["query"],
            "reference": [{"kind": line["kind"]} for line in stored["reference"]],
            "emphasis": stored["emphasis"],
        }
    )


def _completed(
    name: str, plan: dict[str, Any], path: str, data: dict[str, Any] | None = None
) -> tuple[ChartSpec, list[tuple[str, str]]]:
    """The chart at ``path`` completed on its input, and the faults completion found."""
    return complete_chart(_planned(_get(plan, path)), _input(name, data), path)


def _line(name: str, path: str, kind: str, **query: Any) -> dict[str, Any]:
    plan = _anchor(name)
    _set(plan, f"{path}.reference", [{"kind": kind, "key": "placeholder"}])
    _get(plan, f"{path}.query").update(query)
    return plan


def _key_at(name: str, plan: dict[str, Any], path: str) -> str:
    completed, faults = _completed(name, plan, path)
    assert not faults, faults
    (reference,) = completed.reference
    return str(reference.key)


def _line_fault(name: str, plan: dict[str, Any], path: str, data: dict[str, Any] | None = None) -> str:
    _, faults = _completed(name, plan, path, data)
    found = [why for at, why in faults if at == f"{path}.reference[0].kind"]
    assert found, faults
    return " ".join(found)


def _chart_paths(plan: dict[str, Any]) -> list[str]:
    return [
        *(f"summary[{index}]" for index in range(len(plan["summary"]))),
        *(f"sections[{index}].chart" for index in range(len(plan["sections"]))),
    ]


class TestReferenceLines:
    @pytest.mark.parametrize("name", ["r1", "r2", "r3", "r4", "r5"])
    def test_each_anchor_completes_with_the_reference_lines_it_stores(self, name: str):
        plan = _anchor(name)
        for path in _chart_paths(plan):
            completed, faults = _completed(name, plan, path)
            assert not faults, (path, faults)
            assert completed.reference == ChartSpec.model_validate(_get(plan, path)).reference, path

    def test_a_margin_line_sits_at_the_asked_aspects_margin(self):
        assert _key_at("r5", _anchor("r5"), "sections[0].chart") == "aspect.margin"

    def test_a_floor_sits_at_the_one_criterion_reading_the_charts_measure(self):
        assert _key_at("r2", _line("r2", "summary[0]", "floor"), "summary[0]") == "criteria.pass_rate.threshold"

    def test_a_floor_on_a_task_reads_the_task_criterion_where_the_chart_keeps_that_task(self):
        assert _get(_anchor("r1"), "sections[1].chart.query.strata") == ["specul"]
        key = _key_at("r1", _line("r1", "sections[1].chart", "floor"), "sections[1].chart")
        assert key == "criteria.specul.declines.threshold"

    def test_a_task_criterion_reads_nothing_on_a_chart_keeping_every_task(self):
        plan = _line("r1", "sections[1].chart", "floor", strata=[])
        assert "no criterion reads 'declines'" in _line_fault("r1", plan, "sections[1].chart")

    def test_a_budget_on_range_bars_sits_at_the_p95_criterion(self):
        plan = _line("r1", "sections[2].chart", "budget")
        _set(plan, "sections[2].chart.kind", "range_bars")
        _set(plan, "sections[2].chart.query.small_multiples_by", "none")
        assert _key_at("r1", plan, "sections[2].chart") == "criteria.p95.threshold"

    def test_a_budget_on_bars_of_the_mean_reads_no_p95_criterion(self):
        why = _line_fault("r1", _line("r1", "sections[2].chart", "budget"), "sections[2].chart")
        assert "no criterion reads 'latency_s'" in why and "['mean']" in why

    def test_a_floor_never_sits_at_a_budget(self):
        plan = _line("r1", "sections[2].chart", "floor")
        _set(plan, "sections[2].chart.kind", "range_bars")
        _set(plan, "sections[2].chart.query.small_multiples_by", "none")
        assert "higher is better" in _line_fault("r1", plan, "sections[2].chart")

    def test_a_floor_on_a_chart_of_cells_sits_at_the_each_cell_criterion(self):
        key = _key_at("r3", _line("r3", "sections[1].chart", "floor"), "sections[1].chart")
        assert key == "criteria.cell_floor.threshold"

    def test_two_criteria_reading_the_charts_measure_are_refused_naming_both(self):
        data = _data("r2")
        data["criteria"].append({**data["criteria"][0], "id": "pass_rate.again", "threshold": 95.0})
        why = _line_fault("r2", _line("r2", "summary[0]", "floor"), "summary[0]", data)
        assert "['pass_rate', 'pass_rate.again']" in why

    def test_a_chart_reading_several_measures_takes_no_floor(self):
        plan = _line("r2", "summary[0]", "floor", measures=["pass_rate", "outcome"])
        assert "the measures ['pass_rate', 'outcome']" in _line_fault("r2", plan, "summary[0]")

    def test_a_chart_of_readings_reads_no_measure_for_a_floor(self):
        plan = _anchor("r5")
        _set(plan, "sections[0].chart.reference", [{"kind": "floor", "key": "placeholder"}])
        assert "reads no measure" in _line_fault("r5", plan, "sections[0].chart")

    def test_a_no_preference_line_is_refused(self):
        plan = _line("r2", "summary[0]", "no_preference")
        assert "a margin, floor or budget line is what it can draw" in _line_fault("r2", plan, "summary[0]")


def _level_at(name: str, plan: dict[str, Any], path: str, data: dict[str, Any] | None = None) -> str:
    completed, faults = _completed(name, plan, path, data)
    assert not faults, faults
    return completed.level


class TestChartLevel:
    def test_diverging_strata_with_standard_errors_and_no_interval_are_drawn_at_the_descriptive_level(self):
        plan = _anchor("r5")
        assert (_get(plan, "sections[1].chart.kind"), _get(plan, "sections[1].chart.query.source")) == (
            "diverging_bars",
            "strata",
        )
        assert _level_at("r5", plan, "sections[1].chart") == "95% uncorrected"

    def test_bars_over_the_same_strata_draw_no_level(self):
        plan = _set(_anchor("r5"), "sections[1].chart.kind", "bars")
        assert _level_at("r5", plan, "sections[1].chart") == ""

    def test_strata_figures_carrying_an_interval_are_not_drawn_at_the_descriptive_level(self):
        data = _data()
        for stratum in data["strata"]:
            for figures in stratum["per_arm"].values():
                for stat in figures.values():
                    stat["ci"] = [0.0, 1.0]
        assert _level_at("r5", _anchor("r5"), "sections[1].chart", data) == ""
