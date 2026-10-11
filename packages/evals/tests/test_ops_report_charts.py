"""``ops.report_charts``: every chart of a report drawn by the host's renderer, and one refusal per chart that is not.

A chart fails for reasons of its own — its stored chart could not be decided into an intent, or the renderer
refused the intent — so one failure must cost that chart alone: the others are still drawn, and the failed one
is returned as a refusal naming its block, never dropped. Only a renderer's refusal (``ChartRefused``) is one chart's
refusal: anything else it raises — a plain ``ValueError`` from its compiler included — is a defect, and propagates.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.evals.analysis.viz import ChartIntent, ChartRefused, ChartType, ChartValue, chart_intent
from threetears.evals.ops import ChartRefusal, DrawnChart, report_charts
from threetears.evals.vega import (
    CompiledChart,
    RasterRefused,
    SpecPolicyError,
    VegaRenderer,
    packaged_palette,
    render_svg,
)
from packages.evals.tests.chart_examples import EVERY_TYPE
from packages.evals.tests.report_support import minimal_report


def _chart(viz_type: str, *, finding: int | None = 0) -> dict[str, Any]:
    intent = chart_intent(viz_type, EVERY_TYPE[viz_type])  # type: ignore[arg-type]
    return {"kind": "chart", "section": "findings", "finding": finding, "viz_type": viz_type, "intent": intent}


def _undrawable(viz_type: str, error: str) -> dict[str, Any]:
    return {"kind": "chart", "section": "findings", "finding": 0, "viz_type": viz_type, "error": error}


_TITLE = {"kind": "text", "section": "findings", "role": "finding_title", "finding": 0, "body": "t"}


class _ScriptedRenderer:
    """Draws every intent as its type, except the types it was told to refuse (``ChartRefused``) or break on.

    It breaks on a type with a ``RuntimeError``, or with a plain ``ValueError`` where ``breaks_with_value_error``
    — a compiler defect that happens to be a ``ValueError``.
    """

    def __init__(
        self,
        *,
        refuses: frozenset[str] = frozenset(),
        breaks_on: frozenset[str] = frozenset(),
        breaks_with_value_error: bool = False,
    ) -> None:
        self.refuses = refuses
        self.breaks_on = breaks_on
        self.breaks_with_value_error = breaks_with_value_error
        self.drawn: list[ChartType] = []

    def draw(self, intent: ChartIntent) -> str:
        if intent.type in self.breaks_on:
            if self.breaks_with_value_error:
                raise ValueError("could not convert string to float: 'n/a'")
            raise RuntimeError("the renderer itself is broken")
        if intent.type in self.refuses:
            raise ChartRefused(f"a {intent.type} spec breaks rule 9")
        self.drawn.append(intent.type)
        return f"drawing of {intent.type}"

    def drawn_data(self, drawing: str) -> list[dict[str, ChartValue]]:
        return []


def test_every_chart_draws_with_the_packaged_vega_renderer():
    """End to end on the package's own adapter: every chart type a report carries compiles."""
    report = minimal_report(blocks=[_TITLE, *(_chart(viz_type) for viz_type in EVERY_TYPE)])
    charts = report_charts(report, VegaRenderer.packaged())
    assert charts.refused == ()
    assert [chart.viz_type for chart in charts.drawn] == list(EVERY_TYPE)
    assert [chart.block for chart in charts.drawn] == list(range(1, len(EVERY_TYPE) + 1))
    assert all(isinstance(chart.drawing, CompiledChart) for chart in charts.drawn)
    assert all(chart.drawing.intent.type == chart.viz_type for chart in charts.drawn)


def test_one_chart_the_renderer_refuses_costs_that_chart_alone():
    report = minimal_report(blocks=[_TITLE, _chart("breakdown"), _chart("frontier"), _chart("delta_table")])
    renderer = _ScriptedRenderer(refuses=frozenset({"frontier"}))
    charts = report_charts(report, renderer)
    assert charts.drawn == (
        DrawnChart(block=1, finding=0, viz_type="breakdown", drawing="drawing of breakdown"),
        DrawnChart(block=3, finding=0, viz_type="delta_table", drawing="drawing of delta_table"),
    )
    assert charts.refused == (
        ChartRefusal(
            block=2,
            finding=0,
            viz_type="frontier",
            cause="refused_by_renderer",
            reason="ChartRefused: a frontier spec breaks rule 9",
        ),
    )


def test_a_chart_the_report_could_not_decide_is_refused_with_its_own_reason_and_never_drawn():
    report = minimal_report(
        blocks=[_TITLE, _undrawable("frontier", "readings[1]: names no priced cell"), _chart("breakdown")]
    )
    renderer = _ScriptedRenderer()
    charts = report_charts(report, renderer)
    assert renderer.drawn == ["breakdown"]
    assert [chart.block for chart in charts.drawn] == [2]
    assert charts.refused == (
        ChartRefusal(
            block=1,
            finding=0,
            viz_type="frontier",
            cause="undrawable",
            reason="readings[1]: names no priced cell",
        ),
    )


def test_a_chart_code_chose_names_no_finding_drawn_or_refused():
    report = minimal_report(
        basis="code_only",
        headline="",
        finding_count=0,
        source={**minimal_report().source.model_dump(), "analysis_id": None, "generator_model": None},
        blocks=[_chart("breakdown", finding=None), _chart("frontier", finding=None)],
    )
    charts = report_charts(report, _ScriptedRenderer(refuses=frozenset({"frontier"})))
    assert [chart.finding for chart in charts.drawn] == [None]
    assert [refusal.finding for refusal in charts.refused] == [None]


def test_a_report_with_no_chart_draws_nothing_and_refuses_nothing():
    charts = report_charts(minimal_report(blocks=[_TITLE]), _ScriptedRenderer())
    assert (charts.drawn, charts.refused) == ((), ())


def test_a_renderer_that_breaks_is_not_reported_as_one_charts_refusal():
    report = minimal_report(blocks=[_TITLE, _chart("breakdown"), _chart("frontier")])
    with pytest.raises(RuntimeError, match="the renderer itself is broken"):
        report_charts(report, _ScriptedRenderer(breaks_on=frozenset({"frontier"})))


def test_a_compiler_defect_that_is_a_value_error_is_not_reported_as_one_charts_refusal():
    """Any ValueError read as a refusal: a compiler that broke on every chart read as a report of undrawable charts."""
    report = minimal_report(blocks=[_TITLE, _chart("breakdown"), _chart("frontier")])
    renderer = _ScriptedRenderer(breaks_on=frozenset({"frontier"}), breaks_with_value_error=True)
    with pytest.raises(ValueError, match="could not convert string to float") as raised:
        report_charts(report, renderer)
    assert not isinstance(raised.value, ChartRefused)


def test_the_vega_adapters_refusals_are_chart_refusals():
    assert issubclass(SpecPolicyError, ChartRefused) and issubclass(RasterRefused, ChartRefused)


def test_vl_converts_refusal_to_convert_is_a_raster_refusal_and_nothing_else_is():
    """vl-convert words its refusal as a failed conversion; a string it cannot parse is the caller's defect."""
    palette = packaged_palette("dark")
    with pytest.raises(RasterRefused, match="Vega-Lite to SVG conversion failed"):
        render_svg({"mark": 5}, palette=palette)
    with pytest.raises(ValueError, match="Failed to parse") as raised:
        render_svg({"mark": "bar", "data": {"values": [{"a": float("nan")}]}}, palette=palette)
    assert not isinstance(raised.value, ChartRefused)
