"""The kit contract: every value a package chart draws reaches the kit as decided text, which the kit prints as it is.

A page decides each figure's form once (``analysis/report/forms.py``); a chart that handed the kit a bare number and
a format would let the kit decide it a second time, on its own rule (``kShort``'s "as many decimals as it needs, up to
two"), so a chart could read ``85%`` beside its table's ``85.3%``. So every value a written page's chart draws carries
its text (``value_text``, ``lo_text``, ``hi_text`` and their kin), set by the page's forms
(:meth:`~threetears.evals.analysis.report.forms.PageForms.chart_texts`), and both drawers print it: the kit's
``kit.js`` (and its byte-identical copy under ``docs/eval-report-kit/kit/``, ``test_report_kit_pages``) and the static
page's ``chart_svg``. Only a tick label is still the kit's to write: a tick's position depends on the drawn width.

- **Every drawn value carries its text**, on every chart the stored plans draw on every fixture input.
- **Both drawers read every text field the models define** that names a value they draw.
- **Each drawer prints each value's own text in its own place**, per kind: the static page here, the kit in
  ``test_report_kit_charts`` (a browser, integration).

What this does not hold yet, and why: a gate matrix's and a lattice's costs and a heatmap cell's count still draw by
their formats (:data:`_DRAWN_BY_FORMAT`), and rung 5's preference strip carries no texts (the kit rounds its shares as
before), since only the policy flip (C5) moves what a reader sees; and the package's ``fmt`` still names no digits on some charts (``pct``, ``num``,
``ms``). Those digits now only write tick labels, and naming them moves the drawn ticks, which the policy flip (C5)
does in one step with every other form. ``scripts/check.py``'s render of the reference pages, which drives a browser,
holds the kit's drawing itself (integration).
"""

from __future__ import annotations

import functools
import html
import itertools
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from threetears.evals.analysis import ChartBlock
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report import charts as chart_models
from threetears.evals.analysis.report.chart_svg import DRAWERS, chart_drawing
from packages.evals.tests.fixtures.report_inputs import dataset
from packages.evals.tests.plan_charts_support import PLAN_CHARTS, StoredCharts, drawn_charts, stored_charts
from packages.evals.tests.report_kit_support import MARKED_CHARTS

_KIT = Path(chart_models.__file__).parent / "kit" / "kit.js"
_TWIN = Path(chart_models.__file__).parent / "chart_svg.py"

#: Each value field a drawer prints, to the field carrying its text, as the contract names them (JSON keys).
_TEXT_OF = {
    "value": "value_text",
    "lo": "lo_text",
    "hi": "hi_text",
    "values": "values_text",
    "x": "x_text",
    "y": "y_text",
    "from": "from_text",
    "to": "to_text",
    "cost": "cost_text",
    "latency": "latency_text",
    "low": "low_text",
    "high": "high_text",
    "detail": "detail_text",
    "matrix": "matrix_text",
}

#: Values only a chart's format names whether to draw: a detail or a latency the chart gives no format for is not
#: drawn, and needs no text.
_DRAWN_WITH = {"detail": "detail_fmt", "latency": "latency_fmt"}

#: Values the kit still draws by their format, each with why, held to today's drawing until the policy flip (C5).
_DRAWN_BY_FORMAT = {
    "cost": "a gate matrix's and a lattice's cost, drawn at `cost_fmt`'s digits ($5) where its table reads $4.60",
    "n": "a heatmap cell's count, a count's fixed form: drawn by `n_fmt` (1.2k), its tooltip and table grouped",
}

#: Text fields only the values table prints: a stacked bar's parts and a reference line's value are drawn as lengths
#: and lines, never as numbers, and a heatmap draws no cell's interval.
_TABLE_ONLY = {
    (chart_models.BarSegment, "value_text"),
    (chart_models.ChartReference, "value_text"),
    (chart_models.HeatmapCell, "low_text"),
    (chart_models.HeatmapCell, "high_text"),
}

#: The lines a chart draws at a value, each named by its label and never printed as a number: a reference (a floor, a
#: budget, a margin) and a price.
_LINES = frozenset({"reference", "x_reference", "price_line"})

#: The kinds the static page draws as a picture; it shows the rest as their tables.
_TWIN_DRAWS = (
    chart_models.BarRow,
    chart_models.ForestRow,
    chart_models.LineSeries,
    chart_models.ScatterPoint,
    chart_models.DotIntervalRow,
    chart_models.RangeRow,
    chart_models.HeatmapCell,
    chart_models.DiffHeatmapCell,
    chart_models.ConfusionChart,
    chart_models.ConfusionPanel,
)


@functools.cache
def _page_charts() -> list[tuple[str, ChartBlock]]:
    """Every chart the stored plans draw on every fixture input, by where it came from."""
    drawn = []
    for name, plan in itertools.product(
        ("r1", "r2", "r3", "r4", "r5"), sorted(p.stem for p in PLAN_CHARTS.glob("*.json"))
    ):
        report_input, stored = dataset.report_input_of(name), stored_charts(plan)
        for spec in stored.charts():
            one = StoredCharts(stored.slots, stored.order, stored.recommended, [spec], [])
            try:
                ((_, chart),) = drawn_charts(one, report_input)
            except UnresolvableReference:
                continue
            if isinstance(chart.chart, ChartBlock) and chart.chart.data is not None:
                drawn.append((f"{name}/{plan}/{spec.kind}", chart.chart))
    return drawn


def _untexted(node: Any, where: str, formats: dict[str, Any]) -> Iterator[str]:
    """Every value under ``node`` the kit draws with no text beside it, by its path."""
    if isinstance(node, list):
        for index, item in enumerate(node):
            yield from _untexted(item, f"{where}[{index}]", formats)
        return
    if not isinstance(node, dict):
        return
    formats = {**formats, **{key: value for key, value in node.items() if key.endswith("_fmt")}}
    for key, value in node.items():
        if key in _LINES:
            continue
        text = _TEXT_OF.get(key)
        drawn = (key not in _DRAWN_WITH or _DRAWN_WITH[key] in formats) and key not in _DRAWN_BY_FORMAT
        if text is not None and value is not None and drawn and node.get(text) is None:
            yield f"{where}.{key}"
        if key == "segments" and node.get("value_text") is None:
            yield f"{where}.value_text (a stacked bar's total)"
        if text is None:
            yield from _untexted(value, f"{where}.{key}", formats)


def test_the_fixtures_draw_charts_of_every_kind_a_page_resolves() -> None:
    kinds = {block.chart for _, block in _page_charts()}

    assert kinds >= {"bars", "diverging_bars", "forest", "lines", "dot_interval", "range_bars", "gate_scatter"}
    assert kinds >= {"gate_matrix", "lattice", "heatmap", "confusion"}


def test_every_value_a_page_s_chart_draws_reaches_the_kit_as_text() -> None:
    missing = [
        f"{where}: {path}"
        for where, block in _page_charts()
        for path in _untexted(block.kit_chart()[1], "data", {})  # type: ignore[index]
    ]

    assert not missing, "values drawn without their text:\n  " + "\n  ".join(missing)


def _text_fields() -> list[tuple[type[BaseModel], str]]:
    """Every text field a public chart model defines, with the model."""
    return sorted(
        (
            (model, field)
            for name in chart_models.__all__
            if isinstance(model := getattr(chart_models, name), type) and issubclass(model, BaseModel)
            for field in model.model_fields
            if field.endswith("_text")
        ),
        key=lambda pair: (pair[0].__name__, pair[1]),
    )


def _drawn_text_fields() -> list[tuple[type[BaseModel], str]]:
    """Every text field naming a value a drawer draws, with its model."""
    return [pair for pair in _text_fields() if pair not in _TABLE_ONLY]


def test_the_models_define_a_text_field_for_every_value_the_contract_names() -> None:
    fields = {field for _, field in _text_fields()}

    assert fields == set(_TEXT_OF.values())


@pytest.mark.parametrize(("model", "field"), _drawn_text_fields())
def test_both_drawers_read_every_text_field_naming_a_value_they_draw(model: type[BaseModel], field: str) -> None:
    assert field in _KIT.read_text(encoding="utf-8"), f"kit.js never reads {model.__name__}.{field}"
    if issubclass(model, _TWIN_DRAWS):
        assert field in _TWIN.read_text(encoding="utf-8"), f"chart_svg never reads {model.__name__}.{field}"


def _drawn(svg: str) -> tuple[list[tuple[str, ...]], list[str]]:
    """A static drawing's row and cell tooltips (their lines; a tooltip of one line, a heading's, left out), each
    once in drawn order across the drawing's widths, and every text it sets, in drawn order."""
    titles = [html.unescape(title) for title in re.findall(r"<title>([^<]*)</title>", svg)]
    tips = list(dict.fromkeys(tuple(title.split("\n")) for title in titles if "\n" in title))
    return tips, [html.unescape(text) for text in re.findall(r">([^<>]+)</(?:tspan|text)>", svg)]


def test_every_kind_the_static_page_draws_has_a_marked_chart() -> None:
    """Each kind the static page draws (its drawers' own registry, :data:`~threetears.evals.analysis.report.chart_svg.
    DRAWERS`, which :func:`~threetears.evals.analysis.report.chart_svg.chart_drawing` draws from) has a marked chart of
    its own (:data:`MARKED_CHARTS`), and every marked chart is of a kind it draws. So a kind is never left out of the
    per-kind check below because no fixture's page happens to draw it: a new drawer fails here until its markers are
    written."""
    marked = {one.kind for one in MARKED_CHARTS.values()}

    assert set(DRAWERS) - marked == set(), (
        f"drawn by the static page with no marked chart: {sorted(set(DRAWERS) - marked)}"
    )
    assert marked - set(DRAWERS) == set(), f"marked, but no kind the static page draws: {sorted(marked - set(DRAWERS))}"


@pytest.mark.parametrize("name", list(MARKED_CHARTS))
def test_the_static_page_draws_each_value_s_own_text_in_its_own_place(name: str) -> None:
    """Every value of a chart of each kind the static page draws carries a marker of its own: each tooltip reads its
    value's, then its interval's ends' markers, each in its place, and each value drawn as text of its own reads its
    marker, so a low end's text drawn as the high end's, or a value's drawn by its format, fails."""
    marked = MARKED_CHARTS[name]
    block = ChartBlock(section="memo", chart=marked.kind, title="t", data=marked.data)  # type: ignore[arg-type]

    drawing = chart_drawing(block)

    assert drawing is not None, f"the static page draws no {marked.kind}"
    tips, texts = _drawn(drawing)
    assert tips == list(marked.tips)
    found = [texts.index(text) for text in marked.texts if text in texts]
    assert len(found) == len(marked.texts) and found == sorted(found), (marked.texts, texts)
