"""Axes and labels that do not lie: every axis states its unit, its ticks fall where its scale puts them, no label is
dropped or cut, and no mark implies a magnitude the data does not have.

Held offline on the prototype-5 writer pages (Kairo, r2 with one gate, r3 without its configs) as their accepted
records draw them, on the code pages of the shape campaigns, and on charts resolved from the reference inputs:

- a log axis takes ticks where its scale puts them (decades, else 1 and 3, else 1, 2 and 5 of each), three at least,
  none of them a linear step's, and no decade at either end of the data is left unlabelled;
- a magnitude on a log axis is a dot, never a bar, and the chart says the axis is logarithmic and what an open mark is;
- every row label a chart draws is whole, on one line or two, or every row of the chart is its short label; never cut,
  never its tag alone;
- every mark on a scatter is named, by its label or its tag; a hollow mark is explained in the chart's key; no line
  joins configs whose factor the input does not order;
- a count is spelled one way ("per 1,000 calls", never "1k");
- range bars label each row beside its bar, need no legend, and say they run from p50 to p95;
- an outcome bar is its arm's trials as shares, 0 to 100%, its note the share passed and its counts;
- a code page's chart titles say what is compared and in what unit;
- the price line is drawn in ink, since green is a status's colour alone;
- changes per stratum, per task and per config draw as dots, a row with no interval an open mark;
- a dot strip's ticks share the precision their step needs ("$0.00, $0.05, $0.10"; "0%, 50%, 100%"), never the
  decimals a value is stated to;
- a line chart's end labels and a static matrix's headings and row labels wrap, never cut;
- a family-corrected level reads in every chart head of contrasts as the plain form, "99.7% intervals (95% across 16
  comparisons)".

Mutations that turn this file red: linear ticks on a log axis (``kFmtTicks`` in ``kScatter``'s log branch); bars drawn
on a log axis; a label too long for its column drawn as its tag; points labelled only where they carry a label; the
gate scatter's paths restored; ``MoneyRuler.label`` leaving a label's own count as it is; the range bars' legend kept
or their row labels not drawn; outcome bars drawn in counts; "Outcomes" as a title; the price line in ``--good``;
``_AS_DOTS`` without strata, tasks or configs; a dot strip's ticks written with its values' digits or its format's own
rounding; an end label or a matrix heading cut; the level printed as "99.69% intervals" on a forest, diverging bars or a
dot strip of contrasts.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator
from functools import cache
from typing import Any

import pytest

from packages.evals.tests.fixtures.report_inputs import dataset
from packages.evals.tests.report_doc_support import REPORT_DOCS, snapshot_of
from packages.evals.tests.report_shapes_support import s1_change, s2_choice
from threetears.evals.analysis import build_campaign_report
from threetears.evals.analysis.report import ChartBlock
from threetears.evals.analysis.report.block_spec import block_look
from threetears.evals.analysis.report.chart_svg import chart_drawing
from threetears.evals.analysis.report.charts import (
    BarsChart,
    ChartColumn,
    DotIntervalChart,
    DotIntervalRow,
    ForestChart,
    GateScatterChart,
    HeatmapCell,
    HeatmapChart,
    HeatmapRow,
    LineSeries,
    LinesChart,
    PriceLine,
    RangeBarsChart,
    ScatterChart,
    ScatterPoint,
)
from threetears.evals.analysis.report.doc_page.build import build
from threetears.evals.analysis.report.doc_page.model import DocPage
from threetears.evals.analysis.report.parts.resolve import resolve_chart
from threetears.evals.analysis.report_arms import fact_arms
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel.chart_query import ChartSpec
from threetears.evals.kernel.report_doc import ReportDoc, ReportFinding

#: The writer pages held here.
_PAGES = ("prototype5/kairo", "prototype5/r2-with-one-gate", "prototype5/r3-without-its-configs")

#: Every drawn text node's text, its spans joined: ``<text ...><tspan>a</tspan><tspan>b</tspan></text>``.
_TEXT = re.compile(r"<text[^>]*>(.*?)</text>", re.S)
_TAG = re.compile(r"<[^>]+>")
_SVG = re.compile(r"<svg class=\"viz-chart ([a-z-]+)\" viewBox=\"0 0 ([\d.]+) [\d.]+\".*?</svg>", re.S)

#: A tick label under an axis, centred on its tick.
_TICK = re.compile(r'<text x="[^"]+" y="([^"]+)" text-anchor="middle" class="tick">([^<]*)</text>')

#: A tick label's number: money, a duration or a percent, grouped by thousands.
_NUMBER = re.compile(r"-?\$?(\d[\d,]*(?:\.\d+)?)(?: ?(?:ms|s|%))?")

#: What a log axis's caption and an open mark's caption say.
_LOG_NOTE = "The horizontal axis is logarithmic: each step is a multiple."
_OPEN_NOTE = "Open marks: no interval."


@cache
def _record(name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((REPORT_DOCS / name / "record.json").read_text())
    return loaded


@cache
def _input(name: str) -> ReportInput:
    return ReportInput.model_validate(snapshot_of(name).report_input)


@cache
def _page(name: str) -> DocPage:
    record = _record(name)
    return build(
        ReportDoc.model_validate(record["doc"]),
        _input(name),
        strict=True,
        findings=[ReportFinding.model_validate(finding) for finding in record.get("findings", [])],
        draft=record["ending"] == "stored_with_findings",
    )


def _charts(name: str) -> Iterator[tuple[str, str, ChartBlock, list[str]]]:
    """Each chart on a page: the kind the document asks for, its source, the block drawn and its notes."""
    page = _page(name)
    for claim, asked in zip(page.claims, _record(name)["doc"]["claims"], strict=True):
        for evidence, block in zip(claim.evidence, asked["evidence"], strict=True):
            if evidence.chart is None or not isinstance(evidence.chart.chart, ChartBlock):
                continue
            yield block["block"], block["query"]["source"], evidence.chart.chart, evidence.chart.notes


def _one(name: str, kind: str, source: str) -> tuple[ChartBlock, list[str]]:
    (found,) = [(drawn, notes) for asked, at, drawn, notes in _charts(name) if (asked, at) == (kind, source)]
    return found


def _svgs(block: ChartBlock) -> list[tuple[float, str]]:
    """Each drawing the static page holds of a chart: its width and its markup (three widths, or a panel each)."""
    drawing = chart_drawing(block)
    assert drawing is not None
    return [(float(width), found[0]) for found in _SVG.finditer(drawing) for width in [found[2]]]


def _texts(svg: str) -> list[str]:
    return [" ".join(_TAG.sub(" ", found).split()) for found in _TEXT.findall(svg)]


def _axis_ticks(svg: str) -> list[float]:
    """The values of the ticks under a drawing's value axis: the one row of centred tick labels that read as
    numbers."""
    rows: dict[str, list[float]] = {}
    for y, label in _TICK.findall(svg):
        number = _NUMBER.fullmatch(label)
        if number is not None:
            rows.setdefault(y, []).append(float(number[1].replace(",", "")))
    (ticks,) = [row for row in rows.values() if len(row) >= 2]
    return ticks


def _on_log_scale(value: float) -> bool:
    """Whether a tick is where a log scale puts one: 1, 2, 3, 5 or 7 times a power of ten."""
    lead = value / 10 ** math.floor(math.log10(value))
    return any(math.isclose(lead, step, rel_tol=1e-9) for step in (1, 2, 3, 5, 7))


# --- log axes ---------------------------------------------------------------------------------------------------------

#: The charts on a log axis: r2's cost scatter, r3's latency gate scatter and its cost per configuration.
_LOGGED = (
    ("prototype5/r2-with-one-gate", "scatter_frontier", "measures"),
    ("prototype5/r3-without-its-configs", "gate_scatter", "configs"),
    ("prototype5/r3-without-its-configs", "bars", "measures"),
)


@pytest.mark.parametrize(("name", "kind", "source"), _LOGGED)
def test_a_log_axis_is_labelled_where_its_scale_puts_its_ticks(name: str, kind: str, source: str) -> None:
    """r2's cost scatter took a linear step's ticks on its log axis ($2, $4, $6, $8), leaving eight of its twelve
    points over an unlabelled stretch; r3's gate scatter read 500, 1000 and 1500 ms alone. Each width now draws three
    ticks at least, every one where the log scale puts one, and no decade at either end of the data unlabelled."""
    block, _ = _one(name, kind, source)
    data = block.data
    if isinstance(data, ScatterChart | GateScatterChart):
        assert data.x_log
        values = [point.x for point in data.points or []]
    else:
        assert isinstance(data, BarsChart) and data.log
        values = [row.total for row in data.rows or []]
    for width, svg in _svgs(block):
        ticks = _axis_ticks(svg)
        assert len(ticks) >= 3, (width, ticks)
        assert all(_on_log_scale(tick) for tick in ticks), (width, ticks)
        assert min(ticks) <= min(values) * 10 and max(ticks) >= max(values) / 10, (width, ticks, values)


def test_no_bar_is_drawn_on_a_log_axis() -> None:
    """r3's cost per configuration drew bars from $1 on a log axis, where a bar's length measures nothing. It draws a
    dot per configuration now, open where it has no interval, and says the axis is logarithmic."""
    block, notes = _one("prototype5/r3-without-its-configs", "bars", "measures")
    data = block.data
    assert isinstance(data, BarsChart) and data.log and data.dots and data.rows
    for width, svg in _svgs(block):
        assert not re.findall(r'<rect[^>]*style="fill:var\(--[a-z0-9-]+\);opacity', svg), width
        assert len(re.findall(r"<circle", svg)) == len(data.rows), width
    assert _LOG_NOTE in notes
    assert _OPEN_NOTE in notes


# --- labels -----------------------------------------------------------------------------------------------------------


def _drawn_rows(block: ChartBlock) -> Iterator[tuple[str, list[Any], list[str]]]:
    """Each drawing of a chart with rows, beside the rows it draws: every width of a chart, or each panel."""
    data = block.data
    svgs = _svgs(block)
    panels = getattr(data, "panels", None)
    if panels:
        for panel, (_, svg) in zip(panels, svgs, strict=True):
            yield panel.title or "", list(panel.rows or []), _texts(svg)
        return
    if isinstance(data, RangeBarsChart):
        rows: list[Any] = [row for group in data.groups for row in group.rows]
    else:
        rows = list(getattr(data, "rows", None) or [])
    for width, svg in svgs:
        yield str(width), rows, _texts(svg)


@pytest.mark.parametrize("name", _PAGES)
def test_no_row_label_is_dropped_or_cut(name: str) -> None:
    """r3's cost bars drew three configurations as their tag alone ("C", "F", "I"), the name gone, and range bars
    named no row at all. Every row label is drawn whole now, on one line or two, else every row of the chart is its
    short label; none is cut, and none is its tag alone."""
    checked = 0
    for asked, source, block, _ in _charts(name):
        if not isinstance(block.data, BarsChart | ForestChart | DotIntervalChart | RangeBarsChart):
            continue
        for at, rows, texts in _drawn_rows(block):
            whole = [row.label in texts for row in rows]
            short = [getattr(row, "short_label", None) in texts for row in rows]
            assert all(whole) or all(short), (asked, source, at, [row.label for row in rows], texts)
            assert not any(text.endswith("…") for text in texts), (asked, source, at, texts)
            checked += 1
    assert checked, name


#: Series and row names longer than a phone's margin or label column holds on one line.
_LONG_NAMES = ("Brisk-S · reasoning high · 1 retry", "Atlas-M · reasoning low · no retry")


def test_a_line_charts_end_labels_wrap_and_are_never_cut() -> None:
    """A line chart cut each series' end label to its margin ("Brisk-S · r…" on a phone). Each end label is drawn whole
    now, wrapped at its spaces, at every width."""
    data = LinesChart(
        categories=["v1", "v2", "v3"],
        series=[
            LineSeries(name=_LONG_NAMES[0], values=[90, 92, 95]),
            LineSeries(name=_LONG_NAMES[1], values=[85, 86, 88]),
        ],
        domain=(80, 100),
        fmt="pct",
    )
    block = ChartBlock(section="memo", chart="lines", title="Pass rate by version", data=data)
    for width, svg in _svgs(block):
        texts = _texts(svg)
        assert all(name in texts for name in _LONG_NAMES), (width, texts)
        assert not any("…" in text for text in texts), (width, texts)


def test_a_short_line_charts_wrapped_end_labels_all_sit_inside_its_drawing() -> None:
    """A short line chart whose three long series end close together wrapped their end labels past its foot, where
    the drawing clips them: a lost label. The drawing grows until every label line sits inside it, at every width."""
    names = (*_LONG_NAMES, "Nano-XS · reasoning off · 1 retry")
    data = LinesChart(
        categories=["v1", "v2"],
        series=[LineSeries(name=name, values=[90, value]) for name, value in zip(names, (95, 94.5, 94), strict=True)],
        domain=(80, 100),
        fmt="pct",
        height=120,
    )
    block = ChartBlock(section="memo", chart="lines", title="Pass rate by version", data=data)
    for width, svg in _svgs(block):
        height = float(re.search(r'viewBox="0 0 [\d.]+ ([\d.]+)"', svg)[1])  # type: ignore[index]
        bottoms = [
            float(y) + 13 * (inner.count("<tspan") - 1) + 4
            for y, inner in re.findall(r'<text x="[^"]+" y="([^"]+)" class="lbl(?:-strong)?">(.*?)</text>', svg)
        ]
        assert len(bottoms) == len(names), (width, bottoms)
        assert max(bottoms) <= height, (width, height, bottoms)
        assert height > 120, "the labels wrap past the stated height, so the growth is reached"
        assert all(name in _texts(svg) for name in names), width


def test_a_static_matrixs_headings_and_row_labels_wrap_and_are_never_cut() -> None:
    """The static page's heatmap cut its column headings and row labels to their room ("Ambiguous e…", "Brisk-S ·
    reas…"), where the HTML grid wraps them. Both are drawn whole now, wrapped at their spaces, at every width."""
    columns = ["Ambiguous entity resolution", "Date and unit math", "Single lookup"]
    data = HeatmapChart(
        columns=[ChartColumn(label=column) for column in columns],
        rows=[
            HeatmapRow(label=name, cells=[HeatmapCell(value=value) for value in (91, 88, 97)]) for name in _LONG_NAMES
        ],
        domain=(80, 100),
        fmt="pct",
    )
    block = ChartBlock(section="memo", chart="heatmap", title="Pass rate by task", data=data)
    for width, svg in _svgs(block):
        texts = _texts(svg)
        assert all(column in texts for column in columns), (width, texts)
        assert all(name in texts for name in _LONG_NAMES), (width, texts)
        assert not any("…" in text for text in texts), (width, texts)


# --- scatters ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "kind", "source"),
    [("prototype5/r2-with-one-gate", "scatter_frontier", "measures"), *_LOGGED[1:2]],
)
def test_every_mark_on_a_scatter_is_named_and_a_hollow_one_explained(name: str, kind: str, source: str) -> None:
    """r2's scatter drew eighteen configurations in five colours with one label: colour was the only identity, and
    hollow was never explained. Every mark is named by its label or its tag now, and the key says what hollow is."""
    block, _ = _one(name, kind, source)
    data = block.data
    assert isinstance(data, ScatterChart | GateScatterChart) and data.points
    assert all(point.tag for point in data.points)
    for width, svg in _svgs(block):
        texts = _texts(svg)
        for point in data.points:
            own = point.narrow_label if width < 560 else point.label
            assert (own or point.tag) in texts, (width, point.id, own, point.tag)
    hollow = any(point.hollow for point in data.points)
    keyed = [item.label for item in data.legend or [] if item.hollow]
    assert keyed == (["Hollow: fails a gate"] if hollow else []), (hollow, data.legend)
    assert hollow, "some point fails a gate, so the key is reached"
    drawing = chart_drawing(block)
    assert drawing is not None and "Hollow: fails a gate" in drawing


def test_no_line_joins_configs_the_input_does_not_order() -> None:
    """r3's gate scatter joined each model's prompt configs by a line, over categories with no order. It draws the
    configs as points alone now."""
    block, _ = _one("prototype5/r3-without-its-configs", "gate_scatter", "configs")
    assert isinstance(block.data, GateScatterChart) and block.data.paths is None
    for width, svg in _svgs(block):
        assert not re.findall(r'<path d="[^"]*" style="fill:none;stroke:var\(--s', svg), width


# --- units ------------------------------------------------------------------------------------------------------------


def test_a_count_is_spelled_one_way() -> None:
    """r2's scatter titled its axis "Cost per 1k calls" where the page writes "per 1,000" everywhere else. The axis
    reads "Cost per 1,000 calls" now, and no chart on a writer page spells a count "1k"."""
    block, _ = _one("prototype5/r2-with-one-gate", "scatter_frontier", "measures")
    assert isinstance(block.data, ScatterChart) and block.data.x_title == "Cost per 1,000 calls"
    for name in _PAGES:
        for asked, source, drawn, notes in _charts(name):
            headers, _ = drawn.values_table()
            svgs = _svgs(drawn) if chart_drawing(drawn) is not None else []
            said = [*(text for _, svg in svgs for text in _texts(svg)), *headers, *notes]
            assert not [text for text in said if re.search(r"\b1k\b", text)], (name, asked, source)


def test_range_bars_label_each_row_and_say_what_a_bar_spans() -> None:
    """r2's range bars keyed four arms in a legend and named none beside its bar. Each row is labelled beside its bar
    now, the legend is gone, and the head says a bar runs from p50 to p95."""
    block, _ = _one("prototype5/r2-with-one-gate", "range_bars", "measures")
    data = block.data
    assert isinstance(data, RangeBarsChart) and data.legend is None
    for width, svg in _svgs(block):
        texts = _texts(svg)
        assert "p50 to p95" in texts, width
        assert all(row.label in texts for group in data.groups for row in group.rows), (width, texts)


# --- code pages -------------------------------------------------------------------------------------------------------


@cache
def _code_pages() -> list[tuple[str, ChartBlock]]:
    """Every chart on the shape campaigns' code pages: each campaign's page and each aspect's own."""
    found: list[tuple[str, ChartBlock]] = []
    for shape in (s1_change(), s2_choice()):
        page = build_campaign_report(shape.campaign, shape.readings, shape.reports)
        found += [(shape.name, block) for block in page.blocks if isinstance(block, ChartBlock)]
        for label, report in shape.reports.items():
            found += [(f"{shape.name} · {label}", block) for block in report.blocks if isinstance(block, ChartBlock)]
    return found


def test_an_outcome_bar_is_each_arms_trials_as_shares_with_the_share_passed() -> None:
    """The outcome chart drew counts on an axis to the longest arm's total (836 against 760 trials), so shares were
    compared across unequal bars by eye, and the share passed was never printed. Each bar runs 0 to 100% of its arm's
    trials now, its note "20% passed · 167 of 836"."""
    outcomes = [(name, block) for name, block in _code_pages() if block.chart == "outcome_bar"]
    assert outcomes
    for name, block in outcomes:
        data = BarsChart.model_validate((block.kit_chart() or ("", {}))[1])
        assert data.domain == (0, 100) and data.fmt == "pct", name
        for row, bar in zip(data.rows or [], block.bars, strict=True):
            assert math.isclose(sum(segment.value for segment in row.segments or []), 100), (name, row.label)
            passed = sum(segment.count for segment in bar.segments if segment.state == "passed")
            said = re.fullmatch(r"([\d.]+)% passed · (\d+) of (\d+)", row.note or "")
            assert said is not None, (name, row.note)
            assert (int(said[2]), int(said[3])) == (passed, bar.total), (name, row.note)
            assert abs(float(said[1]) - passed / bar.total * 100) < 0.05, (name, row.note)


def test_a_code_page_chart_title_says_what_is_compared_and_in_what_unit() -> None:
    """The code page titled its charts "Outcomes" and "Cost and pass rate": topics, never what is compared or in what
    unit. Each title says it now: the outcome chart in shares of each arm's trials, the cost chart per 1,000 replies
    at its level."""
    pages = _code_pages()
    assert {block.chart for _, block in pages} >= {"outcome_bar", "cost_quality"}
    for name, block in pages:
        assert len(block.title.split()) > 1, (name, block.title)
        if block.chart == "outcome_bar":
            assert block.title == "Outcomes, share of each arm's trials", name
        if block.chart == "cost_quality":
            assert block.title.startswith("Pass rate and cost per 1,000 replies, each arm, 95% intervals"), name


# --- colour -----------------------------------------------------------------------------------------------------------


def test_the_price_line_is_drawn_in_ink() -> None:
    """The cost/gate scatter drew its price line, its label and its region's name in status green, which means a pass
    alone. They are ink now, and the region a neutral wash."""
    data = ScatterChart(
        points=[ScatterPoint(x=1, y=80, id="a", label="A"), ScatterPoint(x=4, y=90, id="b", label="B")],
        x_domain=(0, 5),
        y_domain=(70, 100),
        x_fmt="usd",
        y_fmt="pct",
        price_line=PriceLine(value=0.2, label="$0.20 per point", region_label="Worth it at this price"),
    )
    drawing = chart_drawing(ChartBlock(section="memo", chart="scatter_frontier", title="Price", data=data))
    assert drawing is not None and "$0.20 per point" in drawing and "Worth it at this price" in drawing
    assert "--good" not in drawing


# --- changes as dots --------------------------------------------------------------------------------------------------


def _with_control(name: str) -> ReportInput:
    """Reference input ``name`` with its first arm the control, every other a candidate."""
    data = dataset.report_input_of(name).model_dump(by_alias=True, exclude_none=True)
    first = data["arms"][0]["id"]
    for arm in data["arms"]:
        arm["role"] = "control" if arm["id"] == first else "challenger"
    return ReportInput.model_validate(data)


def _changes(report_input: ReportInput, source: str, measures: list[str]) -> tuple[ChartBlock, tuple[str, ...]]:
    spec = ChartSpec.model_validate(
        {
            "kind": "diverging_bars",
            "title": "",
            "query": {
                "source": source,
                "aspect": "",
                "measures": measures,
                "arms": [],
                "strata": [],
                "limit": 0,
                "transform": "none",
                "small_multiples_by": "none",
            },
            "reference": [],
            "emphasis": "",
            "level": "",
            "unit": "",
            "domain": [],
        }
    )
    drawn = resolve_chart(spec, report_input, block_look(report_input, fact_arms(report_input)), title="chart")
    assert isinstance(drawn.chart, ChartBlock)
    return drawn.chart, tuple(drawn.notes)


@pytest.mark.parametrize(
    ("source", "report_input", "measures"),
    [
        ("strata", lambda: dataset.report_input_of("r4"), []),
        ("tasks", lambda: _with_control("r1"), ["overall"]),
        ("configs", lambda: dataset.report_input_of("r2"), ["pass_rate"]),
    ],
    ids=["strata", "tasks", "configs"],
)
def test_changes_per_stratum_task_and_config_are_dots_open_without_an_interval(
    source: str, report_input: Any, measures: list[str]
) -> None:
    """Per-stratum, per-task and per-config changes draw as dots, as per-place and per-lever ones do: a row with no
    interval an open mark (drawn hollow), never a solid bar, and the chart says what an open mark is."""
    block, notes = _changes(report_input(), source, measures)
    data = block.data
    assert block.chart == "diverging_bars" and isinstance(data, BarsChart) and data.dots
    rows = data.rows or [row for panel in data.panels or [] for row in panel.rows or []]
    assert rows and any(row.lo is None for row in rows)
    assert _OPEN_NOTE in notes
    for _, svg in _svgs(block)[:1]:
        marks = re.findall(r"<circle[^>]*style=\"([^\"]*)\"", svg)
        assert not re.findall(r'<rect[^>]*style="fill:var\(--[a-z0-9-]+\);opacity', svg)
        if data.rows:
            assert [style.startswith("fill:var(--surface)") for style in marks] == [row.lo is None for row in rows]


# --- tick precision ---------------------------------------------------------------------------------------------------


def _decimals(label: str) -> int:
    """How many decimals a tick label is written to."""
    number = _NUMBER.fullmatch(label)
    assert number is not None, label
    whole, _, fraction = number[1].partition(".")
    return len(fraction)


@pytest.mark.parametrize(
    ("fmt", "domain", "row", "stated"),
    [
        # A near-even share stated to three decimals: the ticks step by tens of points, so they read whole.
        ("pct:3", (0, 100), DotIntervalRow(label="lead-first", value=49.998, lo=26.5, hi=73.5), "49.998%"),
        # A money strip under $0.20: its ticks step by cents, so every one reads to two decimals ($0.00, $0.05 …).
        ("usd", (0, 0.2), DotIntervalRow(label="Nano-XS", value=0.0731, lo=0.051, hi=0.094), "$0.0731"),
        # A fractional percent step (0.5%): every tick reads to one decimal, 0.0% beside 0.5% and 1.0%.
        ("pct:2", (0, 2), DotIntervalRow(label="flaky", value=1.23, lo=0.81, hi=1.65), "1.23%"),
    ],
    ids=["near-even-share", "money-under-20-cents", "fractional-percent-step"],
)
def test_a_dot_strips_ticks_share_the_precision_their_step_needs(
    fmt: str, domain: tuple[float, float], row: DotIntervalRow, stated: str
) -> None:
    """A dot strip's ticks are written to the decimals its axis's step needs, every tick alike, never to the decimals a
    value is stated to ("50.000%") nor to a format's own rounding ("$0", "$0.0500", "$0.10"); the value keeps its
    decimals."""
    data = DotIntervalChart(rows=[row], domain=domain, fmt=fmt, level="95%")
    block = ChartBlock(section="memo", chart="dot_interval", title="Strip", data=data)
    for width, svg in _svgs(block):
        ticks = [label for _, label in _TICK.findall(svg)]
        values = [float(_NUMBER.fullmatch(tick)[1].replace(",", "")) for tick in ticks]  # type: ignore[index]
        assert len(ticks) >= 2, (width, ticks)
        step = min(b - a for a, b in zip(values, values[1:], strict=False))
        needed = next(places for places in range(7) if abs(round(step * 10**places) - step * 10**places) < 1e-6)
        assert {_decimals(tick) for tick in ticks} == {needed}, (width, ticks)
        assert f"<title>{row.label}\n{stated}" in svg, (width, stated)


# --- the interval level -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["diverging_bars", "dot_interval"])
def test_a_family_corrected_level_reads_plainly_on_every_chart_of_contrasts(kind: str) -> None:
    """Kairo's corrected contrasts drawn as diverging bars or a dot strip printed "99.69% intervals" in the head, as
    the forest did. Every chart of them reads "99.7% intervals (95% across 16 comparisons)" now."""
    report_input = _input("prototype5/kairo")
    spec = ChartSpec.model_validate(
        {
            "kind": kind,
            "title": "",
            "query": {
                "source": "contrasts",
                "aspect": "character adherence (v2)",
                "measures": ["passed", "persona.holds_character"],
                "arms": [],
                "strata": [],
                "limit": 0,
                "transform": "none",
                "small_multiples_by": "none",
            },
            "reference": [],
            "emphasis": "",
            "level": "99.69%",
            "unit": "",
            "domain": [],
        }
    )
    drawn = resolve_chart(spec, report_input, block_look(report_input, fact_arms(report_input)), title="chart")
    block = drawn.chart
    assert isinstance(block, ChartBlock) and block.data is not None
    assert block.data.level == "99.7% (95% across 16 comparisons)"  # type: ignore[union-attr]
    for width, svg in _svgs(block):
        texts = _texts(svg)
        assert "99.7% intervals (95% across 16 comparisons)" in texts, (width, texts)
        assert not [text for text in texts if "99.69" in text], width


def test_a_family_corrected_level_reads_plainly_in_the_chart_head() -> None:
    """Kairo's corrected forest printed "99.69% intervals" in its head, a level a reader cannot read. Its head reads
    "99.7% intervals (95% across 16 comparisons)" now, on the chart and in its values table, and 99.69% is printed
    nowhere on it."""
    for name in ("prototype5/kairo",):
        forests = [block for asked, source, block, _ in _charts(name) if (asked, source) == ("forest", "contrasts")]
        assert forests
        for block in forests:
            assert isinstance(block.data, ForestChart)
            assert block.data.level == "99.7% (95% across 16 comparisons)"
            headers, _ = block.values_table()
            assert "99.7% interval (95% across 16 comparisons)" in headers
            for width, svg in _svgs(block):
                texts = _texts(svg)
                assert "99.7% intervals (95% across 16 comparisons)" in texts, width
                assert not [text for text in texts if "99.69" in text], width
