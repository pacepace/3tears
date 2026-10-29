"""A static page's charts, drawn on the server as inline SVG for a host whose policy runs no script.

- **Every kind a report can carry is drawn, or says it is not.** Each kind the kit draws as a picture is drawn here
  by the kit's geometry; a kind the kit lays out as HTML (a gate matrix, a lattice, example cards) and a report's
  interval rows draw nothing, and the figure says in one line that it is shown as its table.
- **It reads at any width.** A chart is drawn at a phone's width, the kit's desktop measure and a wide page's, each
  scaling to its frame by its ``viewBox``; a small multiple once, at a panel's width.
- **A tooltip is a ``<title>``.** What the kit's tooltip shows on a mark, its ``<title>`` holds, as text.
- **Light and dark.** Every colour a drawing names is one of the kit's tokens, defined for both themes; none is
  written as a colour of its own, and a colour that is not a token name never reaches a style.
- **The numbers read as the kit draws them.** Ticks and notes are written by the kit's own named formats, read here
  against the same money cases the kit is held to in the browser.
- **The default page is unchanged.** It draws nothing on the server and carries none of the drawings' styles.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from threetears.evals.analysis import ChartBlock, report_html
from threetears.evals.analysis.report.charts import BarRow, BarsChart
from threetears.evals.analysis.report.chart_svg import AS_TABLE_NOTE, CHART_STYLE, chart_drawing, kit_format
from packages.evals.tests.hostile_page_support import hostile_kit_charts
from packages.evals.tests.page_contract_support import parsed
from packages.evals.tests.report_kit_support import report_of

#: The number format cases every renderer reads, the kit's money formats among them.
_NUMBER_CASES = Path(__file__).resolve().parent / "fixtures" / "number-format-cases.json"

#: The kit's stylesheet, whose tokens a drawing's colours name.
_KIT_CSS = (
    Path(__file__).resolve().parents[1] / "src" / "threetears" / "evals" / "analysis" / "report" / "kit" / "kit.css"
).read_text(encoding="utf-8")

#: The kinds not drawn on the server, each shown as its table and saying so.
_NOT_DRAWN = {"gate_matrix", "lattice", "examples"}


def _block(**fields: Any) -> ChartBlock:
    return ChartBlock.model_validate({"section": "memo", "finding": 0, "title": "Chart", **fields})


def _svgs(markup: str) -> list[dict[str, str | None]]:
    return [
        dict(attrs)
        for tag, attrs in parsed(markup).tags
        if tag == "svg" and "viz-chart" in (dict(attrs).get("class") or "")
    ]


def _tokens(block: str) -> set[str]:
    return set(re.findall(r"(--[a-z][a-z0-9-]*)\s*:", block))


def test_each_kind_the_kit_draws_as_a_picture_is_drawn_and_the_rest_are_not() -> None:
    for chart in hostile_kit_charts():
        drawing = chart_drawing(_block(**chart))
        if chart["chart"] in _NOT_DRAWN:
            assert drawing is None, chart["chart"]
            continue
        assert drawing is not None and drawing.startswith(f'<div class="chart-drawn" data-kind="{chart["chart"]}">')
        assert _svgs(drawing), f"a {chart['chart']} chart draws"
    rows = {"label": "a", "value": 1.0, "low": 0.5, "high": 1.5, "n_cases": 6}
    assert chart_drawing(_block(chart="interval_rows", rows=[rows])) is None


def test_a_chart_not_drawn_says_in_one_line_that_it_is_shown_as_its_table() -> None:
    rows = {"label": "a", "value": 1.0, "low": 0.5, "high": 1.5, "n_cases": 6}
    gate = next(chart for chart in hostile_kit_charts() if chart["chart"] == "gate_matrix")
    page = report_html(report_of([{**gate, "title": "Gates"}, _block(chart="interval_rows", rows=[rows])]), static=True)
    for figure in re.findall(r'<figure class="fig".*?</figure>', page, flags=re.DOTALL):
        assert figure.count(f'<p class="chart-note">{AS_TABLE_NOTE}</p>') == 1 and "chart-values" in figure


def test_a_chart_is_drawn_at_three_widths_and_a_small_multiple_once_at_a_panel_s() -> None:
    bars = _block(chart="bars", data={"rows": [{"label": "a", "value": 2}], "domain": [0, 4], "fmt": "int"})
    drawn = _svgs(chart_drawing(bars) or "")
    assert [(svg["class"], (svg["viewbox"] or "").split()[2]) for svg in drawn] == [
        ("viz-chart viz-narrow", "360"),
        ("viz-chart viz-mid", "640"),
        ("viz-chart viz-wide", "960"),
    ]
    assert all(svg["role"] == "img" and "width" not in svg and "height" not in svg for svg in drawn), "it scales"
    panels = {"panels": [{"title": "one", "rows": [{"label": "a", "value": 1}]}, {"title": "two"}]}
    drawn = _svgs(chart_drawing(_block(chart="bars", data={**(bars.kit_chart() or ("", {}))[1], **panels})) or "")
    assert [svg["class"] for svg in drawn] == ["viz-chart viz-panel"], "a panel once; a text card none"


def test_the_style_shows_the_drawing_nearest_the_chart_s_own_width() -> None:
    assert "container-type: inline-size" in CHART_STYLE
    assert "@container (max-width: 499px)" in CHART_STYLE and "@container (min-width: 800px)" in CHART_STYLE


def test_what_a_tooltip_shows_is_each_mark_s_title() -> None:
    data = {
        "rows": [{"label": "arm a", "value": 2, "lo": 1, "hi": 3}, {"label": "arm b", "value": 1, "tip": ["b", "one"]}],
        "domain": [0, 4],
        "fmt": "int",
        "level": "95%",
    }
    titles = [
        text
        for text in re.findall(
            r"<title>(.*?)</title>", chart_drawing(_block(chart="bars", data=data)) or "", flags=re.DOTALL
        )
    ]
    assert titles[:3] == ["Chart", "arm a\n2 (1 to 3)", "b\none"], "the chart's name, then each row's tooltip"


def test_every_colour_a_drawing_names_is_a_kit_token_defined_for_light_and_dark() -> None:
    light = _tokens(_KIT_CSS[: _KIT_CSS.index("@media (prefers-color-scheme:dark)")])
    dark = _tokens(_KIT_CSS[_KIT_CSS.index("@media (prefers-color-scheme:dark)") :])
    named: set[str] = set()
    for chart in hostile_kit_charts():
        drawing = chart_drawing(_block(**chart)) or ""
        styles = [value or "" for _, attrs in parsed(drawing).tags for name, value in attrs if name == "style"]
        for style in styles:
            for prop, value in re.findall(r"(fill|stroke|background|color):([^;]+)", style):
                stripped = re.sub(r"var\(--[a-z][a-z0-9-]*\)", "", value)
                assert stripped.strip() in {"", "none"} or re.fullmatch(r"color-mix\(in oklab,  \d+%, \)", stripped), (
                    f"{prop}:{value} names a colour of its own"
                )
                named |= set(re.findall(r"var\((--[a-z][a-z0-9-]*)\)", value))
        assert not re.search(r"#[0-9a-f]{3,8}\b|rgb\(|hsl\(", drawing, flags=re.IGNORECASE)
    assert named, "the drawings name colours"
    assert named <= light and named <= dark, f"undefined in a theme: {sorted(named - (light & dark))}"
    assert not re.search(r"#[0-9a-f]{3,8}\b|rgb\(", CHART_STYLE, flags=re.IGNORECASE)


def test_a_colour_that_is_not_a_token_name_never_reaches_a_style() -> None:
    """A block whose data skipped validation (built with ``model_construct``) is read again as its kind's model before
    it is drawn, so a colour carrying CSS of its own is refused, never written into a style."""
    data = BarsChart.model_construct(
        rows=[BarRow.model_construct(label="a", value=2, color="red;x:url(y)")], domain=(0, 4), fmt="int"
    )
    smuggled = ChartBlock.model_construct(section="memo", finding=0, title="Chart", chart="bars", data=data)
    with pytest.raises(ValueError, match="pattern"):
        chart_drawing(smuggled)


def test_the_drawn_numbers_are_the_kit_s_named_formats() -> None:
    cases = json.loads(_NUMBER_CASES.read_text(encoding="utf-8"))["kit_money"]["cases"]
    assert [kit_format(case["fmt"])(case["value"]) for case in cases] == [case["expected"] for case in cases]
    assert [
        kit_format(spec)(value) for spec, value in [("pct", 12.5), ("+pts:1", 2), ("int", 1234.5), ("num", 0.125)]
    ] == [
        "12.5%",
        "+2.0 pts",
        "1,235",
        "0.13",
    ]
    with pytest.raises(ValueError, match="fmt"):
        kit_format("bogus")


def test_the_default_page_draws_nothing_on_the_server() -> None:
    page = report_html(
        report_of([_block(chart="bars", data={"rows": [{"label": "a", "value": 2}], "domain": [0, 4], "fmt": "int"})])
    )
    assert "chart-drawn" not in page and "chart-note" not in page and CHART_STYLE not in page
    assert 'class="chart"' in page
