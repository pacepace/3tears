"""Each interval level a page draws is stated once, and a chart at another level is marked (#1128; owner ruling,
2026-11-14).

Contracts held here (``analysis.report.chart_levels``, the charts' ``level_head``, ``chart_svg``, ``kit.js``):

- **the page's main level is stated once**: the level most of a page's charts draw (the first of them on a tie) heads
  the first chart that draws it, and no later chart at it repeats it;
- **a chart at another level is marked**: its head prints its level and the main level's figure it is not
  (``98.3% intervals, not 95%``), on every chart at it; a level differing in its gloss alone keeps the gloss as its mark;
- **only a head that prints counts**: a scatter that draws no whisker states no level, so it sets no main level, and
  a chart alone keeps its head as it was;
- **every page draws it so**: the static page's drawings, the kit's ``data-chart`` and the kit's drawing in a browser
  print the heads the page decided, and a written report states its main level once across its claims' evidence.
"""

from __future__ import annotations

import html
import json
import re
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from packages.evals.tests.report_doc_support import converted_doc, snapshot_of
from packages.evals.tests.report_kit_support import chromium_unavailable, report_of, rendered
from threetears.evals.analysis import report_html
from threetears.evals.analysis.report.blocks import ChartBlock
from threetears.evals.analysis.report.chart_levels import levels_said
from threetears.evals.analysis.report.charts import drawn_head
from threetears.evals.analysis.report.doc_page.build import build
from threetears.evals.analysis.report.doc_page.html import doc_page_html
from threetears.evals.analysis.report.doc_page.model import DocPage
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel.report_doc import ReportDoc

_ADJUSTED = "95% (adjusted for 3 comparisons)"


def _forest(level: str, title: str = "Change") -> ChartBlock:
    """A forest of two changes at ``level``."""
    return ChartBlock.model_validate(
        {
            "section": "memo",
            "chart": "forest",
            "title": title,
            "data": {
                "rows": [
                    {"label": "a", "value": -2.0, "lo": -4.0, "hi": 0.5},
                    {"label": "b", "value": 1.0, "lo": -1.0, "hi": 3.0},
                ],
                "domain": [-6.0, 6.0],
                "fmt": "+num:1",
                "level": level,
            },
        }
    )


def _scatter(*, whiskered: bool) -> ChartBlock:
    """A scatter of two points at 98.3%, each with its intervals where ``whiskered``."""
    ends = {"x_lo": 0.5, "x_hi": 1.5, "y_lo": 70.0, "y_hi": 90.0} if whiskered else {}
    return ChartBlock.model_validate(
        {
            "section": "memo",
            "chart": "scatter_frontier",
            "title": "Cost and quality",
            "data": {
                "points": [{"x": 1.0, "y": 80.0, "label": "a", **ends}, {"x": 2.0, "y": 85.0, "label": "b", **ends}],
                "x_domain": [0.0, 3.0],
                "y_domain": [60.0, 100.0],
                "x_fmt": "usd:2",
                "y_fmt": "pct",
                "level": "98.3%",
            },
        }
    )


def _heads(blocks: list[ChartBlock]) -> list[str]:
    """What each chart's head prints of its level, the page's rule applied."""
    return [drawn_head(block.data) for block in levels_said(blocks)]


def test_the_main_level_heads_its_first_chart_alone() -> None:
    assert _heads([_forest("95%"), _forest("95%"), _forest("95%")]) == ["95% intervals", "", ""]


def test_a_chart_at_another_level_is_marked_on_every_chart_at_it() -> None:
    """Two levels two charts each: the first met is the main one."""
    heads = _heads([_forest("95%"), _forest("98.3%"), _forest("95%"), _forest("98.3%")])
    assert heads == ["95% intervals", "98.3% intervals, not 95%", "", "98.3% intervals, not 95%"]


def test_the_main_level_is_the_one_most_charts_draw() -> None:
    heads = _heads([_forest("98.3%"), _forest("95%"), _forest("95%")])
    assert heads == ["98.3% intervals, not 95%", "95% intervals", ""]


def test_a_level_differing_in_its_gloss_alone_is_marked_by_its_gloss() -> None:
    heads = _heads([_forest(_ADJUSTED), _forest("95%"), _forest(_ADJUSTED), _forest("95% unpaired")])
    assert heads == ["95% intervals (adjusted for 3 comparisons)", "95% intervals", "", "95% unpaired intervals"]


def test_a_chart_whose_head_the_rule_keeps_is_left_unset() -> None:
    """The first chart at the main level and a marked chart whose head is its level's own carry no ``level_head``, so
    their data reads as it did."""
    first, glossed, _ = levels_said([_forest(_ADJUSTED), _forest("95%"), _forest(_ADJUSTED)])
    assert first.data is not None and glossed.data is not None
    assert getattr(first.data, "level_head") is None and getattr(glossed.data, "level_head") is None


def test_a_chart_alone_and_a_scatter_without_whiskers_keep_their_heads() -> None:
    assert _heads([_forest("95%")]) == ["95% intervals"]
    bare = _scatter(whiskered=False)
    # Two bare scatters beside a forest set no main level: the forest is the page's one stated level, left as it was.
    assert _heads([_forest("95%"), bare, bare])[0] == "95% intervals"
    # Two whiskered ones state 98.3%, so it is the main level and the forest is marked against it.
    assert _heads([_forest("95%"), _scatter(whiskered=True), _scatter(whiskered=True)]) == [
        "95% intervals, not 98.3%",
        "98.3% intervals",
        "",
    ]
    # The bare scatter states no level, so the forests' 95% is stated once and the scatter's is not counted.
    assert [drawn_head(block.data) for block in levels_said([_forest("95%"), bare, _forest("95%")])][::2] == [
        "95% intervals",
        "",
    ]
    whiskered = levels_said([_forest("95%"), _scatter(whiskered=True), _forest("95%")])[1]
    assert drawn_head(whiskered.data) == "98.3% intervals, not 95%"


def _three() -> list[ChartBlock]:
    return [_forest("95%", "First"), _forest("98.3%", "Corrected"), _forest("95%", "Second")]


def _drawn_heads(page: str) -> list[str]:
    """Every chart head the static page draws, in page order (each chart is drawn at three widths)."""
    return [
        html.unescape(text) for text in re.findall(r'<text[^>]*text-anchor="end"[^>]*>([^<]*intervals[^<]*)<', page)
    ]


def test_the_static_page_draws_each_head_the_page_decided() -> None:
    heads = _drawn_heads(report_html(report_of(_three()), static=True))
    assert heads == ["95% intervals"] * 3 + ["98.3% intervals, not 95%"] * 3, heads


def test_the_kit_page_serves_each_head_the_page_decided() -> None:
    page = report_html(report_of(_three()))
    served = [json.loads(html.unescape(found)) for found in re.findall(r'data-chart="([^"]*)"', page)]
    assert [chart.get("level_head") for chart in served] == [None, "98.3% intervals, not 95%", ""]
    assert [chart["level"] for chart in served] == ["95%", "98.3%", "95%"], "the level itself is never dropped"


#: Every chart head the kit draws, in page order.
_KIT_HEADS = """() => [...document.querySelectorAll("div.chart svg text")].map((t) => t.textContent)
  .filter((text) => text.includes("intervals"))"""


@pytest.mark.browser
def test_the_kit_draws_each_head_the_page_decided(tmp_path: Path) -> None:
    reason = chromium_unavailable()
    if reason is not None:
        pytest.skip(f"headless Chromium cannot launch: {reason}")
    page = tmp_path / "levels.html"
    page.write_text(report_html(report_of(_three())), encoding="utf-8")
    heads, errors = rendered(page, _KIT_HEADS)
    assert errors == []
    assert heads == ["95% intervals", "98.3% intervals, not 95%"], heads


@cache
def _doc_page(name: str) -> DocPage:
    held = ReportInput.model_validate(snapshot_of(name).report_input)
    return build(ReportDoc.model_validate(converted_doc(name)), held, strict=True)


def test_a_written_report_states_its_main_level_once_across_its_claims() -> None:
    """s2's written report draws three charts at 95%: the first states it, the others are drawn without a head. s1's
    draws its adjusted level four times and 95% three: the adjusted level heads its first chart alone, and each 95%
    chart keeps its own head, the mark of a level other than the main one."""
    plain = _drawn_heads(doc_page_html(_doc_page("prototype2/s2"), static=True))
    assert plain == ["95% intervals"] * 3, plain
    mixed = _drawn_heads(doc_page_html(_doc_page("prototype2/s1"), static=True))
    adjusted = "95% intervals (adjusted for 11 comparisons)"
    assert mixed == [adjusted] * 3 + ["95% intervals"] * 9, mixed


def _own_heads(page: DocPage) -> list[str]:
    """The head each chart of a written report prints of its own level, before its page's rule moves any."""
    charts: list[Any] = [evidence.chart.chart for claim in page.claims for evidence in claim.evidence if evidence.chart]
    return [drawn_head(chart.data) for chart in charts if isinstance(chart, ChartBlock) and chart.data is not None]


def test_the_written_reports_draw_the_levels_the_test_reads() -> None:
    """The fixtures read above draw what the test says they draw, so a change to them fails here, not as a miscount."""
    assert [head for head in _own_heads(_doc_page("prototype2/s2")) if head] == ["95% intervals"] * 3
    s1 = [head for head in _own_heads(_doc_page("prototype2/s1")) if head]
    assert s1.count("95% intervals (adjusted for 11 comparisons)") == 4 and s1.count("95% intervals") == 3
