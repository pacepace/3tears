"""The campaign page's skeleton: its own sections and their headings, a cell's flag, the cost/quality chart, and what
the page calls an aspect.

None of it changes a page laid out today: no block sits in a campaign's own sections yet, no cell is flagged and no
cost/quality chart is drawn. These hold the shapes the page's parts are built into, over the shared campaign shapes.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import get_args

import pytest
from pydantic import ValidationError

from threetears.evals.analysis import campaign_report_html, campaign_report_markdown, serialize_report
from threetears.evals.analysis.report.campaign.names import aspect_name
from threetears.evals.analysis.report.model import (
    CAMPAIGN_JSON_SHAPE,
    FIRST_SCREEN,
    SECTION_TITLES,
    ChartBlock,
    CostQualityPoint,
    ReportSection,
    TableBlock,
    TableCell,
)

from packages.evals.tests.report_shapes_support import SHAPES, Shape, s1_change, s3_measurement, s4_change

#: A campaign's own sections, which no aspect page holds.
_CAMPAIGN_SECTIONS = ("answer", "warnings", "scorecard", "findings", "next", "provenance")


@pytest.mark.parametrize("name", ["s1", "s4"])
def test_a_campaign_page_heads_every_section_in_reading_order_its_own_among_an_aspect_page_s(name: str) -> None:
    page = SHAPES[name]().page()
    titles = page.section_titles
    assert list(titles) == list(get_args(ReportSection))
    assert {section: titles[section] for section in SECTION_TITLES} == SECTION_TITLES
    assert set(titles) - set(SECTION_TITLES) == set(_CAMPAIGN_SECTIONS)
    assert all(len(titles[section].split()) <= 3 for section in _CAMPAIGN_SECTIONS)
    # The first screen carries no heading; every other section of the campaign's own is headed.
    assert {section for section in _CAMPAIGN_SECTIONS if not titles[section]} == {"answer", "warnings"}
    assert {"answer", "warnings"} <= FIRST_SCREEN
    # An aspect's own page is headed as before.
    assert all(fold.report.section_titles.keys() == SECTION_TITLES.keys() for fold in page.aspects)


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_no_block_sits_in_a_campaign_s_own_sections_yet(name: str) -> None:
    page = SHAPES[name]().page()
    assert not [block for block in page.blocks if block.section in _CAMPAIGN_SECTIONS]


def test_a_flag_is_printed_after_its_value_and_a_cell_without_one_prints_as_before() -> None:
    assert TableCell(value=-60.0).text == "-60"
    assert TableCell(value=-60.0, flag="floor broken").text == "-60 (floor broken)"
    assert TableCell(value=None, flag="cannot show a drop").text == "— (cannot show a drop)"
    with pytest.raises(ValidationError):
        TableCell(value=1.0, flag="")


@pytest.mark.parametrize("build", [s1_change, s4_change], ids=["s1", "s4"])
def test_a_flagged_cell_reads_beside_its_value_in_markdown_and_html(build: Callable[[], Shape]) -> None:
    page = build().page()
    (table,) = [block for block in page.blocks if isinstance(block, TableBlock) and block.name == "aspects"]
    first = table.rows[0]
    flagged = {**first, "reading": first["reading"].model_copy(update={"flag": "cannot show a gain"})}
    marked = table.model_copy(update={"rows": [flagged, *table.rows[1:]]})
    changed = page.model_copy(update={"blocks": [marked if block is table else block for block in page.blocks]})
    said = f"{first['reading'].text} (cannot show a gain)"
    assert said in campaign_report_markdown(changed) and said not in campaign_report_markdown(page)
    assert said in campaign_report_html(changed)


def _point(arm: str, cost: float, rate: float) -> CostQualityPoint:
    return CostQualityPoint(
        arm=arm,
        cost=cost,
        cost_low=cost * 0.9,
        cost_high=cost * 1.1,
        pass_rate=rate,
        pass_rate_low=rate - 5,
        pass_rate_high=rate + 5,
    )


def test_a_cost_quality_chart_draws_a_point_per_arm_and_prints_every_figure_as_a_table() -> None:
    chart = ChartBlock(
        section="scorecard",
        chart="cost_quality",
        title="Cost and pass rate",
        unit="usd",
        points=[_point("alpha", 0.004, 85.0), _point("beta", 0.0008, 60.0)],
    )
    headers, rows = chart.values_table()
    assert headers == ["Arm", "Cost ($)", "Cost interval ($)", "Pass rate (%)", "Interval (%)"]
    assert [row[0] for row in rows] == ["alpha", "beta"]
    assert rows[0][3] == "85" and rows[0][4] == "[80, 90]"


def test_a_cost_quality_chart_of_one_point_or_another_kind_s_marks_is_refused() -> None:
    with pytest.raises(ValidationError, match="for two arms or more"):
        ChartBlock(section="scorecard", chart="cost_quality", title="t", points=[_point("alpha", 0.004, 85.0)])
    with pytest.raises(ValidationError, match="no interval row"):
        ChartBlock(section="scoreboard", chart="outcome_bar", title="t", points=[_point("alpha", 0.004, 85.0)])
    with pytest.raises(ValidationError, match="is not within"):
        CostQualityPoint(
            arm="alpha", cost=0.004, cost_low=0.005, cost_high=0.006, pass_rate=50, pass_rate_low=40, pass_rate_high=60
        )


def test_an_aspect_is_called_by_its_full_label_as_declared() -> None:
    for shape in (s1_change(), s3_measurement()):
        assert [aspect_name(bundle) for bundle in shape.bundles.values()] == [
            aspect.label for aspect in shape.campaign.aspects
        ]


@pytest.mark.parametrize("name", ["s1", "s2"])
def test_the_page_json_is_shape_5(name: str) -> None:
    document = json.loads(serialize_report(SHAPES[name]().page(), "json"))
    assert CAMPAIGN_JSON_SHAPE == 5 and document["shape"] == 5
