"""The campaign page's skeleton: its own sections and their headings, a cell's flag, the cost/quality chart, and what
the page calls an aspect.

These hold the shapes the page's parts are built into, over the shared campaign shapes: every campaign section is
headed and laid out in the page's order, a cell's flag prints after its value, and a cost/quality chart draws its points.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import get_args

import pytest
from pydantic import ValidationError

from threetears.evals.analysis import campaign_report_html, campaign_report_markdown, serialize_report
from threetears.evals.analysis.report.campaign.names import AspectNames, aspect_name
from threetears.evals.analysis.report.model import (
    CAMPAIGN_JSON_SHAPE,
    FIRST_SCREEN,
    SECTION_TITLES,
    CampaignReport,
    ChartBlock,
    CostQualityPoint,
    DisclosureBlock,
    IntervalRow,
    OutcomeBar,
    OutcomeSegment,
    Report,
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


@pytest.mark.parametrize("name", ["s1", "s2"])
@pytest.mark.parametrize("section", ["scorecard", "findings", "next", "provenance"])
def test_a_block_in_a_campaign_s_own_headed_section_is_laid_out_under_its_heading_before_the_folds(
    section: str, name: str
) -> None:
    page = SHAPES[name]().page()
    said = f"A line placed in {section}."
    placed = page.model_copy(
        update={"blocks": [*page.blocks, DisclosureBlock(section=section, source="comparisons", text=said)]}
    )
    heading = page.section_titles[section]
    markdown, html = campaign_report_markdown(placed), campaign_report_html(placed)
    first_fold = f"## {page.aspects[0].aspect}"
    assert markdown.index(f"## {heading}") < markdown.index(said) < markdown.index(first_fold)
    assert html.index(f'<section id="{section}">') < html.index(said) < html.index('<section id="aspect-')
    assert f'<a href="#{section}">' in html


@pytest.mark.parametrize("section", sorted(_CAMPAIGN_SECTIONS))
def test_an_aspect_s_page_refuses_a_block_in_a_campaign_s_own_section(section: str) -> None:
    report = next(iter(s4_change().reports.values()))
    with pytest.raises(ValidationError, match=f"lays out no section {section}"):
        Report.model_validate(
            {
                **dict(report),
                "blocks": [*report.blocks, DisclosureBlock(section=section, source="comparisons", text="A line.")],
            }
        )


@pytest.mark.parametrize("build", [s1_change, s4_change], ids=["s1", "s4"])
def test_no_page_holds_a_block_in_the_glossary_which_is_written_from_its_terms(build: Callable[[], Shape]) -> None:
    shape = build()
    report = next(iter(shape.reports.values()))
    stray = DisclosureBlock(section="glossary", source="comparisons", text="A line.")
    with pytest.raises(ValidationError, match="lays out no section glossary"):
        Report.model_validate({**dict(report), "blocks": [*report.blocks, stray]})
    page = shape.page()
    with pytest.raises(ValidationError, match="the glossary is written from the terms"):
        CampaignReport.model_validate({**dict(page), "blocks": [*page.blocks, stray]})


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
    # The table states cost per 1,000 replies, as the chart's axis does: $0.004 a reply is $4.
    assert headers == ["Arm", "Cost per 1,000 replies ($)", "Interval ($)", "Pass rate (%)", "Interval (%)"]
    assert chart.kit_chart()[1]["x_title"] == "Cost per 1,000 replies ($)"  # type: ignore[index]
    assert [row[0] for row in rows] == ["alpha", "beta"]
    # One form over the chart's costs (#993): $0.80 needs cents, so $4 beside it reads $4.00.
    assert (rows[0][1], rows[0][2], rows[1][1]) == ("4.00", "[3.60, 4.40]", "0.80"), rows
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
    with pytest.raises(ValidationError, match="its pass rate 50.0 is not within"):
        CostQualityPoint(
            arm="alpha", cost=0.004, cost_low=0.003, cost_high=0.005, pass_rate=50, pass_rate_low=55, pass_rate_high=60
        )
    two = [_point("alpha", 0.004, 85.0), _point("beta", 0.0008, 60.0)]
    row = IntervalRow(label="alpha", value=1.0, low=0.0, high=2.0, n_cases=5)
    bar = OutcomeBar(arm="alpha", segments=[OutcomeSegment(state="passed", label="passed", count=3)])
    with pytest.raises(ValidationError, match="an interval chart draws at least one row and no outcome bar"):
        ChartBlock(section="contrasts", chart="interval_rows", title="t", rows=[row], points=two)
    for marks in ({"rows": [row]}, {"bars": [bar]}):
        with pytest.raises(ValidationError, match="for two arms or more, and nothing else"):
            ChartBlock(section="scorecard", chart="cost_quality", title="t", points=two, **marks)


def test_a_cost_quality_chart_is_drawn_one_point_per_arm_with_its_figures_in_words() -> None:
    chart = ChartBlock(
        section="scorecard",
        chart="cost_quality",
        title="Cost and pass rate",
        unit="usd",
        points=[_point("alpha", 0.004, 85.0), _point("beta", 0.0008, 60.0)],
    )
    drawn = chart.kit_chart()
    assert drawn is not None
    kind, data = drawn
    assert kind == "scatter_frontier" and "frontier" not in data, "no frontier is tested, so none is drawn"
    points = data["points"]
    assert isinstance(points, list) and [point["label"] for point in points] == ["alpha", "beta"]  # type: ignore[index]
    assert points[0]["tip"] == [  # type: ignore[index]
        "alpha",
        # One form over the chart (#993), its interval on the chart's ruler: never $4 beside $0.8, nor a cost under a
        # cent spelled per reply.
        "$4.00 [$3.60, $4.40] per 1,000 replies",
        "pass rate 85% [80%, 90%]",
    ]
    assert data["y_domain"] == [0, 100] and data["x_domain"][0] == 0, "no axis is cut to flatter a gap"  # type: ignore[index]


def test_an_aspect_is_called_by_its_full_label_as_declared() -> None:
    for shape in (s1_change(), s3_measurement()):
        labels = [aspect.label for aspect in shape.campaign.aspects]
        assert [aspect_name(bundle) for bundle in shape.bundles.values()] == labels
        names = AspectNames.of(shape.bundles.values())
        assert [names(label) for label in labels] == labels
        # A label no evidence was given for is printed whole, never shortened.
        assert names("an aspect no bundle reads") == "an aspect no bundle reads"


@pytest.mark.parametrize("name", ["s1", "s2"])
def test_the_page_json_is_shape_5(name: str) -> None:
    document = json.loads(serialize_report(SHAPES[name]().page(), "json"))
    assert CAMPAIGN_JSON_SHAPE == 5 and document["shape"] == 5
