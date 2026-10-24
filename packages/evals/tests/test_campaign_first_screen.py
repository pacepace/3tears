"""A campaign's first screen, in order: code's warnings, the question, the writer's summary and recommendation, code's
readings, then the person's call.

**Who changes what here.** Chunk 03 of the senior-report plan owns this module and the order. Each part's marker
sits in one row of a table below, tagged by the chunk that lays the part out, so a chunk that changes its part edits
its own rows (04 the scorecard's, :data:`_SCORECARD`) and nothing else.
"""

from __future__ import annotations

import pytest

from threetears.evals.analysis.report.campaign_page import NO_CALL, NO_RECOMMENDATION
from threetears.evals.analysis.report.model import TableBlock

from packages.evals.tests.campaign_page_support import (
    SUMMARY,
    first_screen_markdown,
    header_html,
    kind_page_of,
    opening,
    page_of,
    recommending_nothing,
    two_aspect_memo,
)
from packages.evals.tests.campaign_support import CANDIDATE_LABEL, CONTROL_LABEL, QUESTION

#: Where code's readings are found on the first screen, by the campaign's kind (chunk 04's rows): the table's name,
#: its caption in Markdown, and its mark in HTML.
_SCORECARD: dict[str, tuple[str, str, str]] = {
    "change": ("aspects", "**By aspect**", 'data-table="aspects"'),
    "measurement": ("measures", "**By aspect, measure by measure**", 'data-table="measures"'),
}

#: A change's first screen in Markdown, in order, each marker by the chunk that lays its part out.
_CHANGE_MARKDOWN: list[tuple[str, str]] = [
    ("03", "**Floor broken on billing"),
    ("03", f'"{QUESTION}"'),
    ("memo", SUMMARY),
    ("memo", f"**Recommended:** {CONTROL_LABEL} (high confidence)"),
    ("memo", "- The billing floor is broken"),
    ("04", _SCORECARD["change"][1]),
    ("memo", NO_CALL),
]

#: The same first screen in HTML, in order.
_CHANGE_HTML: list[tuple[str, str]] = [
    ("03", 'role="alert"'),
    ("03", "<h1>"),
    ("memo", 'data-role="summary"'),
    ("memo", 'data-role="recommendation"'),
    ("04", _SCORECARD["change"][2]),
]


async def test_both_aspects_readings_are_on_the_first_screen_and_the_broken_floor_is_first() -> None:
    page = await page_of("billing", "tone")
    name, _, mark = _SCORECARD["change"]
    table = next(block for block in page.blocks if isinstance(block, TableBlock) and block.name == name)
    assert [(row["aspect"].text, row["arm"].text, row["reading"].text) for row in table.rows] == [
        ("billing", CANDIDATE_LABEL, "shown worse"),
        ("tone", CANDIDATE_LABEL, "shown better"),
    ]
    first = first_screen_markdown(page)
    assert first.startswith("**Floor broken on billing: v2's pass rate falls past the 10-point margin")
    assert "shown worse" in first and "shown better" in first
    header = header_html(page)
    assert header.index('role="alert"') < header.index("<h1>") < header.index(mark)
    assert "shown worse" in header and "shown better" in header
    # The broken floor is code's: the readings carry it, and the page warns once.
    assert [status.floor for status in page.readings.broken_floors()] == ["billing/rate_drop/*"]
    assert header.count('role="alert"') == 1


@pytest.mark.parametrize(("screen", "markers"), [("markdown", _CHANGE_MARKDOWN), ("html", _CHANGE_HTML)])
async def test_the_first_screen_runs_in_order_floor_question_summary_recommendation_table_call(
    screen: str, markers: list[tuple[str, str]]
) -> None:
    """A change with a recommendation reads its summary, then the recommendation, then the aspect table."""
    page = await page_of("billing", "tone", memo=two_aspect_memo())
    said = first_screen_markdown(page) if screen == "markdown" else header_html(page)
    order = [said.index(marker) for _, marker in markers]
    assert order == sorted(order)
    assert page.basis == "analysis" and page.finding_count == 1
    assert opening(page) == [("text", "summary"), ("text", "recommendation"), ("text", "reason")]


async def test_a_code_only_page_has_no_summary_and_says_no_recommendation_was_written_before_the_table() -> None:
    page = await page_of("billing", "tone")
    first = first_screen_markdown(page)
    assert first.index(f'"{QUESTION}"') < first.index(NO_RECOMMENDATION) < first.index(_SCORECARD["change"][1])
    assert opening(page)[0] == ("memo", NO_RECOMMENDATION)


async def test_a_measurement_opens_after_its_warning_on_its_summary_then_its_measures() -> None:
    summary = "Alpha passes most cases on solo and costs too much a reply to meet its bar."
    page = await kind_page_of("solo", memo=recommending_nothing(summary))
    _, caption, mark = _SCORECARD["measurement"]
    assert page.readings.kind == "measurement"
    assert opening(page) == [("text", "summary")]
    first = first_screen_markdown(page)
    assert first.startswith("**Bar missed on solo:")
    assert first.index("# ") < first.index(summary) < first.index(caption)
    header = header_html(page)
    assert header.index('role="alert"') < header.index("<h1>") < header.index('data-role="summary"')
    assert header.index('data-role="summary"') < header.index(mark)
