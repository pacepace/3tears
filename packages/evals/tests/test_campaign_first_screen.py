"""A campaign's first screen, in order: the title, the question, the writer's summary and recommendation, what warns,
then the person's call; code's readings follow it, under the Scorecard. Nothing stands above the title, and nothing on
it is announced over it.

**Who changes what here.** Chunk 03 of the senior-report plan owns this module and the order. Each part's marker
sits in one row of a table below, tagged by the chunk that lays the part out, so a chunk that changes its part edits
its own rows (04 the scorecard's, :data:`_SCORECARD`) and nothing else.
"""

from __future__ import annotations

import pytest

import re
from html import escape

from threetears.evals.analysis import campaign_report_html, campaign_report_markdown
from threetears.evals.analysis.report.campaign_page import NO_CALL, NO_RECOMMENDATION
from threetears.evals.analysis.report.model import CampaignReport, DisclosureBlock, TableBlock, TableCell, TableColumn

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
from packages.evals.tests.report_shapes_support import SHAPES, s1_change, s4_change

#: Where code's readings are found, under the Scorecard after the first screen, by the campaign's kind (chunk 04's
#: rows): the table's name, its caption in Markdown, and its mark in HTML.
_SCORECARD: dict[str, tuple[str, str, str]] = {
    "change": ("aspects", "**By aspect**", 'data-table="scorecard"'),
    "measurement": ("measures", "**alpha by aspect**", 'data-table="scorecard"'),
}

#: The line a change's first screen warns of billing's broken floor in.
_FLOOR_WARNING = "Adopting v2 overrides the 10-point floor it breaks on billing."

#: A change's first screen in Markdown, in order, each marker by the chunk that lays its part out.
_CHANGE_MARKDOWN: list[tuple[str, str]] = [
    ("03", "# support queue"),
    ("03", f'"{QUESTION}"'),
    ("memo", SUMMARY),
    ("memo", f"**Recommended:** {CONTROL_LABEL} (high confidence)"),
    ("memo", "- The billing floor is broken"),
    ("03", f"**{_FLOOR_WARNING}**"),
    ("memo", NO_CALL),
]

#: The same first screen in HTML, in order.
_CHANGE_HTML: list[tuple[str, str]] = [
    ("03", "<h1>"),
    ("03", '<p class="question">'),
    ("memo", 'data-role="summary"'),
    ("memo", 'data-role="recommendation"'),
    ("03", '<p class="warning">'),
]


async def test_both_aspects_readings_are_in_the_scorecard_after_the_broken_floor_s_warning() -> None:
    page = await page_of("billing", "tone")
    name, _, mark = _SCORECARD["change"]
    table = next(block for block in page.blocks if isinstance(block, TableBlock) and block.name == name)
    assert [(row["aspect"].text, row["arm"].text, row["reading"].text) for row in table.rows] == [
        ("billing", CANDIDATE_LABEL, "shown worse"),
        ("tone", CANDIDATE_LABEL, "shown better"),
    ]
    first = first_screen_markdown(page)
    assert first.startswith("# support queue\n")
    assert first.index(f'"{QUESTION}"') < first.index(f"**{_FLOOR_WARNING}**")
    # Code's readings follow what warns, wherever the scorecard lays them out.
    markdown, html = campaign_report_markdown(page), campaign_report_html(page)
    assert markdown.index(f"**{_FLOOR_WARNING}**") < markdown.index(_SCORECARD["change"][1])
    assert "shown worse" in markdown and "shown better" in markdown
    header = header_html(page)
    assert header.index("<h1>") < header.index('<p class="warning">')
    assert html.index('<p class="warning">') < html.index(mark)
    # The broken floor is code's: the readings carry it, and the page warns once, never announced over the title.
    assert [status.floor for status in page.readings.broken_floors()] == ["billing/rate_drop/*"]
    assert header.count('<p class="warning">') == 1 and 'role="alert"' not in header


@pytest.mark.parametrize(("screen", "markers"), [("markdown", _CHANGE_MARKDOWN), ("html", _CHANGE_HTML)])
async def test_the_first_screen_runs_title_question_summary_recommendation_warning_call(
    screen: str, markers: list[tuple[str, str]]
) -> None:
    """A change with a recommendation reads its summary, then the recommendation, then what warns, then the call; code's
    table after what warns, wherever the scorecard lays it out."""
    page = await page_of("billing", "tone", memo=two_aspect_memo())
    said = first_screen_markdown(page) if screen == "markdown" else header_html(page)
    order = [said.index(marker) for _, marker in markers]
    assert order == sorted(order)
    if screen == "markdown":
        whole, warning, table = campaign_report_markdown(page), f"**{_FLOOR_WARNING}**", _SCORECARD["change"][1]
    else:
        whole, warning, table = campaign_report_html(page), '<p class="warning">', _SCORECARD["change"][2]
    assert whole.index(warning) < whole.index(table)
    assert page.basis == "analysis" and page.finding_count == 1
    assert opening(page) == [("text", "summary"), ("text", "recommendation"), ("text", "reason")]


async def test_a_code_only_page_has_no_summary_and_says_no_recommendation_was_written_before_the_table() -> None:
    page = await page_of("billing", "tone")
    first = first_screen_markdown(page)
    assert first.index(f'"{QUESTION}"') < first.index(NO_RECOMMENDATION)
    # Code's table is the Scorecard's, after the first screen.
    assert _SCORECARD["change"][1] not in first
    markdown = campaign_report_markdown(page)
    assert (
        markdown.index(NO_RECOMMENDATION) < markdown.index("\n## Scorecard\n") < markdown.index(_SCORECARD["change"][1])
    )
    assert opening(page)[0] == ("memo", NO_RECOMMENDATION)


async def test_a_measurement_opens_on_its_summary_then_its_warning_then_its_measures() -> None:
    summary = "Alpha passes most cases on solo and costs too much a reply to meet its bar."
    page = await kind_page_of("solo", memo=recommending_nothing(summary))
    _, caption, mark = _SCORECARD["measurement"]
    assert page.readings.kind == "measurement"
    assert opening(page) == [("text", "summary")]
    first = first_screen_markdown(page)
    assert first.startswith("# ")
    assert first.index(summary) < first.index("**alpha falls short of the bar on solo.**")
    markdown, html = campaign_report_markdown(page), campaign_report_html(page)
    assert markdown.index("**alpha falls short of the bar on solo.**") < markdown.index(caption)
    header = header_html(page)
    assert header.index("<h1>") < header.index('data-role="summary"') < header.index('<p class="warning">')
    assert html.index('<p class="warning">') < html.index(mark)


@pytest.mark.parametrize("name", ["s1", "s2", "s4"])
def test_the_first_thing_after_the_title_is_the_question_and_nothing_stands_above_the_title(name: str) -> None:
    page = SHAPES[name]().page()
    question = next(block for block in page.blocks if isinstance(block, DisclosureBlock) and block.source == "basis")
    html = campaign_report_html(page)
    header = html[html.index("<header>") + len("<header>") : html.index("</header>")]
    assert header.lstrip().startswith("<h1>")
    after_title = header[header.index("</h1>") + len("</h1>") :].lstrip()
    assert after_title.startswith(f'<p class="question">{escape(question.text)}</p>')
    assert 'role="alert"' not in header
    markdown = campaign_report_markdown(page).splitlines()
    assert markdown[0] == f"# {page.title}" and markdown[1] == question.text


@pytest.mark.parametrize("name", ["s1", "s2", "s3"])
def test_each_warning_is_one_plain_line_per_arm_with_no_expression_or_figure_of_the_tables(name: str) -> None:
    page = SHAPES[name]().page()
    warned = [block for block in page.blocks if block.section == "warnings" and isinstance(block, DisclosureBlock)]
    lines = [block for block in warned if block.source in ("bar", "floor", "guardrails")]
    assert lines, "each of these shapes warns"
    arms = list(page.readings.arm_labels().values())
    said: list[tuple[str, str]] = []
    for block in lines:
        (arm,) = [arm for arm in arms if re.search(rf"(?<!\w){re.escape(arm)}(?!\w)", block.text)]
        said.append((block.source, arm))
        # No code span and no interval: the tables print those.
        assert "`" not in block.text and "[" not in block.text and "%" not in block.text
    assert len(said) == len(set(said)), "one line per arm per kind"


def test_a_page_with_nothing_that_warns_has_no_warnings_section() -> None:
    page = s4_change().page()
    assert not [block for block in page.blocks if block.section == "warnings"]
    assert '<p class="warning">' not in campaign_report_html(page)


def test_a_matrix_draws_every_column_s_glyphs_on_one_scale_so_they_compare_across_columns() -> None:
    page = s4_change().page()

    def cell(value: float, low: float, high: float) -> TableCell:
        return TableCell(value=value, low=low, high=high, zero_line=True)

    columns = [
        TableColumn(key="aspect", header="Aspect"),
        TableColumn(key="a", header="a"),
        TableColumn(key="b", header="b"),
    ]
    rows = [{"aspect": TableCell(value="billing"), "a": cell(1, -2, 4), "b": cell(30, 20, 40)}]
    # The table says whether it is drawn on one scale; its name decides nothing.
    tables = [
        TableBlock(
            section="scorecard",
            name=name,
            title="t",
            columns=columns,
            rows=rows,
            order="o",
            total_rows=1,
            shared_scale=shared,
        )
        for name, shared in (("probe", True), ("other", False))
    ]

    def axis(table: TableBlock) -> str:
        html = campaign_report_html(page.model_copy(update={"blocks": [*page.blocks, table]}))
        start = html.index(f'data-table="{table.name}"')
        line = html.index('<p class="axis">', start)
        return html[line : html.index("</p>", line)]

    matrix, other = (axis(table) for table in tables)
    assert matrix.startswith('<p class="axis">Each column drawn from ') and " · " not in matrix
    assert "a drawn from" in other and "b drawn from" in other
    # The page's own matrix is one.
    (own,) = [block for block in page.blocks if isinstance(block, TableBlock) and block.name == "scorecard"]
    assert own.shared_scale


#: A check's expression longer than a cell shows whole: 64 characters.
_LONG = "any(item.item_id == 'i3' and item.state == 'queued' for it in q)"


def _with_table(page: CampaignReport, table: TableBlock) -> str:
    """The page's HTML with ``table`` laid out among its blocks, and that table's own markup."""
    html = campaign_report_html(page.model_copy(update={"blocks": [*page.blocks, table]}))
    start = html.index(f'data-table="{table.name}"')
    return html[start : html.index("</table>", start)]


@pytest.mark.parametrize("name", ["s1", "s2", "s3"])
def test_every_aspect_a_campaign_table_names_links_its_fold_from_the_cell_built_for_it(name: str) -> None:
    page = SHAPES[name]().page()
    folds = {fold.aspect: f"aspect-{slug}" for fold, slug in zip(page.aspects, page.fold_ids, strict=True)}
    named = {fold.aspect for fold in page.aspects}
    linked = [
        (row["aspect"].text, row["aspect"].ref)
        for block in page.blocks
        if isinstance(block, TableBlock) and any(column.key == "aspect" for column in block.columns)
        for row in block.rows
        if row["aspect"].text in named
    ]
    assert linked, "the page's tables name its aspects"
    assert all(ref == folds[aspect] for aspect, ref in linked)
    html = campaign_report_html(page)
    assert all(f'<a href="#{folds[aspect]}">{aspect}</a>' in html for aspect, _ in linked)
    assert "](#aspect-" not in campaign_report_markdown(page), "Markdown links no fold"


def test_the_html_links_a_cell_where_the_cell_says_and_reads_no_aspect_from_its_text() -> None:
    page = s1_change().page()
    columns = [TableColumn(key="aspect", header="Aspect"), TableColumn(key="arm", header="Arm")]
    rows = [
        {"aspect": TableCell(value="billing"), "arm": TableCell(value="v2")},
        {"aspect": TableCell(value="tone", ref="aspect-billing"), "arm": TableCell(value="v3")},
        {"aspect": TableCell(value="no such aspect"), "arm": TableCell(value="v4")},
    ]
    for section in ("findings", "answer"):
        table = TableBlock(
            section=section, name="probe", title="Probe", columns=columns, rows=rows, order="o", total_rows=3
        )
        markup = _with_table(page, table)
        assert "<td>billing</td>" in markup, "a cell naming an aspect links only where it says so"
        assert '<td><a href="#aspect-billing">tone</a></td>' in markup, "its own link, kept"
        assert "<td>no such aspect</td>" in markup and "<td>v2</td>" in markup, "nothing else is linked"


def test_a_long_expression_in_a_cell_or_a_header_shows_its_start_and_the_whole_one_click_away() -> None:
    page = s4_change().page()
    short = _LONG[:60]
    columns = [TableColumn(key="check", header=f"`{_LONG}`"), TableColumn(key="short", header=f"`{short}`")]
    rows = [{"check": TableCell(value=f"`{_LONG}`"), "short": TableCell(value=f"`{short}`")}]
    table = TableBlock(
        section="findings", name="probe", title="Probe", columns=columns, rows=rows, order="o", total_rows=1
    )
    markup = _with_table(page, table)
    folded = (
        f'<details class="expr"><summary><code>{escape(_LONG[:60].rstrip())}…</code></summary>'
        f"<code>{escape(_LONG)}</code></details>"
    )
    assert markup.count(folded) == 2, "the header and the cell"
    assert f"<code>{escape(short)}</code>" in markup and markup.count("<details") == 2, "60 characters show whole"
    markdown = campaign_report_markdown(page.model_copy(update={"blocks": [*page.blocks, table]}))
    assert f"`{_LONG}`" in markdown and "…" not in markdown.split("**Probe**")[1].split("\n\n")[1]


def test_a_linked_cell_holding_a_long_expression_sets_the_link_beside_its_fold_never_around_it() -> None:
    page = s1_change().page()
    columns = [TableColumn(key="aspect", header="Aspect"), TableColumn(key="value", header="Value")]
    rows = [{"aspect": TableCell(value="billing"), "value": TableCell(value=f"`{_LONG}`", ref="aspect-tone")}]
    table = TableBlock(
        section="findings", name="probe", title="Probe", columns=columns, rows=rows, order="o", total_rows=1
    )
    markup = _with_table(page, table)
    cell = re.search(r'<td[^>]*><details class="expr">.*?</td>', markup)
    assert cell is not None
    assert re.fullmatch(
        r'<td[^>]*><details class="expr"><summary>.*?</summary><code>.*?</code></details> '
        r'<a class="cell-link" href="#aspect-tone">where it is</a></td>',
        cell.group(0),
    )
    # No link anywhere holds an expandable fold.
    assert not re.search(r"<a [^>]*>(?:(?!</a>).)*<details", markup)
