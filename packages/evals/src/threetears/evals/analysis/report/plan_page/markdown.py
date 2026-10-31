"""A planned page as Markdown: the HTML page's parts in its order, every chart printed as its table of values.

Line for line what :func:`~threetears.evals.analysis.report.plan_page.html.planned_page_html` lays out, so a reader of
Markdown checks every number the page draws: the masthead line, the headline as the page's heading, the lede, the
flag and the confidence (its clauses, then code's checked lines), the decision's slots, the answers, the options
table, each summary chart's values, the arms key, each evidence section (its eyebrow, headline, takeaway, chart and
caption), the explore part and Full results where the page holds them, Method, the made line, and the appendix the
caller passes (the per-aspect folds) under its own heading.

Nothing quoted is structure: planner prose is laid out by the writers every page shares
(:mod:`~threetears.evals.analysis.report.md_parts`), which escape a line's opening so it opens no block, and print a
table cell on one line with its pipes escaped.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from threetears.evals.analysis.report.blocks import ChartBlock
from threetears.evals.analysis.report.md_parts import chart_lines, inert, one_line, paragraphs, table_lines
from threetears.evals.analysis.report.plan_page.model import PageChart, PageConfidence, PageMasthead, PlannedPage
from threetears.evals.analysis.report.plan_page.words import (
    APPENDIX_TITLE,
    CONFIDENCE_LABEL,
    FULL_RESULTS_TITLE,
    RUN_TEMPLATE,
    SYNTHETIC,
)

__all__ = ["planned_page_markdown"]

#: How a checked line is marked in a list: code's, not a caveat left open.
_CHECKED_MARK = "(checked)"


def _masthead(masthead: PageMasthead) -> str:
    parts = [
        masthead.kind,
        masthead.system,
        masthead.counts,
        RUN_TEMPLATE.format(date=masthead.run_date) if masthead.run_date else "",
        SYNTHETIC if masthead.synthetic else "",
    ]
    return inert(" · ".join(one_line(part) for part in parts if part))


def _confidence(confidence: PageConfidence) -> list[str]:
    lines = [inert(f"**{CONFIDENCE_LABEL}: {confidence.words}.**")]
    items = [f"- {inert(one_line(clause.text))}" for clause in confidence.clauses]
    items += [f"- {_CHECKED_MARK} {inert(one_line(line))}" for line in confidence.checked]
    return [*lines, *([""] + items if items else [])]


def _chart(chart: PageChart) -> list[str]:
    block = chart.chart
    lines = (
        chart_lines(block)
        if isinstance(block, ChartBlock)
        else [f"**{one_line(block.title)}**", "", *table_lines(block)]
    )
    return [*lines, *(line for note in chart.notes for line in ("", inert(note)))]


def _lines(page: PlannedPage) -> Iterator[list[str]]:
    """Each part's lines, in the page's order; a blank line goes between parts."""
    yield [_masthead(page.masthead)]
    yield [f"# {one_line(page.headline)}"]
    yield paragraphs(page.lede)
    if page.flag is not None:
        yield [inert(f"**{one_line(page.flag.title)}** {one_line(page.flag.text)}")]
    yield _confidence(page.confidence)
    for slot in page.decision:
        said = " ".join(one_line(line) for line in slot.text.splitlines())
        yield [inert(f"**{slot.title}.** {said} {slot.note}" if slot.note else f"**{slot.title}.** {said}")]
    for answer in page.answers:
        yield [inert(f"**{answer.warning}** {one_line(answer.text)}")]
    yield table_lines(page.options.table)
    for chart in page.summary:
        yield _chart(chart)
    if page.arms_key is not None:
        yield [f"## {page.arms_key.eyebrow}", "", inert(page.arms_key.takeaway), "", *table_lines(page.arms_key.table)]
    for section in page.sections:
        lines = [f"## {one_line(section.headline)}", "", inert(f"_{one_line(section.eyebrow)}_")]
        if section.takeaway:
            lines += ["", *paragraphs(section.takeaway)]
        lines += ["", *_chart(section.chart)]
        if section.caption:
            lines += ["", *paragraphs(section.caption)]
        yield lines
    if page.explore is not None:
        yield [
            f"## {one_line(page.explore.title)}",
            "",
            *table_lines(page.explore.flips),
            "",
            inert(page.explore.holds),
        ]
    if page.full_results:
        lines = [f"## {FULL_RESULTS_TITLE}"]
        for table in page.full_results:
            lines += ["", f"**{one_line(table.title)}**", "", *table_lines(table)]
        yield lines
    if page.method:
        yield [f"- **{one_line(item.term)}** {one_line(item.text)}" for item in page.method]
    if page.made:
        yield [inert(f"_{one_line(page.made)}_")]


def planned_page_markdown(page: PlannedPage, *, appendix: Sequence[str] = ()) -> str:
    """Write a planned page as Markdown (this module's account).

    :param page: The page.
    :ptype page: PlannedPage
    :param appendix: Lines this package wrote (the campaign's per-aspect folds), laid out under the appendix's heading
        after the made line; none by default.
    :ptype appendix: Sequence[str]
    :return: The Markdown, ending in a newline.
    :rtype: str
    """
    out: list[str] = []
    for lines in _lines(page):
        if out:
            out.append("")
        out += lines
    if appendix:
        out += ["", f"## {APPENDIX_TITLE}", *([""] if appendix[0] else []), *appendix]
    return "\n".join(out) + "\n"
