"""A report document's page as Markdown: the HTML page's parts in its order, every block printed as its values table.

Line for line what :func:`~threetears.evals.analysis.report.doc_page.html.doc_page_html` lays out, so a reader of
Markdown checks every number the page draws: the masthead line, a draft's line (code's, saying its editor's review
left findings open), code's note where part of a stored document no longer reads, the headline as the page's heading,
the lede, the flag, the confidence (each caveat's label, then its clause or code's checked line), the decision's
entries, the options table, each claim (its text as a heading, the labels of
the caveats it lands, each block's values, notes, caption and code's lines), explore and the next steps, then the
appendix under its heading: the arms key, every table the input holds, what was left out and why, Method under its
own heading (each term with the labels of the caveats it lands), the review findings left open, and the made line.

A block's values are :func:`~threetears.evals.analysis.analyst.blocks.values_table`'s, the table the analyst and the
editor read. Nothing quoted is structure: the document's prose is laid out by the writers every page shares
(:mod:`~threetears.evals.analysis.report.md_parts`), which escape a line's opening so it opens no block, and print a
table cell on one line with its pipes escaped.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from threetears.evals.analysis.analyst.blocks import values_table
from threetears.evals.analysis.report.blocks import ChartBlock, TableBlock
from threetears.evals.analysis.report.doc_page.model import DocClaim, DocPage, caveats_line, draft_line
from threetears.evals.analysis.report.doc_page.words import (
    APPENDIX_TITLE,
    CHECKED_MARK,
    EVIDENCE_TITLE,
    FINDINGS_TITLE,
    METHOD_TITLE,
    NEXT_TITLE,
    OMITTED_TITLE,
)
from threetears.evals.analysis.report.md_parts import inert, markdown_table, one_line, paragraphs, table_lines
from threetears.evals.analysis.report.plan_page.model import PageMasthead
from threetears.evals.analysis.report.plan_page.words import CONFIDENCE_LABEL, RUN_TEMPLATE, SYNTHETIC

#: The module's entry point, which the campaign page's serializer calls for a report document's page (the report
#: read).
__all__ = ["doc_page_markdown"]

#: Each claim weight's heading level: its type size, and nothing else.
_WEIGHT_HEADINGS = {"lead": "## ", "support": "### ", "detail": "#### "}


def _masthead(masthead: PageMasthead) -> str:
    parts = [
        masthead.kind,
        masthead.system,
        masthead.counts,
        RUN_TEMPLATE.format(date=masthead.run_date) if masthead.run_date else "",
        SYNTHETIC if masthead.synthetic else "",
    ]
    return inert(" · ".join(one_line(part) for part in parts if part))


def _values(block: ChartBlock | TableBlock) -> list[str]:
    headers, rows = values_table(block)
    return markdown_table(headers, rows) if rows else ["_(no rows)_"]


def _caveats(labels: list[str]) -> str:
    """The caveats landed beside a claim or a Method term, by label, as code's line (:func:`caveats_line`)."""
    return caveats_line(labels)


def _claim(claim: DocClaim) -> list[str]:
    lines = [f"{_WEIGHT_HEADINGS[claim.weight]}{one_line(claim.text)}"]
    if claim.caveats:
        lines += ["", inert(f"_{_caveats(claim.caveats)}_")]
    for evidence in claim.evidence:
        if evidence.chart is not None:
            lines += ["", *_values(evidence.chart.chart)]
            lines += [line for note in evidence.chart.notes for line in ("", inert(note))]
        lines += ["", *paragraphs(evidence.caption)]
        if evidence.degraded:
            lines += ["", inert(f"_{evidence.degraded}_")]
    return lines


def _lines(page: DocPage) -> Iterator[list[str]]:
    """Each part's lines, in the page's order; a blank line goes between parts."""
    yield [_masthead(page.masthead)]
    if page.draft:
        yield [inert(f"**{draft_line(page)}**")]
    if page.degraded:
        yield [inert(f"**{page.degraded}**")]
    yield [f"# {one_line(page.headline)}"]
    yield paragraphs(page.lede)
    if page.flag:
        yield [inert(f"**{one_line(page.flag)}**")]
    confidence = page.confidence
    items = [
        f"- {CHECKED_MARK + ' ' if line.checked else ''}**{one_line(line.label)}.** {inert(one_line(line.text))}"
        for line in confidence.caveats
    ]
    yield [inert(f"**{CONFIDENCE_LABEL}: {confidence.words}.**"), *([""] + items if items else [])]
    for slot in page.decision:
        yield [inert(f"**{slot.title}.** {one_line(slot.text)}")]
    yield table_lines(page.options)
    for claim in page.claims:
        yield _claim(claim)
    if page.explore is not None:
        yield [f"## {page.explore.title}", "", *table_lines(page.explore)]
    if page.next:
        yield [
            f"## {NEXT_TITLE}",
            "",
            *(f"- {inert(one_line(step.text))} _{one_line(step.settles)}_" for step in page.next),
        ]
    yield [f"## {APPENDIX_TITLE}"]
    key = page.arms_key
    yield [f"### {key.eyebrow}", "", inert(key.takeaway), "", *table_lines(key.table)]
    for table in page.full_results:
        yield [f"### {one_line(table.title)}", "", *table_lines(table)]
    if page.omitted:
        yield [
            f"### {OMITTED_TITLE}",
            "",
            *(f"- **{one_line(item.what)}** {one_line(item.why)}" for item in page.omitted),
        ]
    yield [
        f"### {METHOD_TITLE}",
        "",
        *(
            f"- **{one_line(item.term)}** {one_line(item.text)}"
            + (f" _{_caveats(item.caveats)}_" if item.caveats else "")
            for item in page.method
        ),
    ]
    if page.findings:
        yield [
            f"### {FINDINGS_TITLE}",
            "",
            *(f"- **{finding.severity}** · {finding.about}: {one_line(finding.finding)}" for finding in page.findings),
        ]
    if page.made:
        yield [inert(f"_{one_line(page.made)}_")]


def doc_page_markdown(page: DocPage, *, record: Sequence[str] = ()) -> str:
    """Write a report document's page as Markdown (this module's account).

    :param page: The page.
    :ptype page: DocPage
    :param record: Lines this package wrote (the campaign's folds, each failed or lost trial's record in them), laid
        out after the document under their own heading, the document left as it is; none by default.
    :ptype record: Sequence[str]
    :return: The Markdown, ending in a newline.
    :rtype: str
    """
    out: list[str] = []
    for lines in _lines(page):
        if out:
            out.append("")
        out += lines
    if record:
        out += ["", f"## {EVIDENCE_TITLE}", *record]
    return "\n".join(out) + "\n"
