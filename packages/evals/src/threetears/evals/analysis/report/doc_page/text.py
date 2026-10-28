"""A report document's page as text: what a reader reads on it, every block as its values table.

Two readings of one page, both deterministic (a page reads to the same text every time):

- :func:`doc_page_text`, the whole page, for a reader that cannot see the picture: the editor's input, and what an
  analyst's ``report_check`` returns for a draft. Each claim is headed by its id and weight (the editor's findings name
  a claim by its id), and each block by its kind and what code fixed about it (its interval level, unit, axis and
  reference line), then its values table.
- :func:`above_appendix_text`, the text a reader reads above the appendix, as it is printed and nothing more (no claim
  id, no block's kind): the page's half of the invariant that no internal id appears above the appendix. Example
  cards are left out of it: they show a trial's and a case's ids so a reader can look the trial up, which the
  invariant exempts, and only the renderer draws them.

Every block's values are :func:`~threetears.evals.analysis.analyst.blocks.values_table`'s, the table ``block_render``
returns, so the editor reads a block's values as the analyst read them. A table is laid out in Markdown's pipes.
"""

from __future__ import annotations

from collections.abc import Iterator

from threetears.evals.analysis.analyst.blocks import values_table
from threetears.evals.analysis.report.blocks import ChartBlock, TableBlock
from threetears.evals.analysis.report.charts import ExamplesChart
from threetears.evals.analysis.report.doc_page.model import DocClaim, DocEvidence, DocPage
from threetears.evals.analysis.report.doc_page.words import (
    APPENDIX_TITLE,
    CHECKED_MARK,
    FINDINGS_TITLE,
    LEVEL_WORDS,
    NEXT_TITLE,
    OMITTED_TITLE,
)
from threetears.evals.analysis.report.md_parts import markdown_table, one_line
from threetears.evals.analysis.report.plan_page.words import CONFIDENCE_LABEL, OPTIONS_TITLE, RUN_TEMPLATE, SYNTHETIC

#: Empty until a production caller imports this module's entry point (``doc_page_text``, ``above_appendix_text``): its
#: callers are the writing loop's editor and ``report_check`` (the page text), and the report read (the page);
#: ``tests/test_doc_page_exports.py`` lists the name as awaiting one, and goes red when it has one, so it joins this
#: list then.
__all__: list[str] = []


def _table(block: ChartBlock | TableBlock) -> list[str]:
    headers, rows = values_table(block)
    return markdown_table(headers, rows) if rows else ["(no rows)"]


def _cards(evidence: DocEvidence) -> bool:
    """Whether a block is drawn as example cards, which show trial and case ids by design."""
    if evidence.chart is None:
        return False
    block = evidence.chart.chart
    return isinstance(block, ChartBlock) and isinstance(block.data, ExamplesChart)


def _fixed(evidence: DocEvidence) -> str:
    """What code fixed about a block, for the editor: its kind, then its level, unit, axis and reference line."""
    said = [LEVEL_WORDS[name].format(value=value) for name, value in _facts(evidence) if value]
    return f"[{evidence.block}{'; ' + '; '.join(said) if said else ''}]"


def _facts(evidence: DocEvidence) -> Iterator[tuple[str, str]]:
    yield "level", evidence.level
    yield "unit", evidence.unit
    yield "domain", evidence.domain
    yield "reference", evidence.reference


def _evidence(evidence: DocEvidence, *, editor: bool) -> list[str]:
    lines = [_fixed(evidence)] if editor else []
    if evidence.chart is not None and (editor or not _cards(evidence)):
        lines += _table(evidence.chart.chart)
        lines += evidence.chart.notes
    lines.append(one_line(evidence.caption))
    if evidence.degraded:
        lines.append(evidence.degraded)
    return lines


def _claim(claim: DocClaim, *, editor: bool) -> list[str]:
    lines = [f"## Claim {claim.id} ({claim.weight})"] if editor else []
    lines.append(one_line(claim.text) if editor else f"## {one_line(claim.text)}")
    for evidence in claim.evidence:
        lines += ["", *_evidence(evidence, editor=editor)]
    return lines


def _above(page: DocPage, *, editor: bool) -> Iterator[list[str]]:
    """Each part above the appendix, its lines, in the page's order."""
    masthead = page.masthead
    yield [
        " · ".join(
            part
            for part in (
                masthead.kind,
                masthead.system,
                masthead.counts,
                RUN_TEMPLATE.format(date=masthead.run_date) if masthead.run_date else "",
                SYNTHETIC if masthead.synthetic else "",
            )
            if part
        )
    ]
    if page.degraded:
        yield [page.degraded]
    yield [f"# {one_line(page.headline)}"]
    yield [one_line(page.lede)]
    if page.flag:
        yield [one_line(page.flag)]
    confidence = page.confidence
    yield [
        f"{CONFIDENCE_LABEL}: {confidence.words}.",
        *(
            f"- {CHECKED_MARK + ' ' if line.checked else ''}{line.label}. {one_line(line.text)}"
            for line in confidence.caveats
        ),
    ]
    yield [f"{slot.title}. {one_line(slot.text)}" for slot in page.decision]
    yield [OPTIONS_TITLE, *_table(page.options)]
    for claim in page.claims:
        yield _claim(claim, editor=editor)
    if page.explore is not None:
        yield [f"## {page.explore.title}", *_table(page.explore)]
    if page.next:
        yield [f"## {NEXT_TITLE}", *(f"- {one_line(step.text)} {one_line(step.settles)}" for step in page.next)]


def _appendix(page: DocPage) -> Iterator[list[str]]:
    """Each part of the appendix, its lines, in the page's order."""
    yield [f"# {APPENDIX_TITLE}"]
    key = page.arms_key
    yield [f"## {key.eyebrow}", key.takeaway, *_table(key.table)]
    for table in page.full_results:
        yield [f"## {table.title}", *_table(table)]
    if page.omitted:
        yield [f"## {OMITTED_TITLE}", *(f"- {one_line(item.what)}: {one_line(item.why)}" for item in page.omitted)]
    yield [f"- {one_line(item.term)}: {one_line(item.text)}" for item in page.method]
    if page.findings:
        yield [
            f"## {FINDINGS_TITLE}",
            *(f"- {finding.severity} · {finding.about}: {one_line(finding.finding)}" for finding in page.findings),
        ]
    if page.made:
        yield [page.made]


def _joined(parts: Iterator[list[str]]) -> str:
    out: list[str] = []
    for lines in parts:
        if out:
            out.append("")
        out += lines
    return "\n".join(out) + "\n"


def doc_page_text(page: DocPage) -> str:
    """The whole page as text, every block as its values table, each claim by its id (this module's account).

    :param page: The page.
    :ptype page: DocPage
    :return: The text, ending in a newline.
    :rtype: str
    """

    def parts() -> Iterator[list[str]]:
        yield from _above(page, editor=True)
        yield from _appendix(page)

    return _joined(parts())


def above_appendix_text(page: DocPage) -> str:
    """The text a reader reads above the page's appendix, as printed, example cards left out (this module's account).

    :param page: The page.
    :ptype page: DocPage
    :return: The text, ending in a newline.
    :rtype: str
    """
    return _joined(_above(page, editor=False))
