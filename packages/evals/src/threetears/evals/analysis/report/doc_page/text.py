"""A report document's page as text: what a reader reads on it, every block as its values table.

Two readings of one page, both deterministic (a page reads to the same text every time):

- :func:`doc_page_text`, the whole page, for a reader that cannot see the picture: the editor's input, and what an
  analyst's ``report_check`` returns for a draft. Each claim is headed by its id and weight (the editor's findings name
  a claim by its id), and each block by its kind and what code fixed about it (its interval level, unit, axis and
  reference line), then its values table.
- :func:`above_appendix_text`, the text a reader reads above the appendix, as it is printed and nothing more (no claim
  id, no block's kind): the page's half of the invariant that no internal id appears above the appendix. Example
  cards are left out of it: they show a trial's and a case's ids so a reader can look the trial up, which the
  invariant exempts, and only the renderer draws them. Besides its values table, each chart gives its title (the
  values table's caption, which a screen reader reads and the page shows where the chart is not drawn), then every
  string its drawn chart carries for a reader (:meth:`~threetears.evals.analysis.report.blocks.ChartBlock.kit_chart`'s
  data, as resolved, never the drawing script's): legend labels, tooltip lines, short and narrow labels, notes, group
  heads and axis titles, each once, in the data's order. Left out are the fields that only select a mark (an ``id``, the
  ``ids`` a path joins, the ``emphasis`` a chart stresses) and colour tokens.

Both carry, beside each claim and each Method term, the labels of the caveats the document lands there.

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
    CAVEATS_TEMPLATE,
    CHECKED_MARK,
    FINDINGS_TITLE,
    LEVEL_WORDS,
    METHOD_TITLE,
    NEXT_TITLE,
    OMITTED_TITLE,
)
from threetears.evals.analysis.report.md_parts import markdown_table, one_line
from threetears.evals.analysis.report.plan_page.words import CONFIDENCE_LABEL, OPTIONS_TITLE, RUN_TEMPLATE, SYNTHETIC

#: The entry points, with their callers: the writing loop's editor reads ``doc_page_text``, and the store gate checks
#: ``above_appendix_text`` for internal ids.
__all__ = ["above_appendix_text", "doc_page_text"]


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


#: The kit data's fields that select a mark and are never shown: what ``emphasis`` and a path name.
_UNSHOWN = frozenset({"id", "ids", "emphasis"})

#: How a kit colour token opens: a token, never text.
_TOKEN = "--"


def _shown(node: object, field: str = "") -> Iterator[str]:
    """Every string a reader is shown in a chart's kit data, in its order."""
    if field in _UNSHOWN:
        return
    if isinstance(node, str):
        if not node.startswith(_TOKEN):
            yield node
    elif isinstance(node, dict):
        for key, value in node.items():
            yield from _shown(value, str(key))
    elif isinstance(node, list):
        for value in node:
            yield from _shown(value, field)


def _drawn_text(block: ChartBlock | TableBlock) -> list[str]:
    """Every string a chart shows a reader beyond its values, each once: its title, which its values table prints as
    its caption, then what its drawing shows; none for a table, which its values table is."""
    if not isinstance(block, ChartBlock):
        return []
    drawn = block.kit_chart()
    return list(dict.fromkeys([block.title, *(_shown(drawn[1]) if drawn is not None else ())]))


def _caveats(labels: list[str]) -> str:
    """The caveats landed beside a claim or a Method term, by label, as code's line."""
    return CAVEATS_TEMPLATE.format(labels="; ".join(one_line(label) for label in labels))


def _evidence(evidence: DocEvidence, *, editor: bool) -> list[str]:
    lines = [_fixed(evidence)] if editor else []
    if evidence.chart is not None and (editor or not _cards(evidence)):
        lines += _table(evidence.chart.chart)
        lines += evidence.chart.notes
        if not editor:
            lines += _drawn_text(evidence.chart.chart)
    lines.append(one_line(evidence.caption))
    if evidence.degraded:
        lines.append(evidence.degraded)
    return lines


def _claim(claim: DocClaim, *, editor: bool) -> list[str]:
    lines = [f"## Claim {claim.id} ({claim.weight})"] if editor else []
    lines.append(one_line(claim.text) if editor else f"## {one_line(claim.text)}")
    if claim.caveats:
        lines.append(_caveats(claim.caveats))
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
            f"- {CHECKED_MARK + ' ' if line.checked else ''}{one_line(line.label)}. {one_line(line.text)}"
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
    yield [
        f"## {METHOD_TITLE}",
        *(
            f"- {one_line(item.term)}: {one_line(item.text)}" + (f" {_caveats(item.caveats)}" if item.caveats else "")
            for item in page.method
        ),
    ]
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
