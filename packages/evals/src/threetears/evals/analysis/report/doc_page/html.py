"""A report document's page as one self-contained HTML page, in the default mode (the kit's one script draws each
chart) or the static mode (nothing runs or loads; every chart is its table of values).

The page is the kit's, as the planned page lays it out (:mod:`~threetears.evals.analysis.report.plan_page.html`),
with no class, style or script of its own:

- ``.page`` > ``.mast``; where part of a stored document no longer reads, code's note as a ``.callout.warn``
  (``data-part="degraded"``);
- ``.bluf``: the ``h1`` headline, the ``.lede``, the amber ``.flag`` where the document has one, the confidence
  (``data-part="confidence"``: amber where caveats stay open or the evidence is weak, a muted ``.callout`` for plain
  high; each caveat its label in bold, then its clause, a checked one as ``li.checked``), the ``.decision`` grid (an
  entry set apart by its word as ``.rec``) and the options table (each recommended row ``tr.hl``);
- each claim a ``section`` (``data-part="claim"``, ``id="claim-<place>"``) headed by its text at its weight's size
  (an ``h2`` for a lead claim, an ``h3`` for support, a ``.takeaway`` paragraph for detail), then each evidence
  block's ``.fig`` (or the table of its values), its caption and code's lines as ``.cap``;
- the explore table and the next steps, each its own ``section``;
- the appendix under its ``.part`` divider (``data-part="appendix"``): the arms key, every table the input holds,
  what was left out and why, Method under its own heading (``dl.method``), the review findings left open, and the made
  line; the caveats a claim or a Method term lands, by label, as code's ``.cap`` line under it;
- the kit's script, last, on the default page only.

**Nothing written into a field can make the page run or load anything**, as on every page this package writes: every
text node is escaped before any markup is made of it (:mod:`~threetears.evals.analysis.report.html_parts`), and a
chart's data reaches the page only as JSON escaped into its ``data-chart`` attribute. **The static page** is the
default page less exactly the kit's script, the font links and each chart's ``div.chart``.
"""

from __future__ import annotations

from collections.abc import Sequence

from threetears.evals.analysis.report.blocks import ChartBlock, TableBlock
from threetears.evals.analysis.report.doc_page.model import (
    DocClaim,
    DocConfidence,
    DocEvidence,
    DocFinding,
    DocNext,
    DocOmitted,
    DocPage,
    DocTerm,
)
from threetears.evals.analysis.report.doc_page.words import (
    APPENDIX_TITLE,
    CAVEATS_TEMPLATE,
    FINDINGS_TITLE,
    METHOD_TITLE,
    NEXT_TITLE,
    OMITTED_TITLE,
)
from threetears.evals.analysis.report.html_parts import (
    KIT_JS,
    chart_html,
    escape,
    grid_html,
    inline_html,
    one_line,
    page_head,
    table_html,
)
from threetears.evals.analysis.report.plan_page.model import PageArmsKey, PageMasthead, PageSlot
from threetears.evals.analysis.report.plan_page.words import CONFIDENCE_LABEL, RUN_TEMPLATE, SYNTHETIC

#: The module's entry point, which the campaign page's serializer calls for a report document's page (the report
#: read).
__all__ = ["doc_page_html"]

#: The confidence forms set in the amber box: caveats stay open, or the evidence is weak.
_AMBER_FORMS = frozenset({"high_if_immaterial", "low"})

#: A wrapper the first screen's grid may narrow below its content's width, so a wide table scrolls in its own frame.
_SHRINKS = ' style="min-width:0"'

#: The flag's mark, hidden from a screen reader, which reads the words.
_FLAG_MARK = '<b aria-hidden="true">!</b>'

#: A checked caveat's list item: code's line, the kit's ``li.checked``.
_CHECKED = ' class="checked"'

#: Each claim weight's heading: its type size, and nothing else.
_WEIGHT_TAGS = {"lead": ("<h2>", "</h2>"), "support": ("<h3>", "</h3>"), "detail": ('<p class="takeaway">', "</p>")}


def _prose(text: str) -> str:
    """Analyst prose as a page line: escaped, its bold and code made markup, each of its lines broken where it
    breaks."""
    return "<br>".join(inline_html(line.strip()) for line in text.splitlines() if line.strip())


def _masthead(masthead: PageMasthead) -> str:
    spans = [
        f'<span class="kind">{escape(masthead.kind)}</span>',
        f"<span><b>{escape(masthead.system)}</b></span>",
        *([f"<span>{escape(masthead.counts)}</span>"] if masthead.counts else []),
        *([f"<span>{escape(RUN_TEMPLATE.format(date=masthead.run_date))}</span>"] if masthead.run_date else []),
        *([f"<span>{escape(SYNTHETIC)}</span>"] if masthead.synthetic else []),
    ]
    return '<div class="mast">' + "".join(spans) + "</div>"


def _confidence(confidence: DocConfidence) -> str:
    lines = [
        f"<li{_CHECKED if line.checked else ''}><b>{escape(line.label)}.</b> {_prose(line.text)}</li>"
        for line in confidence.caveats
    ]
    listed = f"<ul>{''.join(lines)}</ul>" if lines else ""
    said = f"<b>{escape(CONFIDENCE_LABEL)}: {escape(confidence.words)}.</b>{listed}"
    form = escape(confidence.form)
    if confidence.form in _AMBER_FORMS:
        return f'<div class="flag" data-part="confidence" data-form="{form}">{_FLAG_MARK}<div>{said}</div></div>'
    return f'<div class="callout" data-part="confidence" data-form="{form}">{said}</div>'


def _slot(slot: PageSlot) -> str:
    css = ' class="rec"' if slot.highlighted else ""
    return (
        f'<div{css} data-slot="{escape(slot.key)}"><span class="k">{escape(slot.title)}</span>'
        f'<span class="v">{_prose(slot.text)}</span></div>'
    )


def _block(evidence: DocEvidence, chart_id: str, *, static: bool) -> str:
    """An evidence block as the page draws it, its caption under it, then code's lines."""
    drawn = ""
    if evidence.chart is not None:
        block = evidence.chart.chart
        if isinstance(block, ChartBlock):
            drawn = chart_html(block, chart_id, static=static, figcaption=False)
        else:
            drawn = f'<div class="table-block" data-table="{escape(block.name)}">{grid_html(block)}</div>'
        drawn += "".join(f'<p class="cap">{escape(note)}</p>' for note in evidence.chart.notes)
    degraded = f'<p class="cap">{escape(evidence.degraded)}</p>' if evidence.degraded else ""
    return f'{drawn}<p class="cap">{_prose(evidence.caption)}</p>{degraded}'


def _claim(claim: DocClaim, *, static: bool) -> str:
    opening, closing = _WEIGHT_TAGS[claim.weight]
    blocks = "".join(
        _block(evidence, f"claim-{claim.place}-{index}", static=static) for index, evidence in enumerate(claim.evidence)
    )
    return (
        f'<section data-part="claim" data-weight="{escape(claim.weight)}" id="claim-{claim.place}">'
        f"<header>{opening}{_prose(claim.text)}{closing}{_caveats(claim.caveats)}</header>{blocks}</section>"
    )


def _caveats(labels: Sequence[str]) -> str:
    """The caveats landed beside a claim or a Method term, by label, as code's line; nothing for none."""
    if not labels:
        return ""
    return f'<p class="cap" data-part="caveats">{escape(CAVEATS_TEMPLATE.format(labels="; ".join(labels)))}</p>'


def _explore(explore: TableBlock) -> str:
    return (
        f'<section data-part="explore"><header><h2>{escape(explore.title)}</h2></header>{grid_html(explore)}</section>'
    )


def _next(steps: Sequence[DocNext]) -> str:
    items = "".join(f"<li>{_prose(step.text)} <em>{_prose(step.settles)}</em></li>" for step in steps)
    return f'<section data-part="next"><header><h2>{escape(NEXT_TITLE)}</h2></header><ul>{items}</ul></section>'


def _arms_key(key: PageArmsKey) -> str:
    return (
        f'<section data-part="arms"><header><span class="eyebrow">{escape(key.eyebrow)}</span>'
        f'<p class="takeaway">{escape(key.takeaway)}</p></header>'
        f'<div class="table-block" data-table="{escape(key.table.name)}">{grid_html(key.table)}</div></section>'
    )


def _omitted(omitted: Sequence[DocOmitted]) -> str:
    terms = "".join(f"<dt>{_prose(item.what)}</dt><dd>{_prose(item.why)}</dd>" for item in omitted)
    return f'<section data-part="omitted"><h3>{escape(OMITTED_TITLE)}</h3><dl class="method">{terms}</dl></section>'


def _method(method: Sequence[DocTerm]) -> str:
    terms = "".join(
        f"<dt>{_prose(item.term)}</dt><dd>{_prose(item.text)}{_caveats(item.caveats)}</dd>" for item in method
    )
    return f'<section data-part="method"><h3>{escape(METHOD_TITLE)}</h3><dl class="method">{terms}</dl></section>'


def _finding(finding: DocFinding) -> str:
    about = escape(finding.about)
    if finding.claim_place is not None:
        about = f'<a href="#claim-{finding.claim_place}">{about}</a>'
    return f"<li><b>{escape(finding.severity)}</b> · {about}: {_prose(finding.finding)}</li>"


def _findings(findings: Sequence[DocFinding]) -> str:
    items = "".join(_finding(finding) for finding in findings)
    return f'<section data-part="findings"><h3>{escape(FINDINGS_TITLE)}</h3><ul>{items}</ul></section>'


def _appendix(page: DocPage) -> list[str]:
    return [
        f'<div class="part" data-part="appendix"><h2>{escape(APPENDIX_TITLE)}</h2></div>',
        _arms_key(page.arms_key),
        "<section>" + "".join(table_html(table) for table in page.full_results) + "</section>",
        *([_omitted(page.omitted)] if page.omitted else []),
        _method(page.method),
        *([_findings(page.findings)] if page.findings else []),
        *([f'<p class="cap" data-part="made">{escape(page.made)}</p>'] if page.made else []),
    ]


def doc_page_html(page: DocPage, *, static: bool = False) -> str:
    """Write a report document's page as one standalone HTML page (this module's account).

    :param page: The page.
    :ptype page: DocPage
    :param static: Write the static page: no ``<script>``, no ``<link>``, every chart as its table of values.
    :ptype static: bool
    :return: The page.
    :rtype: str
    """
    highlighted = page.recommended_rows
    bluf = [
        f"<h1>{_prose(one_line(page.headline))}</h1>",
        f'<p class="lede">{_prose(page.lede)}</p>',
        *(
            [f'<div class="flag" data-part="flag">{_FLAG_MARK}<div>{_prose(page.flag)}</div></div>']
            if page.flag
            else []
        ),
        _confidence(page.confidence),
        '<div class="decision" data-part="decision">' + "".join(_slot(slot) for slot in page.decision) + "</div>",
        f'<div data-table="options"{_SHRINKS}>{grid_html(page.options, highlighted=highlighted)}</div>',
    ]
    parts = [
        *page_head(page.title, static=static),
        "<body>",
        '<div class="page">',
        '<article data-page="doc">',
        _masthead(page.masthead),
        *([f'<p class="callout warn" data-part="degraded">{escape(page.degraded)}</p>'] if page.degraded else []),
        '<div class="bluf">',
        *bluf,
        "</div>",
        *(_claim(claim, static=static) for claim in page.claims),
        *([_explore(page.explore)] if page.explore is not None else []),
        *([_next(page.next)] if page.next else []),
        *_appendix(page),
        "</article>",
        "</div>",
        *([] if static else [f"<script>{KIT_JS}</script>"]),
        "</body>",
        "</html>",
    ]
    return "\n".join(part for part in parts if part) + "\n"
