"""A report document's page as one self-contained HTML page, in the default mode (the kit's one script draws each
chart) or the static mode (nothing runs or loads; every chart is drawn on the server as inline SVG above its table of
values).

The page is the kit's, as the retired planned page laid it out,
with no class, style or script of its own:

- ``.page`` > ``.mast``, the campaign's name its first line; on a page read from evidence of an earlier shape this
  build reads as stored, code's one line saying so, as a ``.callout`` (``data-part="stored-shape"``), first after the masthead; on a draft, code's one line
  saying its editor's review left findings open, as a ``.callout.warn`` (``data-part="draft"``); on a finished page
  with findings open, code's
  one line counting them, as a ``.callout`` (``data-part="open"``) linked to where they are listed; where part of a
  stored document no longer reads, code's note as a ``.callout.warn`` (``data-part="degraded"``);
- ``.bluf``: the ``h1`` headline, the ``.lede``, the amber ``.flag`` where the document has one, the confidence
  (``data-part="confidence"``: amber where caveats stay open or the evidence is weak, a muted ``.callout`` for plain
  high; each caveat its label in bold, then its clause, a checked one as ``li.checked``), the ``.decision`` grid (an
  entry set apart by its word as ``.rec``) and the options table (each recommended row ``tr.hl``);
- each claim a ``section`` (``data-part="claim"``, ``id="claim-<place>"``) headed by its text at its weight's size
  (an ``h2`` for a lead claim, an ``h3`` for support, a ``.takeaway`` paragraph for detail), then each evidence
  block's ``.fig`` (or the table of its values), its caption and code's lines as ``.cap``;
- the explore table and the next steps, each its own ``section``;
- the appendix under its ``.part`` divider (``data-part="appendix"``): the arms key, every table the input holds,
  what was left out and why, Method under its own heading (``dl.method``), the review findings left open
  (``id="findings"``), and the made line; the caveats a claim or a Method term lands, by label, as code's ``.cap``
  line under it;
- the kit's script, last, on the default page only.

**Nothing written into a field can make the page run or load anything**, as on every page this package writes: every
text node is escaped before any markup is made of it (:mod:`~threetears.evals.analysis.report.html_parts`), and a
chart's data reaches the page only as JSON escaped into its ``data-chart`` attribute. **The static page** is the
default page less exactly the kit's script and each chart's ``div.chart``, plus each chart drawn on
the server as inline SVG.
"""

from __future__ import annotations

from collections.abc import Sequence

from threetears.evals.analysis.report.blocks import ChartBlock, TableBlock
from threetears.evals.analysis.report.doc_page.model import (
    DocCheck,
    DocClaim,
    DocConfidence,
    DocEvidence,
    DocFinding,
    DocNext,
    DocOmitted,
    DocPage,
    DocTerm,
    caveats_line,
    draft_line,
    masthead_parts,
    open_findings_line,
)
from threetears.evals.analysis.report.doc_page.words import (
    APPENDIX_TITLE,
    CHECKS_TITLE,
    EVIDENCE_TITLE,
    FINDINGS_TITLE,
    METHOD_TITLE,
    NEXT_TITLE,
    OMITTED_TITLE,
)
from threetears.evals.analysis.report.html_parts import (
    KIT_JS,
    chart_html,
    check_anchor,
    check_header_links,
    escape,
    grid_html,
    inline_html,
    one_line,
    page_head,
    table_html,
)
from threetears.evals.analysis.report.parts.full_results import laid_out
from threetears.evals.analysis.style import check_definition
from threetears.evals.analysis.report.parts.model import PageArmsKey, PageMasthead, PageSlot
from threetears.evals.analysis.report.parts.words import CONFIDENCE_LABEL

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


def _masthead(masthead: PageMasthead, campaign: str) -> str:
    # The campaign's name its own first line, as a code page's title is; then each part as a reader's page prints it
    # (masthead_parts): the kind, the system in bold, then the rest.
    kind, system, *rest = masthead_parts(masthead)
    spans = [
        f'<span class="name">{escape(one_line(campaign))}</span>',
        f'<span class="kind">{escape(kind)}</span>',
        f"<span><b>{escape(system)}</b></span>",
        *(f"<span>{escape(part)}</span>" for part in rest),
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
    return f'<p class="cap" data-part="caveats">{escape(caveats_line(labels))}</p>'


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


def _checks(checks: Sequence[DocCheck]) -> str:
    """The appendix's Checks list: each goal check's name, then the expression it stands for, escaped and set as code,
    whole, under the check's own id, which its headers link to."""
    terms = "".join(
        f'<dt id="{check_anchor(check.name)}">{escape(check.name)}</dt>'
        f"<dd>{check_definition([(check.expression, None)], span=lambda said: f'<code>{escape(said)}</code>')}</dd>"
        for check in checks
    )
    return f'<section data-part="checks"><h3>{escape(CHECKS_TITLE)}</h3><dl class="method">{terms}</dl></section>'


def _finding(finding: DocFinding) -> str:
    about = escape(finding.about)
    if finding.claim_place is not None:
        about = f'<a href="#claim-{finding.claim_place}">{about}</a>'
    return f"<li><b>{escape(finding.severity)}</b> · {about}: {_prose(finding.said)}</li>"


def _findings(findings: Sequence[DocFinding]) -> str:
    """The review findings left open, at the appendix's type size and measure (the kit's ``.method``), as Method and
    Checks beside them read."""
    items = "".join(_finding(finding) for finding in findings)
    return (
        f'<section data-part="findings" id="findings"><h3>{escape(FINDINGS_TITLE)}</h3>'
        f'<ul class="method">{items}</ul></section>'
    )


def _appendix(page: DocPage) -> list[str]:
    return [
        f'<div class="part" data-part="appendix"><h2>{escape(APPENDIX_TITLE)}</h2></div>',
        _arms_key(page.arms_key),
        "<section>" + "".join(table_html(laid_out(table)) for table in page.full_results) + "</section>",
        *([_omitted(page.omitted)] if page.omitted else []),
        _method(page.method),
        *([_checks(page.checks)] if page.checks else []),
        *([_findings(page.findings)] if page.findings else []),
        *([f'<p class="cap" data-part="made">{escape(page.made)}</p>'] if page.made else []),
    ]


def _evidence_record(record: Sequence[str]) -> list[str]:
    """Code's record after the document, closed, under its own heading; nothing where none is passed."""
    if not record:
        return []
    return [
        f'<details class="section" data-part="evidence"><summary><h2>{escape(EVIDENCE_TITLE)}</h2></summary>'
        + "".join(record)
        + "</details>"
    ]


#: The most characters a headline sets in at the hero size: a longer one is a paragraph's worth, which sets in at the
#: scale's 20 px step (``h1.long``) rather than run six lines at a 1104 px frame.
_LONG_HEADLINE = 90


def doc_page_html(page: DocPage, *, static: bool = False, record: Sequence[str] = ()) -> str:
    """Write a report document's page as one standalone HTML page (this module's account).

    :param page: The page.
    :ptype page: DocPage
    :param static: Write the static page: no ``<script>``, no ``<link>``, every chart drawn as inline SVG above its
        table of values.
    :ptype static: bool
    :param record: Markup this package wrote (the campaign's folds, each failed or lost trial's record in them), laid
        out closed after the document, which it leaves as it is; none by default.
    :ptype record: Sequence[str]
    :return: The page.
    :rtype: str
    """
    highlighted = page.recommended_rows
    headline = one_line(page.headline)
    opened = '<h1 class="long">' if len(headline) > _LONG_HEADLINE else "<h1>"
    bluf = [
        f"{opened}{_prose(headline)}</h1>",
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
        _masthead(page.masthead, page.title),
        *(
            [f'<p class="callout" data-part="stored-shape">{escape(page.stored_shape)}</p>']
            if page.stored_shape
            else []
        ),
        *([f'<p class="callout warn" data-part="draft">{escape(draft_line(page))}</p>'] if page.draft else []),
        *(
            [f'<p class="callout" data-part="open"><a href="#findings">{escape(opened)}</a></p>']
            if (opened := open_findings_line(page))
            else []
        ),
        *([f'<p class="callout warn" data-part="degraded">{escape(page.degraded)}</p>'] if page.degraded else []),
        '<div class="bluf">',
        *bluf,
        "</div>",
        *(_claim(claim, static=static) for claim in page.claims),
        *([_explore(page.explore)] if page.explore is not None else []),
        *([_next(page.next)] if page.next else []),
        *_appendix(page),
        *_evidence_record(record),
        "</article>",
        "</div>",
        *([] if static else [f"<script>{KIT_JS}</script>"]),
        "</body>",
        "</html>",
    ]
    # A header naming a goal check links its row in the appendix's Checks list, as a code page's links its glossary.
    return check_header_links("\n".join(part for part in parts if part) + "\n", [check.name for check in page.checks])
