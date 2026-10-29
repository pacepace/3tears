"""A planned page as one self-contained HTML page, in the default mode (the kit's one script draws each chart) or the
static mode (nothing runs or loads; every chart is drawn on the server as inline SVG above its table of values).

The page is the kit's (``docs/eval-report-kit/references/``), part by part in GENERATOR §2 order:

- ``.page`` > ``.mast`` (kind · system · counts · run date · synthetic);
- ``.bluf``: the ``h1`` headline, the ``.lede``, the amber ``.flag`` (``data-part="flag"``) where the plan has one,
  the decision's confidence (``data-part="confidence"``: amber where caveats stay open or the evidence is weak,
  a muted ``.callout`` for plain high; code's checked lines as ``li.checked``), the ``.decision`` grid (its
  recommendation or lean ``.rec``), the answers (``data-part="answers"``), the options table (its set-apart row
  ``tr.hl``) and the ``.grid2`` of summary charts;
- the arms key's ``section`` (eyebrow and takeaway in code's words, then its table);
- each evidence ``section``: a ``header`` of ``.eyebrow``, ``h2`` and ``.takeaway``, then its ``.fig``, its caption
  and code's lines as ``.cap``;
- ``.part`` dividers for the explore part and Full results where the page holds them; ``dl.method``; the made line;
- the appendix the caller passes (the per-aspect folds), in one closed ``<details>`` (``data-part="appendix"``);
- the kit's script, last, on the default page only.

**Nothing written into a field can make the page run or load anything.** Every text node is escaped before any
markup is made of it (:mod:`~threetears.evals.analysis.report.html_parts`, the writers every page shares), a chart's
data reaches the page only as JSON escaped into its ``data-chart`` attribute, and the appendix is markup this package
wrote. **The static page** is the default page less exactly the kit's script, the font links and each chart's
``div.chart``, plus each chart drawn on the server as inline SVG: it carries no ``<script>`` and no ``<link>``, and
every chart is its picture with its table shown under it.
"""

from __future__ import annotations

from collections.abc import Sequence

from threetears.evals.analysis.report.blocks import ChartBlock, TableBlock
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
from threetears.evals.analysis.report.plan_page.model import (
    PageAnswer,
    PageArmsKey,
    PageChart,
    PageConfidence,
    PageExplore,
    PageFlag,
    PageMasthead,
    PageOptions,
    PageSection,
    PageSlot,
    PageTerm,
    PlannedPage,
)
from threetears.evals.analysis.report.plan_page.words import (
    APPENDIX_TITLE,
    CONFIDENCE_LABEL,
    FULL_RESULTS_TITLE,
    RUN_TEMPLATE,
    SYNTHETIC,
)

__all__ = ["planned_page_html"]

#: The confidence forms set in the amber box: caveats stay open, or the evidence is weak.
_AMBER_FORMS = frozenset({"high_if_immaterial", "low"})

#: A wrapper the first screen's grid may narrow below its content's width, so a wide table scrolls in its own frame
#: rather than widening the page (the kit's ``.tbl`` and ``.fig`` shrink so; a wrapper of them must too).
_SHRINKS = ' style="min-width:0"'

#: The flag's mark, as the kit's reference pages draw it, hidden from a screen reader, which reads the words.
_FLAG_MARK = '<b aria-hidden="true">!</b>'


def _prose(text: str) -> str:
    """Planner prose as a page line: escaped, its bold and code made markup, each of its lines broken where it
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


def _flag(flag: PageFlag) -> str:
    return (
        f'<div class="flag" data-part="flag">{_FLAG_MARK}'
        f"<div><b>{_prose(flag.title)}</b> {_prose(flag.text)}</div></div>"
    )


def _confidence(confidence: PageConfidence) -> str:
    lines = [f"<li>{_prose(clause.text)}</li>" for clause in confidence.clauses]
    lines += [f'<li class="checked">{escape(line)}</li>' for line in confidence.checked]
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
        f'<span class="v">{_prose(slot.text)}{f" {escape(slot.note)}" if slot.note else ""}</span></div>'
    )


def _answers(answers: Sequence[PageAnswer]) -> str:
    said = "".join(
        f'<p class="takeaway"><strong>{escape(answer.warning)}</strong> {_prose(answer.text)}</p>' for answer in answers
    )
    return f'<div data-part="answers">{said}</div>'


def _options(options: PageOptions) -> str:
    highlighted = [options.highlighted] if options.highlighted is not None else []
    return f'<div data-table="options"{_SHRINKS}>{grid_html(options.table, highlighted=highlighted)}</div>'


def _chart(chart: PageChart, chart_id: str, *, static: bool) -> str:
    """A chart as the page draws it, then code's lines under it; a table as its own block."""
    block = chart.chart
    if isinstance(block, ChartBlock):
        drawn = chart_html(block, chart_id, static=static, figcaption=bool(chart.title))
    else:
        drawn = table_html(block)
    return drawn + "".join(f'<p class="cap">{escape(note)}</p>' for note in chart.notes)


def _arms_key(key: PageArmsKey) -> str:
    return (
        f'<section data-part="arms"><header><span class="eyebrow">{escape(key.eyebrow)}</span>'
        f'<p class="takeaway">{escape(key.takeaway)}</p></header>'
        f'<div class="table-block" data-table="{escape(key.table.name)}">{grid_html(key.table)}</div></section>'
    )


def _section(section: PageSection, chart_id: str, *, static: bool) -> str:
    takeaway = f'<p class="takeaway">{_prose(section.takeaway)}</p>' if section.takeaway else ""
    caption = f'<p class="cap">{_prose(section.caption)}</p>' if section.caption else ""
    return (
        f'<section data-section="{escape(section.id)}"><header>'
        f'<span class="eyebrow">{_prose(section.eyebrow)}</span><h2>{_prose(section.headline)}</h2>{takeaway}</header>'
        f"{_chart(section.chart, chart_id, static=static)}{caption}</section>"
    )


def _part(title: str, part: str) -> str:
    return f'<div class="part" data-part="{escape(part)}"><h2>{_prose(title)}</h2></div>'


def _explore(explore: PageExplore) -> str:
    return (
        _part(explore.title, "explore")
        + f'<section>{table_html(explore.flips)}<p class="takeaway">{escape(explore.holds)}</p></section>'
    )


def _method(method: Sequence[PageTerm]) -> str:
    if not method:
        return ""
    terms = "".join(f"<dt>{_prose(item.term)}</dt><dd>{_prose(item.text)}</dd>" for item in method)
    return f'<dl class="method">{terms}</dl>'


def _full_results(tables: Sequence[TableBlock], method: Sequence[PageTerm]) -> str:
    """Full results under its divider, Method closing it; Method alone where the page holds no full-results table."""
    if not tables:
        return f'<section data-part="method">{_method(method)}</section>' if method else ""
    return (
        _part(FULL_RESULTS_TITLE, "full_results")
        + "<section>"
        + "".join(table_html(table) for table in tables)
        + _method(method)
        + "</section>"
    )


def _appendix(appendix: Sequence[str]) -> str:
    if not appendix:
        return ""
    return (
        f'<details class="section" data-part="appendix"><summary><h2>{escape(APPENDIX_TITLE)}</h2></summary>'
        + "".join(appendix)
        + "</details>"
    )


def planned_page_html(page: PlannedPage, *, appendix: Sequence[str] = (), static: bool = False) -> str:
    """Write a planned page as one standalone HTML page (this module's account).

    :param page: The page.
    :ptype page: PlannedPage
    :param appendix: Markup this package wrote (the campaign's per-aspect folds), laid out closed after Method.
    :ptype appendix: Sequence[str]
    :param static: Write the static page: no ``<script>``, no ``<link>``, every chart drawn as inline SVG above its
        table of values.
    :ptype static: bool
    :return: The page.
    :rtype: str
    """
    bluf = [
        f"<h1>{_prose(one_line(page.headline))}</h1>",
        f'<p class="lede">{_prose(page.lede)}</p>',
        *([_flag(page.flag)] if page.flag is not None else []),
        _confidence(page.confidence),
        '<div class="decision" data-part="decision">' + "".join(_slot(slot) for slot in page.decision) + "</div>",
        *([_answers(page.answers)] if page.answers else []),
        _options(page.options),
    ]
    if page.summary:
        charts = "".join(
            f"<div{_SHRINKS}>{_chart(chart, f'summary-{index}', static=static)}</div>"
            for index, chart in enumerate(page.summary)
        )
        bluf.append(f'<div class="grid2" data-part="summary">{charts}</div>')
    parts = [
        *page_head(page.title, static=static),
        "<body>",
        '<div class="page">',
        '<article data-page="planned">',
        _masthead(page.masthead),
        '<div class="bluf">',
        *bluf,
        "</div>",
        *([_arms_key(page.arms_key)] if page.arms_key is not None else []),
        *(_section(section, f"section-{index}", static=static) for index, section in enumerate(page.sections)),
        *([_explore(page.explore)] if page.explore is not None else []),
        _full_results(page.full_results, page.method),
        *([f'<p class="cap" data-part="made">{escape(page.made)}</p>'] if page.made else []),
        _appendix(appendix),
        "</article>",
        "</div>",
        *([] if static else [f"<script>{KIT_JS}</script>"]),
        "</body>",
        "</html>",
    ]
    return "\n".join(part for part in parts if part) + "\n"
