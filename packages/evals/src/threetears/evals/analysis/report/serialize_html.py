"""The report as one self-contained HTML page that needs no script.

**Nothing an author or a model wrote can make the page run or load anything.** Every text node is escaped, so no
field can open a ``<script>``, a ``<link>``, an event-handler attribute or a ``javascript:`` URL; the page has no
``<script>`` of its own either. The one thing it loads is the package's own: the kit's fonts (IBM Plex Sans, Sans
Condensed and Mono), two preconnects and one stylesheet link to the Google Fonts hosts, written by this module and
never from a field, the same three links the kit's reference memos carry. Where a host's policy blocks those hosts
the page reads the same in the system faces the kit names after each Plex face. Every other link is a ``#`` link to a
place on the page itself. Drill-down is ``<details>``; moving between the screens is ``#`` anchors. The outcome bar
is inline SVG (:mod:`~threetears.evals.analysis.report.svg`) beside a visually hidden table of the values it draws,
and a table cell carrying an interval draws it as a small glyph beside its number, on one scale for the column, so a
screen reader and a page with styles off lose nothing: every figure a glyph draws is in the row's text.

**The first screen is the page's ``<header>``**: the title (the campaign's name) first, nothing above it; the
declared question quoted whole on its own line where there is one, then the arms line; the arms line alone (line 2)
where none was; the verdict line; what warns (a broken floor's line, set apart, never announced over the title) and
each report-wide caveat once; the scoreboard with its outcome bar, and the one caveat in normal weight that changes
how the verdict is read. Everything a reader needs to decide whether to read on
is there, written by code. Every term of art links to the glossary at the foot of the page.

**Depth on demand, with the least ink.** The result, the floors, what it means and the contrasts are open;
the cases, the trials, what was not tested, the methods and the glossary each sit in a closed ``<details>`` whose
summary is the section heading, so a reader opens what they came for.

**The kit's visual system.** The stylesheet is the eval report kit's ``kit.css`` (shipped beside this module as
package data, byte for byte the kit's documentation copy), then a short stylesheet of this page's own for the
structures the kit has no class for, drawn from the kit's tokens alone. The page sits in the kit's ``.page`` (prose
held to a readable measure, tables and figures breaking out to the full width), each table scrolls inside its own
``.tbl`` frame so the page never scrolls sideways, what warns and each report-wide caveat is a ``.flag``, a trial's
state is a ``.pill`` (an icon and the word, never a colour alone), headings are in the kit's display face, and the
glossary is a ``.method`` list. Light and dark follow the reader's setting, and a ``data-theme`` a host's page sets.

**A host restyles one thing: ``--host-accent``.** It is a CSS token, not a parameter: a host framing the page
overrides it (light and dark values) in its own stylesheet, and the series, status and sequential colours stay
fixed, so a brand colour can never make a fail look like a pass.

The author's words are Markdown; this renders the part of it a memo uses — paragraphs, bullet lists, ``**bold**``
and ``code`` — over text escaped first, so nothing an author wrote can become markup.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from importlib.resources import files
from html.parser import HTMLParser
from collections.abc import Sequence
from typing import TYPE_CHECKING

from threetears.evals.analysis.bundle.schema import TrialState
from threetears.evals.analysis.report.blocks import (
    FIRST_SCREEN,
    FOLDED_SECTIONS,
    SECTION_TITLES,
    ChartBlock,
    DisclosureBlock,
    ReportBlock,
    TableBlock,
    TableCell,
    TextBlock,
    TrialBlock,
    fold_anchor,
)
from threetears.evals.analysis.report.serialize_md import RESOLUTION_WORDS, SUMMARY_LEADS, trial_summary
from threetears.evals.analysis.report.svg import SVG_STYLE, chart_svg, interval_glyph_svg
from threetears.evals.analysis.quantities import PERCENT, is_money, number_in, with_unit
from threetears.evals.analysis.style import GLOSSARY
from threetears.evals.schema.logs import contain, get_logger

if TYPE_CHECKING:
    # The documents import this module to write themselves, so their types are named for the checker alone.
    from threetears.evals.analysis.report.model import CampaignAspectFold, CampaignReport, Report

log = get_logger(__name__)

#: The kit's stylesheet, read once from the package: tokens for light and dark, type, the page and prose columns,
#: tables, pills and the flag (``kit/kit.css``, the kit's documentation copy byte for byte).
_KIT_CSS = files("threetears.evals.analysis.report").joinpath("kit", "kit.css").read_text(encoding="utf-8")

#: The kit's fonts, as the reference memos load them: a preconnect to each Google Fonts host, then the one stylesheet
#: naming the three Plex faces. Written here and only here: no field of a report reaches the page's head.
_FONT_LINKS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500'
    '&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,600;1,400&display=swap">'
)

#: The page's own structures the kit has no class for, from the kit's tokens alone (no colour, face or measure of its
#: own): the first screen, the sticky section links, the folds and their summaries, a table's title and its numeric,
#: prose and glyph columns, a chart's frame, the verdict line's tone, a trial's record and the code-written lines. A
#: column header keeps the case it was declared in, since a header can be an arm's name or a check's expression. Prose
#: is held to a measure in characters near the kit's 44rem column, so a smaller line keeps a readable length too.
_PAGE_STYLE = """
html { scroll-padding-top: 3.5rem; }
p, li, .disclosure, .caveat { max-width: 78ch; }
ul { margin: 0; padding-left: 1.25rem; }
h4, h5 { font-family: var(--f-display); font-weight: 600; margin: 0; }
h4 { font-size: var(--t-base); }
h5 { font-size: var(--t-sm); }
article > header { display: grid; gap: .75rem; padding-block: 1.75rem 1.5rem; }
article > header > h1 { max-width: var(--col); }
.question { font-size: var(--t-md); line-height: 1.45; color: var(--ink-2); }
.arms { font-size: var(--t-sm); color: var(--muted); }
.verdict { font-size: var(--t-md); line-height: 1.45; padding: .15rem 0 .15rem .9rem; border-left: 3px solid var(--base); }
.verdict[data-tone="decided"] { border-left-color: var(--accent); }
.verdict[data-tone="open"] { border-left-color: var(--warn); }
.verdict[data-tone="failed"] { border-left-color: var(--bad); }
.verdict strong { font-weight: 600; }
.line { font-size: var(--t-sm); color: var(--ink-2); }
.line[data-source="caveat"] { font-size: var(--t-base); color: var(--ink); }
.flag.report-caveat { color: var(--ink-2); }
.flag strong { font-weight: 600; color: var(--ink); }
.facts, .order, .count, .trial-links { font-size: var(--t-sm); color: var(--muted); }
.axis { font-size: var(--t-xs); color: var(--muted); }
a.term { font-size: var(--t-xs); color: var(--muted); text-decoration: underline dotted; }
nav { position: sticky; top: 0; z-index: 1; display: flex; flex-wrap: wrap; gap: .25rem 1.25rem; padding: .6rem 0;
  font-size: var(--t-sm); background: var(--bg); border-bottom: 1px solid var(--line); }
nav a { color: var(--ink-2); text-decoration: none; }
nav a:hover { color: var(--accent); }
section > *, .section-body > * { min-width: 0; }
.section-body { display: grid; gap: 1rem; padding-top: 1rem; }
.section-body section { padding-top: 1rem; }
summary { cursor: pointer; list-style: none; }
summary::-webkit-details-marker { display: none; }
summary::before { content: "\\203A"; display: inline-block; width: 1em; color: var(--muted); transition: transform .1s; }
details[open] > summary::before { transform: rotate(90deg); }
details.section > summary { padding-bottom: .4rem; border-bottom: 1px solid var(--line); }
details.section > summary > h2, details.section > summary > h3, details.section > summary > h4 { display: inline; }
details.fold > summary, details[data-table] > summary { font-size: var(--t-sm); color: var(--ink-2); }
details.fold > summary + *, details[data-table] > summary + * { margin-top: .5rem; }
details.fold > * + *:not(summary) { margin-top: 1rem; }
details.expr { display: inline-block; }
details.expr > summary::before { content: none; }
details.expr[open] > summary { display: none; }
.table-block { display: grid; gap: .5rem; }
th { text-transform: none; letter-spacing: normal; }
th a { color: inherit; text-decoration: underline dotted; }
th code { font-size: 1em; }
td { vertical-align: top; }
th.num, td.num { text-align: right; font-variant-numeric: tabular-nums; }
td.key { font-weight: 600; }
[data-table="scoreboard"] table { font-size: var(--t-base); }
td.wrap { max-width: 32rem; }
td.glyph { white-space: nowrap; font-variant-numeric: tabular-nums; }
td.glyph .figure { display: inline-block; min-width: 6ch; text-align: right; }
td.glyph .no-glyph { display: inline-block; width: 7.5em; margin-right: 6px; }
td:target, tr:target > td { background: var(--accent-wash); }
figure { margin: 0; }
.fig figcaption { font-family: var(--f-display); font-size: var(--t-base); font-weight: 600; margin-bottom: .5rem; }
.fig .tbl { border: 0; border-radius: 0; background: none; }
.finding { font-family: var(--f-display); font-size: var(--t-md); margin-top: .75rem; }
.finding strong { font-weight: 600; }
[data-role] > p + p, [data-role] > p + ul, [data-role] > ul + p { margin-top: .6rem; }
p[data-role] > strong:first-child, .caveat > strong:first-child { font-size: var(--t-sm); color: var(--muted);
  letter-spacing: .02em; }
.disclosure, .caveat { font-size: var(--t-sm); color: var(--ink-2); padding-left: .75rem;
  border-left: 1px solid var(--line); }
pre { white-space: pre-wrap; overflow-wrap: anywhere; margin: .25rem 0; padding: .5rem .6rem; border-radius: 4px;
  font: var(--t-xs)/1.45 var(--f-mono); color: var(--ink-2); background: var(--sunk); max-height: 24rem;
  overflow-y: auto; }
details.trial { padding: .5rem 0 .5rem .75rem; border-top: 1px solid var(--line); border-left: 2px solid transparent; }
details.trial:last-of-type { border-bottom: 1px solid var(--line); }
details.trial[data-state="failed"] { border-left-color: var(--bad); }
details.trial[data-state="lost"] { border-left-color: var(--base); }
details.trial > summary { font-size: var(--t-sm); }
details.trial dl { margin: .5rem 0 0; font-size: var(--t-sm); }
details.trial dt { font-weight: 600; margin-top: .5rem; }
details.trial dd { margin: .15rem 0 0; }
@media (max-width: 640px) { nav { position: static; } .verdict { font-size: var(--t-base); } }
@media print {
  nav { display: none; }
  .tbl { overflow: visible; }
  th { position: static; }
  svg.viz { min-width: 0; }
  details.trial, .tbl, figure { break-inside: avoid; }
}
"""


#: The folded sections whose own table lists every one of their units, so the summary can say how many: the cases.
#: Not the trials — their table lists only the failed and lost, and the section's census line counts the rest.
_COUNTED_SECTIONS: frozenset[str] = frozenset({"cases"})

#: Inline Markdown a memo uses, applied to text already escaped — so its delimiters are the only markup it can make.
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_CODE = re.compile(r"`([^`]+)`")

#: A trial id in a table cell, linked to its drill-down.
_TRIAL_ID = re.compile(r"\bT\d+\b")

#: The columns whose trial ids link to their records: the cases table's, the trials table's and a finding's facts.
_TRIAL_COLUMNS: frozenset[tuple[str, str]] = frozenset(
    {("cases", "trials"), ("trials", "trial"), ("finding_facts", "trials")}
)

#: A case as the cases table numbers it, whose number its row's anchor is made of; a case named by its author anchors
#: by its name (``case-double-charge``).
_CASE_NUMBER = re.compile(r"case (\d+)")

#: A cell of words at least this long is prose, laid out to wrap at a measure rather than widen its column.
_PROSE_CELL = 40


def report_html(report: Report) -> str:
    """Render a report as one standalone HTML page that needs no script.

    :param report: The report.
    :ptype report: Report
    :return: The page.
    :rtype: str
    """
    head = [block for block in report.blocks if block.section in FIRST_SCREEN]
    parts = [
        *_head(report.title),
        "<body>",
        '<div class="page">',
        f'<article data-basis="{_escape(report.basis)}">',
        "<header>",
        # The title first, then the first screen in order: what warns is under the verdict, never above the title.
        f"<h1>{_escape(_one_line(report.title))}</h1>",
        *(_block(block, index) for index, block in enumerate(head)),
        "</header>",
    ]
    titles = report.section_titles
    present = _present_sections(report)
    nav = [*present, *(["glossary"] if report.terms else [])]
    parts.append(
        "<nav>" + "".join(f'<a href="#{section}">{_escape(titles[section])}</a>' for section in nav) + "</nav>"
    )
    parts += _sections(report, present)
    parts += _glossary(report.terms)
    parts += ["</article>", "</div>", "</body>", "</html>"]
    return _links_in_sight("\n".join(parts) + "\n")


def campaign_report_html(report: CampaignReport) -> str:
    """Render a campaign's page as one standalone HTML page that needs no script.

    The first screen is the page's ``<header>``: the title first, then the rest of the first screen in order (the
    question, the answer, what warns, set apart but never announced over the title). The sections the page reads
    before its folds follow, open (the findings, the next steps, Design & provenance); then one closed ``<section>``
    per aspect, its summary the aspect's label and what code read there, its
    body the writer's summary of the aspect and the aspect's own page laid out as its page lays it out, a heading
    level down. Every id inside an aspect's fold is prefixed with the aspect's, so a link inside it lands inside it
    (``#a-billing-T4``); a glossary link lands on the page's one glossary, which defines every term any fold uses.

    :param report: The campaign's page.
    :ptype report: CampaignReport
    :return: The page.
    :rtype: str
    """
    head = [block for block in report.blocks if block.section in FIRST_SCREEN]
    slugs = report.fold_ids
    parts = [
        *_head(report.title),
        "<body>",
        '<div class="page">',
        f'<article data-basis="{_escape(report.basis)}" data-page="campaign">',
        "<header>",
        # The title first, then the decision: the question, the answer, then what warns, as consequences.
        f"<h1>{_escape(_one_line(report.title))}</h1>",
        *(_block(block, index) for index, block in enumerate(head)),
        "</header>",
    ]
    # What the page reads and the writer says sit before the folds; how the page was made (the methods) after them.
    before, after = report.headed_sections
    titles = report.section_titles
    # Each nav link names the section itself, whose heading every browser shows, a closed one's included.
    links = [(section, titles[section]) for section in before]
    links += [(fold_anchor(slug), fold.aspect) for fold, slug in zip(report.aspects, slugs, strict=True)]
    links += [(section, titles[section]) for section in after]
    links += [("glossary", SECTION_TITLES["glossary"])] if report.terms else []
    parts.append("<nav>" + "".join(f'<a href="#{href}">{_escape(text)}</a>' for href, text in links) + "</nav>")
    for section in before:
        parts += _campaign_section(report, section)
    for fold, slug in zip(report.aspects, slugs, strict=True):
        parts += _aspect_fold(fold, slug)
    for section in after:
        parts += _campaign_section(report, section)
    parts += _glossary(report.terms)
    parts += ["</article>", "</div>", "</body>", "</html>"]
    return _links_in_sight("\n".join(parts) + "\n")


def _head(title: str) -> list[str]:
    """The page's ``<head>``: its title, the kit's font links and the stylesheet — the kit's, then the page's own and
    the charts'. Nothing in it comes from a field but the title, escaped."""
    return [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{_escape(_one_line(title))}</title>",
        _FONT_LINKS,
        f"<style>{_KIT_CSS}{_PAGE_STYLE}{SVG_STYLE}</style>",
        "</head>",
    ]


def _campaign_section(report: CampaignReport, section: str) -> list[str]:
    """One section of a campaign's page after its first screen: what it means open, the methods folded."""
    blocks = [(index, block) for index, block in enumerate(report.blocks) if block.section == section]
    return [
        f'<section id="{section}">',
        *_section(section, report.section_titles[section], _folded(blocks), count=None),
        "</section>",
    ]


#: An id or a link to one, as the page writes them: every id a fold holds is prefixed, every link to a place in it.
_ID_ATTRIBUTE = re.compile(r'\b(id|aria-labelledby)="([^"]*)"')
_LOCAL_LINK = re.compile(r'href="#(?!g-)([^"]*)"')


def _aspect_fold(fold: CampaignAspectFold, slug: str) -> list[str]:
    """One aspect's closed fold: the writer's summary of it, then its page's first screen and sections, a heading level
    down, every id in it prefixed ``a-<slug>-``.

    The page's own title, its quoted question, its broken floors and its verdict line are left out
    (:attr:`~threetears.evals.analysis.report.model.CampaignAspectFold.head`): the campaign's first screen carries the
    first three once and reads each aspect itself. Its glossary is the campaign page's.
    """
    report = fold.report
    head = fold.head
    body = [
        *(_block(block, index) for index, block in enumerate(fold.summary)),
        *(_block(block, index) for index, block in enumerate(head)),
        *_sections(report, _present_sections(report)),
    ]
    prefix = f"a-{slug}-"
    inner = _prefixed("\n".join(body), prefix)
    return [
        f'<section id="{fold_anchor(slug)}">',
        f'<details class="section"><summary><h2>{_escape(fold.aspect)}</h2> '
        f'<span class="count">· {_escape(fold.shows)}</span></summary>',
        f'<div class="section-body" id="{fold_anchor(slug)}-body">',
        inner,
        "</div></details>",
        "</section>",
    ]


def _prefixed(markup: str, prefix: str) -> str:
    """``markup`` with every id it holds and every link to a place on the page prefixed, its headings a level down.

    Safe on the page's own markup alone: every text node is escaped, so a quote, an ``id="`` or an ``<h2>`` in it is
    one the serializer wrote. A glossary link (``#g-...``) is left alone: the glossary is the page's, not the fold's.
    """

    def ids(match: re.Match[str]) -> str:
        return f'{match.group(1)}="' + " ".join(prefix + part for part in match.group(2).split()) + '"'

    markup = _ID_ATTRIBUTE.sub(ids, markup)
    markup = _LOCAL_LINK.sub(lambda match: f'href="#{prefix}{match.group(1)}"', markup)
    for level in (4, 3, 2):
        markup = markup.replace(f"<h{level}>", f"<h{level + 1}>").replace(f"</h{level}>", f"</h{level + 1}>")
    return markup


def _present_sections(report: Report) -> list[str]:
    """The sections after the first screen that hold a block, in reading order; never the glossary."""
    sections = [section for section in report.section_titles if section not in FIRST_SCREEN and section != "glossary"]
    return [section for section in sections if any(block.section == section for block in report.blocks)]


def _sections(report: Report, present: Sequence[str]) -> list[str]:
    """Each section after the first screen, in a ``<section>`` of its id."""
    titles = report.section_titles
    parts: list[str] = []
    for section in present:
        blocks = [(index, block) for index, block in enumerate(report.blocks) if block.section == section]
        parts.append(f'<section id="{section}">')
        parts.extend(_section(section, titles[section], _folded(blocks), count=_section_count(section, blocks)))
        parts.append("</section>")
    return parts


def _glossary(terms: Sequence[str]) -> list[str]:
    """The glossary section defining ``terms``, closed; nothing when the page links none."""
    if not terms:
        return []
    entries = "".join(
        f'<dt id="g-{_slug(term)}">{_escape(term)}</dt><dd>{_escape(GLOSSARY[term])}</dd>' for term in terms
    )
    return [
        '<section id="glossary">',
        *_section("glossary", SECTION_TITLES["glossary"], [f'<dl class="method glossary">{entries}</dl>'], count=None),
        "</section>",
    ]


def _section(section: str, title: str, body: Sequence[str], *, count: int | None) -> list[str]:
    """A section's heading and body: open, or closed in a ``<details>`` whose summary is the heading and its size.

    ``title`` is the heading as this page heads the section (:attr:`Report.section_titles`), which can differ from
    the section's fixed id: a one-arm page heads ``contrasts`` as ``Readings``.

    A closed section's body is one element with an id of its own (``cases-body``), which the stylesheet keeps clear of
    the sticky nav; the nav links the ``<section>`` around it, whose heading every browser shows.
    """
    heading = f"<h2>{_escape(title)}</h2>"
    if section not in FOLDED_SECTIONS:
        return [heading, *body]
    size = f' <span class="count">· {count}</span>' if count is not None else ""
    return [
        f'<details class="section"><summary>{heading}{size}</summary>',
        f'<div class="section-body" id="{section}-body">',
        *body,
        "</div></details>",
    ]


def _section_count(section: str, blocks: Sequence[tuple[int, ReportBlock]]) -> int | None:
    """How many units a section lists in full (its cases), for its closed summary; None for every other section."""
    if section not in _COUNTED_SECTIONS:
        return None
    table = next((block for _, block in blocks if isinstance(block, TableBlock) and block.name == section), None)
    return None if table is None else table.total_rows


def _folded(blocks: Sequence[tuple[int, ReportBlock]]) -> list[str]:
    """A section's blocks, each consecutive run naming one fold inside one closed ``<details>``."""
    parts: list[str] = []
    open_fold: str | None = None
    for index, block in blocks:
        if block.fold != open_fold:
            if open_fold is not None:
                parts.append("</details>")
            if block.fold is not None:
                parts.append(f'<details class="fold"><summary>{_escape(block.fold)}</summary>')
            open_fold = block.fold
        parts.append(_block(block, index))
    if open_fold is not None:
        parts.append("</details>")
    return parts


def _block(block: ReportBlock, index: int) -> str:
    """One block as HTML; ``index`` makes a chart's ids unique on the page."""
    if isinstance(block, TextBlock):
        return _text(block)
    if isinstance(block, TableBlock):
        return _table(block)
    if isinstance(block, ChartBlock):
        return _chart(block, f"chart-{index}")
    if isinstance(block, TrialBlock):
        return _trial(block)
    return _disclosure(block)


def _text(block: TextBlock) -> str:
    """An author's words, laid out by what they were written as."""
    facts = {fact.name: fact.value for fact in block.facts}
    body = _inline(_one_line(block.body))
    if block.role == "finding_title":
        anchor = f' id="{_escape(block.anchor)}"' if block.anchor else ""
        return f'<p class="finding"{anchor}><strong>{body}</strong></p>'
    if block.role == "answer":
        lead = RESOLUTION_WORDS.get(facts.get("Resolution", ""), "")
        return '<p data-role="answer">' + (f"<strong>{_escape(lead)}.</strong> " if lead else "") + f"{body}</p>"
    if block.role == "recommendation":
        disposition = facts.get("Disposition", "")
        lead = f"<strong>{_escape(disposition[:1].upper() + disposition[1:])}:</strong> " if disposition else ""
        sure = (
            f' <span class="facts">({_escape(facts["Confidence"])} confidence)</span>' if "Confidence" in facts else ""
        )
        return f'<p data-role="recommendation">{lead}{body}{sure}</p>'
    if block.role == "override":
        over = (
            f"The memo recommends {_escape(facts['Arm'])} over {_escape(facts.get('Over', 'it'))}"
            if "Arm" in facts
            else "The memo's why"
        )
        return f'<p class="override" data-role="override"><strong>{over}:</strong> {body}</p>'
    if block.role in SUMMARY_LEADS:
        return f'<p data-role="{block.role}"><strong>{SUMMARY_LEADS[block.role]}</strong> {body}</p>'
    if block.role == "reason":
        return f'<p data-role="reason">{body}</p>'
    if block.role == "next_step":
        return f'<p data-role="next_step"><strong>Next:</strong> {body}</p>'
    if block.role == "caveat":
        return f'<div class="caveat"><strong>Caveat</strong> ({_escape(facts.get("Kind", "unkinded"))}): {body}</div>'
    if block.role == "carried_forward":
        return f"<p><em>Carried forward:</em> {body}</p>"
    return f'<div data-role="{_escape(block.role)}">{_markdown(block.body)}</div>'


def _table(block: TableBlock) -> str:
    """A table block: its title and rows, or a line saying it has none; folded when it is drill-down."""
    if not block.rows:
        grid = '<p class="order">(no rows)</p>'
    else:
        grid = _grid(block)
        if block.total_rows > len(block.rows):
            grid += f'<p class="order">{len(block.rows)} of {block.total_rows} rows shown.</p>'
    title = _escape(block.title)
    if block.title_ref is not None:
        title = f'<a href="#{_escape(block.title_ref)}">{title}</a>'
    if block.details:
        # A closed table carries its own id where a finding's figure lands on it: on the fold, which every browser
        # shows, never on a cell inside it, which a browser that leaves the fold shut would not.
        anchor = f' id="{_escape(block.anchor)}"' if block.anchor is not None else ""
        return f'<details{anchor} data-table="{_escape(block.name)}"><summary>{title}</summary>{grid}</details>'
    if block.fold is not None:
        return f'<div class="table-block" data-table="{_escape(block.name)}">{grid}</div>'
    return f'<div class="table-block" data-table="{_escape(block.name)}"><h3>{title}</h3>{grid}</div>'


def _header(text: str, term: str | None) -> str:
    """A column header, linked to its glossary entry where it is a term of art; a code span (a check) as code."""
    shown = _cell_code(text)
    return _term_link(term, shown) if term else shown


def _term_link(term: str, label: str, *, css: str = "") -> str:
    """A link to a glossary term, its sentence as the link's title so a hover reads it without leaving the place."""
    style = f' class="{css}"' if css else ""
    return f'<a{style} href="#g-{_slug(term)}" title="{_escape(GLOSSARY[term])}">{label}</a>'


def _code(text: str) -> str:
    """Text escaped, its code spans (a check's expression, a call to copy) set as code."""
    return _CODE.sub(r"<code>\1</code>", _escape(text))


#: The longest expression a table cell or header shows whole; a longer one shows this many characters and a visible
#: ellipsis, the whole expression one click away.
_EXPRESSION_CHARS = 60


def _cell_code(text: str) -> str:
    """A cell's or a header's text escaped, its code spans set as code, each longer than :data:`_EXPRESSION_CHARS`
    shown by its start and a visible ellipsis, the whole of it in a fold that opens in place. Never shortened silently:
    the whole expression is in the page."""

    def span(match: re.Match[str]) -> str:
        whole = match.group(1)
        if len(html.unescape(whole)) <= _EXPRESSION_CHARS:
            return f"<code>{whole}</code>"
        start = _escape(html.unescape(whole)[:_EXPRESSION_CHARS].rstrip())
        return f'<details class="expr"><summary><code>{start}…</code></summary><code>{whole}</code></details>'

    return _CODE.sub(span, _escape(text))


def _grid(block: TableBlock) -> str:
    """A table of served cells' text, scrolling inside its own ``.tbl`` frame rather than widening the page.

    A column of numbers is classed ``num`` on every cell, header included, so the stylesheet right-aligns it; a cell
    of prose is classed ``wrap`` (the kit's) so it wraps at a measure instead of widening its column. A column whose cells carry
    an interval is classed ``glyph``: each such cell draws its interval beside its number
    (:func:`~threetears.evals.analysis.report.svg.interval_glyph_svg`), every cell of the column in one unit on one
    scale, and the scale's ends are written once under the table (:func:`_scale_line`).
    """
    scales = _glyph_scales(block)
    classes = {
        column.key: ' class="glyph"'
        if any(key == column.key for key, _ in scales)
        else _column_class(block, column.key)
        for column in block.columns
    }
    head = "".join(
        f'<th scope="col"{classes[column.key]}>{_header(column.header, column.term)}</th>' for column in block.columns
    )
    body = []
    for row in block.rows:
        cells = []
        for column in block.columns:
            cell = row[column.key]
            text = _cell_code(cell.text)
            if cell.ref is not None:
                # A finding's figure, linked to a cell printing the same figure, or to the closed fold holding it. A
                # link never holds a fold that opens: a long expression's fold sits beside the link, not inside it.
                text = (
                    f'<a href="#{cell.ref}">{text}</a>'
                    if "<details" not in text
                    else f'{text} <a class="cell-link" href="#{cell.ref}">where it is</a>'
                )
            if (block.name, column.key) in _TRIAL_COLUMNS:
                text = _TRIAL_ID.sub(lambda match: f'<a href="#{match.group(0)}">{match.group(0)}</a>', text)
            if block.name == "cases" and column.key == "trials":
                # Navigation beside the marks, not information: the stylesheet mutes it.
                css = ' class="trial-links"'
            elif classes[column.key] == ' class="glyph"':
                css = classes[column.key]
                text = _glyph(cell, column.key, scales) + f'<span class="figure">{text}</span>'
            else:
                css = classes[column.key] or (' class="wrap"' if len(cell.text) >= _PROSE_CELL else "")
            anchor = f' id="{cell.anchor}"' if cell.anchor is not None else ""
            cells.append(f"<td{anchor}{css}>{text}</td>")
        anchor = ""
        if block.name == "cases" and "case" in row:
            # prose-canary: allow — the cases table's "case N" cell, which code numbered, not anything a model wrote
            number = _CASE_NUMBER.fullmatch(row["case"].text)
            anchor = f' id="case-{number.group(1) if number else _slug(row["case"].text)}"'
        # A passed trial has no record of its own, so the link every table makes of its id lands on its row here.
        # prose-canary: allow — a trial id (T4), which code numbered, not anything a model wrote
        elif block.name == "passed_trials" and "trial" in row and _TRIAL_ID.fullmatch(row["trial"].text):
            anchor = f' id="{row["trial"].text}"'
        body.append(f"<tr{anchor}>" + "".join(cells) + "</tr>")
    grid = f'<div class="tbl"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'
    return grid + _scale_line(block, scales)


def _glyph_scales(block: TableBlock) -> dict[tuple[str, str], tuple[float, float]]:
    """The scale each column's glyphs are drawn on, per unit: one for every cell of the column in that unit, or, in a
    table drawn on one scale (``TableBlock.shared_scale``: a matrix), whose columns are one quantity read for each arm, one for every cell of the table
    in that unit, so a glyph compares across columns as well as down them.

    The scale holds every interval end, every point and every margin band of the group, and zero where any cell is a
    difference, padded a little at either end so no mark sits on the edge.

    :return: ``{(column key, unit): (low, high)}``, for the columns with a cell carrying an interval.
    :rtype: dict[tuple[str, str], tuple[float, float]]
    """
    ends: dict[tuple[str, str], list[float]] = {}
    for row in block.rows:
        for key, cell in row.items():
            if cell.low is None or cell.high is None or not cell.is_number:
                continue
            group = ends.setdefault((key, cell.unit), [])
            group += [cell.low, cell.high, float(cell.value)]  # type: ignore[arg-type]
            if cell.margin is not None:
                group += [-cell.margin, cell.margin]
            if cell.zero_line:
                group.append(0.0)
    if block.shared_scale:
        by_unit: dict[str, list[float]] = {}
        for (_, unit), figures in ends.items():
            by_unit.setdefault(unit, []).extend(figures)
        ends = {(key, unit): by_unit[unit] for key, unit in ends}
    scales = {}
    for group_key, figures in ends.items():
        low, high = min(figures), max(figures)
        pad = (high - low) * 0.05 or abs(high) * 0.05 or 1.0
        scales[group_key] = (low - pad, high + pad)
    return scales


def _glyph(cell: TableCell, key: str, scales: dict[tuple[str, str], tuple[float, float]]) -> str:
    """A cell's interval glyph on its column's scale, or a blank of the same width where the cell has none, so the
    column's figures stay aligned."""
    if cell.low is None or cell.high is None or not cell.is_number:
        return '<span class="no-glyph"></span>'
    return interval_glyph_svg(
        float(cell.value),  # type: ignore[arg-type]
        cell.low,
        cell.high,
        unit=cell.unit,
        scale=scales[(key, cell.unit)],
        margin=cell.margin,
    )


def _scale_line(block: TableBlock, scales: dict[tuple[str, str], tuple[float, float]]) -> str:
    """One quiet line under a table with glyphs: each glyph column's scale, its ends written once per unit."""
    if not scales:
        return ""
    headers = {column.key: column.header for column in block.columns}
    spans: dict[str, list[str]] = {}
    if block.shared_scale:
        # One scale per unit for every column: written once, not once per column.
        scales = {("", unit): scale for (_, unit), scale in scales.items()}
        headers[""] = "Each column"
    for (key, unit), (low, high) in scales.items():
        if unit == PERCENT or is_money(unit):
            ends = f"{with_unit(low, unit)} to {with_unit(high, unit)}"
        else:
            ends = f"{number_in(low, unit)} to {number_in(high, unit)}" + (f" {unit}" if unit else "")
        spans.setdefault(key, []).append(ends)
    said = [f"{_code(headers[key])} drawn from {' and from '.join(ends)}" for key, ends in spans.items()]
    return f'<p class="axis">{" · ".join(said)}</p>'


def _column_class(block: TableBlock, key: str) -> str:
    """The class attribute a column's cells share: ``num`` for a column of numbers, with ``key`` on the scoreboard's
    passed column, the one a reader weighs first; empty for a column of words.

    A column is numeric when it holds a number and nothing but numbers and absences (a dash in a column of numbers
    sits with the numbers, not with the words).
    """
    values = [row[key].value for row in block.rows]
    numbers = [value for value in values if isinstance(value, int | float) and not isinstance(value, bool)]
    if not numbers or len(numbers) + values.count(None) != len(values):
        return ""
    emphasis = " key" if block.name == "scoreboard" and key == "passed" else ""
    return f' class="num{emphasis}"'


def _chart(block: ChartBlock, chart_id: str) -> str:
    """A chart as the kit's ``.fig``: its title, the SVG, and a visually hidden table of every value it draws.

    The SVG keeps a floor width so its words stay legible on a phone; below it the chart scrolls inside its own frame,
    as a wide table does, and the page never scrolls sideways."""
    headers, rows = block.values_table()
    head = "".join(f'<th scope="col">{_escape(header)}</th>' for header in headers)
    body = "".join("<tr>" + "".join(f"<td>{_escape(cell)}</td>" for cell in row) + "</tr>" for row in rows)
    return (
        f'<figure class="fig" data-chart="{_escape(block.chart)}">'
        f"<figcaption>{_escape(block.title)}</figcaption>"
        f'<div class="chart tbl">{chart_svg(block, chart_id=chart_id)}</div>'
        f'<table class="visually-hidden"><caption>{_escape(block.title)}</caption><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></figure>"
    )


def _trial(block: TrialBlock) -> str:
    """One trial's record, open when the block says so: what it asked, what the judge saw and said, what failed,
    and where it is stored."""
    parts: list[str] = []
    if block.cause_words:
        parts.append(f"<dt>Why</dt><dd>{_escape(block.cause_words)}</dd>")
    if block.first_input:
        parts.append(f"<dt>What the case asked</dt><dd><pre>{_escape(block.first_input)}</pre></dd>")
    if block.excerpt:
        parts.append(f"<dt>What the judge saw (its last part)</dt><dd><pre>{_escape(block.excerpt)}</pre></dd>")
    if block.failed_checks:
        checks = "".join(
            f"<li><code>{_escape(check.expression)}</code></li>"
            if check.expression in check.name
            else f"<li>{_escape(check.name)}: <code>{_escape(check.expression)}</code></li>"
            for check in block.failed_checks
        )
        parts.append(f"<dt>Failed checks</dt><dd><ul>{checks}</ul></dd>")
    if block.judge:
        judged = "".join(
            f"<li><strong>{_escape(entry.dim)}</strong>: "
            + ("could not tell" if entry.score is None else f"{entry.score} ({_escape(entry.scale)})")
            + (f'<br><span class="facts">Asked: {_escape(entry.criterion)}</span>' if entry.criterion else "")
            + (f"<br>{_escape(entry.reason)}" if entry.reason else "")
            + "</li>"
            for entry in block.judge
        )
        parts.append(f"<dt>What the judge said</dt><dd><ul>{judged}</ul></dd>")
    if block.error_words:
        parts.append(f"<dt>Recorded error</dt><dd><pre>{_escape(block.error_words)}</pre></dd>")
    parts.append(f"<dt>Stored result</dt><dd><code>{_escape(block.result_id)}</code> (read it with result_get)</dd>")
    opened = " open" if block.open else ""
    return (
        f'<details class="trial" id="{_escape(block.trial_id)}" data-state="{_escape(block.state)}"{opened}>'
        f"<summary>{_trial_summary(block)}</summary><dl>{''.join(parts)}</dl></details>"
    )


#: Each trial state as the kit's pill shows it: its kind (the pill's colour) and the icon beside the word, so a state
#: never rests on a colour alone.
_STATE_PILLS: dict[TrialState, tuple[str, str]] = {
    "passed": ("good", "✓"),
    "failed": ("bad", "✕"),
    "lost": ("neutral", "–"),
}


def _trial_summary(block: TrialBlock) -> str:
    """A trial's one-line summary, its state the kit's pill: the same words as every serializer's, the icon hidden
    from a screen reader, which reads the word."""
    kind, icon = _STATE_PILLS[block.state]
    lead = trial_summary(block).removesuffix(block.state)
    return (
        f'{_escape(lead)}<span class="pill {kind}"><span aria-hidden="true">{icon}</span>{_escape(block.state)}</span>'
    )


#: The sources of what warns, set apart under the title: a missed bar, a broken floor, a guardrail an arm is not
#: shown safe on.
_WARNING_SOURCES = frozenset({"bar", "floor", "guardrails"})

#: The kit's flag icon, beside the words that carry the meaning; hidden from a screen reader, which reads the words.
_FLAG_ICON = '<b aria-hidden="true">!</b>'


def _disclosure(block: DisclosureBlock) -> str:
    """Something code had to say: the first screen's lines as lines, every other one set apart from the author's."""
    text = _code(block.text)
    if block.term is not None:
        text += " " + _term_link(block.term, f"({_escape(block.term)})", css="term")
    if block.source == "arms":
        return f'<p class="arms">{text}</p>'
    if block.section in ("header", "answer") and block.source == "basis":
        return f'<p class="question">{text}</p>'
    anchor = f' id="{_escape(block.anchor)}"' if block.anchor is not None else ""
    if block.section in ("header", "warnings") and block.source in _WARNING_SOURCES:
        # What warns, under the title: the kit's flag, set apart, never announced over it.
        return f'<p class="flag warning"{anchor}>{_FLAG_ICON}<strong>{text}</strong></p>'
    if block.section == "warnings" or (block.section == "header" and block.anchor is not None):
        # A report-wide caveat, once, beside what warns: a flag too, in normal weight.
        return f'<p class="flag report-caveat"{anchor} data-source="{_escape(block.source)}">{_FLAG_ICON}<span>{text}</span></p>'
    if block.source == "verdict":
        tone = f' data-tone="{_escape(block.tone)}"' if block.tone is not None else ""
        return f'<p class="verdict"{tone}><strong>{text}</strong></p>'
    if block.section in FIRST_SCREEN:
        return f'<p class="line" data-source="{_escape(block.source)}">{text}</p>'
    return f'<p class="disclosure" data-source="{_escape(block.source)}">{text}</p>'


def _slug(term: str) -> str:
    """A glossary term as an anchor id."""
    return re.sub(r"[^a-z0-9]+", "-", term.casefold()).strip("-")


def _markdown(text: str) -> str:
    """The Markdown a memo uses — paragraphs, bullet lists, bold, code — over text escaped first.

    :param text: An author's words.
    :ptype text: str
    :return: The HTML.
    :rtype: str
    """
    out: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text.strip()):
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if not lines:
            continue
        if all(line[:2] in ("- ", "* ") for line in lines):
            out.append("<ul>" + "".join(f"<li>{_inline(line[2:])}</li>" for line in lines) + "</ul>")
        else:
            out.append("<p>" + " ".join(_inline(line) for line in lines) + "</p>")
    return "".join(out)


def _inline(text: str) -> str:
    """Escape ``text``, then turn its bold and code spans into markup."""
    return _CODE.sub(r"<code>\1</code>", _BOLD.sub(r"<strong>\1</strong>", _escape(text)))


#: Elements HTML never closes, which hold nothing.
_VOID_ELEMENTS: frozenset[str] = frozenset({"meta", "br", "hr", "img", "input", "link", "wbr"})

#: A link's target in the start tag the parser read as that link: the one ``href`` attribute, double-quoted.
_HREF = re.compile(r'(?<![\w-])href="#([^"]*)"')


@dataclass
class _Link:
    """One ``<a href="#…">`` as the parser read it: where its start tag and its end tag sit in the page, what it
    names, and the closed folds holding it."""

    start: int
    tag: str
    target: str
    shut: frozenset[int]
    end: int | None = None


class _Outline(HTMLParser):
    """The page's own markup read once: each id with the ids around it, each with the closed folds holding it, and
    each link with where it sits and the closed folds holding it.

    A closed fold is a ``<details>`` without ``open``. A place inside one is shown only once the reader opens it, which
    a browser does on its own for a link landing there in some browsers and not in others, so where a link may land
    depends on the folds the reader has already opened to see the link.
    """

    def __init__(self, page: str) -> None:
        super().__init__(convert_charrefs=True)
        self._lines = [0, *(index + 1 for index, char in enumerate(page) if char == "\n")]
        self._open: list[tuple[str, str | None, int | None, frozenset[int]]] = []
        self._folds = 0
        self._link: _Link | None = None
        self.places: dict[str, list[tuple[str, frozenset[int]]]] = {}
        self.links: list[_Link] = []

    def _offset(self) -> int:
        line, column = self.getpos()
        return self._lines[line - 1] + column

    def _shut(self) -> frozenset[int]:
        return frozenset(fold for _, _, fold, _ in self._open if fold is not None)

    def _read(self, tag: str, attrs: list[tuple[str, str | None]]) -> tuple[str | None, int | None]:
        found = dict(attrs)
        shut = self._shut()
        href = found.get("href") or ""
        if tag == "a" and href.startswith("#"):
            self._link = _Link(self._offset(), self.get_starttag_text() or "", href[1:], shut)
            self.links.append(self._link)
        if (name := found.get("id")) is not None:
            around = [(held, above) for _, held, _, above in reversed(self._open) if held is not None]
            self.places[name] = [(name, shut), *around]
        fold = None
        if tag == "details" and "open" not in found:
            self._folds += 1
            fold = self._folds
        return found.get("id"), fold

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        shut = self._shut()
        name, fold = self._read(tag, attrs)
        if tag not in _VOID_ELEMENTS:
            self._open.append((tag, name, fold, shut))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._read(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._link is not None:
            self._link.end = self._offset()
            self._link = None
        if any(open_tag == tag for open_tag, _, _, _ in self._open):
            while self._open.pop()[0] != tag:
                pass


def _links_in_sight(page: str) -> str:
    """``page`` with every link drawn to the nearest place a reader sees: the place it names where no closed fold the
    reader has not opened holds it, else the nearest place around it outside every such fold — a closed section's own
    ``<section>``, a closed table's own ``<details>``, an aspect's fold's ``<section>`` — whose heading every browser
    shows. No link may land inside a closed ``<details>``: a browser that does not open it on its own would leave the
    reader where they were, and the page runs no script to open it. A trial id naming a trial the page does not hold
    is not linked.

    The links are the parser's own, rewritten where it found them, so nothing else has to agree with it on what a link
    is. A link this cannot redraw keeps the target it was written with, and is recorded; the page always renders.

    :param page: The page, as the serializer laid it out.
    :ptype page: str
    :return: The page, every link landing in sight.
    :rtype: str
    """
    try:
        outline = _Outline(page)
        outline.feed(page)
        outline.close()
    except Exception as exc:
        contain(exc, log=log, what="the page's outline could not be read, so its links keep the targets they name")
        return page
    drawn: list[str] = []
    cursor = 0
    for link in outline.links:
        drawn.append(page[cursor : link.start])
        try:
            replacement, cursor = _redrawn(page, link, outline)
        except ValueError as exc:
            contain(
                exc,
                log=log,
                what="a link could not be redrawn, so it keeps the target it names",
                extra={"target": link.target},
            )
            replacement, cursor = link.tag, link.start + len(link.tag)
        drawn.append(replacement)
    drawn.append(page[cursor:])
    return "".join(drawn)


def _redrawn(page: str, link: _Link, outline: _Outline) -> tuple[str, int]:
    """One link redrawn to land in sight, and where in the page what it replaces ends: its start tag with its target
    moved, or, for a trial id the page does not hold, the id alone in place of the whole link.

    :raises ValueError: The page does not hold the start tag where the parser read it, or the tag names its target
        other than once, double-quoted.
    """
    after = link.start + len(link.tag)
    if page[link.start : after] != link.tag:
        raise ValueError(f"the link to #{link.target} is not where the outline read it")
    chain = outline.places.get(link.target)
    if chain is None:
        text = page[after : link.end] if link.end is not None else ""
        # prose-canary: allow — a trial id (T4), which code numbered, not anything a model wrote
        if link.end is not None and _TRIAL_ID.fullmatch(text):
            return text, page.index(">", link.end) + 1
        return link.tag, after
    landing = next((place for place, above in chain if above <= link.shut), link.target)
    if _HREF.findall(link.tag) != [link.target]:
        raise ValueError(f"the link to #{link.target} names it other than once, double-quoted: {link.tag}")
    return _HREF.sub(f'href="#{landing}"', link.tag), after


def _escape(text: str) -> str:
    """Escape text for an element or a quoted attribute."""
    return html.escape(text, quote=True)


def _one_line(text: str) -> str:
    """Text a heading carries, on one line."""
    return " ".join(text.split("\n")).strip()


__all__ = [
    "campaign_report_html",
    "report_html",
]
