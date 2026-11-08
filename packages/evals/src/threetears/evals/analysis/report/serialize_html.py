"""The report as one self-contained HTML page, in one of two modes: the default page, whose one script is the eval
report kit's own, and the static page, which runs and loads nothing at all.

**Nothing an author or a model wrote can make the page run or load anything.** Every text node is escaped, so no
field can open a ``<script>``, a ``<link>``, an event-handler attribute or a ``javascript:`` URL. The default page carries
exactly one script, written by this module and never from a field: the kit's ``kit.js``, shipped beside this module
as package data, inlined verbatim as the last element of the body. It draws each chart from its data, and a chart's
data reaches the page only as JSON escaped into its ``data-chart`` attribute, every string in it text the kit sets as
text. The page loads nothing: its faces are a local stack (IBM Plex Sans, Sans Condensed and Mono where the reader
has them, then the system's faces, ending in a generic family), so a sandboxed frame that blocks every host sets it as
it would anywhere. Every link is a ``#`` link to a place on the page itself. Drill-down is ``<details>``; moving
between the screens is ``#`` anchors; neither needs the script.

**The static page** (``static=True`` on :func:`report_html` and :func:`campaign_report_html`) is for a host whose
policy lets a page neither run nor load anything (``default-src 'none'``), even were escaping ever to fail. It is the
default page less exactly two things, the kit's script and each chart's ``div.chart`` with its data, and plus each chart drawn on the server as inline SVG
(:func:`~threetears.evals.analysis.report.chart_svg.chart_drawing`) with the drawings' stylesheet. So it carries no
``<script>`` and no ``<link>``; every chart the kit draws as a picture is that picture, drawn with no script, and its table of values stays shown
under it, since the one rule that hides the table needs a ``div.chart`` beside it; a chart not drawn there (interval
rows, a gate matrix, a lattice, example cards) says in one line that it is shown as its table. The default page is
unchanged by the option, byte for byte.

**A chart reads without its picture.** Each chart is the kit's ``.fig``: its title, a ``div.chart`` the kit draws at
its real width, and a table of every value it draws. While the chart is drawn the table is visually hidden, for a
screen reader; where no script ran, the kit refused the data, or the kit has no kind for the chart (interval rows,
which name no interval level), the table is what the figure shows. A table cell carrying an interval draws it as a
small inline SVG glyph beside its number, on one scale for the column, so a page with styles off loses nothing: every
figure a glyph draws is in the row's text.

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

**A host restyles it through seven tokens, and only those.** The stylesheet reads ``--host-font-body``,
``--host-font-mono``, ``--host-color-text``, ``--host-color-muted``, ``--host-color-surface``, ``--host-color-link`` and
``--host-color-accent``, each with the kit's own value as its fallback, and declares none of them, so a host's value
wins in either theme; secondary text, rules and tints mix from them. The series, outcome, status and sequential
colours stay fixed, so a brand colour can never make a fail look like a pass. The page opens with exactly one
``<html lang="en">`` tag, right after the doctype, which a host framing it in a ``srcdoc`` rewrites to stamp
``data-theme`` and an inline ``style`` with its tokens (``docs/adopting-a-host.md``); ``data-theme`` wins over the
reader's colour-scheme setting either way.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from collections.abc import Collection, Mapping, Sequence
from typing import TYPE_CHECKING

from threetears.evals.analysis.bundle.schema import TrialState
from threetears.evals.analysis.report.blocks import (
    FIRST_SCREEN,
    FLAGGED_CAVEATS,
    FOLD_SECTION_TITLES,
    FOLDED_SECTIONS,
    SECTION_TITLES,
    ChartBlock,
    DisclosureBlock,
    ReportBlock,
    TableBlock,
    TrialBlock,
    fold_anchor,
)
from threetears.evals.analysis.report.charts import ChartStatus
from threetears.evals.analysis.report.html_parts import (
    KIT_JS,
    TRIAL_ID,
    chart_html,
    CHECK_ANCHOR_PREFIX,
    check_anchor,
    check_header_links,
    code_html,
    escape,
    glossary_slug,
    one_line,
    page_head,
    pill_html,
    table_html,
    term_link,
)
from threetears.evals.analysis.report.doc_page.html import doc_page_html
from threetears.evals.analysis.report.serialize_md import trial_summary, world_count
from threetears.evals.analysis.style import CANNOT_TELL_APART, CANNOT_TELL_APART_PHRASE, GLOSSARY, check_definition
from threetears.evals.schema.logs import contain, get_logger

if TYPE_CHECKING:
    # The documents import this module to write themselves, so their types are named for the checker alone.
    from threetears.evals.analysis.report.model import CampaignAspectFold, CampaignReport, Report

log = get_logger(__name__)


#: The folded sections whose own table lists every one of their units, so the summary can say how many: the cases.
#: Not the trials — their table lists only the failed and lost, and the section's census line counts the rest.
_COUNTED_SECTIONS: frozenset[str] = frozenset({"cases"})


def report_html(report: Report, *, static: bool = False) -> str:
    """Render a report as one standalone HTML page whose one script is the kit's, or, ``static``, with none.

    :param report: The report.
    :ptype report: Report
    :param static: Write the static page: no ``<script>``, no ``<link>``, and every chart drawn as inline SVG above its
        table of values. False (the default) writes the page with the kit's script, byte for byte as before the option
        existed.
    :ptype static: bool
    :return: The page.
    :rtype: str
    """
    head = [block for block in report.blocks if block.section in FIRST_SCREEN]
    parts = [
        *page_head(report.title, static=static),
        "<body>",
        '<div class="page">',
        # An aspect's page is code's alone; the attribute stays, so the page's bytes do not move.
        '<article data-basis="code_only">',
        "<header>",
        # The title first, then the first screen in order: what warns is under the verdict, never above the title.
        f"<h1>{escape(one_line(report.title))}</h1>",
        *(_block(block, index, static=static) for index, block in enumerate(head)),
        "</header>",
    ]
    titles = report.section_titles
    present = _present_sections(report)
    nav = [*present, *(["glossary"] if report.glossary else [])]
    parts.append("<nav>" + "".join(f'<a href="#{section}">{escape(titles[section])}</a>' for section in nav) + "</nav>")
    parts += _sections(report, present, static=static)
    parts = [_first_uses(_check_headers("\n".join(parts), report), report, _printed_names(report.blocks))]
    parts += _glossary(report)
    parts += ["</article>", "</div>", *_script(static), "</body>", "</html>"]
    return _links_in_sight("\n".join(parts) + "\n")


def campaign_report_html(report: CampaignReport, *, static: bool = False) -> str:
    """Render a campaign's page as one standalone HTML page whose one script is the kit's, or, ``static``, with none.

    The first screen is the page's ``<header>``: the title first, then the rest of the first screen in order (the
    question, the answer, what warns, set apart but never announced over the title). The sections the page reads
    before its folds follow, open (the findings, the next steps, Design & provenance); then one closed ``<section>``
    per aspect, its summary the aspect's label and what code read there, its body the aspect's own page laid out as
    its page lays it out, a heading level down. Every id inside an aspect's fold is prefixed with the aspect's, so a
    link inside it lands inside it (``#a-billing-T4``); a glossary link lands on the page's one glossary, which defines every term any fold uses.

    A report document's page (``doc_page`` set) is the page its stored document draws
    (:func:`~threetears.evals.analysis.report.doc_page.html.doc_page_html`), static where this page is, written whole:
    its own appendix holds every table its snapshot holds. With ``doc_trials`` (``trials="all"``), the aspects' folds
    and the glossary they link follow it, closed, as its evidence record, each failed or lost trial's record in them.

    :param report: The campaign's page.
    :ptype report: CampaignReport
    :param static: Write the static page, as :func:`report_html` does: no script, no link, every chart drawn as inline SVG above its table.
    :ptype static: bool
    :return: The page.
    :rtype: str
    """
    if report.doc_page is not None:
        if not report.doc_trials:
            return doc_page_html(report.doc_page, static=static)
        return _links_in_sight(doc_page_html(report.doc_page, static=static, record=_appendix(report, static)))
    head = [block for block in report.blocks if block.section in FIRST_SCREEN]
    slugs = report.fold_ids
    parts = [
        *page_head(report.title, static=static),
        "<body>",
        '<div class="page">',
        f'<article data-basis="{escape(report.basis)}" data-page="campaign">',
        "<header>",
        # The title first, then the decision: the question, the answer, then what warns, as consequences.
        f"<h1>{escape(one_line(report.title))}</h1>",
        *(_block(block, index, static=static) for index, block in enumerate(head)),
        "</header>",
    ]
    # What the page reads and the writer says sit before the folds; how the page was made (the methods) after them.
    before, after = report.headed_sections
    titles = report.section_titles
    # Each nav link names the section itself, whose heading every browser shows, a closed one's included. An aspect's
    # link names it by its declared short name where it has one (its fold's heading keeps the full label); a link is
    # one line, cut with an ellipsis where it is still too long. A label is a host's text, which no attribute carries,
    # so its whole is the heading of the fold the link lands on, never a title.
    links = [(section, titles[section]) for section in before]
    links += [
        (fold_anchor(slug), fold.report.bundle.short_name or fold.heading)
        for fold, slug in zip(report.aspects, slugs, strict=True)
    ]
    links += [(section, titles[section]) for section in after]
    links += [("glossary", SECTION_TITLES["glossary"])] if report.glossary else []
    parts.append("<nav>" + "".join(f'<a href="#{href}">{escape(text)}</a>' for href, text in links) + "</nav>")
    for section in before:
        parts += _campaign_section(report, section, static=static)
    for fold, slug in zip(report.aspects, slugs, strict=True):
        parts += _aspect_fold(fold, slug, static=static)
    for section in after:
        parts += _campaign_section(report, section, static=static)
    every = [*report.blocks, *(block for fold in report.aspects for block in fold.report.blocks)]
    parts = [_first_uses(_check_headers("\n".join(parts), report), report, _printed_names(every))]
    parts += _glossary(report)
    parts += ["</article>", "</div>", *_script(static), "</body>", "</html>"]
    return _links_in_sight("\n".join(parts) + "\n")


#: A start or end tag in the page's markup, and its element's name.
_TAG = re.compile(r"(<[^>]*>)")
_TAG_NAME = re.compile(r"</?\s*([a-zA-Z][a-zA-Z0-9]*)")
_TITLE = re.compile(r'\btitle="([^"]*)"')

#: Where a glossary term is never linked: inside a link, code a reader copies, a heading, a fold's summary, a table
#: (a header links its own term where it names one, and a cell links only where it says), the navigation, a chart,
#: the head of the page, or the declared question (``_QUOTED``, the author's words).
_QUOTED = "quoted:"

#: Where no term of art is linked but the undecided reading: the verdict line, the page's answer in one bold sentence,
#: which a link on its every term would break up; its undecided reading is the one term of art a reader must have
#: defined to read it. A goal check's name there still links the check's entry, which says what the name stands for.
_VERDICT = "verdict:"
_UNLINKED_IN: frozenset[str] = frozenset(
    {"a", "code", "pre", "summary", "h1", "h2", "h3", "h4", "h5", "h6", "th", "td", "nav", "svg", "script", "style"}
    | {"title", f"{_QUOTED}p"}
)


def _first_uses(markup: str, report: Report | CampaignReport, names: Sequence[str] = ()) -> str:
    """``markup`` with each glossary term the page defines linked where the page first uses it in its prose: the first
    time its words stand whole in a line a reader reads, unless a link to it came first (a column header's, a line's
    own), the verdict line linking no term of art but the undecided reading. Each term is linked once this way; the
    glossary at the foot defines it. A goal check's name is matched whole and in its own case, never inside a longer
    one (``Goal check`` in ``Goal check 2``); a name that is a term of art's words (``Outcome``) is left to the term.
    Nor is a term linked inside a measure's or a column's name (``names``, :func:`_printed_names`), wherever a line
    prints it: ``ran`` in ``Turns that ran over budget`` is the name's word, not the term; a name that is the term's
    own words is the term."""
    terms = list(report.terms)
    of_art = {term.casefold() for term in terms}
    checks = [name for name, _ in report.check_glossary if name.casefold() not in of_art]
    # A link's title names its term: a check's name carries one title for every check, so it is read off the link.
    by_definition = {GLOSSARY[term]: term for term in terms}
    pending = {
        # The undecided reading is linked in its sentence form too: ``cannot tell v2 and v1 apart``.
        term: CANNOT_TELL_APART_PHRASE
        if term == CANNOT_TELL_APART
        else re.compile(rf"(?<![\w-]){re.escape(escape(term))}(?![\w-])", re.IGNORECASE)
        for term in terms
    }
    pending.update({name: re.compile(rf"(?<![\w-]){re.escape(escape(name))}(?!(?:[\w-]| \d))") for name in checks})
    opened: list[str] = []
    out: list[str] = []
    named = _NamedIn(names)
    for token in _TAG.split(markup):
        if not token.startswith("<"):
            if token and pending and not any(name in _UNLINKED_IN for name in opened):
                if any(name.startswith(_VERDICT) for name in opened):
                    held = {
                        term: pattern
                        for term, pattern in pending.items()
                        if term == CANNOT_TELL_APART or term not in GLOSSARY
                    }
                    token = _linked_once(token, held, named)
                    # A term linked here is pending no longer.
                    for term in [
                        term
                        for term in pending
                        if (term == CANNOT_TELL_APART or term not in GLOSSARY) and term not in held
                    ]:
                        pending.pop(term)
                else:
                    token = _linked_once(token, pending, named)
            out.append(token)
            continue
        out.append(token)
        found = _TAG_NAME.match(token)
        if found is None:
            continue
        name = found.group(1).lower()
        if token.startswith("</"):
            while opened and opened.pop().removeprefix(_QUOTED).removeprefix(_VERDICT) != name:
                pass
            continue
        if name == "a" and (title := _TITLE.search(token)) is not None:
            pending.pop(by_definition.get(html.unescape(title.group(1)), ""), None)
        if name not in _VOID_ELEMENTS and not token.endswith("/>"):
            # The question is the author's words, quoted whole: nothing is linked inside it; the verdict line links its
            # undecided reading alone.
            prefix = _QUOTED if 'class="question"' in token else _VERDICT if 'class="verdict"' in token else ""
            opened.append(f"{prefix}{name}")
    return "".join(out)


def _check_headers(markup: str, report: Report | CampaignReport) -> str:
    """``markup`` with each column header that names a goal check linked to its glossary entry
    (:func:`~threetears.evals.analysis.report.html_parts.check_header_links`); a name that is a term of art's words is
    left to the term."""
    of_art = {term.casefold() for term in report.terms}
    return check_header_links(markup, [name for name, _ in report.check_glossary if name.casefold() not in of_art])


def _linked_once(text: str, pending: dict[str, re.Pattern[str]], named: _NamedIn | None = None) -> str:
    """``text`` with the first use of each pending term in it linked, the longest first where two start alike, never
    one inside a name ``named`` holds; each term linked is no longer pending."""
    spans = named.spans(text) if named is not None else []
    found = sorted(
        (
            (match.start(), -len(match.group(0)), match.end(), term)
            for term, pattern in pending.items()
            if (match := _first_free(pattern, text, spans)) is not None
        ),
    )
    said: list[str] = []
    at = 0
    for start, _, end, term in found:
        if start < at:
            continue
        said += [text[at:start], term_link(term, text[start:end])]
        pending.pop(term)
        at = end
    return "".join([*said, text[at:]])


def _first_free(pattern: re.Pattern[str], text: str, spans: Sequence[tuple[int, int]]) -> re.Match[str] | None:
    """The first match of ``pattern`` in ``text`` that is not inside a name's span (``spans``): one a name holds whole
    is the name's own words; one that is the whole name is the term."""
    for match in pattern.finditer(text):
        if not any(
            start <= match.start() and match.end() <= end and (start, end) != match.span() for start, end in spans
        ):
            return match
    return None


#: A column head's unit, which is no part of the name it follows: ``(%)``, ``(pts)``, and the count money is drawn
#: per (``, per 1,000 replies``).
_HEAD_UNIT = re.compile(r"(?: \((?:%|pts)\)|, per [\d,]+ [\w ]+)$")


def _printed_names(blocks: Sequence[ReportBlock]) -> list[str]:
    """Every measure's and column's name a page's tables print: each column's header without the unit it closes on
    (``Pass rate`` of ``Pass rate (%)``, ``Measuring spend, all roles`` of its ``, per 1,000 replies``), so a term that
    is a measure's whole name (``measuring spend, all roles``) is that name, wherever the unit follows it."""
    headers = {column.header for block in blocks if isinstance(block, TableBlock) for column in block.columns}
    return sorted({_HEAD_UNIT.sub("", header) for header in headers} - {""})


class _NamedIn:
    """Where the names a page prints stand in a line of its markup: each the longest that starts there, matched whole
    and in any case (a name in a sentence keeps the sentence's case), so a term inside one is left to the name."""

    def __init__(self, names: Sequence[str]) -> None:
        held = sorted({escape(name) for name in names if name.strip()}, key=len, reverse=True)
        self._pattern = (
            re.compile(rf"(?<![\w-])(?:{'|'.join(re.escape(name) for name in held)})(?![\w-])", re.IGNORECASE)
            if held
            else None
        )

    def spans(self, text: str) -> list[tuple[int, int]]:
        """Each name's span in ``text``, the longest at each place, none overlapping."""
        return [] if self._pattern is None else [match.span() for match in self._pattern.finditer(text)]


def _appendix(report: CampaignReport, static: bool) -> list[str]:
    """A planned page's appendix: each aspect's fold, as every campaign page lays it out, then the glossary its folds
    link."""
    folds = [
        "\n".join(_aspect_fold(fold, slug, static=static))
        for fold, slug in zip(report.aspects, report.fold_ids, strict=True)
    ]
    # The document's own Checks list defines each check it names: the record's glossary defines only the others.
    glossary = _glossary(report, defined=[check.name for check in report.doc_page.checks] if report.doc_page else [])
    return [*folds, *(["\n".join(glossary)] if glossary else [])]


def _script(static: bool) -> list[str]:
    """The page's one script, the kit's, as the last element of its body; nothing on a static page."""
    return [] if static else [f"<script>{KIT_JS}</script>"]


def _campaign_section(report: CampaignReport, section: str, *, static: bool) -> list[str]:
    """One section of a campaign's page after its first screen: what it means open, the methods folded."""
    blocks = [(index, block) for index, block in enumerate(report.blocks) if block.section == section]
    return [
        f'<section id="{section}">',
        *_section(
            section,
            report.section_titles[section],
            _folded(blocks, static=static, heading=report.section_titles[section]),
            count=None,
        ),
        "</section>",
    ]


#: An id or a link to one, as the page writes them: every id a fold holds is prefixed, every link to a place in it.
_ID_ATTRIBUTE = re.compile(r'\b(id|aria-labelledby)="([^"]*)"')
_LOCAL_LINK = re.compile(r'href="#(?!g-)([^"]*)"')


def _aspect_fold(fold: CampaignAspectFold, slug: str, *, static: bool) -> list[str]:
    """One aspect's closed fold: its page's first screen and sections, a heading
    level down, every id in it prefixed ``a-<slug>--``.

    Ids derived from the slug join it with ``--``, which :func:`~threetears.evals.analysis.report.blocks.fold_ids`
    never writes (it folds every run of other characters into one hyphen), so an aspect labelled "billing body" can
    never take the id of the "billing" fold's body, nor any id inside it.

    The page's own title, its quoted question, its broken floors and its verdict line are left out
    (:attr:`~threetears.evals.analysis.report.model.CampaignAspectFold.head`): the campaign's first screen carries the
    first three once and reads each aspect itself. Its glossary is the campaign page's.
    """
    report = fold.report
    head = fold.head
    body = [
        *(_block(block, index, static=static) for index, block in enumerate(head)),
        *_sections(report, _present_sections(report), static=static, renamed=FOLD_SECTION_TITLES),
    ]
    prefix = f"a-{slug}--"
    inner = _prefixed("\n".join(body), prefix)
    return [
        f'<section id="{fold_anchor(slug)}">',
        # The separator binds to the words after it (a no-break space), so a line never ends on it.
        f'<details class="section"><summary><h2>{escape(fold.heading)}</h2> '
        f'<span class="count">·\u00a0{escape(fold.shows)}</span></summary>',
        f'<div class="section-body" id="{fold_anchor(slug)}--body">',
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


def _sections(
    report: Report, present: Sequence[str], *, static: bool, renamed: Mapping[str, str] | None = None
) -> list[str]:
    """Each section after the first screen, in a ``<section>`` of its id."""
    titles = report.section_titles
    parts: list[str] = []
    for section in present:
        blocks = [(index, block) for index, block in enumerate(report.blocks) if block.section == section]
        parts.append(f'<section id="{section}">')
        body = _folded(blocks, static=static, heading=titles[section])
        title = (renamed or {}).get(section, titles[section])
        parts.extend(_section(section, title, body, count=_section_count(section, blocks)))
        parts.append("</section>")
    return parts


def _glossary(report: Report | CampaignReport, *, defined: Collection[str] = ()) -> list[str]:
    """The glossary section, closed: each term the page links, then each goal check's name, its expression escaped and
    set as code, whole, under an id of the checks' own (``g-check-…``) so a declared name that is a term's words never
    repeats a term's id; none of the checks ``defined`` names, which a report document's own Checks list defines;
    nothing when the page defines none."""
    entries = "".join(
        f'<dt id="g-{glossary_slug(term)}">{escape(term)}</dt><dd>{escape(GLOSSARY[term])}</dd>'
        for term in report.terms
    )
    entries += "".join(
        f'<dt id="{check_anchor(name)}">{escape(name)}</dt>'
        f"<dd>{check_definition(said, span=lambda check: f'<code>{escape(check)}</code>', words=escape)}</dd>"
        for name, said in report.check_glossary
        if name not in defined
    )
    if not entries:
        return []
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
    heading = f"<h2>{escape(title)}</h2>"
    if section not in FOLDED_SECTIONS:
        return [heading, *body]
    size = f' <span class="count">·\u00a0{count}</span>' if count is not None else ""
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


def _folded(blocks: Sequence[tuple[int, ReportBlock]], *, static: bool, heading: str | None = None) -> list[str]:
    """A section's blocks, each consecutive run naming one fold inside one closed ``<details>``; a table titled as the
    section's ``heading`` untitled, so no heading prints twice."""
    parts: list[str] = []
    open_fold: str | None = None
    for index, block in blocks:
        if block.fold != open_fold:
            if open_fold is not None:
                parts.append("</details>")
            if block.fold is not None:
                parts.append(f'<details class="fold"><summary>{escape(block.fold)}</summary>')
            open_fold = block.fold
        if isinstance(block, TableBlock) and block.fold is None and not block.details and block.title == heading:
            # A table in a section of its own name is titled once, by the section's heading: drawn as one in a fold of
            # that name is, without a title of its own.
            block = block.model_copy(update={"fold": block.title})
        parts.append(_block(block, index, static=static))
    if open_fold is not None:
        parts.append("</details>")
    return parts


def _block(block: ReportBlock, index: int, *, static: bool) -> str:
    """One block as HTML; ``index`` makes a chart's ids unique on the page, and a static page draws its charts as
    inline SVG."""
    if isinstance(block, TableBlock):
        return table_html(block)
    if isinstance(block, ChartBlock):
        return chart_html(block, f"chart-{index}", static=static)
    if isinstance(block, TrialBlock):
        return _trial(block)
    return _disclosure(block)


def _trial(block: TrialBlock) -> str:
    """One trial's record, open when the block says so: what it asked, what the judge saw and said, what failed,
    and where it is stored."""
    parts: list[str] = []
    if block.cause_words:
        parts.append(f"<dt>Why</dt><dd>{escape(block.cause_words)}</dd>")
    if block.first_input:
        parts.append(f"<dt>What the case asked</dt><dd><pre>{escape(block.first_input)}</pre></dd>")
    if block.world:
        parts.append(
            f"<dt>{escape(world_count(block.world_parts))}</dt><dd><details><summary>"
            f"{escape(block.case_label)}'s world state</summary><pre>{escape(block.world)}</pre></details></dd>"
        )
    if block.excerpt:
        parts.append(f"<dt>What the judge saw (its last part)</dt><dd><pre>{escape(block.excerpt)}</pre></dd>")
    if block.failed_checks:
        checks = "".join(
            f"<li><code>{escape(check.expression)}</code></li>"
            if check.expression in check.name
            else f"<li>{escape(check.name)}: <code>{escape(check.expression)}</code></li>"
            for check in block.failed_checks
        )
        parts.append(f"<dt>Failed checks</dt><dd><ul>{checks}</ul></dd>")
    if block.judge:
        judged = "".join(
            f"<li><strong>{escape(entry.dim)}</strong>: "
            + ("could not tell" if entry.score is None else f"{entry.score} ({escape(entry.scale)})")
            + (f'<br><span class="facts">Asked: {escape(entry.criterion)}</span>' if entry.criterion else "")
            + (f"<br>{escape(entry.reason)}" if entry.reason else "")
            + "</li>"
            for entry in block.judge
        )
        parts.append(f"<dt>What the judge said</dt><dd><ul>{judged}</ul></dd>")
    if block.error_words:
        parts.append(f"<dt>Recorded error</dt><dd><pre>{escape(block.error_words)}</pre></dd>")
    parts.append(f"<dt>Stored result</dt><dd><code>{escape(block.result_id)}</code></dd>")
    opened = " open" if block.open else ""
    return (
        f'<details class="trial" id="{escape(block.trial_id)}" data-state="{escape(block.state)}"{opened}>'
        f"<summary>{_trial_summary(block)}</summary><dl>{''.join(parts)}</dl></details>"
    )


#: Each trial state as the kit's pill shows it: its status (the pill's colour, and the icon beside the word, so a state
#: never rests on a colour alone).
_STATE_PILLS: dict[TrialState, ChartStatus] = {"passed": "good", "failed": "bad", "lost": "neutral"}


def _trial_summary(block: TrialBlock) -> str:
    """A trial's one-line summary, its state the kit's pill: the same words as every serializer's, the icon hidden
    from a screen reader, which reads the word."""
    lead = trial_summary(block).removesuffix(block.state)
    return escape(lead) + pill_html(_STATE_PILLS[block.state], escape(block.state))


#: The sources of what warns, set apart under the title: a missed bar, a broken floor, a guardrail an arm is not
#: shown safe on.
_WARNING_SOURCES = frozenset({"bar", "floor", "guardrails"})

#: The kit's flag icon, beside the words that carry the meaning; hidden from a screen reader, which reads the words.
_FLAG_ICON = '<b aria-hidden="true">!</b>'


def _disclosure(block: DisclosureBlock) -> str:
    """Something code had to say: the first screen's lines as lines, every other one set apart from the author's."""
    if block.source == "verdict":
        tone = f' data-tone="{escape(block.tone)}"' if block.tone is not None else ""
        return f'<p class="verdict"{tone}>{_verdict_markup(block)}</p>'
    text = code_html(block.text)
    if block.term is not None:
        text += " " + term_link(block.term, f"({escape(block.term)})", css="term")
    if block.source == "arms":
        return f'<p class="arms">{text}</p>'
    if block.section in ("header", "answer") and block.source == "basis":
        return f'<p class="question">{text}</p>'
    anchor = f' id="{escape(block.anchor)}"' if block.anchor is not None else ""
    if block.section in ("header", "warnings") and block.source in _WARNING_SOURCES:
        # What warns, under the title: the kit's flag, set apart, never announced over it.
        return f'<p class="flag warning"{anchor}>{_FLAG_ICON}<strong>{text}</strong></p>'
    if block.anchor in FLAGGED_CAVEATS:
        # When the criteria were set, once, beside what warns: the one report-wide caveat set apart, in normal weight.
        return f'<p class="flag report-caveat"{anchor} data-source="{escape(block.source)}">{_FLAG_ICON}<span>{text}</span></p>'
    if block.section == "warnings" or (block.section == "header" and block.anchor is not None):
        # Every other report-wide caveat, once: a plain line under the flag, never a callout of the same weight.
        return f'<p class="line report-caveat"{anchor} data-source="{escape(block.source)}">{text}</p>'
    if block.section in FIRST_SCREEN:
        return f'<p class="line" data-source="{escape(block.source)}">{text}</p>'
    return f'<p class="disclosure" data-source="{escape(block.source)}">{text}</p>'


def _verdict_markup(block: DisclosureBlock) -> str:
    """The verdict line's markup: each piece of its raw text escaped on its own (:meth:`DisclosureBlock.emphasis_pieces`),
    its decisions set in bold, its figures and guardrails regular, so one weight carries the one emphasis. The
    undecided reading is linked where its sentence form stands whole in one piece; where it does not (a phrase cut by
    the bold, or longer than the form allows), the line links the term after its words, as every other line does."""
    linked = block.term != CANNOT_TELL_APART
    out = []
    for piece, bold in block.emphasis_pieces():
        said = code_html(piece)
        # linked only outside a code span, so neither half of the piece cuts one: the first such match, since an
        # earlier one may cross a span while a whole one follows it
        found = (
            None
            if linked
            else next(
                (
                    match
                    for match in CANNOT_TELL_APART_PHRASE.finditer(piece)
                    if "`" not in match.group(0) and not piece.count("`", 0, match.start()) % 2
                ),
                None,
            )
        )
        if found is not None:
            said = (
                code_html(piece[: found.start()])
                + term_link(CANNOT_TELL_APART, escape(found.group(0)))
                + code_html(piece[found.end() :])
            )
            linked = True
        out.append(f"<strong>{said}</strong>" if bold else said)
    if block.term is not None and (block.term != CANNOT_TELL_APART or not linked):
        out.append(" " + term_link(block.term, f"({escape(block.term)})", css="term"))
    return "".join(out)


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
    reader where they were, and nothing on the page opens it. The one exception is a link to a goal check's row
    (``g-check-…``), which lands on the row: what a reader following a check's name wants is its expression. A trial id naming a trial the page does not hold
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
    if chain is not None and link.target.startswith(CHECK_ANCHOR_PREFIX):
        # A goal check's name links its own row, the one place its expression stands: a reader following a header
        # wants that row, and the glossary's heading names no check. The one link that may land inside a closed fold.
        return link.tag, after
    if chain is None:
        text = page[after : link.end] if link.end is not None else ""
        # prose-canary: allow — a trial id (T4), which code numbered, not anything a model wrote
        if link.end is not None and TRIAL_ID.fullmatch(text):
            return text, page.index(">", link.end) + 1
        return link.tag, after
    landing = next((place for place, above in chain if above <= link.shut), link.target)
    if _HREF.findall(link.tag) != [link.target]:
        raise ValueError(f"the link to #{link.target} names it other than once, double-quoted: {link.tag}")
    return _HREF.sub(f'href="#{landing}"', link.tag), after


__all__ = [
    "campaign_report_html",
    "report_html",
]
