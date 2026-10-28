"""The report as one self-contained HTML page, in one of two modes: the default page, whose one script is the eval
report kit's own, and the static page, which runs and loads nothing at all.

**Nothing an author or a model wrote can make the page run or load anything.** Every text node is escaped, so no
field can open a ``<script>``, a ``<link>``, an event-handler attribute or a ``javascript:`` URL. The default page carries
exactly one script, written by this module and never from a field: the kit's ``kit.js``, shipped beside this module
as package data, inlined verbatim as the last element of the body. It draws each chart from its data, and a chart's
data reaches the page only as JSON escaped into its ``data-chart`` attribute, every string in it text the kit sets as
text. The one thing the page loads is the package's own too: the kit's fonts (IBM Plex Sans, Sans Condensed and
Mono), two preconnects and one stylesheet link to the Google Fonts hosts, the same three links the kit's reference
memos carry. Where a host's policy blocks those hosts the page reads the same in the system faces the kit names after
each Plex face. Every other link is a ``#`` link to a place on the page itself. Drill-down is ``<details>``; moving
between the screens is ``#`` anchors; neither needs the script.

**The static page** (``static=True`` on :func:`report_html` and :func:`campaign_report_html`) is for a host whose
policy lets a page neither run nor load anything (``default-src 'none'``), even were escaping ever to fail. It is the
default page less exactly three things: the kit's script, the three font links, and each chart's ``div.chart`` with
its data. So it carries no ``<script>`` and no ``<link>``; its text is set in the system faces the kit's stylesheet
names after each Plex face; and every chart is its table of values, shown, since the one rule that hides the table
needs a drawn chart beside it. The default page is unchanged by the option, byte for byte.

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

**A host restyles one thing: ``--host-accent``.** It is a CSS token, not a parameter: a host framing the page
overrides it (light and dark values) in its own stylesheet, and the series, status and sequential colours stay
fixed, so a brand colour can never make a fail look like a pass.

The author's words are Markdown; this renders the part of it a memo uses — paragraphs, bullet lists, ``**bold**``
and ``code`` — over text escaped first, so nothing an author wrote can become markup.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
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
    TextBlock,
    TrialBlock,
    fold_anchor,
)
from threetears.evals.analysis.report.charts import ChartStatus
from threetears.evals.analysis.report.html_parts import (
    KIT_JS,
    TRIAL_ID,
    chart_html,
    code_html,
    escape,
    glossary_slug,
    inline_html,
    markdown_html,
    one_line,
    page_head,
    pill_html,
    table_html,
    term_link,
)
from threetears.evals.analysis.report.doc_page.html import doc_page_html
from threetears.evals.analysis.report.plan_page.html import planned_page_html
from threetears.evals.analysis.report.serialize_md import RESOLUTION_WORDS, SUMMARY_LEADS, trial_summary
from threetears.evals.analysis.style import GLOSSARY
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
    :param static: Write the static page: no ``<script>``, no ``<link>``, the text in the system faces the kit's
        stylesheet names, and every chart as its table of values. False (the default) writes the page with the kit's
        script and font links, byte for byte as before the option existed.
    :ptype static: bool
    :return: The page.
    :rtype: str
    """
    head = [block for block in report.blocks if block.section in FIRST_SCREEN]
    parts = [
        *page_head(report.title, static=static),
        "<body>",
        '<div class="page">',
        f'<article data-basis="{escape(report.basis)}">',
        "<header>",
        # The title first, then the first screen in order: what warns is under the verdict, never above the title.
        f"<h1>{escape(one_line(report.title))}</h1>",
        *(_block(block, index, static=static) for index, block in enumerate(head)),
        "</header>",
    ]
    titles = report.section_titles
    present = _present_sections(report)
    nav = [*present, *(["glossary"] if report.terms else [])]
    parts.append("<nav>" + "".join(f'<a href="#{section}">{escape(titles[section])}</a>' for section in nav) + "</nav>")
    parts += _sections(report, present, static=static)
    parts += _glossary(report.terms)
    parts += ["</article>", "</div>", *_script(static), "</body>", "</html>"]
    return _links_in_sight("\n".join(parts) + "\n")


def campaign_report_html(report: CampaignReport, *, static: bool = False) -> str:
    """Render a campaign's page as one standalone HTML page whose one script is the kit's, or, ``static``, with none.

    The first screen is the page's ``<header>``: the title first, then the rest of the first screen in order (the
    question, the answer, what warns, set apart but never announced over the title). The sections the page reads
    before its folds follow, open (the findings, the next steps, Design & provenance); then one closed ``<section>``
    per aspect, its summary the aspect's label and what code read there, its
    body the writer's summary of the aspect and the aspect's own page laid out as its page lays it out, a heading
    level down. Every id inside an aspect's fold is prefixed with the aspect's, so a link inside it lands inside it
    (``#a-billing-T4``); a glossary link lands on the page's one glossary, which defines every term any fold uses.

    A planned analysis's page (``plan_page`` set) is the page its report plan draws
    (:func:`~threetears.evals.analysis.report.plan_page.html.planned_page_html`), static where this page is, with the
    aspects' folds and the glossary they link as its closed appendix after Method, so a trial's id and its stored
    result are reached as on every other campaign page.

    A report document's page (``doc_page`` set) is the page its stored document draws
    (:func:`~threetears.evals.analysis.report.doc_page.html.doc_page_html`), static where this page is, written whole:
    its own appendix holds every table its snapshot holds. With ``doc_trials`` (``trials="all"``), the aspects' folds
    and the glossary they link follow it, closed, as its evidence record, each failed or lost trial's record in them.

    :param report: The campaign's page.
    :ptype report: CampaignReport
    :param static: Write the static page, as :func:`report_html` does: no script, no link, every chart its table.
    :ptype static: bool
    :return: The page.
    :rtype: str
    """
    if report.doc_page is not None:
        if not report.doc_trials:
            return doc_page_html(report.doc_page, static=static)
        return _links_in_sight(doc_page_html(report.doc_page, static=static, record=_appendix(report, static)))
    if report.plan_page is not None:
        return _links_in_sight(planned_page_html(report.plan_page, appendix=_appendix(report, static), static=static))
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
    # Each nav link names the section itself, whose heading every browser shows, a closed one's included.
    links = [(section, titles[section]) for section in before]
    links += [(fold_anchor(slug), fold.aspect) for fold, slug in zip(report.aspects, slugs, strict=True)]
    links += [(section, titles[section]) for section in after]
    links += [("glossary", SECTION_TITLES["glossary"])] if report.terms else []
    parts.append("<nav>" + "".join(f'<a href="#{href}">{escape(text)}</a>' for href, text in links) + "</nav>")
    for section in before:
        parts += _campaign_section(report, section, static=static)
    for fold, slug in zip(report.aspects, slugs, strict=True):
        parts += _aspect_fold(fold, slug, static=static)
    for section in after:
        parts += _campaign_section(report, section, static=static)
    parts += _glossary(report.terms)
    parts += ["</article>", "</div>", *_script(static), "</body>", "</html>"]
    return _links_in_sight("\n".join(parts) + "\n")


def _appendix(report: CampaignReport, static: bool) -> list[str]:
    """A planned page's appendix: each aspect's fold, as every campaign page lays it out, then the glossary its folds
    link."""
    folds = [
        "\n".join(_aspect_fold(fold, slug, static=static))
        for fold, slug in zip(report.aspects, report.fold_ids, strict=True)
    ]
    glossary = _glossary(report.terms)
    return [*folds, *(["\n".join(glossary)] if glossary else [])]


def _script(static: bool) -> list[str]:
    """The page's one script, the kit's, as the last element of its body; nothing on a static page."""
    return [] if static else [f"<script>{KIT_JS}</script>"]


def _campaign_section(report: CampaignReport, section: str, *, static: bool) -> list[str]:
    """One section of a campaign's page after its first screen: what it means open, the methods folded."""
    blocks = [(index, block) for index, block in enumerate(report.blocks) if block.section == section]
    return [
        f'<section id="{section}">',
        *_section(section, report.section_titles[section], _folded(blocks, static=static), count=None),
        "</section>",
    ]


#: An id or a link to one, as the page writes them: every id a fold holds is prefixed, every link to a place in it.
_ID_ATTRIBUTE = re.compile(r'\b(id|aria-labelledby)="([^"]*)"')
_LOCAL_LINK = re.compile(r'href="#(?!g-)([^"]*)"')


def _aspect_fold(fold: CampaignAspectFold, slug: str, *, static: bool) -> list[str]:
    """One aspect's closed fold: the writer's summary of it, then its page's first screen and sections, a heading level
    down, every id in it prefixed ``a-<slug>--``.

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
        *(_block(block, index, static=static) for index, block in enumerate(fold.summary)),
        *(_block(block, index, static=static) for index, block in enumerate(head)),
        *_sections(report, _present_sections(report), static=static),
    ]
    prefix = f"a-{slug}--"
    inner = _prefixed("\n".join(body), prefix)
    return [
        f'<section id="{fold_anchor(slug)}">',
        f'<details class="section"><summary><h2>{escape(fold.aspect)}</h2> '
        f'<span class="count">· {escape(fold.shows)}</span></summary>',
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


def _sections(report: Report, present: Sequence[str], *, static: bool) -> list[str]:
    """Each section after the first screen, in a ``<section>`` of its id."""
    titles = report.section_titles
    parts: list[str] = []
    for section in present:
        blocks = [(index, block) for index, block in enumerate(report.blocks) if block.section == section]
        parts.append(f'<section id="{section}">')
        body = _folded(blocks, static=static)
        parts.extend(_section(section, titles[section], body, count=_section_count(section, blocks)))
        parts.append("</section>")
    return parts


def _glossary(terms: Sequence[str]) -> list[str]:
    """The glossary section defining ``terms``, closed; nothing when the page links none."""
    if not terms:
        return []
    entries = "".join(
        f'<dt id="g-{glossary_slug(term)}">{escape(term)}</dt><dd>{escape(GLOSSARY[term])}</dd>' for term in terms
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
    heading = f"<h2>{escape(title)}</h2>"
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


def _folded(blocks: Sequence[tuple[int, ReportBlock]], *, static: bool) -> list[str]:
    """A section's blocks, each consecutive run naming one fold inside one closed ``<details>``."""
    parts: list[str] = []
    open_fold: str | None = None
    for index, block in blocks:
        if block.fold != open_fold:
            if open_fold is not None:
                parts.append("</details>")
            if block.fold is not None:
                parts.append(f'<details class="fold"><summary>{escape(block.fold)}</summary>')
            open_fold = block.fold
        parts.append(_block(block, index, static=static))
    if open_fold is not None:
        parts.append("</details>")
    return parts


def _block(block: ReportBlock, index: int, *, static: bool) -> str:
    """One block as HTML; ``index`` makes a chart's ids unique on the page, and a static page's charts are their
    tables."""
    if isinstance(block, TextBlock):
        return _text(block)
    if isinstance(block, TableBlock):
        return table_html(block)
    if isinstance(block, ChartBlock):
        return chart_html(block, f"chart-{index}", static=static)
    if isinstance(block, TrialBlock):
        return _trial(block)
    return _disclosure(block)


def _text(block: TextBlock) -> str:
    """An author's words, laid out by what they were written as."""
    facts = {fact.name: fact.value for fact in block.facts}
    body = inline_html(one_line(block.body))
    if block.role == "finding_title":
        anchor = f' id="{escape(block.anchor)}"' if block.anchor else ""
        return f'<p class="finding"{anchor}><strong>{body}</strong></p>'
    if block.role == "answer":
        lead = RESOLUTION_WORDS.get(facts.get("Resolution", ""), "")
        return '<p data-role="answer">' + (f"<strong>{escape(lead)}.</strong> " if lead else "") + f"{body}</p>"
    if block.role == "recommendation":
        disposition = facts.get("Disposition", "")
        lead = f"<strong>{escape(disposition[:1].upper() + disposition[1:])}:</strong> " if disposition else ""
        sure = (
            f' <span class="facts">({escape(facts["Confidence"])} confidence)</span>' if "Confidence" in facts else ""
        )
        return f'<p data-role="recommendation">{lead}{body}{sure}</p>'
    if block.role == "override":
        over = (
            f"The memo recommends {escape(facts['Arm'])} over {escape(facts.get('Over', 'it'))}"
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
        return f'<div class="caveat"><strong>Caveat</strong> ({escape(facts.get("Kind", "unkinded"))}): {body}</div>'
    if block.role == "carried_forward":
        return f"<p><em>Carried forward:</em> {body}</p>"
    return f'<div data-role="{escape(block.role)}">{markdown_html(block.body)}</div>'


def _trial(block: TrialBlock) -> str:
    """One trial's record, open when the block says so: what it asked, what the judge saw and said, what failed,
    and where it is stored."""
    parts: list[str] = []
    if block.cause_words:
        parts.append(f"<dt>Why</dt><dd>{escape(block.cause_words)}</dd>")
    if block.first_input:
        parts.append(f"<dt>What the case asked</dt><dd><pre>{escape(block.first_input)}</pre></dd>")
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
    parts.append(f"<dt>Stored result</dt><dd><code>{escape(block.result_id)}</code> (read it with result_get)</dd>")
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
    if block.section == "warnings" or (block.section == "header" and block.anchor is not None):
        # A report-wide caveat, once, beside what warns: a flag too, in normal weight.
        return f'<p class="flag report-caveat"{anchor} data-source="{escape(block.source)}">{_FLAG_ICON}<span>{text}</span></p>'
    if block.source == "verdict":
        tone = f' data-tone="{escape(block.tone)}"' if block.tone is not None else ""
        return f'<p class="verdict"{tone}><strong>{text}</strong></p>'
    if block.section in FIRST_SCREEN:
        return f'<p class="line" data-source="{escape(block.source)}">{text}</p>'
    return f'<p class="disclosure" data-source="{escape(block.source)}">{text}</p>'


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
    reader where they were, and nothing on the page opens it. A trial id naming a trial the page does not hold
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
