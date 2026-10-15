"""The report as one self-contained HTML page that needs no script.

**No script, ever, and nothing that runs.** The page is a document: every text node escaped, no ``<script>``, no
event-handler attribute, no external resource and no URL but a ``#`` link to a place on the page itself. Drill-down
is ``<details>``; moving between the screens is ``#`` anchors. The two charts are inline SVG
(:mod:`~threetears.evals.analysis.report.svg`), each beside a visually hidden table of the values it draws, so a
screen reader and a page with styles off lose nothing.

**The first screen is the page's ``<header>``**: the title (the question), the arms line (always line 2), the verdict
line, the scoreboard with its outcome bar, and the caveats in normal weight. Everything a reader needs to decide
whether to read on is there, written by code. Every term of art links to the glossary at the foot of the page.

**Depth on demand, with the least ink.** The result, the decision rule, what it means and the contrasts are open;
the cases, the trials, what was not tested, the methods and the glossary each sit in a closed ``<details>`` whose
summary is the section heading, so a reader opens what they came for. The stylesheet draws hairlines, not boxes:
hierarchy is type size, weight and spacing, a thin rule in a hue is the only tint (the verdict line's kind, a failed
or lost trial), and every meaning a colour carries is also in words.

The author's words are Markdown; this renders the part of it a memo uses — paragraphs, bullet lists, ``**bold**``
and ``code`` — over text escaped first, so nothing an author wrote can become markup.
"""

from __future__ import annotations

import html
import re
from collections.abc import Sequence

from threetears.evals.analysis.report.model import (
    FIRST_SCREEN,
    SECTION_TITLES,
    ChartBlock,
    DisclosureBlock,
    Report,
    ReportBlock,
    TableBlock,
    TextBlock,
    TrialBlock,
)
from threetears.evals.analysis.report.serialize_md import RESOLUTION_WORDS, trial_summary
from threetears.evals.analysis.report.svg import SVG_STYLE, chart_svg
from threetears.evals.analysis.style import GLOSSARY

#: The page's stylesheet: the system's own fonts (named once, as variables — the page loads no font), a readable
#: measure, light and dark from the reader's setting (and a ``data-theme`` a host's page may set), with the charts'
#: own classes after it. Every colour is a variable both schemes define; spacing is on one scale (4, 8, 16, 24, 40).
#: The tone hues set the verdict line's kind and a failed or lost trial apart by a thin rule only, never a wash.
_STYLE = """
:root { color-scheme: light dark; --fg: #1d1d1f; --muted: #5f6368; --bg: #ffffff; --rule: #e3e3e8;
  --rule-strong: #c4c4cc; --tone-decided: #2a78d6; --tone-open: #b8791a; --tone-failed: #d2501f;
  --tone-summary: #8a8a92; --font-sans: system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-mono: ui-monospace, SFMono-Regular, Menlo, monospace; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { --fg: #ececf1; --muted: #a1a1aa; --bg: #16161a; --rule: #2c2c33;
    --rule-strong: #4a4a54; --tone-decided: #4a90e2; --tone-open: #d9a44a; --tone-failed: #e0623a;
    --tone-summary: #8a8a92; }
}
:root[data-theme="dark"] { --fg: #ececf1; --muted: #a1a1aa; --bg: #16161a; --rule: #2c2c33; --rule-strong: #4a4a54;
  --tone-decided: #4a90e2; --tone-open: #d9a44a; --tone-failed: #e0623a; --tone-summary: #8a8a92; }
html { scroll-padding-top: 48px; }
body { font: 16px/1.5 var(--font-sans); color: var(--fg); background: var(--bg); margin: 0 auto;
  padding: 24px 16px 64px; max-width: 1040px; }
h1 { font-size: 1.6rem; line-height: 1.25; letter-spacing: -0.01em; margin: 0 0 8px; max-width: 60ch; }
h2 { font-size: 1.25rem; line-height: 1.3; font-weight: 600; margin: 40px 0 12px; padding-bottom: 6px;
  border-bottom: 1px solid var(--rule); }
h3, figcaption { font-size: 0.9375rem; font-weight: 600; margin: 24px 0 4px; }
a { color: inherit; text-decoration: underline; text-decoration-color: var(--rule-strong); text-underline-offset: 2px; }
a:hover { text-decoration-color: var(--fg); }
a.term { font-size: 0.8125rem; color: var(--muted); text-decoration: underline dotted; }
th a { text-decoration: underline dotted; }
p, li, .disclosure, .caveat { max-width: 75ch; }
code { font-family: var(--font-mono); font-size: 0.9em; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; margin: 4px 0; padding: 0 0 0 12px;
  border-left: 1px solid var(--rule); font: 0.8125rem/1.45 var(--font-mono); max-height: 24rem; overflow-y: auto; }
header { padding: 0 0 16px; }
.arms { margin: 0 0 16px; color: var(--muted); }
.verdict { font-size: 1.125rem; line-height: 1.4; margin: 16px 0 24px; padding: 2px 0 2px 14px;
  border-left: 3px solid var(--rule-strong); }
.verdict[data-tone="decided"] { border-left-color: var(--tone-decided); }
.verdict[data-tone="open"] { border-left-color: var(--tone-open); }
.verdict[data-tone="failed"] { border-left-color: var(--tone-failed); }
.verdict[data-tone="summary"] { border-left-color: var(--tone-summary); }
.facts, .order, .line { color: var(--muted); font-size: 0.875rem; }
.line { margin: 4px 0; }
.line[data-source="caveat"] { color: var(--fg); font-size: 0.9375rem; }
nav { position: sticky; top: 0; z-index: 1; display: flex; gap: 20px; overflow-x: auto; white-space: nowrap;
  font-size: 0.875rem; padding: 10px 0; margin: 0 0 8px; background: var(--bg); border-bottom: 1px solid var(--rule); }
nav a { color: var(--muted); text-decoration: none; }
nav a:hover { color: var(--fg); }
.table-wrap { overflow: auto; max-height: 80vh; margin: 4px 0 8px; }
table { border-collapse: separate; border-spacing: 0; font-size: 0.9375rem; font-variant-numeric: tabular-nums; }
th, td { padding: 6px 20px 6px 0; text-align: left; vertical-align: top; border-bottom: 1px solid var(--rule); }
th:last-child, td:last-child { padding-right: 0; }
thead th { font-weight: 600; border-bottom: 1px solid var(--rule-strong); position: sticky; top: 0;
  background: var(--bg); }
th.num, td.num { text-align: right; }
td.key { font-weight: 600; }
td.prose { max-width: 36ch; }
[data-table="scoreboard"] table { font-size: 1rem; }
figure { margin: 16px 0; overflow-x: auto; }
figcaption { margin: 0 0 4px; }
details { margin: 8px 0; }
summary { cursor: pointer; list-style: none; }
summary::-webkit-details-marker { display: none; }
summary::before { content: "\\203A"; display: inline-block; width: 1em; color: var(--muted); transition: transform 0.1s; }
details[open] > summary::before { transform: rotate(90deg); }
details.section { margin: 40px 0 0; }
details.section > summary { padding-bottom: 6px; border-bottom: 1px solid var(--rule); }
details.section > summary > h2 { display: inline; margin: 0; padding: 0; border: 0; }
.count { color: var(--muted); font-size: 0.9375rem; }
details.fold, details[data-table] { margin: 8px 0 16px; }
details.fold > summary, details[data-table] > summary { font-size: 0.875rem; color: var(--muted); }
.finding { font-size: 1.0625rem; margin: 28px 0 8px; }
p[data-role] > strong:first-child, .caveat > strong:first-child { font-size: 0.8125rem; color: var(--muted);
  letter-spacing: 0.02em; }
.disclosure, .caveat { font-size: 0.875rem; color: var(--muted); margin: 8px 0; padding: 0 0 0 12px;
  border-left: 1px solid var(--rule-strong); }
details.trial { margin: 0; padding: 8px 0 8px 12px; border-top: 1px solid var(--rule);
  border-left: 2px solid transparent; }
details.trial:last-of-type { border-bottom: 1px solid var(--rule); }
details.trial[data-state="failed"] { border-left-color: var(--tone-failed); }
details.trial[data-state="lost"] { border-left-color: var(--tone-summary); }
details.trial > summary { font-size: 0.9375rem; }
details.trial dl { margin: 8px 0 0; font-size: 0.9375rem; }
details.trial dt { font-weight: 600; margin-top: 8px; }
details.trial dd { margin: 2px 0 0 0; }
.trial-links { color: var(--muted); font-size: 0.875rem; }
dl.glossary { margin: 12px 0 0; }
dl.glossary dt { font-weight: 600; margin-top: 10px; }
dl.glossary dd { margin: 2px 0 0 16px; max-width: 72ch; }
@media (max-width: 480px) {
  h1 { font-size: 1.35rem; }
  .verdict { font-size: 1rem; }
  nav { gap: 16px; }
}
@media print {
  nav { display: none; }
  body { padding: 0; max-width: none; }
  .table-wrap { overflow: visible; max-height: none; }
  thead th { position: static; }
  figure { overflow: visible; }
  svg.viz { min-width: 0; }
  details.trial, .table-wrap, figure { break-inside: avoid; }
}
"""

#: The sections a reader opens on demand: everything after the result, the decision, what it means and the contrasts.
_FOLDED_SECTIONS: frozenset[str] = frozenset({"cases", "trials", "not_tested", "methods", "glossary"})

#: The folded sections whose own table lists every one of their units, so the summary can say how many: the cases.
#: Not the trials — their table lists only the failed and lost, and the section's census line counts the rest.
_COUNTED_SECTIONS: frozenset[str] = frozenset({"cases"})

#: Inline Markdown a memo uses, applied to text already escaped — so its delimiters are the only markup it can make.
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_CODE = re.compile(r"`([^`]+)`")

#: A trial id in a table cell, linked to its drill-down.
_TRIAL_ID = re.compile(r"\bT\d+\b")

#: A case as the cases table numbers it, whose number its row's anchor is made of.
_CASE_NUMBER = re.compile(r"case (\d+)")

#: A cell of words at least this long is prose, laid out to wrap at a measure rather than widen its column.
_PROSE_CELL = 40


def report_html(report: Report) -> str:
    """Render a report as one standalone HTML page that needs no script.

    Args:
        report: The report.

    Returns:
        The page.
    """
    head = [block for block in report.blocks if block.section in FIRST_SCREEN]
    parts = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{_escape(_one_line(report.title))}</title>",
        f"<style>{_STYLE}{SVG_STYLE}</style>",
        "</head>",
        "<body>",
        f'<article data-basis="{_escape(report.basis)}">',
        "<header>",
        f"<h1>{_escape(_one_line(report.title))}</h1>",
        *(_block(block, index) for index, block in enumerate(head)),
        "</header>",
    ]
    sections = [section for section in SECTION_TITLES if section not in FIRST_SCREEN and section != "glossary"]
    present = [section for section in sections if any(block.section == section for block in report.blocks)]
    nav = [*present, *(["glossary"] if report.terms else [])]
    parts.append(
        "<nav>" + "".join(f'<a href="#{section}">{_escape(SECTION_TITLES[section])}</a>' for section in nav) + "</nav>"
    )
    for section in present:
        blocks = [(index, block) for index, block in enumerate(report.blocks) if block.section == section]
        parts.append(f'<section id="{section}">')
        parts.extend(_section(section, _folded(blocks), count=_section_count(section, blocks)))
        parts.append("</section>")
    if report.terms:
        entries = "".join(
            f'<dt id="g-{_slug(term)}">{_escape(term)}</dt><dd>{_escape(GLOSSARY[term])}</dd>' for term in report.terms
        )
        parts.append('<section id="glossary">')
        parts.extend(_section("glossary", [f'<dl class="glossary">{entries}</dl>'], count=None))
        parts.append("</section>")
    parts += ["</article>", "</body>", "</html>"]
    return "\n".join(parts) + "\n"


def _section(section: str, body: Sequence[str], *, count: int | None) -> list[str]:
    """A section's heading and body: open, or closed in a ``<details>`` whose summary is the heading and its size."""
    heading = f"<h2>{_escape(SECTION_TITLES[section])}</h2>"
    if section not in _FOLDED_SECTIONS:
        return [heading, *body]
    size = f' <span class="count">· {count}</span>' if count is not None else ""
    return [f'<details class="section"><summary>{heading}{size}</summary>', *body, "</details>"]


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
        return f'<p data-role="recommendation">{lead}{body}</p>'
    if block.role == "next_step":
        return f'<p data-role="next_step"><strong>Next:</strong> {body}</p>'
    if block.role == "revisit_when":
        return f"<p><em>Revisit when:</em> {body}</p>"
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
    if block.details:
        return f'<details data-table="{_escape(block.name)}"><summary>{title}</summary>{grid}</details>'
    if block.fold is not None:
        return f'<div class="table-block" data-table="{_escape(block.name)}">{grid}</div>'
    return f'<div class="table-block" data-table="{_escape(block.name)}"><h3>{title}</h3>{grid}</div>'


def _header(text: str, term: str | None) -> str:
    """A column header, linked to its glossary entry where it is a term of art; a code span (a check) as code."""
    shown = _code(text)
    return _term_link(term, shown) if term else shown


def _term_link(term: str, label: str, *, css: str = "") -> str:
    """A link to a glossary term, its sentence as the link's title so a hover reads it without leaving the place."""
    style = f' class="{css}"' if css else ""
    return f'<a{style} href="#g-{_slug(term)}" title="{_escape(GLOSSARY[term])}">{label}</a>'


def _code(text: str) -> str:
    """Text escaped, its code spans (a check's expression, a call to copy) set as code."""
    return _CODE.sub(r"<code>\1</code>", _escape(text))


def _grid(block: TableBlock) -> str:
    """A table of served cells' text, scrolling inside its own box rather than widening the page.

    A column of numbers is classed ``num`` on every cell, header included, so the stylesheet right-aligns it; a cell
    of prose is classed ``prose`` so it wraps at a measure instead of widening its column.
    """
    classes = {column.key: _column_class(block, column.key) for column in block.columns}
    head = "".join(
        f'<th scope="col"{classes[column.key]}>{_header(column.header, column.term)}</th>' for column in block.columns
    )
    body = []
    for row in block.rows:
        cells = []
        for column in block.columns:
            cell = row[column.key]
            text = _code(cell.text)
            if (block.name == "cases" and column.key == "trials") or (block.name == "trials" and column.key == "trial"):
                text = _TRIAL_ID.sub(lambda match: f'<a href="#{match.group(0)}">{match.group(0)}</a>', text)
            if block.name == "cases" and column.key == "trials":
                # Navigation beside the marks, not information: the stylesheet mutes it.
                css = ' class="trial-links"'
            else:
                css = classes[column.key] or (' class="prose"' if len(cell.text) >= _PROSE_CELL else "")
            cells.append(f"<td{css}>{text}</td>")
        anchor = ""
        # prose-canary: allow — the cases table's "case N" cell, which code numbered, not anything a model wrote
        if block.name == "cases" and "case" in row and (number := _CASE_NUMBER.fullmatch(row["case"].text)):
            anchor = f' id="case-{number.group(1)}"'
        # A passed trial has no record of its own, so the link every table makes of its id lands on its row here.
        elif block.name == "passed_trials" and "trial" in row and _TRIAL_ID.fullmatch(row["trial"].text):
            anchor = f' id="{row["trial"].text}"'
        body.append(f"<tr{anchor}>" + "".join(cells) + "</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


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
    """A chart as a figure: its title, the SVG, and a visually hidden table of every value it draws."""
    headers, rows = block.values_table()
    head = "".join(f'<th scope="col">{_escape(header)}</th>' for header in headers)
    body = "".join("<tr>" + "".join(f"<td>{_escape(cell)}</td>" for cell in row) + "</tr>" for row in rows)
    return (
        f'<figure class="chart" data-chart="{_escape(block.chart)}">'
        f"<figcaption>{_escape(block.title)}</figcaption>"
        f"{chart_svg(block, chart_id=chart_id)}"
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
        f"<summary>{_escape(trial_summary(block))}</summary><dl>{''.join(parts)}</dl></details>"
    )


def _disclosure(block: DisclosureBlock) -> str:
    """Something code had to say: the first screen's lines as lines, every other one set apart from the author's."""
    text = _code(block.text)
    if block.term is not None:
        text += " " + _term_link(block.term, f"({_escape(block.term)})", css="term")
    if block.source == "arms":
        return f'<p class="arms">{text}</p>'
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

    Args:
        text: An author's words.

    Returns:
        The HTML.
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


def _escape(text: str) -> str:
    """Escape text for an element or a quoted attribute."""
    return html.escape(text, quote=True)


def _one_line(text: str) -> str:
    """Text a heading carries, on one line."""
    return " ".join(text.split("\n")).strip()


__all__ = [
    "report_html",
]
