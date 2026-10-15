"""The report as one self-contained HTML page that needs no script.

**No script, ever, and nothing that runs.** The page is a document: every text node escaped, no ``<script>``, no
event-handler attribute, no external resource and no URL but a ``#`` link to a place on the page itself. Drill-down
is ``<details>``; moving between the screens is ``#`` anchors. The two charts are inline SVG
(:mod:`~threetears.evals.analysis.report.svg`), each beside a visually hidden table of the values it draws, so a
screen reader and a page with styles off lose nothing.

**The first screen is the page's ``<header>``**: the title (the campaign's name), the arms line (always line 2), the
declared question quoted whole where there is one, the verdict line, the scoreboard with its outcome bar, and the one
caveat in normal weight that changes how the verdict is read. Everything a reader needs to decide whether to read on
is there, written by code. Every term of art links to the glossary at the foot of the page.

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

#: The page's stylesheet: the system's own fonts (named once, as variables — the page loads no font), a readable measure, light and dark from the reader's setting (and a
#: ``data-theme`` a host's page may set), with the charts' own classes after it.
_STYLE = """
:root { color-scheme: light dark; --fg: #1d1d1f; --muted: #5f6368; --bg: #ffffff; --rule: #d9d9de; --note: #f4f4f6;
  --verdict: #fff4e5; --font-sans: system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-mono: ui-monospace, SFMono-Regular, Menlo, monospace; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { --fg: #ececf1; --muted: #a1a1aa; --bg: #16161a; --rule: #3a3a42; --note: #222228;
    --verdict: #2b2418; }
}
:root[data-theme="dark"] { --fg: #ececf1; --muted: #a1a1aa; --bg: #16161a; --rule: #3a3a42; --note: #222228;
  --verdict: #2b2418; }
body { font: 16px/1.55 var(--font-sans); color: var(--fg); background: var(--bg);
  margin: 0 auto; padding: 24px 16px 64px; max-width: 1040px; }
h1 { font-size: 1.45rem; line-height: 1.3; margin: 0 0 6px; }
h2 { font-size: 1.25rem; margin: 36px 0 8px; padding-bottom: 4px; border-bottom: 1px solid var(--rule); }
h3 { font-size: 1.05rem; margin: 24px 0 6px; }
header { border-bottom: 2px solid var(--rule); padding-bottom: 12px; }
.arms { margin: 0 0 8px; color: var(--muted); }
.finding { margin: 18px 0 6px; }
a.term { font-size: 0.8rem; color: var(--muted); }
.verdict { background: var(--verdict); border-left: 4px solid var(--muted); padding: 10px 14px; margin: 8px 0 12px; }
nav { font-size: 0.9rem; margin: 12px 0; }
nav a { margin-right: 14px; }
.facts, .order, .line { color: var(--muted); font-size: 0.9rem; }
.disclosure, .caveat { background: var(--note); border-left: 3px solid var(--rule); padding: 8px 12px; margin: 10px 0; }
.table-wrap { overflow-x: auto; margin: 8px 0; }
table { border-collapse: collapse; font-size: 0.9rem; font-variant-numeric: tabular-nums; }
th, td { border-bottom: 1px solid var(--rule); padding: 4px 10px; text-align: left; vertical-align: top; }
th { font-weight: 600; }
figure { margin: 16px 0; }
figcaption { margin-bottom: 6px; font-weight: 600; }
details { margin: 8px 0; }
details > summary { cursor: pointer; }
details.trial { border: 1px solid var(--rule); border-radius: 4px; padding: 6px 10px; }
details.trial dl { margin: 8px 0 0; }
details.trial dt { font-weight: 600; margin-top: 8px; }
details.trial dd { margin: 2px 0 0 0; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; background: var(--note); padding: 8px; margin: 4px 0;
  font: 0.85rem/1.45 var(--font-mono); max-height: 28rem; overflow-y: auto; }
p, li, .disclosure, .caveat { max-width: 75ch; }
code { font-family: var(--font-mono); font-size: 0.9em; }
"""

#: Inline Markdown a memo uses, applied to text already escaped — so its delimiters are the only markup it can make.
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_CODE = re.compile(r"`([^`]+)`")

#: A trial id in a table cell, linked to its drill-down.
_TRIAL_ID = re.compile(r"\bT\d+\b")

#: A case as the cases table numbers it, whose number its row's anchor is made of; a case named by its author anchors
#: by its name (``case-double-charge``).
_CASE_NUMBER = re.compile(r"case (\d+)")


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
    titles = report.section_titles
    sections = [section for section in SECTION_TITLES if section not in FIRST_SCREEN and section != "glossary"]
    present = [section for section in sections if any(block.section == section for block in report.blocks)]
    nav = [*present, *(["glossary"] if report.terms else [])]
    parts.append(
        "<nav>" + "".join(f'<a href="#{section}">{_escape(titles[section])}</a>' for section in nav) + "</nav>"
    )
    for section in present:
        blocks = [(index, block) for index, block in enumerate(report.blocks) if block.section == section]
        parts.append(f'<section id="{section}">')
        parts.append(f"<h2>{_escape(titles[section])}</h2>")
        parts.extend(_folded(blocks))
        parts.append("</section>")
    if report.terms:
        entries = "".join(
            f'<dt id="g-{_slug(term)}">{_escape(term)}</dt><dd>{_escape(GLOSSARY[term])}</dd>' for term in report.terms
        )
        parts.append(
            f'<section id="glossary"><h2>{titles["glossary"]}</h2>'
            f'<details class="glossary"><summary>Terms used on this page</summary><dl>{entries}</dl></details></section>'
        )
    parts += ["</article>", "</body>", "</html>"]
    return "\n".join(parts) + "\n"


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
    return f'<a href="#g-{_slug(term)}">{shown}</a>' if term else shown


def _code(text: str) -> str:
    """Text escaped, its code spans (a check's expression, a call to copy) set as code."""
    return _CODE.sub(r"<code>\1</code>", _escape(text))


def _grid(block: TableBlock) -> str:
    """A table of served cells' text, scrolling inside its own box rather than widening the page."""
    head = "".join(f'<th scope="col">{_header(column.header, column.term)}</th>' for column in block.columns)
    body = []
    for row in block.rows:
        cells = []
        for column in block.columns:
            text = _code(row[column.key].text)
            if (block.name == "cases" and column.key == "trials") or (block.name == "trials" and column.key == "trial"):
                text = _TRIAL_ID.sub(lambda match: f'<a href="#{match.group(0)}">{match.group(0)}</a>', text)
            cells.append(f"<td>{text}</td>")
        anchor = ""
        # prose-canary: allow — the cases table's "case N" cell, which code numbered, not anything a model wrote
        if block.name == "cases" and "case" in row:
            number = _CASE_NUMBER.fullmatch(row["case"].text)
            anchor = f' id="case-{number.group(1) if number else _slug(row["case"].text)}"'
        body.append(f"<tr{anchor}>" + "".join(cells) + "</tr>")
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


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
    """One trial's record, closed: what it asked, what the judge saw and said, what failed, and where it is stored."""
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
        text += f' <a class="term" href="#g-{_slug(block.term)}">({_escape(block.term)})</a>'
    if block.source == "arms":
        return f'<p class="arms">{text}</p>'
    if block.source == "verdict":
        return f'<p class="verdict"><strong>{text}</strong></p>'
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
