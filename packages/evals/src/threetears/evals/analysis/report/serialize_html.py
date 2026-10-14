"""The report as one self-contained HTML page that needs no script.

**No script, ever, and nothing that runs.** The page is a document: every text node escaped, no ``<script>``, no
event-handler attribute, no external resource and no URL but a ``#`` link to a place on the page itself. Drill-down
is ``<details>``; moving between the screens is ``#`` anchors. The two charts are inline SVG
(:mod:`~threetears.evals.analysis.report.svg`), each beside a visually hidden table of the values it draws, so a
screen reader and a page with styles off lose nothing.

**The first screen is the page's ``<header>``**: the title line, the basis line (always line 2), the verdict line,
and the scoreboard with its outcome bar. Everything a reader needs to decide whether to read on is there, written by
code.

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
    Fact,
    Report,
    ReportBlock,
    TableBlock,
    TextBlock,
    TrialBlock,
)
from threetears.evals.analysis.report.svg import SVG_STYLE, chart_svg

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
.basis { font-weight: 600; margin: 0 0 8px; }
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

#: A case as the cases table numbers it, whose number its row's anchor is made of.
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
    sections = [section for section in SECTION_TITLES if section not in FIRST_SCREEN]
    present = [section for section in sections if any(block.section == section for block in report.blocks)]
    parts.append(
        "<nav>" + "".join(f'<a href="#{section}">{_escape(SECTION_TITLES[section])}</a>' for section in present) + "</nav>"
    )
    for section in present:
        blocks = [(index, block) for index, block in enumerate(report.blocks) if block.section == section]
        parts.append(f'<section id="{section}">')
        parts.append(f"<h2>{_escape(SECTION_TITLES[section])}</h2>")
        parts.extend(_block(block, index) for index, block in blocks)
        parts.append("</section>")
    parts += ["</article>", "</body>", "</html>"]
    return "\n".join(parts) + "\n"


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
    """An author's words, laid out by what they were written as, with code's facts beside them."""
    facts = _facts(block.facts)
    rests = (
        '<p class="facts">Rests on '
        + ", ".join(f'<a href="#F{number}">F{number}</a>' for number in block.rests_on_numbers)
        + ".</p>"
        if block.rests_on
        else ""
    )
    if block.role == "finding_title":
        anchor = f' id="{_escape(block.anchor)}"' if block.anchor else ""
        heading = f"<h3{anchor}>F{block.finding_number}. {_inline(_one_line(block.body))}</h3>"
        return heading + (f'<p class="facts">{facts}</p>' if facts else "")
    if block.role == "caveat":
        caveat = f'<strong>Caveat</strong> <span class="facts">({facts})</span>'
        return f'<div class="caveat">{caveat} {_markdown(block.body)}</div>'
    if block.role in ("revisit_when", "next_step_why"):
        label = "Revisit when" if block.role == "revisit_when" else "Why"
        return f"<div><em>{label}:</em> {_markdown(block.body)}</div>"
    if block.role == "carried_forward":
        return f"<div><em>Carried forward:</em> {_markdown(block.body)}</div>"
    if block.role in ("recommendation", "next_step"):
        lead = "Recommendation: " if block.role == "recommendation" else "Next: "
        body = f"<p><strong>{lead}{_inline(_one_line(block.body))}</strong></p>"
    else:
        body = _markdown(block.body)
    return (
        f'<div data-role="{_escape(block.role)}">{body}'
        + (f'<p class="facts">{facts}</p>' if facts else "")
        + f"{rests}</div>"
    )


def _table(block: TableBlock) -> str:
    """A table block: its caption and order, its rows, or a line saying it has none; folded when it is drill-down."""
    if not block.rows:
        grid = '<p class="order">(no rows)</p>'
    else:
        grid = _grid(block)
        if block.total_rows > len(block.rows):
            grid += f'<p class="order">{len(block.rows)} of {block.total_rows} rows shown.</p>'
    caption = f'<strong>{_escape(block.title)}</strong> <span class="order">({_escape(block.order)})</span>'
    if block.details:
        return f'<details data-table="{_escape(block.name)}"><summary>{caption}</summary>{grid}</details>'
    return f'<div class="table-block" data-table="{_escape(block.name)}"><p>{caption}</p>{grid}</div>'


def _grid(block: TableBlock) -> str:
    """A table of served cells' text, scrolling inside its own box rather than widening the page."""
    head = "".join(f'<th scope="col">{_escape(column.header)}</th>' for column in block.columns)
    body = []
    for row in block.rows:
        cells = []
        for column in block.columns:
            text = _escape(row[column.key].text)
            if block.name == "cases" and column.key == "trials":
                text = _TRIAL_ID.sub(lambda match: f'<a href="#{match.group(0)}">{match.group(0)}</a>', text)
            cells.append(f"<td>{text}</td>")
        anchor = ""
        # prose-canary: allow — the cases table's "case N" cell, which code numbered, not anything a model wrote
        if block.name == "cases" and "case" in row and (number := _CASE_NUMBER.fullmatch(row["case"].text)):
            anchor = f' id="case-{number.group(1)}"'
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
    """One trial's drill-down: what it asked, what the judge saw and said, what failed, and where the record is."""
    state = block.state if block.cause_words is None else f"{block.state}: {block.cause_words}"
    summary = (
        f"<strong>{_escape(block.trial_id)}</strong> · {_escape(block.case_label)} · {_escape(block.arm)} · "
        f"repeat {block.repeat} · {_escape(state)}"
    )
    parts: list[str] = []
    if block.first_input:
        parts.append(f"<dt>What the case asked</dt><dd><pre>{_escape(block.first_input)}</pre></dd>")
    if block.excerpt:
        parts.append(f"<dt>What the judge saw (its last part)</dt><dd><pre>{_escape(block.excerpt)}</pre></dd>")
    if block.failed_checks:
        checks = "".join(
            f"<li>{_escape(check.name)}: <code>{_escape(check.expression)}</code></li>" for check in block.failed_checks
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
        f"<summary>{summary}</summary><dl>{''.join(parts)}</dl></details>"
    )


def _disclosure(block: DisclosureBlock) -> str:
    """Something code had to say: the first screen's lines as lines, every other one set apart from the author's."""
    if block.source == "basis":
        return f'<p class="basis">{_escape(block.text)}</p>'
    if block.source == "verdict":
        return f'<p class="verdict"><strong>{_escape(block.text)}</strong></p>'
    if block.section in FIRST_SCREEN:
        return f'<p class="line" data-source="{_escape(block.source)}">{_escape(block.text)}</p>'
    return f'<div class="disclosure" data-source="{_escape(block.source)}">{_escape(block.text)}</div>'


def _facts(facts: Sequence[Fact]) -> str:
    """Facts code states beside an author's words, escaped, on one line; a ``See`` fact's cases linked."""
    rendered = []
    for fact in facts:
        if fact.name == "See":
            links = ", ".join(
                f'<a href="#case-{number.group(1)}">{_escape(case)}</a>'
                if (number := _CASE_NUMBER.fullmatch(case))
                else _escape(case)
                for case in fact.value.split(", ")
            )
            rendered.append(f"See {links}")
        else:
            rendered.append(f"{_escape(fact.name)}: {_escape(fact.value)}")
    return " · ".join(rendered)


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
