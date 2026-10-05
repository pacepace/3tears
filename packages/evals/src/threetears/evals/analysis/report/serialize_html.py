"""The report as HTML that reads without a script.

**No script, ever, and nothing that runs.** The page is a document: every text node escaped, no
``<script>``, no event-handler attribute, no external resource, no URL at all. A chart block is a
``<figure>`` holding its title, the author's caption, its values as drawn as a ``<table>`` and its
disclosures — so the page loses nothing a chart claims with scripting off — and it carries its
:class:`~threetears.evals.analysis.viz.intent.ChartIntent` as JSON in a ``data-chart-intent`` attribute,
which a host's own renderer reads to draw the picture beside or over the table. The package draws no
picture here: how a chart looks is the host's.

The author's words are Markdown; this renders the part of it a memo uses — paragraphs, bullet lists,
``**bold**`` and ``code`` — over text that is escaped first, so nothing an author wrote can become
markup.
"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Sequence

from threetears.evals.analysis.report.model import (
    SECTION_TITLES,
    ChartBlock,
    DisclosureBlock,
    Fact,
    Report,
    ReportBlock,
    TableBlock,
    TextBlock,
    finding_number,
)
from threetears.evals.analysis.report.words import positions
from threetears.evals.analysis.viz.intent import Cell
from threetears.evals.analysis.viz.quantities import render_cell

#: The page's whole stylesheet: system fonts, a readable measure, light and dark from the reader's setting.
_STYLE = """
:root { color-scheme: light dark; --fg: #1d1d1f; --muted: #5f6368; --bg: #ffffff; --rule: #d9d9de; --note: #f4f4f6; }
@media (prefers-color-scheme: dark) {
  :root { --fg: #ececf1; --muted: #a1a1aa; --bg: #16161a; --rule: #3a3a42; --note: #222228; }
}
body { font: 16px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--fg); background: var(--bg);
  margin: 0 auto; padding: 24px 16px 64px; max-width: 920px; }
h1 { font-size: 1.7rem; line-height: 1.25; margin: 0 0 4px; }
h2 { font-size: 1.25rem; margin: 36px 0 8px; padding-bottom: 4px; border-bottom: 1px solid var(--rule); }
h3 { font-size: 1.05rem; margin: 24px 0 6px; }
.source, .facts, .order { color: var(--muted); font-size: 0.9rem; }
.disclosure, .caveat { background: var(--note); border-left: 3px solid var(--rule); padding: 8px 12px; margin: 10px 0; }
.table-wrap { overflow-x: auto; margin: 8px 0; }
table { border-collapse: collapse; font-size: 0.9rem; font-variant-numeric: tabular-nums; }
th, td { border-bottom: 1px solid var(--rule); padding: 4px 10px; text-align: left; vertical-align: top; }
th { font-weight: 600; }
figure { margin: 16px 0; }
figcaption { margin-bottom: 6px; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.9em; }
"""

#: Inline Markdown a memo uses, applied to text already escaped — so its delimiters are the only markup it can make.
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_CODE = re.compile(r"`([^`]+)`")


def report_html(report: Report) -> str:
    """Render a report as a standalone HTML page that needs no script.

    Args:
        report: The report.

    Returns:
        The page.
    """
    source = report.source
    title = report.headline.strip() or "(blank headline)"
    parts = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{_escape(_one_line(title))}</title>",
        f"<style>{_STYLE}</style>",
        "</head>",
        "<body>",
        "<article>",
        f"<h1>{_inline(_one_line(title))}</h1>",
        (
            f'<p class="source">Analysis {_escape(source.analysis_id)} of campaign {_escape(source.campaign_id)} — '
            f"{_escape(source.behavior)}; generated {_escape(source.generated_at)} by {_escape(source.generator_model)}.</p>"
        ),
    ]
    for section, heading in SECTION_TITLES.items():
        blocks = [block for block in report.blocks if block.section == section]
        if not blocks:
            continue
        parts.append(f'<section data-section="{section}">')
        parts.append(f"<h2>{_escape(heading)}</h2>")
        parts.extend(_block(block) for block in blocks)
        parts.append("</section>")
    parts += ["</article>", "</body>", "</html>"]
    return "\n".join(parts) + "\n"


def _block(block: ReportBlock) -> str:
    """One block as HTML."""
    if isinstance(block, TextBlock):
        return _text(block)
    if isinstance(block, TableBlock):
        return (
            f'<div class="table-block" data-table="{_escape(block.name)}">'
            f'<p><strong>{_escape(block.title)}</strong> <span class="order">({_escape(block.order)})</span></p>'
            f"{_table(block)}</div>"
        )
    if isinstance(block, ChartBlock):
        return _chart(block)
    return _disclosure(block)


def _text(block: TextBlock) -> str:
    """An author's words, laid out by what they were written as, with code's facts beside them."""
    facts = _facts(block.facts)
    rests = f'<p class="facts">Rests on finding {positions(block.rests_on)}.</p>' if block.rests_on else ""
    if block.role == "finding_title":
        heading = f"<h3>{finding_number(block)}. {_inline(_one_line(block.body))}</h3>"
        return heading + (f'<p class="facts">{facts}</p>' if facts else "")
    if block.role == "caveat":
        return f'<div class="caveat"><strong>Caveat</strong> <span class="facts">({facts})</span> {_markdown(block.body)}</div>'
    if block.role in ("revisit_when", "next_step_why"):
        label = "Revisit when" if block.role == "revisit_when" else "Why"
        return f"<div><em>{label}:</em> {_markdown(block.body)}</div>"
    if block.role == "carried_forward":
        return f"<div><em>Carried forward:</em> {_markdown(block.body)}</div>"
    lead = (
        f"<p><strong>{_inline(_one_line(block.body))}</strong></p>" if block.role in ("decision", "next_step") else ""
    )
    body = "" if lead else _markdown(block.body)
    return (
        f'<div data-role="{block.role}">{lead}{body}'
        + (f'<p class="facts">{facts}</p>' if facts else "")
        + f"{rests}</div>"
    )


def _chart(block: ChartBlock) -> str:
    """A chart as a figure: its values as drawn, its words, and its intent for a host renderer to draw."""
    if block.intent is None:
        return (
            f'<div class="disclosure" data-chart-error="{_escape(block.viz_type)}">A {_escape(block.viz_type)} chart is '
            f"stored for this finding and cannot be drawn: {_escape(block.error)}</div>"
        )
    intent = block.intent
    payload = json.dumps(intent.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
    caption = f" {_inline(_one_line(intent.caption))}" if intent.caption.strip() else ""
    notes = [line for line in (intent.footnote, *intent.disclosures) if line]
    listed = "<ul>" + "".join(f"<li>{_escape(note)}</li>" for note in notes) + "</ul>" if notes else ""
    table = _grid(
        [column.header for column in intent.columns],
        [[row.get(column.key) for column in intent.columns] for row in intent.rows],
    )
    return (
        f'<figure class="chart" data-chart-type="{_escape(intent.type)}" data-chart-intent="{_escape(payload)}">'
        f"<figcaption><strong>{_escape(intent.title)}</strong>{caption}</figcaption>"
        f"{table}{listed}</figure>"
    )


def _disclosure(block: DisclosureBlock) -> str:
    """Something code had to say, set apart from the author's words."""
    return f'<div class="disclosure" data-source="{block.source}">{_escape(block.text)}</div>'


def _table(block: TableBlock) -> str:
    """A table block's rows, or a line saying it has none."""
    if not block.rows:
        return '<p class="order">(no rows)</p>'
    grid = _grid(
        [column.header for column in block.columns],
        [[row.get(column.key) for column in block.columns] for row in block.rows],
    )
    if block.total_rows > len(block.rows):
        grid += f'<p class="order">{len(block.rows)} of {block.total_rows} rows shown.</p>'
    return grid


def _grid(headers: Sequence[str], rows: Sequence[Sequence[Cell]]) -> str:
    """A table, scrolling inside its own box rather than widening the page."""
    head = "".join(f'<th scope="col">{_escape(header)}</th>' for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_escape(render_cell(value))}</td>" for value in row) + "</tr>" for row in rows
    )
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _facts(facts: Sequence[Fact]) -> str:
    """Facts code states beside an author's words, escaped, on one line."""
    return " · ".join(f"{_escape(fact.name)}: {_escape(fact.value)}" for fact in facts)


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
