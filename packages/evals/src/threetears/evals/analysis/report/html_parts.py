"""The pieces of a report page every HTML writer shares: the page's head, a table, a chart, a status pill, an arm
chip, and the escaping and inline Markdown under every text node.

One owner for each, so a campaign's memo page and a page built from a report plan lay a table, a chart or a pill out
alike, byte for byte. Every function here escapes what it is given as text before it makes any markup of it, so no
field can open a ``<script>``, a ``<link>``, an event-handler attribute or a ``javascript:`` URL; a function taking
``markup`` takes HTML this package already wrote. The page's contract (one script, the kit's; nothing loaded but the
kit's fonts) is :mod:`~threetears.evals.analysis.report.serialize_html`'s.
"""

from __future__ import annotations

import html
import json
import re
from importlib.resources import files
from threetears.evals.analysis.report.blocks import ChartBlock, TableBlock, TableCell
from threetears.evals.analysis.report.charts import ChartStatus
from threetears.evals.analysis.report.svg import GLYPH_STYLE, interval_glyph_svg
from threetears.evals.analysis.quantities import PERCENT, is_money, number_in, with_unit
from threetears.evals.analysis.style import GLOSSARY

#: The kit's stylesheet, read once from the package: tokens for light and dark, type, the page and prose columns,
#: tables, pills and the flag (``kit/kit.css``, the kit's documentation copy byte for byte).
_KIT_CSS = files("threetears.evals.analysis.report").joinpath("kit", "kit.css").read_text(encoding="utf-8")

#: The kit's script, read once from the package (``kit/kit.js``, the kit's documentation copy byte for byte): it draws
#: every ``div.chart`` from the JSON in its ``data-chart`` at the chart's real width. The page carries it verbatim as its
#: one script, at the end of the body, and nothing else that runs.
KIT_JS = files("threetears.evals.analysis.report").joinpath("kit", "kit.js").read_text(encoding="utf-8")

#: The kit's fonts, as the reference memos load them: a preconnect to each Google Fonts host, then the one stylesheet
#: naming the three Plex faces. Written here and only here: no field of a report reaches the page's head.
_FONT_LINKS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500'
    '&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,600;1,400&display=swap">'
)

#: The page's own structures the kit has no class for, from the kit's tokens (no colour or face of its own): the first
#: screen, the sticky section links, the folds and their summaries, a table's title and its numeric, prose and glyph
#: columns, a chart's frame, the verdict line's tone, a trial's record and the code-written lines. A column header
#: keeps the case it was declared in, since a header can be an arm's name or a check's expression. The one measure of
#: its own is prose's 78ch cap: the page's measure test caps a line in characters, and 78ch at the body size is about
#: the kit's 44rem column, so prose set smaller still keeps a readable length.
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
.chart-values caption { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); }
.chart:not(:empty) + .chart-values { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px;
  overflow: hidden; clip-path: inset(50%); white-space: nowrap; border: 0; }
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
  details.trial, .tbl, figure { break-inside: avoid; }
}
"""


#: Inline Markdown a memo uses, applied to text already escaped — so its delimiters are the only markup it can make.
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_CODE = re.compile(r"`([^`]+)`")

#: A trial id in a table cell, linked to its drill-down.
TRIAL_ID = re.compile(r"\bT\d+\b")

#: The columns whose trial ids link to their records: the cases table's, the trials table's and a finding's facts.
_TRIAL_COLUMNS: frozenset[tuple[str, str]] = frozenset(
    {("cases", "trials"), ("trials", "trial"), ("finding_facts", "trials")}
)

#: A case as the cases table numbers it, whose number its row's anchor is made of; a case named by its author anchors
#: by its name (``case-double-charge``).
_CASE_NUMBER = re.compile(r"case (\d+)")

#: A cell of words at least this long is prose, laid out to wrap at a measure rather than widen its column.
_PROSE_CELL = 40


def page_head(title: str, *, static: bool) -> list[str]:
    """The page's ``<head>``: its title, the kit's font links (none on a static page) and the stylesheet — the kit's,
    then the page's own and the charts'. Nothing in it comes from a field but the title, escaped."""
    return [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{escape(one_line(title))}</title>",
        *([] if static else [_FONT_LINKS]),
        f"<style>{_KIT_CSS}{_PAGE_STYLE}{GLYPH_STYLE}</style>",
        "</head>",
    ]


def table_html(block: TableBlock) -> str:
    """A table block: its title and rows, or a line saying it has none; folded when it is drill-down."""
    if not block.rows:
        grid = '<p class="order">(no rows)</p>'
    else:
        grid = _grid(block)
        if block.total_rows > len(block.rows):
            grid += f'<p class="order">{len(block.rows)} of {block.total_rows} rows shown.</p>'
    title = escape(block.title)
    if block.title_ref is not None:
        title = f'<a href="#{escape(block.title_ref)}">{title}</a>'
    if block.details:
        # A closed table carries its own id where a finding's figure lands on it: on the fold, which every browser
        # shows, never on a cell inside it, which a browser that leaves the fold shut would not.
        anchor = f' id="{escape(block.anchor)}"' if block.anchor is not None else ""
        return f'<details{anchor} data-table="{escape(block.name)}"><summary>{title}</summary>{grid}</details>'
    if block.fold is not None:
        return f'<div class="table-block" data-table="{escape(block.name)}">{grid}</div>'
    return f'<div class="table-block" data-table="{escape(block.name)}"><h3>{title}</h3>{grid}</div>'


def _header(text: str, term: str | None) -> str:
    """A column header, linked to its glossary entry where it is a term of art; a code span (a check) as code."""
    shown = _cell_code(text)
    return term_link(term, shown) if term else shown


def term_link(term: str, label: str, *, css: str = "") -> str:
    """A link to a glossary term, its sentence as the link's title so a hover reads it without leaving the place."""
    style = f' class="{css}"' if css else ""
    return f'<a{style} href="#g-{glossary_slug(term)}" title="{escape(GLOSSARY[term])}">{label}</a>'


def code_html(text: str) -> str:
    """Text escaped, its code spans (a check's expression, a call to copy) set as code."""
    return _CODE.sub(r"<code>\1</code>", escape(text))


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
        start = escape(html.unescape(whole)[:_EXPRESSION_CHARS].rstrip())
        return f'<details class="expr"><summary><code>{start}…</code></summary><code>{whole}</code></details>'

    return _CODE.sub(span, escape(text))


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
                text = TRIAL_ID.sub(lambda match: f'<a href="#{match.group(0)}">{match.group(0)}</a>', text)
            if cell.status is not None:
                text = pill_html(cell.status, text)
            elif cell.tag is not None:
                text = _arm_chip(cell.tag, cell.color, text)
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
            anchor = f' id="case-{number.group(1) if number else glossary_slug(row["case"].text)}"'
        # A passed trial has no record of its own, so the link every table makes of its id lands on its row here.
        # prose-canary: allow — a trial id (T4), which code numbered, not anything a model wrote
        elif block.name == "passed_trials" and "trial" in row and TRIAL_ID.fullmatch(row["trial"].text):
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
    said = [f"{code_html(headers[key])} drawn from {' and from '.join(ends)}" for key, ends in spans.items()]
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


def chart_html(block: ChartBlock, chart_id: str, *, static: bool) -> str:
    """A chart as the kit's ``.fig``: its title, the chart the kit draws, and a table of every value it draws.

    The chart is a ``div.chart`` whose ``data-chart`` is its JSON, escaped into the attribute: the kit's script draws
    it at the chart's real width, every string in it as text. The table follows it. While the kit has drawn the chart
    the table is visually hidden, read by a screen reader; where the chart is empty (no script ran, or the kit refused
    the data) or there is no chart to draw, the table is the figure, in its own scrolling frame.

    A static page writes no ``div.chart`` at all, so no script is wanted to draw one, no empty box stands where it
    would be, and the one rule that hides the table (``.chart:not(:empty) + .chart-values``) never matches: every
    figure is its title and its table."""
    headers, rows = block.values_table()
    head = "".join(f'<th scope="col">{escape(header)}</th>' for header in headers)
    body = "".join("<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in row) + "</tr>" for row in rows)
    drawn = None if static else block.kit_chart()
    chart = ""
    if drawn is not None:
        kind, data = drawn
        chart = (
            f'<div class="chart" id="{escape(chart_id)}" data-kind="{escape(kind)}" '
            f'data-chart="{escape(json.dumps(data, separators=(",", ":"), allow_nan=False))}"></div>'
        )
    return (
        f'<figure class="fig" data-figure="{escape(block.chart)}">'
        f"<figcaption>{escape(block.title)}</figcaption>{chart}"
        f'<div class="tbl chart-values"><table><caption>{escape(block.title)}</caption><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div></figure>"
    )


def glossary_slug(term: str) -> str:
    """A glossary term as an anchor id."""
    return re.sub(r"[^a-z0-9]+", "-", term.casefold()).strip("-")


def markdown_html(text: str) -> str:
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
            out.append("<ul>" + "".join(f"<li>{inline_html(line[2:])}</li>" for line in lines) + "</ul>")
        else:
            out.append("<p>" + " ".join(inline_html(line) for line in lines) + "</p>")
    return "".join(out)


def inline_html(text: str) -> str:
    """Escape ``text``, then turn its bold and code spans into markup."""
    return _CODE.sub(r"<code>\1</code>", _BOLD.sub(r"<strong>\1</strong>", escape(text)))


def escape(text: str) -> str:
    """Escape text for an element or a quoted attribute."""
    return html.escape(text, quote=True)


def one_line(text: str) -> str:
    """Text a heading carries, on one line."""
    return " ".join(text.split("\n")).strip()


#: Each status's icon on the kit's pill, beside the word that carries the meaning, so a status never rests on a colour
#: alone: the kit's own icons (``kit.js``'s ``pill``).
_PILL_ICONS: dict[ChartStatus, str] = {"good": "✓", "warn": "!", "bad": "✕", "neutral": "–"}


def pill_html(status: ChartStatus, markup: str) -> str:
    """A status as the kit's pill: its colour, its icon (hidden from a screen reader, which reads the word) and
    ``markup``, the word, already escaped.

    :param status: The status: the pill's colour and icon.
    :ptype status: ChartStatus
    :param markup: The word the pill says, as HTML this package wrote (escaped text).
    :ptype markup: str
    :return: The pill.
    :rtype: str
    """
    return f'<span class="pill {status}"><span aria-hidden="true">{_PILL_ICONS[status]}</span>{markup}</span>'


#: The colour an arm chip's tag takes where the cell names none: the kit's (``kit.js``'s ``kArmChip``), the control's
#: ink-gray.
_CHIP_TOKEN = "--ink-2"


def _arm_chip(tag: str, color: str | None, markup: str) -> str:
    """An arm as the kit's chip: its letter tag on its identity colour, then ``markup``, its name, already escaped.

    The tag's letter is the kit's on-fill colour, except on the control's ink-gray, which is light in the dark theme
    and takes the surface colour instead (``kit.js``'s ``kOnTag``).
    """
    token = color or _CHIP_TOKEN
    on_tag = "--surface" if token == "--ink-2" else "--on-fill"
    return (
        f'<span class="armchip"><span class="tag" style="background:var({token});color:var({on_tag})">'
        f"{escape(tag)}</span>{markup}</span>"
    )


__all__ = [
    "KIT_JS",
    "TRIAL_ID",
    "chart_html",
    "code_html",
    "escape",
    "glossary_slug",
    "inline_html",
    "markdown_html",
    "one_line",
    "page_head",
    "pill_html",
    "table_html",
    "term_link",
]
