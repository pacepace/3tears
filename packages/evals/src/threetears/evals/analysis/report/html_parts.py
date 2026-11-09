"""The pieces of a report page every HTML writer shares: the page's head, a table, a chart, a status pill, an arm
chip, and the escaping and inline Markdown under every text node.

One owner for each, so a campaign's page and a report document's page lay a table, a chart or a pill out
alike, byte for byte. Every function here escapes what it is given as text before it makes any markup of it, so no
field can open a ``<script>``, a ``<link>``, an event-handler attribute or a ``javascript:`` URL; a function taking
``markup`` takes HTML this package already wrote. The page's contract (one script, the kit's; nothing loaded) is
:mod:`~threetears.evals.analysis.report.serialize_html`'s.
"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from importlib.resources import files
from threetears.evals.analysis.report.blocks import ChartBlock, TableBlock, TableCell
from threetears.evals.analysis.report.chart_svg import AS_TABLE_NOTE, CHART_STYLE, chart_drawing
from threetears.evals.analysis.report.charts import ChartStatus
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.analysis.report.kit_colours import ink_on
from threetears.evals.analysis.report.svg import GLYPH_STYLE, interval_glyph_svg
from threetears.evals.analysis.quantities import PERCENT, is_money
from threetears.evals.analysis.style import CHECK_LINK_TITLE, GLOSSARY
from threetears.evals.schema.hashing import round_floats

#: The kit's stylesheet, read once from the package: tokens for light and dark, type, the page and prose columns,
#: tables, pills and the flag (``kit/kit.css``, the kit's documentation copy byte for byte).
_KIT_CSS = files("threetears.evals.analysis.report").joinpath("kit", "kit.css").read_text(encoding="utf-8")

#: The kit's script, read once from the package (``kit/kit.js``, the kit's documentation copy byte for byte): it draws
#: every ``div.chart`` from the JSON in its ``data-chart`` at the chart's real width. The page carries it verbatim as its
#: one script, at the end of the body, and nothing else that runs.
KIT_JS = files("threetears.evals.analysis.report").joinpath("kit", "kit.js").read_text(encoding="utf-8")

#: The page's one opening ``<html>`` tag, written once, right after the doctype, with nothing but its language: a host
#: that frames the page rewrites this tag to stamp ``data-theme`` and an inline ``style`` setting the kit's
#: ``--host-*`` tokens (``docs/adopting-a-host.md``), and leaves a page with any other count of it untouched.
_HTML_OPEN = '<html lang="en">'

#: The page's own structures the kit has no class for, from the kit's tokens (no colour or face of its own): the first
#: screen, the sticky section links (painted over every table's pinned cells, which their frame isolates), the folds
#: and their summaries, a table's title and its numeric, prose and glyph columns, a chart's frame, the verdict line's
#: tone, a trial's record and the code-written lines. A column header keeps the case it was declared in, since a header
#: can be an arm's name or a check's expression. The one measure of
#: its own is prose's 78ch cap: the page's measure test caps a line in characters, and 78ch at the body size is about
#: the kit's 44rem column, so prose set smaller still keeps a readable length. Nothing may widen the page at phone
#: width: a long unbroken token in prose (a dotted metric path) wraps, and a cell grid too narrow for its labels scrolls
#: inside its own ``.tbl`` frame (a cell-grid chart is written inside one, since the render checks exempt only that frame).
_PAGE_STYLE = """
html { scroll-padding-top: 3.5rem; }
p, li, .disclosure, .caveat { max-width: 78ch; }
ul { margin: 0; padding-left: 1.25rem; }
p, li, dd, figcaption, summary, .disclosure, .caveat { overflow-wrap: anywhere; }
.mx { min-width: max-content; }
h4, h5 { font-family: var(--f-display); font-weight: 600; margin: 0; }
h4 { font-size: var(--t-base); }
h5 { font-size: var(--t-sm); }
article > header { display: grid; gap: .75rem; padding-block: 1.75rem 1.5rem; }
/* a code page's title is the campaign's name, a label over the lead, so it sets at the scale's 20 px step */
article > header > h1 { max-width: var(--col); font-size: var(--t-lg); line-height: 1.25; }
.question { font-size: var(--t-base); line-height: 1.45; color: var(--ink-2); }
.arms { font-size: var(--t-sm); color: var(--muted); }
.verdict { font-size: var(--t-base); line-height: 1.45; padding: .15rem 0 .15rem .9rem; border-left: 3px solid var(--base); }
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
nav { position: sticky; top: 0; z-index: 3; display: flex; flex-wrap: wrap; gap: .25rem 1.25rem; padding: .6rem 0;
  font-size: var(--t-sm); background: var(--bg); border-bottom: 1px solid var(--line); }
nav a { color: var(--ink-2); text-decoration: none; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  max-width: min(100%, 22rem); }
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
/* a header's term is underlined dotted at the first header linking it; a later one (seen) keeps its link, plain */
th a { color: inherit; text-decoration: underline dotted; }
th a.seen { text-decoration: none; }
th code { font-size: 1em; }
td, tbody th { vertical-align: top; }
th.num, td.num { text-align: right; font-variant-numeric: tabular-nums; }
td.num { white-space: nowrap; }
td.key { font-weight: 600; }
[data-table="scoreboard"] table { font-size: var(--t-base); }
td.wrap { max-width: 32rem; }
td.glyph { font-variant-numeric: tabular-nums; }
/* a glyph cell reads its figure first, then the glyph with the reading after it, as a sentence reads them */
td.glyph .figure { display: block; }
td.glyph .reading { color: var(--ink-2); }
td.glyph .no-glyph { display: inline-block; width: 7.5em; max-width: 100%; min-width: 4.5em; margin-right: 6px; }
td:target, tr:target > td { background: var(--accent-wash); }
figure { margin: 0; }
.fig figcaption { font-family: var(--f-display); font-size: var(--t-base); font-weight: 600; margin-bottom: .5rem; }
.chart-values caption { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); }
.chart:not(:empty) + .chart-values { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px;
  overflow: hidden; clip-path: inset(50%); white-space: nowrap; border: 0; }
.finding { font-family: var(--f-display); font-size: var(--t-base); margin-top: .75rem; }
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
@media (max-width: 640px) { nav { position: static; } }
@media print {
  nav { display: none; }
  .tbl { overflow: visible; }
  th { position: static; }
  details.trial, .tbl, figure { break-inside: avoid; }
}
"""


#: Inline Markdown a report uses, applied to text already escaped — so its delimiters are the only markup it can make.
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_CODE = re.compile(r"`([^`]+)`")

#: A trial id in a table cell, linked to its drill-down.
TRIAL_ID = re.compile(r"\bT\d+\b")

#: The columns whose trial ids link to their records: the cases table's, the trials table's and a finding's facts.
_TRIAL_COLUMNS: frozenset[tuple[str, str]] = frozenset(
    {("cases", "trials"), ("trials", "trial"), ("finding_facts", "trials")}
)

#: The columns whose cells head their rows (``<th scope="row">``): a turned Every arm by measure's figure, and a turned
#: one-arm scorecard's measure, each naming what its row reads, so a header link (a goal check's,
#: :func:`check_header_links`) reaches it as a column's does.
_ROW_HEADER_COLUMNS: frozenset[tuple[str, str]] = frozenset({("arms_by_measure", "measure"), ("scorecard", "measure")})

#: A case as the cases table numbers it, whose number its row's anchor is made of; a case named by its author anchors
#: by its name (``case-double-charge``).
_CASE_NUMBER = re.compile(r"case (\d+)")

#: A cell of words at least this long is prose, laid out to wrap at a measure rather than widen its column.
_PROSE_CELL = 40


def page_head(title: str, *, static: bool) -> list[str]:
    """The page's doctype, its one :data:`_HTML_OPEN` tag and its ``<head>``: its title and the stylesheet — the kit's,
    then the page's own and the glyphs', and on a static page its drawn charts'. Nothing is loaded: the kit's faces are
    a local stack ending in a generic family. Nothing in it comes from a field but the title, escaped."""
    return [
        "<!doctype html>",
        _HTML_OPEN,
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{escape(one_line(title))}</title>",
        f"<style>{_KIT_CSS}{_PAGE_STYLE}{GLYPH_STYLE}{CHART_STYLE if static else ''}</style>",
        "</head>",
    ]


def table_html(block: TableBlock) -> str:
    """A table block: its title and rows, or a line saying it has none; folded when it is drill-down."""
    if not block.rows:
        grid = '<p class="order">(no rows)</p>'
    else:
        grid = grid_html(block)
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
    """A link to a glossary term, its sentence as the link's title so a hover reads it without leaving the place; a
    term the page defines itself (a goal check's name) carries :data:`CHECK_LINK_TITLE`, since its definition is a
    host's text, which no attribute carries."""
    style = f' class="{css}"' if css else ""
    if term not in GLOSSARY:
        return f'<a{style} href="#{check_anchor(term)}" title="{escape(CHECK_LINK_TITLE)}">{label}</a>'
    said = GLOSSARY[term]
    return f'<a{style} href="#g-{glossary_slug(term)}" title="{escape(said)}">{label}</a>'


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


def grid_html(block: TableBlock, *, highlighted: Collection[int] = ()) -> str:
    """A table of served cells' text, scrolling inside its own ``.tbl`` frame rather than widening the page: a table
    block's rows alone, which :func:`table_html` heads with the block's title and a page that names the table itself
    (the planned page's options table) lays out as it is. Its first column stays in view as it scrolls, and a table a
    row is told apart in by two columns (``TableBlock.key_columns``) is framed ``tbl keys-2``, which keeps both.

    A turned Every arm by measure heads each row with its figure (``<th scope="row">``, which reads as a body cell).
    A column of numbers is classed ``num`` on every cell, header included, so the stylesheet right-aligns it; a cell
    of prose is classed ``wrap`` (the kit's) so it wraps at a measure instead of widening its column. A column whose cells carry
    an interval is classed ``glyph``: each such cell states its figure, then draws its interval under it
    (:func:`~threetears.evals.analysis.report.svg.interval_glyph_svg`) with its reading (the cell's flag) after the
    glyph, every cell of the column in one unit on one scale, and the scale's ends are written once under the table
    (:func:`_scale_line`). A row at a position in
    ``highlighted`` is the kit's highlighted row (``tr.hl``): the one a page sets apart, as the recommendation.

    :param block: The table.
    :ptype block: TableBlock
    :param highlighted: The positions of the rows set apart; none by default.
    :ptype highlighted: Collection[int]
    :return: The table's markup.
    :rtype: str
    """
    scales = _glyph_scales(block)
    # A stored table holds no page's forms, so its glyphs' titles and the line under it ask forms of their own.
    forms = PageForms.off_page()
    classes = {
        column.key: ' class="glyph"'
        if any(key == column.key for key, _ in scales)
        else _column_class(block, column.key)
        for column in block.columns
    }
    pinned = {column.key for column in block.columns[: 2 if block.key_columns == 2 else 1]}
    head = "".join(
        f'<th scope="col"{_pinned_class(classes[column.key], column.header, column.key in pinned)}'
        f"{'' if column.key in pinned else _head_width(column.header)}>"
        f"{_pinned_text(_header(column.header, column.term), column.key in pinned)}</th>"
        for column in block.columns
    )
    body = []
    for position, row in enumerate(block.rows):
        cells = []
        for column in block.columns:
            cell = row[column.key]
            glyphed = classes[column.key] == ' class="glyph"'
            # A glyph cell sets its reading after its glyph, so its figure alone is the cell's text here.
            text = _cell_code(cell.figure if glyphed else cell.text)
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
            elif glyphed:
                css = classes[column.key]
                reading = "" if cell.flag is None else f' <span class="reading">{_cell_code(cell.flag)}</span>'
                text = f'<span class="figure">{text}</span>' + _glyph(cell, column.key, scales, forms) + reading
            else:
                css = classes[column.key] or (' class="wrap"' if len(cell.text) >= _PROSE_CELL else "")
            anchor = f' id="{cell.anchor}"' if cell.anchor is not None else ""
            pin = column.key in pinned
            tag, scope = ("th", ' scope="row"') if (block.name, column.key) in _ROW_HEADER_COLUMNS else ("td", "")
            # A cell holds a hyphenated word whole, as a pinned one does, so it wraps at its spaces alone; a prose cell
            # (wrap) wraps as a paragraph does.
            held = held_whole(text) if pin or css != ' class="wrap"' else text
            cells.append(f"<{tag}{scope}{anchor}{_pinned_class(css, cell.text, pin)}>{held}</{tag}>")
        anchor = ""
        if block.name == "cases" and "case" in row:
            # prose-canary: allow — the cases table's "case N" cell, which code numbered, not anything a model wrote
            number = _CASE_NUMBER.fullmatch(row["case"].text)
            anchor = f' id="case-{number.group(1) if number else glossary_slug(row["case"].text)}"'
        # A passed trial has no record of its own, so the link every table makes of its id lands on its row here.
        # prose-canary: allow — a trial id (T4), which code numbered, not anything a model wrote
        elif block.name == "passed_trials" and "trial" in row and TRIAL_ID.fullmatch(row["trial"].text):
            anchor = f' id="{row["trial"].text}"'
        if position in highlighted:
            anchor += ' class="hl"'
        body.append(f"<tr{anchor}>" + "".join(cells) + "</tr>")
    frame = "tbl keys-2" if block.key_columns == 2 else "tbl"
    grid = f'<div class="{frame}"{_frame_widths(block)}><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'
    return grid + _scale_line(block, scales, forms)


@dataclass(frozen=True)
class _Scale:
    """One glyph scale: the data's ends, which the line under the table states; the track's, the data's own but run a
    little past a declared margin so its ticks sit inside the track (the glyph's inset keeps a mark off the edge); the
    unit its spans are read in (a cell's own, else the one its column's header names); and a cell of the scale, which
    spells a number as the column's cells do (its money form), as the forms read it
    (:meth:`~threetears.evals.analysis.report.forms.PageForms.scale_span`,
    :meth:`~threetears.evals.analysis.report.forms.PageForms.glyph_said`)."""

    low: float
    high: float
    unit: str
    cell: TableCell
    track: tuple[float, float]

    def span(self, forms: PageForms) -> str:
        """The scale's ends, its unit once, as the line under the table states them."""
        return forms.scale_span(self.cell, self.unit, self.low, self.high).text


#: The unit a column's header names in its closing parentheses (``v2 (pts)``, ``Pass rate (%)``), which a cell
#: stated bare under it is read in.
_HEADER_UNIT = re.compile(r"\((%|pts)\)\s*$")


def _read_unit(cell: TableCell, header: str) -> str:
    """The unit a glyph cell's figures are read in: its own, else its column header's (:data:`_HEADER_UNIT`)."""
    if cell.unit:
        return cell.unit
    named = _HEADER_UNIT.search(header)
    return named.group(1) if named is not None else ""


#: How far past a declared margin a glyph's track runs, as a multiple of the margin.
_PAST_MARGIN = 1.15


def _glyph_scales(block: TableBlock) -> dict[tuple[str, str], _Scale]:
    """The scale each column's glyphs are drawn on, per unit: one for every cell of the column in that unit, or, in a
    table drawn on one scale (``TableBlock.shared_scale``: a matrix), whose columns are one quantity read for each arm,
    one for every cell of the table read in that unit, so a glyph compares across columns as well as down them. Money
    keeps a scale per column even there: two money measures are drawn on rulers of their own.

    The scale runs from the least to the greatest of every interval end, every point and every margin either side of
    zero of the group, and holds zero where any cell is a difference or carries a margin: the data's own ends, never
    padded, so the line under the table states what the track spans. A rate's scale stays within 0 to 100. A scale of
    levels alone (no cell a difference, none with a margin) never runs below zero where its measure cannot: money
    floors at 0, and a scale whose one figure is zero (every count read 0) is drawn from 0, never from -1. A level
    whose own interval reaches below zero is drawn there: its measure is not declared non-negative
    (``MetricDescriptor.nonnegative``), so nothing here may say it cannot be.

    :return: ``{(column key, cell unit): scale}``, for the columns with a cell carrying an interval.
    :rtype: dict[tuple[str, str], _Scale]
    """
    headers = {column.key: column.header for column in block.columns}
    ends: dict[tuple[str, str], list[float]] = {}
    margins: dict[tuple[str, str], float] = {}
    first: dict[tuple[str, str], TableCell] = {}
    differences: set[tuple[str, str]] = set()
    for row in block.rows:
        for key, cell in row.items():
            if cell.low is None or cell.high is None or not cell.is_number:
                continue
            held = ends.setdefault((key, cell.unit), [])
            first.setdefault((key, cell.unit), cell)
            held += [cell.low, cell.high, float(cell.value)]  # type: ignore[arg-type]
            if cell.margin is not None:
                held += [-cell.margin, cell.margin, 0.0]
                margins[key, cell.unit] = max(margins.get((key, cell.unit), 0.0), cell.margin)
            if cell.zero_line:
                held.append(0.0)
            if cell.zero_line or cell.margin is not None:
                differences.add((key, cell.unit))
    units = {group: _read_unit(cell, headers.get(group[0], "")) for group, cell in first.items()}

    def pooled(group: tuple[str, str]) -> tuple[str, str]:
        unit = units[group]
        return ("", unit) if block.shared_scale and not is_money(unit) else group

    by_scale: dict[tuple[str, str], list[float]] = {}
    margin_of: dict[tuple[str, str], float] = {}
    signed: set[tuple[str, str]] = set()
    for group, figures in ends.items():
        by_scale.setdefault(pooled(group), []).extend(figures)
        margin_of[pooled(group)] = max(margin_of.get(pooled(group), 0.0), margins.get(group, 0.0))
        if group in differences:
            signed.add(pooled(group))
    scales: dict[tuple[str, str], _Scale] = {}
    for group in ends:
        figures = by_scale[pooled(group)]
        low, high = min(figures), max(figures)
        levels = pooled(group) not in signed
        if units[group] == PERCENT:
            low, high = max(low, 0.0), min(high, 100.0)
        elif levels and is_money(units[group]):
            low = max(low, 0.0)
        if low == high:
            # One figure alone: a track either side of it, so its mark is drawn where it is; a level's never runs
            # below zero from a figure at or above it (every count read 0 is drawn from 0 to 1).
            pad = abs(low) * 0.05 or 1.0
            low, high = (max(low - pad, 0.0) if levels and low >= 0 else low - pad), high + pad
        # A declared margin's ends sit inside the track, so its ticks and its band never fill it edge to edge.
        margin = margin_of.get(pooled(group), 0.0) * _PAST_MARGIN
        track = (min(low, -margin), max(high, margin))
        scales[group] = _Scale(low=low, high=high, unit=units[group], cell=first[group], track=track)
    return scales


def _glyph(cell: TableCell, key: str, scales: dict[tuple[str, str], _Scale], forms: PageForms) -> str:
    """A cell's interval glyph on its column's scale, or a blank of the same width where the cell has none, so the
    column's figures stay aligned. Its title states its figures as the scale's cells write them; its whisker is drawn
    inside the track, where a scale that holds its measure's floor (a rate at 0 to 100, money at 0) ends it."""
    if cell.low is None or cell.high is None or not cell.is_number:
        return '<span class="no-glyph"></span>'
    scale = scales[(key, cell.unit)]
    value = float(cell.value)  # type: ignore[arg-type]
    said = forms.glyph_said(scale.cell, scale.unit, value, cell.low, cell.high, margin=cell.margin).text
    first, last = scale.track
    return interval_glyph_svg(
        float(cell.value),  # type: ignore[arg-type]
        min(max(cell.low, first), last),
        max(min(cell.high, last), first),
        unit=cell.unit,
        scale=scale.track,
        margin=cell.margin,
        said=said,
    )


def _scale_line(block: TableBlock, scales: dict[tuple[str, str], _Scale], forms: PageForms) -> str:
    """One quiet line under a table with glyphs: each scale's ends, written once, in the cells' own form and with the
    unit on every span, each range naming the columns drawn on it (``Pass rate (%) and Reply cites policy check (%)
    drawn from 47.2% to 100% · Turn time drawn from 973 to 1,029``); where one range is every glyph column's, it says
    so (``Each column drawn from -7.8 to 35.4 pts``). A table on one scale that says what its glyphs draw
    (``TableBlock.glyph_key``) opens its range of differences with that sentence, so a change's mark names the
    baseline its tick stands for and which way is better: ``Each mark is a candidate's change from v1 (the tick) with
    its 95% interval; right is better. Drawn from -7.8 to 35.4 pts.``"""
    if not scales:
        return ""
    headers = {column.key: column.header for column in block.columns}
    # Each range: the columns drawn on it (a table on one scale pools every column of a unit but money's), its spans,
    # and whether it is a range of differences.
    ranges: dict[tuple[str, str], tuple[list[str], list[str]]] = {}
    keyed: set[tuple[str, str]] = set()
    for (key, _), scale in scales.items():
        pooled = block.shared_scale and not is_money(scale.unit)
        group = ("", scale.unit) if pooled else (key, "")
        keys, spans = ranges.setdefault(group, ([], []))
        if key not in keys:
            keys.append(key)
        span = scale.span(forms)
        if span not in spans:
            spans.append(span)
        if pooled and block.glyph_key is not None and scale.cell.zero_line:
            keyed.add(group)
    if len(ranges) == 1 and not keyed:
        (((pool, _), (keys, spans)),) = ranges.items()
        named = "Each column" if not pool else code_html(headers[keys[0]])
        return f'<p class="axis">{named} drawn from {" and from ".join(spans)}</p>'
    lines = [
        f"{code_html(block.glyph_key or '')} Drawn from {' and from '.join(spans)}."
        if group in keyed
        else f"{_and([code_html(headers[key]) for key in keys])} drawn from {' and from '.join(spans)}"
        for group, (keys, spans) in ranges.items()
    ]
    return f'<p class="axis">{" · ".join(lines)}</p>'


def _and(names: Sequence[str]) -> str:
    """Names in a run: ``a``, ``a and b``, ``a, b and c``."""
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


#: The longest run a pinned cell holds without a place to wrap (a space, or after a hyphen or a slash) that its floor
#: still holds whole: a longer one (an expression, a digest, a snake_case key) is id-like.
_UNBROKEN_CHARS = 14

#: The places a pinned cell's words wrap at: a space, or after a hyphen or a slash (``deepseek-v4.1-flash``).
_WRAP_AT = re.compile(r"\s+|(?<=[-/])")


def _unbroken(text: str) -> bool:
    """Whether ``text`` holds a run longer than :data:`_UNBROKEN_CHARS` with no place to wrap: an id-like cell (an
    expression, a digest, a key), which a pinned column breaks anywhere rather than be held wider than its cap. Any
    other cell wraps at its spaces and after its hyphens, never inside a word.

    :param text: The cell's text, as served.
    :ptype text: str
    :return: Whether the cell is id-like.
    :rtype: bool
    """
    return any(len(run) > _UNBROKEN_CHARS for run in _WRAP_AT.split(text))


#: The longest hyphenated word (a model id, a case id) a pinned cell keeps on one line: the pinned column's cap holds
#: it whole. A longer one wraps after its hyphens, since no column has room for it.
_ONE_WORD_CHARS = 28

#: The longest one the pinned column's floor holds whole, about, so holding it whole never widens a table. A longer one
#: is classed ``long``: held whole where the table scrolls anyway, free to wrap after its hyphens where breaking it is
#: what lets the table fit, or where a phone's cap is too narrow for it.
_FLOOR_WORD_CHARS = 14

#: A word of a cell's text (a run between spaces) with a hyphen inside it, and a tag of the cell's markup.
_HYPHENATED = re.compile(r"[^\s<>]*\w-\w[^\s<>]*")
_TAG = re.compile(r"(<[^>]*>)")


def _pinned_text(markup: str, pinned: bool) -> str:
    """A cell's markup as its column lays it out: a pinned cell's hyphenated words held whole (:func:`held_whole`)."""
    return held_whole(markup) if pinned else markup


def held_whole(markup: str) -> str:
    """Markup with each hyphenated word of its text set ``<span class="id">``, which the stylesheet keeps on one line:
    ``deepseek-v4.1-flash`` is one word, never broken after a hyphen, in a pinned cell while its column can grow and in
    a written page's running text (its headline, lede and cards) alike.

    Only text is read: a tag (a link's ``href`` holds a case id too) and a code span (an expression, which breaks
    anywhere instead) are left as they are. A word longer than :data:`_ONE_WORD_CHARS` is left free to wrap after its
    hyphens, and one longer than :data:`_FLOOR_WORD_CHARS` is classed ``long``, which the stylesheet holds whole only
    in a frame that scrolls anyway.

    :param markup: The cell's markup, as built.
    :ptype markup: str
    :return: The same markup with its hyphenated words held whole.
    :rtype: str
    """
    out, code = [], 0
    for part in _TAG.split(markup):
        if part.startswith("<"):
            code += part.startswith("<code") - part.startswith("</code")
            out.append(part)
        else:
            out.append(part if code else _HYPHENATED.sub(_id_span, part))
    return "".join(out)


def _id_span(match: re.Match[str]) -> str:
    """One hyphenated word, held on one line where the cap can hold it."""
    word = match.group(0)
    chars = len(html.unescape(word))
    if chars > _ONE_WORD_CHARS:
        return word
    return f'<span class="id{" long" if chars > _FLOOR_WORD_CHARS else ""}">{word}</span>'


#: The widths a pinned column is laid out at, estimated from its text, in rem: its floor and its cap; the width a
#: character of a cell takes at the table's size (an upper estimate, wide enough for a wide fallback face, so a word
#: never outgrows it); the width one of a header takes (smaller, upper case and spaced); a cell's side padding; and an
#: arm chip's tag beside its name.
_KEY_REM = (8.0, 14.0)
_CELL_REM_PER_CHAR = 0.68 * 0.8125
_HEAD_REM_PER_CHAR = 0.75 * 0.75
_CELL_PAD_REM = 1.4
_CHIP_REM = 1.5


def _frame_widths(block: TableBlock) -> str:
    """The frame's style attribute: the widths its pinned columns are laid out at, or none.

    ``--key`` is a two-key table's first column, held at exactly that width (:func:`_key_rem`). ``--id1`` (a one-key
    table's pinned column) and ``--id2`` (a two-key table's second) are the width that holds the column's longest
    ``long`` hyphenated word whole (:func:`_id_rem`): a column with a width is given it before the columns past it
    grow, so the word breaks after a hyphen only where those columns at their narrowest leave no room for it.
    """
    widths = [f"--key:{_key_rem(block)}rem"] if block.key_columns == 2 else []
    # A two-key table's first column is held at --key, which already holds its words.
    for at, column in enumerate(block.columns[: block.key_columns], start=1):
        if at == 1 and block.key_columns == 2:
            continue
        rem = _id_rem(block, column.key)
        if rem is not None:
            widths.append(f"--id{at}:{rem}rem")
    if block.key_columns == 1 and block.columns and (lines := _lines_rem(block, block.columns[0].key)) is not None:
        widths.append(f"--lines1:{lines}rem")
    return f' style="{";".join(widths)}"' if widths else ""


#: The most lines a pinned label (a long aspect's name) takes where the column's cap leaves room for it.
_PINNED_LINES = 3


def _lines_rem(block: TableBlock, key: str) -> float | None:
    """The width, in rem, a one-key table's pinned column needs for its longest label to set in at most
    :data:`_PINNED_LINES` lines, its words wrapped at its spaces, up to the cap; ``None`` where the floor already
    sets every label in that many lines. The frame sets it as its pinned column's floor (``--lines1``), which the
    columns past it never crush, so a long label reads in three lines in a host's 688 px frame, not five.

    :param block: The table, keyed by one column.
    :ptype block: TableBlock
    :param key: The pinned column's key.
    :ptype key: str
    :return: The width, to two places, or ``None``.
    :rtype: float | None
    """
    low, high = _KEY_REM
    at_floor = int((low - _CELL_PAD_REM) / _CELL_REM_PER_CHAR)
    widest = 0.0
    for row in block.rows:
        words = row[key].text.split()
        if _set_lines(words, at_floor) <= _PINNED_LINES:
            continue
        chars = next(
            width
            for width in range(max(len(word) for word in words), len(" ".join(words)) + 1)
            if _set_lines(words, width) <= _PINNED_LINES
        )
        widest = max(widest, chars * _CELL_REM_PER_CHAR + (_CHIP_REM if row[key].tag is not None else 0.0))
    if not widest:
        return None
    return round(min(max(widest + _CELL_PAD_REM, low), high), 2)


def _set_lines(words: Sequence[str], width: int) -> int:
    """How many lines ``words`` set in at ``width`` characters a line, each line filled before the next; a word longer
    than a line takes one of its own."""
    lines, used = 0, 0
    for word in words:
        if lines and used + 1 + len(word) <= width:
            used += 1 + len(word)
        else:
            lines, used = lines + 1, len(word)
    return lines


#: The most lines a column's header takes in a table that scrolls anyway, and the narrowest a header is, in rem (the
#: stylesheet's ``th`` floor).
_HEAD_LINES = 2
_HEAD_FLOOR_REM = 4.5


def _head_width(header: str) -> str:
    """A header's style attribute where its words would set in more than :data:`_HEAD_LINES` lines at the header's
    floor: ``--head``, the width that sets them in that many, at their spaces. The stylesheet holds a header at it only
    in a frame that scrolls anyway, where widening its column costs a little more scroll and saves a header row as tall
    as its longest label's wrap (``Share of output spent reasoning`` in five lines at 688 px); in a table that fits,
    the header wraps as its column needs. Empty for a header the floor sets in that many lines already.

    :param header: The column's header, as declared.
    :ptype header: str
    :return: `` style="--head:<rem>rem"``, or empty.
    :rtype: str
    """
    words = header.split()
    at_floor = int((_HEAD_FLOOR_REM - _CELL_PAD_REM) / _HEAD_REM_PER_CHAR)
    if not words or _set_lines(words, at_floor) <= _HEAD_LINES:
        return ""
    chars = next(
        width
        for width in range(max(len(word) for word in words), len(" ".join(words)) + 1)
        if _set_lines(words, width) <= _HEAD_LINES
    )
    return f' style="--head:{round(chars * _HEAD_REM_PER_CHAR + _CELL_PAD_REM, 2)}rem"'


def _id_rem(block: TableBlock, key: str) -> float | None:
    """The width, in rem, that holds a pinned column's longest ``long`` hyphenated word on one line, its arm chip's tag
    beside it, between the floor and the cap; ``None`` for a column with no such word.

    :param block: The table.
    :ptype block: TableBlock
    :param key: The pinned column's key.
    :ptype key: str
    :return: The width, to two places, or ``None``.
    :rtype: float | None
    """
    widths = [
        len(word) * _CELL_REM_PER_CHAR + (_CHIP_REM if row[key].tag is not None else 0.0)
        for row in block.rows
        # prose-canary: allow — sizes a column to the words a cell sets; layout, never a judgement of what it says
        for word in _HYPHENATED.findall(row[key].text)
        if _FLOOR_WORD_CHARS < len(word) <= _ONE_WORD_CHARS
    ]
    if not widths:
        return None
    low, high = _KEY_REM
    return round(min(max(max(widths) + _CELL_PAD_REM, low), high), 2)


def _key_rem(block: TableBlock) -> float:
    """The width, in rem, of a two-key table's first column: its longest text, a cell's or the header's, as wide as it
    sets on one line, between the floor and the cap. The column is held at exactly this width so the second pins
    where the first ends; sized to its text, a short key ("All cases") leaves the frame to the columns past it instead
    of holding the cap.

    :param block: The table, keyed by two columns.
    :ptype block: TableBlock
    :return: The first column's width, in rem, to two places.
    :rtype: float
    """
    first = block.columns[0]
    cells = [
        len(row[first.key].text) * _CELL_REM_PER_CHAR + (_CHIP_REM if row[first.key].tag is not None else 0.0)
        for row in block.rows
    ]
    widest = max([len(first.header) * _HEAD_REM_PER_CHAR, *cells]) + _CELL_PAD_REM
    low, high = _KEY_REM
    return round(min(max(widest, low), high), 2)


def _pinned_class(css: str, text: str, pinned: bool) -> str:
    """A cell's class attribute (``css``, as built), with ``unbroken`` added where the cell is in a pinned column and
    id-like (:func:`_unbroken`): the one cell the stylesheet lets a pinned column break anywhere."""
    if not pinned or not _unbroken(text):
        return css
    if not css:
        return ' class="unbroken"'
    return css[:-1] + ' unbroken"'


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


#: The kit's chart kinds drawn as a grid of cells (its ``.mx``), whose columns can outnumber what a phone is wide.
_CELL_GRID_KINDS = frozenset({"heatmap", "diff_heatmap", "confusion"})


def chart_html(block: ChartBlock, chart_id: str, *, static: bool, figcaption: bool = True) -> str:
    """A chart as the kit's ``.fig``: its title, the chart the kit draws, and a table of every value it draws.

    The chart is a ``div.chart`` whose ``data-chart`` is its JSON, escaped into the attribute: the kit's script draws
    it at the chart's real width, every string in it as text. Every float in that JSON is served to twelve significant
    figures (:func:`~threetears.evals.schema.hashing.round_floats`), as the page's JSON form is: a point's position is
    a figure code computed, whose bits past that differ between Python builds, so a chart's data does not vary with
    them. The table follows it. While the kit has drawn the chart
    the table is visually hidden, read by a screen reader; where the chart is empty (no script ran, or the kit refused
    the data) or there is no chart to draw, the table is the figure, in its own scrolling frame.

    A static page writes no ``div.chart`` at all, so no script is wanted to draw one and no empty box stands where it
    would be: it draws the chart here, as inline SVG
    (:func:`~threetears.evals.analysis.report.chart_svg.chart_drawing`), in a ``div.chart-drawn``, and its table is
    visually hidden beside it as beside a drawn ``div.chart`` (``.chart-drawn + .chart-values``, in the drawings'
    stylesheet; owner ruling 2026-10-29), read by a screen reader. A chart not drawn there (a report's interval rows, a gate matrix, a lattice,
    example cards) says in one line that it is shown as its table, and that table is in sight.

    Without ``figcaption`` the figure carries no visible title, for a figure whose heading the page writes beside it
    (a section's headline); its table keeps the title as its caption, for a screen reader."""
    headers, rows = block.values_table()
    head = "".join(
        f'<th scope="col"{_pinned_class("", header, at == 0)}>{_pinned_text(escape(header), at == 0)}</th>'
        for at, header in enumerate(headers)
    )
    # A cell of words wraps at a measure, as a table block's does (:data:`_PROSE_CELL`), so a long label or reason never
    # widens the table past what its frame can scroll to; the pinned first column breaks an id-like cell anywhere. Any
    # other cell holds a hyphenated word whole, as a table block's does (``A · qwen3.8-flash``).
    body = "".join(
        "<tr>"
        + "".join(
            f"<td{_pinned_class(' class="wrap"' if len(cell) >= _PROSE_CELL else '', cell, at == 0)}>"
            f"{held_whole(escape(cell)) if at == 0 or len(cell) < _PROSE_CELL else escape(cell)}</td>"
            for at, cell in enumerate(row)
        )
        + "</tr>"
        for row in rows
    )
    drawn = None if static else block.kit_chart()
    chart = ""
    if static:
        chart = chart_drawing(block) or f'<p class="chart-note">{escape(AS_TABLE_NOTE)}</p>'
    elif drawn is not None:
        kind, data = drawn
        chart = (
            f'<div class="chart" id="{escape(chart_id)}" data-kind="{escape(kind)}" '
            f'data-chart="{escape(json.dumps(round_floats(data), separators=(",", ":"), allow_nan=False))}"></div>'
        )
    framed = block.chart in _CELL_GRID_KINDS
    values = (
        f'<div class="tbl chart-values"><table><caption>{held_whole(escape(block.title))}</caption>'
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{body}</tbody></table></div>"
    )
    # a cell grid wider than the phone scrolls inside a frame holding the chart and its table (on every page, drawn or
    # static, so the static page stays the default less its chart boxes), so the table still follows the chart
    drawing = f'<div class="tbl">{chart}{values}</div>' if framed else f"{chart}{values}"
    return (
        f'<figure class="fig" data-figure="{escape(block.chart)}">'
        + (f"<figcaption>{escape(block.title)}</figcaption>" if figcaption else "")
        + f"{drawing}</figure>"
    )


def glossary_slug(term: str) -> str:
    """A glossary term as an anchor id."""
    return re.sub(r"[^a-z0-9]+", "-", term.casefold()).strip("-")


def check_header_links(markup: str, names: Sequence[str]) -> str:
    """``markup`` with each header cell (a column's, or a row's: a turned table's figure) that names one of the goal
    checks ``names`` linked to that check's entry: a header is where a reader first meets a check's name, as a term of
    art's header links its term. A header naming a check reads its name, then its unit in brackets or the aspects it
    reads after a comma. A hyphenated word in it is held whole by a span (``span.id``), which the name is matched
    through and the link keeps. Then every header link to a glossary entry an earlier header already linked is marked
    ``seen`` (:func:`_seen_header_links`), so a term's dotted underline marks its first header alone.

    :param markup: The page's markup.
    :ptype markup: str
    :param names: The checks' names the page defines.
    :ptype names: Sequence[str]
    :return: The markup, each such header linked.
    :rtype: str
    """
    named = [(name, re.compile(rf"{re.escape(escape(name))}(?:,[^<(]*)?(?: \([^<]*\))?", re.S)) for name in names]

    def linked(found: re.Match[str]) -> str:
        said = _HELD.sub(r"\1", found[2])
        if "<" in said:
            # A header holding other markup (a term's link, code) is not a check's bare name.
            return found[0]
        for name, header in named:
            if header.fullmatch(said):
                return f"{found[1]}{term_link(name, found[2])}{found[3]}"
        return found[0]

    return _seen_header_links(_HEADER_CELL.sub(linked, markup) if named else markup)


def _seen_header_links(markup: str) -> str:
    """``markup`` with each header link to a glossary entry (``#g-…``) that an earlier header on the page linked
    classed ``seen``, which the page draws plain: a term's dotted underline marks where a reader first meets it in a
    header, not every table that repeats it."""
    first: set[str] = set()

    def marked(anchor: re.Match[str]) -> str:
        href = anchor["href"]
        if href not in first:
            first.add(href)
            return anchor[0]
        tag = anchor[0]
        return (
            tag.replace(' class="', ' class="seen ', 1)
            if ' class="' in tag
            else tag.replace("<a ", '<a class="seen" ', 1)
        )

    return _HEADER_CELL.sub(lambda cell: f"{cell[1]}{_GLOSSARY_LINK.sub(marked, cell[2])}{cell[3]}", markup)


#: A header cell, its opening tag, its content and its closing tag.
_HEADER_CELL = re.compile(r"(<th\b[^>]*>)(.*?)(</th>)", re.S)

#: A link to a glossary entry, its target the group ``href``.
_GLOSSARY_LINK = re.compile(r'<a\b[^>]*\bhref="(?P<href>#g-[^"]+)"[^>]*>')

#: A hyphenated word held whole in a cell (:func:`held_whole`), its text the group.
_HELD = re.compile(r'<span class="id(?: long)?">([^<]*)</span>')


#: The namespace of a goal check's glossary row's id (:func:`check_anchor`).
CHECK_ANCHOR_PREFIX = "g-check-"


def check_anchor(name: str) -> str:
    """A goal check's glossary entry's id: the checks' own namespace, so a declared name that is a term of art's words
    (``Outcome``) never repeats the term's id."""
    return f"{CHECK_ANCHOR_PREFIX}{glossary_slug(name)}"


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

    The tag's letter is the kit's on-fill colour or its surface, whichever reads better on the tag's colour in either
    theme: on-fill on an arm's colour, the surface on the control's ink-gray, which is light in the dark theme
    (``kit.js``'s ``kOnTag``).
    """
    token = color or _CHIP_TOKEN
    on_tag = ink_on(f"var({token})", ("--on-fill", "--surface"))
    return (
        f'<span class="armchip"><span class="tag" style="background:var({token});color:var({on_tag})">'
        f"{escape(tag)}</span>{markup}</span>"
    )


__all__ = [
    "CHECK_ANCHOR_PREFIX",
    "KIT_JS",
    "TRIAL_ID",
    "chart_html",
    "code_html",
    "escape",
    "check_anchor",
    "check_header_links",
    "glossary_slug",
    "grid_html",
    "held_whole",
    "inline_html",
    "one_line",
    "page_head",
    "pill_html",
    "table_html",
    "term_link",
]
