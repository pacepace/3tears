"""The report's marks as inline SVG, drawn with no dependency: the outcome bar, the interval glyph a table cell
draws beside its number, and the interval rows.

**What earns a picture.** The *outcome bar* is a census, so it is always drawn: one horizontal stacked bar per arm,
its label on the line above it, never cut — passed, failed, and lost to the test setup cause by cause — every
segment's count written as text beside the bar, and a key naming only the states that occur. An *interval glyph* is
the uncertainty of one figure drawn in the row that states it (Tufte's table-graphic): a track, a zero line where
zero is on it, the declared margin as a quiet band, the interval as a whisker and the point as a dot, every row of a
column in one unit on the same scale, so the rows compare as small multiples and the scale's ends are written once
under the table. *Interval rows* are the same marks as a standalone chart (a dot and its interval per row, the
value, the interval and the cases written as words under the marks), which the builder no longer draws, since the
tables carry the intervals; this module draws what it is handed.

**Legible without colour, a script or a stylesheet of its own.** Every mark's value is written as text, and each
segment says what it is in words, so colour never carries identity alone. Colours are classes the page's stylesheet
sets from CSS variables (:data:`SVG_STYLE`), stepped for light and dark; a glyph is drawn in the page's ink alone,
the margin band in its one quiet tint, and its verdict is the word in the next cell, never a hue. Each chart has a
``<title>`` and a ``<desc>`` stating its values, each mark and each glyph a ``<title>`` of its own, and the page keeps
a visually hidden table of a chart's values beside it. There is no ``<script>``, no event attribute, no external
reference and no URL of any kind — not even a ``url(#...)`` fill — so the page's ``default-src 'none'`` policy holds
unchanged.
"""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence

from threetears.evals.analysis.report.model import ChartBlock, IntervalRow, OutcomeBar, interval_text
from threetears.evals.analysis.quantities import PERCENT, is_money, number_in, with_unit
from threetears.evals.analysis.style import GLOSSARY

#: The chart's own coordinate width; the SVG scales to the page's width from it. A line of words that would run past
#: it is wrapped onto more lines (:func:`_wrapped`), its row drawn taller, never cut.
_WIDTH = 720

#: Where an interval chart's plot ends and the value text begins. The label column before the plot is as wide as the
#: longest label needs (:func:`_label_column`), between these two bounds.
_PLOT_END = 470
_LABEL_MIN = 120
_LABEL_MAX = 300

#: Where an outcome bar starts and ends: its label is on the line above it, so the bar has the chart's width less
#: the room its total takes at the right.
_BAR_START = 8
_BAR_END = 600

#: The width a character of a label is laid out at: a generous mean for 13px system sans, so a label fits its column.
_CHAR_WIDTH = 7.2

#: The narrowest segment its count is written inside; a narrower one's count is in the line under the bar.
_COUNT_INSIDE = 22

#: One outcome bar's row: its label, the bar, then a line of counts under it.
_BAR_ROW = 63
_BAR_HEIGHT = 16

#: The interval glyph's own coordinates: its width, its height, the track's inset from either edge, and the dot.
_GLYPH_WIDTH = 120
_GLYPH_HEIGHT = 14
_GLYPH_INSET = 4
_GLYPH_DOT = 2.5

#: One interval row: the marks and the value, then a line of words under them.
_INTERVAL_ROW = 44
_INTERVAL_MARK = 11

#: Each further line a row's words wrap onto, in either chart.
_LINE = 17

#: The gap between two stacked segments, so each reads as its own mark.
_SEGMENT_GAP = 2

#: The page's chart stylesheet: each mark's class, from CSS variables stepped for light and for dark. The categorical
#: steps are the validated default palette's first two slots (blue, orange); a lost segment is a neutral grey with a
#: dashed outline, the convention for "not measured", so it never reads as a third series. A failure's causes share
#: the one failed hue and differ in ink alone (:data:`_FAILURE_INK`): tone, edge weight and hatching. The chart scales
#: to the
#: page's width down to a floor that keeps its 13px text at 11px or more on a phone; past the floor its figure
#: scrolls, as a table does, rather than shrinking the words.
SVG_STYLE = """
:root { --viz-passed: #2a78d6; --viz-failed: #eb6834; --viz-lost: #b9b8b3; --viz-lost-edge: #6f6e69;
  --viz-ink: #0b0b0b; --viz-muted: #52514e; --viz-rule: #c9c8c2; --viz-band: #e9e8e3; --viz-on-fill: #ffffff; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { --viz-passed: #3987e5; --viz-failed: #d95926; --viz-lost: #4a4a46;
    --viz-lost-edge: #a8a7a0; --viz-ink: #f0efec; --viz-muted: #c3c2b7; --viz-rule: #4a4a46; --viz-band: #2c2c29;
    --viz-on-fill: #ffffff; }
}
:root[data-theme="dark"] { --viz-passed: #3987e5; --viz-failed: #d95926; --viz-lost: #4a4a46;
  --viz-lost-edge: #a8a7a0; --viz-ink: #f0efec; --viz-muted: #c3c2b7; --viz-rule: #4a4a46; --viz-band: #2c2c29;
  --viz-on-fill: #ffffff; }
svg.viz { width: 100%; min-width: 610px; height: auto; display: block;
  font: 13px/1.3 var(--font-sans, system-ui, sans-serif); }
svg.viz text { fill: var(--viz-ink); }
svg.viz text.viz-muted { fill: var(--viz-muted); }
svg.viz .viz-passed { fill: var(--viz-passed); }
svg.viz .viz-failed { fill: var(--viz-failed); }
svg.viz .viz-failed.viz-alt { fill-opacity: 0.45; stroke: var(--viz-failed); stroke-width: 1.5; }
svg.viz .viz-failed.viz-alt2 { fill-opacity: 0.15; stroke: var(--viz-failed); stroke-width: 2.5; }
svg.viz .viz-failed.viz-alt3 { fill-opacity: 0.15; stroke: var(--viz-failed); stroke-width: 1; }
svg.viz .viz-failed.viz-alt4 { fill-opacity: 0.15; stroke: var(--viz-failed); stroke-width: 2; }
svg.viz .viz-hatch { stroke: var(--viz-failed); stroke-width: 1; fill: none; pointer-events: none; }
svg.viz .viz-lost { fill: var(--viz-lost); stroke: var(--viz-lost-edge); stroke-width: 1; stroke-dasharray: 3 2; }
svg.viz .viz-lost.viz-alt { stroke-dasharray: 1 2; }
svg.viz text.viz-on-fill { fill: var(--viz-on-fill); font-variant-numeric: tabular-nums; }
svg.viz .viz-rule { stroke: var(--viz-rule); stroke-width: 1; }
svg.viz .viz-zero { stroke: var(--viz-muted); stroke-width: 1; }
svg.viz .viz-band { fill: var(--viz-band); }
svg.viz .viz-whisker { stroke: var(--viz-ink); stroke-width: 2; stroke-linecap: round; }
svg.viz .viz-dot { fill: var(--viz-passed); stroke: var(--viz-ink); stroke-width: 1; }
svg.viz-glyph { width: 7.5em; height: 0.875em; vertical-align: -0.0625em; margin-right: 6px; overflow: visible; }
svg.viz-glyph .viz-track { stroke: var(--viz-rule); stroke-width: 1; }
svg.viz-glyph .viz-zero { stroke: var(--viz-muted); stroke-width: 1; }
svg.viz-glyph .viz-band { fill: var(--viz-band); }
svg.viz-glyph .viz-whisker { stroke: var(--viz-ink); stroke-width: 1.5; stroke-linecap: round; }
svg.viz-glyph .viz-point { fill: var(--viz-ink); }
.visually-hidden { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden;
  clip-path: inset(50%); white-space: nowrap; border: 0; }
"""


def chart_svg(block: ChartBlock, *, chart_id: str) -> str:
    """One chart block as inline SVG.

    Args:
        block: The chart.
        chart_id: An id unique on the page, which the chart's title and description are named by.

    Returns:
        The ``<svg>`` element.
    """
    if block.chart == "outcome_bar":
        return outcome_bar_svg(block.title, block.bars, chart_id=chart_id)
    return interval_rows_svg(
        block.title, block.rows, unit=block.unit, zero_line=block.zero_line, margin=block.margin, chart_id=chart_id
    )


def segment_words(bar: OutcomeBar) -> str:
    """An arm's census as the text under its bar: every segment's count, in words."""
    return " · ".join(f"{segment.label} {segment.count}" for segment in bar.segments)


def outcome_bar_svg(title: str, bars: Sequence[OutcomeBar], *, chart_id: str) -> str:
    """The outcome bar: one stacked bar per arm under its label, its segments scaled to the longest arm, every count
    written.

    An arm's label is the line above its bar, wrapped where it is long and never cut, so the bar has the chart's
    width. Each segment's count is written inside it where it fits, and every arm's whole census is written under
    its bar, so no count is read off a length or a colour. A lost segment is grey with a dashed edge, the next lost
    cause dotted, so two causes side by side read as two. A failure's causes are told apart the same way, in ink
    alone (:func:`_failure_styles`), one style per cause on every bar, so the key names each cause that occurs
    (``failed: check``, ``failed: judge``). A census too long for one line wraps, and its row is taller. The key
    names only the states that occur.

    Args:
        title: What the chart shows.
        bars: One bar per arm, in the scoreboard's order.
        chart_id: The chart's id on the page.

    Returns:
        The ``<svg>`` element.
    """
    longest = max((bar.total for bar in bars), default=0) or 1
    span = _BAR_END - _BAR_START
    labels = [_wrapped(bar.arm, _BAR_START) for bar in bars]
    censuses = [_wrapped(segment_words(bar), _BAR_START) for bar in bars]
    rows = [
        _BAR_ROW + _LINE * (len(label) - 1) + _LINE * (len(lines) - 1)
        for label, lines in zip(labels, censuses, strict=True)
    ]
    states = list(dict.fromkeys(segment.state for bar in bars for segment in bar.segments))
    failures = _failure_styles(bars)
    legend, legend_height = _legend(_BAR_START, states, failures)
    height = legend_height + sum(rows) + 6
    marks = [legend]
    top = legend_height
    for bar, label, lines, row_height in zip(bars, labels, censuses, rows, strict=True):
        for offset, line in enumerate(label):
            marks.append(_text(_BAR_START, top + 13 + _LINE * offset, line))
        bar_top = top + 19 + _LINE * (len(label) - 1)
        x = float(_BAR_START)
        lost = 0
        for segment in bar.segments:
            width = max(span * segment.count / longest - _SEGMENT_GAP, 1.0)
            mark = f"{bar.arm}: {segment.label}, {segment.count} of the {bar.total} trials it ran"
            alt = " viz-alt" if segment.state == "lost" and lost % 2 else ""
            hatch = ""
            if segment.state == "failed":
                alt, hatch = failures[segment.label]
            lost += segment.state == "lost"
            marks.append(
                f'<rect class="viz-{segment.state}{alt}" x="{x:.1f}" y="{bar_top}" width="{width:.1f}" '
                f'height="{_BAR_HEIGHT}" rx="2"><title>{_escape(mark)}</title></rect>'
            )
            marks.append(_hatching(hatch, x, bar_top, width, _BAR_HEIGHT))
            if width >= _COUNT_INSIDE:
                # A count on a light tone is in ink, as on a lost segment; on a solid fill, in the fill's contrast.
                inside = ' class="viz-on-fill"' if segment.state != "lost" and not alt else ""
                marks.append(
                    f'<text x="{x + width / 2:.1f}" y="{bar_top + _BAR_HEIGHT - 4}"{inside} text-anchor="middle">'
                    f"{segment.count}</text>"
                )
            x += width + _SEGMENT_GAP
        marks.append(_text(x + 6, bar_top + _BAR_HEIGHT - 3, f"{bar.total} trials"))
        for offset, line in enumerate(lines):
            marks.append(_text(_BAR_START, bar_top + _BAR_HEIGHT + 17 + _LINE * offset, line, muted=True))
        top += row_height
    desc = "; ".join(f"{bar.arm} ran {bar.total} trials: {segment_words(bar)}" for bar in bars)
    return _svg(chart_id, title, desc, height, marks)


def interval_glyph_svg(
    value: float,
    low: float,
    high: float,
    *,
    unit: str,
    scale: tuple[float, float],
    margin: float | None = None,
) -> str:
    """One figure's interval as the glyph a table cell draws beside it: a track from the scale's low end to its
    high, a zero line where zero is on it, the margin as a quiet band either side of zero, the interval as a whisker
    and the figure as a dot.

    The scale is the caller's, shared by every cell of the column in the same unit, so one row's glyph reads against
    the next; the scale's ends are written once under the table, not here. The glyph's ``<title>`` states its
    numbers, so a hover reads them where the row's cells already print them.

    Args:
        value: The figure, in ``unit``.
        low: The interval's low end, in ``unit``.
        high: The interval's high end, in ``unit``.
        unit: The unit the figures are in; empty for none.
        scale: The track's low and high ends, which hold the interval, the margin and zero where it is drawn.
        margin: A declared margin either side of zero, or None.

    Returns:
        The ``<svg>`` element.
    """
    first, last = scale
    span = (last - first) or 1.0

    def x_of(figure: float) -> float:
        return _GLYPH_INSET + (figure - first) / span * (_GLYPH_WIDTH - 2 * _GLYPH_INSET)

    middle = _GLYPH_HEIGHT / 2
    said = f"{with_unit(value, unit)}, interval {_interval_words(low, high, unit)}"
    if margin is not None:
        said += f", margin ±{with_unit(margin, unit)}"
    marks = [
        f'<line class="viz-track" x1="{x_of(first):.1f}" y1="{middle}" x2="{x_of(last):.1f}" y2="{middle}"></line>'
    ]
    if margin is not None:
        band_low, band_high = x_of(max(-margin, first)), x_of(min(margin, last))
        marks.append(
            f'<rect class="viz-band" x="{band_low:.1f}" y="1" width="{max(band_high - band_low, 0.0):.1f}" '
            f'height="{_GLYPH_HEIGHT - 2}"></rect>'
        )
    if first <= 0.0 <= last:
        marks.append(
            f'<line class="viz-zero" x1="{x_of(0.0):.1f}" y1="0" x2="{x_of(0.0):.1f}" y2="{_GLYPH_HEIGHT}"></line>'
        )
    marks.append(
        f'<line class="viz-whisker" x1="{x_of(low):.1f}" y1="{middle}" x2="{x_of(high):.1f}" y2="{middle}"></line>'
    )
    marks.append(f'<circle class="viz-point" cx="{x_of(value):.1f}" cy="{middle}" r="{_GLYPH_DOT}"></circle>')
    return (
        f'<svg class="viz-glyph" viewBox="0 0 {_GLYPH_WIDTH} {_GLYPH_HEIGHT}" role="img">'
        f"<title>{_escape(said)}</title>" + "".join(marks) + "</svg>"
    )


def _label_column(labels: Sequence[str]) -> tuple[int, list[str]]:
    """Where the plot starts after a column wide enough for the labels, and each label cut to fit it.

    Args:
        labels: The rows' labels.

    Returns:
        ``(x where the plot starts, labels as drawn)``; a label too long for the widest column is cut with an
        ellipsis, its whole text kept in its mark's ``<title>``.
    """
    widest = max((len(label) for label in labels), default=0)
    start = int(min(max(widest * _CHAR_WIDTH + 24, _LABEL_MIN), _LABEL_MAX))
    room = int((start - 24) / _CHAR_WIDTH)
    return start, [label if len(label) <= room else label[: room - 1] + "…" for label in labels]


def _failure_styles(bars: Sequence[OutcomeBar]) -> dict[str, tuple[str, str]]:
    """Each failure cause's ink on the chart, by its segment's label: one style per cause on every bar, in the order
    the causes first occur, so a cause reads alike on each arm and the key names it once."""
    labels = dict.fromkeys(segment.label for bar in bars for segment in bar.segments if segment.state == "failed")
    return {label: _FAILURE_INK[index % len(_FAILURE_INK)] for index, label in enumerate(labels)}


#: A failure cause's ink, as ``(class, hatching)``, in the order causes take it: solid; a lighter tone edged in the hue;
#: lighter still with a heavier edge; that tone with a hairline edge, hatched one way; and with a mid edge,
#: cross-hatched. One per failure cause (``FailCause``), one hue throughout: a failure is one state, its causes told
#: apart without a new colour.
_FAILURE_INK: tuple[tuple[str, str], ...] = (
    ("", ""),
    (" viz-alt", ""),
    (" viz-alt2", ""),
    (" viz-alt3", "/"),
    (" viz-alt4", "x"),
)

#: The gap between two hatch lines, along either edge.
_HATCH_STEP = 4


def _hatching(hatch: str, x: float, y: float, width: float, height: float) -> str:
    """The hatch lines over one mark, drawn inside its box: ``/`` one way, ``x`` both ways, ``""`` none.

    Drawn as lines rather than a pattern fill, so the chart references nothing by URL.
    """
    if not hatch:
        return ""
    strokes: list[str] = []
    for falling in (False, True) if hatch == "x" else (False,):
        for index in range(1, int((width + height) / _HATCH_STEP) + 1):
            # Across the box by u and down it by s: a ``/`` line holds u + s = c, a ``\`` line u - s = c.
            c = index * _HATCH_STEP - (height if falling else 0)
            first, last = (max(0.0, c), min(width, c + height)) if falling else (max(0.0, c - height), min(width, c))
            if last - first < 1:
                continue
            ends = [(u, u - c if falling else c - u) for u in (first, last)]
            strokes.append("M" + " L".join(f"{x + u:.1f} {y + down:.1f}" for u, down in ends))
    return f'<path class="viz-hatch" d="{" ".join(strokes)}"></path>'


def _legend(start: int, states: Sequence[str], failures: Mapping[str, tuple[str, str]]) -> tuple[str, int]:
    """The outcome bar's key and its height: each state that occurs, its swatch beside its name, so the colours are
    read off words; a failure by each cause that occurs (``failed: check``, ``failed: judge``), in its own ink.

    A lost trial's name is the glossary's term, with its definition as the swatch's title, so a hover says what the
    test setup spoiling a trial means without a line of its own. A key too long for one line wraps onto the next.
    """
    entries: list[tuple[str, str, str, str]] = []
    for state in ("passed", "failed", "lost"):
        if state not in states:
            continue
        if state == "failed":
            entries += [("failed" + ink, label, "", hatch) for label, (ink, hatch) in failures.items()]
            continue
        entries.append((state, state, f"<title>{_escape(GLOSSARY['lost'])}</title>" if state == "lost" else "", ""))
    parts = []
    x, line = start, 0
    for classes, name, title, hatch in entries:
        width = 17 + int(_CHAR_WIDTH * len(name)) + 18
        if x > start and x + width - 18 > _BAR_END:
            x, line = start, line + 1
        y = 26 + _LINE * line
        parts.append(f'<rect class="viz-{classes}" x="{x}" y="{y - 20}" width="12" height="12" rx="2">{title}</rect>')
        parts.append(_hatching(hatch, x, y - 20, 12, 12))
        parts.append(_text(x + 17, y - 10, name, muted=True))
        x += width
    return "".join(parts), 26 + _LINE * line


def interval_rows_svg(
    title: str,
    rows: Sequence[IntervalRow],
    *,
    unit: str,
    zero_line: bool,
    margin: float | None,
    chart_id: str,
) -> str:
    """Interval rows: a dot and its interval per row, its value at the right, and under the marks its value, interval,
    cases and verdict in words.

    The words go under the marks, not beside them, wrapped where they are long, so a verdict never runs past the
    chart's edge; the value alone sits at the right of its row, where a reader scanning the dots finds it.

    Args:
        title: What the chart shows.
        rows: The rows, in drawn order.
        unit: The unit the values are in; empty for none.
        zero_line: Whether to draw a line at zero.
        margin: A margin band either side of zero, or None.
        chart_id: The chart's id on the page.

    Returns:
        The ``<svg>`` element.
    """
    ends = [value for row in rows for value in (row.low, row.high, row.value)]
    if zero_line:
        ends.append(0.0)
    if margin is not None:
        ends += [-margin, margin]
    low, high = min(ends), max(ends)
    pad = (high - low) * 0.05 or abs(high) * 0.05 or 1.0
    low, high = low - pad, high + pad
    start, labels = _label_column([row.label for row in rows])

    def x_of(value: float) -> float:
        return start + (value - low) / (high - low) * (_PLOT_END - start)

    axis = 18
    sayings = [_wrapped(row_words(row, unit), start) for row in rows]
    heights = [_INTERVAL_ROW + _LINE * (len(lines) - 1) for lines in sayings]
    plot_bottom = sum(heights)
    height = plot_bottom + axis + 8
    marks = []
    if margin is not None:
        left, right = x_of(-margin), x_of(margin)
        marks.append(
            f'<rect class="viz-band" x="{left:.1f}" y="0" width="{right - left:.1f}" height="{plot_bottom}">'
            f"<title>{_escape(f'margin ±{with_unit(margin, unit)}')}</title></rect>"
        )
    if zero_line:
        zero = x_of(0.0)
        marks.append(f'<line class="viz-zero" x1="{zero:.1f}" y1="0" x2="{zero:.1f}" y2="{plot_bottom}"></line>')
    top = 0
    for row, label, lines, row_height in zip(rows, labels, sayings, heights, strict=True):
        middle = top + _INTERVAL_MARK
        said = row_words(row, unit)
        marks.append(_text(8, middle + 4, label, title=row.label if label != row.label else None))
        marks.append(
            f'<line class="viz-whisker" x1="{x_of(row.low):.1f}" y1="{middle:.1f}" x2="{x_of(row.high):.1f}" '
            f'y2="{middle:.1f}"><title>{_escape(f"{row.label}: {said}")}</title></line>'
        )
        marks.append(
            f'<circle class="viz-dot" cx="{x_of(row.value):.1f}" cy="{middle:.1f}" r="4.5">'
            f"<title>{_escape(f'{row.label}: {said}')}</title></circle>"
        )
        marks.append(_text(_PLOT_END + 10, middle + 4, with_unit(row.value, unit)))
        for offset, line in enumerate(lines):
            marks.append(_text(start, middle + 23 + _LINE * offset, line, muted=True))
        top += row_height
    marks.append(f'<line class="viz-rule" x1="{start}" y1="{plot_bottom}" x2="{_PLOT_END}" y2="{plot_bottom}"></line>')
    marks.append(_text(start, plot_bottom + 14, number_in(low, unit), muted=True))
    marks.append(_text(_PLOT_END, plot_bottom + 14, number_in(high, unit), muted=True, anchor="end"))
    desc = "; ".join(f"{row.label}: {row_words(row, unit)}" for row in rows)
    return _svg(chart_id, title, desc, height, marks)


def _interval_words(low: float, high: float, unit: str) -> str:
    """An interval as a glyph's title says it, as the row's own interval cell spells it: the unit on each end where
    it sits on the number (``[88.1%, 100%]``, ``[-$0.01, $0.06]``), once after the bracket otherwise
    (``[-4.2, 2] pts``)."""
    if unit == PERCENT or is_money(unit):
        return f"[{with_unit(low, unit)}, {with_unit(high, unit)}]"
    return interval_text(low, high, unit) + (f" {unit}" if unit else "")


def row_words(row: IntervalRow, unit: str) -> str:
    """One interval row in words: its value and unit, its interval, its cases, and its verdict where it has one.

    The value is joined to its unit as every surface joins one (``98.9%``, ``-1.1 pts``, ``$0.025``, ``0.5 s``), and
    the interval's ends are spelled at the same precision, the unit said once on the value.
    """
    verdict = f", {row.verdict}" if row.verdict else ""
    return (
        f"{with_unit(row.value, unit)} {interval_text(row.low, row.high, unit)}, "
        f"n={row.n_cases} case{'' if row.n_cases == 1 else 's'}{verdict}"
    )


def _wrapped(words: str, start: int) -> list[str]:
    """``words`` as the lines they fit on from ``start`` to the chart's edge, broken at spaces.

    Args:
        words: A row's line of words.
        start: Where the line begins; it may run to the chart's edge less a small margin.

    Returns:
        At least one line; a single word longer than the room is a line of its own.
    """
    room = int((_WIDTH - start - 8) / _CHAR_WIDTH)
    lines: list[str] = []
    line = ""
    for word in words.split(" "):
        if line and len(line) + 1 + len(word) > room:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}" if line else word
    return [*lines, line]


def _svg(chart_id: str, title: str, desc: str, height: int, marks: Sequence[str]) -> str:
    """The chart's ``<svg>``: its viewBox, its title and description, and its marks."""
    return (
        f'<svg class="viz" viewBox="0 0 {_WIDTH} {height}" role="img" '
        f'aria-labelledby="{chart_id}-title {chart_id}-desc">'
        f'<title id="{chart_id}-title">{_escape(title)}</title>'
        f'<desc id="{chart_id}-desc">{_escape(desc)}</desc>' + "".join(marks) + "</svg>"
    )


def _text(
    x: float, y: float, words: str, *, muted: bool = False, anchor: str = "start", title: str | None = None
) -> str:
    """A text mark, its words escaped, in the page's ink; ``title`` the whole of a label cut to fit."""
    css = ' class="viz-muted"' if muted else ""
    end = ' text-anchor="end"' if anchor == "end" else ""
    whole = f"<title>{_escape(title)}</title>" if title else ""
    return f'<text x="{x:.1f}" y="{y:.1f}"{css}{end}>{whole}{_escape(words)}</text>'


def _escape(text: str) -> str:
    """Escape text for an SVG element or attribute."""
    return html.escape(text, quote=True)


__all__ = [
    "SVG_STYLE",
    "chart_svg",
    "interval_glyph_svg",
    "interval_rows_svg",
    "outcome_bar_svg",
    "row_words",
    "segment_words",
]
