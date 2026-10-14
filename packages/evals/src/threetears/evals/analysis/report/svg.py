"""The report's two charts as inline SVG, drawn with no dependency: the outcome bar and the interval rows.

**What earns a chart.** Two shapes and no more. The *outcome bar* is a census, so it is always drawn: one horizontal
stacked bar per arm — passed, failed, and lost to the test setup cause by cause — every segment's count written as
text beside the bar. *Interval rows* are estimates: a dot and its interval per row, a zero line where the rows are
differences, a margin band where one was declared, and the value, the interval and the cases written as text at the
right. Which rows are drawn at all is the report builder's rule (an arm needs
:data:`~threetears.evals.analysis.report.verdict.MIN_CASES` usable cases and an interval); this module draws what it
is handed.

**Legible without colour, a script or a stylesheet of its own.** Every mark's value is written as text, and each
segment says what it is in words, so colour never carries identity alone. Colours are classes the page's stylesheet
sets from CSS variables (:data:`SVG_STYLE`), stepped for light and dark, and text is drawn in ``currentColor``-based
ink, never in a series colour. Each chart has a ``<title>`` and a ``<desc>`` stating its values, each mark a
``<title>`` of its own, and the page keeps a visually hidden table of the same values beside it. There is no
``<script>``, no event attribute, no external reference and no URL of any kind — not even a ``url(#...)`` fill — so
the page's ``default-src 'none'`` policy holds unchanged.
"""

from __future__ import annotations

import html
from collections.abc import Sequence

from threetears.evals.analysis.report.model import ChartBlock, IntervalRow, OutcomeBar, interval_text
from threetears.evals.analysis.quantities import render_cell

#: The chart's own coordinate width; the SVG scales to the page's width from it.
_WIDTH = 720

#: Where the plot ends and the value text begins. The label column before the plot is as wide as the longest label
#: needs (:func:`_label_column`), between these two bounds.
_PLOT_END = 470
_LABEL_MIN = 120
_LABEL_MAX = 300

#: The width a character of a label is laid out at: a generous mean for 13px system sans, so a label fits its column.
_CHAR_WIDTH = 7.2

#: The narrowest segment its count is written inside; a narrower one's count is in the line under the bar.
_COUNT_INSIDE = 22

#: One outcome bar's row: the bar, then a line of counts under it.
_BAR_ROW = 46
_BAR_HEIGHT = 16

#: One interval row.
_INTERVAL_ROW = 30

#: The gap between two stacked segments, so each reads as its own mark.
_SEGMENT_GAP = 2

#: The page's chart stylesheet: each mark's class, from CSS variables stepped for light and for dark. The categorical
#: steps are the validated default palette's first two slots (blue, orange); a lost segment is a neutral grey with a
#: dashed outline, the convention for "not measured", so it never reads as a third series.
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
svg.viz { width: 100%; height: auto; display: block; font: 13px/1.3 var(--font-sans, system-ui, sans-serif); }
svg.viz text { fill: var(--viz-ink); }
svg.viz text.viz-muted { fill: var(--viz-muted); }
svg.viz .viz-passed { fill: var(--viz-passed); }
svg.viz .viz-failed { fill: var(--viz-failed); }
svg.viz .viz-lost { fill: var(--viz-lost); stroke: var(--viz-lost-edge); stroke-width: 1; stroke-dasharray: 3 2; }
svg.viz .viz-lost.viz-alt { stroke-dasharray: 1 2; }
svg.viz text.viz-on-fill { fill: var(--viz-on-fill); font-variant-numeric: tabular-nums; }
svg.viz .viz-rule { stroke: var(--viz-rule); stroke-width: 1; }
svg.viz .viz-zero { stroke: var(--viz-muted); stroke-width: 1; }
svg.viz .viz-band { fill: var(--viz-band); }
svg.viz .viz-whisker { stroke: var(--viz-ink); stroke-width: 2; stroke-linecap: round; }
svg.viz .viz-dot { fill: var(--viz-passed); stroke: var(--viz-ink); stroke-width: 1; }
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
    """The outcome bar: one stacked bar per arm, its segments scaled to the longest arm, every count written.

    Each segment's count is written inside it where it fits, and every arm's whole census is written under its bar,
    so no count is read off a length or a colour. A lost segment is grey with a dashed edge, the next lost cause
    dotted, so two causes side by side read as two.

    Args:
        title: What the chart shows.
        bars: One bar per arm, in the scoreboard's order.
        chart_id: The chart's id on the page.

    Returns:
        The ``<svg>`` element.
    """
    legend_height = 26
    height = legend_height + _BAR_ROW * len(bars) + 6
    longest = max((bar.total for bar in bars), default=0) or 1
    start, labels = _label_column([bar.arm for bar in bars])
    span = _PLOT_END - start
    marks = [_legend(start, legend_height)]
    for index, (bar, label) in enumerate(zip(bars, labels, strict=True)):
        top = legend_height + index * _BAR_ROW
        marks.append(_text(8, top + _BAR_HEIGHT - 3, label, title=bar.arm if label != bar.arm else None))
        x = float(start)
        lost = 0
        for segment in bar.segments:
            width = max(span * segment.count / longest - _SEGMENT_GAP, 1.0)
            mark = f"{bar.arm}: {segment.label}, {segment.count} of the {bar.total} trials it ran"
            alt = " viz-alt" if segment.state == "lost" and lost % 2 else ""
            lost += segment.state == "lost"
            marks.append(
                f'<rect class="viz-{segment.state}{alt}" x="{x:.1f}" y="{top}" width="{width:.1f}" '
                f'height="{_BAR_HEIGHT}" rx="2"><title>{_escape(mark)}</title></rect>'
            )
            if width >= _COUNT_INSIDE:
                inside = ' class="viz-on-fill"' if segment.state != "lost" else ""
                marks.append(
                    f'<text x="{x + width / 2:.1f}" y="{top + _BAR_HEIGHT - 4}"{inside} text-anchor="middle">'
                    f"{segment.count}</text>"
                )
            x += width + _SEGMENT_GAP
        marks.append(_text(x + 6, top + _BAR_HEIGHT - 3, f"{bar.total} trials"))
        marks.append(_text(start, top + _BAR_HEIGHT + 17, segment_words(bar), muted=True))
    desc = "; ".join(f"{bar.arm} ran {bar.total} trials: {segment_words(bar)}" for bar in bars)
    return _svg(chart_id, title, desc, height, marks)


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


def _legend(start: int, height: int) -> str:
    """The outcome bar's key: each state's swatch beside its name, so the colours are read off words."""
    parts = []
    x = start
    for state, words in (("passed", "passed"), ("failed", "failed"), ("lost", "lost to the test setup")):
        parts.append(f'<rect class="viz-{state}" x="{x}" y="{height - 20}" width="12" height="12" rx="2"></rect>')
        parts.append(_text(x + 17, height - 10, words, muted=True))
        x += 17 + int(_CHAR_WIDTH * len(words)) + 18
    return "".join(parts)


def interval_rows_svg(
    title: str,
    rows: Sequence[IntervalRow],
    *,
    unit: str,
    zero_line: bool,
    margin: float | None,
    chart_id: str,
) -> str:
    """Interval rows: a dot and its interval per row, the value, interval and cases written at the right.

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
    height = _INTERVAL_ROW * len(rows) + axis + 8
    marks = []
    plot_bottom = _INTERVAL_ROW * len(rows)
    if margin is not None:
        left, right = x_of(-margin), x_of(margin)
        marks.append(
            f'<rect class="viz-band" x="{left:.1f}" y="0" width="{right - left:.1f}" height="{plot_bottom}">'
            f"<title>{_escape(f'margin ±{render_cell(margin)}{_unit(unit)}')}</title></rect>"
        )
    if zero_line:
        zero = x_of(0.0)
        marks.append(f'<line class="viz-zero" x1="{zero:.1f}" y1="0" x2="{zero:.1f}" y2="{plot_bottom}"></line>')
    for index, (row, label) in enumerate(zip(rows, labels, strict=True)):
        middle = index * _INTERVAL_ROW + _INTERVAL_ROW / 2
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
        marks.append(_text(_PLOT_END + 10, middle + 4, said))
    marks.append(f'<line class="viz-rule" x1="{start}" y1="{plot_bottom}" x2="{_PLOT_END}" y2="{plot_bottom}"></line>')
    marks.append(_text(start, plot_bottom + 14, render_cell(low), muted=True))
    marks.append(_text(_PLOT_END, plot_bottom + 14, render_cell(high), muted=True, anchor="end"))
    desc = "; ".join(f"{row.label}: {row_words(row, unit)}" for row in rows)
    return _svg(chart_id, title, desc, height, marks)


def row_words(row: IntervalRow, unit: str) -> str:
    """One interval row in words: its value and unit, its interval, its cases, and its verdict where it has one."""
    verdict = f", {row.verdict}" if row.verdict else ""
    return (
        f"{render_cell(row.value)}{_unit(unit)} {interval_text(row.low, row.high)}, "
        f"n={row.n_cases} case{'' if row.n_cases == 1 else 's'}{verdict}"
    )


def _unit(unit: str) -> str:
    """A unit after a number: ``%`` against it, any other unit after a space, none at all for no unit."""
    if not unit:
        return ""
    return unit if unit == "%" else f" {unit}"


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
    "interval_rows_svg",
    "outcome_bar_svg",
    "row_words",
    "segment_words",
]
