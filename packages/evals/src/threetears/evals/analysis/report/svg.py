"""The interval glyph a table cell draws beside its number, as inline SVG with no dependency.

An *interval glyph* is the uncertainty of one figure drawn in the row that states it (Tufte's table-graphic): a
track, a zero line where zero is on it, the declared margin as a quiet band, the interval as a whisker and the point
as a dot, every row of a column in one unit on the same scale, so the rows compare as small multiples and the scale's
ends are written once under the table. The report's charts are drawn by the eval report kit's script from their data
(:meth:`~threetears.evals.analysis.report.blocks.ChartBlock.kit_chart`), not here.

**Legible without colour, a script or a stylesheet of its own.** A glyph is drawn in the page's ink alone, the margin
band in its one quiet tint, from classes the page's stylesheet sets from the kit's tokens (:data:`GLYPH_STYLE`); its
verdict is the word in the next cell, never a hue. Its ``<title>`` states its numbers, and every figure it draws is
in the row's text. There is no ``<script>``, no event attribute, no external reference and no URL of any kind.
"""

from __future__ import annotations

import html

from threetears.evals.analysis.report.blocks import interval_text
from threetears.evals.analysis.quantities import PERCENT, is_money, with_unit

#: The interval glyph's own coordinates: its width, its height, the track's inset from either edge, and the dot.
_GLYPH_WIDTH = 120
_GLYPH_HEIGHT = 14
_GLYPH_INSET = 4
_GLYPH_DOT = 2.5

#: The glyph's stylesheet: each mark's class, coloured from the kit's tokens (``kit/kit.css``), which define every one
#: for light and for dark, so a glyph follows the page's theme with no colour of its own: the page's ink, its margin
#: band the sunken tint, sized in ems so it sits on the row's line of text.
GLYPH_STYLE = """
svg.viz-glyph { width: 7.5em; height: 0.875em; vertical-align: -0.0625em; margin-right: 6px; overflow: visible; }
svg.viz-glyph .viz-track { stroke: var(--base); stroke-width: 1; }
svg.viz-glyph .viz-zero { stroke: var(--ink-2); stroke-width: 1; }
svg.viz-glyph .viz-band { fill: var(--sunk); }
svg.viz-glyph .viz-whisker { stroke: var(--ink); stroke-width: 1.5; stroke-linecap: round; }
svg.viz-glyph .viz-point { fill: var(--ink); }
"""


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

    :param value: The figure, in ``unit``.
    :ptype value: float
    :param low: The interval's low end, in ``unit``.
    :ptype low: float
    :param high: The interval's high end, in ``unit``.
    :ptype high: float
    :param unit: The unit the figures are in; empty for none.
    :ptype unit: str
    :param scale: The track's low and high ends, which hold the interval, the margin and zero where it is drawn.
    :ptype scale: tuple[float, float]
    :param margin: A declared margin either side of zero, or None.
    :ptype margin: float | None
    :return: The ``<svg>`` element.
    :rtype: str
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


def _interval_words(low: float, high: float, unit: str) -> str:
    """An interval as a glyph's title says it, as the row's own interval cell spells it: the unit on each end where
    it sits on the number (``[88.1%, 100%]``, ``[-$0.01, $0.06]``), once after the bracket otherwise
    (``[-4.2, 2] pts``)."""
    if unit == PERCENT or is_money(unit):
        return f"[{with_unit(low, unit)}, {with_unit(high, unit)}]"
    return interval_text(low, high, unit) + (f" {unit}" if unit else "")


def _escape(text: str) -> str:
    """Escape text for an SVG element or attribute."""
    return html.escape(text, quote=True)


__all__ = [
    "GLYPH_STYLE",
    "interval_glyph_svg",
]
