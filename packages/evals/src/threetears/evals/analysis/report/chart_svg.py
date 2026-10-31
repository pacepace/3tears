"""The charts a static page draws: each kit chart as inline SVG, drawn here from its data, for a page that runs no
script.

The default page hands each chart's data to the eval report kit's script (``kit/kit.js``), which draws it in the
browser at the chart's real width. A host whose policy runs no script and loads nothing (``default-src 'none'``, a
frame with an empty ``sandbox``) asks for the static page, and that page draws its charts here instead, on the
server, by the kit's own geometry: the same margins, ticks, marks, label rules and colour tokens, so a chart reads
the same on either page. What a script would do at the chart's width, a stylesheet does here: every chart is drawn
at three widths, a phone's, the kit's desktop measure and a wide page's, each scaling to its container by its
``viewBox``, and the page's stylesheet shows the one nearest the chart's own width by a container query
(:data:`CHART_STYLE`), so a label is drawn near the kit's size wherever the chart sits, never too small to read.
A small multiple's panel is drawn once, at a panel's width.

**Nothing runs, loads or links.** A drawing is SVG elements and the kit's legend markup: no ``<script>``, no event
attribute, no URL of any kind (no ``xmlns``, no ``href``, no ``url(#...)`` fill: a texture is drawn as its own lines
and dots). What a tooltip shows is each mark's ``<title>``, which a browser shows on hover with no script. Every
colour is one of the kit's tokens (``var(--s1)``), which its stylesheet defines for light and for dark, so a drawing
follows the page's theme; a colour reaches a style only as a token name, refused otherwise. Every string is escaped.

**The table stays.** The figure's table of values follows the drawing on the static page as it always has, for a
screen reader and for copying the numbers out; the drawing is the picture of it.

The kinds the kit lays out as HTML rather than drawing (a gate matrix, a lattice, example cards) are not drawn here:
:func:`chart_drawing` returns None for them, and the figure says in one line that it is shown as its table.
"""

from __future__ import annotations

import html
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from pydantic import BaseModel

from threetears.evals.analysis.report.blocks import ChartBlock
from threetears.evals.analysis.report.charts import (
    KIT_CHARTS,
    BarsChart,
    BarsPanel,
    ChartReference,
    ChartShape,
    ConfusionChart,
    ConfusionPanel,
    DiffHeatmapChart,
    DotIntervalChart,
    DotIntervalPanel,
    ForestChart,
    GateScatterChart,
    GateScatterPanel,
    HeatmapChart,
    HeatmapPanel,
    LegendItem,
    LinesChart,
    RangeBarsChart,
    ScatterChart,
    ScatterPanel,
)
from threetears.evals.schema.hashing import round_figures

#: The widths every chart is drawn at, in the kit's px, each with its class: the narrow one under every narrow rule
#: the kit has (480 px for bars, 560 px for a scatter's labels), as a phone draws it, then the kit's desktop measure
#: and a wide page's. The stylesheet shows the one nearest the chart's own width (:data:`CHART_STYLE`).
_WIDTHS: tuple[tuple[str, int], ...] = (("viz-narrow", 360), ("viz-mid", 640), ("viz-wide", 960))

#: A small multiple's width: the kit's narrowest panel (``.mult``'s 15.5rem) and a little over.
_PANEL_WIDTH = 280

#: The one line a figure carries where its kind is not drawn here.
AS_TABLE_NOTE = "This chart is shown as its table."

#: A colour as a style takes one: a token name, so data can never write CSS of its own.
_TOKEN = re.compile(r"^--[a-z][a-z0-9-]*$")

#: The drawing's stylesheet: the kit's chart rules (``kit.css``'s ``.chart`` ones) under the drawing's own class, so a
#: drawn chart reads as the kit's, and the container queries that show the drawing nearest the chart's own width, so
#: its text is drawn near the kit's size wherever the chart sits. Every colour is a kit token.
CHART_STYLE = """
.chart-drawn { width: 100%; min-width: 0; margin-bottom: .75rem; container-type: inline-size; }
.chart-drawn + .chart-values { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden;
  clip-path: inset(50%); white-space: nowrap; border: 0; }
.chart-drawn svg.viz-chart { display: none; width: 100%; height: auto; overflow: visible; }
.chart-drawn svg.viz-mid, .chart-drawn svg.viz-panel { display: block; }
@container (max-width: 499px) {
  .chart-drawn svg.viz-mid { display: none; }
  .chart-drawn svg.viz-narrow { display: block; }
}
@container (min-width: 800px) {
  .chart-drawn svg.viz-mid { display: none; }
  .chart-drawn svg.viz-wide { display: block; }
}
.chart-drawn text { font-family: var(--f-body); fill: var(--ink-2); }
.chart-drawn .tick { font-size: 11px; fill: var(--muted); font-variant-numeric: tabular-nums; }
.chart-drawn .lbl { font-size: 12px; fill: var(--ink-2); }
.chart-drawn .lbl-strong { font-size: 12px; fill: var(--ink); font-weight: 600; }
.chart-drawn .cell-main { font-size: 14px; font-weight: 600; font-variant-numeric: tabular-nums; }
.chart-drawn .cell-sub { font-size: 10px; opacity: .85; font-variant-numeric: tabular-nums; }
.chart-drawn .grid { stroke: var(--line); stroke-width: 1; }
.chart-drawn .axis { stroke: var(--base); stroke-width: 1; }
.chart-drawn .floor { stroke: var(--ink-2); stroke-width: 1.25; stroke-dasharray: 3 3; }
.chart-drawn .viz-hit { fill: none; pointer-events: all; }
.chart-drawn .legend svg { display: inline-block; vertical-align: -1px; margin-right: .4rem; }
.chart-drawn .mult .fig .chart-panel { margin-top: auto; padding-top: .4rem; }
.chart-note { font-size: var(--t-sm); color: var(--muted); margin: 0 0 .5rem; }
"""


# =============================================================================
# Numbers: the kit's named formats, its scales and its ticks
# =============================================================================

#: A kit format spec's parts: the sign flag, the name and the digits.
_FORMAT_SPEC = re.compile(r"^(\+?)([a-z]+)(?::(\d))?$")


def _fixed(value: float, digits: int) -> str:
    """``value`` to ``digits`` decimals, as JavaScript's ``toFixed`` writes it: the float's exact value rounded half
    away from zero (Python's own formatting rounds half to even), and no negative zero."""
    text = str(Decimal(value).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))
    return text[1:] if text.startswith("-") and float(text) == 0 else text


def _short(value: float, digits: int | None) -> str:
    """The kit's ``kShort``: as many decimals as the value needs, up to two, or exactly ``digits``."""
    if digits is not None:
        return _fixed(value, digits)
    text = _fixed(value, 2)
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def _rounded(value: float) -> int:
    """JavaScript's ``Math.round``: halves up."""
    return math.floor(value + 0.5)


def _grouped(value: float, digits: int) -> str:
    """A non-negative amount grouped by thousands, at ``digits`` decimals (``toLocaleString("en-US")``)."""
    return f"{value:,.{digits}f}"


def _usd(value: float) -> str:
    """The kit's money by its size: whole dollars from $10, cents from ten cents, three significant figures below."""
    size = abs(value)
    if size >= 10:
        text = f"{_rounded(size):,}"
    elif size == 0:
        text = "0"
    elif size >= 0.1:
        text = _grouped(size, 2)
    else:
        text = _fixed(size, 2 - math.floor(math.log10(size)))
    return ("-" if value < 0 and size != 0 else "") + "$" + text


def _usd_at(value: float, digits: int | None) -> str:
    """The kit's ``usd`` format, at its own rounding or at ``digits`` decimals."""
    if digits is None:
        return _usd(value)
    return ("-" if value < 0 else "") + "$" + _grouped(abs(value), digits)


def _usdk(value: float) -> str:
    if abs(value) >= 1000:
        return "$" + _fixed(value / 1000, 1) + "k"
    return "$" + f"{_rounded(value):,}"


def _usdpk(value: float, digits: int | None) -> str:
    return "$0" if value == 0 else _usd_at(value * 1000, 2 if digits is None else digits) + "/1000"


def _ms(value: float, digits: int | None) -> str:
    return (str(_rounded(value)) if digits is None else _fixed(value, digits)) + " ms"


def _dur(value: float) -> str:
    return _short(value / 1000, None) + "s" if value >= 1000 else _short(value, None) + "ms"


#: The kit's named formats (``kit.js``'s ``K_FMTS``), each a value and its digits (None for the format's own) to text.
_FORMATS: dict[str, Callable[[float, int | None], str]] = {
    "num": _short,
    "int": lambda value, _digits: f"{_rounded(value):,}",
    "pct": lambda value, digits: _short(value, digits) + "%",
    "pts": lambda value, digits: _short(value, digits) + " pts",
    "usd": _usd_at,
    "usdk": lambda value, _digits: _usdk(value),
    "usdpk": _usdpk,
    "ms": _ms,
    "s": lambda value, digits: _short(value, digits) + " s",
    "dur": lambda value, _digits: _dur(value),
    "x": lambda value, digits: _fixed(value, 1 if digits is None else digits) + "×",
    "k": lambda value, digits: _fixed(value / 1000, 1 if digits is None else digits) + "k",
}


def _kit_format(spec: str) -> Callable[[float], str]:
    """A kit format (``name`` or ``name:digits``, ``+`` first to sign a positive value) as the function that writes a
    value in it, as the kit's ``namedFmt`` does, so a drawn tick or note reads as the kit draws it.

    :param spec: The format, as a chart's data names it.
    :ptype spec: str
    :return: The value's text.
    :rtype: Callable[[float], str]
    :raises ValueError: ``spec`` is not a format the kit names.
    """
    match = _FORMAT_SPEC.fullmatch(spec)
    if match is None or match[2] not in _FORMATS:
        raise ValueError(f"fmt {spec!r} is not one of {', '.join(_FORMATS)}")
    write, digits, signed = _FORMATS[match[2]], None if match[3] is None else int(match[3]), match[1] == "+"

    def formatted(value: float) -> str:
        text = write(value, digits)
        return "+" + text if signed and value > 0 else text

    return formatted


@dataclass(frozen=True)
class _Scale:
    """A value axis mapped onto a pixel range: linear, or log where the chart asks for one."""

    d0: float
    d1: float
    r0: float
    r1: float
    log: bool = False

    def __call__(self, value: float) -> float:
        if self.log:
            low, high = math.log10(max(self.d0, 1e-12)), math.log10(max(self.d1, 1e-12))
            at = math.log10(max(value, 1e-12))
            return self.r0 + (at - low) / ((high - low) or 1.0) * (self.r1 - self.r0)
        return self.r0 + (value - self.d0) / ((self.d1 - self.d0) or 1.0) * (self.r1 - self.r0)


def _clamp(value: float, low: float, high: float) -> float:
    return max(min(low, high), min(max(low, high), value))


def _ticks_raw(a: float, b: float, n: int) -> list[float]:
    if b <= a or n < 1:
        return [a]
    step = 10 ** math.floor(math.log10((b - a) / n))
    err = (b - a) / n / step
    s = (10 if err >= 7.5 else 5 if err >= 3.5 else 2 if err >= 1.5 else 1) * step
    out: list[float] = []
    value = math.ceil(a / s) * s
    while value <= b + 1e-9:
        out.append(round(value, 10))
        value += s
    return out


def _ticks(a: float, b: float, n: int = 5) -> list[float]:
    if n > 1:
        k, found = n, _ticks_raw(a, b, n)
        while len(found) > n + 1 and k > 1:
            k -= 1
            found = _ticks_raw(a, b, k)
        return found
    return _ticks_raw(a, b, n)


def _format_ticks(a: float, b: float, n: int, f: Callable[[float], str]) -> list[float]:
    """Nice ticks whose labels are all distinct: the step widens until each label names one position."""
    for k in range(n, 1, -1):
        found = _ticks(a, b, k)
        if len({f(v) for v in found}) == len(found):
            return found
    return _ticks(a, b, 2)


def _thin_ticks(values: Sequence[float], x: _Scale, f: Callable[[float], str]) -> list[float]:
    """Explicit tick values, thinned left to right so no two labels touch at the drawn width."""
    out: list[float] = []
    last = -1e9
    for value in values:
        half = len(f(value)) * 3.2
        if x(value) - half < last + 6:
            continue
        out.append(value)
        last = x(value) + half
    return out


def _fit_ticks(a: float, b: float, n: int, f: Callable[[float], str], x: _Scale) -> list[float]:
    """Computed ticks, at most ``n``, fewer until no two neighbours' labels touch at the drawn scale."""
    for k in range(max(2, n), 1, -1):
        found = _format_ticks(a, b, k, f)
        if all(
            not i or x(v) - x(found[i - 1]) >= (len(f(v)) + len(f(found[i - 1]))) * 3.2 + 6 for i, v in enumerate(found)
        ):
            return found
    return _thin_ticks(_format_ticks(a, b, 2, f), x, f)


def _text_width(text: str, size: float = 12, strong: bool = False) -> float:
    """A text's width in the chart's face, estimated from its length: the kit measures it, a server cannot."""
    return len(text) * size * (0.58 if strong else 0.55)


# =============================================================================
# Markup
# =============================================================================


def _escape(text: str) -> str:
    return html.escape(text, quote=True)


def _n(value: float) -> str:
    """A coordinate, to a tenth of a px, from its value to twelve significant figures
    (:func:`~threetears.evals.schema.hashing.round_figures`): a coordinate is computed from a figure whose last bits
    differ between Python builds, and one that lands on a twentieth of a px (a 45% rate on a round axis) would round to
    either tenth by those bits alone."""
    value = round_figures(value)
    text = f"{value:.1f}"
    if text.endswith(".0"):
        text = text[:-2]
    return "0" if text == "-0" else text


def _tok(token: str | None, fallback: str) -> str:
    """A colour token for a style, ``fallback`` where none is named.

    :raises ValueError: The token is not a token name, so it could carry CSS of its own.
    """
    chosen = fallback if token is None else token
    if not _TOKEN.fullmatch(chosen):
        raise ValueError(f"color {chosen!r} is not a token name such as '--s1'")
    return chosen


def _el(tag: str, attrs: dict[str, Any], inner: str = "") -> str:
    """One element: its attributes (a number as a coordinate, None left out), each escaped, and ``inner``, markup
    this module wrote."""
    written = "".join(
        f' {name}="{_n(value) if isinstance(value, float | int) and not isinstance(value, bool) else _escape(str(value))}"'
        for name, value in attrs.items()
        if value is not None
    )
    return f"<{tag}{written}>{inner}</{tag}>"


def _text(x: float, y: float, text: str, css: str = "tick", anchor: str = "start", style: str | None = None) -> str:
    return _el("text", {"x": x, "y": y, "text-anchor": anchor, "class": css, "style": style}, _escape(text))


def _line(x1: float, y1: float, x2: float, y2: float, css: str | None = None, style: str | None = None) -> str:
    return _el("line", {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "class": css, "style": style})


def _titled(tip: Sequence[str], inner: str, hit: tuple[float, float, float, float] | None = None) -> str:
    """A mark with the lines its tooltip would show as its ``<title>``, over ``hit`` too where given (the row's
    whole band), so a hover anywhere on the row reads it."""
    target = (
        ""
        if hit is None
        else _el("rect", {"x": hit[0], "y": hit[1], "width": hit[2], "height": hit[3], "class": "viz-hit"})
    )
    return f"<g><title>{_escape(chr(10).join(tip))}</title>{inner}{target}</g>"


def _cut(label: str, room: int) -> str:
    return label if len(label) <= room else label[: max(room - 1, 0)] + "…"


def _mark(
    shape: ChartShape | None, x: float, y: float, r: float, color: str, hollow: bool | None, extra: str = ""
) -> str:
    """A mark in its arm's shape, so an arm reads apart without colour; hollow for a failure or an absent setting."""
    style = (
        f"fill:var(--surface);stroke:var({color});stroke-width:2"
        if hollow
        else f"fill:var({color});stroke:var(--surface);stroke-width:2"
    ) + extra
    if shape == "square":
        return _el(
            "rect",
            {"x": x - r * 0.88, "y": y - r * 0.88, "width": r * 1.76, "height": r * 1.76, "rx": 1.5, "style": style},
        )
    if shape == "triangle":
        path = f"M{_n(x)},{_n(y - r * 1.1)}L{_n(x + r * 1.05)},{_n(y + r * 0.8)}L{_n(x - r * 1.05)},{_n(y + r * 0.8)}Z"
        return _el("path", {"d": path, "style": style})
    if shape == "diamond":
        path = f"M{_n(x)},{_n(y - r * 1.15)}L{_n(x + r * 1.1)},{_n(y)}L{_n(x)},{_n(y + r * 1.15)}L{_n(x - r * 1.1)},{_n(y)}Z"
        return _el("path", {"d": path, "style": style})
    return _el("circle", {"cx": x, "cy": y, "r": r, "style": style})


def _texture(name: str, x: float, y: float, w: float, h: float, color: str, opacity: float = 1.0) -> str:
    """A fill told apart by its texture, as the kit's ``kTexture`` draws it: washed in its token, the texture drawn
    over it as plain lines and dots cut to the box by geometry, so it references nothing outside its own marks."""
    marks = [
        _el(
            "rect",
            {
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "rx": 2,
                "style": f"fill:color-mix(in oklab, var({color}) 22%, var(--surface));stroke:var({color});stroke-width:1",
            },
        )
    ]
    if w > 0 and h > 0:
        step, path = 6.0, []

        def diagonal(sx: float, sy: float, direction: int) -> None:
            a, b = max(0.0, x - sx), min(h, x + w - sx)
            if b > a:
                path.append(f"M{_n(sx + a)},{_n(sy + direction * a)}L{_n(sx + b)},{_n(sy + direction * b)}")

        if name in ("hatch", "cross"):
            k = -h
            while k < w:
                diagonal(x + k, y + h, -1)
                k += step
        if name == "cross":
            k = -h
            while k < w:
                diagonal(x + k, y, 1)
                k += step
        if name == "rows":
            v = y + step / 2
            while v < y + h:
                path.append(f"M{_n(x)},{_n(v)}H{_n(x + w)}")
                v += step
        if name == "cols":
            v = x + step / 2
            while v < x + w:
                path.append(f"M{_n(v)},{_n(y)}V{_n(y + h)}")
                v += step
        if path:
            width = 1 if name == "cross" else 1.4
            marks.append(
                _el("path", {"d": "".join(path), "style": f"fill:none;stroke:var({color});stroke-width:{width}"})
            )
        if name == "dots":
            i = x + step / 2
            while i < x + w:
                j = y + step / 2
                while j < y + h:
                    marks.append(_el("circle", {"cx": i, "cy": j, "r": 1.4, "style": f"fill:var({color})"}))
                    j += step
                i += step
    return _el("g", {"style": f"opacity:{opacity:g}" if opacity != 1 else None}, "".join(marks))


@dataclass(frozen=True)
class _RefLabel:
    """A reference line's label as placed: where it starts, its anchor, its lines and its widest."""

    x: float
    anchor: str
    lines: list[str]
    width: float


def _ref_label(rx: float, text: str, x0: float, x1: float) -> _RefLabel:
    """A reference line's label beside its line at ``rx`` and inside ``[x0, x1]``, as the kit's ``kRefLabel`` places
    it: on one line to the right where it fits, else to the left, else wrapped at its spaces to the wider side."""

    def width(t: str) -> float:
        return _text_width(t, 11)

    tw, lx = width(text), min(rx - 4, x1)
    right, left = x1 - rx - 4, lx - x0
    if tw <= right:
        return _RefLabel(rx + 4, "start", [text], tw)
    if tw <= left:
        return _RefLabel(lx, "end", [text], tw)
    room, lines, current = max(right, left), [], ""
    for word in text.split(" "):
        joined = f"{current} {word}" if current else word
        if current and width(joined) > room:
            lines.append(current)
            current = word
        else:
            current = joined
    lines.append(current)
    widest = max(width(line) for line in lines)
    if widest > room:
        return _RefLabel(x1, "end", lines, widest)
    return _RefLabel(rx + 4, "start", lines, widest) if right >= left else _RefLabel(lx, "end", lines, widest)


def _ref_text(label: _RefLabel, y: float, up: bool) -> str:
    """The label drawn as one text, its lines 13 px apart: stacked up to baseline ``y``, else down from it."""
    count = len(label.lines)
    spans = "".join(
        _el("tspan", {"x": label.x, "dy": 13 if i else 0}, _escape(line + " " if i < count - 1 else line))
        for i, line in enumerate(label.lines)
    )
    top = y - (count - 1) * 13 if up else y
    return _el("text", {"x": label.x, "y": top, "text-anchor": label.anchor, "class": "tick"}, spans)


def _ref_box(label: _RefLabel, y: float) -> tuple[float, float, float, float, float]:
    start = label.x - label.width if label.anchor == "end" else label.x
    return (start, y - 10, start + label.width, y + 3 + (len(label.lines) - 1) * 13, 10.0)


def _need[T](value: T | None, name: str) -> T:
    """A field the kit refuses to draw without, which the chart's model has already required.

    :raises ValueError: It is missing.
    """
    if value is None:
        raise ValueError(f"{name!r} required")
    return value


def _tip_or(tip: list[str] | None, *default: str) -> list[str]:
    return list(tip) if tip else list(default)


# =============================================================================
# The drawn kinds, each the kit's geometry at width w: its marks and its height
# =============================================================================

_Drawn = tuple[str, float]

_BarsLike = BarsChart | BarsPanel


def _bars(d: _BarsLike, w: float) -> _Drawn:
    """Horizontal bars (``barsH``): plain or stacked, diverging from a no-preference point, grouped, with whiskers and
    a reference line."""
    rows, (d0, d1) = _need(d.rows, "rows"), _need(d.domain, "domain")
    f = _kit_format(_need(d.fmt, "fmt"))
    tf = _kit_format(d.tick_fmt) if d.tick_fmt else f
    ref = d.reference
    no_preference = ref is not None and ref.kind == "no_preference"
    origin = ref.value if ref is not None and no_preference else 0.0
    origin_label = ref.label if ref is not None and no_preference else None
    line_ref: ChartReference | None = None if no_preference else ref
    row_h = d.row_height or 28
    narrow = w < 480
    tag_w = 22 if any(r.tag is not None for r in rows) else 0
    lw = (d.label_width_narrow if narrow else d.label_width) or (min(130, w * 0.36) if narrow else min(220, w * 0.3))
    ml, mr, mb = lw, (52 if narrow else 70), 22
    mt = 20 if (line_ref is not None or d.level or origin_label) else 6
    x = _Scale(d0, d1, ml, w - mr, log=bool(d.log))
    level_room = w - _text_width(f"{d.level} intervals", 11) - 8 if d.level else w
    label = _ref_label(x(line_ref.value), line_ref.label, 0, level_room) if line_ref and line_ref.label else None
    if label is not None:
        mt += (len(label.lines) - 1) * 13
    grouped = any(r.group is not None for r in rows)
    gaps = sum(1 for i in range(1, len(rows)) if rows[i].group != rows[i - 1].group)
    h = mt + mb + row_h * len(rows) + gaps * 14 + (14 if grouped else 0)
    out: list[str] = []
    tick_w = max(len(tf(v)) for v in (d0, d1, (d0 + d1) / 2)) * 6.4 + 14
    shown = _thin_ticks(d.ticks, x, tf) if d.ticks else _fit_ticks(d0, d1, math.floor((w - ml - mr) / tick_w), tf, x)
    for v in shown:
        out.append(_line(x(v), mt, x(v), h - mb, "grid"))
        out.append(_text(x(v), h - 6, tf(v), anchor="middle"))
    z = x(_clamp(origin, d0, d1))
    out.append(_line(z, mt - 2, z, h - mb, "axis"))
    if origin_label:
        out.append(_text(z + 4, mt - 8, origin_label))
    y0: float = mt + (14 if grouped else 0)
    for i, r in enumerate(rows):
        if grouped and (i == 0 or r.group != rows[i - 1].group):
            if i:
                y0 += 14
            out.append(_text(0, y0 - 4, r.group or "", "lbl-strong"))
        cy = y0 + row_h / 2
        y0 += row_h
        strong = bool(r.strong) or (d.emphasis is not None and r.id == d.emphasis)
        value = sum(s.value for s in r.segments) if r.segments else _need(r.value, "value")
        v = _clamp(value, d0, d1)
        color = _tok(r.color, "--s1")
        opacity = 0.45 if r.muted else 1
        marks: list[str] = []
        room = math.floor((ml - 14 - tag_w) / 6.6)
        shown_label = r.label
        if len(shown_label) > room:
            if r.short_label and len(r.short_label) <= room:
                shown_label = r.short_label
            elif r.tag is not None:
                shown_label = ""
            else:
                shown_label = _cut(r.short_label or shown_label, room)
        marks.append(_text(ml - 10 - tag_w, cy + 4, shown_label, "lbl-strong" if strong else "lbl", "end"))
        if r.tag is not None:
            tag_color = _tok(r.color, "--ink-2")
            on_tag = "--surface" if tag_color == "--ink-2" else "--on-fill"
            marks.append(
                _el(
                    "rect",
                    {"x": ml - 26, "y": cy - 8, "width": 16, "height": 16, "rx": 3, "style": f"fill:var({tag_color})"},
                )
            )
            marks.append(
                _text(ml - 18, cy + 3.5, r.tag, "tag", "middle", f"fill:var({on_tag});font-size:10px;font-weight:600")
            )
        if r.segments:
            acc = origin
            for j, seg in enumerate(r.segments):
                a, b = x(_clamp(acc, d0, d1)), x(_clamp(acc + seg.value, d0, d1))
                seg_color = _tok(seg.color, color)
                sx, sw = min(a, b) + (1 if j else 0), max(0.0, abs(b - a) - (1 if j else 0))
                sy, sh = cy - row_h * 0.28, row_h * 0.56
                if seg.pattern is not None:
                    marks.append(_texture(seg.pattern, sx, sy, sw, sh, seg_color, opacity))
                else:
                    marks.append(
                        _el(
                            "rect",
                            {
                                "x": sx,
                                "y": sy,
                                "width": sw,
                                "height": sh,
                                "rx": 2,
                                "style": f"fill:var({seg_color});opacity:{opacity}",
                            },
                        )
                    )
                acc += seg.value
        else:
            bar_x, bar_w = min(z, x(v)), max(1.5, abs(x(v) - z))
            marks.append(
                _el(
                    "rect",
                    {
                        "x": bar_x,
                        "y": cy - row_h * 0.28,
                        "width": bar_w,
                        "height": row_h * 0.56,
                        "rx": 3,
                        "style": f"fill:var({color});opacity:{opacity}",
                    },
                )
            )
        open_above = r.lo is not None and r.hi is None
        hi = d1 if open_above else r.hi
        whisker = "stroke:var(--ink);stroke-width:1.25"
        if r.lo is not None and hi is not None:
            marks.append(_line(x(max(d0, r.lo)), cy, x(min(d1, hi)), cy, style=whisker))
            for end in [r.lo] if open_above else [r.lo, hi]:
                at = x(_clamp(end, d0, d1))
                marks.append(_line(at, cy - 4, at, cy + 4, style=whisker))
            if open_above:
                ex = x(d1)
                marks.append(
                    _el(
                        "path",
                        {
                            "d": f"M{_n(ex - 5)},{_n(cy - 4)}L{_n(ex)},{_n(cy)}L{_n(ex - 5)},{_n(cy + 4)}",
                            "style": "fill:none;" + whisker,
                        },
                    )
                )
        right = value >= origin
        if r.lo is not None and hi is not None:
            edge = x(min(d1, hi)) if right else x(max(d0, r.lo))
        else:
            edge = x(v)
        note = r.note if r.note is not None else f(value)
        if right and edge + 6 + len(note) * 6.4 > w:
            note = f(value)
        note_w = len(note) * 6.4
        flip = not right and edge - 6 - note_w < ml + 4
        past = max(z, x(min(d1, hi if hi is not None else origin)))
        if flip and past + 6 + note_w > w:
            note = f(value)
        note_x = edge + 6 if right else past + 6 if flip else edge - 6
        marks.append(
            _text(note_x, cy + 4, note, "lbl-strong" if strong else "tick", "start" if right or flip else "end")
        )
        if r.lo is None:
            spread = ""
        elif open_above:
            spread = f" (from {f(r.lo)} up, no upper bound)"
        else:
            spread = f" ({f(r.lo)} to {f(_need(r.hi, 'hi'))})"
        out.append(_titled(_tip_or(r.tip, r.label, f(value) + spread), "".join(marks), (0, cy - row_h / 2, w, row_h)))
    if line_ref is not None:
        rx = x(line_ref.value)
        out.append(_line(rx, mt - 6, rx, h - mb, "floor"))
        if label is not None:
            out.append(_ref_text(label, mt - 8, True))
    if d.level:
        out.append(_text(w, mt - 8, f"{d.level} intervals", anchor="end"))
    return "".join(out), h


def _forest(d: ForestChart, w: float) -> _Drawn:
    """A forest plot (``forest``): each change against the control with its interval, and an optional margin band."""
    rows, (d0, d1), f = d.rows, d.domain, _kit_format(d.fmt)
    narrow, row_h = w < 480, d.row_height or 30
    need = max(len(r.label) for r in rows) * 6.6 + 18
    lw = (d.label_width_narrow if narrow else d.label_width) or min(need, w * 0.42 if narrow else min(230, w * 0.32))
    ml, mr, mt, mb = lw, (60 if narrow else 96), 22.0, 24
    gaps = sum(1 for i in range(1, len(rows)) if rows[i].group != rows[i - 1].group)
    x = _Scale(d0, d1, ml, w - mr)
    ref = d.reference
    level_room = w - mr - _text_width(f"{d.level} intervals", 11) - 8 if d.level else w
    label = _ref_label(x(max(d0, ref.value)), ref.label, 0, level_room) if ref is not None and ref.label else None
    if label is not None:
        mt += (len(label.lines) - 1) * 13
    grouped = any(r.group for r in rows)
    h = mt + mb + row_h * len(rows) + gaps * 14 + (14 if grouped else 0)
    out: list[str] = []
    if ref is not None:
        mx = x(max(d0, ref.value))
        out.append(
            _el(
                "rect",
                {
                    "x": mx,
                    "y": mt,
                    "width": x(d1) - mx,
                    "height": h - mt - mb,
                    "style": "fill:var(--good-wash);opacity:.6",
                },
            )
        )
        out.append(_line(mx, mt - 4, mx, h - mb, "floor"))
        if label is not None:
            out.append(_ref_text(label, mt - 8, True))
    tick_w = max(len(f(d0)), len(f(d1))) * 6.4 + 16
    for v in _fit_ticks(d0, d1, math.floor((w - ml - mr) / tick_w), f, x):
        out.append(_line(x(v), mt, x(v), h - mb, "grid"))
        out.append(_text(x(v), h - 6, f(v), anchor="middle"))
    out.append(_line(x(0), mt - 4, x(0), h - mb, "axis", "stroke-width:1.5"))
    if d.level:
        out.append(_text(w - mr, mt - 8, f"{d.level} intervals", anchor="end"))
    y0 = mt + (14 if grouped else 0)
    for i, r in enumerate(rows):
        if r.group and (i == 0 or r.group != rows[i - 1].group):
            if i:
                y0 += 14
            out.append(_text(0, y0 - 4, r.group, "lbl-strong"))
        cy = y0 + row_h / 2
        y0 += row_h
        strong = bool(r.strong) or (d.emphasis is not None and r.id == d.emphasis)
        color = _tok(r.color, "--s1")

        def at(value: float) -> float:
            return x(max(d0, min(d1, value)))

        room = math.floor((ml - 14) / 6.6)
        if len(r.label) <= room:
            shown_label = r.label
        elif r.short_label and len(r.short_label) <= room:
            shown_label = r.short_label
        else:
            shown_label = _cut(r.label, room)
        note = r.note if r.note is not None else f(r.value)
        if len(note) * 6.4 > mr - 10:
            note = f(r.value)
        marks = [
            _text(ml - 10, cy + 4, shown_label, "lbl-strong" if strong else "lbl", "end"),
            _line(at(r.lo), cy, at(r.hi), cy, style=f"stroke:var({color});stroke-width:3;stroke-linecap:round"),
            _el(
                "rect",
                {
                    "x": at(r.value) - 5,
                    "y": cy - 5,
                    "width": 10,
                    "height": 10,
                    "rx": 2,
                    "style": f"fill:var({color});stroke:var(--surface);stroke-width:2",
                },
            ),
            _text(w - mr + 8, cy + 4, note, "lbl-strong" if strong else "tick"),
        ]
        tip = _tip_or(r.tip, r.label, f"{f(r.value)} [{f(r.lo)}, {f(r.hi)}]")
        out.append(_titled(tip, "".join(marks), (0, cy - row_h / 2, w, row_h)))
    return "".join(out), h


def _lines(d: LinesChart, w: float) -> _Drawn:
    """A line chart over ordered categories (``lineChart``), each series' band where it carries one."""
    narrow, cats, f = w < 480, d.categories, _kit_format(d.fmt)
    h = d.height or (260 if narrow else 300)
    ml, mr, mt, mb = 44, (92 if narrow else 130), (24 if d.level else 14), (44 if d.x_title else 28)
    d0, d1 = d.domain
    y = _Scale(d0, d1, h - mb, mt)

    def x(i: int) -> float:
        return ml + (0 if len(cats) == 1 else i / (len(cats) - 1)) * (w - ml - mr)

    out: list[str] = []
    for v in _format_ticks(d0, d1, 5, f):
        out.append(_line(ml, y(v), w - mr, y(v), "grid"))
        if v > d0:
            out.append(_text(ml - 8, y(v) + 4, f(v), anchor="end"))
    for i, cat in enumerate(cats):
        out.append(_text(x(i), h - mb + 16, cat, anchor="middle"))
    if d.x_title:
        out.append(_text((ml + w - mr) / 2, h - 6, d.x_title, anchor="middle"))
    if d.level:
        out.append(_text(w - mr, 12, f"{d.level} intervals", anchor="end"))
    for se in d.series:
        if se.lo and se.hi:
            color = _tok(se.color, "--s1")
            upper = "L".join(f"{_n(x(i))},{_n(y(v))}" for i, v in enumerate(se.hi))
            lower = "L".join(f"{_n(x(i))},{_n(y(v))}" for i, v in reversed(list(enumerate(se.lo))))
            out.append(_el("path", {"d": f"M{upper}L{lower}Z", "style": f"fill:var({color});opacity:.12"}))
    ends: list[tuple[float, float, str, bool]] = []
    for se in d.series:
        color = _tok(se.color, "--s1")
        dash = "stroke-dasharray:5 4;" if se.dash else ""
        path = "L".join(f"{_n(x(i))},{_n(y(v))}" for i, v in enumerate(se.values))
        out.append(
            _el(
                "path",
                {
                    "d": f"M{path}",
                    "style": f"fill:none;stroke:var({color});stroke-width:{2.5 if se.strong else 2};{dash}",
                },
            )
        )
        for i, v in enumerate(se.values):
            spread = f" ({f(se.lo[i])}–{f(se.hi[i])})" if se.lo and se.hi else ""
            dot = _el(
                "circle",
                {
                    "cx": x(i),
                    "cy": y(v),
                    "r": 4.5 if se.strong else 3.5,
                    "style": f"fill:var({color});stroke:var(--surface);stroke-width:2",
                },
            )
            out.append(_titled([f"{se.name} · {cats[i]}", f"{f(v)}{spread}"], dot, (x(i) - 10, y(v) - 10, 20, 20)))
        ends.append((x(len(cats) - 1), y(se.values[-1]), se.end_label or se.name, bool(se.strong)))
    ends.sort(key=lambda end: end[1])
    last, room = -1e9, math.floor((mr - 12) / 6.6)
    for px, py, text, strong in ends:
        at = max(py + 4, last + 16)
        last = at
        out.append(_text(px + 10, at, _cut(text, room), "lbl-strong" if strong else "lbl"))
    return "".join(out), h


@dataclass
class _Label:
    """A point's label to place: its mark, its text, and where it lands."""

    px: float
    py: float
    text: str
    strong: bool
    r: float
    x: float = 0.0
    y: float = 0.0
    anchor: str = "start"


def _place_labels(
    items: list[_Label],
    bounds: tuple[float, float, float, float],
    obstacles: list[tuple[float, float, float, float, float]],
    marks: list[tuple[float, float]],
) -> list[_Label]:
    """The kit's ``placeLabels``: each label right, left, above or below its mark, then nudged, at the spot that
    collides least with the labels placed before it and the obstacles, and never beside another arm's mark."""
    placed = list(obstacles)

    def near(box: Sequence[float], x: float, y: float) -> float:
        return math.hypot(max(box[0] - x, 0, x - box[2]), max(box[1] - y, 0, y - box[3]))

    for it in items:
        width = _text_width(it.text, 12, it.strong) + 3
        gap = it.r + 2
        right, left, up, down = it.px + gap + 1.5, it.px - gap - 1.5, it.py - gap - 4.5, it.py + gap + 13
        options: list[tuple[float, float, str]] = [
            (right, it.py + 4, "start"),
            (left, it.py + 4, "end"),
            (it.px, up, "middle"),
            (it.px, down, "middle"),
        ]
        for nudge in (-6, 6, 12, -12, 24, -24, 36, -36):
            options += [(right, it.py + 4 + nudge, "start"), (left, it.py + 4 + nudge, "end")]
        options += [(it.px, up - 14, "middle"), (it.px, down + 14, "middle")]

        def box(option: tuple[float, float, str], width: float = width) -> tuple[float, float, float, float]:
            x0 = (
                option[0] - width
                if option[2] == "end"
                else option[0] - width / 2
                if option[2] == "middle"
                else option[0]
            )
            return (x0, option[1] - 12.5, x0 + width, option[1] + 4)

        def score(b: tuple[float, float, float, float], it: _Label = it) -> float:
            s = 100.0 if b[0] < bounds[0] or b[2] > bounds[2] or b[1] < bounds[1] or b[3] > bounds[3] else 0.0
            for p in placed:
                if b[0] < p[2] and b[2] > p[0] and b[1] < p[3] and b[3] > p[1]:
                    s += p[4]
            own = near(b, it.px, it.py)
            s += own * 0.06
            if any((mx != it.px or my != it.py) and near(b, mx, my) < max(own, 4) for mx, my in marks):
                s += 3
            return s

        pick, best = options[0], 1e9
        for i, option in enumerate(options):
            scored = score(box(option)) + i * 0.08
            if scored < best:
                best, pick = scored, option
        it.x, it.y, it.anchor = pick
        placed.append((*box(pick), 10.0))
    return items


_ScatterLike = ScatterChart | ScatterPanel | GateScatterChart | GateScatterPanel


def _scatter(d: _ScatterLike, w: float, gate: bool) -> _Drawn:
    """Two measures that trade off (``kScatter``); ``gate`` shades where both references pass (``gate_scatter``)."""
    points = _need(d.points, "points")
    xf, yf = _kit_format(_need(d.x_fmt, "x_fmt")), _kit_format(_need(d.y_fmt, "y_fmt"))
    (x0, x1), (y0, y1) = _need(d.x_domain, "x_domain"), _need(d.y_domain, "y_domain")
    narrow = w < 560
    y_ticks = d.y_ticks or _format_ticks(y0, y1, 5, yf)
    h = d.height or round(min(380, max(300, w * 0.45)))
    ml, mr, mt, mb = max(30, max(len(yf(v)) for v in y_ticks) * 6.6 + 16), 18, 16, 40
    x, y = _Scale(x0, x1, ml, w - mr, log=bool(d.x_log)), _Scale(y0, y1, h - mb, mt)
    obstacles: list[tuple[float, float, float, float, float]] = []
    out: list[str] = []

    def pass_side(r: ChartReference, lo: float, hi: float) -> tuple[float, float]:
        return (lo, r.value) if r.kind == "budget" else (r.value, hi)

    if gate:
        x_ref, y_ref = _need(d.x_reference, "x_reference"), _need(d.reference, "reference")
        ax, bx = pass_side(x_ref, x0, x1)
        ay, by = pass_side(y_ref, y0, y1)
        out.append(
            _el(
                "rect",
                {
                    "x": x(ax),
                    "y": y(by),
                    "width": x(bx) - x(ax),
                    "height": y(ay) - y(by),
                    "style": "fill:var(--good-wash)",
                },
            )
        )
        region = getattr(d, "region_label", None)
        if region:
            out.append(_text(x(ax) + 8, y(by) + 16, region, "lbl", style="fill:var(--good);font-weight:600"))
            obstacles.append((x(ax) + 8, y(by) + 6, x(ax) + 8 + len(region) * 6.6, y(by) + 19, 10.0))
    price = d.price_line
    if price is not None:
        p = price.value

        def y_at(v: float) -> float:
            return -(v / max(1e-9, p))

        lx = min(x1, p * -y0)
        region_path = (
            f"M{_n(x(0))},{_n(y(0))}L{_n(x(x1))},{_n(y(0))}L{_n(x(x1))},{_n(y(max(y0, y_at(x1))))}"
            f"L{_n(x(lx))},{_n(y(max(y0, y_at(lx))))}Z"
        )
        out.append(_el("path", {"d": region_path, "style": "fill:var(--good-wash)"}))
    for v in y_ticks:
        out.append(_line(ml, y(v), w - mr, y(v), "grid"))
        if v > y0:
            out.append(_text(ml - 8, y(v) + 4, yf(v), anchor="end"))
    for v in _thin_ticks(d.x_ticks or _format_ticks(x0, x1, 5, xf), x, xf):
        out.append(_line(x(v), mt, x(v), h - mb, "grid"))
        out.append(_text(x(v), h - mb + 16, xf(v), anchor="middle"))
    if d.x_title:
        out.append(_text((ml + w - mr) / 2, h - 6, d.x_title, anchor="middle"))
    if d.y_title:
        out.append(_text(ml + 6, h - mb - 8, d.y_title))
        obstacles.append((ml + 6, h - mb - 20, ml + 6 + len(d.y_title) * 6.4, h - mb - 6, 10.0))
    if d.x_reference is not None:
        rx = x(d.x_reference.value)
        out.append(_line(rx, mt, rx, h - mb, "floor"))
        obstacles.append((rx - 2, mt, rx + 2, h - mb, 1.0))
        if d.x_reference.label:
            label = _ref_label(rx, d.x_reference.label, ml, w - mr)
            out.append(_ref_text(label, mt + 12, False))
            obstacles.append(_ref_box(label, mt + 12))
    if d.reference is not None:
        ry = y(d.reference.value)
        out.append(_line(ml, ry, w - mr, ry, "floor"))
        obstacles.append((ml, ry - 2, w - mr, ry + 2, 1.0))
        if d.reference.label:
            label = _ref_label(w - mr + 4, d.reference.label, ml, w - mr)
            out.append(_ref_text(label, ry + 13, False))
            obstacles.append(_ref_box(label, ry + 13))
    if price is not None:
        p = price.value
        lx = min(x1, p * -y0)
        out.append(
            _line(
                x(0),
                y(0),
                x(lx),
                y(max(y0, -(lx / max(1e-9, p)))),
                style="stroke:var(--good);stroke-width:1.5;stroke-dasharray:5 4",
            )
        )
        if price.label:
            mx = lx * 0.55
            tx, ty = x(mx) + 8, y(max(y0, -(mx / max(1e-9, p)))) + 16
            out.append(_text(tx, ty, price.label, style="fill:var(--good)"))
            obstacles.append((tx, ty - 10, tx + len(price.label) * 6.4, ty + 3, 10.0))
        if price.region_label:
            t = price.region_label
            out.append(_text(w - mr - 8, mt + 16, t, "lbl", "end", "fill:var(--good);font-weight:600"))
            obstacles.append((w - mr - 8 - len(t) * 6.6, mt + 6, w - mr - 8, mt + 19, 10.0))

    def at(px: float, py: float) -> tuple[float, float]:
        return x(_clamp(px, x0, x1)), y(_clamp(py, y0 + (y1 - y0) * 0.025, y1))

    by_id = {point.id: point for point in points if point.id is not None}
    if d.frontier:
        best, front = -math.inf, []
        for point in sorted(points, key=lambda point: point.x):
            if point.y > best:
                front.append(point)
                best = point.y
        path = "L".join(f"{_n(a)},{_n(b)}" for a, b in (at(p.x, p.y) for p in front))
        out.append(_el("path", {"d": f"M{path}", "style": "fill:none;stroke:var(--base);stroke-width:1.5"}))
    for joined in d.paths or []:
        missing = [one for one in joined.ids if one not in by_id]
        if missing:
            raise ValueError(f"path names point {missing[0]!r}, which is not in 'points'")
        path = "L".join(f"{_n(a)},{_n(b)}" for a, b in (at(by_id[one].x, by_id[one].y) for one in joined.ids))
        style = (
            f"fill:none;stroke:var({_tok(joined.color, '--base')});stroke-width:{joined.width or 2:g};"
            f"opacity:{1 if joined.opacity is None else joined.opacity:g}"
        )
        out.append(_el("path", {"d": f"M{path}", "style": style}))
    for point in points:
        px, py = at(point.x, point.y)
        r = point.size or 5.5
        marks = _mark(
            point.shape, px, py, r, _tok(point.color, "--s1"), point.hollow, ";opacity:.45" if point.muted else ""
        )
        if d.emphasis is not None and point.id == d.emphasis:
            marks += _el(
                "circle", {"cx": px, "cy": py, "r": r + 4.5, "style": "fill:none;stroke:var(--ink);stroke-width:1.5"}
            )
        obstacles.append((px - r - 2, py - r - 2, px + r + 2, py + r + 2, 10.0))
        tip = _tip_or(point.tip, point.label or point.id or "", f"{xf(point.x)} · {yf(point.y)}")
        out.append(_titled(tip, marks, (px - 12, py - 12, 24, 24)))
    labels = []
    for point in points:
        text = point.narrow_label if narrow else point.label
        if text:
            px, py = at(point.x, point.y)
            strong = d.emphasis is not None and point.id == d.emphasis
            labels.append(_Label(px, py, text, strong, point.size or 5.5))
    labels.sort(key=lambda one: (0 if one.strong else 1, one.px))
    for placed in _place_labels(labels, (ml, 0, w, h - mb), obstacles, [at(p.x, p.y) for p in points]):
        out.append(_text(placed.x, placed.y, placed.text, "lbl-strong" if placed.strong else "lbl", placed.anchor))
    return "".join(out), h


_DotLike = DotIntervalChart | DotIntervalPanel


def _dot_interval(d: _DotLike, w: float) -> _Drawn:
    """One rate per row against a floor (``kDotInterval``): a dot in its arm's shape, its interval, the floor
    dashed."""
    rows, (d0, d1), f = _need(d.rows, "rows"), _need(d.domain, "domain"), _kit_format(_need(d.fmt, "fmt"))
    row_h = d.row_height or 28
    need = max(len(r.label) for r in rows) * 6.6 + 18
    ml = (d.label_width_narrow if w < 480 else d.label_width) or min(need, w * 0.42)
    mr, mt, mb = 14, 22.0, 22
    room = math.floor((ml - 12) / 6.6)
    x = _Scale(d0, d1, ml, w - mr)
    ref = d.reference
    level_room = w - mr - _text_width(f"{d.level} intervals", 11) - 8 if d.level else w
    label = _ref_label(x(ref.value), ref.label, 0, level_room) if ref is not None and ref.label else None
    if label is not None:
        mt += (len(label.lines) - 1) * 13
    h = mt + mb + row_h * len(rows)
    out: list[str] = []
    tick_w = max(len(f(d0)), len(f(d1))) * 6.4 + 16
    shown = _thin_ticks(d.ticks, x, f) if d.ticks else _fit_ticks(d0, d1, math.floor((w - ml - mr) / tick_w), f, x)
    for v in shown:
        out.append(_line(x(v), mt, x(v), h - mb, "grid"))
        out.append(_text(x(v), h - 6, f(v), anchor="middle"))
    for i, r in enumerate(rows):
        cy, color = mt + row_h * i + row_h / 2, _tok(r.color, "--s1")
        strong = bool(r.strong) or (d.emphasis is not None and r.id == d.emphasis)
        marks = [_text(ml - 8, cy + 4, _cut(r.label, room), "lbl-strong" if strong else "lbl", "end")]
        if r.lo is not None and r.hi is not None:
            style = f"stroke:var({color});stroke-width:3;stroke-linecap:round;opacity:.35"
            marks.append(_line(x(_clamp(r.lo, d0, d1)), cy, x(_clamp(r.hi, d0, d1)), cy, style=style))
        marks.append(_mark(r.shape, x(_clamp(r.value, d0, d1)), cy, 5, color, r.hollow))
        spread = f" ({f(r.lo)} to {f(r.hi)})" if r.lo is not None and r.hi is not None else ""
        out.append(
            _titled(
                _tip_or(r.tip, r.label, f(r.value) + spread), "".join(marks), (ml, cy - row_h / 2, w - ml - mr, row_h)
            )
        )
    if ref is not None:
        rx = x(ref.value)
        out.append(_line(rx, mt - 4, rx, h - mb, "floor"))
        if label is not None:
            out.append(_ref_text(label, mt - 8, True))
    if d.level:
        out.append(_text(w - mr, mt - 8, f"{d.level} intervals", anchor="end"))
    return "".join(out), h


def _range_bars(d: RangeBarsChart, w: float) -> _Drawn:
    """Latency distributions (``kRangeBars``): one bar per row from ``from`` to ``to``, grouped, with a budget line."""
    groups, (d0, d1), f = d.groups, d.domain, _kit_format(d.fmt)

    def band(rows: int) -> float:
        return rows * 13 + 20

    ml, mr, mt, mb = min(190, w * 0.36), 16, 16.0, 26
    x = _Scale(d0, d1, ml, w - mr)
    ref = d.reference
    label = _ref_label(x(ref.value), ref.label, 0, w) if ref is not None and ref.label else None
    if label is not None:
        mt += (len(label.lines) - 1) * 13
    h = mt + mb + sum(band(len(group.rows)) for group in groups)
    out: list[str] = []
    tick_w = max(len(f(d0)), len(f(d1))) * 6.4 + 16
    shown = _thin_ticks(d.ticks, x, f) if d.ticks else _fit_ticks(d0, d1, math.floor((w - ml - mr) / tick_w), f, x)
    for v in shown:
        out.append(_line(x(v), mt, x(v), h - mb, "grid"))
        out.append(_text(x(v), h - 8, f(v), anchor="middle"))
    top = mt
    for group in groups:
        y0 = top + 10
        top += band(len(group.rows))
        heading = group.short_label if w < 480 and group.short_label else group.label
        # cut to the label column, which the kit leaves to overflow; the whole heading is the group's title
        heading_y = y0 + len(group.rows) * 13 / 2 + 3
        cut = _text(ml - 12, heading_y, _cut(heading, math.floor((ml - 14) / 7.0)), "lbl-strong", "end")
        out.append(_titled([group.label], cut))
        for i, r in enumerate(group.rows):
            cy, color = y0 + i * 13 + 4, _tok(r.color, "--s1")
            marks = _line(
                x(_clamp(r.from_, d0, d1)),
                cy,
                x(_clamp(r.to, d0, d1)),
                cy,
                style=f"stroke:var({color});stroke-width:6;stroke-linecap:round",
            ) + _mark(r.shape, x(_clamp(r.to, d0, d1)), cy, 4, color, r.hollow)
            tip = _tip_or(r.tip, f"{r.label} · {group.label}", f"{f(r.from_)} to {f(r.to)}")
            out.append(_titled(tip, marks, (ml, cy - 6.5, w - ml - mr, 13)))
    if ref is not None:
        rx = x(ref.value)
        out.append(_line(rx, mt - 8, rx, h - mb, "floor"))
        if label is not None:
            out.append(_ref_text(label, mt - 2, True))
    return "".join(out), h


@dataclass(frozen=True)
class _Cell:
    """One cell of a matrix as drawn: its fill, its text colour, its figure, a line under it, its tooltip and
    whether it is outlined (a heatmap cell under its floor)."""

    fill: str
    ink: str
    main: str
    sub: str | None
    tip: list[str]
    low: bool = False


def _matrix(columns: Sequence[str], rows: Sequence[tuple[str, Sequence[_Cell]]], w: float) -> _Drawn:
    """A grid of cells (the kit's ``.mx``, shared by the heatmaps and the confusion matrix): the row labels, a heading
    per column, and each cell a tinted tile with its figure, the line under it where it has one."""
    labels = [label for label, _ in rows]
    lw = min(max([len(label) for label in labels] + [0]) * 6.6 + 12, w * 0.3)
    gap = 3.0
    col_w = (w - lw - gap * len(columns)) / max(len(columns), 1)
    any_sub = any(cell.sub is not None for _, cells in rows for cell in cells)
    cell_h, head_h = (44.0 if any_sub else 34.0), 22.0
    h = head_h + len(rows) * (cell_h + gap)
    room, label_room = math.floor((col_w - 4) / 6.0), math.floor((lw - 8) / 6.6)
    out: list[str] = []
    for j, column in enumerate(columns):
        cx = lw + gap + j * (col_w + gap) + col_w / 2
        out.append(_titled([column], _text(cx, head_h - 7, _cut(column, room), "tick", "middle")))
    for i, (label, cells) in enumerate(rows):
        top = head_h + i * (cell_h + gap)
        out.append(_titled([label], _text(0, top + cell_h / 2 + 4, _cut(label, label_room), "lbl")))
        for j, cell in enumerate(cells):
            left = lw + gap + j * (col_w + gap)
            outline = ";stroke:var(--bad);stroke-width:2" if cell.low else ""
            inset = 1.0 if cell.low else 0.0
            tile = _el(
                "rect",
                {
                    "x": left + inset,
                    "y": top + inset,
                    "width": col_w - 2 * inset,
                    "height": cell_h - 2 * inset,
                    "rx": 3,
                    "style": f"fill:{cell.fill}{outline}",
                },
            )
            main_y = top + (cell_h / 2 - 1 if cell.sub is not None else cell_h / 2 + 5)
            tile += _text(left + col_w / 2, main_y, cell.main, "cell-main", "middle", f"fill:{cell.ink}")
            if cell.sub is not None:
                tile += _text(
                    left + col_w / 2, top + cell_h / 2 + 13, cell.sub, "cell-sub", "middle", f"fill:{cell.ink}"
                )
            out.append(_titled(cell.tip, tile))
    return "".join(out), h


def _count(n: float) -> str:
    """A count as a tooltip prints it (``toLocaleString``): grouped, its fraction to three decimals at most."""
    text = f"{n:,.3f}".rstrip("0").rstrip(".")
    return text or "0"


_HeatLike = HeatmapChart | HeatmapPanel


def _heatmap(d: _HeatLike, w: float) -> _Drawn:
    """A target matrix (``kHeatmap``): six sequential shades over the domain, n per cell, a cell under the floor
    outlined."""
    columns = [column.label for column in _need(d.columns, "columns")]
    (d0, d1), f = _need(d.domain, "domain"), _kit_format(_need(d.fmt, "fmt"))
    nf = _kit_format(d.n_fmt) if d.n_fmt else None
    rows = []
    for row in _need(d.rows, "rows"):
        cells = []
        for j, c in enumerate(row.cells):
            k = int(_clamp(math.floor((c.value - d0) / ((d1 - d0) or 1.0) * 6), 0, 5))
            tip = _tip_or(
                c.tip, f"{row.label} · {columns[j]}", f(c.value) + (f" · n = {_count(c.n)}" if c.n is not None else "")
            )
            sub = nf(c.n) if c.n is not None and nf is not None else None
            low = d.reference is not None and c.value < d.reference.value
            cells.append(_Cell(f"var(--seq-{k})", f"var(--seq-t{k})", f(c.value), sub, tip, low))
        rows.append((row.label, cells))
    return _matrix(columns, rows, w)


def _diff_heatmap(d: DiffHeatmapChart, w: float) -> _Drawn:
    """The difference between two configurations (``kDiffHeatmap``): diverging shades, gray within the neutral
    band, full colour at the saturation point."""
    columns, f = [column.label for column in d.columns], _kit_format(d.fmt)
    df = _kit_format(d.detail_fmt) if d.detail_fmt else None
    neutral, saturate = (0.5 if d.neutral is None else d.neutral), (d.saturate or 8)
    rows = []
    for row in d.rows:
        cells = []
        for j, c in enumerate(row.cells):
            size, flat = min(1.0, abs(c.value) / saturate), abs(c.value) < neutral
            side = "--div-pos" if c.value > 0 else "--div-neg"
            fill = (
                "var(--div-mid)"
                if flat
                else f"color-mix(in oklab, var({side}) {round(18 + size * 72)}%, var(--surface))"
            )
            ink = "var(--on-fill)" if size > 0.55 and not flat else "var(--ink)"
            sub = df(c.detail) if c.detail is not None and df is not None else None
            cells.append(_Cell(fill, ink, f(c.value), sub, _tip_or(c.tip, f"{row.label} · {columns[j]}", f(c.value))))
        rows.append((row.label, cells))
    return _matrix(columns, rows, w)


_ConfusionLike = ConfusionChart | ConfusionPanel


def _confusion(d: _ConfusionLike, w: float) -> _Drawn:
    """Errors between classes (``kConfusion``), each row as shares of 100: the diagonal tinted, an off-diagonal cell
    warned and bad by its share."""
    labels, f = _need(d.labels, "labels"), _kit_format(d.fmt or "pct:1")
    warn_at, bad_at = (1 if d.warn_at is None else d.warn_at), (3 if d.bad_at is None else d.bad_at)
    rows = []
    for i, shares in enumerate(d.shares()):
        cells = []
        for j, v in enumerate(shares):
            fill = (
                "var(--seq-1)"
                if i == j
                else "var(--bad-wash)"
                if v >= bad_at
                else "var(--warn-wash)"
                if v >= warn_at
                else "var(--sunk)"
            )
            cells.append(_Cell(fill, "var(--ink)", f(v), None, [f"{labels[i]} → {labels[j]}", f"{f(v)} of the row"]))
        rows.append((labels[i], cells))
    return _matrix(labels, rows, w)


# =============================================================================
# A chart whole: its legend, its panels, its two widths
# =============================================================================


def _svg(drawn: _Drawn, w: float, css: str, title: str) -> str:
    """One drawing as an ``<svg>`` scaling to its container by its ``viewBox``, named by its first ``<title>`` (text,
    never an attribute, so no field reaches an attribute value)."""
    body, h = drawn
    return (
        f'<svg class="viz-chart {css}" viewBox="0 0 {_n(w)} {_n(math.ceil(h))}" role="img">'
        f"<title>{_escape(title)}</title>{body}</svg>"
    )


def _swatch(item: LegendItem) -> str:
    """A legend entry's mark, as the kit's ``kLegend`` draws it, as a small SVG."""
    color, mark = _tok(item.color, "--ink-2"), item.mark or "dot"
    if item.pattern is not None:
        return (
            f'<svg viewBox="0 0 12 12" width="12" height="12">{_texture(item.pattern, 0.5, 0.5, 11, 11, color)}</svg>'
        )
    if mark in ("dot", "square"):
        style = f"fill:var(--surface);stroke:var({color});stroke-width:2" if item.hollow else f"fill:var({color})"
        shape = (
            _el("rect", {"x": 1, "y": 1, "width": 10, "height": 10, "rx": 2, "style": style})
            if mark == "square"
            else _el("circle", {"cx": 6, "cy": 6, "r": 5, "style": style})
        )
        return f'<svg viewBox="0 0 12 12" width="12" height="12">{shape}</svg>'
    if mark == "line":
        opacity = f"{1 if item.opacity is None else item.opacity:g}"
        line = _line(1, 5, 21, 5, style=f"stroke:var({color});stroke-width:2;opacity:{opacity}")
        return f'<svg viewBox="0 0 22 10" width="22" height="10">{line}</svg>'
    shape_mark: ChartShape = mark  # type: ignore[assignment]  # every other legend mark is a marker shape
    return f'<svg viewBox="0 0 14 14" width="14" height="14">{_mark(shape_mark, 7, 7, 5, color, item.hollow)}</svg>'


def _legend(items: Sequence[LegendItem]) -> str:
    entries = "".join(f"<span>{_swatch(item)}{_escape(item.label)}</span>" for item in items)
    return f'<div class="legend">{entries}</div>'


#: Each drawn kind, to the function that draws one chart (or one panel) of it at a width.
_Drawer = Callable[[Any, float], _Drawn]
_DRAWERS: dict[str, _Drawer] = {
    "bars": _bars,
    "diverging_bars": _bars,
    "forest": _forest,
    "lines": _lines,
    "scatter_frontier": lambda d, w: _scatter(d, w, gate=False),
    "gate_scatter": lambda d, w: _scatter(d, w, gate=True),
    "dot_interval": _dot_interval,
    "range_bars": _range_bars,
    "heatmap": _heatmap,
    "diff_heatmap": _diff_heatmap,
    "confusion": _confusion,
}

#: Each chart with small multiples, to the model one panel is drawn from once it is overlaid on the chart's fields.
_PANEL_MODELS: dict[type, type[BaseModel]] = {
    BarsChart: BarsPanel,
    ScatterChart: ScatterPanel,
    GateScatterChart: GateScatterPanel,
    DotIntervalChart: DotIntervalPanel,
    HeatmapChart: HeatmapPanel,
    ConfusionChart: ConfusionPanel,
}

#: A panel's text fields, and the chart fields no panel overrides: as the kit overlays a panel.
_PANEL_TEXT = frozenset({"title", "subtitle", "tag", "color"})
_CHART_ONLY = frozenset({"legend", "panels", "panel_min_rem"})


def _panel_heading(panel: Any) -> str:
    if panel.tag:
        color = _tok(panel.color, "--ink-2")
        on_tag = "--surface" if color == "--ink-2" else "--on-fill"
        return (
            f'<h4><span class="armchip"><span class="tag" style="background:var({color});color:var({on_tag})">'
            f"{_escape(panel.tag)}</span>{_escape(panel.title or '')}</span></h4>"
        )
    return f"<h4>{_escape(panel.title or '')}</h4>"


def _holds_no_row(chart: Any) -> bool:
    """Whether a chart or a panel lays its marks out by row and was given none (``rows: []``): there is nothing to
    draw, so it is shown as its table, never laid out at a width read off its widest label."""
    return getattr(chart, "rows", None) == []


def _panels(chart: Any, draw: _Drawer, title: str) -> str:
    """Small multiples, as the kit lays them out: each panel the chart's fields overlaid with its own, drawn once at
    a panel's width; a panel of text alone a text card."""
    shared = chart.model_dump(mode="json", exclude_none=True, by_alias=True, exclude=set(_CHART_ONLY))
    model = _PANEL_MODELS[type(chart)]
    width = max(_PANEL_WIDTH, (chart.panel_min_rem or 0) * 16)
    style = (
        f' style="grid-template-columns:repeat(auto-fill,minmax(min(100%,{_n(chart.panel_min_rem)}rem),1fr))"'
        if chart.panel_min_rem is not None
        else ""
    )
    cards = []
    for panel in chart.panels:
        own = panel.model_dump(mode="json", exclude_none=True, by_alias=True, exclude=set(_PANEL_TEXT))
        card = _panel_heading(panel)
        if panel.subtitle:
            card += f'<div class="sub">{_escape(panel.subtitle)}</div>'
        if own:
            one = model.model_validate({**shared, **own})
            named = " · ".join(part for part in (title, panel.title, panel.subtitle) if part)
            if _holds_no_row(one):
                card += f'<p class="chart-note">{_escape(AS_TABLE_NOTE)}</p>'
            else:
                card += f'<div class="chart-panel">{_svg(draw(one, width), width, "viz-panel", named)}</div>'
        cards.append(f'<div class="fig">{card}</div>')
    return f'<div class="mult"{style}>{"".join(cards)}</div>'


def chart_drawing(block: ChartBlock) -> str | None:
    """A chart as a static page draws it: its legend, then its picture as inline SVG at each of three widths
    (or its small multiples, each at a panel's width), in a ``div.chart-drawn``; None for a chart not drawn here (a
    report's interval rows, which the kit does not draw either, the kinds the kit lays out as HTML: a gate matrix, a
    lattice, example cards, and a chart of rows given none), which the page shows as its table and says so. A small
    multiple given no row is a card saying the same.

    :param block: The chart.
    :ptype block: ChartBlock
    :return: The drawing's markup, or None.
    :rtype: str | None
    """
    kit = block.kit_chart()
    if kit is None or kit[0] not in _DRAWERS:
        return None
    kind, data = kit
    chart = KIT_CHARTS[kind].model_validate(data)
    if _holds_no_row(chart):
        return None
    draw = _DRAWERS[kind]
    legend: list[LegendItem] | None = getattr(chart, "legend", None)
    parts = [_legend(legend)] if legend else []
    if getattr(chart, "panels", None) is not None:
        parts.append(_panels(chart, draw, block.title))
    else:
        parts += [_svg(draw(chart, width), width, css, block.title) for css, width in _WIDTHS]
    return f'<div class="chart-drawn" data-kind="{_escape(kind)}">{"".join(parts)}</div>'


__all__ = [
    "AS_TABLE_NOTE",
    "CHART_STYLE",
    "chart_drawing",
]
