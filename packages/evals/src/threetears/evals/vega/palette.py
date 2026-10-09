"""The packaged chart palette, and the Vega-Lite config built from any palette.

**Two palettes reach this module, and one config builder serves both.** A host declares its own
:class:`~threetears.evals.contracts.host.ChartPalette` on its style profile, and a renderer built for that
style draws in it; a host that declares none draws in the palette packaged here
(:func:`packaged_palette`). :func:`vega_config` turns either into the Vega-Lite config — the only place a
palette becomes Vega's vocabulary, so the host contract never names a Vega key. The packaged artifact is
checked with the contract's own colour check
(:func:`~threetears.evals.contracts.host.require_resolved_colour`), so "what a palette may hold" has one
answer for both.

The browser reads the palette straight off the CSS custom properties, so it can
never drift from the design tokens. Python cannot do that, and two facts make the
obvious workarounds wrong:

- **The tokens are authored in OKLCH, and the rasteriser cannot parse it.**
  vl-convert rasterises through resvg; an ``oklch(...)`` string passes into the
  SVG ``fill`` verbatim and renders BLACK. A valid PNG is produced and no warning
  is raised, so the failure is invisible from inside the process that caused it.
  The palette therefore reaches this module as resolved sRGB hex.
- **A design system's build outputs are not a runtime dependency this package may
  have.** Reading the palette out of a stylesheet, or out of whatever directory a
  token build happens to write to, works on a machine that has just run that build
  and fails everywhere else. So the palette arrives here as an artifact that ships
  *inside this package*, beside :mod:`~threetears.evals.vega.text_metrics`'s
  ``font_metrics.json``, and is resolved from this module's own directory.

That is the whole of the contract with whoever supplies the palette: produce
:data:`_PALETTE_PATH` — resolved sRGB hex, the shape the loader checks — and this
module will draw with it. What produces it is named in the artifact's own
``$comment``: a token build converting from the same token sources a browser's CSS
custom properties come from, so the two surfaces share one conversion rather than a
hand-copied second one.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from threetears.evals.contracts.host import ChartFont, ChartPalette, StyleError, require_resolved_colour
from threetears.evals.vega.text_metrics import packaged_font
from threetears.observe import get_logger

log = get_logger(__name__)

Theme = Literal["light", "dark"]

#: The palette artifact, beside the module that reads it — packaged data, resolved
#: from this module's own directory exactly as ``font_metrics.json`` is.
#:
#: This used to walk three levels out of the package and into the repo that
#: generates it, which made an installed copy of this package unable to find its own
#: palette. A package may read what ships with it and nothing else; where the
#: generator lives is the generator's business.
_PALETTE_PATH = Path(__file__).resolve().parent / "chart_palette.json"

#: The Vega-Lite config style a mark asks for to be drawn in the chart's rule ink.
#:
#: Named here rather than in the compiler because both sides of the split need it:
#: the compiler writes the name into a spec and :func:`vega_config` gives that name
#: a colour, which is what lets a zero rule be drawn in something other than the
#: series hue without a colour literal reaching the spec.
ZERO_RULE_STYLE = "chart-zero-rule"

#: The Vega-Lite config style a value label asks for when it is drawn ON a mark's fill.
#:
#: The value-label rule puts a value inside its mark when there is no clearance outside, and that
#: is the longest bar on nearly every figure rather than an edge case. Chart ink over
#: the single-series hue measures 2.53:1 in dark and 3.36:1 in light, against the 4.5:1
#: the design rules require; the knockout — each mode's own chart surface — measures
#: 6.27:1 and 5.66:1. The "two contrast levels, not three" rule was amended
#: to admit this third ink, bounded to this one use.
#:
#: A style rather than a colour in the spec, for the reason every other colour here is:
#: what "knockout" resolves to is a property of the surface being painted, which a
#: compiled spec cannot see.
VALUE_ON_FILL_STYLE = "chart-value-on-fill"

#: The Vega-Lite config style a mark asks for to be drawn as context rather than as
#: the answer — a dominated contestant, a row whose hue has been recycled.
#:
#: Recession is a per-THEME decision and this is what hands it to the renderer. A
#: compiler that writes an opacity into the spec has decided what "receded" looks
#: like on a surface it cannot see, and the same alpha is not the same recession on
#: near-black and on pearl.
#:
#: Colour and no opacity, deliberately. Receding here means *drawn in the neutral
#: that carries no identity*, which is a statement about identity; the low alpha a
#: rug or an overlap band uses is a different thing — there the marks ACCUMULATE, so
#: the alpha is the encoding rather than a way of turning one down.
CONTEXT_STYLE = "chart-context"

#: The Vega-Lite config range name an ordered dimension's levels are drawn from.
#:
#: A **path through colour space, not a set of slots**: the spec names this range and
#: gives a ``linear`` scale over the level's rank, so Vega samples the path at however
#: many steps the data has. That is why an ordered dimension needs no level ceiling
#: while a categorical one does — measured at the pins, an ``ordinal`` scale recycles
#: past the authored tones, which in a sequential encoding draws the darkest level in
#: the lightest one's ink and makes a contiguous block read as though it wrapped.
SEQUENTIAL_RANGE = "chart-seq"


class PaletteError(RuntimeError):
    """The chart palette artifact is missing or cannot be drawn with."""


@lru_cache(maxsize=1)
def load_palette() -> dict[str, Any]:
    """Load and check the generated palette artifact.

    Returns:
        The parsed artifact.

    Raises:
        PaletteError: The artifact is absent, or holds a colour that is not resolved
            sRGB hex — refused by the contract's colour check, since the server-side
            rasteriser renders another notation (oklch) as black.
    """
    try:
        raw = _PALETTE_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise PaletteError(
            f"chart palette artifact missing at {_PALETTE_PATH} — it is packaged data and ships beside this "
            "module, so an installation without it is incomplete. The tool that produces it is named in the "
            "artifact's own `$comment`."
        ) from exc
    palette: dict[str, Any] = json.loads(raw)
    return check_palette_artifact(palette)


def check_palette_artifact(palette: dict[str, Any]) -> dict[str, Any]:
    """Hold every colour in a palette artifact to the contract's one colour check.

    Args:
        palette: A parsed palette artifact.

    Returns:
        ``palette``, unchanged.

    Raises:
        PaletteError: A colour is not resolved sRGB hex
            (:func:`~threetears.evals.contracts.host.require_resolved_colour`).
    """
    for path, colour in _colour_values(palette):
        try:
            require_resolved_colour(f"packaged chart palette value {path}", colour)
        except StyleError as unresolved:
            raise PaletteError(str(unresolved)) from unresolved
    return palette


def _colour_values(palette: dict[str, Any]) -> list[tuple[str, str]]:
    """Yield ``(path, value)`` for every colour in the artifact.

    Walks the parsed values rather than the file text: the artifact's own
    ``$comment`` explains the OKLCH hazard by name, and a text scan flags that
    explanation as the defect it warns about.
    """
    found: list[tuple[str, str]] = []
    for mode, entry in palette.items():
        if not isinstance(entry, dict):
            continue
        for key, value in entry.items():
            if isinstance(value, str):
                found.append((f"{mode}.{key}", value))
            elif isinstance(value, list):
                found.extend(
                    (f"{mode}.{key}[{index}]", item) for index, item in enumerate(value) if isinstance(item, str)
                )
    return found


def validated_slots() -> int:
    """How many categorical palette slots may be assigned before falling back."""
    return int(load_palette()["validated_slots"])


def series_slots() -> int:
    """How wide the categorical palette is — where it starts recycling, not where it stops being validated.

    The artifact carries this because the token build COUNTS it off the token tree
    rather than authoring it. Read
    it rather than measuring ``len(series_colors(theme))``: both give the same
    answer today, and only this one still gives it if a mode's block is ever built
    short.
    """
    return int(load_palette()["series_slots"])


def geometry() -> dict[str, Any]:
    """The figure's fixed dimensions, in px.

    Sizes travel with the palette for the same reason the hues do: the report is
    drawn by two renderers in two languages, and a number written twice is a number
    that will disagree. The block is authored in the design tokens' chart group
    and reaches both sides through the generated artifact.

    Returns:
        The geometry block. Keys are stable names (``plot_width``, ``row_step``,
        …); callers read the ones they need rather than unpacking a fixed shape,
        so a value added for one chart type does not break the others.
    """
    return dict(load_palette()["geometry"])


def font_sizes() -> dict[str, float]:
    """The chart type scale, in px, already resolved from rem by the token build.

    Read by the compiler as well as by :func:`vega_config`, because a layout
    decision taken from a string's width is only right if it measures at the size
    the renderer will draw that string at. The two uses are held together by
    ``tests/test_vega_compiler.py``, which pins that the size the compiler
    measures a category label at is the size this config hands the axis — the pair
    is otherwise free to drift silently, and the symptom would be a label that was
    measured to fit and draws truncated.

    Returns:
        The ``font_size`` block. Keys are the scale's own names (``title``,
        ``tick``, ``label``, …); callers read the ones they need.
    """
    return dict(load_palette()["font_size"])


def font_weights() -> dict[str, float]:
    """The chart type scale's weights, one per step of :func:`font_sizes`.

    Read beside the sizes rather than left to the config's blanket defaults,
    because the two steps a chart axis draws at are not the same step: a numeric
    tick and a category name have the same size today and different weights, and
    an axis given one step's size with the other's weight is drawing neither.

    Returns:
        The ``font_weight`` block, keyed by the same step names as
        :func:`font_sizes`.
    """
    return dict(load_palette()["font_weight"])


def series_colors(theme: Theme) -> list[str]:
    """The categorical hues, in fixed order — the whole palette, not the validated prefix.

    Slots 1-4 pass categorical validation and 5-8 are the derived second tier. The
    palette does not refuse to draw past the validated slots: the number of series is a
    property of the data, and refusing
    legitimate data is never the right answer. Past eight Vega-Lite recycles the
    range, which is admitted rather than prevented — and survivable only because a
    direct label on every mark is mandatory from five series up, so identity has left
    the hue channel before the hue channel weakens.
    """
    return list(load_palette()[theme]["chart"])


def sequential_colors(theme: Theme) -> list[str]:
    """The ordered ramp's authored stops, lightest first.

    Handed to the renderer as a range rather than read by the compiler: a spec that
    picked a stop would be stating a colour, and the stop count would then become a
    ceiling on how many levels a dimension may have. As a range it is a path Vega
    samples, so five stops draw seven levels as seven distinct tones.
    """
    return list(load_palette()[theme]["seq"])


def packaged_palette(theme: Theme) -> ChartPalette:
    """The palette packaged with this renderer, in one of its two variants.

    What a renderer draws in when its host declares no palette of its own — the host's stated choice,
    since declaring one is a field on its style profile.

    Args:
        theme: Which variant.

    Returns:
        The palette, held to the contract like any host's.

    Raises:
        PaletteError: The artifact is absent or does not make a palette.
    """
    mode = load_palette()[theme]
    try:
        return ChartPalette(
            series=tuple(mode["chart"]),
            sequential=tuple(mode["seq"]),
            background=mode["surface"],
            ink=mode["ink"],
            muted=mode["muted"],
            grid=mode["grid"],
            rule=mode["rule"],
            context=mode["context"],
            on_fill=mode["on_fill"],
        )
    except StyleError as refused:
        raise PaletteError(f"the packaged {theme} palette is not a palette: {refused}") from refused


def vega_config(palette: ChartPalette, font: ChartFont | None = None) -> dict[str, Any]:
    """Build the Vega-Lite config that themes a compiled spec in ``palette``, set in ``font``.

    The spec itself carries no colour, so this is the whole of a chart's
    appearance on the server side — and the mirror of what the browser assembles
    from its CSS custom properties.

    The two must build the same KEYS; their values differ by construction, resolved
    hex here against a custom property there, the surface colour here against
    ``transparent`` there. A browser renderer's parity test should compare the key
    structures at every depth — reading this function's returned dict literal out of
    the source, since it cannot execute Python. Add a key on one side alone and that
    test is what fails; both renderers ignore a key the other honours without
    complaining, so the symptom otherwise is only that the exported PNG stops looking
    like the report.

    The colours are the palette's — a host's or :func:`packaged_palette` — and the type scale and
    weights are this renderer's own, from the packaged artifact. The typeface is ``font``'s family list:
    the face whose measured advances the spec's layout was computed from, which is why a typeface
    arrives here only as a :class:`~threetears.evals.contracts.host.ChartFont` and never as a bare name.
    Pass the same font the spec was compiled with.

    Args:
        palette: The colours to draw with.
        font: The typeface the spec was laid out in; ``None`` for the packaged face
            (:func:`~threetears.evals.vega.text_metrics.packaged_font`).

    Returns:
        A Vega-Lite ``config`` object.
    """
    artifact = load_palette()
    font_family = (font if font is not None else packaged_font()).family
    # The chart type scale and weights, resolved to px by the token build. Charts read
    # their OWN scale rather than borrowing the page's: they previously took `font.size.xs`
    # for labels and `font.size.sm` for titles — the two steps the design system defines
    # for captions and helper text — because those were the smallest available, not because
    # a chart element belongs there. A tick and the value written on a mark are where a chart
    # states its numbers, and neither of those is helper text.
    sizes, weights = artifact["font_size"], artifact["font_weight"]
    # Three ink levels. `chart.fg` carries everything except footnotes; `fg-2`/`fg-3` cap
    # around Lc 59 in dark by design, which is right for a caption and wrong for a value.
    # `on_fill` is the third, admitted by an amendment to the two-contrast-levels rule and
    # bounded to a value drawn over a mark — see `VALUE_ON_FILL_STYLE`.
    ink, muted, grid, rule, context = palette.ink, palette.muted, palette.grid, palette.rule, palette.context
    on_fill = palette.on_fill
    single = palette.series[0]
    return {
        "background": palette.background,
        "font": font_family,
        # The default single-series mark colour. A chart whose identity channel is
        # its axis labels rather than its hues draws entirely in this one.
        #
        # Per-mark entries are NOT redundant with `mark`: `config.mark.color` does
        # not reach a mark whose colour rides on `stroke`, so an interval drawn as
        # a `rule` renders in Vega's own default — pure black, which on an obsidian
        # surface is an interval the reader cannot see. Established by rendering.
        "mark": {"color": single},
        "bar": {"color": single},
        "rule": {"color": single},
        "line": {"color": single},
        "point": {"color": single},
        # Two ranges, and the difference between them is why only one needs a ceiling.
        # `category` is a SET OF SLOTS: past its width Vega recycles, so the ninth
        # series takes the first's hue and two different things look identical — which
        # is survivable only behind the direct-label mandate. `chart-seq` is a PATH:
        # a linear scale over a level's rank samples it at however many steps the data
        # has, so levels close up but never collide. A categorical ramp recycles; a
        # sequential ramp resamples.
        # `"chart-seq"` written as a literal rather than as `SEQUENTIAL_RANGE`, for the
        # same reason `"chart-zero-rule"` is below: a browser mirror's parity check that
        # reads THIS dict out of the source can only see a string key.
        "range": {"category": list(palette.series), "chart-seq": list(palette.sequential)},
        "title": {
            "color": ink,
            "subtitleColor": muted,
            "font": font_family,
            "subtitleFont": font_family,
            "fontWeight": weights["title"],
            "anchor": "start",
            "fontSize": sizes["title"],
            "subtitleFontSize": sizes["footnote"],
            "subtitleFontWeight": weights["footnote"],
        },
        # Quiet axes, but not faint ones. The grid recedes; the labels do not — a tick
        # label states a value, so it takes full chart ink at the weight floor.
        #
        # `labelFontSize`/`labelFontWeight` are the step for a NUMERIC tick, which is
        # what an axis carrying no statement of its own is most likely to be. An axis
        # whose labels are category names states the name step on itself, because the
        # two steps are different roles that happen to share a size — a config keyed by
        # channel cannot tell them apart, since one figure here draws names on y and
        # bin ranges on x.
        #
        # `grid: False` is the default in BOTH directions, and the compiler opts an
        # axis in. Gridlines are a per-encoding rule rather than a per-chart choice:
        # a bar has a shared baseline and its value written on it, so a grid is lines
        # the reader does not need, while an interval has no baseline to read against.
        "axis": {
            "labelColor": ink,
            "titleColor": ink,
            "labelFont": font_family,
            "titleFont": font_family,
            "labelFontSize": sizes["tick"],
            "labelFontWeight": weights["tick"],
            "titleFontSize": sizes["label"],
            "titleFontWeight": weights["label"],
            "gridColor": grid,
            "domainColor": rule,
            "tickColor": rule,
            "grid": False,
        },
        # A small-multiple's row header names the same categories the value panel's
        # axis does, so it takes the scale's own step for that. Vega's default is 11px
        # — four steps below the type scale's floor — which drew one figure's category
        # names at two sizes depending on which panel they sat in.
        "header": {
            "labelColor": ink,
            "titleColor": ink,
            "labelFont": font_family,
            "titleFont": font_family,
            "labelFontSize": sizes["facet-label"],
            "labelFontWeight": weights["facet-label"],
        },
        "legend": {
            "labelColor": ink,
            "titleColor": ink,
            "labelFont": font_family,
            "titleFont": font_family,
            "labelFontSize": sizes["label"],
            "labelFontWeight": weights["label"],
            "titleFontSize": sizes["label"],
            "titleFontWeight": weights["label"],
            # A legend is a fallback, and the one place it still earns its keep is a
            # faceted figure where one key serves every panel. Where it does survive,
            # it goes under the title and reads across: Vega's own default is a
            # right-side vertical key, which takes its width out of the plot and pushes
            # the figure past the column its geometry is fixed to. Stated here rather
            # than left to the producer, because a generator-emitted spec that never
            # mentions orientation is exactly the one that would take the default.
            #
            # `top`, NOT `top-left`, and the difference is not cosmetic — measured
            # rather than read: Vega's corner orients place a legend INSIDE the data
            # rectangle, so `top-left` draws the key on top of the marks (18px in from
            # the plot's corner, occluding whatever is there). `top` places it above
            # the plot, left-aligned to the plot's own left edge, which is what "top-left
            # under the title" describes.
            "orient": "top",
            # The key's symbols are drawn in the MUTED ink, never the series hue. A key
            # that survives here is a SHAPE key — a circle against a diamond — and Vega
            # fills its symbols with the default series colour, which told the reader
            # both classes share one ink while the plot drew the beaten one in the
            # context neutral. Neutral symbols make the key say what it is keying: the
            # glyph. Colour on the marks stays the marks' to carry.
            "symbolFillColor": muted,
            "symbolStrokeColor": muted,
            # Clear of the y axis title, which is drawn FLAT above the plot on its own
            # line and anchored to the same left edge — so at Vega's default offset the
            # key and the quantity's name land in one band and read as noise. The two
            # are stacked instead: legend above, title under it, plot under that.
            # Measured against the title's own placement (12px above the plot, at the
            # label size) rather than picked, so a type-scale change moves both.
            "offset": 36,
            "direction": "horizontal",
        },
        "view": {"stroke": None},
        # Values written on a mark, and words standing in for one. These are the numbers
        # the reader lands on first, so they sit a step above ticks and never in muted ink.
        "text": {"color": ink, "font": font_family, "fontSize": sizes["value"], "fontWeight": weights["value"]},
        # A named style is how a spec asks for the RULE ink without carrying a colour:
        # a zero line drawn in the series hue reads as data, and the spec may not say
        # which hex that is. The name travels in the spec; the value stays here, which
        # is the same division every other colour in this file is on.
        #
        # Written as a literal rather than as `ZERO_RULE_STYLE` because a browser
        # mirror's parity check that reads THIS dict out of the source cannot resolve a
        # name; `tests/test_vega_compiler.py` pins the literal to the constant instead, so
        # the pair is still held by a gate rather than by memory.
        #
        # `chart-context` is the same division applied to RECESSION. A mark that is
        # present for comparison rather than as the answer asks for it by name, and the
        # renderer decides what receding looks like on the surface it is drawing on —
        # a compiled opacity would be the same alpha on near-black and on pearl, which
        # is the right recession on at most one of them. Colour and no opacity: this
        # says "carries no identity", not "turned down", and the low alpha a rug or an
        # overlap band uses is the other thing.
        #
        # `chart-value-on-fill` is the third of the same kind, and the one that is a
        # CONTRAST decision rather than an identity one: a value drawn on a mark takes the
        # knockout, which is each mode's own chart surface. Bounded to the single-series
        # mark colour — the same ink over a tier-2 slot inverts (dark knockout over slot 5
        # is 2.99:1), so nothing may reach for this style on a mark whose fill came from
        # the categorical range.
        "style": {
            "chart-zero-rule": {"color": rule},
            "chart-context": {"color": context},
            "chart-value-on-fill": {"color": on_fill},
        },
    }


__all__ = [
    "CONTEXT_STYLE",
    "SEQUENTIAL_RANGE",
    "VALUE_ON_FILL_STYLE",
    "ZERO_RULE_STYLE",
    "PaletteError",
    "Theme",
    "check_palette_artifact",
    "font_sizes",
    "font_weights",
    "geometry",
    "load_palette",
    "packaged_palette",
    "sequential_colors",
    "series_colors",
    "series_slots",
    "validated_slots",
    "vega_config",
]
