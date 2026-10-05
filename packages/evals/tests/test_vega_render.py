"""The server-side render — checked by looking at the pixels it actually drew.

**"A PNG was produced" cannot gate this file.** The rasteriser behind
``vl-convert`` is resvg, which does not parse ``oklch()`` — and the design tokens
these charts are painted from are authored in OKLCH. Handed an unconverted
palette it emits a perfectly valid PNG, of the right size, with no warning and
every series black. Every well-formedness assertion passes on exactly the output
this file exists to prevent, so the only check that means anything is what colour
reached the pixels.

:func:`dominant_colours` therefore decodes the PNG rather than trusting it, and
:func:`png_size` reads back the width — a chart drawn at the rasteriser's default
size carries exactly the same colours in the same proportions as one drawn at the
width this module asked for. Both decoders are deliberately dependency-free: the
venv carries no image library, and adding one so a test can read four bytes would
be a dependency bought for a single assertion.

**No typeface is registered here.** The renderer owns none and takes a directory from
its host, so every chart in this module draws in whatever face the machine resolves.
The checks that need a particular host's face -- that it is the one registered, and
that text laid out against its metrics stays inside its column -- belong to the host
that supplies it.
"""

import json
import math
import re
import struct
import zlib
from collections import Counter, defaultdict
from collections.abc import Iterator
from typing import Any

import pytest

from threetears.evals.vega import compile_chart
from threetears.evals.vega.compiler import point_radius
from threetears.evals.vega.palette import geometry, load_palette, series_slots, vega_config
from threetears.evals.analysis.viz.payloads import PAYLOAD_MODELS
from threetears.evals.vega.render import render_png, render_svg


PAYLOAD = {
    "parts": [
        {"label": "budget_exhausted", "value": 42.0, "n": 21},
        {"label": "no_new_sources", "value": 27.0, "n": 13},
        {"label": "confidence_met", "value": 19.0, "n": 9},
        {"label": "tool_error", "value": 12.0, "n": 6},
    ],
    "unit": "%",
    "measure": "share of stops",
    "total": 100.0,
    "total_n": 49,
}

#: One payload per drawable SHAPE, so the pixel checks cover every chart this build
#: can produce rather than the first one it learned to. Keyed by a shape name and
#: carrying its viz type, because a type is not a shape: a distribution that has
#: values to place concatenates a value axis above its counts, while a pre-binned
#: one draws the counts alone and reaches the rasteriser as a bare facet. Type
#: coverage — free while the keys were the types — is kept by
#: `TestPngPixels.test_every_compiled_type_is_drawn_here`.
#:
#: The distinction is not academic. While this file drew only a bar chart, every
#: assertion in it passed against a renderer that was painting interval marks pure
#: black — `config.mark.color` does not reach a mark whose colour rides on
#: `stroke`, so the rule marks in a distribution or a null result were invisible on
#: the obsidian surface and nothing said so.
PAYLOADS: dict[str, tuple[str, dict]] = {
    "breakdown": ("breakdown", PAYLOAD),
    "distribution_with_values": (
        "distribution",
        {
            "groups": [
                {
                    "label": "model-b",
                    "samples": [1200.0, 1450.0, 1310.0, 1600.0],
                    "ci": {
                        "low": 1250.0,
                        "high": 1520.0,
                        "mean": 1390.0,
                        "variability": "across 5 runs",
                        "level": 0.95,
                    },
                    "n": 5,
                },
                {
                    "label": "deepseek",
                    "buckets": [{"range": "0-1s", "count": 2}, {"range": "1-2s", "count": 7}],
                    "ci": {
                        "low": 1400.0,
                        "high": 1900.0,
                        "mean": 1650.0,
                        "variability": "across 5 runs",
                        "level": 0.95,
                    },
                    "n": 9,
                },
            ],
            "unit": "ms",
            "x_label": "pipeline_synthesis_ms",
        },
    ),
    #: Every group pre-binned and no group carrying an interval or samples — the shape
    #: the generator is told to emit at larger n, and the one that used to compile to a
    #: blank frame above the counts. It is the only payload here whose spec is a bare
    #: `facet`, i.e. whose width lives on a nested view and nowhere else, so it is also
    #: what keeps the every-depth size assertions from passing on the top level alone.
    "distribution_buckets_only": (
        "distribution",
        {
            "groups": [
                {
                    "label": "model-b",
                    "buckets": [
                        {"range": "0-1s", "count": 3},
                        {"range": "1-2s", "count": 11},
                        {"range": "2-4s", "count": 6},
                    ],
                    "n": 20,
                },
                {
                    "label": "deepseek",
                    "buckets": [
                        {"range": "0-1s", "count": 9},
                        {"range": "1-2s", "count": 8},
                        {"range": "2-4s", "count": 3},
                    ],
                    "n": 20,
                },
            ],
            "unit": "ms",
            "x_label": "pipeline_synthesis_ms",
        },
    ),
    "null_result": (
        "null_result",
        {
            "groups": [
                {
                    "label": "timeout=4s",
                    "ci": {
                        "low": 0.71,
                        "high": 0.85,
                        "mean": 0.78,
                        "variability": "across the 12 cases",
                        "level": 0.95,
                    },
                    "n": 12,
                },
                {
                    "label": "timeout=8s",
                    "ci": {
                        "low": 0.74,
                        "high": 0.88,
                        "mean": 0.81,
                        "variability": "across the 12 cases",
                        "level": 0.95,
                    },
                    "n": 12,
                },
            ],
            "metric": "mean_composite",
            "mechanism": "The batch never fills before the deadline at either setting.",
        },
    ),
    "delta_table": (
        "delta_table",
        {
            "rows": [
                {
                    "metric": "cost_usd",
                    "a": 0.011,
                    "b": 0.019,
                    "unit": "usd",
                    "p": 0.004,
                    "d_z": 1.2,
                    "n": 24,
                    "significant": True,
                },
                {"metric": "total_ms", "a": 16162.0, "b": 11040.0, "unit": "ms"},
            ],
            "a_label": "model-b",
            "b_label": "deepseek",
        },
    ),
    #: The only point plot, and the only spec whose marks carry a `shape` scale — so
    #: this is what proves the symbols survive rasterisation. A dominated contestant
    #: has to reach the PNG as a different glyph and not merely as a fainter one:
    #: opacity is the redundant channel here, and a reader who cannot separate two
    #: weights is exactly the reader the shape exists for.
    "frontier": (
        "frontier",
        {
            "points": [
                {"label": "model-a-3.5-fast-lite", "cost": 0.0071, "quality": 0.2, "latency_ms": 31000.0},
                {"label": "model-b", "cost": 0.0174, "quality": 0.0, "latency_ms": 48700.0, "dominated": True},
            ],
            "bar": 0.5,
            "cost_label": "Cost per run (USD)",
            "quality_label": "pass^k",
        },
    ),
    #: The one shape drawing TWO ink vocabularies in one figure — an ordered lever
    #: sampled off the sequential ramp and a categorical one taking the validated
    #: hues — which is why it is rasterised rather than only compiled: the two reach
    #: the renderer through different config entries (`range.chart-seq` against
    #: `range.category`), and a spec that named either one wrongly would compile
    #: cleanly and draw a barcode in a single flat colour.
    "sweep_ranking": (
        "sweep_ranking",
        {
            "ranked": {"measure": "pass^k", "unit": None},
            "secondary": {"measure": "cost per run", "unit": "usd"},
            "rows": [
                {
                    "config": {"model": "gpt-5", "fetch_concurrency": "8"},
                    "ranked_value": 0.72,
                    "secondary_value": 0.0111,
                },
                {
                    "config": {"model": "model-b", "fetch_concurrency": "4"},
                    "ranked_value": 0.61,
                    "secondary_value": 0.0094,
                },
                {
                    "config": {"model": "gpt-5", "fetch_concurrency": "2"},
                    "ranked_value": 0.55,
                    "secondary_value": 0.0142,
                },
            ],
        },
    ),
    #: The two attribution cases draw a DIFFERENT NUMBER OF MARKS, which is the whole
    #: behaviour of the type: a remainder the catalog's containment declaration earns
    #: is drawn, and one it does not is a labelled row with no bar. Rasterising only
    #: the earned case would leave the withheld one — the shape most stored analyses
    #: actually carry — never once drawn.
    "attribution_withheld": (
        "attribution",
        {
            "end_to_end": {"measure": "total_ms", "delta": 3000.0, "a": 44000.0, "b": 47000.0},
            "subsystem": {"measure": "pipeline_synthesis_ms", "delta": 87800.0, "a": 5200.0, "b": 93000.0},
            "unit": "ms",
            "unattributed_withheld": (
                "pipeline_synthesis_ms is not declared a component of total_ms — they share a unit but are not "
                "known to be nested, so what looks like an unexplained remainder may be two disjoint stretches "
                "of the same measure."
            ),
            "lever": "pipeline.pipeline_model",
            "a_label": "model-a-3.5-fast-lite",
            "b_label": "model-b",
        },
    ),
    "attribution_earned": (
        "attribution",
        {
            "end_to_end": {"measure": "total_ms", "delta": -31800.0, "a": 58300.0, "b": 26500.0},
            "subsystem": {"measure": "tool_ms", "delta": -3800.0},
            "unit": "ms",
            "contained_by": "total_ms",
            "unattributed_delta": -28000.0,
            "lever": "prompt.pipeline_system",
            "a_label": "terse_prompt",
            "b_label": "verbose_prompt",
        },
    ),
    "timeseries": (
        "timeseries",
        {
            "metric": "rules_accuracy",
            "unit": None,
            "basis": "date",
            "positions": ["2026-10-01", "2026-10-02", "2026-10-03"],
            "series": [
                {
                    "label": "model-a",
                    "points": [
                        {
                            "position": day,
                            "ci": {
                                "low": m - 0.05,
                                "high": m + 0.05,
                                "mean": m,
                                "level": 0.95,
                                "variability": "the cell's observations",
                            },
                            "n": 10,
                        }
                        for day, m in (("2026-10-01", 0.7), ("2026-10-02", 0.74), ("2026-10-03", 0.81))
                    ],
                },
                {
                    "label": "model-b",
                    "points": [
                        {
                            "position": day,
                            "ci": {
                                "low": m - 0.05,
                                "high": m + 0.05,
                                "mean": m,
                                "level": 0.95,
                                "variability": "the cell's observations",
                            },
                            "n": 10,
                        }
                        for day, m in (("2026-10-01", 0.6), ("2026-10-03", 0.66))
                    ],
                },
            ],
            "gaps": [{"series": "model-b", "position": "2026-10-02", "reason": "the cell was not measured there"}],
        },
    ),
}


def _walk_dicts(node: Any) -> Iterator[dict[str, Any]]:
    """Every mapping anywhere in a compiled spec, so a nested layer's mark is found wherever it sits."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk_dicts(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk_dicts(value)


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    """Parse ``#rrggbb`` into a byte triple."""
    return tuple(int(value[index : index + 2], 16) for index in (1, 3, 5))


def _relative_luminance(colour: str) -> float:
    """WCAG 2.x relative luminance of an ``#rrggbb`` colour.

    Written out rather than pulled from a dependency because the artifact this
    measures is already 8-bit sRGB — the whole conversion argument was settled when
    it was generated — and the formula is the definition of the number the design
    rules are stated in.
    """
    channels = []
    for byte in _hex_to_rgb(colour):
        channel = byte / 255
        channels.append(channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4)
    red, green, blue = channels
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _contrast_ratio(foreground: str, background: str) -> float:
    """WCAG contrast between two ``#rrggbb`` colours, lighter over darker."""
    first, second = _relative_luminance(foreground), _relative_luminance(background)
    lighter, darker = max(first, second), min(first, second)
    return (lighter + 0.05) / (darker + 0.05)


def png_size(png: bytes) -> tuple[int, int]:
    """Read a PNG's pixel dimensions out of its IHDR chunk.

    The one property that separates a chart drawn at the width this module asked
    for from one drawn at the rasteriser's default: both carry the same colours in
    the same proportions, so nothing else in this file can tell them apart.

    Args:
        png: The PNG bytes.

    Returns:
        ``(width, height)`` in pixels.
    """
    assert png[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    return struct.unpack(">II", png[16:24])


def pixel_rows(png: bytes) -> list[list[tuple[int, int, int]]]:
    """Decode a PNG into its RGB pixels, row by row from the top.

    The same decoder :func:`dominant_colours` runs, stopping one step earlier: a
    histogram answers *what* was drawn and this answers *where*, which is what a
    check on two marks overlapping needs. One decoder rather than two, because a
    second copy of PNG's per-scanline filters is a second chance to get them wrong.

    Args:
        png: The PNG bytes.

    Returns:
        One list per scanline, each holding ``(r, g, b)`` per pixel — alpha
        dropped, since vl-convert paints the configured surface opaque.
    """
    assert png[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    return _decode(png)


def dominant_colours(png: bytes, limit: int | None = 4) -> list[tuple[tuple[int, int, int], int]]:
    """Decode a PNG and return its most common RGB values, most frequent first.

    Handles the 8-bit non-interlaced output ``vl-convert`` produces, including
    the per-scanline filters PNG always applies.

    ``limit=None`` returns every colour, which is what a caller asking *whether a
    specific colour was drawn* needs: a truncated histogram answers "is it among
    the commonest N", and those are different questions on a chart whose marks are
    small. A point plot's two symbols cover ~226px against a figure of anti-aliased
    text, which put the series colour at rank 12 in one theme and 13 in the other —
    the same chart reading as painted and as unpainted on a tie-break.

    Args:
        png: The PNG bytes.
        limit: How many distinct colours to return, or ``None`` for all of them.

    Returns:
        ``(rgb, pixel_count)`` pairs, most frequent first.
    """
    assert png[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    counts: Counter = Counter()
    for row in _decode(png):
        counts.update(row)
    return counts.most_common(limit)


def _decode(png: bytes) -> list[list[tuple[int, int, int]]]:
    """Undo PNG's container and per-scanline filters, dependency-free.

    Args:
        png: The PNG bytes, 8-bit and non-interlaced — what ``vl-convert`` emits.

    Returns:
        One list of ``(r, g, b)`` per scanline, top to bottom.
    """
    position, compressed = 8, bytearray()
    width = height = channels = 0
    while position < len(png):
        (length,) = struct.unpack(">I", png[position : position + 4])
        chunk = png[position + 4 : position + 8]
        body = png[position + 8 : position + 8 + length]
        if chunk == b"IHDR":
            width, height, depth, colour_type, _, _, interlace = struct.unpack(">IIBBBBB", body)
            assert depth == 8 and interlace == 0, f"unexpected PNG encoding: depth={depth} interlace={interlace}"
            channels = {0: 1, 2: 3, 4: 2, 6: 4}[colour_type]
            # Stated because the rows below read three channels per pixel: a greyscale
            # encoding would hand back a chart whose every colour check compares
            # brightness to a hex, which passes and means nothing.
            assert channels >= 3, f"the render is not colour: colour_type={colour_type}"
        elif chunk == b"IDAT":
            compressed += body
        elif chunk == b"IEND":
            break
        position += 12 + length

    raw = zlib.decompress(bytes(compressed))
    stride = width * channels
    rows: list[list[tuple[int, int, int]]] = []
    previous = bytearray(stride)
    cursor = 0
    for _ in range(height):
        filter_type = raw[cursor]
        cursor += 1
        line = bytearray(raw[cursor : cursor + stride])
        cursor += stride
        for index in range(stride):
            left = line[index - channels] if index >= channels else 0
            up = previous[index]
            up_left = previous[index - channels] if index >= channels else 0
            if filter_type == 1:
                line[index] = (line[index] + left) & 0xFF
            elif filter_type == 2:
                line[index] = (line[index] + up) & 0xFF
            elif filter_type == 3:
                line[index] = (line[index] + (left + up) // 2) & 0xFF
            elif filter_type == 4:
                predictor = left + up - up_left
                deltas = (abs(predictor - left), abs(predictor - up), abs(predictor - up_left))
                nearest = (
                    left
                    if deltas[0] <= deltas[1] and deltas[0] <= deltas[2]
                    else (up if deltas[1] <= deltas[2] else up_left)
                )
                line[index] = (line[index] + nearest) & 0xFF
        rows.append([(line[index], line[index + 1], line[index + 2]) for index in range(0, stride, channels)])
        previous = line
    return rows


@pytest.fixture(scope="module")
def spec():
    """The compiled breakdown spec every render in this module draws."""
    return compile_chart("breakdown", PAYLOAD).spec


def _oklch(hex_colour: str) -> tuple[float, float, float]:
    """An sRGB hex as OKLCH (L, C, hue-degrees).

    The inverse of the conversion the token build applies when it generates the
    artifact. Written here rather than imported because nothing in the app needs to go this
    direction — only a test asserting that one palette tier is DERIVED from another does.
    """
    channels = [int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    r, g, b = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    lms = (
        (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3),
        (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3),
        (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3),
    )
    lightness = 0.2104542553 * lms[0] + 0.7936177850 * lms[1] - 0.0040720468 * lms[2]
    a = 1.9779984951 * lms[0] - 2.4285922050 * lms[1] + 0.4505937099 * lms[2]
    b_ = 0.0259040371 * lms[0] + 0.7827717662 * lms[1] - 0.8086757660 * lms[2]
    return lightness, math.hypot(a, b_), math.degrees(math.atan2(b_, a)) % 360


class TestPaletteArtifact:
    """The artifact the renderer's colour comes from."""

    def test_no_unrenderable_colour_notation_survives_into_the_artifact(self):
        """The defect one layer earlier than the pixels, named more clearly.

        `load_palette` refuses an artifact holding OKLCH, so reaching this
        assertion at all is most of the check; the explicit scan states what is
        being guaranteed rather than leaving it to a loader's side effect.
        """
        palette = load_palette()
        for mode in ("light", "dark"):
            for colour in [*palette[mode]["chart"], palette[mode]["surface"], palette[mode]["ink"]]:
                assert colour.startswith("#") and len(colour) == 7, f"{mode}: {colour!r} is not resolved sRGB hex"

    def test_the_generated_hexes_match_the_validation_record(self):
        """The palette's colourblind/contrast results only describe THESE values.

        Hard-coded on purpose: this is the one place the conversion's output is pinned to
        the record that measured it, so a token edit that changes a drawn colour has to be
        re-validated rather than silently shipped. Slots 1-4 are the validated categorical
        hues in v5.2 order (velvet, azure, champagne, teal); 5-8 are the derived second
        tier, which is NOT claimed to pass validation — that is what makes it tier 2.
        """
        palette = load_palette()
        assert palette["light"]["chart"] == [
            "#8625fe",
            "#00a2c4",
            "#c79100",
            "#00ad8a",
            "#451686",
            "#005d6f",
            "#785b20",
            "#006652",
        ]
        assert palette["dark"]["chart"] == [
            "#ac79ff",
            "#00c1df",
            "#eaab05",
            "#00c7a6",
            "#6a50a5",
            "#007989",
            "#977430",
            "#2d7f6c",
        ]

    def test_tier_two_is_derived_from_tier_one_by_the_stated_rule(self):
        """Slots 5-8 are a function of 1-4, not four more hand-picked hues.

        The authored rule is chroma x0.6 and lightness -0.20 in OKLCH, per hue. What is
        asserted here is deliberately weaker than that arithmetic, because the artifact
        carries 8-bit sRGB and several of these colours are outside the sRGB gamut at their
        authored chroma — dark azure is authored at C 0.18 and resolves to C 0.131 once
        clipped. Re-deriving the exact triple from the hex would therefore fail on values
        that are perfectly correct, and pinning the post-clip numbers instead would pin the
        clipping rather than the rule.

        So: same hue, materially darker, materially duller. That still fails the thing this
        guards — a tier-2 slot quietly replaced by a colour someone preferred, which is how
        the second tier would stop being derivable and become a second palette nobody
        validated.
        """
        palette = load_palette()
        for theme in ("light", "dark"):
            hues = palette[theme]["chart"]
            for index in range(4):
                base_l, base_c, base_h = _oklch(hues[index])
                twin_l, twin_c, twin_h = _oklch(hues[index + 4])
                slot = f"{theme} slot {index + 5}"
                assert twin_l < base_l - 0.12, f"{slot}: tier 2 must be materially darker than its twin"
                assert twin_c < base_c * 0.8, f"{slot}: tier 2 must be materially duller than its twin"
                separation = abs((twin_h - base_h + 180) % 360 - 180)
                assert separation < 10, (
                    f"{slot}: hue drifted {separation:.1f} deg from its twin — it is a different hue, not a tier"
                )

    def test_the_categorical_range_offers_the_whole_palette(self):
        """The palette does not refuse to draw past the validated slots.

        The palette never refuses to draw: the number of series is a property of
        the data, so a fifth series takes the derived tier rather than a neutral. Handing
        Vega a 4-wide range would put it back to cycling, which repaints series 5 as
        series 1 — the failure the old cap was reaching for and did not actually prevent.
        """
        palette = load_palette()
        for theme in ("light", "dark"):
            assert vega_config(theme)["range"]["category"] == palette[theme]["chart"]
            # Against the artifact's own counted width, not against a literal 8. A
            # literal here was the last independently-pinned copy of the palette width:
            # widening the palette would have failed this test as a regression rather
            # than passing it as the change it is.
            assert len(vega_config(theme)["range"]["category"]) == series_slots()

    def test_a_value_drawn_on_a_mark_clears_the_contrast_bar_in_both_themes(self):
        """The knockout ink, measured rather than eyeballed.

        The value-placement rule puts a value inside its mark when there is no clearance outside, which
        is the longest bar on nearly every figure rather than an edge case. Chart ink
        there measures 2.53:1 in dark and 3.36:1 in light against slot 1 — the only fill
        a value is ever written on — and the design rules ask 4.5:1 for text at this
        size. The two-contrast-levels rule admits a third ink for this one use.

        Computed from the committed artifact rather than asserted as a number, so
        retuning a hue fails here instead of quietly dropping the value label below the
        bar it was granted an exception to clear.
        """
        for theme in ("light", "dark"):
            config = vega_config(theme)
            fill = config["bar"]["color"]
            knockout = config["style"]["chart-value-on-fill"]["color"]
            ratio = _contrast_ratio(knockout, fill)
            assert ratio >= 4.5, f"{theme}: value-on-fill {knockout} over {fill} is {ratio:.2f}:1, under 4.5:1"

    def test_the_knockout_is_the_reason_the_ordinary_ink_could_not_be_used(self):
        """Guards the amendment's premise, so it cannot outlive its own justification.

        If chart ink ever clears 4.5:1 over slot 1 on its own, the third ink is a
        complication with no argument left and the palette should go back to two levels.
        A failure here is that news, not a regression.
        """
        for theme in ("light", "dark"):
            config = vega_config(theme)
            ratio = _contrast_ratio(config["text"]["color"], config["bar"]["color"])
            assert ratio < 4.5, (
                f"{theme}: chart ink now clears {ratio:.2f}:1 over the mark fill — the on-fill knockout "
                "exists because it did not, so re-open the third-ink exception rather than deleting this test"
            )

    def test_the_named_roles_are_present_and_are_not_categorical_slots(self):
        """`highlight`, `context` and the seq ramp mean something; they are not spare hues.

        Carried separately from `chart[]` so no index can ever reach them: assigning
        `highlight` as "series 5" would put spot magenta in a categorical set, which is the
        one thing its once-per-figure discipline forbids.
        """
        palette = load_palette()
        for theme in ("light", "dark"):
            mode = palette[theme]
            assert mode["highlight"].startswith("#") and mode["context"].startswith("#")
            assert len(mode["seq"]) == 5
            assert mode["highlight"] not in mode["chart"]
            assert mode["context"] not in mode["chart"]

    def test_the_chart_type_scale_reaches_the_config_at_the_weight_floor(self):
        """Charts carry their own scale, and no chart text drops below 500 except footnotes.

        The floor is the point: light-on-dark optically thins, so a 400-weight tick label
        reads spindly at a size that measures correct. A tick is also the only place a
        value is written down, which is why it takes full chart ink rather than the muted
        level the page reserves for captions.
        """
        palette = load_palette()
        assert palette["font_size"]["tick"] == 14
        assert palette["font_size"]["value"] > palette["font_size"]["tick"]
        for role, weight in palette["font_weight"].items():
            assert weight >= 500 or role == "footnote", f"{role} is below the weight floor"
        for theme in ("light", "dark"):
            config = vega_config(theme)
            assert config["axis"]["labelFontSize"] == palette["font_size"]["tick"]
            assert config["axis"]["labelFontWeight"] == palette["font_weight"]["tick"]
            assert config["axis"]["labelColor"] == palette[theme]["ink"]


class TestSvgRender:
    def test_the_series_colour_reaches_the_svg_as_hex(self, spec):
        for theme in ("light", "dark"):
            svg = render_svg(spec, theme=theme)
            assert load_palette()[theme]["chart"][0] in svg

    def test_no_unparseable_colour_notation_reaches_the_svg(self, spec):
        """The `fill` attribute is exactly where an OKLCH string would survive to."""
        for theme in ("light", "dark"):
            assert "oklch" not in render_svg(spec, theme=theme).lower()

    def test_the_configured_family_reaches_the_svg_as_font_family(self, spec):
        """That the theme's `font` key is honoured — NOT that the brand face drew the text.

        vl-convert writes the configured family into every `font-family`
        attribute whether or not a file providing it was ever registered, so this
        assertion holds with no font directory registered and cannot see the
        fallback `render.py` warns about. It is still worth pinning: it is what
        fails if the `font` key stops reaching text marks at all.
        """
        assert f'font-family="{vega_config("dark")["font"]}"' in render_svg(spec, theme="dark")

    def test_the_drawn_labels_are_the_compiled_labels(self, spec):
        svg = render_svg(spec, theme="dark")
        for part in PAYLOAD["parts"]:
            assert part["label"] in svg

    def test_the_dumbbells_point_is_drawn_inside_the_plot(self):
        """The arm's reserved point radius is a claim about the renderer, so the renderer settles it.

        `delta_table` widens its axis past the largest change by the radius of the
        point that marks it, or that point is drawn half outside the plot and clipped
        into a half-disc against the identity labels. The radius is computed from the
        mark's `size`, and the conversion is not the obvious one: Vega sizes a symbol
        by the area of its BOUNDING BOX, so a circle of `size` 80 has radius
        `sqrt(80)/2`, not d3's `sqrt(80/pi)`. Both are arithmetic that reads as correct
        on the page and only one is what gets drawn, so this measures the path the
        rasteriser emitted rather than recomputing the formula under test.

        The one-row shape is used because a single metric is the ordinary comparison
        and its row necessarily decides the axis.
        """
        payload = {"rows": [{"metric": "llm_ms", "a": 16162.0, "b": 242.0, "unit": "ms"}]}
        spec = compile_chart("delta_table", payload).spec
        # The radius the arm reserves, read off the mark it compiled: the arm clears its axis by
        # ``point_radius`` of the very ``size`` it hands Vega.
        (size,) = {
            node["mark"]["size"]
            for node in _walk_dicts(spec)
            if isinstance(node.get("mark"), dict) and node["mark"].get("type") == "point" and "size" in node["mark"]
        }
        reserved = point_radius(size)
        svg = render_svg(spec, theme="dark")
        drawn = re.search(
            r'aria-roledescription="point"[^>]*transform="translate\(([\d.]+),[\d.]+\)" d="M([\d.]+),0A', svg
        )
        assert drawn, "no point mark reached the document"
        centre, radius = float(drawn.group(1)), float(drawn.group(2))
        assert radius == pytest.approx(reserved, abs=0.01), "the reserved radius is not the one Vega draws"
        assert centre - radius >= 0, f"the point is clipped by the plot's low edge by {radius - centre:.2f}px"


class TestADumbbellsValueIsNotStruckThroughByItsOwnMark:
    """The two arms that draw a thin line along the row, measured against their numbers.

    A label with no room past its mark is pushed INWARD, and inward of a dumbbell's
    value is the line it hangs on: `delta_table`'s connector back to zero, and
    `null_result`'s interval rule back to its low bound. Both are 3px drawn on the
    row's own centreline, which is where a label centred on that row puts its glyphs —
    so the number was drawn with a bar through it, end to end, in both themes.

    `delta_table` had it on EVERY chart and every row count, not as an edge case: the
    room past the row that decides the axis is the point's radius and nothing more, so
    that row's label is always the pushed one. `null_result` had it only where the
    outermost arm's number is too wide for the domain pad — the case
    `_MarkValue.filled` already describes, reproduced here at `1.235e-05`.

    **Measured off the raster, not off the spec's offsets.** What settles it is where
    the glyphs landed, and the lift that clears them is arithmetic over a font size
    only the renderer resolves — the same reason the point's radius is measured rather
    than recomputed. Two details of the method carry the weight:

    * The comparison is per COLUMN. Numerals are sparse, so a band taken across a whole
      label would count the daylight between glyphs as ink and could report an overlap
      the reader cannot see — or hide one inside a wide empty column.
    * Only the EXACT series hue counts as the mark. Its anti-aliased fringe is covered
      by `FRINGE`; the shaded overlap band `null_result` draws under both is a
      blend rather than a line, and a number over a wash is not what this is about.
    """

    #: How far a mark's anti-aliased edge bleeds past the pixels drawn in its exact
    #: colour, in px. One, measured on this rasteriser at scale 1: a 3px rule lands two
    #: rows of the flat hue with one blended row on each side of them.
    FRINGE = 1

    #: The ordinary one-metric comparison, whose single row necessarily decides the axis.
    ONE_ROW = {"rows": [{"metric": "llm_ms", "a": 16162.0, "b": 242.0, "unit": "ms"}]}

    #: Several rows, so the connector is a fraction of a much shorter band — the shape
    #: the arm's own comment claimed was the safe one.
    MANY_ROWS = {
        "rows": [{"metric": f"m{index}", "a": 16162.0, "b": 242.0 * (index + 1), "unit": "ms"} for index in range(8)]
    }

    #: An established null whose outermost arm carries a number too wide for the room
    #: past it, which is what pushes a `null_result` label onto its own interval rule.
    WIDE_NULL = {
        "metric": "score",
        "unit": "",
        "groups": [
            {
                "label": "control",
                "n": 12,
                "ci": {
                    "mean": 1.2345e-05,
                    "low": 1.0e-05,
                    "high": 1.4e-05,
                    "level": 0.95,
                    "variability": "across 5 runs",
                },
            },
            {
                "label": "treatment",
                "n": 12,
                "ci": {
                    "mean": 1.23456e-05,
                    "low": 1.1e-05,
                    "high": 1.5e-05,
                    "level": 0.95,
                    "variability": "across 5 runs",
                },
            },
        ],
    }

    def _drawn(self, viz_type, payload, theme):
        """Which rows of pixels the value ink and the series hue cover, column by column.

        Args:
            viz_type: The arm to compile.
            payload: Its payload.
            theme: The palette to draw in — the defect was in both, so the check is.

        Returns:
            ``(ink, hue)``, each mapping a column to the set of rows it is drawn in.
        """
        png = render_png(compile_chart(viz_type, payload).spec, theme=theme, scale=1)
        mode = load_palette()[theme]
        value_ink, series_hue = _hex_to_rgb(mode["ink"]), _hex_to_rgb(mode["chart"][0])
        ink: dict[int, set[int]] = defaultdict(set)
        hue: dict[int, set[int]] = defaultdict(set)
        for row, scanline in enumerate(pixel_rows(png)):
            for column, pixel in enumerate(scanline):
                if pixel == value_ink:
                    ink[column].add(row)
                elif pixel == series_hue:
                    hue[column].add(row)
        return ink, hue

    # One rasterisation and a dependency-free PNG decode in pure Python, per case.

    def test_a_lifted_label_stays_on_the_row_whose_value_it_states(self):
        """The other half, and the bound on the fix: clearing the line must not cost the row.

        Lift a label far enough and it leaves the row it names — on a chart of eight
        rows it starts reading as the row above's. Half the row step is the line it may
        not cross, measured between each label and its OWN connector, which the drawn
        document pairs for us: both marks carry the row in their `aria-label`.

        The glyph band is bounded by the font size the document states rather than by a
        cap-height guess, which errs the only safe way — the numerals draw shorter than
        their line box, so a band that passes this measured wide passes drawn.
        """
        svg = render_svg(compile_chart("delta_table", self.MANY_ROWS).spec, theme="dark")
        connectors = {
            match["row"]: float(match["top"]) + float(match["height"]) / 2
            for match in re.finditer(
                r'aria-label="[^"]*display: (?P<row>[^";]+)"[^>]*aria-roledescription="bar"'
                r'[^>]*d="M[\d.eE+-]+,(?P<top>[\d.eE+-]+)h[\d.eE+-]+v(?P<height>[\d.eE+-]+)',
                svg,
            )
        }
        labels = {
            match["row"]: (float(match["baseline"]), float(match["size"]))
            for match in re.finditer(
                r'aria-label="[^"]*display: (?P<row>[^";]+); text: [^"]*"[^>]*aria-roledescription="text mark"'
                r'[^>]*transform="translate\([\d.eE+-]+,(?P<baseline>[\d.eE+-]+)\)"[^>]*font-size="(?P<size>[\d.]+)px"',
                svg,
            )
        }
        assert labels and labels.keys() == connectors.keys(), (
            f"every row must draw both marks — labelled {sorted(labels)}, connected {sorted(connectors)}"
        )
        bound = geometry()["row_step"] / 2
        for row, (baseline, size) in labels.items():
            reach = max(abs(baseline - connectors[row]), abs(baseline - size - connectors[row]))
            assert reach <= bound, (
                f"{row}'s value is drawn {reach:.1f}px from the connector it belongs to, "
                f"past the {bound}px half row step that keeps it attached to its own dumbbell"
            )


class TestPngPixels:
    """What actually reached the raster — the only check the black-chart defect fails."""

    def test_the_bars_are_drawn_in_the_expected_series_colour(self, spec):
        """Samples the drawn pixels, because a black chart is a VALID PNG.

        The bars are the largest non-background region, so the second most common
        colour is them. Asserting on the count as well as the value is what
        distinguishes "the series colour appears" from "the series colour was
        drawn".
        """
        for theme in ("light", "dark"):
            colours = dominant_colours(render_png(spec, theme=theme, scale=1))
            expected = _hex_to_rgb(load_palette()[theme]["chart"][0])
            drawn = {rgb: count for rgb, count in colours}
            assert expected in drawn, f"{theme}: expected bars in {expected}, got {colours}"
            assert drawn[expected] > 1000, f"{theme}: series colour present but barely drawn: {colours}"

    def test_the_chart_is_not_drawn_black(self, spec):
        """The exact signature of an unconverted palette: every series pure black."""
        colours = dominant_colours(render_png(spec, theme="dark", scale=1))
        drawn = {rgb: count for rgb, count in colours}
        assert drawn.get((0, 0, 0), 0) < 1000, f"chart drew a large black region — palette did not resolve: {colours}"

    def test_the_background_is_the_validated_chart_surface(self, spec):
        """Contrast was measured against this surface; drawing on another invalidates it."""
        for theme in ("light", "dark"):
            colours = dominant_colours(render_png(spec, theme=theme, scale=1))
            assert colours[0][0] == _hex_to_rgb(load_palette()[theme]["surface"])

    def test_every_compiled_type_is_drawn_here(self):
        """The coverage the type-keyed dict used to give for free.

        `PAYLOADS` is keyed by shape now, so a viz type gaining a payload model and
        a compiler arm no longer forces an entry. This is what forces it: the pixel
        checks are the only place a new mark type is caught drawing black.
        """
        assert {viz_type for viz_type, _ in PAYLOADS.values()} == set(PAYLOAD_MODELS)

    # Same measurement, same reason — this one rasterises one shape per case in both themes and
    # its slowest parametrisation sat >3s idle, so it crosses the ceiling under the same load.
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("shape", sorted(PAYLOADS), ids=sorted(PAYLOADS))
    def test_no_compiled_chart_shape_draws_an_uncoloured_mark(self, shape):
        """Every shape, not just the one this file was written for.

        A chart's marks come from the theme config, and the config is a per-mark
        map — so "the palette resolves" is a claim about the mark types actually
        used, and a new compiler case introduces new ones. Asserted at a low
        threshold because a rule or a dot covers far less area than a bar; what is
        being caught is a mark drawn in Vega's default black, which covers plenty.

        Parametrised per shape rather than looped, because the work here is two
        rasterisations and two pixel histograms per shape and the shape set grows
        with every compiler arm. As one case it measured 24s against the suite's
        30s per-test budget — passing, but close enough that host contention
        crossed it, and the next type added would cross it unloaded. Split, each
        case carries one shape's cost and the total is unchanged; nothing is
        asserted less.
        """
        viz_type, payload = PAYLOADS[shape]
        spec = compile_chart(viz_type, payload).spec
        for theme in ("light", "dark"):
            counts = dominant_colours(render_png(spec, theme=theme, scale=1), limit=None)
            # Counted over the WHOLE histogram and reported over the head of it: the
            # question is how much of this colour was drawn, not whether it out-ranked
            # the anti-aliasing, and a shape whose marks are small loses that ranking
            # while being painted perfectly.
            drawn = dict(counts)
            commonest = dict(counts[:12])
            expected = _hex_to_rgb(load_palette()[theme]["chart"][0])
            assert drawn.get(expected, 0) > 100, f"{shape}/{theme}: series colour barely drawn: {commonest}"
            if theme == "dark":
                assert drawn.get((0, 0, 0), 0) < 500, (
                    f"{shape}: a mark drew black — the palette did not reach it: {commonest}"
                )

    # Every shape in PAYLOADS, rasterised in both themes, through matplotlib. Measured 20.8s of
    # call time on an idle box — 69% of the global 30s ceiling — so on a loaded one it crosses and
    # pytest-timeout kills the worker with no assertion and no traceback, which reads as a
    # regression in whatever branch happens to be running. The bound is declared rather than the
    # work reduced: the whole point is that EVERY compiled shape reaches a raster, and dropping
    # shapes to fit a ceiling nobody chose would trade the coverage for the schedule.

    def test_no_compiled_shape_asks_a_renderer_to_size_it(self):
        """The premise of the check above: nothing is left for a renderer to decide.

        `"width": "container"` is Vega-Lite's "fill your parent", and it was once
        what every compiled spec carried — each renderer substituting the width it
        had, which is why one chart drew at two sizes and why a nested one drew at
        neither. Vega-Lite does not support it under `facet` or `concat` at all, so
        the composed shapes were the ones that broke.

        This asserts the whole tree rather than the top level, because that nesting
        is exactly where it hid, and it asserts on the JSON text so a `container`
        reintroduced under a key this test does not know about is still caught.
        """
        for shape, (viz_type, payload) in PAYLOADS.items():
            spec = compile_chart(viz_type, payload).spec
            assert '"container"' not in json.dumps(spec), f"{shape} left a size for a renderer to substitute"

    def test_the_shapes_whose_size_is_nested_are_still_among_them(self):
        """The other premise: some spec must still carry its size below the top.

        The rasterised-width check is only load-bearing while a compiled spec sizes
        its inner views rather than itself, and each such shape is a compiler
        decision this file does not control — a concatenated distribution, a
        pre-binned one whose counts panel is its only panel, a sweep whose
        barcode and ranking are two panels of one row, and a time series faceted
        into a panel per series. If the compiler ever wraps
        them in a sized parent, that check stops covering the nesting and passes for
        the wrong reason; this is what says so.
        """
        nested = {
            shape
            for shape, (viz_type, payload) in PAYLOADS.items()
            for spec in [compile_chart(viz_type, payload).spec]
            if "width" not in spec and '"width"' in json.dumps(spec)
        }
        assert nested == {"distribution_with_values", "distribution_buckets_only", "sweep_ranking", "timeseries"}, (
            f"nested-size shapes moved: {nested}"
        )

        buckets_only = compile_chart(*PAYLOADS["distribution_buckets_only"]).spec
        [panel] = buckets_only["vconcat"]
        assert "facet" in panel, f"no longer a faceted counts panel: {sorted(panel)}"
        assert panel["spec"]["width"] == geometry()["plot_width"]

    def test_an_unresolved_palette_would_be_caught(self, spec):
        """Proves the assertions above can fail — the defect, reproduced deliberately.

        Without this, every check in this class would pass just as happily against
        a renderer that had stopped honouring the palette at all, and there would
        be no evidence that "renders black" is a thing they detect rather than a
        thing the docstrings claim.
        """
        import vl_convert as vlc

        broken = {
            **vega_config("dark"),
            "mark": {"color": "oklch(0.70 0.22 295)"},
            "bar": {"color": "oklch(0.70 0.22 295)"},
        }
        png = vlc.vegalite_to_png(json.dumps(spec), config=broken, scale=1)
        drawn = {rgb: count for rgb, count in dominant_colours(png)}
        assert drawn.get((0, 0, 0), 0) > 1000, "OKLCH no longer renders black — this suite's premise needs rechecking"
        assert _hex_to_rgb(load_palette()["dark"]["chart"][0]) not in drawn


class TestALabelTooWideForTheColumnOverrunsRatherThanTruncating:
    """Which way the two rules break when they cannot both hold.

    A category name that will not fit the 176px gutter moves onto its own line
    inside the plot, and a name that will not fit there either has nowhere left to
    go: it is one token, so it cannot wrap the way a title does, and the label rule
    forbids both of the remaining options — never truncate, never shrink.

    So the figure widens past its column, and the figure card's horizontal scroll
    is what carries it. That is the deliberate trade rather than an oversight, and
    it is pinned here because it is the one place the fixed geometry gives way:
    without this, a later change could "fix" the overrun by reinstating a
    `labelLimit` and every other test in the suite would stay green while model IDs
    started drawing truncated again.
    """

    #: One token, no whitespace to wrap on, wider than the 1040 column at label size.
    _UNBREAKABLE = "anthropic/" + "x" * 150

    def _spec(self):
        payload = {
            "parts": [{"label": self._UNBREAKABLE, "value": 42.0}, {"label": "openai/gpt-5.2", "value": 19.0}],
            "unit": "%",
            "measure": "share of stops",
        }
        return compile_chart("breakdown", payload).spec

    def test_the_name_survives_whole_into_the_drawn_document(self):
        assert self._UNBREAKABLE in render_svg(self._spec(), theme="dark")

    def test_nothing_in_the_figure_is_ellipsised(self):
        """Vega's own truncation marker. Its absence is the rule holding."""
        assert "…" not in render_svg(self._spec(), theme="dark")

    def test_the_figure_widens_past_its_column_and_that_is_the_cost(self):
        """Stated as a measurement, so the trade is visible rather than discovered."""
        width, _height = png_size(render_png(self._spec(), theme="dark", scale=1))
        assert width > geometry()["figure_width"]
