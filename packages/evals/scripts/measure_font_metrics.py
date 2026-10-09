"""Measure a chart typeface through vl-convert and write the advance table the chart compiler lays out from.

Dev tooling for ``3tears-evals``: it lives beside the package, not in it, and is not installed. It needs
the ``[vega]`` extra (``vl-convert-python``).

    # Re-measure the packaged face and rewrite src/threetears/evals/vega/font_metrics.json:
    uv run python packages/evals/scripts/measure_font_metrics.py

    # Measure a host's own face, registered from a font directory, into a file the host keeps:
    uv run python packages/evals/scripts/measure_font_metrics.py \\
        --family "Inter, Arial, sans-serif" --font-dir path/to/fonts --out inter_metrics.json

A host then declares the face with ``StyleProfile(chart_font=load_chart_font(Path("inter_metrics.json")))``.

**What is measured, and how.** Each probe is a Vega text mark drawn at :data:`PROBE_SIZE` px in the FIRST
family of ``--family``, and its width is read back from the SVG vl-convert sizes to fit it (``autosize:
pad``, no padding) — Vega's own text measurement, which is what lays the chart out, rather than the font
file's declared advances. Every printable ASCII character is measured in two contexts, and at every
weight the packaged type scale draws at:

* **isolated**, between two ``H`` glyphs: ``width("H" + c + "H") - width("HH")``;
* **repeated**, in a run of itself between the same two ``H`` glyphs, divided by the run length.

The table keeps each character's widest result across contexts and weights, so the per-character sum
leans long: too wide moves a label to its own line, too narrow truncates it. Weights are folded in
rather than tabulated because the packaged face draws only two (500 resolves to regular, 600 to bold), and
a title measured at a body weight would wrap too late.

**Then it checks itself.** A corpus of real model IDs, measure names, numbers and deliberately
kern-heavy strings is drawn whole at each weight and compared with the table's sum. The worst
``drawn / estimated`` ratio is recorded in the artifact, and ``write_font_metrics`` refuses to write a table
whose corpus underestimates past ``SAFETY_MARGIN``.

**It refuses a face that does not resolve.** vl-convert draws NO text for a family it cannot find (it
does not fall back within a single-family list), and its measurement still returns widths, so a probe of
a missing face measures something other than the face named. A rasterised probe that comes out identical
to an empty one stops the run.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

from threetears.evals.contracts.host import CHART_FONT_CHARACTERS
from threetears.evals.vega.palette import font_weights
from threetears.evals.vega.text_metrics import METRICS_PATH, write_font_metrics

#: The packaged default: Liberation Sans, which vl-convert embeds (so a raster draws it on a host with no
#: fonts installed), then Arial, which is metric-compatible with it (so a browser without Liberation Sans
#: lays text out at the same widths), then the generic family.
DEFAULT_FAMILY = "Liberation Sans, Arial, sans-serif"

#: The font size every probe is drawn at, in px. Vega reports the sized SVG in whole px, so a larger probe
#: makes that rounding a smaller fraction of an advance: at 1000 px it is under 0.001 of the font size.
PROBE_SIZE = 1000

#: How many copies of a character the repeated context draws.
RUN_LENGTH = 10

#: Strings a chart lays out, drawn whole to check the per-character sum against Vega's own measurement.
CORPUS = (
    "anthropic/claude-opus-4-20250514",
    "anthropic-0/claude-opus-4-20250514-extended-thinking-preview",
    "openai/gpt-4o-2024-08-06",
    "google/gemini-2.5-pro-preview-05-06",
    "meta-llama/Llama-3.1-405B-Instruct-Turbo",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "accuracy",
    "cost_usd",
    "mean_composite",
    "p95 latency (s)",
    "share of stops",
    "budget_exhausted",
    "End-to-end",
    "Subsystem",
    "Unattributed",
    "timeout=4s",
    "0.1234",
    "1,234,567",
    "1.235e-05",
    "+12.5%",
    "-0.0081",
    "11111111",
    "AVAVAVAVAVAV",
    "To Ty Te Yo LT LV",
    "WWWWWWWWWWWWWWWWWWWW",
    "/////",
    "The batch never fills before the deadline at either setting.",
)


def _width(text: str, *, face: str, weight: int) -> float:
    """How wide ``text`` draws at :data:`PROBE_SIZE`, in px, by Vega's own measurement."""
    import vl_convert as vlc

    spec = _probe_spec(text, face=face, weight=weight)
    svg = vlc.vega_to_svg(json.dumps(spec))
    match = re.search(r'<svg[^>]*\swidth="([\d.]+)"', svg)
    if match is None:
        raise RuntimeError(f"vl-convert returned an SVG with no width for {text!r}")
    return float(match.group(1))


def _probe_spec(text: str, *, face: str, weight: int) -> dict[str, Any]:
    """A Vega spec drawing ``text`` alone, left-aligned at the origin, sized to fit it."""
    return {
        "$schema": "https://vega.github.io/schema/vega/v5.json",
        "autosize": "pad",
        "padding": 0,
        "width": 0,
        "height": 0,
        "marks": [
            {
                "type": "text",
                "encode": {
                    "enter": {
                        "x": {"value": 0},
                        "y": {"value": 0},
                        "text": {"value": text},
                        "align": {"value": "left"},
                        "baseline": {"value": "top"},
                        "font": {"value": face},
                        "fontSize": {"value": PROBE_SIZE},
                        "fontWeight": {"value": weight},
                    }
                },
            }
        ],
    }


def _require_face_resolves(face: str) -> None:
    """Stop when vl-convert cannot draw ``face``: it would draw nothing, and measure something else."""
    import vl_convert as vlc

    drawn = vlc.vega_to_png(json.dumps(_probe_spec("HHHH", face=face, weight=400) | {"width": 4000, "height": 1200}))
    empty = vlc.vega_to_png(json.dumps(_probe_spec("", face=face, weight=400) | {"width": 4000, "height": 1200}))
    if drawn == empty:
        raise SystemExit(
            f"vl-convert draws no text in {face!r}: the face is not embedded, installed, or in --font-dir. "
            "Measuring it would record another face's widths."
        )


def measure(family: str) -> dict[str, Any]:
    """Measure ``family``'s first face at every weight the type scale draws at.

    Args:
        family: The CSS family list; its first family is the face measured.

    Returns:
        The keyword arguments :func:`write_font_metrics` takes, less ``path``.
    """
    face = family.split(",")[0].strip()
    _require_face_resolves(face)
    weights = sorted({int(weight) for weight in font_weights().values()})
    advances: dict[str, float] = {}
    for weight in weights:
        pair = _width("HH", face=face, weight=weight)
        for character in CHART_FONT_CHARACTERS:
            isolated = _width(f"H{character}H", face=face, weight=weight) - pair
            repeated = (_width(f"H{character * RUN_LENGTH}H", face=face, weight=weight) - pair) / RUN_LENGTH
            widest = max(isolated, repeated) / PROBE_SIZE
            advances[character] = round(max(advances.get(character, 0.0), widest), 5)
    worst_label, worst_ratio = "", 0.0
    for label in CORPUS:
        estimated = sum(advances[character] for character in label) * PROBE_SIZE
        for weight in weights:
            ratio = _width(label, face=face, weight=weight) / estimated
            if ratio > worst_ratio:
                worst_label, worst_ratio = label, ratio
    return {
        "advances": advances,
        "fallback_advance": max(advances.values()),
        "worst_label": worst_label,
        "worst_ratio": worst_ratio,
        "font": family,
        "measured_with": f"vl-convert-python {version('vl-convert-python')}",
        "probe_size": PROBE_SIZE,
        "weights": weights,
    }


def main(argv: list[str] | None = None) -> int:
    """Measure, check and write a font metrics table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family", default=DEFAULT_FAMILY, help=f"CSS family list (default: {DEFAULT_FAMILY!r})")
    parser.add_argument("--font-dir", type=Path, help="a directory of font files to register before measuring")
    parser.add_argument("--out", type=Path, default=METRICS_PATH, help="where to write (default: the packaged table)")
    args = parser.parse_args(argv)
    if args.font_dir is not None:
        import vl_convert as vlc

        vlc.register_font_directory(str(args.font_dir.resolve()))
    measured = measure(args.family)
    written = write_font_metrics(**measured, path=args.out)
    print(
        f"wrote {written}: {args.family!r}, weights {measured['weights']}, worst corpus ratio "
        f"{measured['worst_ratio']:.4f} on {measured['worst_label']!r}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
