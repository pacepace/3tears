"""Measure a chart typeface through vl-convert and write the advance table the chart compiler lays out from.

It ships with the renderer and needs the ``[vega]`` extra (``vl-convert-python``)::

    # Re-measure the packaged face and rewrite the packaged table (from a checkout of this repository):
    python -m threetears.evals.vega.measure_font

    # Measure a host's own face, registered from a font directory, into a file the host keeps:
    python -m threetears.evals.vega.measure_font \\
        --family "Inter, Arial, sans-serif" --font-dir path/to/fonts --out inter_metrics.json

A host then declares the face with ``StyleProfile(chart_font=load_chart_font(Path("inter_metrics.json")))``
and builds its renderer with the same directory (``VegaRenderer.for_style(style, font_dir=...)``).

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

**It records the files the face was drawn from.** The advances describe a face's bytes, not its name, so
the table carries the SHA-256 of each font file under ``--font-dir`` whose own name table declares the
measured family — those files only: a directory may hold other faces, and adding or updating one of them
must not invalidate this table. A renderer then refuses the table once those files change
(:func:`~threetears.evals.vega.text_metrics.check_font_files`). Which files provide a family is read from
each file's ``name`` table (:func:`font_families`), the same names the rasteriser's font database matches
a family against; this module parses that table itself, so measuring needs nothing past the extra.

**It refuses a face it cannot tie to files.** With ``--font-dir``, a family no file in the directory
declares is refused, since whatever drew it (the machine's own fonts) is not what a host ships. Without
one, only the face the rasteriser embeds — the packaged face — can be measured: any other resolves from
the machine's font database, which does not travel to the image that renders.

**It refuses a face that does not resolve.** vl-convert draws NO text for a family it cannot find (it
does not fall back within a single-family list), and its measurement still returns widths, so a probe of
a missing face measures something other than the face named. A rasterised probe that comes out identical
to an empty one stops the run.
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

from threetears.evals.kernel.host import CHART_FONT_CHARACTERS
from threetears.evals.vega.palette import font_weights
from threetears.evals.vega.render import register_fonts, vega_png, vega_svg
from threetears.evals.vega.text_metrics import (
    METRICS_PATH,
    PACKAGED_FAMILY,
    font_file_digest,
    write_font_metrics,
)

#: The face the rasteriser embeds, and so the one face measurable with no font directory.
EMBEDDED_FACE = PACKAGED_FAMILY.split(",")[0].strip()

#: The font size every probe is drawn at, in px. Vega reports the sized SVG in whole px, so a larger probe
#: makes that rounding a smaller fraction of an advance: at 1000 px it is under 0.001 of the font size.
PROBE_SIZE = 1000

#: How many copies of a character the repeated context draws.
RUN_LENGTH = 10

#: The file suffixes the rasteriser's font database loads from a registered directory, lowercased.
FONT_SUFFIXES = frozenset({".ttf", ".otf", ".ttc", ".otc"})

#: The ``name`` table records that name a family: 1 is the family, 16 the typographic family a face that
#: has more than four styles declares beside it. A face is matched against either.
_FAMILY_NAME_IDS = frozenset({1, 16})

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


class FontMeasureError(SystemExit):
    """The face cannot be measured as asked: it does not draw, or cannot be tied to the files it draws from.

    A :class:`SystemExit`, because this module is a command and its refusals are the command's exit: the
    message is what the person running it reads.
    """


def font_families(data: bytes) -> frozenset[str]:
    """Every family a font file's ``name`` table declares, in every face it holds.

    Reads an sfnt font (TrueType or OpenType, ``.ttf``/``.otf``) or a collection of them (``.ttc``/
    ``.otc``): name IDs 1 and 16, on the Unicode and Windows platforms (UTF-16BE) and the Macintosh
    platform's Roman encoding. Nothing else is read.

    Args:
        data: The file's bytes.

    Returns:
        The families, as declared. Empty for bytes that are not a font, or a font with no family name.
    """
    try:
        if data[:4] == b"ttcf":
            (count,) = struct.unpack_from(">I", data, 8)
            offsets = struct.unpack_from(f">{count}I", data, 12)
        else:
            offsets = (0,)
        return frozenset(family for offset in offsets for family in _sfnt_families(data, offset))
    # NOSILENT: bytes that do not parse as a font declare no family; the rasteriser's font database skips
    # such a file too, so it cannot be what drew the face, and the caller refuses when no file declares it.
    except struct.error:
        return frozenset()


def _sfnt_families(data: bytes, offset: int) -> list[str]:
    """The families one sfnt face at ``offset`` declares in its ``name`` table.

    Args:
        data: The file's bytes.
        offset: Where the face's table directory starts.

    Returns:
        The families; empty when the face has no ``name`` table.

    Raises:
        struct.error: The bytes end inside a structure they declare.
    """
    (tables,) = struct.unpack_from(">H", data, offset + 4)
    for record in range(tables):
        tag, _checksum, start, _length = struct.unpack_from(">4sIII", data, offset + 12 + 16 * record)
        if tag == b"name":
            _format, count, strings = struct.unpack_from(">HHH", data, start)
            families = []
            for entry in range(count):
                platform, encoding, _language, name_id, length, at = struct.unpack_from(
                    ">HHHHHH", data, start + 6 + 12 * entry
                )
                if name_id not in _FAMILY_NAME_IDS:
                    continue
                raw = data[start + strings + at : start + strings + at + length]
                if platform in {0, 3}:
                    families.append(raw.decode("utf-16-be", errors="replace"))
                elif platform == 1 and encoding == 0:
                    families.append(raw.decode("mac_roman"))
            return families
    return []


def resolving_files(face: str, font_dir: Path) -> dict[str, str]:
    """Every font file under ``font_dir`` that declares ``face`` as its family, with its digest.

    The directory is read recursively, as the rasteriser registers it, and a family matches without
    regard to case. Files declaring other families are not in the result: the table is tied to the files
    of the face it measured, not to the directory.

    Args:
        face: The family measured.
        font_dir: The directory registered for it.

    Returns:
        Each file's path relative to ``font_dir`` (``/``-separated) to the SHA-256 of its bytes.
    """
    wanted = face.casefold()
    return {
        path.relative_to(font_dir).as_posix(): font_file_digest(path)
        for path in sorted(font_dir.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in FONT_SUFFIXES
        and wanted in {family.casefold() for family in font_families(path.read_bytes())}
    }


def _width(text: str, *, face: str, weight: int) -> float:
    """How wide ``text`` draws at :data:`PROBE_SIZE`, in px, by Vega's own measurement."""
    svg = vega_svg(_probe_spec(text, face=face, weight=weight))
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
    """Stop when vl-convert cannot draw ``face``: it would draw nothing, and measure something else.

    Raises:
        FontMeasureError: The face draws nothing.
    """
    drawn = vega_png(_probe_spec("HHHH", face=face, weight=400) | {"width": 4000, "height": 1200})
    empty = vega_png(_probe_spec("", face=face, weight=400) | {"width": 4000, "height": 1200})
    if drawn == empty:
        raise FontMeasureError(
            f"vl-convert draws no text in {face!r}: the face is not embedded, installed, or in --font-dir. "
            "Measuring it would record another face's widths."
        )


def face_files(family: str, font_dir: Path | None) -> dict[str, str]:
    """The files ``family``'s first face will be measured from, registering ``font_dir`` when there is one.

    Args:
        family: The CSS family list; its first family is the face measured.
        font_dir: The directory holding the face's files, or ``None`` for the embedded face.

    Returns:
        What :func:`resolving_files` finds; empty for the embedded face.

    Raises:
        FontMeasureError: ``font_dir`` is not a directory, or no file in it declares the face, or there is
            no ``font_dir`` and the face is not the one the rasteriser embeds.
    """
    face = family.split(",")[0].strip()
    if font_dir is None:
        if face.casefold() != EMBEDDED_FACE.casefold():
            raise FontMeasureError(
                f"{face!r} is not the face the rasteriser embeds ({EMBEDDED_FACE!r}), so without --font-dir it "
                "could only be drawn from this machine's own fonts, which neither travel to the image that renders "
                "nor can be named in the table. Pass --font-dir with the directory holding its files."
            )
        return {}
    if not font_dir.is_dir():
        raise FontMeasureError(f"--font-dir {font_dir} is not a directory")
    files = resolving_files(face, font_dir)
    if not files:
        raise FontMeasureError(
            f"no font file under {font_dir} declares the family {face!r} in its name table, so whatever drew it "
            "would not be the files a renderer is given. Check the family's spelling against the font, or point "
            "--font-dir at the directory holding its files."
        )
    register_fonts(font_dir)
    return files


def measure(family: str) -> dict[str, Any]:
    """Measure ``family``'s first face at every weight the type scale draws at.

    Args:
        family: The CSS family list; its first family is the face measured.

    Returns:
        The keyword arguments :func:`write_font_metrics` takes, less ``font_files`` and ``path``.

    Raises:
        FontMeasureError: The face draws nothing.
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
    """Measure, check and write a font metrics table.

    Args:
        argv: The arguments; ``None`` reads the command line.

    Returns:
        The exit status.

    Raises:
        FontMeasureError: The face cannot be measured as asked.
        TextMetricsError: The measurement does not clear the gate :func:`write_font_metrics` holds it to.
    """
    parser = argparse.ArgumentParser(
        prog="python -m threetears.evals.vega.measure_font", description=(__doc__ or "").splitlines()[0]
    )
    parser.add_argument("--family", default=PACKAGED_FAMILY, help=f"CSS family list (default: {PACKAGED_FAMILY!r})")
    parser.add_argument("--font-dir", type=Path, help="the directory holding the face's font files, registered first")
    parser.add_argument("--out", type=Path, default=METRICS_PATH, help="where to write (default: the packaged table)")
    args = parser.parse_args(argv)
    font_dir = args.font_dir.resolve() if args.font_dir is not None else None
    files = face_files(args.family, font_dir)
    measured = measure(args.family)
    written = write_font_metrics(**measured, font_files=files, path=args.out)
    digits = [measured["advances"][digit] for digit in "0123456789"]
    figures = "tabular" if max(digits) - min(digits) <= 0.001 else "PROPORTIONAL — numeric ticks will not align"
    print(
        f"wrote {written}: {args.family!r}, weights {measured['weights']}, worst corpus ratio "
        f"{measured['worst_ratio']:.4f} on {measured['worst_label']!r}; figures {figures}; "
        f"font files {sorted(files) or 'none (embedded face)'}"
    )
    return 0


__all__ = [
    "CORPUS",
    "EMBEDDED_FACE",
    "FONT_SUFFIXES",
    "PROBE_SIZE",
    "RUN_LENGTH",
    "FontMeasureError",
    "face_files",
    "font_families",
    "main",
    "measure",
    "resolving_files",
]


if __name__ == "__main__":
    sys.exit(main())
