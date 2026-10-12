"""Rasterise a compiled chart spec, for surfaces that cannot run a browser.

The same spec the browser embeds, drawn here through vl-convert — which runs Vega
in an embedded JS runtime and rasterises with resvg, so no browser and no Node are
needed at runtime. That "same spec" is the whole reason this module is thin: every
decision about what the chart says was made by the compiler, and anything decided
here would be a second answer only one surface ever sees.

**Size is one of those decisions, and it did not use to be.** Specs once carried
``"width": "container"`` and each renderer substituted the width it had — a fixed
number here, the measured element there — so the two surfaces drew one chart at
two sizes by design. The compiler now emits fixed dimensions: a plot handed its
container's width overflows by exactly the gutter its axis labels are drawn in,
and height that follows the row count while width follows the card makes three
bars 12px tall in a 1400px plot. Nothing is resized here, and a server-rendered
PNG comes out at the dimensions the browser draws.

Two facts about the rasteriser shape this module, both established by rendering
rather than by reading documentation:

- **It renders unparseable colour as black, silently.** Hence the palette module's
  resolved-hex contract, and hence the caller getting a real error from it rather
  than a chart nobody can tell is wrong.
- **It resolves fonts from what it embeds, the registered directories, and the
  machine's own font database — and draws NOTHING for a family none of them has.**
  A single-family list naming a face that is absent is not substituted: the text
  marks are simply not painted, while the layout still measures them (measured,
  vl-convert 1.9). Within a list it falls through to the next family. vl-convert
  embeds exactly one face, Liberation Sans, which is why the packaged chart face is
  that one (:func:`~threetears.evals.vega.text_metrics.packaged_font`): it draws on
  every host, including a slim container with no font package at all, and needs no
  directory registered.

  A host face is the other case. Registering its directory is what makes it
  available where the machine offers nothing, and the hazard is hard to see: on a
  developer machine that has the face installed, a render with nothing registered
  came back byte-identical to one with the directory registered. So a missing face
  shows up only in the deployed image. The gate for that is not a render either:
  vl-convert writes the configured family into the SVG's ``font-family`` whether or
  not any file provides it, so the rendered document cannot tell the two cases apart.

**The font directory is injected; the palette and the default face are packaged.**
The palette is a small JSON file that ships inside the package and is read from the
module's own directory (:data:`~threetears.evals.vega.palette._PALETTE_PATH`), and the
default face is embedded in the rasteriser, with its measured metrics shipped beside
the palette. A host's own face cannot sensibly live in this package — it is
megabytes of licensed files, it belongs to whoever owns it, and two products would
need two of them — so it arrives as :func:`register_fonts`, or as the ``font_dir``
argument to a render call, beside the host's
:class:`~threetears.evals.kernel.host.ChartFont`, which carries its family and its
measured advances.

**A host face drawn with no directory is loud, because its consequence is
invisible.** If the machine lacks the face, the chart is a well-formed PNG of the
right size with its text drawn in a fallback or not at all — the same shape as the
black-series hazard above. A render in a font other than the packaged one that
reaches the rasteriser with no directory registered anywhere in this process logs a
warning naming the consequence and the argument that fixes it, **once per process**,
because the fact it reports is a property of the process rather than of the chart:
repeating it per render would be noise readers learn to filter.

**A host face is drawn only from the files it was measured from.** Its
:class:`~threetears.evals.kernel.host.ChartFont` records the SHA-256 of each file that
provided the face when it was measured, and a render that registers a directory — or
runs after one was registered — refuses the font when those files are not there with
those bytes (:func:`~threetears.evals.vega.text_metrics.check_font_files`). The table's
advances describe the bytes measured; a face updated in place keeps its family name, so
nothing else would notice that labels are being fitted against widths it no longer has.

**The font measurer draws through here too** (:func:`vega_svg`, :func:`vega_png`): it
probes raw Vega text marks rather than compiled Vega-Lite specs, and keeping those two
calls in this module keeps the rasteriser in one module of the renderer.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from threetears.evals.analysis.viz.renderer import ChartRefused
from threetears.evals.kernel.host import ChartFont, ChartPalette
from threetears.evals.vega.palette import vega_config
from threetears.evals.vega.text_metrics import check_font_files, packaged_font
from threetears.evals.schema.logs import get_logger

log = get_logger(__name__)

# ``vl_convert`` is imported inside the calls that reach the rasteriser and nowhere
# at module level. It is the ``[vega]`` extra's one dependency (a native extension, and the
# only third-party import the package matrix allows this adapter), and this module is
# re-exported from the ``threetears.evals.vega`` root: imported here at module level,
# compiling a spec for a browser to draw would require the rasteriser too.

#: Device-pixel multiplier for the PNG. 2 is a retina-legible chart that still fits
#: a chat transcript; the SVG path exists for when a caller wants it resolution-free.
DEFAULT_SCALE = 2

_font_lock = threading.Lock()

#: Directories already handed to the rasteriser in this process, resolved.
#:
#: A set rather than a flag, because the directory is now the caller's to name: two
#: hosts in one process, or a test registering a stand-in beside the real one, are
#: both legitimate and neither is served by "fonts have been registered".
_registered_font_dirs: set[Path] = set()

#: Whether any directory has actually reached the rasteriser in this process.
#:
#: Not derivable from :data:`_registered_font_dirs`, which records *attempts* so a
#: repeated call is cheap: a directory that turned out to be absent is on that set
#: and registered nothing. This flag is the one that answers "will this chart draw
#: in a face somebody chose?", which is what the unfonted-render warning turns on.
_fonts_registered = False

#: Whether the unfonted-render warning has been emitted in this process. See the
#: module docstring for why it is stated once rather than per render.
_warned_unfonted = False


def register_fonts(font_dir: Path) -> None:
    """Make a directory of font files available to the rasteriser.

    Idempotent per directory and lock-guarded: the render path is reachable from
    concurrent request handlers, and vl-convert's font registry is process-global
    state rather than per-call.

    An absent directory is a warning, not an error. The chart still draws, in
    whatever face the machine's font database resolves — the host's own on a machine
    that has it installed, a substitute or no text at all in a slim container, which
    is the only place this bites and the one place nobody is looking. The difference is
    invisible in the output, so it would otherwise be discovered by someone
    wondering why an exported chart looks unlike the report.

    Args:
        font_dir: Directory holding the font files to register.
    """
    global _fonts_registered

    resolved = font_dir.resolve()
    if resolved in _registered_font_dirs:
        return
    with _font_lock:
        if resolved in _registered_font_dirs:
            return
        if resolved.is_dir():
            import vl_convert as vlc

            vlc.register_font_directory(str(resolved))
            _fonts_registered = True
        else:
            log.warning(
                "chart font directory %s is absent — server-rendered charts will use a fallback typeface", resolved
            )
        _registered_font_dirs.add(resolved)


def _prepare_fonts(font_dir: Path | None, font: ChartFont | None = None) -> None:
    """Register the caller's font directory, or say what drawing a host face without one means.

    The two branches are the two ways a render can end up in a substitute face, and
    both are reported. A named-but-absent directory is :func:`register_fonts`'
    warning; a render in a host's own face that names none, in a process where none
    has ever been registered, is this one. The packaged face is embedded in the
    rasteriser, so drawing it with no directory is the ordinary case and says nothing.

    The condition is "nothing registered in this process", not "this call passed
    ``None``": vl-convert's registry is process-global, so a render that names no
    directory after the host registered one still draws in the host's face and has
    nothing to warn about.

    A font that records the files it was measured from is checked against the
    directory named, before it is registered, or — when none is named — against every
    directory registered in this process; with none registered there is nothing to
    check against, and the warning stands for that.

    Args:
        font_dir: The directory the caller named, or ``None`` for "draw in whatever
            this host's font database resolves".
        font: The typeface the render is set in; ``None`` for the packaged face.

    Raises:
        TextMetricsError: ``font``'s recorded files are not in the directories it would
            be drawn from with the bytes it was measured from.
    """
    global _warned_unfonted

    if font_dir is not None:
        if font is not None:
            check_font_files(font, [font_dir.resolve()])
        register_fonts(font_dir)
        return
    if font is None or font == packaged_font():
        return
    if _fonts_registered and font.font_files:
        check_font_files(font, sorted(directory for directory in _registered_font_dirs if directory.is_dir()))
        return
    if _fonts_registered or _warned_unfonted:
        return
    with _font_lock:
        if _fonts_registered or _warned_unfonted:
            return
        _warned_unfonted = True
        log.warning(
            "rendering a chart in %r with no font directory registered — it is drawn in that face only if "
            "this machine has it installed, and otherwise in a fallback or with no text at all, which is not "
            "visible in the output. Pass font_dir=, or call register_fonts(), to name one. "
            "Said once per process.",
            font.measured_face,
        )


#: How vl-convert words its own refusal to convert a spec — the JavaScript runtime would not draw it — as against a
#: string it could not parse or an argument it could not take, each of which is a defect in what called it.
_CONVERSION_FAILED = re.compile(r"^(?:Vega-Lite|Vega) to \w+ conversion failed:")


class RasterRefused(ChartRefused):
    """vl-convert refused to convert a chart's spec: its runtime would not draw it.

    Raised only for vl-convert's own conversion failure (``Vega-Lite to PNG conversion failed: ...``), whose text
    it carries. Every other ``ValueError`` vl-convert raises — a spec string it cannot parse as JSON (a ``NaN`` the
    compiler let through), an argument it does not take — is a defect in the code that called it and propagates as
    it was raised.
    """


def _converted[T](convert: Callable[[], T]) -> T:
    """Run one vl-convert conversion, raising :class:`RasterRefused` for its refusal and nothing else.

    Args:
        convert: The conversion.

    Returns:
        What it returns.

    Raises:
        RasterRefused: vl-convert reported the conversion failed.
    """
    try:
        return convert()
    except ValueError as failed:
        if _CONVERSION_FAILED.match(str(failed)) is None:
            raise
        raise RasterRefused(str(failed)) from failed


def render_png(
    spec: dict[str, Any],
    *,
    palette: ChartPalette,
    scale: int = DEFAULT_SCALE,
    font_dir: Path | None = None,
    font: ChartFont | None = None,
) -> bytes:
    """Rasterise a compiled spec to PNG.

    Args:
        spec: The colourless Vega-Lite spec.
        palette: The colours to draw it in — a host's, or ``packaged_palette(theme)``.
        scale: Device-pixel multiplier.
        font_dir: Directory of font files to register before drawing, per
            :func:`register_fonts`. ``None`` registers nothing — all the packaged
            face needs, and warned about once per process for a host face, per
            :func:`_prepare_fonts`.
        font: The typeface the spec was laid out in; ``None`` for the packaged face.

    Returns:
        PNG bytes.

    Raises:
        TextMetricsError: ``font`` was measured from files its directory no longer holds as measured.
        RasterRefused: vl-convert refused to convert the spec.
    """
    import vl_convert as vlc

    _prepare_fonts(font_dir, font)
    return _converted(lambda: vlc.vegalite_to_png(json.dumps(spec), config=vega_config(palette, font), scale=scale))


def render_svg(
    spec: dict[str, Any], *, palette: ChartPalette, font_dir: Path | None = None, font: ChartFont | None = None
) -> str:
    """Render a compiled spec to SVG.

    Args:
        spec: The colourless Vega-Lite spec.
        palette: The colours to draw it in — a host's, or ``packaged_palette(theme)``.
        font_dir: Directory of font files to register before drawing, per
            :func:`register_fonts`. ``None`` registers nothing — all the packaged
            face needs, and warned about once per process for a host face, per
            :func:`_prepare_fonts`.
        font: The typeface the spec was laid out in; ``None`` for the packaged face.

    Returns:
        The SVG document.

    Raises:
        TextMetricsError: ``font`` was measured from files its directory no longer holds as measured.
        RasterRefused: vl-convert refused to convert the spec.
    """
    import vl_convert as vlc

    _prepare_fonts(font_dir, font)
    return _converted(lambda: vlc.vegalite_to_svg(json.dumps(spec), config=vega_config(palette, font)))


def vega_svg(spec: dict[str, Any]) -> str:
    """Draw a raw Vega spec (not Vega-Lite) to SVG, unthemed, in whatever fonts are registered.

    What the font measurer probes the rasteriser's text layout with: a Vega text mark sized to fit,
    whose SVG width is Vega's own measurement of the string.

    Args:
        spec: A Vega spec.

    Returns:
        The SVG document.
    """
    import vl_convert as vlc

    return vlc.vega_to_svg(json.dumps(spec))


def vega_png(spec: dict[str, Any]) -> bytes:
    """Rasterise a raw Vega spec (not Vega-Lite) to PNG, unthemed, in whatever fonts are registered.

    What the font measurer compares to tell a face that draws from one the rasteriser cannot find,
    which it lays out but does not paint.

    Args:
        spec: A Vega spec.

    Returns:
        PNG bytes.
    """
    import vl_convert as vlc

    return vlc.vega_to_png(json.dumps(spec))


__all__ = [
    "DEFAULT_SCALE",
    "RasterRefused",
    "register_fonts",
    "render_png",
    "render_svg",
    "vega_png",
    "vega_svg",
]
