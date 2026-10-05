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
- **It resolves fonts from the registered directories AND the host's own font
  database.** Registering a font directory is what makes a brand face available
  where the host offers nothing — a slim container image typically carries no font
  package at all. It is also why the hazard is hard to see: on a developer machine
  that has the brand face installed, a render with nothing registered came back
  byte-identical to one with the directory registered (measured, macOS,
  vl-convert 1.9). So the fallback shows up only in the deployed image, where a
  server-rendered chart would silently draw in a substitute face while the web
  report uses the brand one.

  That asymmetry is why the gate for it is not a render. The host that supplies the
  directory checks that it exists and declares the family
  :func:`~threetears.evals.vega.palette.vega_config` asks for, rather than looking
  for that family name in the output: vl-convert writes the configured family
  into the SVG's ``font-family`` whether or not any file provides it, so the
  rendered document cannot tell the two cases apart.

**The font directory is injected; the palette is packaged.** The two assets this
subsystem needs are solved by opposite mechanisms on purpose. The palette is a
small JSON file that ships inside the package and is read from the module's own
directory (:data:`~threetears.evals.vega.palette._PALETTE_PATH`). A directory of font
binaries cannot sensibly live in a Python package — it is megabytes of licensed
files, it belongs to whoever owns the brand, and the same package rendering for
two products would need two of them — so it arrives as
:func:`register_fonts`, or as the ``font_dir`` argument to a render call, with the
host naming the directory it wants used. Passing nothing registers nothing and
draws in whatever the host's font database resolves, which is the honest default
for a package that owns no typeface.

**That default is loud, because its consequence is invisible.** A chart drawn in a
substitute face is a well-formed PNG of the right size that simply does not match
the report it sits beside, so nothing downstream can notice — the same shape as
the black-series hazard above, and the reason neither is left to be discovered.
A render that reaches the rasteriser with no directory registered anywhere in this
process logs a warning naming the consequence and the argument that fixes it. It
is stated **once per process**, because the fact it reports is a property of the
process rather than of the chart: repeating it per render would be noise readers
learn to filter, which is the silence it exists to break. A host that owns no
typeface and wants none still gets one line saying so, which is the difference
between a stated choice and an omission.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from threetears.evals.vega.palette import Theme, vega_config
from threetears.observe import get_logger

log = get_logger(__name__)

# ``vl_convert`` is imported inside the three calls that reach the rasteriser and nowhere
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
    whatever face the host's font database resolves — the brand one on a machine
    that has it installed, a substitute in a slim container, which is the only
    place this bites and the one place nobody is looking. The difference is
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


def _prepare_fonts(font_dir: Path | None) -> None:
    """Register the caller's font directory, or say what drawing without one means.

    The two branches are the two ways a render can end up in a substitute face, and
    both are reported. A named-but-absent directory is :func:`register_fonts`'
    warning; a render that names none, in a process where none has ever been
    registered, is this one — the case that was previously silent, and the only one
    where nothing in the call even mentions a typeface.

    The condition is "nothing registered in this process", not "this call passed
    ``None``": vl-convert's registry is process-global, so a render that names no
    directory after the host registered one still draws in the host's face and has
    nothing to warn about.

    Args:
        font_dir: The directory the caller named, or ``None`` for "draw in whatever
            this host's font database resolves".
    """
    global _warned_unfonted

    if font_dir is not None:
        register_fonts(font_dir)
        return
    if _fonts_registered or _warned_unfonted:
        return
    with _font_lock:
        if _fonts_registered or _warned_unfonted:
            return
        _warned_unfonted = True
        log.warning(
            "rendering a chart with no font directory registered — it will draw in whatever typeface this "
            "host's font database resolves, which is not the brand face the web report uses and is not "
            "visible in the output. Pass font_dir=, or call register_fonts(), to name one. "
            "Said once per process."
        )


def render_png(
    spec: dict[str, Any],
    *,
    theme: Theme = "dark",
    scale: int = DEFAULT_SCALE,
    font_dir: Path | None = None,
) -> bytes:
    """Rasterise a compiled spec to PNG.

    Args:
        spec: The colourless Vega-Lite spec.
        theme: Which palette to draw it in.
        scale: Device-pixel multiplier.
        font_dir: Directory of font files to register before drawing, per
            :func:`register_fonts`. ``None`` registers nothing and draws in
            whatever the host's font database resolves — permitted, and warned
            about once per process, per :func:`_prepare_fonts`.

    Returns:
        PNG bytes.
    """
    import vl_convert as vlc

    _prepare_fonts(font_dir)
    return vlc.vegalite_to_png(json.dumps(spec), config=vega_config(theme), scale=scale)


def render_svg(spec: dict[str, Any], *, theme: Theme = "dark", font_dir: Path | None = None) -> str:
    """Render a compiled spec to SVG.

    Args:
        spec: The colourless Vega-Lite spec.
        theme: Which palette to draw it in.
        font_dir: Directory of font files to register before drawing, per
            :func:`register_fonts`. ``None`` registers nothing and draws in
            whatever the host's font database resolves — permitted, and warned
            about once per process, per :func:`_prepare_fonts`.

    Returns:
        The SVG document.
    """
    import vl_convert as vlc

    _prepare_fonts(font_dir)
    return vlc.vegalite_to_svg(json.dumps(spec), config=vega_config(theme))


__all__ = [
    "DEFAULT_SCALE",
    "register_fonts",
    "render_png",
    "render_svg",
]
