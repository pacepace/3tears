"""An icon set from an SVG source (the ``svg`` extra: ``pip install '3tears-pwa[svg]'``).

The SVG is rendered by resvg, a self-contained renderer: no cairo or other system library, and
the same SVG gives the same pixels on every machine. System fonts are not loaded, so a mark that
uses ``<text>`` renders without it: convert text to paths first. External references
(``<image href>`` to a file or URL) are not resolved either; embed them as data URLs.
"""

from __future__ import annotations

from typing import Final

import resvg_py

from threetears.pwa.icons import MAX_SOURCE_PX, IconSet, render_icons

__all__ = ["render_svg_icons"]

#: the raster size the icon set is rendered from: twice the largest icon, for clean downsampling
_RASTER_PX: Final[int] = 1024


def render_svg(svg: bytes, px: int = _RASTER_PX) -> bytes:
    """``svg`` rendered to a ``px`` square PNG.

    :param svg: the SVG document, UTF-8; it should be square (its viewBox is stretched to fit)
    :ptype svg: bytes
    :param px: the output's width and height
    :ptype px: int
    :return: the PNG
    :rtype: bytes
    :raises ValueError: an SVG resvg cannot parse, or a size out of range
    """
    if not 1 <= px <= MAX_SOURCE_PX:
        raise ValueError(f"px must be 1 to {MAX_SOURCE_PX}: {px}")
    try:
        text = svg.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("the SVG must be UTF-8") from exc
    return bytes(resvg_py.svg_to_bytes(svg_string=text, width=px, height=px, skip_system_fonts=True))


def render_svg_icons(
    svg: bytes,
    *,
    background: str,
    maskable_scale: float = 0.8,
    monochrome: bool = False,
) -> IconSet:
    """The icon set rendered from ``svg``, which is also served as ``icon.svg``.

    Options are :func:`threetears.pwa.icons.render_icons`'s.

    :raises ValueError: as :func:`render_svg` and :func:`threetears.pwa.icons.render_icons`
    """
    return render_icons(
        render_svg(svg),
        background=background,
        maskable_scale=maskable_scale,
        monochrome=monochrome,
        svg=svg,
    )
