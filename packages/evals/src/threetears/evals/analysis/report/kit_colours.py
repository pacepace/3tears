"""The kit's colours as a reader sees them, in each of the kit's themes: which of some inks reads best on a fill.

A page drawn on the server (the static page's SVG, an arm chip) cannot know the theme it will be shown in, so it
names an ink by token, and the token has to read in light and in dark alike. :func:`ink_on` picks, of the inks
offered, the one whose worse contrast across the kit's two themes is the higher, from the kit's own token values
(``kit/kit.css``): the choice ``kit.js``'s ``kInk`` makes at draw time in the theme the page is shown in. A host's
colours are not known here; the drawn page reads them.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from functools import cache
from importlib.resources import files

__all__ = ["ink_on"]

#: The kit's stylesheet, whose token blocks hold the colours.
_KIT_CSS = files("threetears.evals.analysis.report").joinpath("kit", "kit.css").read_text(encoding="utf-8")

_VAR = re.compile(r"^var\((--[\w-]+)(?:,(.*))?\)$", re.S)
_MIX = re.compile(r"^color-mix\(in (srgb|oklab),\s*(.+?)\s+(\d+(?:\.\d+)?)%,\s*(.+)\)$", re.S)

_RGB = tuple[float, float, float]


@cache
def _themes() -> tuple[dict[str, str], ...]:
    """The kit's tokens in light, then in dark (the light block's with the first dark block's laid over them)."""
    blocks = [
        {name.strip(): value.strip() for name, _, value in (part.partition(":") for part in body.split(";")) if value}
        for body in re.findall(r":root[^{]*\{([^}]*)\}", _KIT_CSS)
    ]
    light = {name: value for name, value in blocks[0].items() if name.startswith("--")}
    dark = light | {name: value for name, value in blocks[1].items() if name.startswith("--")}
    return light, dark


def _srgb(value: str, tokens: dict[str, str]) -> _RGB:
    """A colour the kit writes (a hex, a token, a ``color-mix`` in srgb or oklab) as sRGB channels in 0 to 1."""
    value = value.strip()
    if found := _VAR.match(value):
        name, fallback = found.group(1), found.group(2)
        return _srgb(tokens[name] if name in tokens else str(fallback), tokens)
    if found := _MIX.match(value):
        space, first, share, second = found.groups()
        a, b, p = _srgb(first, tokens), _srgb(second, tokens), float(share) / 100
        if space == "srgb":
            return (a[0] * p + b[0] * (1 - p), a[1] * p + b[1] * (1 - p), a[2] * p + b[2] * (1 - p))
        oka, okb = _oklab(a), _oklab(b)
        return _from_oklab(
            (oka[0] * p + okb[0] * (1 - p), oka[1] * p + okb[1] * (1 - p), oka[2] * p + okb[2] * (1 - p))
        )
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", value):
        raise ValueError(f"the kit writes no colour this reads: {value!r}")
    return (int(value[1:3], 16) / 255, int(value[3:5], 16) / 255, int(value[5:7], 16) / 255)


def _linear(channel: float) -> float:
    return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4


def _gamma(channel: float) -> float:
    encoded = 12.92 * channel if channel <= 0.0031308 else 1.055 * channel ** (1 / 2.4) - 0.055
    return min(1.0, max(0.0, encoded))


def _oklab(rgb: _RGB) -> _RGB:
    r, g, b = (_linear(channel) for channel in rgb)
    l_, m_, s_ = (
        math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b),
        math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b),
        math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b),
    )
    return (
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    )


def _from_oklab(lab: _RGB) -> _RGB:
    lightness, a, b = lab
    l_ = (lightness + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m_ = (lightness - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (lightness - 0.0894841775 * a - 1.2914855480 * b) ** 3
    return (
        _gamma(4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_),
        _gamma(-1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_),
        _gamma(-0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_),
    )


def _contrast(first: _RGB, second: _RGB) -> float:
    """The WCAG contrast ratio of two colours."""

    def luminance(rgb: _RGB) -> float:
        r, g, b = (_linear(channel) for channel in rgb)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    lighter, darker = sorted((luminance(first), luminance(second)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def ink_on(fill: str, inks: Sequence[str]) -> str:
    """The token of ``inks`` that reads best on ``fill`` in whichever of the kit's themes it reads worse in; the first
    offered where two read alike, and where ``fill`` names a token the kit does not set (a browser draws that fill
    as nothing, so no ink is judged against it).

    :param fill: The fill, as the drawing writes it: ``var(--s1)``, or ``color-mix(in oklab, var(--div-pos) 74%,
        var(--surface))``.
    :ptype fill: str
    :param inks: The ink tokens offered, by name (``--ink``).
    :ptype inks: Sequence[str]
    :return: The ink's token.
    :rtype: str
    """

    light, _ = _themes()
    if any(name not in light for name in re.findall(r"var\((--[\w-]+)", fill)):
        return inks[0]

    def worst(ink: str) -> float:
        return min(_contrast(_srgb(f"var({ink})", tokens), _srgb(fill, tokens)) for tokens in _themes())

    best = inks[0]
    for ink in inks[1:]:
        if worst(ink) > worst(best):
            best = ink
    return best
