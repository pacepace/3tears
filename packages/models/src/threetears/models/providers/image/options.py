"""Size, aspect and seed as the image providers read them, and what they report back.

Every provider turns the same ``size`` / ``aspect`` arguments into a width and a
height, and every one says what it did in the result's metadata under the
``GENERATION_*`` keys of :mod:`threetears.media.contracts`. Both live here so the
five providers read and report them one way.
"""

from __future__ import annotations

import re
from typing import Any

from threetears.media.contracts import GENERATION_IGNORED, GENERATION_SEED, GENERATION_SIZE, GENERATION_SOURCE_USED

__all__ = [
    "ASPECTS",
    "dimensions_for",
    "parse_size",
    "result_metadata",
]

#: The shapes ``aspect`` takes by name, as width over height.
ASPECTS: dict[str, tuple[int, int]] = {
    "square": (1, 1),
    "landscape": (3, 2),
    "portrait": (2, 3),
    "wide": (16, 9),
    "tall": (9, 16),
}

_SIZE = re.compile(r"^\s*(\d{2,5})\s*[x×*]\s*(\d{2,5})\s*$", re.IGNORECASE)
_RATIO = re.compile(r"^\s*(\d{1,3})\s*[:/x]\s*(\d{1,3})\s*$", re.IGNORECASE)


def _ratio(aspect: str) -> tuple[int, int]:
    """Width and height of a named shape or a written ratio such as ``16:9``.

    :param aspect: the shape as written
    :ptype aspect: str
    :return: width and height of the ratio
    :rtype: tuple[int, int]
    :raises ValueError: when the aspect is neither, with the forms taken
    """
    named = ASPECTS.get(aspect.strip().lower())
    if named:
        return named
    m = _RATIO.match(aspect)
    if not m or int(m.group(1)) == 0 or int(m.group(2)) == 0:
        raise ValueError(
            f"aspect {aspect!r} is not a shape: write square, landscape, portrait, wide, tall or a ratio such as 16:9"
        )
    return int(m.group(1)), int(m.group(2))


def parse_size(size: str) -> tuple[int, int]:
    """Width and height of a size written ``WIDTHxHEIGHT``.

    :param size: the size as written, e.g. ``"1024x1536"``
    :ptype size: str
    :return: width and height
    :rtype: tuple[int, int]
    :raises ValueError: when it is not ``WIDTHxHEIGHT``
    """
    m = _SIZE.match(size)
    if not m:
        raise ValueError(f"size {size!r} is not WIDTHxHEIGHT, e.g. 1024x1536")
    return int(m.group(1)), int(m.group(2))


def dimensions_for(
    size: str | None,
    aspect: str | None,
    *,
    default: tuple[int, int],
    longest: int,
) -> tuple[int, int]:
    """The width and height to ask a service for: from a size, else an aspect, else the default.

    An aspect keeps the default's longest side and rounds each side to a multiple
    of 64, which every diffusion model takes.

    :param size: ``WIDTHxHEIGHT`` as written, or ``None``
    :ptype size: str | None
    :param aspect: a named shape or a ratio, or ``None``
    :ptype aspect: str | None
    :param default: the provider's configured width and height
    :ptype default: tuple[int, int]
    :param longest: the largest side the service takes
    :ptype longest: int
    :return: width and height
    :rtype: tuple[int, int]
    :raises ValueError: when the size or aspect cannot be read, or the size is past the largest side
    """
    if size:
        width, height = parse_size(size)
        if max(width, height) > longest:
            raise ValueError(f"size {size} is larger than this model makes: at most {longest} on a side")
        return width, height
    if aspect:
        w, h = _ratio(aspect)
        side = float(max(default))
        long_w, long_h = (side, side * h / w) if w >= h else (side * w / h, side)
        return max(64, int(round(long_w / 64) * 64)), max(64, int(round(long_h / 64) * 64))
    return default


def result_metadata(
    base: dict[str, Any],
    *,
    size: str | None,
    seed: int | None,
    source_given: bool,
    source_used: bool,
    ignored: dict[str, str],
) -> dict[str, Any]:
    """The metadata a provider returns: its own keys plus what it did with what it was given.

    :param base: the provider's own keys (model, ...)
    :ptype base: dict[str, Any]
    :param size: the size the image was made at, ``WIDTHxHEIGHT``, when known
    :ptype size: str | None
    :param seed: the seed it was made with, when the service has one
    :ptype seed: int | None
    :param source_given: whether a source image was passed
    :ptype source_given: bool
    :param source_used: whether that source shaped the result
    :ptype source_used: bool
    :param ignored: each option it could not apply, with why
    :ptype ignored: dict[str, str]
    :return: the metadata
    :rtype: dict[str, Any]
    """
    metadata = dict(base)
    if size:
        metadata[GENERATION_SIZE] = size
    if seed is not None:
        metadata[GENERATION_SEED] = seed
    if source_given:
        metadata[GENERATION_SOURCE_USED] = source_used
    if ignored:
        metadata[GENERATION_IGNORED] = dict(ignored)
    return metadata
