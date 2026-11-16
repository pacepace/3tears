"""The icon set an installable app needs, rendered from one square source image.

What each platform reads, and so what the set holds:

* Chrome, Edge and Firefox read the manifest's ``icons``: a 192 and a 512 px PNG (Chrome's install
  criteria), plus ``maskable`` variants that Android and Chrome OS crop to their own shape.
* iOS and iPadOS read none of the manifest's icons for the home screen. They take
  ``<link rel="apple-touch-icon">``, a 180 px PNG that must be opaque: transparency turns black.
* The browser tab takes ``<link rel="icon">``: a 32 px PNG, and the SVG itself when the source was one.

A maskable icon is cropped to anything from a circle to a squircle, and only the centre circle of
80 % of its width is guaranteed visible. So it is the source scaled into that circle
(``maskable_scale``) over a solid background colour. The apple-touch icon is composed the same way:
iOS rounds its corners, and an opaque square with the mark inside reads as an app icon rather than a
cropped picture. A round mark fits the safe circle at 0.8; a square picture needs about 0.6.

Output is deterministic: the same source and options give byte-identical files (Pillow writes no
timestamps into a PNG), so a content hash names a version and the URLs can be cached for good.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Final

from PIL import Image, UnidentifiedImageError

__all__ = [
    "APPLE_TOUCH_ICON",
    "FAVICON",
    "SVG_ICON",
    "IconSet",
    "render_icons",
]

#: the file iOS takes for its home screen
APPLE_TOUCH_ICON: Final[str] = "apple-touch-icon.png"
#: the browser tab icon
FAVICON: Final[str] = "favicon-32.png"
#: the source SVG itself, when the set was rendered from one
SVG_ICON: Final[str] = "icon.svg"

#: the smallest source that renders the 512 px icon without upscaling
MIN_SOURCE_PX: Final[int] = 512
#: the largest source accepted: well past any icon size, and a bound on decoding work
MAX_SOURCE_PX: Final[int] = 4096

_APPLE_PX: Final[int] = 180
_FAVICON_PX: Final[int] = 32
_SIZES: Final[tuple[int, ...]] = (192, 512)


@dataclass(frozen=True, slots=True)
class IconFile:
    """One file of the set.

    :param name: the file name, stable across versions (``icon-512.png``)
    :param body: the file's bytes
    :param media_type: its Content-Type
    :param sizes: the manifest ``sizes`` value (``512x512``, or ``any`` for an SVG)
    :param purpose: the manifest ``purpose`` (``any``, ``maskable``, ``monochrome``), or ``None``
        for a file the manifest does not list
    """

    name: str
    body: bytes
    media_type: str
    sizes: str
    purpose: str | None

    @property
    def version(self) -> str:
        """A short content hash: the ``v`` query that makes a URL cacheable for good."""
        return hashlib.sha256(self.body).hexdigest()[:12]

    @property
    def etag(self) -> str:
        """A strong ETag for the bytes."""
        return f'"{hashlib.sha256(self.body).hexdigest()[:32]}"'


@dataclass(frozen=True, slots=True)
class IconSet(Mapping[str, IconFile]):
    """The rendered set, by file name."""

    files: tuple[IconFile, ...]

    def __getitem__(self, name: str) -> IconFile:
        for icon in self.files:
            if icon.name == name:
                return icon
        raise KeyError(name)

    def __iter__(self) -> Iterator[str]:
        return (icon.name for icon in self.files)

    def __len__(self) -> int:
        return len(self.files)

    def url(self, name: str, base_url: str, *, versioned: bool = True) -> str:
        """Where ``name`` is served under ``base_url`` (a directory URL), with ``?v=<hash>`` if
        ``versioned``.

        :raises KeyError: ``name`` is not in the set
        """
        icon = self[name]
        url = f"{base_url.rstrip('/')}/{icon.name}"
        return f"{url}?v={icon.version}" if versioned else url

    def manifest_icons(self, base_url: str, *, versioned: bool = True) -> list[dict[str, str]]:
        """The manifest's ``icons`` list: every file with a purpose, served under ``base_url``."""
        return [
            {
                "src": self.url(icon.name, base_url, versioned=versioned),
                "sizes": icon.sizes,
                "type": icon.media_type,
                "purpose": icon.purpose,
            }
            for icon in self.files
            if icon.purpose is not None
        ]


def _png(image: Image.Image) -> bytes:
    out = io.BytesIO()
    image.save(out, format="PNG", optimize=False, compress_level=9)
    return out.getvalue()


def _resized(source: Image.Image, px: int) -> Image.Image:
    return source if source.width == px else source.resize((px, px), Image.Resampling.LANCZOS)


def _composed(source: Image.Image, px: int, background: str, scale: float) -> Image.Image:
    """``source`` scaled to ``scale`` of ``px``, centred on an opaque ``background`` square."""
    canvas = Image.new("RGBA", (px, px), background)
    inner = max(1, round(px * scale))
    mark = _resized(source, inner)
    offset = (px - inner) // 2
    canvas.alpha_composite(mark, (offset, offset))
    return canvas.convert("RGB")


def _monochrome(source: Image.Image, px: int) -> Image.Image:
    """White where the source is opaque, keeping its alpha: what ``purpose: monochrome`` reads."""
    alpha = _resized(source, px).getchannel("A")
    white = Image.new("RGBA", (px, px), (255, 255, 255, 0))
    white.putalpha(alpha)
    return white


def _open_source(png: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(png)) as probe:
            fmt = probe.format
            width, height = probe.size
            if fmt != "PNG":
                raise ValueError(f"the icon source must be a PNG, not {fmt}")
            if width != height:
                raise ValueError(f"the icon source must be square, not {width}x{height}")
            if not MIN_SOURCE_PX <= width <= MAX_SOURCE_PX:
                raise ValueError(f"the icon source must be {MIN_SOURCE_PX} to {MAX_SOURCE_PX} px square, not {width}")
            probe.load()
            return probe.convert("RGBA")
    except UnidentifiedImageError as exc:
        raise ValueError("the icon source is not an image Pillow can read") from exc


def render_icons(
    png: bytes,
    *,
    background: str,
    maskable_scale: float = 0.8,
    monochrome: bool = False,
    svg: bytes | None = None,
) -> IconSet:
    """Render the set from a square PNG.

    :param png: the source, square, 512 to 4096 px; transparency is kept in the ``any`` icons
    :ptype png: bytes
    :param background: the colour behind the mark in the maskable and apple-touch icons (CSS hex)
    :ptype background: str
    :param maskable_scale: the mark's width as a fraction of the maskable and apple-touch icons;
        0.8 fills the maskable safe circle with a round mark, about 0.6 fits a square one
    :ptype maskable_scale: float
    :param monochrome: also render ``purpose: monochrome`` (a white silhouette from the alpha);
        useful only for a mark on a transparent background
    :ptype monochrome: bool
    :param svg: the SVG the PNG was rendered from, served as ``icon.svg`` (sizes ``any``)
    :ptype svg: bytes | None
    :return: the set
    :rtype: IconSet
    :raises ValueError: a source that is not a square PNG in range, or a scale outside (0, 1]
    """
    if not 0 < maskable_scale <= 1:
        raise ValueError(f"maskable_scale must be in (0, 1]: {maskable_scale}")
    source = _open_source(png)
    try:
        Image.new("RGB", (1, 1), background)
    except ValueError as exc:
        raise ValueError(f"background must be a colour Pillow reads, e.g. '#07060c': {background!r}") from exc
    files: list[IconFile] = []
    for px in _SIZES:
        files.append(IconFile(f"icon-{px}.png", _png(_resized(source, px)), "image/png", f"{px}x{px}", "any"))
    for px in _SIZES:
        body = _png(_composed(source, px, background, maskable_scale))
        files.append(IconFile(f"icon-maskable-{px}.png", body, "image/png", f"{px}x{px}", "maskable"))
    if monochrome:
        files.append(
            IconFile("icon-monochrome-512.png", _png(_monochrome(source, 512)), "image/png", "512x512", "monochrome")
        )
    if svg is not None:
        files.append(IconFile(SVG_ICON, svg, "image/svg+xml", "any", "any"))
    files.append(
        IconFile(
            APPLE_TOUCH_ICON,
            _png(_composed(source, _APPLE_PX, background, maskable_scale)),
            "image/png",
            f"{_APPLE_PX}x{_APPLE_PX}",
            None,
        )
    )
    files.append(
        IconFile(FAVICON, _png(_resized(source, _FAVICON_PX)), "image/png", f"{_FAVICON_PX}x{_FAVICON_PX}", None)
    )
    return IconSet(tuple(files))
