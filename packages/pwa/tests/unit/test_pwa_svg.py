"""An SVG source renders through resvg into the same set, with the SVG itself served too."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from threetears.pwa import SVG_ICON
from threetears.pwa.svg import render_svg, render_svg_icons

_MARK = b"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48">
<circle cx="24" cy="24" r="21" fill="#7c5cff"/><circle cx="24" cy="24" r="4" fill="#ffffff"/></svg>"""


def test_render_svg_gives_a_square_png_of_the_size_asked() -> None:
    png = render_svg(_MARK, 600)
    image = Image.open(io.BytesIO(png))
    assert (image.format, image.size) == ("PNG", (600, 600))
    assert image.convert("RGBA").getpixel((300, 300)) == (255, 255, 255, 255)
    assert image.convert("RGBA").getpixel((0, 0))[3] == 0  # type: ignore[index]


def test_svg_icons_are_the_full_set_plus_the_svg_and_deterministic() -> None:
    first = render_svg_icons(_MARK, background="#07060c", monochrome=True)
    second = render_svg_icons(_MARK, background="#07060c", monochrome=True)
    assert first[SVG_ICON].body == _MARK
    assert "icon-maskable-512.png" in first and "apple-touch-icon.png" in first
    assert [f.body for f in first.files] == [f.body for f in second.files]


@pytest.mark.parametrize(
    ("svg", "px"),
    [(b"<svg", 64), (_MARK, 0), (_MARK, 5000), (b"\xff\xfe", 64)],
    ids=["unparseable", "zero-px", "too-big", "not-utf8"],
)
def test_bad_input_is_a_value_error(svg: bytes, px: int) -> None:
    with pytest.raises(ValueError):
        render_svg(svg, px)
