"""The icon set: every file each platform reads, opaque where iOS needs it, and deterministic."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from threetears.pwa import APPLE_TOUCH_ICON, FAVICON, SVG_ICON, render_icons


def _source(px: int = 1024, *, mode: str = "RGBA", fmt: str = "PNG", height: int | None = None) -> bytes:
    """A red disc on transparency: a round mark, as a station or app icon usually is."""
    image = Image.new(mode, (px, height or px), (0, 0, 0, 0) if mode == "RGBA" else (0, 0, 0))
    pixels = image.load()
    assert pixels is not None
    r = px // 2
    for y in range(height or px):
        for x in range(px):
            if (x - r) ** 2 + (y - r) ** 2 <= (r - 1) ** 2:
                pixels[x, y] = (220, 30, 30, 255) if mode == "RGBA" else (220, 30, 30)
    out = io.BytesIO()
    image.save(out, format=fmt)
    return out.getvalue()


@pytest.fixture(scope="module")
def source() -> bytes:
    return _source(512)


def _image(body: bytes) -> Image.Image:
    return Image.open(io.BytesIO(body))


def test_the_set_holds_what_each_platform_reads(source: bytes) -> None:
    icons = render_icons(source, background="#07060c")
    assert set(icons) == {
        "icon-192.png",
        "icon-512.png",
        "icon-maskable-192.png",
        "icon-maskable-512.png",
        APPLE_TOUCH_ICON,
        FAVICON,
    }
    for name, px in [("icon-192.png", 192), ("icon-512.png", 512), (APPLE_TOUCH_ICON, 180), (FAVICON, 32)]:
        assert _image(icons[name].body).size == (px, px), name
        assert icons[name].media_type == "image/png"


def test_any_icons_keep_transparency_and_maskable_and_apple_are_opaque(source: bytes) -> None:
    icons = render_icons(source, background="#07060c")
    corner = (0, 0)
    assert _image(icons["icon-512.png"].body).convert("RGBA").getpixel(corner)[3] == 0  # type: ignore[index]
    for name in ("icon-maskable-512.png", APPLE_TOUCH_ICON):
        image = _image(icons[name].body)
        assert image.mode == "RGB", f"{name}: iOS turns transparency black"
        assert image.getpixel(corner) == (7, 6, 12), name


def test_maskable_keeps_the_mark_inside_the_safe_circle(source: bytes) -> None:
    """Only the centre circle of 80 % of the width survives every mask shape."""
    maskable = _image(render_icons(source, background="#000000", maskable_scale=0.8)["icon-maskable-512.png"].body)
    # just outside the safe circle, on the horizontal axis: background, not mark
    assert maskable.getpixel((512 // 2 + int(512 * 0.4) + 4, 256)) == (0, 0, 0)
    # the centre is the mark
    red, _, _ = maskable.getpixel((256, 256))  # type: ignore[misc]
    assert red > 200


def test_manifest_icons_list_purposes_and_versioned_urls(source: bytes) -> None:
    icons = render_icons(source, background="#07060c")
    listed = icons.manifest_icons("/app/icons/")
    by_src = {entry["src"].split("?")[0]: entry for entry in listed}
    assert set(by_src) == {
        "/app/icons/icon-192.png",
        "/app/icons/icon-512.png",
        "/app/icons/icon-maskable-192.png",
        "/app/icons/icon-maskable-512.png",
    }
    assert by_src["/app/icons/icon-512.png"] == {
        "src": f"/app/icons/icon-512.png?v={icons['icon-512.png'].version}",
        "sizes": "512x512",
        "type": "image/png",
        "purpose": "any",
    }
    assert by_src["/app/icons/icon-maskable-192.png"]["purpose"] == "maskable"
    assert icons.url(APPLE_TOUCH_ICON, "/app/icons", versioned=False) == "/app/icons/apple-touch-icon.png"


def test_rendering_is_deterministic(source: bytes) -> None:
    first = render_icons(source, background="#07060c", monochrome=True)
    second = render_icons(source, background="#07060c", monochrome=True)
    assert [(f.name, f.body) for f in first.files] == [(f.name, f.body) for f in second.files]
    assert first["icon-512.png"].etag == second["icon-512.png"].etag
    other = render_icons(source, background="#ffffff")
    assert other["icon-maskable-512.png"].version != first["icon-maskable-512.png"].version


def test_monochrome_is_a_white_silhouette_from_the_alpha(source: bytes) -> None:
    icons = render_icons(source, background="#07060c", monochrome=True)
    mono = _image(icons["icon-monochrome-512.png"].body).convert("RGBA")
    assert mono.getpixel((256, 256)) == (255, 255, 255, 255)
    assert mono.getpixel((0, 0))[3] == 0  # type: ignore[index]
    assert {e["purpose"] for e in icons.manifest_icons("/i")} == {"any", "maskable", "monochrome"}


def test_an_svg_given_is_served_as_any_size(source: bytes) -> None:
    svg = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1"/>'
    icons = render_icons(source, background="#07060c", svg=svg)
    assert icons[SVG_ICON].body == svg
    assert icons[SVG_ICON].sizes == "any"
    assert icons[SVG_ICON].media_type == "image/svg+xml"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (_source(512, fmt="JPEG", mode="RGB"), "must be a PNG"),
        (_source(512, height=500), "square"),
        (_source(256), "512 to 4096"),
        (b"not an image", "not an image"),
    ],
    ids=["jpeg", "not-square", "too-small", "garbage"],
)
def test_a_bad_source_is_refused_by_name(body: bytes, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        render_icons(body, background="#07060c")


def test_bad_options_are_refused(source: bytes) -> None:
    with pytest.raises(ValueError, match="maskable_scale"):
        render_icons(source, background="#07060c", maskable_scale=0)
    with pytest.raises(ValueError, match="background"):
        render_icons(source, background="not-a-colour")
    # flattened onto it, so a transparent background would come out black
    with pytest.raises(ValueError, match="opaque"):
        render_icons(source, background="#00000000")


def test_a_truncated_source_is_a_value_error(source: bytes) -> None:
    with pytest.raises(ValueError, match="decoded"):
        render_icons(source[: len(source) // 2], background="#07060c")


def test_an_icon_outside_the_set_is_a_key_error(source: bytes) -> None:
    icons = render_icons(source, background="#07060c")
    assert "nope.png" not in icons
    with pytest.raises(KeyError):
        icons.url("nope.png", "/i")
