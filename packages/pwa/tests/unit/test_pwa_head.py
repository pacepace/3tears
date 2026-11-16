"""The head tags: what iOS reads instead of the manifest, theme-color per scheme, escaping."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from threetears.pwa.head import VIEWPORT, IconLink
from threetears.pwa import (
    AppIdentity,
    head_tags,
    identity_head_tags,
    insert_head_tags,
    render_icons,
)


def test_tags_cover_the_manifest_ios_and_one_theme_colour() -> None:
    tags = head_tags(
        manifest_url="/api/kairo/app/manifest.webmanifest",
        title="Kairo",
        theme_color="#07060c",
        apple_touch_icon_url="/api/kairo/app/apple-touch-icon.png",
        icons=[IconLink("/f.png", "image/png", "32x32")],
    )
    lines = tags.splitlines()
    assert '<link rel="manifest" href="/api/kairo/app/manifest.webmanifest">' in lines
    assert '<meta name="theme-color" content="#07060c">' in lines
    assert '<meta name="apple-mobile-web-app-title" content="Kairo">' in lines
    assert '<meta name="apple-mobile-web-app-capable" content="yes">' in lines
    assert '<meta name="mobile-web-app-capable" content="yes">' in lines
    assert '<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">' in lines
    assert '<link rel="apple-touch-icon" sizes="180x180" href="/api/kairo/app/apple-touch-icon.png">' in lines
    assert '<link rel="icon" type="image/png" sizes="32x32" href="/f.png">' in lines


def test_a_dark_theme_colour_gets_its_own_media_scoped_meta() -> None:
    tags = head_tags(manifest_url="/m", title="T", theme_color="#ffffff", dark_theme_color="#000000")
    assert '<meta name="theme-color" content="#ffffff" media="(prefers-color-scheme: light)">' in tags
    assert '<meta name="theme-color" content="#000000" media="(prefers-color-scheme: dark)">' in tags
    same = head_tags(manifest_url="/m", title="T", theme_color="#000000", dark_theme_color="#000000")
    assert same.count('name="theme-color"') == 1


def test_values_are_escaped() -> None:
    tags = head_tags(manifest_url='/m?a=1&b="x"', title="<script>alert(1)</script>", theme_color="#000")
    assert "<script>" not in tags
    assert 'href="/m?a=1&amp;b=&quot;x&quot;"' in tags
    assert "&lt;script&gt;" in tags


def test_identity_head_tags_use_the_rendered_icons() -> None:
    out = io.BytesIO()
    Image.new("RGBA", (512, 512), (0, 0, 200, 255)).save(out, format="PNG")
    icons = render_icons(out.getvalue(), background="#07060c", svg=b"<svg/>")
    identity = AppIdentity(
        id="/stations/kairo",
        name="Kairo Radio",
        short_name="Kairo",
        start_url="/player/",
        scope="/",
        theme_color="#07060c",
        background_color="#07060c",
    )
    tags = identity_head_tags(identity, manifest_url="/m", icon_set=icons, icon_base_url="/i")
    assert f'href="/i/apple-touch-icon.png?v={icons["apple-touch-icon.png"].version}"' in tags
    assert 'content="Kairo"' in tags  # iOS's label is the short name
    assert 'type="image/svg+xml" sizes="any"' in tags


def test_insert_head_tags_prefers_the_marker_else_closes_the_head() -> None:
    assert (
        insert_head_tags("<head><!-- pwa-head --><title>x</title></head>", "<T>") == "<head><T><title>x</title></head>"
    )
    assert insert_head_tags("<HEAD><title>x</title></HEAD>", "<T>") == "<HEAD><title>x</title><T>\n</HEAD>"
    with pytest.raises(ValueError, match="</head>"):
        insert_head_tags("<body></body>", "<T>")


def test_viewport_covers_the_safe_areas() -> None:
    assert "viewport-fit=cover" in VIEWPORT
