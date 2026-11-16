"""The manifest's fields and the cache rules for it and its icons."""

from __future__ import annotations

import io
import json

import pytest
from PIL import Image

from threetears.pwa import (
    MANIFEST_MEDIA_TYPE,
    AppIdentity,
    build_manifest,
    icon_cache_control,
    manifest_body,
    not_modified,
    render_icons,
)


@pytest.fixture(scope="module")
def icons():  # type: ignore[no-untyped-def]
    out = io.BytesIO()
    Image.new("RGBA", (512, 512), (200, 0, 0, 255)).save(out, format="PNG")
    return render_icons(out.getvalue(), background="#07060c")


def _identity(**changes: object) -> AppIdentity:
    fields: dict[str, object] = {
        "id": "/stations/kairo",
        "name": "Kairo Radio",
        "short_name": "Kairo",
        "start_url": "/player/?station=kairo",
        "scope": "/",
        "theme_color": "#07060c",
        "background_color": "#07060c",
    }
    fields.update(changes)
    return AppIdentity(**fields)  # type: ignore[arg-type]


def test_the_manifest_carries_identity_and_the_install_criteria(icons) -> None:  # type: ignore[no-untyped-def]
    manifest = build_manifest(
        _identity(description="Live radio.", categories=("music",)), icons, icon_base_url="/api/kairo/app"
    )
    assert manifest["id"] == "/stations/kairo"
    assert manifest["name"] == "Kairo Radio"
    assert manifest["short_name"] == "Kairo"
    assert manifest["start_url"] == "/player/?station=kairo"
    assert manifest["scope"] == "/"
    assert manifest["display"] == "standalone"
    assert manifest["theme_color"] == manifest["background_color"] == "#07060c"
    assert manifest["description"] == "Live radio."
    assert manifest["categories"] == ["music"]
    sizes = {(i["sizes"], i["purpose"]) for i in manifest["icons"]}  # type: ignore[attr-defined]
    # Chrome's install criteria: a 192 and a 512
    assert {("192x192", "any"), ("512x512", "any"), ("512x512", "maskable")} <= sizes


def test_optional_fields_are_left_out_when_unset(icons) -> None:  # type: ignore[no-untyped-def]
    manifest = build_manifest(_identity(short_name=None), icons, icon_base_url="/i")
    assert "description" not in manifest and "categories" not in manifest
    assert manifest["short_name"] == "Kairo Radio"


def test_the_body_is_utf8_json_and_the_type_is_registered(icons) -> None:  # type: ignore[no-untyped-def]
    body = manifest_body(build_manifest(_identity(name="Café"), icons, icon_base_url="/i"))
    assert json.loads(body.decode("utf-8"))["name"] == "Café"
    assert "Café".encode() in body
    assert MANIFEST_MEDIA_TYPE == "application/manifest+json"


def test_an_icon_under_its_current_hash_is_cached_for_good_else_a_day() -> None:
    assert icon_cache_control("abc", "abc") == "public, max-age=31536000, immutable"
    assert icon_cache_control("abc", "old") == "public, max-age=86400"
    assert icon_cache_control("abc", None) == "public, max-age=86400"


@pytest.mark.parametrize(
    ("header", "expected"),
    [(None, False), ("", False), ('"a"', True), ('W/"a"', True), ('"b", "a"', True), ('"b"', False), ("*", True)],
)
def test_not_modified_reads_if_none_match(header: str | None, expected: bool) -> None:
    assert not_modified(header, '"a"') is expected
