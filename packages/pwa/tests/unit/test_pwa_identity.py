"""The identity rules: a stable id from a slug, and a scope that holds the start URL."""

from __future__ import annotations

import pytest

from threetears.pwa import AppIdentity, install_id


def _identity(**changes: object) -> AppIdentity:
    fields: dict[str, object] = {
        "id": "/stations/kairo",
        "name": "Kairo",
        "start_url": "/player/?station=kairo",
        "scope": "/",
        "theme_color": "#07060c",
        "background_color": "#07060c",
    }
    fields.update(changes)
    return AppIdentity(**fields)  # type: ignore[arg-type]


def test_install_id_joins_prefix_and_slug() -> None:
    assert install_id("/stations", "kairo") == "/stations/kairo"
    assert install_id("/stations/", "dj-2") == "/stations/dj-2"


@pytest.mark.parametrize("slug", ["", "Kairo", "kai ro", "-kairo", "kairo-", "k/airo", "x" * 64, "kairo?x=1"])
def test_install_id_refuses_a_slug_that_is_not_one(slug: str) -> None:
    with pytest.raises(ValueError, match="slug"):
        install_id("/stations", slug)


@pytest.mark.parametrize("prefix", ["stations", "//evil.example/x", "https://evil.example/", "\\x"])
def test_install_id_refuses_a_prefix_off_the_origin(prefix: str) -> None:
    with pytest.raises(ValueError, match="path-absolute"):
        install_id(prefix, "kairo")


def test_a_valid_identity_labels_itself_by_short_name_else_name() -> None:
    assert _identity().label == "Kairo"
    assert _identity(short_name="K").label == "K"


@pytest.mark.parametrize("field", ["id", "start_url", "scope"])
def test_urls_must_be_on_the_apps_own_origin(field: str) -> None:
    with pytest.raises(ValueError, match="path-absolute"):
        _identity(**{field: "https://elsewhere.example/"})


@pytest.mark.parametrize("url", ["/player/../admin/", "/player/./x", "/.."])
def test_dot_segments_are_refused(url: str) -> None:
    """A browser resolves them, so "/player/../admin/" would pass a prefix check and leave the scope."""
    with pytest.raises(ValueError, match="path-absolute"):
        _identity(scope="/player/", start_url=url)


def test_start_url_must_be_inside_scope() -> None:
    with pytest.raises(ValueError, match="outside scope"):
        _identity(scope="/player/", start_url="/auth/login")
    assert _identity(scope="/player/", start_url="/player/?station=kairo").scope == "/player/"


def test_scope_must_end_in_a_slash() -> None:
    # "/player" would also cover "/players-admin"
    with pytest.raises(ValueError, match="end in '/'"):
        _identity(scope="/player", start_url="/player/")


@pytest.mark.parametrize(
    ("field", "value"),
    [("theme_color", "black"), ("background_color", "#12345"), ("dark_theme_color", "rgb(0,0,0)")],
)
def test_colours_are_hex(field: str, value: str) -> None:
    with pytest.raises(ValueError, match="hex colour"):
        _identity(**{field: value})


def test_names_and_categories_are_checked() -> None:
    with pytest.raises(ValueError, match="name"):
        _identity(name="  ")
    with pytest.raises(ValueError, match="short_name"):
        _identity(short_name="")
    with pytest.raises(ValueError, match="category"):
        _identity(categories=("Music",))
    assert _identity(categories=("music", "entertainment")).categories == ("music", "entertainment")
