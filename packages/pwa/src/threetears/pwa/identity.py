"""Who an installed app is: its stable identity, its names, where it starts and what it covers.

A browser knows an installed app by the manifest's ``id`` and by nothing else. The id is opaque and
is only ever compared: change it and every existing install becomes a stranger to the page, which
offers to install a second copy beside it. So the id is built from a stable slug, chosen once per
app (a station id, a DJ's slug), and never from a display name or a URL that may move.

Several apps can live on one host -- one per station, one per persona -- each with its own id and
its own home-screen icon, sharing a scope. The scope is a path prefix of the URLs the app keeps
inside its own window; anything else opens in the browser. It must cover every page of the app's
flows (a sign-in that leaves the scope drops an iOS user into a browser sheet mid-flow), so it is
usually the site root or the app's directory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Final, Literal
from urllib.parse import urlsplit

__all__ = ["AppIdentity", "install_id"]

#: a slug: lowercase letters, digits and inner hyphens or underscores, up to 63 characters
_SLUG: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9](?:[a-z0-9_-]{0,61}[a-z0-9])?$")

#: a CSS hex colour, the one form every manifest consumer and every theme-color reader accepts
_HEX_COLOUR: Final[re.Pattern[str]] = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")

#: the manifest display modes an installed app can ask for
Display = Literal["standalone", "fullscreen", "minimal-ui"]


def install_id(prefix: str, slug: str) -> str:
    """The manifest ``id`` for one app on a host: ``<prefix>/<slug>``.

    :param prefix: a path-absolute prefix shared by the host's apps, e.g. ``/stations``
    :ptype prefix: str
    :param slug: the app's stable slug; never a display name, which may change
    :ptype slug: str
    :return: the id, path-absolute
    :rtype: str
    :raises ValueError: a slug that is not lowercase letters, digits, ``-`` or ``_``, or a prefix
        that is not path-absolute
    """
    if not _SLUG.match(slug):
        raise ValueError(f"an install slug is lowercase letters, digits, '-' or '_' (max 63): {slug!r}")
    _require_path(prefix, "prefix")
    return f"{prefix.rstrip('/')}/{slug}"


def _require_path(value: str, what: str) -> None:
    """``value`` is a same-origin, path-absolute URL (``/x``, never ``//host`` or ``https:``)."""
    parts = urlsplit(value)
    dot_segment = any(segment in (".", "..") for segment in parts.path.split("/"))
    if (
        not value.startswith("/")
        or value.startswith("//")
        or parts.scheme
        or parts.netloc
        or "\\" in value
        or dot_segment
    ):
        raise ValueError(f"{what} must be a path-absolute URL on the app's own origin: {value!r}")


def _require_colour(value: str, what: str) -> None:
    if not _HEX_COLOUR.match(value):
        raise ValueError(f"{what} must be a hex colour (#rgb or #rrggbb): {value!r}")


@dataclass(frozen=True, slots=True)
class AppIdentity:
    """One installable app.

    :param id: the manifest id, from :func:`install_id`; stable for the app's whole life
    :param name: the full name (install dialogs, the app switcher, iOS's home-screen label)
    :param start_url: where the installed app opens; inside ``scope``
    :param scope: the path prefix the app keeps in its own window
    :param theme_color: the title bar and status bar colour, and the ``theme-color`` without a scheme
    :param background_color: the splash screen colour; match the page's first paint
    :param dark_theme_color: the ``theme-color`` under a dark colour scheme, if it differs
    :param short_name: the label where space is short (home screens); defaults to ``name``
    :param description: one sentence for install dialogs and stores
    :param categories: manifest categories (lowercase words, e.g. ``music``)
    :param lang: the language of ``name`` and ``description``
    :param display: the display mode; ``standalone`` gives an app window with no browser chrome
    """

    id: str
    name: str
    start_url: str
    scope: str
    theme_color: str
    background_color: str
    dark_theme_color: str | None = None
    short_name: str | None = None
    description: str | None = None
    categories: tuple[str, ...] = field(default=())
    lang: str = "en"
    display: Display = "standalone"

    def __post_init__(self) -> None:
        for what in ("id", "start_url", "scope"):
            _require_path(getattr(self, what), what)
        if not self.scope.endswith("/"):
            raise ValueError(
                f"scope must end in '/', or it also covers paths that merely share its prefix: {self.scope!r}"
            )
        if not urlsplit(self.start_url).path.startswith(self.scope):
            raise ValueError(f"start_url {self.start_url!r} is outside scope {self.scope!r}")
        if not self.name.strip():
            raise ValueError("name must not be blank")
        if self.short_name is not None and not self.short_name.strip():
            raise ValueError("short_name must not be blank; leave it out to use name")
        _require_colour(self.theme_color, "theme_color")
        _require_colour(self.background_color, "background_color")
        if self.dark_theme_color is not None:
            _require_colour(self.dark_theme_color, "dark_theme_color")
        for category in self.categories:
            if not category or category != category.lower() or not category.replace("-", "").isalpha():
                raise ValueError(f"a manifest category is one lowercase word (hyphens allowed): {category!r}")

    @property
    def label(self) -> str:
        """The home-screen label: ``short_name``, else ``name``."""
        return self.short_name or self.name
