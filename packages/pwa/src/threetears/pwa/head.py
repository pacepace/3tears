"""The ``<head>`` tags an installable page carries, for every platform's reader.

The manifest is not enough on its own:

* **iOS takes the home-screen name and icon from the page, not the manifest**: the name from
  ``apple-mobile-web-app-title`` (else ``<title>``), the icon from ``apple-touch-icon``. So for
  several apps on one host these tags are per app, written into the page each request.
* ``theme-color`` colours the status bar and title bar, and takes a ``media`` query, so a dark
  colour scheme gets its own. The manifest has only one ``theme_color``.
* ``apple-mobile-web-app-status-bar-style: black-translucent`` lets the page draw under the iOS
  status bar in standalone mode. The page must then pad its edges by the safe-area insets
  (``safe-area.css``), with ``viewport-fit=cover`` in the viewport (:data:`VIEWPORT`).
* ``mobile-web-app-capable`` is the standard name of Apple's ``apple-mobile-web-app-capable``.
  Both are written: the manifest's ``display`` is what current browsers read, and the two metas
  cost nothing where they are ignored.

Every value is HTML-escaped.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
from typing import Final, Literal

from threetears.pwa.icons import APPLE_TOUCH_ICON, FAVICON, SVG_ICON, IconSet
from threetears.pwa.identity import AppIdentity

__all__ = [
    "head_tags",
    "identity_head_tags",
    "insert_head_tags",
]

#: the viewport an installed page wants: device width, and drawing into the safe-area insets
VIEWPORT: Final[str] = "width=device-width, initial-scale=1, viewport-fit=cover"

#: where :func:`insert_head_tags` puts the tags, if the page has it (else before ``</head>``)
_HEAD_MARKER: Final[str] = "<!-- pwa-head -->"

#: the iOS status bar in standalone mode: ``black-translucent`` draws the page under it
StatusBarStyle = Literal["default", "black", "black-translucent"]


@dataclass(frozen=True, slots=True)
class IconLink:
    """A ``<link rel="icon">``: its URL, media type and sizes (``32x32``, ``any``)."""

    href: str
    media_type: str
    sizes: str


def _attr(value: str) -> str:
    return escape(value, quote=True)


def head_tags(
    *,
    manifest_url: str,
    title: str,
    theme_color: str,
    dark_theme_color: str | None = None,
    apple_touch_icon_url: str | None = None,
    icons: Sequence[IconLink] = (),
    status_bar_style: StatusBarStyle = "black-translucent",
) -> str:
    """The tags, one per line.

    :param manifest_url: the manifest's URL
    :ptype manifest_url: str
    :param title: the home-screen name on iOS
    :ptype title: str
    :param theme_color: the status/title bar colour (the light scheme's, when a dark one is given)
    :ptype theme_color: str
    :param dark_theme_color: the colour under ``prefers-color-scheme: dark``
    :ptype dark_theme_color: str | None
    :param apple_touch_icon_url: the 180 px opaque PNG iOS puts on the home screen
    :ptype apple_touch_icon_url: str | None
    :param icons: browser tab icons
    :ptype icons: Sequence[IconLink]
    :param status_bar_style: the iOS status bar in standalone mode
    :ptype status_bar_style: StatusBarStyle
    :return: the HTML
    :rtype: str
    """
    lines = [f'<link rel="manifest" href="{_attr(manifest_url)}">']
    if dark_theme_color is None or dark_theme_color == theme_color:
        lines.append(f'<meta name="theme-color" content="{_attr(theme_color)}">')
    else:
        lines.append(f'<meta name="theme-color" content="{_attr(theme_color)}" media="(prefers-color-scheme: light)">')
        lines.append(
            f'<meta name="theme-color" content="{_attr(dark_theme_color)}" media="(prefers-color-scheme: dark)">'
        )
    lines.append('<meta name="mobile-web-app-capable" content="yes">')
    lines.append('<meta name="apple-mobile-web-app-capable" content="yes">')
    lines.append(f'<meta name="apple-mobile-web-app-title" content="{_attr(title)}">')
    lines.append(f'<meta name="apple-mobile-web-app-status-bar-style" content="{_attr(status_bar_style)}">')
    if apple_touch_icon_url is not None:
        lines.append(f'<link rel="apple-touch-icon" sizes="180x180" href="{_attr(apple_touch_icon_url)}">')
    for icon in icons:
        lines.append(
            f'<link rel="icon" type="{_attr(icon.media_type)}" sizes="{_attr(icon.sizes)}" href="{_attr(icon.href)}">'
        )
    return "\n".join(lines)


def identity_head_tags(
    identity: AppIdentity,
    *,
    manifest_url: str,
    icon_set: IconSet,
    icon_base_url: str,
    status_bar_style: StatusBarStyle = "black-translucent",
) -> str:
    """:func:`head_tags` for ``identity`` and its rendered icons served under ``icon_base_url``."""
    links = [IconLink(icon_set.url(FAVICON, icon_base_url), "image/png", icon_set[FAVICON].sizes)]
    if SVG_ICON in icon_set:
        links.append(IconLink(icon_set.url(SVG_ICON, icon_base_url), "image/svg+xml", "any"))
    return head_tags(
        manifest_url=manifest_url,
        title=identity.label,
        theme_color=identity.theme_color,
        dark_theme_color=identity.dark_theme_color,
        apple_touch_icon_url=icon_set.url(APPLE_TOUCH_ICON, icon_base_url),
        icons=links,
        status_bar_style=status_bar_style,
    )


def insert_head_tags(document: str, tags: str) -> str:
    """``document`` with ``tags`` in place of ``<!-- pwa-head -->``, else just before ``</head>``.

    :raises ValueError: the document has neither
    """
    if _HEAD_MARKER in document:
        return document.replace(_HEAD_MARKER, tags, 1)
    at = document.lower().find("</head>")
    if at < 0:
        raise ValueError(f"the document has no {_HEAD_MARKER} and no </head> to put the tags before")
    return f"{document[:at]}{tags}\n{document[at:]}"
