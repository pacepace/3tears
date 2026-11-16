"""The web app manifest, and how to serve it.

Browsers fetch the manifest without credentials (a ``<link rel="manifest">`` request carries no
cookies unless it says ``crossorigin="use-credentials"``, and whether iOS honours that is
unverified), and they fetch its icons the same way. So both must be served without sign-in. They
hold nothing private: the app's name, colours and icon.

The manifest itself revalidates on every fetch (``no-cache``): browsers re-read it to update an
installed app, and a stale one would hold old colours or icons for days. Icons are served under
``?v=<content hash>`` and cached for good; a request without the current hash (an old manifest, or a
page that cannot know the hash) gets a day.
"""

from __future__ import annotations

import json
from typing import Final

from threetears.pwa.icons import IconSet
from threetears.pwa.identity import AppIdentity

__all__ = [
    "MANIFEST_CACHE_CONTROL",
    "MANIFEST_MEDIA_TYPE",
    "build_manifest",
    "icon_cache_control",
    "manifest_body",
    "not_modified",
]

#: the manifest's registered media type (W3C Web App Manifest)
MANIFEST_MEDIA_TYPE: Final[str] = "application/manifest+json"
#: the manifest revalidates every time it is read
MANIFEST_CACHE_CONTROL: Final[str] = "no-cache"
#: an icon requested under its current content hash: never changes, cache it for a year
_VERSIONED_ICON_CACHE_CONTROL: Final[str] = "public, max-age=31536000, immutable"
#: an icon requested without its current hash: a day
_ICON_CACHE_CONTROL: Final[str] = "public, max-age=86400"


def build_manifest(identity: AppIdentity, icons: IconSet, *, icon_base_url: str) -> dict[str, object]:
    """The manifest for ``identity``, its icons served under ``icon_base_url``.

    :param identity: the app
    :ptype identity: AppIdentity
    :param icons: its icon set
    :ptype icons: IconSet
    :param icon_base_url: the directory URL the icon files are served under
    :ptype icon_base_url: str
    :return: the manifest, ready for :func:`manifest_body`
    :rtype: dict[str, object]
    """
    manifest: dict[str, object] = {
        "id": identity.id,
        "name": identity.name,
        "short_name": identity.label,
        "start_url": identity.start_url,
        "scope": identity.scope,
        "display": identity.display,
        "theme_color": identity.theme_color,
        "background_color": identity.background_color,
        "lang": identity.lang,
        "icons": icons.manifest_icons(icon_base_url),
    }
    if identity.description:
        manifest["description"] = identity.description
    if identity.categories:
        manifest["categories"] = list(identity.categories)
    return manifest


def manifest_body(manifest: dict[str, object]) -> bytes:
    """``manifest`` as the bytes to serve (UTF-8 JSON) with :data:`MANIFEST_MEDIA_TYPE`."""
    return json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def icon_cache_control(version: str, requested: str | None) -> str:
    """The Cache-Control for an icon whose current version is ``version``, requested with
    ``?v=<requested>`` (``None`` when the request had none)."""
    return _VERSIONED_ICON_CACHE_CONTROL if requested == version else _ICON_CACHE_CONTROL


def not_modified(if_none_match: str | None, etag: str) -> bool:
    """Whether a request's ``If-None-Match`` already names ``etag`` (answer 304)."""
    if not if_none_match:
        return False
    if if_none_match.strip() == "*":
        return True
    tags = (tag.strip().removeprefix("W/") for tag in if_none_match.split(","))
    return etag in tags
