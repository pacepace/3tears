"""3tears-pwa: what a web app needs to install well and behave well once installed.

* :mod:`~threetears.pwa.identity` -- :class:`AppIdentity`: a stable manifest ``id`` per app
  (several apps per host), names, ``start_url``, ``scope``, colours.
* :mod:`~threetears.pwa.icons` -- :func:`render_icons`: the icon set (192/512, maskable,
  apple-touch, favicon, optional monochrome) from one square PNG, deterministic.
  :mod:`threetears.pwa.svg` renders it from an SVG (the ``svg`` extra).
* :mod:`~threetears.pwa.manifest` -- :func:`build_manifest` and how to serve it and its icons.
* :mod:`~threetears.pwa.head` -- :func:`head_tags`: the manifest link, ``theme-color`` per
  colour scheme and the iOS tags that iOS reads instead of the manifest.
* :mod:`~threetears.pwa.assets` -- ``pwa.js`` (display mode, Media Session, Screen Wake Lock)
  and ``safe-area.css``, for the app to serve.

No service worker: none is needed to install on current Chrome, Edge, Safari or Firefox, and the
family keeps live pages live (no offline cache, no push) unless an app opts in on its own. See the
package README, "Service worker".
"""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

from threetears.pwa.assets import STATIC_FILES, export_static
from threetears.pwa.head import head_tags, identity_head_tags, insert_head_tags
from threetears.pwa.icons import APPLE_TOUCH_ICON, FAVICON, SVG_ICON, IconSet, render_icons
from threetears.pwa.identity import AppIdentity, install_id
from threetears.pwa.manifest import (
    MANIFEST_CACHE_CONTROL,
    MANIFEST_MEDIA_TYPE,
    build_manifest,
    icon_cache_control,
    manifest_body,
    not_modified,
)

# read from installed metadata, so a release that bumps pyproject cannot leave a stale runtime
# ``__version__`` behind. The fallback keeps the package importable from a source tree.
try:
    __version__ = _version("3tears-pwa")
except _PackageNotFoundError:  # pragma: no cover - dev fallback
    __version__ = "unknown"

__all__ = [
    "APPLE_TOUCH_ICON",
    "FAVICON",
    "MANIFEST_CACHE_CONTROL",
    "MANIFEST_MEDIA_TYPE",
    "STATIC_FILES",
    "SVG_ICON",
    "AppIdentity",
    "IconSet",
    "__version__",
    "build_manifest",
    "export_static",
    "head_tags",
    "icon_cache_control",
    "identity_head_tags",
    "insert_head_tags",
    "install_id",
    "manifest_body",
    "not_modified",
    "render_icons",
]
