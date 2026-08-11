"""dependency-free media capability contracts.

pure interface types -- ``typing.Protocol`` classes and stdlib
dataclasses -- shared between media providers (``3tears-models``) and
media consumers (``3tears-agent-tools``). this package depends on
nothing so that implementing or accepting a contract never inherits a
feature package's dependency closure. purity is enforced by the
contract-purity check in the workspace's ``tests/enforcement/``.
"""

# Version derived from package metadata so the metadata is the single
# source of truth -- a future release that bumps pyproject without
# updating ``__init__.py`` can't drift the runtime ``__version__``.
# This module is the one that did: it carried a hardcoded 0.10.6 through
# fourteen minor releases, reporting a version this package had not been
# for months. The except guard handles the rare case where the package
# isn't installed via importlib.metadata (e.g. running directly from a
# checked-out source tree without ``uv sync``); the fallback keeps
# imports working but reports ``unknown`` rather than crashing.
# ``importlib.metadata`` is stdlib, so the dependency-free floor this
# package is pinned at is untouched.
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("3tears-media-contracts")
except _PackageNotFoundError:  # pragma: no cover - dev fallback
    __version__ = "unknown"

from threetears.media.contracts.facets import (
    LOCATOR_KIND_CONTAINING_PAGE,
    LOCATOR_KIND_DIRECT_FILE,
    MediaFacets,
)
from threetears.media.contracts.keys import build_object_key, sanitize_segment
from threetears.media.contracts.protocols import (
    OBJECT_HANDLE_METADATA_KEY,
    GeneratedImage,
    ImageGenerationBackend,
    MediaInfo,
    MediaStorage,
    ObjectHandle,
    ObjectListing,
    ObjectStore,
    TextProvider,
    TranscriptionProvider,
    VisionProvider,
)

__all__ = [
    "LOCATOR_KIND_CONTAINING_PAGE",
    "LOCATOR_KIND_DIRECT_FILE",
    "OBJECT_HANDLE_METADATA_KEY",
    "GeneratedImage",
    "ImageGenerationBackend",
    "MediaFacets",
    "MediaInfo",
    "MediaStorage",
    "ObjectHandle",
    "ObjectListing",
    "ObjectStore",
    "TextProvider",
    "TranscriptionProvider",
    "VisionProvider",
    "build_object_key",
    "sanitize_segment",
]
