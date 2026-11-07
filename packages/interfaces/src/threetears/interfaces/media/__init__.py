"""Media capability interfaces.

Pure interface types -- ``typing.Protocol`` classes and stdlib dataclasses -- shared between
media providers (``3tears-models``, ``3tears-object-store``) and media consumers
(``3tears-agent-tools``). Implementing or accepting one never inherits a feature package's
dependency closure: ``3tears-interfaces`` depends on nothing (see :mod:`threetears.interfaces`).
"""

from threetears.interfaces.media.facets import (
    LOCATOR_KIND_CONTAINING_PAGE,
    LOCATOR_KIND_DIRECT_FILE,
    MediaFacets,
)
from threetears.interfaces.media.keys import build_object_key
from threetears.interfaces.media.protocols import (
    EXTRACTION_STATUS_COMPLETE,
    EXTRACTION_STATUS_FAILED,
    EXTRACTION_STATUS_NONE,
    EXTRACTION_STATUS_PENDING,
    EXTRACTION_STATUS_REFUSED,
    EXTRACTION_STATUS_UNCHANGED,
    GENERATION_IGNORED,
    GENERATION_SEED,
    GENERATION_SIZE,
    GENERATION_SOURCE_USED,
    OBJECT_HANDLE_METADATA_KEY,
    GeneratedImage,
    ImageGenerationBackend,
    MediaInfo,
    MediaSizeLimitExceeded,
    MediaStorage,
    ObjectHandle,
    ObjectListing,
    ObjectStore,
    TextProvider,
    TranscriptionProvider,
    ReferenceVisionProvider,
    VisionProvider,
)

__all__ = [
    "EXTRACTION_STATUS_COMPLETE",
    "EXTRACTION_STATUS_FAILED",
    "EXTRACTION_STATUS_NONE",
    "EXTRACTION_STATUS_PENDING",
    "EXTRACTION_STATUS_REFUSED",
    "EXTRACTION_STATUS_UNCHANGED",
    "GENERATION_IGNORED",
    "GENERATION_SEED",
    "GENERATION_SIZE",
    "GENERATION_SOURCE_USED",
    "LOCATOR_KIND_CONTAINING_PAGE",
    "LOCATOR_KIND_DIRECT_FILE",
    "OBJECT_HANDLE_METADATA_KEY",
    "GeneratedImage",
    "ImageGenerationBackend",
    "MediaFacets",
    "MediaInfo",
    "MediaSizeLimitExceeded",
    "MediaStorage",
    "ObjectHandle",
    "ObjectListing",
    "ObjectStore",
    "TextProvider",
    "TranscriptionProvider",
    "ReferenceVisionProvider",
    "VisionProvider",
    "build_object_key",
]
