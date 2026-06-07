"""back-compat shim -- re-exports :mod:`threetears.media.contracts`.

the media capability contracts moved to the dependency-free
``3tears-media-contracts`` package so that providers (``3tears-models``)
can implement them without inheriting agent-tools' dependency closure.
this module preserves the historical import path for installed
agent-tools consumers; new code should import from
``threetears.media.contracts`` directly.
"""

from __future__ import annotations

from threetears.media.contracts import (
    GeneratedImage,
    ImageGenerationBackend,
    MediaInfo,
    MediaStorage,
    TextProvider,
    TranscriptionProvider,
    VisionProvider,
)

__all__ = [
    "GeneratedImage",
    "ImageGenerationBackend",
    "MediaInfo",
    "MediaStorage",
    "TextProvider",
    "TranscriptionProvider",
    "VisionProvider",
]
