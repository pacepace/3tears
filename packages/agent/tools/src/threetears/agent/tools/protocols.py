"""back-compat shim -- re-exports :mod:`threetears.interfaces.media`.

the media capability contracts live in the dependency-free
``3tears-interfaces`` package so that providers (``3tears-models``)
can implement them without inheriting agent-tools' dependency closure.
this module preserves the historical import path for installed
agent-tools consumers; new code should import from
``threetears.interfaces.media`` directly.
"""

from __future__ import annotations

from threetears.interfaces.media import (
    GeneratedImage,
    ImageGenerationBackend,
    MediaInfo,
    MediaStorage,
    ReferenceVisionProvider,
    TextProvider,
    TranscriptionProvider,
    VisionProvider,
)

__all__ = [
    "GeneratedImage",
    "ImageGenerationBackend",
    "MediaInfo",
    "MediaStorage",
    "ReferenceVisionProvider",
    "TextProvider",
    "TranscriptionProvider",
    "VisionProvider",
]
