"""dependency-free media capability contracts.

pure interface types -- ``typing.Protocol`` classes and stdlib
dataclasses -- shared between media providers (``3tears-models``) and
media consumers (``3tears-agent-tools``). this package depends on
nothing so that implementing or accepting a contract never inherits a
feature package's dependency closure. purity is enforced by the
contract-purity check in the workspace's ``tests/enforcement/``.
"""

from threetears.media.contracts.protocols import (
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

__version__ = "0.10.6"
