"""Protocols for media-related tool capabilities.

These define the contracts that host applications implement. Tool
implementations can depend on these protocols without coupling to
any specific infrastructure (S3, specific vision APIs, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

__all__ = [
    "GeneratedImage",
    "ImageGenerationBackend",
    "MediaInfo",
    "MediaStorage",
    "TextProvider",
    "TranscriptionProvider",
    "VisionProvider",
]


@dataclass
class GeneratedImage:
    """Result from an image generation backend."""

    data: bytes
    mime_type: str
    width: int | None = None
    height: int | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class MediaInfo:
    """Metadata about a media item, returned by :meth:`MediaStorage.get_media`."""

    media_id: UUID
    media_category: str  # "image", "audio", "video", "document"
    mime_type: str
    extraction_status: str | None = None  # "pending", "complete", None
    has_downloadable_data: bool = True


@runtime_checkable
class ImageGenerationBackend(Protocol):
    """Protocol for image generation backends."""

    async def generate(
        self,
        prompt: str,
        *,
        style: str | None = None,
        source_image: bytes | None = None,
        source_mime_type: str | None = None,
    ) -> GeneratedImage:
        """Generate an image from a text prompt.

        :param prompt: text description of the image to generate
        :ptype prompt: str
        :param style: optional style modifier
        :ptype style: str | None
        :param source_image: optional source image for img2img
        :ptype source_image: bytes | None
        :param source_mime_type: MIME type of source image
        :ptype source_mime_type: str | None
        :return: generated image result
        :rtype: GeneratedImage
        """
        ...


@runtime_checkable
class MediaStorage(Protocol):
    """Protocol for media item access and storage.

    Host applications implement this to bridge their specific storage
    infrastructure (S3, database, etc.) to the generic tool interface.
    """

    async def get_media(
        self,
        media_id: UUID,
    ) -> MediaInfo | None:
        """Look up media metadata by ID.

        :param media_id: media item UUID
        :ptype media_id: UUID
        :return: media info or None
        :rtype: MediaInfo | None
        """
        ...

    async def download_media(
        self,
        media_id: UUID,
    ) -> tuple[bytes, str] | None:
        """Download raw media bytes.

        :param media_id: media item UUID
        :ptype media_id: UUID
        :return: (data, mime_type) or None if unavailable
        :rtype: tuple[bytes, str] | None
        """
        ...

    async def get_content(
        self,
        media_id: UUID,
        content_type: str,
        *,
        model_name: str | None = None,
    ) -> str | None:
        """Retrieve extracted content for a media item.

        :param media_id: media item UUID
        :ptype media_id: UUID
        :param content_type: type of content to retrieve
        :ptype content_type: str
        :param model_name: optional model name filter
        :ptype model_name: str | None
        :return: extracted content or None
        :rtype: str | None
        """
        ...

    async def store_content(
        self,
        media_id: UUID,
        user_id: UUID,
        content_type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store extracted content for a media item.

        :param media_id: media item UUID
        :ptype media_id: UUID
        :param user_id: user who triggered extraction
        :ptype user_id: UUID
        :param content_type: type of content being stored
        :ptype content_type: str
        :param content: extracted content text
        :ptype content: str
        :param metadata: optional metadata
        :ptype metadata: dict[str, Any] | None
        :return: content ID
        :rtype: str
        """
        ...


@runtime_checkable
class VisionProvider(Protocol):
    """Protocol for image analysis via a vision-capable model."""

    async def analyze(
        self,
        image_data: bytes,
        mime_type: str,
        prompt: str,
    ) -> str:
        """Analyze an image and return a text description.

        :param image_data: raw image bytes
        :ptype image_data: bytes
        :param mime_type: image MIME type
        :ptype mime_type: str
        :param prompt: analysis prompt
        :ptype prompt: str
        :return: analysis result text
        :rtype: str
        """
        ...


@runtime_checkable
class TextProvider(Protocol):
    """Protocol for text-only queries (document QA, summarization).

    Separate from :class:`VisionProvider` so that apps can use different
    models for image analysis vs. text reasoning.  Apps that don't
    handle documents can ignore this entirely.
    """

    async def answer(
        self,
        prompt: str,
    ) -> str:
        """Answer a text prompt.

        :param prompt: text prompt
        :ptype prompt: str
        :return: answer text
        :rtype: str
        """
        ...


@runtime_checkable
class TranscriptionProvider(Protocol):
    """Protocol for audio/video transcription."""

    async def transcribe(
        self,
        audio_data: bytes,
        mime_type: str,
    ) -> str:
        """Transcribe audio data to text.

        :param audio_data: raw audio bytes
        :ptype audio_data: bytes
        :param mime_type: audio MIME type
        :ptype mime_type: str
        :return: transcription text
        :rtype: str
        """
        ...
