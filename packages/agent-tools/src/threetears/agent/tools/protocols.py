"""Protocols for media-related tool capabilities.

These define the contracts that host applications implement. Tool
implementations can depend on these protocols without coupling to
any specific infrastructure (S3, specific vision APIs, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
from uuid import UUID


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
    ) -> GeneratedImage: ...


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
        """Look up media metadata by ID."""
        ...

    async def download_media(
        self,
        media_id: UUID,
    ) -> tuple[bytes, str] | None:
        """Download raw media bytes.

        :return: ``(data, mime_type)`` or ``None`` if unavailable
        """
        ...

    async def get_content(
        self,
        media_id: UUID,
        content_type: str,
        *,
        model_name: str | None = None,
    ) -> str | None: ...

    async def store_content(
        self,
        media_id: UUID,
        user_id: UUID,
        content_type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str: ...


@runtime_checkable
class VisionProvider(Protocol):
    """Protocol for image analysis via a vision-capable model."""

    async def analyze(
        self,
        image_data: bytes,
        mime_type: str,
        prompt: str,
    ) -> str: ...


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
    ) -> str: ...


@runtime_checkable
class TranscriptionProvider(Protocol):
    """Protocol for audio/video transcription."""

    async def transcribe(
        self,
        audio_data: bytes,
        mime_type: str,
    ) -> str: ...
