"""Protocols for media-related tool capabilities.

These define the contracts that host applications implement. Tool
implementations can depend on these protocols without coupling to
any specific infrastructure (S3, specific vision APIs, etc.).

The actual tool logic stays in the host application for now — these
protocols prepare the path for future extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    """Protocol for media item access and storage."""

    async def get_media(
        self,
        media_id: UUID,
        user_id: UUID,
    ) -> dict[str, Any] | None: ...

    async def get_content(
        self,
        media_id: UUID,
        content_type: str,
    ) -> str | None: ...

    async def store_content(
        self,
        media_id: UUID,
        user_id: UUID,
        content_type: str,
        content: str,
    ) -> str: ...


@runtime_checkable
class VisionProvider(Protocol):
    """Protocol for image/document analysis via a vision-capable model."""

    async def analyze(
        self,
        image_data: bytes,
        mime_type: str,
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
