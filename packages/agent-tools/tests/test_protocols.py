"""Tests for media provider protocols."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from threetears.agent.tools.protocols import (
    GeneratedImage,
    ImageGenerationBackend,
    MediaStorage,
    TextProvider,
    TranscriptionProvider,
    VisionProvider,
)


# -- GeneratedImage dataclass -------------------------------------------------


class TestGeneratedImage:
    def test_required_fields(self):
        img = GeneratedImage(data=b"png-bytes", mime_type="image/png")
        assert img.data == b"png-bytes"
        assert img.mime_type == "image/png"
        assert img.width is None
        assert img.height is None
        assert img.metadata is None

    def test_all_fields(self):
        img = GeneratedImage(
            data=b"data",
            mime_type="image/jpeg",
            width=512,
            height=768,
            metadata={"model": "dall-e-3"},
        )
        assert img.width == 512
        assert img.height == 768
        assert img.metadata["model"] == "dall-e-3"


# -- Protocol compliance via concrete implementations -------------------------


class FakeImageBackend:
    async def generate(
        self,
        prompt: str,
        *,
        style: str | None = None,
        source_image: bytes | None = None,
        source_mime_type: str | None = None,
    ) -> GeneratedImage:
        return GeneratedImage(data=b"fake", mime_type="image/png")


class FakeMediaStorage:
    async def get_media(self, media_id: UUID) -> Any:
        return None

    async def download_media(self, media_id: UUID) -> tuple[bytes, str] | None:
        return None

    async def get_content(
        self, media_id: UUID, content_type: str, *, model_name: str | None = None,
    ) -> str | None:
        return None

    async def store_content(
        self,
        media_id: UUID,
        user_id: UUID,
        content_type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return "stored"


class FakeVisionProvider:
    async def analyze(self, image_data: bytes, mime_type: str, prompt: str) -> str:
        return "analysis result"


class FakeTextProvider:
    async def answer(self, prompt: str) -> str:
        return "answer result"


class FakeTranscriptionProvider:
    async def transcribe(self, audio_data: bytes, mime_type: str) -> str:
        return "transcription"


class TestProtocolCompliance:
    def test_image_generation_backend(self):
        assert isinstance(FakeImageBackend(), ImageGenerationBackend)

    def test_media_storage(self):
        assert isinstance(FakeMediaStorage(), MediaStorage)

    def test_vision_provider(self):
        assert isinstance(FakeVisionProvider(), VisionProvider)

    def test_text_provider(self):
        assert isinstance(FakeTextProvider(), TextProvider)

    def test_transcription_provider(self):
        assert isinstance(FakeTranscriptionProvider(), TranscriptionProvider)


class TestProtocolRejection:
    def test_non_compliant_rejected(self):
        assert not isinstance("a string", ImageGenerationBackend)
        assert not isinstance(42, MediaStorage)
        assert not isinstance(object(), VisionProvider)
        assert not isinstance([], TranscriptionProvider)
        assert not isinstance(123, TextProvider)
