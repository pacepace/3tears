"""Tests for the analyze_media builtin tool."""

from __future__ import annotations

import io
from typing import Any
from uuid import UUID, uuid4

import pytest

from threetears.agent.tools.builtin.analyze_media import (
    AnalyzerConfig,
    create_analyze_media_tool,
)
from threetears.agent.tools.protocols import MediaInfo


# ---------------------------------------------------------------------------
# Fake protocol implementations
# ---------------------------------------------------------------------------


class FakeVisionProvider:
    """Records calls and returns canned responses."""

    def __init__(self, response: str = "A red square."):
        self.response = response
        self.analyze_calls: list[tuple[bytes, str, str]] = []

    async def analyze(self, image_data: bytes, mime_type: str, prompt: str) -> str:
        self.analyze_calls.append((image_data, mime_type, prompt))
        return self.response


class FakeTextProvider:
    """Records calls and returns canned responses."""

    def __init__(self, response: str = "A red square."):
        self.response = response
        self.answer_calls: list[str] = []

    async def answer(self, prompt: str) -> str:
        self.answer_calls.append(prompt)
        return self.response


class FakeTranscriptionProvider:
    def __init__(self, response: str = "Hello world."):
        self.response = response
        self.calls: list[tuple[bytes, str]] = []

    async def transcribe(self, audio_data: bytes, mime_type: str) -> str:
        self.calls.append((audio_data, mime_type))
        return self.response


class FakeMediaStorage:
    """In-memory storage for testing."""

    def __init__(self) -> None:
        self._media: dict[UUID, MediaInfo] = {}
        self._downloads: dict[UUID, tuple[bytes, str]] = {}
        self._content: dict[tuple[UUID, str, str | None], str] = {}
        self.store_calls: list[dict[str, Any]] = []

    def add_media(
        self,
        media_id: UUID,
        info: MediaInfo,
        data: bytes | None = None,
        mime_type: str = "image/jpeg",
    ) -> None:
        self._media[media_id] = info
        if data is not None:
            self._downloads[media_id] = (data, mime_type)

    def add_content(
        self,
        media_id: UUID,
        content_type: str,
        content: str,
        model_name: str | None = None,
    ) -> None:
        self._content[(media_id, content_type, model_name)] = content

    async def get_media(self, media_id: UUID) -> MediaInfo | None:
        return self._media.get(media_id)

    async def download_media(self, media_id: UUID) -> tuple[bytes, str] | None:
        return self._downloads.get(media_id)

    async def get_content(
        self,
        media_id: UUID,
        content_type: str,
        *,
        model_name: str | None = None,
    ) -> str | None:
        # Try exact match first, then without model_name
        val = self._content.get((media_id, content_type, model_name))
        if val is None:
            val = self._content.get((media_id, content_type, None))
        return val

    async def store_content(
        self,
        media_id: UUID,
        user_id: UUID,
        content_type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        self.store_calls.append({
            "media_id": media_id,
            "user_id": user_id,
            "content_type": content_type,
            "content": content,
            "metadata": metadata,
        })
        key = (media_id, content_type, (metadata or {}).get("model_name"))
        self._content[key] = content
        return "stored"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _small_jpeg() -> bytes:
    """Create a small JPEG for testing."""
    from PIL import Image

    img = Image.new("RGB", (50, 50), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_tool(
    storage: FakeMediaStorage,
    vision: FakeVisionProvider | None = None,
    text: FakeTextProvider | None = None,
    transcription: FakeTranscriptionProvider | None = None,
    user_id: UUID | None = None,
    analyzer_name: str = "TestVision",
    categories: set[str] | None = None,
    on_analysis: Any = None,
    media_url_fn: Any = None,
    response_suffix: str | None = None,
):
    if vision is None:
        vision = FakeVisionProvider()
    if text is None:
        text = FakeTextProvider()

    acfg = AnalyzerConfig(
        name=analyzer_name,
        vision=vision,
        text=text,
        transcription=transcription,
        supported_categories=categories or {"image"},
    )

    config: dict[str, Any] = {
        "storage": storage,
        "analyzers": {analyzer_name: acfg},
        "user_id": user_id,
    }
    if on_analysis:
        config["on_analysis"] = on_analysis
    if media_url_fn:
        config["media_url_fn"] = media_url_fn
    if response_suffix is not None:
        config["response_suffix"] = response_suffix

    return create_analyze_media_tool(
        config,
        "Analyze media using vision/STT.",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVisionAnalysis:
    """Image analysis via VisionProvider."""

    @pytest.mark.asyncio
    async def test_basic_image_analysis(self):
        storage = FakeMediaStorage()
        vision = FakeVisionProvider("A beautiful sunset.")
        mid = uuid4()
        storage.add_media(
            mid,
            MediaInfo(mid, "image", "image/jpeg"),
            _small_jpeg(),
            "image/jpeg",
        )

        tool = _make_tool(storage, vision=vision, user_id=uuid4())
        result = await tool.ainvoke({
            "media_ids": [str(mid)],
            "question": "Describe this image",
            "analyzer": "TestVision",
        })

        assert "A beautiful sunset." in result
        assert len(vision.analyze_calls) == 1

    @pytest.mark.asyncio
    async def test_display_hint_with_url_fn(self):
        """Display hint uses the configurable media_url_fn."""
        storage = FakeMediaStorage()
        mid = uuid4()
        storage.add_media(mid, MediaInfo(mid, "image", "image/jpeg"), _small_jpeg())

        tool = _make_tool(
            storage,
            user_id=uuid4(),
            media_url_fn=lambda mid_str: f"/custom/media/{mid_str}",
        )
        result = await tool.ainvoke({
            "media_ids": [str(mid)],
            "question": "Describe",
            "analyzer": "TestVision",
        })

        assert f"![description](/custom/media/{mid})" in result

    @pytest.mark.asyncio
    async def test_no_display_hint_without_url_fn(self):
        """Without media_url_fn, no display hint is appended."""
        storage = FakeMediaStorage()
        mid = uuid4()
        storage.add_media(mid, MediaInfo(mid, "image", "image/jpeg"), _small_jpeg())

        tool = _make_tool(storage, user_id=uuid4())
        result = await tool.ainvoke({
            "media_ids": [str(mid)],
            "question": "Describe",
            "analyzer": "TestVision",
        })

        assert "![description]" not in result
        assert "A red square." in result

    @pytest.mark.asyncio
    async def test_unknown_analyzer_error(self):
        storage = FakeMediaStorage()
        mid = uuid4()
        storage.add_media(mid, MediaInfo(mid, "image", "image/jpeg"), _small_jpeg())

        tool = _make_tool(storage)
        result = await tool.ainvoke({
            "media_ids": [str(mid)],
            "question": "Describe",
            "analyzer": "NonExistent",
        })

        assert "Unknown analyzer" in result
        assert "TestVision" in result

    @pytest.mark.asyncio
    async def test_no_media_found(self):
        """When storage has no download for the media_id."""
        storage = FakeMediaStorage()
        mid = uuid4()
        storage.add_media(mid, MediaInfo(mid, "image", "image/jpeg"))
        # No download data added

        tool = _make_tool(storage)
        result = await tool.ainvoke({
            "media_ids": [str(mid)],
            "question": "Describe",
            "analyzer": "TestVision",
        })

        assert "No valid media" in result


class TestConfigValidation:
    """Config validation."""

    def test_missing_storage_raises(self):
        with pytest.raises(TypeError, match="storage"):
            create_analyze_media_tool({}, "description")


class TestResponseSuffix:
    """Configurable response suffix."""

    @pytest.mark.asyncio
    async def test_custom_suffix_in_prompt(self):
        storage = FakeMediaStorage()
        vision = FakeVisionProvider("result")
        mid = uuid4()
        storage.add_media(mid, MediaInfo(mid, "image", "image/jpeg"), _small_jpeg())

        tool = _make_tool(
            storage, vision=vision,
            response_suffix="Reply in JSON format.",
        )
        await tool.ainvoke({
            "media_ids": [str(mid)],
            "question": "Analyze",
            "analyzer": "TestVision",
        })

        assert "JSON format" in vision.analyze_calls[0][2]
        assert "markdown" not in vision.analyze_calls[0][2].lower()

    @pytest.mark.asyncio
    async def test_empty_suffix(self):
        storage = FakeMediaStorage()
        vision = FakeVisionProvider("result")
        mid = uuid4()
        storage.add_media(mid, MediaInfo(mid, "image", "image/jpeg"), _small_jpeg())

        tool = _make_tool(storage, vision=vision, response_suffix="")
        await tool.ainvoke({
            "media_ids": [str(mid)],
            "question": "Analyze",
            "analyzer": "TestVision",
        })

        prompt = vision.analyze_calls[0][2]
        assert prompt == "Analyze"


class TestCachedDescription:
    """Cached descriptions should be returned without calling the provider."""

    @pytest.mark.asyncio
    async def test_cached_description_returned(self):
        storage = FakeMediaStorage()
        vision = FakeVisionProvider()
        mid = uuid4()
        storage.add_media(mid, MediaInfo(mid, "image", "image/jpeg"), _small_jpeg())
        storage.add_content(mid, "description", "Cached result", model_name="TestVision")

        tool = _make_tool(storage, vision=vision)
        result = await tool.ainvoke({
            "media_ids": [str(mid)],
            "question": "Describe",
            "analyzer": "TestVision",
        })

        assert result == "Cached result"
        assert len(vision.analyze_calls) == 0


class TestDocumentRouting:
    """Documents should route through extracted text via TextProvider."""

    @pytest.mark.asyncio
    async def test_document_uses_extracted_text(self):
        storage = FakeMediaStorage()
        text_prov = FakeTextProvider("Summary of the document.")
        mid = uuid4()
        storage.add_media(
            mid,
            MediaInfo(mid, "document", "application/pdf", extraction_status="complete"),
        )
        storage.add_content(mid, "extracted_text", "This is the document text.")

        tool = _make_tool(storage, text=text_prov, user_id=uuid4())
        result = await tool.ainvoke({
            "media_ids": [str(mid)],
            "question": "Summarize",
            "analyzer": "TestVision",
        })

        assert "Summary of the document." in result
        assert len(text_prov.answer_calls) == 1
        assert "document text" in text_prov.answer_calls[0].lower()

    @pytest.mark.asyncio
    async def test_document_pending_extraction(self):
        storage = FakeMediaStorage()
        mid = uuid4()
        storage.add_media(
            mid,
            MediaInfo(mid, "document", "application/pdf", extraction_status="pending"),
        )

        tool = _make_tool(storage, user_id=uuid4())
        result = await tool.ainvoke({
            "media_ids": [str(mid)],
            "question": "Summarize",
            "analyzer": "TestVision",
        })

        assert "still being processed" in result

    @pytest.mark.asyncio
    async def test_document_no_text_extracted(self):
        storage = FakeMediaStorage()
        mid = uuid4()
        storage.add_media(
            mid,
            MediaInfo(mid, "document", "application/pdf", extraction_status="complete"),
        )
        # No extracted_text content added

        tool = _make_tool(storage, user_id=uuid4())
        result = await tool.ainvoke({
            "media_ids": [str(mid)],
            "question": "Summarize",
            "analyzer": "TestVision",
        })

        assert "No text could be extracted" in result

    @pytest.mark.asyncio
    async def test_document_no_text_provider(self):
        """Document analysis without a TextProvider gives a clear error."""
        storage = FakeMediaStorage()
        mid = uuid4()
        storage.add_media(
            mid,
            MediaInfo(mid, "document", "application/pdf", extraction_status="complete"),
        )
        storage.add_content(mid, "extracted_text", "Some text.")

        acfg = AnalyzerConfig(
            name="VisionOnly",
            vision=FakeVisionProvider(),
            text=None,  # No text provider
        )
        tool = create_analyze_media_tool(
            {"storage": storage, "analyzers": {"VisionOnly": acfg}},
            "Analyze",
        )
        result = await tool.ainvoke({
            "media_ids": [str(mid)],
            "question": "Summarize",
            "analyzer": "VisionOnly",
        })

        assert "no text QA capability" in result


class TestAudioVideoRouting:
    """Audio/video should route through transcription."""

    @pytest.mark.asyncio
    async def test_audio_transcription(self):
        storage = FakeMediaStorage()
        transcription = FakeTranscriptionProvider("Hello from audio.")
        mid = uuid4()
        storage.add_media(
            mid,
            MediaInfo(mid, "audio", "audio/mpeg", extraction_status=None),
            b"fake-audio-data",
            "audio/mpeg",
        )

        tool = _make_tool(
            storage,
            transcription=transcription,
            categories={"audio", "video"},
            user_id=uuid4(),
        )
        result = await tool.ainvoke({
            "media_ids": [str(mid)],
            "question": "What is said?",
            "analyzer": "TestVision",
        })

        assert "Hello from audio." in result
        assert "Transcript" in result
        assert len(transcription.calls) == 1

    @pytest.mark.asyncio
    async def test_cached_transcript_returned(self):
        storage = FakeMediaStorage()
        transcription = FakeTranscriptionProvider()
        mid = uuid4()
        storage.add_media(
            mid,
            MediaInfo(mid, "audio", "audio/mpeg", extraction_status="complete"),
            b"fake-audio-data",
        )
        storage.add_content(mid, "transcript", "Previously transcribed.")

        tool = _make_tool(
            storage,
            transcription=transcription,
            categories={"audio"},
            user_id=uuid4(),
        )
        result = await tool.ainvoke({
            "media_ids": [str(mid)],
            "question": "What is said?",
            "analyzer": "TestVision",
        })

        assert "Previously transcribed." in result
        assert len(transcription.calls) == 0  # Not called


class TestCapabilityCheck:
    """Analyzer capability filtering."""

    @pytest.mark.asyncio
    async def test_unsupported_category_rejected(self):
        storage = FakeMediaStorage()
        mid = uuid4()
        storage.add_media(
            mid,
            MediaInfo(mid, "audio", "audio/mpeg"),
            b"data",
        )

        tool = _make_tool(storage, categories={"image"})
        result = await tool.ainvoke({
            "media_ids": [str(mid)],
            "question": "Analyze",
            "analyzer": "TestVision",
        })

        assert "doesn't support audio" in result


class TestDescriptionPersistence:
    """Results should be stored via MediaStorage."""

    @pytest.mark.asyncio
    async def test_description_stored_after_vision(self):
        storage = FakeMediaStorage()
        vision = FakeVisionProvider("Nice photo!")
        uid = uuid4()
        mid = uuid4()
        storage.add_media(mid, MediaInfo(mid, "image", "image/jpeg"), _small_jpeg())

        tool = _make_tool(storage, vision=vision, user_id=uid)
        await tool.ainvoke({
            "media_ids": [str(mid)],
            "question": "Describe",
            "analyzer": "TestVision",
        })

        assert len(storage.store_calls) == 1
        call = storage.store_calls[0]
        assert call["media_id"] == mid
        assert call["content_type"] == "description"
        assert call["content"] == "Nice photo!"
        assert call["metadata"]["model_name"] == "TestVision"

    @pytest.mark.asyncio
    async def test_on_analysis_callback_fires_for_vision(self):
        """The on_analysis callback fires after vision analysis."""
        storage = FakeMediaStorage()
        uid = uuid4()
        mid = uuid4()
        storage.add_media(mid, MediaInfo(mid, "image", "image/jpeg"), _small_jpeg())

        callback_calls: list[tuple[str, str, str]] = []

        async def _cb(media_id_str: str, content_type: str, text: str) -> None:
            callback_calls.append((media_id_str, content_type, text))

        tool = _make_tool(storage, user_id=uid, on_analysis=_cb)
        await tool.ainvoke({
            "media_ids": [str(mid)],
            "question": "Describe",
            "analyzer": "TestVision",
        })

        assert len(callback_calls) == 1
        assert callback_calls[0][0] == str(mid)
        assert callback_calls[0][1] == "description"

    @pytest.mark.asyncio
    async def test_on_analysis_callback_fires_for_transcription(self):
        """The on_analysis callback also fires for new transcripts."""
        storage = FakeMediaStorage()
        transcription = FakeTranscriptionProvider("Hello!")
        uid = uuid4()
        mid = uuid4()
        storage.add_media(
            mid,
            MediaInfo(mid, "audio", "audio/mpeg"),
            b"fake-audio",
            "audio/mpeg",
        )

        callback_calls: list[tuple[str, str, str]] = []

        async def _cb(media_id_str: str, content_type: str, text: str) -> None:
            callback_calls.append((media_id_str, content_type, text))

        tool = _make_tool(
            storage,
            transcription=transcription,
            categories={"audio"},
            user_id=uid,
            on_analysis=_cb,
        )
        await tool.ainvoke({
            "media_ids": [str(mid)],
            "question": "Transcribe",
            "analyzer": "TestVision",
        })

        assert len(callback_calls) == 1
        assert callback_calls[0][1] == "transcript"
        assert callback_calls[0][2] == "Hello!"

    @pytest.mark.asyncio
    async def test_on_analysis_callback_fires_for_document(self):
        """The on_analysis callback fires for document descriptions."""
        storage = FakeMediaStorage()
        text_prov = FakeTextProvider("Doc summary.")
        uid = uuid4()
        mid = uuid4()
        storage.add_media(
            mid,
            MediaInfo(mid, "document", "application/pdf", extraction_status="complete"),
        )
        storage.add_content(mid, "extracted_text", "Doc text.")

        callback_calls: list[tuple[str, str, str]] = []

        async def _cb(media_id_str: str, content_type: str, text: str) -> None:
            callback_calls.append((media_id_str, content_type, text))

        tool = _make_tool(storage, text=text_prov, user_id=uid, on_analysis=_cb)
        await tool.ainvoke({
            "media_ids": [str(mid)],
            "question": "Summarize",
            "analyzer": "TestVision",
        })

        assert len(callback_calls) == 1
        assert callback_calls[0][1] == "description"
        assert callback_calls[0][2] == "Doc summary."


class TestNoUserIdMode:
    """Tool should work without a user_id (no persistence, still analyzes)."""

    @pytest.mark.asyncio
    async def test_analysis_works_without_user_id(self):
        storage = FakeMediaStorage()
        vision = FakeVisionProvider("No user result.")
        mid = uuid4()
        storage.add_media(mid, MediaInfo(mid, "image", "image/jpeg"), _small_jpeg())

        tool = _make_tool(storage, vision=vision, user_id=None)
        result = await tool.ainvoke({
            "media_ids": [str(mid)],
            "question": "Describe",
            "analyzer": "TestVision",
        })

        assert "No user result." in result
        assert len(storage.store_calls) == 0  # Nothing stored
