"""tests for result dataclasses returned by AI model providers."""

from __future__ import annotations

import dataclasses

from pydantic import BaseModel

from threetears.models.results import (
    ChatChunk,
    ChatResult,
    EmbeddingResult,
    RerankResult,
    SpeechResult,
    TranscriptionResult,
    TranscriptionSegment,
)


class TestTranscriptionSegment:
    """tests for TranscriptionSegment dataclass."""

    def test_is_dataclass(self) -> None:
        """TranscriptionSegment is a dataclass."""
        assert dataclasses.is_dataclass(TranscriptionSegment)

    def test_is_not_pydantic(self) -> None:
        """TranscriptionSegment is not a Pydantic model."""
        assert not issubclass(TranscriptionSegment, BaseModel)

    def test_required_fields(self) -> None:
        """TranscriptionSegment requires start, end, and text."""
        segment = TranscriptionSegment(start=0.0, end=1.5, text="hello")
        assert segment.start == 0.0
        assert segment.end == 1.5
        assert segment.text == "hello"


class TestTranscriptionResult:
    """tests for TranscriptionResult dataclass."""

    def test_is_dataclass(self) -> None:
        """TranscriptionResult is a dataclass."""
        assert dataclasses.is_dataclass(TranscriptionResult)

    def test_is_not_pydantic(self) -> None:
        """TranscriptionResult is not a Pydantic model."""
        assert not issubclass(TranscriptionResult, BaseModel)

    def test_required_fields(self) -> None:
        """TranscriptionResult requires text."""
        result = TranscriptionResult(text="transcribed")
        assert result.text == "transcribed"

    def test_defaults(self) -> None:
        """TranscriptionResult optional fields default to None."""
        result = TranscriptionResult(text="transcribed")
        assert result.language is None
        assert result.duration_seconds is None
        assert result.segments is None

    def test_all_fields(self) -> None:
        """TranscriptionResult stores all fields correctly."""
        segments = [TranscriptionSegment(start=0.0, end=1.0, text="hello")]
        result = TranscriptionResult(
            text="hello world",
            language="en",
            duration_seconds=5.0,
            segments=segments,
        )
        assert result.text == "hello world"
        assert result.language == "en"
        assert result.duration_seconds == 5.0
        assert result.segments is not None
        assert len(result.segments) == 1


class TestEmbeddingResult:
    """tests for EmbeddingResult dataclass."""

    def test_is_dataclass(self) -> None:
        """EmbeddingResult is a dataclass."""
        assert dataclasses.is_dataclass(EmbeddingResult)

    def test_is_not_pydantic(self) -> None:
        """EmbeddingResult is not a Pydantic model."""
        assert not issubclass(EmbeddingResult, BaseModel)

    def test_required_fields(self) -> None:
        """EmbeddingResult requires vector, token_count, dimensions, and model."""
        result = EmbeddingResult(
            vector=[0.1, 0.2, 0.3],
            token_count=2,
            dimensions=3,
            model="test-embed",
        )
        assert result.vector == [0.1, 0.2, 0.3]
        assert result.token_count == 2
        assert result.dimensions == 3
        assert result.model == "test-embed"


class TestChatResult:
    """tests for ChatResult dataclass."""

    def test_is_dataclass(self) -> None:
        """ChatResult is a dataclass."""
        assert dataclasses.is_dataclass(ChatResult)

    def test_is_not_pydantic(self) -> None:
        """ChatResult is not a Pydantic model."""
        assert not issubclass(ChatResult, BaseModel)

    def test_required_fields(self) -> None:
        """ChatResult requires content."""
        result = ChatResult(content="response")
        assert result.content == "response"

    def test_defaults(self) -> None:
        """ChatResult optional fields have correct defaults."""
        result = ChatResult(content="response")
        assert result.tool_calls is None
        assert result.model == ""
        assert result.usage is None

    def test_all_fields(self) -> None:
        """ChatResult stores all fields correctly."""
        result = ChatResult(
            content="response",
            tool_calls=[],
            model="claude-sonnet-4-20250514",
            usage={"input_tokens": 10, "output_tokens": 5},
        )
        assert result.content == "response"
        assert result.tool_calls == []
        assert result.model == "claude-sonnet-4-20250514"
        assert result.usage == {"input_tokens": 10, "output_tokens": 5}


class TestChatChunk:
    """tests for ChatChunk dataclass."""

    def test_is_dataclass(self) -> None:
        """ChatChunk is a dataclass."""
        assert dataclasses.is_dataclass(ChatChunk)

    def test_is_not_pydantic(self) -> None:
        """ChatChunk is not a Pydantic model."""
        assert not issubclass(ChatChunk, BaseModel)

    def test_defaults(self) -> None:
        """ChatChunk fields have correct defaults."""
        chunk = ChatChunk()
        assert chunk.content == ""
        assert chunk.tool_calls is None
        assert chunk.finish_reason is None

    def test_all_fields(self) -> None:
        """ChatChunk stores all fields correctly."""
        chunk = ChatChunk(
            content="hello",
            tool_calls=[],
            finish_reason="stop",
        )
        assert chunk.content == "hello"
        assert chunk.tool_calls == []
        assert chunk.finish_reason == "stop"


class TestSpeechResult:
    """tests for SpeechResult dataclass."""

    def test_is_dataclass(self) -> None:
        """SpeechResult is a dataclass."""
        assert dataclasses.is_dataclass(SpeechResult)

    def test_is_not_pydantic(self) -> None:
        """SpeechResult is not a Pydantic model."""
        assert not issubclass(SpeechResult, BaseModel)

    def test_required_fields(self) -> None:
        """SpeechResult requires data and mime_type."""
        result = SpeechResult(data=b"audio", mime_type="audio/mp3")
        assert result.data == b"audio"
        assert result.mime_type == "audio/mp3"

    def test_defaults(self) -> None:
        """SpeechResult optional fields default to None."""
        result = SpeechResult(data=b"audio", mime_type="audio/mp3")
        assert result.duration_seconds is None
        assert result.sample_rate is None

    def test_all_fields(self) -> None:
        """SpeechResult stores all fields correctly."""
        result = SpeechResult(
            data=b"audio-bytes",
            mime_type="audio/wav",
            duration_seconds=3.5,
            sample_rate=44100,
        )
        assert result.data == b"audio-bytes"
        assert result.mime_type == "audio/wav"
        assert result.duration_seconds == 3.5
        assert result.sample_rate == 44100


class TestRerankResult:
    """tests for RerankResult dataclass."""

    def test_is_dataclass(self) -> None:
        """RerankResult is a dataclass."""
        assert dataclasses.is_dataclass(RerankResult)

    def test_is_not_pydantic(self) -> None:
        """RerankResult is not a Pydantic model."""
        assert not issubclass(RerankResult, BaseModel)

    def test_required_fields(self) -> None:
        """RerankResult requires index and relevance_score."""
        result = RerankResult(index=0, relevance_score=0.95)
        assert result.index == 0
        assert result.relevance_score == 0.95

    def test_defaults(self) -> None:
        """RerankResult optional fields default to None."""
        result = RerankResult(index=0, relevance_score=0.9)
        assert result.text is None

    def test_all_fields(self) -> None:
        """RerankResult stores all fields correctly."""
        result = RerankResult(
            index=2,
            relevance_score=0.87,
            text="document content",
        )
        assert result.index == 2
        assert result.relevance_score == 0.87
        assert result.text == "document content"
