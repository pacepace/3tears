"""result types returned by AI model providers."""

from __future__ import annotations

from dataclasses import dataclass

from threetears.models.messages import ToolCallRequest

__all__ = [
    "ChatChunk",
    "ChatResult",
    "EmbeddingResult",
    "RerankResult",
    "SpeechResult",
    "TranscriptionResult",
    "TranscriptionSegment",
]


@dataclass
class TranscriptionSegment:
    """time-aligned segment within transcription result.

    :param start: segment start time in seconds
    :ptype start: float
    :param end: segment end time in seconds
    :ptype end: float
    :param text: transcribed text for segment
    :ptype text: str
    """

    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    """result from audio transcription provider.

    :param text: full transcription text
    :ptype text: str
    :param language: detected or specified language code
    :ptype language: str | None
    :param duration_seconds: audio duration in seconds
    :ptype duration_seconds: float | None
    :param segments: time-aligned transcript segments
    :ptype segments: list[TranscriptionSegment] | None
    """

    text: str
    language: str | None = None
    duration_seconds: float | None = None
    segments: list[TranscriptionSegment] | None = None


@dataclass
class EmbeddingResult:
    """result from text embedding provider.

    :param vector: embedding vector as list of floats
    :ptype vector: list[float]
    :param token_count: number of tokens consumed
    :ptype token_count: int
    :param dimensions: dimensionality of embedding vector
    :ptype dimensions: int
    :param model: model identifier used for embedding
    :ptype model: str
    """

    vector: list[float]
    token_count: int
    dimensions: int
    model: str


@dataclass
class ChatResult:
    """result from chat completion provider.

    :param content: generated text content
    :ptype content: str
    :param tool_calls: tool invocation requests from model
    :ptype tool_calls: list[ToolCallRequest] | None
    :param model: model identifier used for completion
    :ptype model: str
    :param usage: token usage counts with input_tokens and output_tokens keys
    :ptype usage: dict[str, int] | None
    """

    content: str
    tool_calls: list[ToolCallRequest] | None = None
    model: str = ""
    usage: dict[str, int] | None = None


@dataclass
class ChatChunk:
    """single chunk from streaming chat completion.

    :param content: partial text content in chunk
    :ptype content: str
    :param tool_calls: partial tool call data in chunk
    :ptype tool_calls: list[ToolCallRequest] | None
    :param finish_reason: reason stream ended, if final chunk
    :ptype finish_reason: str | None
    """

    content: str = ""
    tool_calls: list[ToolCallRequest] | None = None
    finish_reason: str | None = None


@dataclass
class SpeechResult:
    """result from text-to-speech synthesis provider.

    :param data: generated audio bytes
    :ptype data: bytes
    :param mime_type: audio MIME type (e.g. "audio/mp3", "audio/wav")
    :ptype mime_type: str
    :param duration_seconds: duration of generated audio in seconds
    :ptype duration_seconds: float | None
    :param sample_rate: sample rate in Hz
    :ptype sample_rate: int | None
    """

    data: bytes
    mime_type: str
    duration_seconds: float | None = None
    sample_rate: int | None = None


@dataclass
class RerankResult:
    """result for single document from re-ranking provider.

    :param index: original index of document in input list
    :ptype index: int
    :param relevance_score: relevance score (0.0-1.0)
    :ptype relevance_score: float
    :param text: optionally echoed back document text
    :ptype text: str | None
    """

    index: int
    relevance_score: float
    text: str | None = None
