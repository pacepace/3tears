"""provider protocols defining contracts for AI model integrations."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from threetears.agent.tools.protocols import ImageGenerationBackend
from threetears.models.messages import ChatMessage, ToolDefinition
from threetears.models.results import (
    ChatChunk,
    ChatResult,
    EmbeddingResult,
    RerankResult,
    SpeechResult,
    TranscriptionResult,
)


@runtime_checkable
class ChatProvider(Protocol):
    """protocol for chat completion model providers.

    implementations wrap specific AI model APIs (anthropic, openai, openrouter)
    behind unified interface for chat completion and streaming.
    """

    async def complete(self, messages: list[ChatMessage], **kwargs: Any) -> ChatResult:
        """generate chat completion from message history.

        :param messages: ordered list of conversation messages
        :ptype messages: list[ChatMessage]
        :param kwargs: provider-specific parameters
        :ptype kwargs: Any
        :return: chat completion result
        :rtype: ChatResult
        """
        ...

    async def stream(self, messages: list[ChatMessage], **kwargs: Any) -> AsyncIterator[ChatChunk]:
        """stream chat completion chunks from message history.

        :param messages: ordered list of conversation messages
        :ptype messages: list[ChatMessage]
        :param kwargs: provider-specific parameters
        :ptype kwargs: Any
        :return: async iterator of chat completion chunks
        :rtype: AsyncIterator[ChatChunk]
        """
        ...

    def bind_tools(self, tools: list[ToolDefinition]) -> None:
        """bind tool definitions for subsequent completions.

        :param tools: tool definitions available to model
        :ptype tools: list[ToolDefinition]
        """
        ...

    def preprocess(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """preprocess messages before sending to model.

        applies provider-specific transformations such as role alternation
        or content format conversion.

        :param messages: raw conversation messages
        :ptype messages: list[ChatMessage]
        :return: preprocessed messages ready for model
        :rtype: list[ChatMessage]
        """
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """protocol for text embedding model providers.

    implementations wrap embedding APIs (voyageai, openai) behind unified
    interface for single and batch embedding operations.
    """

    @property
    def dimensions(self) -> int:
        """number of dimensions in embedding vectors.

        :return: embedding vector dimensionality
        :rtype: int
        """
        ...

    async def embed(self, text: str) -> EmbeddingResult:
        """generate embedding vector for single text input.

        :param text: text to embed
        :ptype text: str
        :return: embedding result with vector and metadata
        :rtype: EmbeddingResult
        """
        ...

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """generate embedding vectors for batch of text inputs.

        :param texts: list of texts to embed
        :ptype texts: list[str]
        :return: list of embedding results in same order as inputs
        :rtype: list[EmbeddingResult]
        """
        ...


@runtime_checkable
class TranscriptionProvider(Protocol):
    """protocol for audio transcription model providers.

    implementations wrap transcription APIs (whisper, etc.) behind unified
    interface for audio-to-text conversion with optional segmentation.
    """

    async def transcribe(
        self,
        audio_data: bytes,
        mime_type: str,
        *,
        language_hint: str | None = None,
    ) -> TranscriptionResult:
        """transcribe audio data to text.

        :param audio_data: raw audio bytes
        :ptype audio_data: bytes
        :param mime_type: MIME type of audio data
        :ptype mime_type: str
        :param language_hint: optional language code hint for transcription
        :ptype language_hint: str | None
        :return: transcription result with text and optional segments
        :rtype: TranscriptionResult
        """
        ...


@runtime_checkable
class SpeechProvider(Protocol):
    """protocol for text-to-speech synthesis model providers.

    implementations wrap speech synthesis APIs behind unified
    interface for text-to-audio conversion with voice and format options.
    """

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        output_format: str | None = None,
        speed: float | None = None,
    ) -> SpeechResult:
        """synthesize speech audio from text input.

        :param text: text to synthesize as speech
        :ptype text: str
        :param voice: voice identifier for synthesis
        :ptype voice: str | None
        :param output_format: desired audio output format (e.g. "mp3", "wav")
        :ptype output_format: str | None
        :param speed: playback speed multiplier
        :ptype speed: float | None
        :return: speech synthesis result with audio data
        :rtype: SpeechResult
        """
        ...


@runtime_checkable
class RerankingProvider(Protocol):
    """protocol for document re-ranking model providers.

    implementations wrap reranking APIs behind unified interface
    for scoring and sorting documents by relevance to query.
    """

    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """rerank documents by relevance to query.

        :param query: query string to rank documents against
        :ptype query: str
        :param documents: list of document texts to rerank
        :ptype documents: list[str]
        :param top_k: maximum number of results to return
        :ptype top_k: int | None
        :return: reranked documents sorted by relevance score descending
        :rtype: list[RerankResult]
        """
        ...


ImageGenerationProvider = ImageGenerationBackend
"""alias for image generation protocol from agent-tools package."""
