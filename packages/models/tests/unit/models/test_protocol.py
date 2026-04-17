"""tests for provider protocols: Chat, Embedding, Transcription, ImageGeneration, Speech, and Reranking."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from threetears.agent.tools.protocols import GeneratedImage, ImageGenerationBackend
from threetears.models.messages import ChatMessage, MessageRole, ToolDefinition
from threetears.models.protocol import (
    ChatProvider,
    EmbeddingProvider,
    ImageGenerationProvider,
    RerankingProvider,
    SpeechProvider,
    TranscriptionProvider,
)
from threetears.models.results import (
    ChatChunk,
    ChatResult,
    EmbeddingResult,
    RerankResult,
    SpeechResult,
    TranscriptionResult,
)


# -- ChatProvider tests --


class TestChatProvider:
    """tests for ChatProvider protocol."""

    def test_chat_provider_is_runtime_checkable(self) -> None:
        """ChatProvider is a runtime_checkable Protocol."""
        assert hasattr(ChatProvider, "__protocol_attrs__") or hasattr(ChatProvider, "__abstractmethods__")

    def test_conforming_class_satisfies_protocol(self) -> None:
        """class implementing all ChatProvider methods satisfies isinstance check."""

        class _MockChat:
            async def complete(self, messages: list[ChatMessage], **kwargs: object) -> ChatResult:
                return ChatResult(content="ok")

            async def stream(self, messages: list[ChatMessage], **kwargs: object) -> AsyncIterator[ChatChunk]:
                yield ChatChunk(content="chunk")

            def bind_tools(self, tools: list[ToolDefinition]) -> None:
                pass

            def preprocess(self, messages: list[ChatMessage]) -> list[ChatMessage]:
                return messages

        provider = _MockChat()
        assert isinstance(provider, ChatProvider)

    def test_non_conforming_class_fails_protocol(self) -> None:
        """class without ChatProvider methods does not satisfy isinstance check."""

        class _NotAChat:
            async def some_method(self) -> None:
                pass

        obj = _NotAChat()
        assert not isinstance(obj, ChatProvider)

    @pytest.mark.asyncio
    async def test_conforming_chat_complete(self) -> None:
        """conforming ChatProvider can execute complete and return ChatResult."""

        class _EchoChat:
            async def complete(self, messages: list[ChatMessage], **kwargs: object) -> ChatResult:
                last = messages[-1].content if messages else ""
                content = last if isinstance(last, str) else str(last)
                return ChatResult(content=f"echo: {content}", model="test-model")

            async def stream(self, messages: list[ChatMessage], **kwargs: object) -> AsyncIterator[ChatChunk]:
                yield ChatChunk(content="chunk")

            def bind_tools(self, tools: list[ToolDefinition]) -> None:
                pass

            def preprocess(self, messages: list[ChatMessage]) -> list[ChatMessage]:
                return messages

        provider = _EchoChat()
        result = await provider.complete(
            [
                ChatMessage(role=MessageRole.USER, content="hello"),
            ]
        )
        assert result.content == "echo: hello"
        assert result.model == "test-model"

    @pytest.mark.asyncio
    async def test_conforming_chat_stream(self) -> None:
        """conforming ChatProvider can execute stream and yield ChatChunk."""

        class _StreamChat:
            async def complete(self, messages: list[ChatMessage], **kwargs: object) -> ChatResult:
                return ChatResult(content="ok")

            async def stream(self, messages: list[ChatMessage], **kwargs: object) -> AsyncIterator[ChatChunk]:
                yield ChatChunk(content="hello ")
                yield ChatChunk(content="world", finish_reason="stop")

            def bind_tools(self, tools: list[ToolDefinition]) -> None:
                pass

            def preprocess(self, messages: list[ChatMessage]) -> list[ChatMessage]:
                return messages

        provider = _StreamChat()
        chunks: list[ChatChunk] = []
        async for chunk in provider.stream(
            [
                ChatMessage(role=MessageRole.USER, content="hi"),
            ]
        ):
            chunks.append(chunk)
        assert len(chunks) == 2
        assert chunks[0].content == "hello "
        assert chunks[1].finish_reason == "stop"


# -- EmbeddingProvider tests --


class TestEmbeddingProvider:
    """tests for EmbeddingProvider protocol."""

    def test_embedding_provider_is_runtime_checkable(self) -> None:
        """EmbeddingProvider is a runtime_checkable Protocol."""
        assert hasattr(EmbeddingProvider, "__protocol_attrs__") or hasattr(EmbeddingProvider, "__abstractmethods__")

    def test_conforming_class_satisfies_protocol(self) -> None:
        """class implementing all EmbeddingProvider methods satisfies isinstance check."""

        class _MockEmbed:
            @property
            def dimensions(self) -> int:
                return 1536

            async def embed(self, text: str) -> EmbeddingResult:
                return EmbeddingResult(
                    vector=[0.0] * 1536,
                    token_count=1,
                    dimensions=1536,
                    model="test",
                )

            async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
                return []

        provider = _MockEmbed()
        assert isinstance(provider, EmbeddingProvider)

    def test_non_conforming_class_fails_protocol(self) -> None:
        """class without EmbeddingProvider methods does not satisfy isinstance check."""

        class _NotAnEmbed:
            pass

        obj = _NotAnEmbed()
        assert not isinstance(obj, EmbeddingProvider)

    @pytest.mark.asyncio
    async def test_conforming_embed_returns_result(self) -> None:
        """conforming EmbeddingProvider can execute embed and return EmbeddingResult."""

        class _SimpleEmbed:
            @property
            def dimensions(self) -> int:
                return 3

            async def embed(self, text: str) -> EmbeddingResult:
                return EmbeddingResult(
                    vector=[0.1, 0.2, 0.3],
                    token_count=len(text.split()),
                    dimensions=3,
                    model="test-embed",
                )

            async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
                results = []
                for text in texts:
                    results.append(await self.embed(text))
                return results

        provider = _SimpleEmbed()
        assert provider.dimensions == 3
        result = await provider.embed("hello world")
        assert result.dimensions == 3
        assert result.token_count == 2
        assert len(result.vector) == 3

    @pytest.mark.asyncio
    async def test_conforming_embed_batch(self) -> None:
        """conforming EmbeddingProvider can execute embed_batch."""

        class _BatchEmbed:
            @property
            def dimensions(self) -> int:
                return 2

            async def embed(self, text: str) -> EmbeddingResult:
                return EmbeddingResult(vector=[0.1, 0.2], token_count=1, dimensions=2, model="test")

            async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
                results = []
                for text in texts:
                    results.append(await self.embed(text))
                return results

        provider = _BatchEmbed()
        results = await provider.embed_batch(["one", "two", "three"])
        assert len(results) == 3


# -- TranscriptionProvider tests --


class TestTranscriptionProvider:
    """tests for TranscriptionProvider protocol."""

    def test_transcription_provider_is_runtime_checkable(self) -> None:
        """TranscriptionProvider is a runtime_checkable Protocol."""
        assert hasattr(TranscriptionProvider, "__protocol_attrs__") or hasattr(
            TranscriptionProvider, "__abstractmethods__"
        )

    def test_conforming_class_satisfies_protocol(self) -> None:
        """class implementing transcribe satisfies TranscriptionProvider isinstance check."""

        class _MockTranscribe:
            async def transcribe(
                self,
                audio_data: bytes,
                mime_type: str,
                *,
                language_hint: str | None = None,
            ) -> TranscriptionResult:
                return TranscriptionResult(text="hello")

        provider = _MockTranscribe()
        assert isinstance(provider, TranscriptionProvider)

    def test_non_conforming_class_fails_protocol(self) -> None:
        """class without transcribe does not satisfy TranscriptionProvider isinstance check."""

        class _NotATranscriber:
            async def other_method(self) -> None:
                pass

        obj = _NotATranscriber()
        assert not isinstance(obj, TranscriptionProvider)

    @pytest.mark.asyncio
    async def test_conforming_transcribe_returns_result(self) -> None:
        """conforming TranscriptionProvider can execute transcribe and return TranscriptionResult."""

        class _SimpleTranscribe:
            async def transcribe(
                self,
                audio_data: bytes,
                mime_type: str,
                *,
                language_hint: str | None = None,
            ) -> TranscriptionResult:
                return TranscriptionResult(
                    text="transcribed text",
                    language=language_hint or "en",
                    duration_seconds=5.0,
                )

        provider = _SimpleTranscribe()
        result = await provider.transcribe(b"audio-bytes", "audio/wav", language_hint="en")
        assert result.text == "transcribed text"
        assert result.language == "en"
        assert result.duration_seconds == 5.0

    @pytest.mark.asyncio
    async def test_conforming_transcribe_without_language_hint(self) -> None:
        """conforming TranscriptionProvider works without language_hint."""

        class _AutoDetect:
            async def transcribe(
                self,
                audio_data: bytes,
                mime_type: str,
                *,
                language_hint: str | None = None,
            ) -> TranscriptionResult:
                return TranscriptionResult(text="detected", language="fr")

        provider = _AutoDetect()
        result = await provider.transcribe(b"audio-bytes", "audio/mp3")
        assert result.text == "detected"
        assert result.language == "fr"


# -- ImageGenerationProvider tests --


class TestImageGenerationProvider:
    """tests for ImageGenerationProvider alias."""

    def test_image_generation_provider_is_alias(self) -> None:
        """ImageGenerationProvider is alias for ImageGenerationBackend."""
        assert ImageGenerationProvider is ImageGenerationBackend

    def test_image_generation_provider_is_runtime_checkable(self) -> None:
        """ImageGenerationProvider is a runtime_checkable Protocol."""
        assert hasattr(ImageGenerationProvider, "__protocol_attrs__") or hasattr(
            ImageGenerationProvider, "__abstractmethods__"
        )

    def test_conforming_class_satisfies_protocol(self) -> None:
        """class implementing generate satisfies ImageGenerationProvider isinstance check."""

        class _MockImageGen:
            async def generate(
                self,
                prompt: str,
                *,
                style: str | None = None,
                source_image: bytes | None = None,
                source_mime_type: str | None = None,
            ) -> GeneratedImage:
                return GeneratedImage(
                    data=b"image-bytes",
                    mime_type="image/png",
                    width=512,
                    height=512,
                )

        provider = _MockImageGen()
        assert isinstance(provider, ImageGenerationProvider)

    def test_non_conforming_class_fails_protocol(self) -> None:
        """class without generate does not satisfy ImageGenerationProvider isinstance check."""

        class _NotAnImageGen:
            pass

        obj = _NotAnImageGen()
        assert not isinstance(obj, ImageGenerationProvider)

    @pytest.mark.asyncio
    async def test_conforming_generate_returns_result(self) -> None:
        """conforming ImageGenerationProvider can execute generate and return GeneratedImage."""

        class _SimpleImageGen:
            async def generate(
                self,
                prompt: str,
                *,
                style: str | None = None,
                source_image: bytes | None = None,
                source_mime_type: str | None = None,
            ) -> GeneratedImage:
                return GeneratedImage(
                    data=b"png-data",
                    mime_type="image/png",
                    width=1024,
                    height=1024,
                )

        provider = _SimpleImageGen()
        result = await provider.generate("a sunset over mountains")
        assert result.mime_type == "image/png"
        assert result.width == 1024


# -- SpeechProvider tests --


class TestSpeechProvider:
    """tests for SpeechProvider protocol."""

    def test_speech_provider_is_runtime_checkable(self) -> None:
        """SpeechProvider is a runtime_checkable Protocol."""
        assert hasattr(SpeechProvider, "__protocol_attrs__") or hasattr(SpeechProvider, "__abstractmethods__")

    def test_conforming_class_satisfies_protocol(self) -> None:
        """class implementing synthesize satisfies SpeechProvider isinstance check."""

        class _MockSpeech:
            async def synthesize(
                self,
                text: str,
                *,
                voice: str | None = None,
                output_format: str | None = None,
                speed: float | None = None,
            ) -> SpeechResult:
                return SpeechResult(data=b"audio", mime_type="audio/mp3")

        provider = _MockSpeech()
        assert isinstance(provider, SpeechProvider)

    def test_non_conforming_class_fails_protocol(self) -> None:
        """class without synthesize does not satisfy SpeechProvider isinstance check."""

        class _NotASpeech:
            async def other_method(self) -> None:
                pass

        obj = _NotASpeech()
        assert not isinstance(obj, SpeechProvider)

    @pytest.mark.asyncio
    async def test_conforming_synthesize_returns_result(self) -> None:
        """conforming SpeechProvider can execute synthesize and return SpeechResult."""

        class _SimpleSpeech:
            async def synthesize(
                self,
                text: str,
                *,
                voice: str | None = None,
                output_format: str | None = None,
                speed: float | None = None,
            ) -> SpeechResult:
                return SpeechResult(
                    data=b"audio-bytes",
                    mime_type="audio/mp3",
                    duration_seconds=2.5,
                    sample_rate=24000,
                )

        provider = _SimpleSpeech()
        result = await provider.synthesize("hello world", voice="alloy")
        assert result.data == b"audio-bytes"
        assert result.mime_type == "audio/mp3"
        assert result.duration_seconds == 2.5
        assert result.sample_rate == 24000


# -- RerankingProvider tests --


class TestRerankingProvider:
    """tests for RerankingProvider protocol."""

    def test_reranking_provider_is_runtime_checkable(self) -> None:
        """RerankingProvider is a runtime_checkable Protocol."""
        assert hasattr(RerankingProvider, "__protocol_attrs__") or hasattr(RerankingProvider, "__abstractmethods__")

    def test_conforming_class_satisfies_protocol(self) -> None:
        """class implementing rerank satisfies RerankingProvider isinstance check."""

        class _MockRerank:
            async def rerank(
                self,
                query: str,
                documents: list[str],
                *,
                top_k: int | None = None,
            ) -> list[RerankResult]:
                return [RerankResult(index=0, relevance_score=0.9)]

        provider = _MockRerank()
        assert isinstance(provider, RerankingProvider)

    def test_non_conforming_class_fails_protocol(self) -> None:
        """class without rerank does not satisfy RerankingProvider isinstance check."""

        class _NotAReranker:
            pass

        obj = _NotAReranker()
        assert not isinstance(obj, RerankingProvider)

    @pytest.mark.asyncio
    async def test_conforming_rerank_returns_result(self) -> None:
        """conforming RerankingProvider can execute rerank and return list of RerankResult."""

        class _SimpleRerank:
            async def rerank(
                self,
                query: str,
                documents: list[str],
                *,
                top_k: int | None = None,
            ) -> list[RerankResult]:
                scored = [
                    RerankResult(index=i, relevance_score=1.0 / (i + 1), text=doc) for i, doc in enumerate(documents)
                ]
                scored.sort(key=lambda r: r.relevance_score, reverse=True)
                if top_k is not None:
                    scored = scored[:top_k]
                return scored

        provider = _SimpleRerank()
        results = await provider.rerank(
            "python programming",
            ["python guide", "java tutorial", "python basics"],
            top_k=2,
        )
        assert len(results) == 2
        assert results[0].relevance_score >= results[1].relevance_score
        assert results[0].text is not None
