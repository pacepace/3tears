"""openai-compatible chat and embedding provider adapters wrapping langchain-openai."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from threetears.models.messages import ChatMessage, ToolDefinition
from threetears.models.providers._conversions import (
    ai_chunk_to_chat_chunk,
    ai_message_to_result,
    messages_to_lc,
    tool_def_to_lc,
)
from threetears.models.results import ChatChunk, ChatResult, EmbeddingResult

__all__ = [
    "OpenAIChatProvider",
    "OpenAIEmbeddingProvider",
]

# default embedding dimensions for ada-002 compatibility
_DEFAULT_EMBEDDING_DIMENSIONS = 1536


class OpenAIChatProvider:
    """chat provider adapter for OpenAI-compatible models via langchain-openai.

    wraps ChatOpenAI with lazy instantiation, converting between
    threetears message types and LangChain message types at boundaries.
    supports any OpenAI-compatible API via base_url parameter.

    :param model_name: OpenAI model identifier (e.g. gpt-4o)
    :ptype model_name: str
    :param api_key: API key for authentication
    :ptype api_key: str
    :param base_url: optional custom API base URL (passed through as-is)
    :ptype base_url: str | None
    :param timeout: request timeout in seconds
    :ptype timeout: int
    :param max_retries: maximum retry attempts for failed requests
    :ptype max_retries: int
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        *,
        base_url: str | None = None,
        timeout: int = 120,
        max_retries: int = 2,
    ) -> None:
        self.model_name = model_name
        self._api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self._max_retries = max_retries
        self.model: Any = None
        self.tools: list[ToolDefinition] | None = None

    def _get_model(self) -> Any:
        """lazily creates and caches ChatOpenAI instance.

        imports langchain_openai on first call to avoid module-level
        dependency on optional package. enables stream_usage for token
        counting in streaming responses.

        :return: configured ChatOpenAI instance, optionally with tools bound
        :rtype: Any
        """
        if self.model is not None:
            return self.model

        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "api_key": self._api_key,
            "timeout": self.timeout,
            "max_retries": self._max_retries,
            "stream_usage": True,
        }
        if self.base_url is not None:
            kwargs["base_url"] = self.base_url

        base_model: Any = ChatOpenAI(**kwargs)

        if self.tools:
            lc_tools = [tool_def_to_lc(t) for t in self.tools]
            base_model = base_model.bind_tools(lc_tools)

        self.model = base_model
        return self.model

    async def complete(self, messages: list[ChatMessage], **kwargs: Any) -> ChatResult:
        """generates chat completion from message history.

        converts threetears messages to LangChain format, invokes model,
        and converts response back to ChatResult.

        :param messages: ordered list of conversation messages
        :ptype messages: list[ChatMessage]
        :param kwargs: additional parameters passed to LangChain ainvoke
        :ptype kwargs: Any
        :return: chat completion result with content, tool calls, and usage
        :rtype: ChatResult
        """
        lc_messages = messages_to_lc(messages)
        response = await self._get_model().ainvoke(lc_messages, **kwargs)
        result = ai_message_to_result(response)
        return result

    async def stream(self, messages: list[ChatMessage], **kwargs: Any) -> AsyncIterator[ChatChunk]:
        """streams chat completion chunks from message history.

        converts threetears messages to LangChain format and yields
        converted chunks from async stream.

        :param messages: ordered list of conversation messages
        :ptype messages: list[ChatMessage]
        :param kwargs: additional parameters passed to LangChain astream
        :ptype kwargs: Any
        :return: async iterator of chat completion chunks
        :rtype: AsyncIterator[ChatChunk]
        """
        lc_messages = messages_to_lc(messages)
        async for chunk in self._get_model().astream(lc_messages, **kwargs):
            yield ai_chunk_to_chat_chunk(chunk)

    def bind_tools(self, tools: list[ToolDefinition]) -> None:
        """binds tool definitions for subsequent completions.

        stores tools and clears cached model instance so next call
        recreates model with tools bound.

        :param tools: tool definitions available to model
        :ptype tools: list[ToolDefinition]
        """
        self.tools = list(tools)
        self.model = None

    def preprocess(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """preprocesses messages before sending to OpenAI model.

        applies capability-based transforms via preprocessing pipeline.
        OpenAI models do not require alternating roles, so this is
        effectively passthrough for standard configurations.

        :param messages: raw conversation messages
        :ptype messages: list[ChatMessage]
        :return: preprocessed messages ready for model
        :rtype: list[ChatMessage]
        """
        from threetears.models.capabilities import ModelCapabilities
        from threetears.models.enums import ModelStatus, ModelTier, ModelType
        from threetears.models.preprocessing import preprocess_messages

        capabilities = ModelCapabilities(
            model_name=self.model_name,
            model_type=ModelType.CHAT,
            model_tier=ModelTier.LARGE,
            model_status=ModelStatus.ACTIVE,
            requires_alternating_roles=False,
        )
        result = preprocess_messages(messages, capabilities)
        return result


class OpenAIEmbeddingProvider:
    """embedding provider adapter for OpenAI-compatible models via langchain-openai.

    wraps OpenAIEmbeddings with lazy instantiation for single and batch
    embedding operations. supports configurable dimensions for models
    that allow it (e.g. text-embedding-3-small).

    :param model_name: OpenAI embedding model identifier
    :ptype model_name: str
    :param api_key: API key for authentication
    :ptype api_key: str
    :param base_url: optional custom API base URL (passed through as-is)
    :ptype base_url: str | None
    :param embedding_dimensions: optional output vector dimensionality
    :ptype embedding_dimensions: int | None
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        *,
        base_url: str | None = None,
        embedding_dimensions: int | None = None,
    ) -> None:
        self.model_name = model_name
        self._api_key = api_key
        self.base_url = base_url
        self._embedding_dimensions = embedding_dimensions
        self.model: Any = None

    def _get_model(self) -> Any:
        """lazily creates and caches OpenAIEmbeddings instance.

        imports langchain_openai on first call to avoid module-level
        dependency on optional package.

        :return: configured OpenAIEmbeddings instance
        :rtype: Any
        """
        if self.model is not None:
            return self.model

        from langchain_openai import OpenAIEmbeddings

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "api_key": self._api_key,
        }
        if self.base_url is not None:
            kwargs["base_url"] = self.base_url
        if self._embedding_dimensions is not None:
            kwargs["dimensions"] = self._embedding_dimensions

        self.model = OpenAIEmbeddings(**kwargs)
        return self.model

    @property
    def dimensions(self) -> int:
        """number of dimensions in embedding vectors.

        returns configured dimensions if set, otherwise defaults to
        1536 for ada-002 compatibility.

        :return: embedding vector dimensionality
        :rtype: int
        """
        if self._embedding_dimensions is not None:
            return self._embedding_dimensions
        return _DEFAULT_EMBEDDING_DIMENSIONS

    async def embed(self, text: str) -> EmbeddingResult:
        """generates embedding vector for single text input.

        delegates to embed_batch and returns first result.

        :param text: text to embed
        :ptype text: str
        :return: embedding result with vector and metadata
        :rtype: EmbeddingResult
        """
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """generates embedding vectors for batch of text inputs.

        calls OpenAIEmbeddings.aembed_documents and converts raw float
        vectors into EmbeddingResult objects with metadata.

        :param texts: list of texts to embed
        :ptype texts: list[str]
        :return: list of embedding results in same order as inputs
        :rtype: list[EmbeddingResult]
        """
        raw_embeddings: list[list[float]] = await self._get_model().aembed_documents(texts)

        results: list[EmbeddingResult] = []
        for text, embedding in zip(texts, raw_embeddings):
            results.append(
                EmbeddingResult(
                    vector=embedding,
                    dimensions=len(embedding),
                    model=self.model_name,
                    token_count=len(text) // 4,
                )
            )
        return results
