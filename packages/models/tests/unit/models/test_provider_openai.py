"""tests for OpenAIChatProvider and OpenAIEmbeddingProvider adapters."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from threetears.models.messages import ChatMessage, MessageRole, ToolDefinition
from threetears.models.protocol import ChatProvider, EmbeddingProvider
from threetears.models.providers.openai import (
    OpenAIChatProvider,
    OpenAIEmbeddingProvider,
    _DEFAULT_EMBEDDING_DIMENSIONS,
)
from threetears.models.results import ChatChunk, ChatResult, EmbeddingResult


def _mock_ai_message(
    content: str = "hello",
    tool_calls: list[dict[str, object]] | None = None,
    usage_metadata: dict[str, int] | None = None,
    response_metadata: dict[str, str] | None = None,
) -> MagicMock:
    """creates mock LangChain AIMessage for testing.

    :param content: message content text
    :ptype content: str
    :param tool_calls: list of tool call dicts
    :ptype tool_calls: list[dict[str, object]] | None
    :param usage_metadata: token usage metadata
    :ptype usage_metadata: dict[str, int] | None
    :param response_metadata: response metadata from provider
    :ptype response_metadata: dict[str, str] | None
    :return: mock AIMessage object
    :rtype: MagicMock
    """
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls or []
    msg.usage_metadata = usage_metadata
    msg.response_metadata = response_metadata or {}
    return msg


async def _mock_astream(*args: object, **kwargs: object) -> AsyncMock:
    """async generator yielding mock AIMessageChunks for stream testing.

    :param args: positional arguments (ignored)
    :ptype args: object
    :param kwargs: keyword arguments (ignored)
    :ptype kwargs: object
    :return: yields mock chunk objects
    :rtype: AsyncMock
    """
    chunk1 = MagicMock()
    chunk1.content = "hel"
    chunk1.tool_calls = []
    chunk1.response_metadata = {}
    yield chunk1

    chunk2 = MagicMock()
    chunk2.content = "lo"
    chunk2.tool_calls = []
    chunk2.response_metadata = {"stop_reason": "stop"}
    yield chunk2


class TestOpenAIChatProvider:
    """tests for OpenAIChatProvider class."""

    def test_satisfies_chat_provider_protocol(self) -> None:
        """OpenAIChatProvider instance satisfies ChatProvider protocol check."""
        provider = OpenAIChatProvider("gpt-4o", "sk-test-key")
        assert isinstance(provider, ChatProvider)

    def test_base_url_passed_through(self) -> None:
        """base_url is stored as-is without stripping."""
        provider = OpenAIChatProvider(
            "gpt-4o",
            "sk-test-key",
            base_url="https://api.openai.com/v1",
        )
        assert provider._base_url == "https://api.openai.com/v1"

    def test_base_url_none_accepted(self) -> None:
        """base_url=None is stored as None without error."""
        provider = OpenAIChatProvider(
            "gpt-4o",
            "sk-test-key",
            base_url=None,
        )
        assert provider._base_url is None

    async def test_complete_returns_chat_result(self) -> None:
        """complete converts AIMessage response to ChatResult."""
        provider = OpenAIChatProvider("gpt-4o", "sk-test-key")
        mock_response = _mock_ai_message(
            content="hello world",
            response_metadata={"model": "gpt-4o-2024-08-06"},
        )
        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(return_value=mock_response)
        provider._model = mock_model

        messages = [ChatMessage(role=MessageRole.USER, content="hi")]
        result = await provider.complete(messages)

        assert isinstance(result, ChatResult)
        assert result.content == "hello world"
        assert result.model == "gpt-4o-2024-08-06"
        assert result.tool_calls is None

    async def test_complete_with_tool_calls(self) -> None:
        """complete maps AIMessage tool_calls to ToolCallRequest list."""
        provider = OpenAIChatProvider("gpt-4o", "sk-test-key")
        mock_response = _mock_ai_message(
            content="",
            tool_calls=[
                {"id": "call_abc", "name": "get_weather", "args": {"city": "NYC"}},
            ],
            response_metadata={"model": "gpt-4o"},
        )
        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(return_value=mock_response)
        provider._model = mock_model

        messages = [ChatMessage(role=MessageRole.USER, content="weather?")]
        result = await provider.complete(messages)

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_abc"
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].args == {"city": "NYC"}

    async def test_complete_with_usage_metadata(self) -> None:
        """complete extracts usage metadata into usage dict."""
        provider = OpenAIChatProvider("gpt-4o", "sk-test-key")
        mock_response = _mock_ai_message(
            content="response",
            usage_metadata={"input_tokens": 15, "output_tokens": 25},
            response_metadata={"model": "gpt-4o"},
        )
        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(return_value=mock_response)
        provider._model = mock_model

        messages = [ChatMessage(role=MessageRole.USER, content="count tokens")]
        result = await provider.complete(messages)

        assert result.usage is not None
        assert result.usage["input_tokens"] == 15
        assert result.usage["output_tokens"] == 25

    async def test_stream_yields_chat_chunks(self) -> None:
        """stream yields ChatChunk objects from async LangChain stream."""
        provider = OpenAIChatProvider("gpt-4o", "sk-test-key")
        mock_model = MagicMock()
        mock_model.astream = _mock_astream
        provider._model = mock_model

        messages = [ChatMessage(role=MessageRole.USER, content="stream test")]
        chunks: list[ChatChunk] = []
        async for chunk in provider.stream(messages):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].content == "hel"
        assert chunks[0].finish_reason is None
        assert chunks[1].content == "lo"
        assert chunks[1].finish_reason == "stop"

    def test_bind_tools_clears_model_cache(self) -> None:
        """bind_tools stores tools and clears cached model instance."""
        provider = OpenAIChatProvider("gpt-4o", "sk-test-key")
        provider._model = MagicMock()

        tools = [
            ToolDefinition(
                name="get_weather",
                description="gets current weather",
                parameters={"type": "object", "properties": {"city": {"type": "string"}}},
            ),
        ]
        provider.bind_tools(tools)

        assert provider._tools is not None
        assert len(provider._tools) == 1
        assert provider._tools[0].name == "get_weather"
        assert provider._model is None

    def test_preprocess_returns_messages(self) -> None:
        """preprocess delegates to preprocessing pipeline and returns messages."""
        provider = OpenAIChatProvider("gpt-4o", "sk-test-key")
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="system prompt"),
            ChatMessage(role=MessageRole.USER, content="hello"),
        ]
        result = provider.preprocess(messages)

        assert len(result) == 2
        assert result[0].role == MessageRole.SYSTEM
        assert result[1].role == MessageRole.USER
        assert result[0].content == "system prompt"
        assert result[1].content == "hello"


class TestOpenAIEmbeddingProvider:
    """tests for OpenAIEmbeddingProvider class."""

    def test_satisfies_embedding_provider_protocol(self) -> None:
        """OpenAIEmbeddingProvider instance satisfies EmbeddingProvider protocol check."""
        provider = OpenAIEmbeddingProvider("text-embedding-3-small", "sk-test-key")
        assert isinstance(provider, EmbeddingProvider)

    def test_dimensions_property_returns_configured(self) -> None:
        """dimensions property returns configured embedding_dimensions."""
        provider = OpenAIEmbeddingProvider(
            "text-embedding-3-small",
            "sk-test-key",
            embedding_dimensions=1024,
        )
        assert provider.dimensions == 1024

    def test_dimensions_property_default(self) -> None:
        """dimensions property returns default when not configured."""
        provider = OpenAIEmbeddingProvider("text-embedding-ada-002", "sk-test-key")
        assert provider.dimensions == _DEFAULT_EMBEDDING_DIMENSIONS

    async def test_embed_returns_single_result(self) -> None:
        """embed returns single EmbeddingResult from aembed_documents."""
        provider = OpenAIEmbeddingProvider("text-embedding-3-small", "sk-test-key")
        mock_model = MagicMock()
        mock_model.aembed_documents = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
        provider._model = mock_model

        result = await provider.embed("test text")

        assert isinstance(result, EmbeddingResult)
        assert result.vector == [0.1, 0.2, 0.3]
        assert result.dimensions == 3
        assert result.model == "text-embedding-3-small"

    async def test_embed_batch_returns_multiple_results(self) -> None:
        """embed_batch returns list of EmbeddingResult for each input text."""
        provider = OpenAIEmbeddingProvider("text-embedding-3-small", "sk-test-key")
        mock_model = MagicMock()
        mock_model.aembed_documents = AsyncMock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        provider._model = mock_model

        results = await provider.embed_batch(["text one", "text two"])

        assert len(results) == 2
        assert results[0].vector == [0.1, 0.2, 0.3]
        assert results[0].model == "text-embedding-3-small"
        assert results[1].vector == [0.4, 0.5, 0.6]
        assert results[1].dimensions == 3

    async def test_embed_batch_token_count_estimation(self) -> None:
        """embed_batch estimates token_count as len(text) // 4."""
        provider = OpenAIEmbeddingProvider("text-embedding-3-small", "sk-test-key")
        mock_model = MagicMock()
        mock_model.aembed_documents = AsyncMock(return_value=[[0.1, 0.2]])
        provider._model = mock_model

        # 20 characters -> 20 // 4 = 5 estimated tokens
        results = await provider.embed_batch(["twelve chars plus 8x"])

        assert results[0].token_count == 5

    def test_base_url_passed_through(self) -> None:
        """base_url is stored as-is without modification."""
        provider = OpenAIEmbeddingProvider(
            "text-embedding-3-small",
            "sk-test-key",
            base_url="https://custom.api.com/v1",
        )
        assert provider._base_url == "https://custom.api.com/v1"
