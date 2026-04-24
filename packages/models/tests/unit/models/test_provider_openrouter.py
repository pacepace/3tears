"""tests for OpenRouterChatProvider adapter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from threetears.models.messages import ChatMessage, MessageRole, ToolDefinition
from threetears.models.protocol import ChatProvider
from threetears.models.providers.openrouter import OpenRouterChatProvider
from threetears.models.results import ChatChunk, ChatResult


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


async def _mock_astream_none_finish(*args: object, **kwargs: object) -> AsyncMock:
    """async generator yielding chunk with None finish_reason for defensive testing.

    :param args: positional arguments (ignored)
    :ptype args: object
    :param kwargs: keyword arguments (ignored)
    :ptype kwargs: object
    :return: yields mock chunk with no finish reason metadata
    :rtype: AsyncMock
    """
    chunk = MagicMock()
    chunk.content = "partial"
    chunk.tool_calls = []
    chunk.response_metadata = None
    yield chunk


class TestOpenRouterChatProvider:
    """tests for OpenRouterChatProvider class."""

    def test_satisfies_chat_provider_protocol(self) -> None:
        """OpenRouterChatProvider instance satisfies ChatProvider protocol check."""
        provider = OpenRouterChatProvider("deepseek/deepseek-chat-v3-0324", "sk-or-test")
        assert isinstance(provider, ChatProvider)

    def test_timeout_stored_in_seconds(self) -> None:
        """timeout parameter stored in seconds for internal conversion."""
        provider = OpenRouterChatProvider("deepseek/deepseek-chat-v3-0324", "sk-or-test", timeout=90)
        assert provider._timeout == 90

    def test_timeout_default_value(self) -> None:
        """timeout defaults to 120 seconds when not specified."""
        provider = OpenRouterChatProvider("deepseek/deepseek-chat-v3-0324", "sk-or-test")
        assert provider._timeout == 120

    async def test_complete_returns_chat_result(self) -> None:
        """complete converts AIMessage response to ChatResult."""
        provider = OpenRouterChatProvider("deepseek/deepseek-chat-v3-0324", "sk-or-test")
        mock_response = _mock_ai_message(
            content="hello world",
            response_metadata={"model": "deepseek/deepseek-chat-v3-0324"},
        )
        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(return_value=mock_response)
        provider.model = mock_model

        messages = [ChatMessage(role=MessageRole.USER, content="hi")]
        result = await provider.complete(messages)

        assert isinstance(result, ChatResult)
        assert result.content == "hello world"
        assert result.model == "deepseek/deepseek-chat-v3-0324"
        assert result.tool_calls is None

    async def test_complete_with_tool_calls(self) -> None:
        """complete maps AIMessage tool_calls to ToolCallRequest list."""
        provider = OpenRouterChatProvider("deepseek/deepseek-chat-v3-0324", "sk-or-test")
        mock_response = _mock_ai_message(
            content="",
            tool_calls=[
                {"id": "call_abc", "name": "get_weather", "args": {"city": "NYC"}},
            ],
            response_metadata={"model": "deepseek/deepseek-chat-v3-0324"},
        )
        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(return_value=mock_response)
        provider.model = mock_model

        messages = [ChatMessage(role=MessageRole.USER, content="weather?")]
        result = await provider.complete(messages)

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_abc"
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].args == {"city": "NYC"}

    async def test_complete_with_usage_metadata(self) -> None:
        """complete extracts usage metadata into usage dict."""
        provider = OpenRouterChatProvider("deepseek/deepseek-chat-v3-0324", "sk-or-test")
        mock_response = _mock_ai_message(
            content="response",
            usage_metadata={"input_tokens": 15, "output_tokens": 25},
            response_metadata={"model": "deepseek/deepseek-chat-v3-0324"},
        )
        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(return_value=mock_response)
        provider.model = mock_model

        messages = [ChatMessage(role=MessageRole.USER, content="count tokens")]
        result = await provider.complete(messages)

        assert result.usage is not None
        assert result.usage["input_tokens"] == 15
        assert result.usage["output_tokens"] == 25

    async def test_stream_yields_chat_chunks(self) -> None:
        """stream yields ChatChunk objects from async LangChain stream."""
        provider = OpenRouterChatProvider("deepseek/deepseek-chat-v3-0324", "sk-or-test")
        mock_model = MagicMock()
        mock_model.astream = _mock_astream
        provider.model = mock_model

        messages = [ChatMessage(role=MessageRole.USER, content="stream test")]
        chunks: list[ChatChunk] = []
        async for chunk in provider.stream(messages):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].content == "hel"
        assert chunks[0].finish_reason is None
        assert chunks[1].content == "lo"
        assert chunks[1].finish_reason == "stop"

    async def test_stream_handles_none_finish_reason(self) -> None:
        """stream handles chunk with None finish_reason without error."""
        provider = OpenRouterChatProvider("deepseek/deepseek-chat-v3-0324", "sk-or-test")
        mock_model = MagicMock()
        mock_model.astream = _mock_astream_none_finish
        provider.model = mock_model

        messages = [ChatMessage(role=MessageRole.USER, content="partial stream")]
        chunks: list[ChatChunk] = []
        async for chunk in provider.stream(messages):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].content == "partial"
        assert chunks[0].finish_reason is None

    def test_bind_tools_clears_model_cache(self) -> None:
        """bind_tools stores tools and clears cached model instance."""
        provider = OpenRouterChatProvider("deepseek/deepseek-chat-v3-0324", "sk-or-test")
        provider.model = MagicMock()

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
        assert provider.model is None

    def test_preprocess_returns_messages(self) -> None:
        """preprocess delegates to preprocessing pipeline and returns messages."""
        provider = OpenRouterChatProvider("deepseek/deepseek-chat-v3-0324", "sk-or-test")
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

    async def test_valueerror_propagates_from_complete(self) -> None:
        """ValueError from OpenRouter SDK propagates unmodified."""
        provider = OpenRouterChatProvider("deepseek/deepseek-chat-v3-0324", "sk-or-test")
        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(
            side_effect=ValueError("OpenRouter API returned an error: 429"),
        )
        provider.model = mock_model

        with pytest.raises(ValueError, match="OpenRouter API"):
            await provider.complete([ChatMessage(role=MessageRole.USER, content="test")])
