"""tests for AnthropicChatProvider adapter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from threetears.models.messages import ChatMessage, MessageRole, ToolCallRequest, ToolDefinition
from threetears.models.protocol import ChatProvider
from threetears.models.providers.anthropic import (
    AnthropicChatProvider,
    _ai_chunk_to_chat_chunk,
    _ai_message_to_result,
    _messages_to_lc,
    _strip_v1_suffix,
    _tool_def_to_lc,
)
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
    chunk2.response_metadata = {"stop_reason": "end_turn"}
    yield chunk2


class TestAnthropicChatProvider:
    """tests for AnthropicChatProvider class."""

    def test_satisfies_chat_provider_protocol(self) -> None:
        """AnthropicChatProvider instance satisfies ChatProvider protocol check."""
        provider = AnthropicChatProvider("claude-sonnet-4-20250514", "sk-test-key")
        assert isinstance(provider, ChatProvider)

    def test_base_url_v1_stripped(self) -> None:
        """base_url ending with /v1 has suffix stripped."""
        provider = AnthropicChatProvider(
            "claude-sonnet-4-20250514",
            "sk-test-key",
            base_url="https://api.anthropic.com/v1",
        )
        assert provider._base_url == "https://api.anthropic.com"

    def test_base_url_v1_slash_stripped(self) -> None:
        """base_url ending with /v1/ has suffix stripped."""
        provider = AnthropicChatProvider(
            "claude-sonnet-4-20250514",
            "sk-test-key",
            base_url="https://api.anthropic.com/v1/",
        )
        assert provider._base_url == "https://api.anthropic.com"

    def test_base_url_without_v1_unchanged(self) -> None:
        """base_url without /v1 suffix is preserved unchanged."""
        provider = AnthropicChatProvider(
            "claude-sonnet-4-20250514",
            "sk-test-key",
            base_url="https://custom.api.com",
        )
        assert provider._base_url == "https://custom.api.com"

    def test_base_url_none_accepted(self) -> None:
        """base_url=None is stored as None without error."""
        provider = AnthropicChatProvider(
            "claude-sonnet-4-20250514",
            "sk-test-key",
            base_url=None,
        )
        assert provider._base_url is None

    async def test_complete_returns_chat_result(self) -> None:
        """complete converts AIMessage response to ChatResult."""
        provider = AnthropicChatProvider("claude-sonnet-4-20250514", "sk-test-key")
        mock_response = _mock_ai_message(
            content="hello world",
            response_metadata={"model": "claude-sonnet-4-20250514"},
        )
        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(return_value=mock_response)
        provider.model = mock_model

        messages = [ChatMessage(role=MessageRole.USER, content="hi")]
        result = await provider.complete(messages)

        assert isinstance(result, ChatResult)
        assert result.content == "hello world"
        assert result.model == "claude-sonnet-4-20250514"
        assert result.tool_calls is None

    async def test_complete_with_tool_calls(self) -> None:
        """complete maps AIMessage tool_calls to ToolCallRequest list."""
        provider = AnthropicChatProvider("claude-sonnet-4-20250514", "sk-test-key")
        mock_response = _mock_ai_message(
            content="",
            tool_calls=[
                {"id": "call_123", "name": "get_weather", "args": {"city": "NYC"}},
            ],
            response_metadata={"model": "claude-sonnet-4-20250514"},
        )
        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(return_value=mock_response)
        provider.model = mock_model

        messages = [ChatMessage(role=MessageRole.USER, content="weather?")]
        result = await provider.complete(messages)

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_123"
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].args == {"city": "NYC"}

    async def test_complete_with_usage_metadata(self) -> None:
        """complete extracts usage metadata into usage dict."""
        provider = AnthropicChatProvider("claude-sonnet-4-20250514", "sk-test-key")
        mock_response = _mock_ai_message(
            content="response",
            usage_metadata={"input_tokens": 10, "output_tokens": 20},
            response_metadata={"model": "claude-sonnet-4-20250514"},
        )
        mock_model = MagicMock()
        mock_model.ainvoke = AsyncMock(return_value=mock_response)
        provider.model = mock_model

        messages = [ChatMessage(role=MessageRole.USER, content="count tokens")]
        result = await provider.complete(messages)

        assert result.usage is not None
        assert result.usage["input_tokens"] == 10
        assert result.usage["output_tokens"] == 20

    async def test_stream_yields_chat_chunks(self) -> None:
        """stream yields ChatChunk objects from async LangChain stream."""
        provider = AnthropicChatProvider("claude-sonnet-4-20250514", "sk-test-key")
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
        assert chunks[1].finish_reason == "end_turn"

    def test_bind_tools_clears_model_cache(self) -> None:
        """bind_tools stores tools and clears cached model instance."""
        provider = AnthropicChatProvider("claude-sonnet-4-20250514", "sk-test-key")
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
        provider = AnthropicChatProvider("claude-sonnet-4-20250514", "sk-test-key")
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


class TestStripV1Suffix:
    """tests for _strip_v1_suffix helper."""

    def test_strips_v1(self) -> None:
        """strips /v1 suffix from URL."""
        assert _strip_v1_suffix("https://api.anthropic.com/v1") == "https://api.anthropic.com"

    def test_strips_v1_slash(self) -> None:
        """strips /v1/ suffix from URL."""
        assert _strip_v1_suffix("https://api.anthropic.com/v1/") == "https://api.anthropic.com"

    def test_no_v1_unchanged(self) -> None:
        """URL without /v1 suffix is returned unchanged."""
        assert _strip_v1_suffix("https://custom.api.com") == "https://custom.api.com"

    def test_v1_in_middle_unchanged(self) -> None:
        """URL with /v1 in middle path is returned unchanged."""
        assert _strip_v1_suffix("https://api.com/v1/extra") == "https://api.com/v1/extra"


class TestMessagesToLc:
    """tests for _messages_to_lc conversion helper."""

    def test_converts_system_message(self) -> None:
        """system message converts to LangChain SystemMessage."""
        from langchain_core.messages import SystemMessage

        messages = [ChatMessage(role=MessageRole.SYSTEM, content="be helpful")]
        result = _messages_to_lc(messages)

        assert len(result) == 1
        assert isinstance(result[0], SystemMessage)
        assert result[0].content == "be helpful"

    def test_converts_user_message(self) -> None:
        """user message converts to LangChain HumanMessage."""
        from langchain_core.messages import HumanMessage

        messages = [ChatMessage(role=MessageRole.USER, content="hello")]
        result = _messages_to_lc(messages)

        assert len(result) == 1
        assert isinstance(result[0], HumanMessage)
        assert result[0].content == "hello"

    def test_converts_assistant_message(self) -> None:
        """assistant message converts to LangChain AIMessage."""
        from langchain_core.messages import AIMessage

        messages = [ChatMessage(role=MessageRole.ASSISTANT, content="hi there")]
        result = _messages_to_lc(messages)

        assert len(result) == 1
        assert isinstance(result[0], AIMessage)
        assert result[0].content == "hi there"

    def test_converts_assistant_with_tool_calls(self) -> None:
        """assistant message with tool calls includes LangChain tool_calls format."""
        from langchain_core.messages import AIMessage

        messages = [
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content="",
                tool_calls=[ToolCallRequest(id="tc_1", name="search", args={"q": "test"})],
            ),
        ]
        result = _messages_to_lc(messages)

        assert len(result) == 1
        assert isinstance(result[0], AIMessage)
        assert len(result[0].tool_calls) == 1
        assert result[0].tool_calls[0]["name"] == "search"
        assert result[0].tool_calls[0]["id"] == "tc_1"

    def test_converts_tool_message(self) -> None:
        """tool message converts to LangChain ToolMessage."""
        from langchain_core.messages import ToolMessage

        messages = [
            ChatMessage(
                role=MessageRole.TOOL,
                content="result data",
                tool_call_id="tc_1",
            ),
        ]
        result = _messages_to_lc(messages)

        assert len(result) == 1
        assert isinstance(result[0], ToolMessage)
        assert result[0].content == "result data"
        assert result[0].tool_call_id == "tc_1"

    def test_tool_message_none_tool_call_id_uses_empty_string(self) -> None:
        """tool message with None tool_call_id uses empty string fallback."""
        from langchain_core.messages import ToolMessage

        messages = [
            ChatMessage(role=MessageRole.TOOL, content="result", tool_call_id=None),
        ]
        result = _messages_to_lc(messages)

        assert isinstance(result[0], ToolMessage)
        assert result[0].tool_call_id == ""

    def test_converts_mixed_conversation(self) -> None:
        """mixed message roles convert to correct LangChain types."""
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="system"),
            ChatMessage(role=MessageRole.USER, content="user"),
            ChatMessage(role=MessageRole.ASSISTANT, content="assistant"),
        ]
        result = _messages_to_lc(messages)

        assert len(result) == 3
        assert isinstance(result[0], SystemMessage)
        assert isinstance(result[1], HumanMessage)
        assert isinstance(result[2], AIMessage)


class TestAiMessageToResult:
    """tests for _ai_message_to_result conversion helper."""

    def test_basic_content(self) -> None:
        """extracts string content from AIMessage."""
        msg = _mock_ai_message(content="hello world")
        result = _ai_message_to_result(msg)

        assert result.content == "hello world"
        assert result.tool_calls is None
        assert result.usage is None

    def test_non_string_content_converted(self) -> None:
        """non-string content is converted to string."""
        msg = _mock_ai_message()
        msg.content = ["block1", "block2"]
        result = _ai_message_to_result(msg)

        assert result.content == "['block1', 'block2']"

    def test_with_tool_calls(self) -> None:
        """extracts tool calls into ToolCallRequest list."""
        msg = _mock_ai_message(
            content="",
            tool_calls=[
                {"id": "tc_1", "name": "search", "args": {"query": "test"}},
                {"id": "tc_2", "name": "calc", "args": {"expr": "1+1"}},
            ],
        )
        result = _ai_message_to_result(msg)

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].name == "search"
        assert result.tool_calls[1].name == "calc"

    def test_with_usage_metadata(self) -> None:
        """extracts usage metadata into dict."""
        msg = _mock_ai_message(
            usage_metadata={"input_tokens": 50, "output_tokens": 100},
        )
        result = _ai_message_to_result(msg)

        assert result.usage is not None
        assert result.usage["input_tokens"] == 50
        assert result.usage["output_tokens"] == 100

    def test_with_response_metadata_model(self) -> None:
        """extracts model name from response_metadata."""
        msg = _mock_ai_message(
            response_metadata={"model": "claude-sonnet-4-20250514"},
        )
        result = _ai_message_to_result(msg)

        assert result.model == "claude-sonnet-4-20250514"

    def test_no_response_metadata(self) -> None:
        """handles None response_metadata gracefully."""
        msg = _mock_ai_message()
        msg.response_metadata = None
        result = _ai_message_to_result(msg)

        assert result.model == ""


class TestAiChunkToChatChunk:
    """tests for _ai_chunk_to_chat_chunk conversion helper."""

    def test_basic_chunk(self) -> None:
        """converts chunk with string content."""
        chunk = MagicMock()
        chunk.content = "partial"
        chunk.tool_calls = []
        chunk.response_metadata = {}

        result = _ai_chunk_to_chat_chunk(chunk)

        assert result.content == "partial"
        assert result.tool_calls is None
        assert result.finish_reason is None

    def test_chunk_with_finish_reason(self) -> None:
        """extracts stop_reason from response_metadata as finish_reason."""
        chunk = MagicMock()
        chunk.content = ""
        chunk.tool_calls = []
        chunk.response_metadata = {"stop_reason": "end_turn"}

        result = _ai_chunk_to_chat_chunk(chunk)

        assert result.finish_reason == "end_turn"

    def test_chunk_with_tool_calls(self) -> None:
        """converts chunk tool_calls to ToolCallRequest list."""
        chunk = MagicMock()
        chunk.content = ""
        chunk.tool_calls = [
            {"id": "tc_1", "name": "search", "args": {"q": "test"}},
        ]
        chunk.response_metadata = {}

        result = _ai_chunk_to_chat_chunk(chunk)

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"

    def test_non_string_content_becomes_empty(self) -> None:
        """non-string chunk content converts to empty string."""
        chunk = MagicMock()
        chunk.content = [{"type": "text", "text": "hi"}]
        chunk.tool_calls = []
        chunk.response_metadata = {}

        result = _ai_chunk_to_chat_chunk(chunk)

        assert result.content == ""


class TestToolDefToLc:
    """tests for _tool_def_to_lc conversion helper."""

    def test_converts_tool_definition(self) -> None:
        """converts ToolDefinition to LangChain tool dict format."""
        tool = ToolDefinition(
            name="get_weather",
            description="gets current weather for city",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        )
        result = _tool_def_to_lc(tool)

        assert result["name"] == "get_weather"
        assert result["description"] == "gets current weather for city"
        assert result["input_schema"] == tool.parameters

    def test_empty_parameters(self) -> None:
        """converts ToolDefinition with empty parameters."""
        tool = ToolDefinition(name="ping", description="health check")
        result = _tool_def_to_lc(tool)

        assert result["name"] == "ping"
        assert result["input_schema"] == {}
