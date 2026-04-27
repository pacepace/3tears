"""integration tests for full chat completion pipeline."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from threetears.models.cache import ModelCache
from threetears.models.capabilities import ModelCapabilities
from threetears.models.circuit_breaker import CircuitBreakerRegistry, CircuitState
from threetears.models.enums import ModelStatus, ModelTier, ModelType
from threetears.models.messages import ChatMessage, MessageRole, ToolDefinition
from threetears.models.preprocessing import preprocess_messages
from threetears.models.providers.anthropic import AnthropicChatProvider
from threetears.models.results import ChatChunk, ChatResult
from threetears.models.streaming import merge_chunks
from threetears.models.tracking import LlmPurpose, UsageRecord, UsageTracker


class TestChatCompletionPipeline:
    """integration tests for full chat completion pipeline."""

    @pytest.mark.asyncio
    async def test_chat_completion_pipeline(self) -> None:
        """full chat pipeline: cache, breaker, complete, track."""
        # 1. create provider with mocked model
        provider = AnthropicChatProvider("claude-sonnet-4-20250514", "sk-test")
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Hello world"
        mock_response.tool_calls = []
        mock_response.usage_metadata = {"input_tokens": 10, "output_tokens": 5}
        mock_response.response_metadata = {"model": "claude-sonnet-4-20250514"}
        mock_model.ainvoke = AsyncMock(return_value=mock_response)
        provider.model = mock_model

        # 2. store in cache
        cache = ModelCache()
        cache.put("anthropic", "claude-sonnet-4-20250514", provider)

        # 3. check circuit breaker
        registry = CircuitBreakerRegistry()
        breaker = registry.get("anthropic")
        breaker.check()
        assert breaker.state == CircuitState.CLOSED

        # 4. retrieve from cache and complete
        cached_provider = cache.get("anthropic", "claude-sonnet-4-20250514")
        assert cached_provider is provider

        messages = [ChatMessage(role=MessageRole.USER, content="Hello")]
        result = await provider.complete(messages)

        # 5. record success
        breaker.record_success()
        assert breaker.state == CircuitState.CLOSED

        # 6. verify result
        assert isinstance(result, ChatResult)
        assert result.content == "Hello world"
        assert result.usage is not None
        assert result.usage["input_tokens"] == 10

        # 7. track usage
        tracker = UsageTracker()
        usage = UsageRecord(
            model_name="claude-sonnet-4-20250514",
            provider_name="anthropic",
            purpose=LlmPurpose.CHAT,
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            latency_ms=200,
        )
        tracker.record(usage)

    @pytest.mark.asyncio
    async def test_chat_with_tool_calls_pipeline(self) -> None:
        """chat pipeline with tool binding and tool call result."""
        provider = AnthropicChatProvider("claude-sonnet-4-20250514", "sk-test")

        # bind tools first, then inject mock (bind_tools clears model cache)
        tools = [
            ToolDefinition(
                name="get_weather",
                description="get current weather for city",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            ),
        ]
        provider.bind_tools(tools)

        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.content = ""
        mock_response.tool_calls = [
            {"id": "tc_001", "name": "get_weather", "args": {"city": "Seattle"}},
        ]
        mock_response.usage_metadata = {"input_tokens": 20, "output_tokens": 12}
        mock_response.response_metadata = {"model": "claude-sonnet-4-20250514"}
        mock_model.ainvoke = AsyncMock(return_value=mock_response)
        provider.model = mock_model

        messages = [
            ChatMessage(role=MessageRole.USER, content="What is the weather in Seattle?"),
        ]
        result = await provider.complete(messages)

        assert isinstance(result, ChatResult)
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].args == {"city": "Seattle"}

    @pytest.mark.asyncio
    async def test_chat_streaming_pipeline(self) -> None:
        """streaming chat pipeline with chunk merging."""
        provider = AnthropicChatProvider("claude-sonnet-4-20250514", "sk-test")

        # build mock chunks as LangChain AIMessageChunk-like objects
        chunk_1 = MagicMock()
        chunk_1.content = "Hello "
        chunk_1.tool_calls = []
        chunk_1.response_metadata = {}

        chunk_2 = MagicMock()
        chunk_2.content = "world"
        chunk_2.tool_calls = []
        chunk_2.response_metadata = {"stop_reason": "end_turn"}

        async def mock_astream(*_args, **_kwargs):  # noqa: ANN002, ANN003
            """yield mock chunks."""
            yield chunk_1
            yield chunk_2

        mock_model = MagicMock()
        mock_model.astream = mock_astream
        provider.model = mock_model

        messages = [ChatMessage(role=MessageRole.USER, content="Hello")]
        chunks: list[ChatChunk] = []
        async for chunk in provider.stream(messages):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].content == "Hello "
        assert chunks[1].content == "world"
        assert chunks[1].finish_reason == "end_turn"

        merged = merge_chunks(chunks)
        assert isinstance(merged, ChatResult)
        assert merged.content == "Hello world"

    @pytest.mark.asyncio
    async def test_chat_preprocessing_pipeline(self) -> None:
        """preprocessing applied before completion merges consecutive roles."""
        # consecutive USER messages should be merged
        messages = [
            ChatMessage(role=MessageRole.USER, content="first message"),
            ChatMessage(role=MessageRole.USER, content="second message"),
        ]

        capabilities = ModelCapabilities(
            model_name="test-model",
            model_type=ModelType.CHAT,
            model_tier=ModelTier.LARGE,
            model_status=ModelStatus.ACTIVE,
            requires_alternating_roles=True,
        )

        processed = preprocess_messages(messages, capabilities)

        # consecutive USER messages are merged into one
        user_messages = [m for m in processed if m.role == MessageRole.USER]
        assert len(user_messages) == 1
        assert "first message" in str(user_messages[0].content)
        assert "second message" in str(user_messages[0].content)
