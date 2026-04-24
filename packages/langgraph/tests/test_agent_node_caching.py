"""integration tests for :class:`PromptCachingHook` wired into :func:`agent_node`.

exercises the end-to-end path: stub chat model reports cache_read
counters on the second call; the hook annotates the system prompt
on anthropic-family models; tool-binding is memoized across
invocations when the tool set is stable.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from threetears.langgraph import PromptCachingHook, agent_node
from threetears.langgraph.hooks import _BOUND_MODEL_CACHE


class _StubPydanticArgsSchema:
    """minimal ``args_schema`` exposing ``model_json_schema``."""

    def __init__(self, schema: dict[str, Any]) -> None:
        """stash the target schema dict.

        :param schema: schema dict returned from
            :meth:`model_json_schema`
        :ptype schema: dict[str, Any]
        """
        self._schema = schema

    def model_json_schema(self) -> dict[str, Any]:
        """return the stashed schema dict.

        :return: schema dict
        :rtype: dict[str, Any]
        """
        return self._schema


class _StubTool:
    """minimal tool double with ``name`` and ``args_schema``."""

    def __init__(
        self,
        name: str,
        args_schema: _StubPydanticArgsSchema | None = None,
    ) -> None:
        """capture name and schema.

        :param name: tool name
        :ptype name: str
        :param args_schema: optional schema stub
        :ptype args_schema: _StubPydanticArgsSchema | None
        """
        self.name = name
        self.args_schema = args_schema


class _FakeAnthropicChatModel:
    """stub chat model posing as :class:`langchain_anthropic.ChatAnthropic`.

    the class name is aliased to ``ChatAnthropic`` at definition
    time so :func:`detect_capabilities` recognizes it; the `model`
    attribute carries a cache-capable identifier; `bind_tools`
    returns a wrapped handle that records the bound tools and
    forwards ``ainvoke`` to the parent.
    """

    def __init__(self, model: str = "claude-sonnet-4-5") -> None:
        """capture model identifier and per-call state.

        :param model: anthropic model identifier
        :ptype model: str
        """
        self.model = model
        self.bind_tools_calls: int = 0
        self.ainvoke_calls: int = 0
        self.last_messages: list[Any] | None = None
        # usage shape toggles across invocations: first call reports
        # cache_creation only (warming the cache), second reports
        # cache_read > 0 (cache hit).
        self._invocation_index = 0

    def bind_tools(self, tools: list[Any]) -> _FakeAnthropicChatModel:
        """record the bind and return self as the bound model.

        :param tools: tool list being bound
        :ptype tools: list[Any]
        :return: self acting as the bound model
        :rtype: _FakeAnthropicChatModel
        """
        self.bind_tools_calls += 1
        return self

    async def ainvoke(
        self,
        messages: list[Any],
        config: Any = None,
    ) -> AIMessage:
        """return a canned :class:`AIMessage` with cache telemetry.

        :param messages: message list
        :ptype messages: list[Any]
        :param config: runtime config (ignored)
        :ptype config: Any
        :return: :class:`AIMessage` with populated ``usage_metadata``
        :rtype: AIMessage
        """
        self.ainvoke_calls += 1
        self.last_messages = list(messages)
        invocation = self._invocation_index
        self._invocation_index += 1
        if invocation == 0:
            details = {"cache_read": 0, "cache_creation": 1100}
        else:
            details = {"cache_read": 1100, "cache_creation": 0}
        return AIMessage(
            content=f"reply-{invocation}",
            usage_metadata={
                "input_tokens": 1200,
                "output_tokens": 40,
                "total_tokens": 1240,
                "input_token_details": details,
            },
        )


_FakeAnthropicChatModel.__name__ = "ChatAnthropic"


class _FakeOpenAIChatModel:
    """stub chat model posing as :class:`langchain_openai.ChatOpenAI`."""

    def __init__(self, model_name: str = "gpt-4o") -> None:
        """capture identifier and per-call counters.

        :param model_name: openai model identifier
        :ptype model_name: str
        """
        self.model_name = model_name
        self.bind_tools_calls: int = 0
        self.ainvoke_calls: int = 0
        self.last_messages: list[Any] | None = None

    def bind_tools(self, tools: list[Any]) -> _FakeOpenAIChatModel:
        """record the bind and return self as the bound model.

        :param tools: tool list being bound
        :ptype tools: list[Any]
        :return: self acting as the bound model
        :rtype: _FakeOpenAIChatModel
        """
        self.bind_tools_calls += 1
        return self

    async def ainvoke(
        self,
        messages: list[Any],
        config: Any = None,
    ) -> AIMessage:
        """return a canned :class:`AIMessage` with no cache fields.

        :param messages: message list
        :ptype messages: list[Any]
        :param config: runtime config (ignored)
        :ptype config: Any
        :return: :class:`AIMessage` carrying only regular usage
        :rtype: AIMessage
        """
        self.ainvoke_calls += 1
        self.last_messages = list(messages)
        return AIMessage(
            content="openai-reply",
            usage_metadata={
                "input_tokens": 500,
                "output_tokens": 20,
                "total_tokens": 520,
            },
        )


_FakeOpenAIChatModel.__name__ = "ChatOpenAI"


class _FakeUnknownChatModel:
    """stub chat model with an unrecognized class name."""

    def __init__(self) -> None:
        """initialize counters; no model attribute needed.

        :return: nothing
        :rtype: None
        """
        self.bind_tools_calls: int = 0
        self.ainvoke_calls: int = 0
        self.last_messages: list[Any] | None = None

    def bind_tools(self, tools: list[Any]) -> _FakeUnknownChatModel:
        """record the bind and return self.

        :param tools: tool list
        :ptype tools: list[Any]
        :return: self
        :rtype: _FakeUnknownChatModel
        """
        self.bind_tools_calls += 1
        return self

    async def ainvoke(
        self,
        messages: list[Any],
        config: Any = None,
    ) -> AIMessage:
        """return a canned :class:`AIMessage`.

        :param messages: message list
        :ptype messages: list[Any]
        :param config: runtime config (ignored)
        :ptype config: Any
        :return: plain response
        :rtype: AIMessage
        """
        self.ainvoke_calls += 1
        self.last_messages = list(messages)
        return AIMessage(content="unknown-reply")


@pytest.fixture(autouse=True)
def _clear_bound_model_cache() -> Any:
    """reset the module-level bound-model cache between tests.

    the :data:`_BOUND_MODEL_CACHE` is shared across test functions
    because it lives on the module singleton. each test needs a
    clean slate so tool-binding counters reflect only its own
    calls.

    :return: yields nothing
    :rtype: Any
    """
    _BOUND_MODEL_CACHE.clear()
    yield
    _BOUND_MODEL_CACHE.clear()


class TestSystemPromptAnnotationOnCacheCapableModel:
    """cache-capable models see structured-content system messages."""

    @pytest.mark.asyncio
    async def test_system_prompt_rewritten_to_structured_content(self) -> None:
        """anthropic model sees SystemMessage with list content carrying cache_control.

        :raises AssertionError: when the structured shape is absent
        """
        chat = _FakeAnthropicChatModel()
        state: dict[str, Any] = {"messages": [HumanMessage(content="hi")]}
        config: RunnableConfig = {
            "configurable": {
                "chat_model": chat,
                "system_prompt": "You are helpful.",
                "_hooks": {"agent": [PromptCachingHook()]},
            },
        }
        await agent_node(state, config)  # type: ignore[arg-type]
        assert chat.last_messages is not None
        first = chat.last_messages[0]
        assert isinstance(first, SystemMessage)
        assert isinstance(first.content, list)
        assert first.content[0]["cache_control"] == {"type": "ephemeral"}
        assert first.content[0]["text"] == "You are helpful."


class TestSystemPromptLeftAloneOnNonCachingModel:
    """non-caching adapters get bare-string system messages."""

    @pytest.mark.asyncio
    async def test_openai_model_keeps_bare_string_system_prompt(self) -> None:
        """openai model sees SystemMessage with plain string content.

        :raises AssertionError: when structured shape leaks through
        """
        chat = _FakeOpenAIChatModel()
        state: dict[str, Any] = {"messages": [HumanMessage(content="hi")]}
        config: RunnableConfig = {
            "configurable": {
                "chat_model": chat,
                "system_prompt": "You are helpful.",
                "_hooks": {"agent": [PromptCachingHook()]},
            },
        }
        await agent_node(state, config)  # type: ignore[arg-type]
        assert chat.last_messages is not None
        first = chat.last_messages[0]
        assert isinstance(first, SystemMessage)
        assert first.content == "You are helpful."

    @pytest.mark.asyncio
    async def test_unknown_model_keeps_bare_string_system_prompt(self) -> None:
        """unrecognized adapter degrades cleanly; no structured content.

        :raises AssertionError: when degradation is not silent
        """
        chat = _FakeUnknownChatModel()
        state: dict[str, Any] = {"messages": [HumanMessage(content="hi")]}
        config: RunnableConfig = {
            "configurable": {
                "chat_model": chat,
                "system_prompt": "You are helpful.",
                "_hooks": {"agent": [PromptCachingHook()]},
            },
        }
        await agent_node(state, config)  # type: ignore[arg-type]
        assert chat.last_messages is not None
        first = chat.last_messages[0]
        assert isinstance(first, SystemMessage)
        assert first.content == "You are helpful."


class TestCacheUsageTelemetry:
    """after_invoke attaches normalized cache counters to the response."""

    @pytest.mark.asyncio
    async def test_second_call_shows_cache_read_in_usage_metadata(self) -> None:
        """invoking twice in a row surfaces cache_read > 0 on call 2.

        this is the canonical telemetry gate the shard mandates: a
        fake chat model alternating between cache-creation and
        cache-read shapes must thread through ``extract_cache_usage``
        and surface the hit count on the second call.

        :raises AssertionError: when the counter is zero on call 2
        """
        chat = _FakeAnthropicChatModel()
        hook = PromptCachingHook()
        base_messages: list[Any] = [HumanMessage(content="hi")]
        config: RunnableConfig = {
            "configurable": {
                "chat_model": chat,
                "system_prompt": "You are helpful.",
                "_hooks": {"agent": [hook]},
            },
        }
        first = await agent_node({"messages": base_messages}, config)  # type: ignore[arg-type]
        second = await agent_node(
            {"messages": [*base_messages, HumanMessage(content="again")]},
            config,
        )  # type: ignore[arg-type]
        first_response = first["messages"][0]
        second_response = second["messages"][0]
        first_usage = first_response.usage_metadata["cache_usage"]
        second_usage = second_response.usage_metadata["cache_usage"]
        assert first_usage["cache_read_input_tokens"] == 0
        assert first_usage["cache_creation_input_tokens"] == 1100
        assert second_usage["cache_read_input_tokens"] == 1100
        assert second_usage["cache_read_input_tokens"] > 0

    @pytest.mark.asyncio
    async def test_response_without_usage_metadata_still_gets_cache_usage(
        self,
    ) -> None:
        """responses lacking usage_metadata receive a fresh dict.

        :raises AssertionError: when cache_usage key is missing
        """
        chat = _FakeUnknownChatModel()
        state: dict[str, Any] = {"messages": [HumanMessage(content="hi")]}
        config: RunnableConfig = {
            "configurable": {
                "chat_model": chat,
                "system_prompt": "hello",
                "_hooks": {"agent": [PromptCachingHook()]},
            },
        }
        result = await agent_node(state, config)  # type: ignore[arg-type]
        response = result["messages"][0]
        assert "cache_usage" in response.usage_metadata
        assert response.usage_metadata["cache_usage"]["cache_read_input_tokens"] == 0


class TestToolBindingMemoization:
    """tool binding happens once per (model, tool_key) tuple."""

    @pytest.mark.asyncio
    async def test_same_tool_set_triggers_bind_once(self) -> None:
        """back-to-back invocations with identical tools bind once only.

        :raises AssertionError: when the bind count exceeds one
        """
        chat = _FakeAnthropicChatModel()
        tools = [
            _StubTool("alpha", _StubPydanticArgsSchema({"type": "object"})),
            _StubTool("beta", _StubPydanticArgsSchema({"type": "object"})),
        ]
        config: RunnableConfig = {
            "configurable": {
                "chat_model": chat,
                "system_prompt": "sys",
                "tools": tools,
                "_hooks": {"agent": [PromptCachingHook()]},
            },
        }
        state: dict[str, Any] = {"messages": [HumanMessage(content="hi")]}
        await agent_node(state, config)  # type: ignore[arg-type]
        await agent_node(state, config)  # type: ignore[arg-type]
        assert chat.bind_tools_calls == 1
        assert chat.ainvoke_calls == 2

    @pytest.mark.asyncio
    async def test_changing_tool_set_triggers_rebind(self) -> None:
        """adding / removing a tool forces a fresh bind on the next call.

        :raises AssertionError: when the cache does not invalidate
        """
        chat = _FakeAnthropicChatModel()
        tool_a = _StubTool("alpha", _StubPydanticArgsSchema({"type": "object"}))
        tool_b = _StubTool("beta", _StubPydanticArgsSchema({"type": "object"}))
        tool_c = _StubTool("gamma", _StubPydanticArgsSchema({"type": "object"}))
        state: dict[str, Any] = {"messages": [HumanMessage(content="hi")]}

        config_ab: RunnableConfig = {
            "configurable": {
                "chat_model": chat,
                "system_prompt": "sys",
                "tools": [tool_a, tool_b],
                "_hooks": {"agent": [PromptCachingHook()]},
            },
        }
        await agent_node(state, config_ab)  # type: ignore[arg-type]
        assert chat.bind_tools_calls == 1

        config_ac: RunnableConfig = {
            "configurable": {
                "chat_model": chat,
                "system_prompt": "sys",
                "tools": [tool_a, tool_c],
                "_hooks": {"agent": [PromptCachingHook()]},
            },
        }
        await agent_node(state, config_ac)  # type: ignore[arg-type]
        assert chat.bind_tools_calls == 2

    @pytest.mark.asyncio
    async def test_no_tools_skips_bind(self) -> None:
        """empty tool list leaves the chat_model untouched.

        :raises AssertionError: when bind is called on an empty list
        """
        chat = _FakeAnthropicChatModel()
        config: RunnableConfig = {
            "configurable": {
                "chat_model": chat,
                "system_prompt": "sys",
                "_hooks": {"agent": [PromptCachingHook()]},
            },
        }
        state: dict[str, Any] = {"messages": [HumanMessage(content="hi")]}
        await agent_node(state, config)  # type: ignore[arg-type]
        assert chat.bind_tools_calls == 0


class TestSystemPromptAnnotationIsIdempotent:
    """double-installing the hook does not nest cache_control blocks."""

    @pytest.mark.asyncio
    async def test_two_caching_hooks_do_not_double_wrap(self) -> None:
        """running the hook twice in sequence leaves structured content stable.

        :raises AssertionError: when the second run nests the content
        """
        chat = _FakeAnthropicChatModel()
        state: dict[str, Any] = {"messages": [HumanMessage(content="hi")]}
        config: RunnableConfig = {
            "configurable": {
                "chat_model": chat,
                "system_prompt": "sys",
                "_hooks": {
                    "agent": [PromptCachingHook(), PromptCachingHook()],
                },
            },
        }
        await agent_node(state, config)  # type: ignore[arg-type]
        assert chat.last_messages is not None
        first = chat.last_messages[0]
        assert isinstance(first, SystemMessage)
        assert isinstance(first.content, list)
        # one block, not nested; idempotent re-annotation
        assert len(first.content) == 1
        assert first.content[0]["cache_control"] == {"type": "ephemeral"}
