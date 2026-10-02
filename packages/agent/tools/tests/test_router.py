"""Tests for tool routing: recall-intent detection and ToolRouter."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock


from threetears.agent.tools.router import (
    ToolRouter,
    is_recall_intent,
)


# ---------------------------------------------------------------------------
# is_recall_intent
# ---------------------------------------------------------------------------


def test_is_recall_intent_true():
    assert is_recall_intent("show me the results") is True


def test_is_recall_intent_with_verbs():
    assert is_recall_intent("recall the output from opus") is True


def test_is_recall_intent_false_for_new_task():
    assert is_recall_intent("search for python tutorials") is False


def test_is_recall_intent_false_for_general():
    assert is_recall_intent("hello how are you") is False


def test_is_recall_intent_new_task_overrides():
    assert is_recall_intent("create a new search query and show results") is False


# ---------------------------------------------------------------------------
# Mock factory
# ---------------------------------------------------------------------------


class MockChatModelFactory:
    """Chat model factory that returns a mock producing a fixed response."""

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text

    async def create_chat_model(self, purpose: str = "routing") -> Any:
        model = AsyncMock()
        response = MagicMock()
        response.content = self._response_text
        model.ainvoke = AsyncMock(return_value=response)
        return model


# ---------------------------------------------------------------------------
# reading the routing model's answer
# ---------------------------------------------------------------------------

_SEARCH_TOOL = [{"name": "web_search", "description": "Search the web", "tool_llm_id": "t1"}]
_CALCULATOR_TOOL = [{"name": "calculator", "description": "Do math", "tool_llm_id": "t2"}]


async def test_a_plain_json_answer_is_read():
    router = ToolRouter(
        chat_model_factory=MockChatModelFactory('{"tool_name": "web_search", "reasoning": "needs search"}')
    )
    decision = await router.route("what is the weather in tokyo", available_tool_llms=_SEARCH_TOOL)
    assert decision.tool_name == "web_search"
    assert decision.reasoning == "needs search"


async def test_a_markdown_wrapped_answer_is_read():
    text = '```json\n{"tool_name": "calculator", "reasoning": "math"}\n```'
    router = ToolRouter(chat_model_factory=MockChatModelFactory(text))
    decision = await router.route("what is 2 plus 2", available_tool_llms=_CALCULATOR_TOOL)
    assert decision.tool_name == "calculator"
    assert decision.reasoning == "math"


async def test_an_unreadable_answer_routes_to_no_tool():
    router = ToolRouter(chat_model_factory=MockChatModelFactory("this is not json at all"))
    decision = await router.route("what is the weather in tokyo", available_tool_llms=_SEARCH_TOOL)
    assert decision.tool_name is None
    assert decision.tool_type is None
    assert decision.reasoning == "parse error"


# ---------------------------------------------------------------------------
# ToolRouter.route
# ---------------------------------------------------------------------------


async def test_route_no_tools():
    factory = MockChatModelFactory('{"tool_name": null, "reasoning": "none"}')
    router = ToolRouter(chat_model_factory=factory)
    decision = await router.route("hello")
    assert decision.tool_name is None
    assert decision.reasoning == "no tools available"


async def test_route_recall_intent():
    factory = MockChatModelFactory('{"tool_name": "search", "reasoning": "search"}')
    router = ToolRouter(chat_model_factory=factory)
    decision = await router.route(
        "show me the previous results",
        available_tool_llms=[{"name": "search", "description": "web search", "tool_llm_id": "t1"}],
    )
    assert decision.tool_name is None
    assert "recall" in decision.reasoning


async def test_route_selects_tool_llm():
    factory = MockChatModelFactory('{"tool_name": "web_search", "reasoning": "user wants to search"}')
    router = ToolRouter(chat_model_factory=factory)
    decision = await router.route(
        "what is the weather in tokyo",
        available_tool_llms=[
            {"name": "web_search", "description": "Search the web", "tool_llm_id": "t1"},
        ],
    )
    assert decision.tool_name == "web_search"
    assert decision.tool_type == "tool_llm"
    assert decision.tool_id == "t1"


async def test_route_selects_mcp_tool():
    factory = MockChatModelFactory('{"tool_name": "mcp:code_exec", "reasoning": "run code"}')
    router = ToolRouter(chat_model_factory=factory)
    decision = await router.route(
        "run this python code",
        available_mcp_tools=[
            {"name": "code_exec", "description": "Execute code", "mcp_server_config_id": "m1"},
        ],
    )
    assert decision.tool_name == "code_exec"
    assert decision.tool_type == "mcp"
    assert decision.tool_id == "m1"


async def test_route_no_match():
    factory = MockChatModelFactory('{"tool_name": null, "reasoning": "general conversation"}')
    router = ToolRouter(chat_model_factory=factory)
    decision = await router.route(
        "hello how are you",
        available_tool_llms=[
            {"name": "web_search", "description": "Search the web", "tool_llm_id": "t1"},
        ],
    )
    assert decision.tool_name is None
    assert decision.tool_type is None
