"""Tests for ToolExecutor."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from threetears.agent.tools.executor import ToolExecutor, ToolExecutionResult


def _make_text_response(text: str) -> MagicMock:
    """Create a mock LLM response with text content and no tool calls."""
    resp = MagicMock()
    resp.content = text
    resp.tool_calls = []
    return resp


def _make_tool_call_response(calls: list[dict[str, Any]]) -> MagicMock:
    """Create a mock LLM response with tool calls."""
    resp = MagicMock()
    resp.content = ""
    resp.tool_calls = calls
    return resp


def _make_mock_tool(name: str, return_value: str = "tool result") -> MagicMock:
    """Create a mock LangChain tool."""
    tool = MagicMock()
    tool.name = name
    tool.ainvoke = AsyncMock(return_value=return_value)
    return tool


async def test_invoke_text_response():
    model = AsyncMock()
    model.ainvoke = AsyncMock(return_value=_make_text_response("Hello!"))
    executor = ToolExecutor(max_rounds=3)
    result = await executor.invoke_with_tools(model, [], [])
    assert result.output == "Hello!"
    assert result.rounds_used == 1
    assert result.tool_calls_made == []
    assert result.error is None


async def test_invoke_with_tool_calls():
    tool = _make_mock_tool("calculator", "42")
    tc_response = _make_tool_call_response([
        {"name": "calculator", "args": {"expr": "6*7"}, "id": "tc_1"},
    ])
    text_response = _make_text_response("The answer is 42.")

    model = AsyncMock()
    model.ainvoke = AsyncMock(side_effect=[tc_response, text_response])

    executor = ToolExecutor(max_rounds=3)
    result = await executor.invoke_with_tools(model, [], [tool])

    assert result.output == "The answer is 42."
    assert result.rounds_used == 2
    assert len(result.tool_calls_made) == 1
    assert result.tool_calls_made[0]["name"] == "calculator"
    tool.ainvoke.assert_awaited_once_with({"expr": "6*7"})


async def test_max_rounds_exhausted():
    tool = _make_mock_tool("search", "result")
    tc_response = _make_tool_call_response([
        {"name": "search", "args": {"q": "test"}, "id": "tc_1"},
    ])
    final_response = _make_text_response("Done after exhaustion.")

    model = AsyncMock()
    # 2 rounds of tool calls, then the final invocation after exhaustion
    model.ainvoke = AsyncMock(side_effect=[tc_response, tc_response, final_response])

    executor = ToolExecutor(max_rounds=2)
    result = await executor.invoke_with_tools(model, [], [tool])

    assert result.output == "Done after exhaustion."
    assert result.rounds_used == 2
    assert len(result.tool_calls_made) == 2
    assert result.error == "max rounds exhausted"


async def test_tool_not_found():
    tc_response = _make_tool_call_response([
        {"name": "nonexistent", "args": {}, "id": "tc_1"},
    ])
    text_response = _make_text_response("Handled.")

    model = AsyncMock()
    model.ainvoke = AsyncMock(side_effect=[tc_response, text_response])

    executor = ToolExecutor(max_rounds=3)
    result = await executor.invoke_with_tools(model, [], [])

    assert result.rounds_used == 2
    assert result.tool_calls_made[0]["name"] == "nonexistent"


async def test_execution_result_fields():
    tool = _make_mock_tool("t1", "r1")
    tc_response = _make_tool_call_response([
        {"name": "t1", "args": {"a": 1}, "id": "tc_1"},
        {"name": "t1", "args": {"a": 2}, "id": "tc_2"},
    ])
    text_response = _make_text_response("Final.")

    model = AsyncMock()
    model.ainvoke = AsyncMock(side_effect=[tc_response, text_response])

    executor = ToolExecutor(max_rounds=5)
    result = await executor.invoke_with_tools(model, [], [tool])

    assert result.output == "Final."
    assert result.rounds_used == 2
    assert len(result.tool_calls_made) == 2
    assert result.tool_calls_made[0] == {"name": "t1", "args": {"a": 1}}
    assert result.tool_calls_made[1] == {"name": "t1", "args": {"a": 2}}
    assert result.error is None
