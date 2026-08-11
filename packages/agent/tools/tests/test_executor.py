"""Tests for ToolExecutor."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import ToolMessage

from threetears.agent.tools.executor import ToolExecutor


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
    tc_response = _make_tool_call_response(
        [
            {"name": "calculator", "args": {"expr": "6*7"}, "id": "tc_1"},
        ]
    )
    text_response = _make_text_response("The answer is 42.")

    model = AsyncMock()
    model.ainvoke = AsyncMock(side_effect=[tc_response, text_response])

    executor = ToolExecutor(max_rounds=3)
    result = await executor.invoke_with_tools(model, [], [tool])

    assert result.output == "The answer is 42."
    assert result.rounds_used == 2
    assert len(result.tool_calls_made) == 1
    assert result.tool_calls_made[0]["name"] == "calculator"
    # The whole tool call, not just its args: that is what makes LangChain
    # build the ToolMessage and keep a content_and_artifact tool's artifact
    # (§4.7). This assertion previously pinned the args-only shape, which is
    # the defect -- a test can encode a bug as firmly as the code does.
    tool.ainvoke.assert_awaited_once_with(
        {"name": "calculator", "args": {"expr": "6*7"}, "id": "tc_1", "type": "tool_call"}
    )


async def test_a_tool_message_answer_is_kept_whole_with_its_artifact():
    """§4.7: the structured artifact survives the executor.

    ``page_finder`` runs through this executor and returns its structure as
    an artifact. Stringifying the invocation result dropped it, so the
    structure never reached the caller -- check 4 fails without this.
    """
    artifact = {"pages": [{"url": "https://example.org/a", "score": 0.9}]}
    tool = _make_mock_tool("page_finder")
    tool.ainvoke = AsyncMock(return_value=ToolMessage(content="found 1 page", tool_call_id="tc_1", artifact=artifact))
    tc_response = _make_tool_call_response([{"name": "page_finder", "args": {"q": "a"}, "id": "tc_1"}])

    model = AsyncMock()
    model.ainvoke = AsyncMock(side_effect=[tc_response, _make_text_response("done")])

    executor = ToolExecutor(max_rounds=3)
    await executor.invoke_with_tools(model, [], [tool])

    appended = model.ainvoke.await_args_list[1].args[0]
    tool_messages = [m for m in appended if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].artifact == artifact
    assert tool_messages[0].content == "found 1 page"


async def test_a_plain_value_answer_still_becomes_content():
    """A tool that is not artifact-aware behaves exactly as it did before."""
    tool = _make_mock_tool("calculator", "42")
    tc_response = _make_tool_call_response([{"name": "calculator", "args": {"expr": "6*7"}, "id": "tc_1"}])

    model = AsyncMock()
    model.ainvoke = AsyncMock(side_effect=[tc_response, _make_text_response("done")])

    executor = ToolExecutor(max_rounds=3)
    await executor.invoke_with_tools(model, [], [tool])

    appended = model.ainvoke.await_args_list[1].args[0]
    tool_messages = [m for m in appended if isinstance(m, ToolMessage)]
    assert tool_messages[0].content == "42"
    assert tool_messages[0].artifact is None


async def test_a_raising_tool_still_answers_with_error_text():
    """The failure path keeps its shape: one ToolMessage, naming the tool."""
    tool = _make_mock_tool("calculator")
    tool.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))
    tc_response = _make_tool_call_response([{"name": "calculator", "args": {}, "id": "tc_1"}])

    model = AsyncMock()
    model.ainvoke = AsyncMock(side_effect=[tc_response, _make_text_response("done")])

    executor = ToolExecutor(max_rounds=3)
    await executor.invoke_with_tools(model, [], [tool])

    appended = model.ainvoke.await_args_list[1].args[0]
    tool_messages = [m for m in appended if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert "Error executing calculator" in str(tool_messages[0].content)
    assert "boom" in str(tool_messages[0].content)


async def test_max_rounds_exhausted():
    tool = _make_mock_tool("search", "result")
    tc_response = _make_tool_call_response(
        [
            {"name": "search", "args": {"q": "test"}, "id": "tc_1"},
        ]
    )
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
    tc_response = _make_tool_call_response(
        [
            {"name": "nonexistent", "args": {}, "id": "tc_1"},
        ]
    )
    text_response = _make_text_response("Handled.")

    model = AsyncMock()
    model.ainvoke = AsyncMock(side_effect=[tc_response, text_response])

    executor = ToolExecutor(max_rounds=3)
    result = await executor.invoke_with_tools(model, [], [])

    assert result.rounds_used == 2
    assert result.tool_calls_made[0]["name"] == "nonexistent"


async def test_execution_result_fields():
    tool = _make_mock_tool("t1", "r1")
    tc_response = _make_tool_call_response(
        [
            {"name": "t1", "args": {"a": 1}, "id": "tc_1"},
            {"name": "t1", "args": {"a": 2}, "id": "tc_2"},
        ]
    )
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
