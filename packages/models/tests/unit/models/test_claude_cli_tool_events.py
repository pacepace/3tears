"""Tests for the subscription-backend tool-status event dispatch (claude-max-convergence Chunk 7).

``ClaudeCodeChatModel`` gives no visibility into the Claude Agent SDK subprocess's internal tool
calls through LangChain's own instrumentation -- ``_wrap_langchain_tool``'s ``wrapped`` closure is
the one place with real-time visibility, so it dispatches the SAME typed events node-path tools
already emit. These tests exercise that closure directly (via ``SdkMcpTool.handler``), mocking
only ``dispatch_event`` -- the SDK subprocess itself is not involved.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.tools import BaseTool

pytest.importorskip("langchain_claude_code")
pytest.importorskip("claude_agent_sdk")

from threetears.models import DEFAULT_CHAT_MODEL
from threetears.models.providers._claude_cli import create_subscription_chat


class _EchoTool(BaseTool):
    """A trivial :class:`BaseTool` that returns its args, for tool-event tests."""

    name: str = "echo"
    description: str = "echoes its input"

    def _run(self, **kwargs: Any) -> str:
        return f"echo:{kwargs}"

    async def _arun(self, **kwargs: Any) -> str:
        return f"echo:{kwargs}"


class _FailingTool(BaseTool):
    """A :class:`BaseTool` that always raises, for the failure-path event test."""

    name: str = "boom"
    description: str = "always raises"

    def _run(self, **kwargs: Any) -> str:
        raise RuntimeError("boom")

    async def _arun(self, **kwargs: Any) -> str:
        raise RuntimeError("boom")


def _wrapped_handler(tool: BaseTool):
    """Build the subscription model, wrap ``tool``, and return the SDK tool's raw async handler."""
    model = create_subscription_chat(DEFAULT_CHAT_MODEL, "sk-ant-oat01-faketokenfortest")
    schema = {"properties": {"x": {"type": "string"}}, "required": []}
    sdk_tool = model._wrap_langchain_tool(tool, schema)  # noqa: SLF001 -- the method under test  # type: ignore[attr-defined]
    return sdk_tool.handler


class TestSubscriptionToolEvents:
    async def test_success_dispatches_dispatched_started_completed_in_order(self) -> None:
        with patch(
            "threetears.models.providers._claude_cli.dispatch_event",
            AsyncMock(),
        ) as mock_dispatch:
            handler = _wrapped_handler(_EchoTool())
            result = await handler({"x": "hi"})

        assert result["content"][0]["text"] == "echo:{'x': 'hi'}"
        assert mock_dispatch.await_count == 3
        types = [call.args[0].__class__.__name__ for call in mock_dispatch.await_args_list]
        assert types == ["ToolDispatchedEvent", "ToolStartedEvent", "ToolCompletedEvent"]

        dispatched, started, completed = (c.args[0] for c in mock_dispatch.await_args_list)
        assert dispatched.tool_name == "echo"
        assert started.tool_name == "echo"
        assert started.tool_args == {"x": "hi"}
        assert completed.tool_name == "echo"
        assert completed.tool_status == "completed"
        assert completed.tool_duration_ms >= 0

    async def test_no_config_is_threaded_through_explicitly(self) -> None:
        """`dispatch_event` resolves the ambient RunnableConfig -- this wrapper never holds one."""
        with patch(
            "threetears.models.providers._claude_cli.dispatch_event",
            AsyncMock(),
        ) as mock_dispatch:
            handler = _wrapped_handler(_EchoTool())
            await handler({"x": "hi"})

        for call in mock_dispatch.await_args_list:
            assert call.kwargs.get("config") is None

    async def test_tool_failure_dispatches_a_failed_completed_event(self) -> None:
        with patch(
            "threetears.models.providers._claude_cli.dispatch_event",
            AsyncMock(),
        ) as mock_dispatch:
            handler = _wrapped_handler(_FailingTool())
            result = await handler({})

        assert result["is_error"] is True
        assert "boom" in result["content"][0]["text"]
        completed = mock_dispatch.await_args_list[-1].args[0]
        assert completed.__class__.__name__ == "ToolCompletedEvent"
        assert completed.tool_status == "failed"

    async def test_a_broken_event_bus_never_breaks_the_tool_call(self) -> None:
        """Best-effort dispatch: a dispatch_event failure must not propagate."""
        with patch(
            "threetears.models.providers._claude_cli.dispatch_event",
            AsyncMock(side_effect=RuntimeError("event bus is down")),
        ):
            handler = _wrapped_handler(_EchoTool())
            result = await handler({"x": "hi"})

        assert result["content"][0]["text"] == "echo:{'x': 'hi'}"
        assert "is_error" not in result
