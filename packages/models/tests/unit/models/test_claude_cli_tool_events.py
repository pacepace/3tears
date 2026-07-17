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


class _DottedNameTool(BaseTool):
    """A :class:`BaseTool` with a canonical dotted 3tears name, like every real builtin
    (``threetears.web_search``, ``threetears.calculator``, ...) -- see
    :meth:`threetears.agent.tools.base_tool.BaseAgentTool.mcp_name`."""

    name: str = "threetears.web_search"
    description: str = "search the web"

    def _run(self, **kwargs: Any) -> str:
        return f"result:{kwargs}"

    async def _arun(self, **kwargs: Any) -> str:
        return f"result:{kwargs}"


def _wrap(tool: BaseTool):
    """Build the subscription model and wrap ``tool``, returning the raw ``SdkMcpTool``."""
    model = create_subscription_chat(DEFAULT_CHAT_MODEL, "sk-ant-oat01-faketokenfortest")
    schema = {"properties": {"x": {"type": "string"}}, "required": []}
    return model._wrap_langchain_tool(tool, schema)  # noqa: SLF001 -- the method under test  # type: ignore[attr-defined]


def _wrapped_handler(tool: BaseTool):
    """Build the subscription model, wrap ``tool``, and return the SDK tool's raw async handler."""
    return _wrap(tool).handler


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

    async def test_a_broken_event_bus_is_logged_not_silently_swallowed(self) -> None:
        """A dispatch failure is exactly the signal that verifying ambient-config
        propagation across the SDK subprocess boundary needs -- it must be logged,
        not disappear into a bare `except: pass`."""
        with (
            patch(
                "threetears.models.providers._claude_cli.dispatch_event",
                AsyncMock(side_effect=RuntimeError("event bus is down")),
            ),
            patch("threetears.models.providers._claude_cli._logger") as mock_logger,
        ):
            handler = _wrapped_handler(_EchoTool())
            await handler({"x": "hi"})

        assert mock_logger.warning.call_count == 3  # dispatched + started + completed all fail
        first_call = mock_logger.warning.call_args_list[0]
        assert "event bus is down" in first_call.kwargs["extra"]["extra_data"]["error"]


class TestDottedToolNamePermissionFix:
    """Every 3tears builtin's canonical name is dotted (``threetears.web_search``, per
    ``BaseAgentTool.mcp_name()``). Live-observed bug: every real call to a bound dotted tool was
    silently DENIED under a subscription turn ("Claude requested permissions to use ... but you
    haven't granted it yet"), with nothing logged anywhere, while the same tool worked fine on
    every other backend. Root cause: the base class's own ``bind_tools`` already tries to
    auto-approve bound tools by deriving ``allowed_tools`` from each tool's raw (dotted) ``.name``,
    but the SDK/CLI normalizes dots out of tool identities on the wire, so that entry never matched
    -- the auto-approval attempt silently failed to match its own target. ``bind_tools`` here
    substitutes each dotted tool for a ``NameMangledToolProxy`` (the same translation
    ``anthropic.py``/``openrouter.py`` already apply for the identical Anthropic tool-name
    constraint) before the base class ever derives ``allowed_tools``, so the entry matches.
    """

    def test_allowed_tools_matches_the_underscored_wire_identity(self) -> None:
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, "sk-ant-oat01-faketokenfortest")
        bound = model.bind_tools([_DottedNameTool()])
        assert bound.bound.allowed_tools == ["mcp__langchain-tools__threetears_web_search"]  # type: ignore[attr-defined]

    def test_dotless_tool_name_is_unaffected(self) -> None:
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, "sk-ant-oat01-faketokenfortest")
        bound = model.bind_tools([_EchoTool()])
        assert bound.bound.allowed_tools == ["mcp__langchain-tools__echo"]  # type: ignore[attr-defined]

    def test_permission_mode_is_left_at_the_default(self) -> None:
        """The fix is a precise allowed_tools match, not a blanket permission bypass."""
        model = create_subscription_chat(DEFAULT_CHAT_MODEL, "sk-ant-oat01-faketokenfortest")
        bound = model.bind_tools([_DottedNameTool()])
        assert bound.bound.permission_mode == "default"  # type: ignore[attr-defined]

    async def test_dotted_tool_still_dispatches_and_executes_correctly(self) -> None:
        """The SDK-registered/allowed_tools name is mangled; execution and event-tracking must not
        be -- exercised on the exact wire-form proxy the real ``bind_tools`` call substitutes
        (:func:`build_name_translation`), matching what the base class's ``bind_tools`` internally
        wraps via ``_wrap_langchain_tool``.
        """
        from threetears.models.tool_name_translation import build_name_translation

        [wire_tool], _reverse_map = build_name_translation([_DottedNameTool()])
        schema = {"properties": {"x": {"type": "string"}}, "required": []}

        with patch(
            "threetears.models.providers._claude_cli.dispatch_event",
            AsyncMock(),
        ) as mock_dispatch:
            model = create_subscription_chat(DEFAULT_CHAT_MODEL, "sk-ant-oat01-faketokenfortest")
            sdk_tool = model._wrap_langchain_tool(wire_tool, schema)  # noqa: SLF001 -- the method under test  # type: ignore[attr-defined]
            assert sdk_tool.name == "threetears_web_search"
            result = await sdk_tool.handler({"x": "hi"})

        assert result["content"][0]["text"] == "result:{'x': 'hi'}"
        dispatched, started, completed = (c.args[0] for c in mock_dispatch.await_args_list)
        assert dispatched.tool_name == "threetears.web_search"
        assert started.tool_name == "threetears.web_search"
        assert completed.tool_name == "threetears.web_search"
