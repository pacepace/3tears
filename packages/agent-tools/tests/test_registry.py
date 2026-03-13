"""Tests for ToolRegistry."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.tools import BaseTool, StructuredTool

from threetears.agent.tools.registry import ToolRegistry


def _mock_factory(config: dict[str, Any], description: str) -> BaseTool:
    """A trivial factory that returns a mock tool."""
    return StructuredTool.from_function(
        func=lambda: "ok",
        name="mock_tool",
        description=description,
    )


def _error_factory(config: dict[str, Any], description: str) -> BaseTool:
    """A factory that always raises."""
    raise RuntimeError("factory boom")


def test_register_and_create():
    reg = ToolRegistry()
    reg.register("mock", _mock_factory)
    tool = reg.create("mock", {}, "A mock tool")
    assert tool.name == "mock_tool"


def test_create_unknown_raises():
    reg = ToolRegistry()
    with pytest.raises(KeyError, match="Unknown tool type: nope"):
        reg.create("nope", {}, "desc")


def test_list_types():
    reg = ToolRegistry()
    reg.register("alpha", _mock_factory)
    reg.register("beta", _mock_factory)
    assert reg.list_types() == ["alpha", "beta"]


def test_unregister():
    reg = ToolRegistry()
    reg.register("temp", _mock_factory)
    assert reg.has("temp")
    reg.unregister("temp")
    assert not reg.has("temp")


def test_has():
    reg = ToolRegistry()
    assert not reg.has("x")
    reg.register("x", _mock_factory)
    assert reg.has("x")


async def test_create_tools_skips_unknown():
    reg = ToolRegistry()
    reg.register("known", _mock_factory)
    tools = await reg.create_tools(
        [
            {"tool_type": "known", "description": "good"},
            {"tool_type": "unknown", "description": "bad"},
        ]
    )
    assert len(tools) == 1
    assert tools[0].name == "mock_tool"


async def test_create_tools_skips_on_error():
    reg = ToolRegistry()
    reg.register("broken", _error_factory)
    reg.register("good", _mock_factory)
    tools = await reg.create_tools(
        [
            {"tool_type": "broken", "description": "breaks"},
            {"tool_type": "good", "description": "works"},
        ]
    )
    assert len(tools) == 1
    assert tools[0].name == "mock_tool"
