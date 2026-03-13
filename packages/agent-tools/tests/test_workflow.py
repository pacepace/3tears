"""Tests for workflow tools."""

from __future__ import annotations

from threetears.agent.tools.context import ToolContextManager
from threetears.agent.tools.workflow import load_workflow_tools


def _make_tools():
    ctx = ToolContextManager("conv1", "user1")
    tools = load_workflow_tools(ctx)
    tool_map = {t.name: t for t in tools}
    return ctx, tool_map


def test_set_variable_tool():
    ctx, tools = _make_tools()
    result = tools["set_variable"].invoke({"key": "color", "value": "blue"})
    assert "Variable 'color' saved" in result
    var = ctx.get_variable("color")
    assert var is not None
    assert var["value"] == "blue"


def test_get_variable_tool():
    ctx, tools = _make_tools()
    ctx.set_variable("city", "Paris")
    result = tools["get_variable"].invoke({"key": "city"})
    assert "Paris" in result


def test_get_variable_not_found():
    _ctx, tools = _make_tools()
    result = tools["get_variable"].invoke({"key": "nope"})
    assert "not found" in result


def test_recall_context_tool():
    ctx, tools = _make_tools()
    cid = ctx.save_tool_result("calc", "42")
    result = tools["recall_context"].invoke({"context_id": cid})
    assert "42" in result


def test_recall_context_not_found():
    _ctx, tools = _make_tools()
    result = tools["recall_context"].invoke({"context_id": "bad-id"})
    assert "not found" in result


def test_declare_workflow_tool():
    ctx, tools = _make_tools()
    result = tools["declare_workflow"].invoke(
        {
            "plan": "Test plan",
            "steps": ["step1", "step2"],
        }
    )
    assert "Workflow declared" in result
    assert ctx.has_active_workflow
