"""Tests for ToolContextManager."""

from __future__ import annotations

import pytest

from threetears.agent.tools.context import ToolContextManager


def test_set_get_variable():
    ctx = ToolContextManager("conv1", "user1")
    ctx.set_variable("name", "Alice")
    var = ctx.get_variable("name")
    assert var is not None
    assert var["value"] == "Alice"
    assert var["value_type"] == "string"


def test_set_variable_limit():
    ctx = ToolContextManager("conv1", "user1", var_limit=2)
    ctx.set_variable("a", "1")
    ctx.set_variable("b", "2")
    with pytest.raises(ValueError, match="Variable limit reached"):
        ctx.set_variable("c", "3")


def test_set_variable_truncates():
    ctx = ToolContextManager("conv1", "user1", var_max_chars=10)
    ctx.set_variable("long", "x" * 100)
    var = ctx.get_variable("long")
    assert var is not None
    assert len(var["value"]) == 10


def test_set_variable_upsert():
    """Updating an existing variable should not consume a new slot."""
    ctx = ToolContextManager("conv1", "user1", var_limit=1)
    ctx.set_variable("key", "v1")
    ctx.set_variable("key", "v2")  # Should not raise
    var = ctx.get_variable("key")
    assert var is not None
    assert var["value"] == "v2"


def test_delete_variable():
    ctx = ToolContextManager("conv1", "user1")
    ctx.set_variable("x", "1")
    assert ctx.delete_variable("x") is True
    assert ctx.get_variable("x") is None


def test_delete_nonexistent():
    ctx = ToolContextManager("conv1", "user1")
    assert ctx.delete_variable("nope") is False


def test_get_all_variables():
    ctx = ToolContextManager("conv1", "user1")
    ctx.set_variable("a", "1")
    ctx.set_variable("b", "2")
    all_vars = ctx.get_all_variables()
    assert len(all_vars) == 2
    keys = {v["key"] for v in all_vars}
    assert keys == {"a", "b"}


def test_save_tool_result():
    ctx = ToolContextManager("conv1", "user1")
    cid = ctx.save_tool_result("calc", "42")
    item = ctx.get_context_item(cid)
    assert item is not None
    assert item["value"] == "42"
    assert item["key"] == "calc"


def test_get_context_item_not_found():
    ctx = ToolContextManager("conv1", "user1")
    assert ctx.get_context_item("nonexistent-id") is None


def test_build_conversation_context():
    ctx = ToolContextManager("conv1", "user1")
    ctx.set_variable("name", "Alice")
    ctx.save_tool_result("calc", "42")
    result = ctx.build_conversation_context()
    assert result is not None
    assert "name" in result
    assert "Alice" in result
    assert "42" in result
    assert "[Conversation Variables]" in result
    assert "[Tool Results]" in result


def test_build_conversation_context_empty():
    ctx = ToolContextManager("conv1", "user1")
    assert ctx.build_conversation_context() is None


def test_register_media():
    ctx = ToolContextManager("conv1", "user1")
    ctx.register_media("image_1", url="http://example.com/img.png")
    slots = ctx.get_slots()
    assert "image_1" in slots
    assert slots["image_1"]["url"] == "http://example.com/img.png"


def test_build_media_context():
    ctx = ToolContextManager("conv1", "user1")
    ctx.register_media("image_1", url="http://example.com/img.png")
    result = ctx.build_media_context()
    assert result is not None
    assert "[Active Media Slots]" in result
    assert "image_1" in result


def test_build_media_context_empty():
    ctx = ToolContextManager("conv1", "user1")
    assert ctx.build_media_context() is None


def test_workflow_lifecycle():
    ctx = ToolContextManager("conv1", "user1")
    state = ctx.declare_workflow("Test plan", ["step1", "step2"])
    assert state["status"] == "active"
    assert state["current_step"] == 0

    state = ctx.advance_workflow_step()
    assert state is not None
    assert state["current_step"] == 1
    assert state["status"] == "active"

    state = ctx.advance_workflow_step()
    assert state is not None
    assert state["current_step"] == 2
    assert state["status"] == "completed"

    state = ctx.complete_workflow()
    assert state is not None
    assert state["status"] == "completed"


def test_has_active_workflow():
    ctx = ToolContextManager("conv1", "user1")
    assert ctx.has_active_workflow is False
    ctx.declare_workflow("plan", ["s1"])
    assert ctx.has_active_workflow is True
    ctx.complete_workflow()
    assert ctx.has_active_workflow is False


def test_workflow_state_none():
    ctx = ToolContextManager("conv1", "user1")
    assert ctx.workflow_state is None


def test_has_context():
    ctx = ToolContextManager("conv1", "user1")
    assert ctx.has_context is False
    ctx.set_variable("x", "1")
    assert ctx.has_context is True
