"""Tests for workflow tools."""

from __future__ import annotations

import uuid

import pytest

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig

from threetears.agent.tools.collections import ContextItemCollection
from threetears.agent.tools.context import ToolContextManager
from threetears.agent.tools.utils import tool_result_status
from threetears.agent.tools.workflow import load_workflow_tools

from packages.agent.tools.tests.testing_utils import FakePool, make_context_metadata, make_nats_mock


@pytest.fixture()
def l1_backend() -> SQLiteBackend:
    b = SQLiteBackend(db_name=f"test_wf_{uuid.uuid4().hex[:8]}")
    b.initialize(make_context_metadata())
    yield b
    from threetears.core.testing import drain_and_shutdown_bridge

    drain_and_shutdown_bridge()
    b.reset()


@pytest.fixture()
def registry(l1_backend: SQLiteBackend) -> CollectionRegistry:
    reg = CollectionRegistry()
    # ``l2_key`` refuses to build a key on a registry with no principal scope.
    reg.configure(l1_backend=l1_backend, kv_key_scope="test-principal")
    return reg


@pytest.fixture()
def collection(registry: CollectionRegistry) -> ContextItemCollection:
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    nats = make_nats_mock()
    coll = ContextItemCollection(registry, config, nats_client=nats)
    coll.l3_pool = FakePool()
    return coll


@pytest.fixture()
def ctx(collection: ContextItemCollection) -> ToolContextManager:
    return ToolContextManager(collection, "00000000-0000-0000-0000-000000000001", "user1")


def _make_tools(ctx: ToolContextManager):
    tools = load_workflow_tools(ctx)
    tool_map = {t.name: t for t in tools}
    return tool_map


@pytest.mark.asyncio
async def test_variable_set_tool(ctx: ToolContextManager) -> None:
    tools = _make_tools(ctx)
    result = await tools["variable_set"].ainvoke({"key": "color", "value": "blue"})
    assert "Variable 'color' saved" in result
    var = await ctx.get_variable("color")
    assert var is not None
    assert var["value"] == "blue"


@pytest.mark.asyncio
async def test_variable_get_tool(ctx: ToolContextManager) -> None:
    tools = _make_tools(ctx)
    await ctx.set_variable("city", "Paris")
    result = await tools["variable_get"].ainvoke({"key": "city"})
    assert "Paris" in result


@pytest.mark.asyncio
async def test_variable_get_not_found(ctx: ToolContextManager) -> None:
    tools = _make_tools(ctx)
    result = await tools["variable_get"].ainvoke({"key": "nope"})
    assert result == "[NO CHANGE] variable_get: no variable named 'nope'. No variables are saved in this conversation."
    assert tool_result_status(result) == "no_change"


@pytest.mark.asyncio
async def test_variable_get_not_found_lists_the_names_that_are_saved(ctx: ToolContextManager) -> None:
    tools = _make_tools(ctx)
    await ctx.set_variable("city", "Paris")
    await ctx.set_variable("budget", "400", "number")
    result = await tools["variable_get"].ainvoke({"key": "town"})
    assert result == "[NO CHANGE] variable_get: no variable named 'town'. Saved names: budget, city."


@pytest.mark.asyncio
async def test_variable_set_at_the_cap_changes_nothing_and_is_not_a_failure(collection: ContextItemCollection) -> None:
    ctx = ToolContextManager(collection, "00000000-0000-0000-0000-000000000001", "user1", var_limit=2)
    tools = _make_tools(ctx)
    await tools["variable_set"].ainvoke({"key": "a", "value": "1"})
    await tools["variable_set"].ainvoke({"key": "b", "value": "2"})
    result = await tools["variable_set"].ainvoke({"key": "c", "value": "3"})
    assert result == (
        "[NO CHANGE] variable_set: Variable limit reached (2). Save over one you no longer need by reusing "
        "its name. Saved names: a, b."
    )
    assert tool_result_status(result) == "no_change"
    assert await ctx.get_variable("c") is None
    assert tool_result_status(await tools["variable_set"].ainvoke({"key": "a", "value": "9"})) == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["variable_set", "variable_get"])
async def test_a_variable_tool_that_broke_says_tool_error(
    ctx: ToolContextManager, monkeypatch: pytest.MonkeyPatch, tool_name: str
) -> None:
    async def _down(*args: object, **kwargs: object) -> None:
        raise RuntimeError("store is down")

    monkeypatch.setattr(ctx, "set_variable", _down)
    monkeypatch.setattr(ctx, "get_variable", _down)
    result = await _make_tools(ctx)[tool_name].ainvoke({"key": "a", "value": "1"})
    assert result == f"[TOOL ERROR] {tool_name}: store is down" and tool_result_status(result) == "error"


@pytest.mark.asyncio
async def test_context_recall_tool(ctx: ToolContextManager) -> None:
    tools = _make_tools(ctx)
    cid = await ctx.save_tool_result("calc", "42")
    result = await tools["context_recall"].ainvoke({"context_id": cid})
    assert "42" in result


@pytest.mark.asyncio
async def test_context_recall_not_found(ctx: ToolContextManager) -> None:
    tools = _make_tools(ctx)
    result = await tools["context_recall"].ainvoke({"context_id": "bad-id"})
    assert "not found" in result


@pytest.mark.asyncio
async def test_workflow_declare_tool(ctx: ToolContextManager) -> None:
    tools = _make_tools(ctx)
    result = tools["workflow_declare"].invoke(
        {
            "plan": "Test plan",
            "steps": ["step1", "step2"],
        }
    )
    assert "Workflow declared" in result
    assert ctx.has_active_workflow
