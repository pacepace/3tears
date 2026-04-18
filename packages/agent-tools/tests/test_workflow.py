"""Tests for workflow tools."""

from __future__ import annotations

import uuid

import pytest

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig

from threetears.agent.tools.collections import ContextItemCollection
from threetears.agent.tools.context import ToolContextManager
from threetears.agent.tools.workflow import load_workflow_tools

from testing_utils import FakePool, make_context_metadata, make_nats_mock


@pytest.fixture()
def l1_backend() -> SQLiteBackend:
    b = SQLiteBackend(db_name=f"test_wf_{uuid.uuid4().hex[:8]}")
    b.initialize(make_context_metadata())
    yield b
    from threetears.core._bridge import drain, shutdown

    drain()
    shutdown()
    b.reset()


@pytest.fixture()
def registry(l1_backend: SQLiteBackend) -> CollectionRegistry:
    reg = CollectionRegistry()
    reg.configure(l1_backend=l1_backend)
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
async def test_set_variable_tool(ctx: ToolContextManager) -> None:
    tools = _make_tools(ctx)
    result = await tools["set_variable"].ainvoke({"key": "color", "value": "blue"})
    assert "Variable 'color' saved" in result
    var = await ctx.get_variable("color")
    assert var is not None
    assert var["value"] == "blue"


@pytest.mark.asyncio
async def test_get_variable_tool(ctx: ToolContextManager) -> None:
    tools = _make_tools(ctx)
    await ctx.set_variable("city", "Paris")
    result = await tools["get_variable"].ainvoke({"key": "city"})
    assert "Paris" in result


@pytest.mark.asyncio
async def test_get_variable_not_found(ctx: ToolContextManager) -> None:
    tools = _make_tools(ctx)
    result = await tools["get_variable"].ainvoke({"key": "nope"})
    assert "not found" in result


@pytest.mark.asyncio
async def test_recall_context_tool(ctx: ToolContextManager) -> None:
    tools = _make_tools(ctx)
    cid = await ctx.save_tool_result("calc", "42")
    result = await tools["recall_context"].ainvoke({"context_id": cid})
    assert "42" in result


@pytest.mark.asyncio
async def test_recall_context_not_found(ctx: ToolContextManager) -> None:
    tools = _make_tools(ctx)
    result = await tools["recall_context"].ainvoke({"context_id": "bad-id"})
    assert "not found" in result


@pytest.mark.asyncio
async def test_declare_workflow_tool(ctx: ToolContextManager) -> None:
    tools = _make_tools(ctx)
    result = tools["declare_workflow"].invoke(
        {
            "plan": "Test plan",
            "steps": ["step1", "step2"],
        }
    )
    assert "Workflow declared" in result
    assert ctx.has_active_workflow
