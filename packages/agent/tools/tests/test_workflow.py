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
async def test_a_long_result_is_recalled_in_parts_and_the_parts_are_the_whole(ctx: ToolContextManager) -> None:
    """One call returns one window, its window line first; the offset it names returns the next."""
    from threetears.agent.tools.context import RECALL_MAX_CHARS

    full = "".join(f"line {n:05d} of the page\n" for n in range(2000))
    cid = await ctx.save_tool_result("web_fetch", full)
    tool = _make_tools(ctx)["context_recall"]
    assert "offset" in tool.args, "the offset a window line names is an argument"
    parts: list[str] = []
    offset = 0
    for _ in range(10):
        result = await tool.ainvoke({"context_id": cid, "offset": offset})
        line, blank, body = result.split("\n", 2)
        assert line.startswith(f"[characters {offset:,}-") and blank == "", line
        assert len(body) <= RECALL_MAX_CHARS
        parts.append(body)
        if "the end of it" in line:
            break
        assert f"call context_recall again with offset={offset + len(body)} " in line
        offset += len(body)
    whole = "".join(parts)
    assert whole.endswith(full) and whole.index(": ") == len(whole) - len(full) - 2, "the key, then all of it"
    assert len(parts) == -(-len(whole) // RECALL_MAX_CHARS) > 2


@pytest.mark.asyncio
async def test_a_result_written_once_is_recalled_with_no_note(ctx: ToolContextManager) -> None:
    cid = await ctx.save_tool_result("web_search", "old results", input_fingerprint="query=moon")
    result = await _make_tools(ctx)["context_recall"].ainvoke({"context_id": cid})
    assert result.endswith(": old results")
    assert "Updated at" not in result


@pytest.mark.asyncio
async def test_a_refreshed_result_says_when_before_its_body(ctx: ToolContextManager) -> None:
    """The same call again replaces the row; a handle from the first call then opens the second's answer."""
    first = await ctx.save_tool_result("web_search", "old results", input_fingerprint="query=moon")
    second = await ctx.save_tool_result("web_search", "fresh results", input_fingerprint="query=moon")
    assert first == second
    item = await ctx.get_context_item(first)
    assert item is not None and item["date_updated"] > item["date_created"], (
        "the row keeps the day it was first written"
    )

    tools = {t.name: t for t in load_workflow_tools(ctx, tz="America/Los_Angeles")}
    result = await tools["context_recall"].ainvoke({"context_id": first})

    note, blank, body = result.split("\n", 2)
    assert note.startswith("(Updated at ") and note.endswith(
        ": a later call with the same input replaced what this first held.)"
    )
    assert note.split(":")[-2].split()[-1] in {"PDT", "PST"}, f"the time names the caller's zone: {note}"
    assert blank == ""
    assert body.endswith(": fresh results")


@pytest.mark.asyncio
async def test_a_refreshed_result_with_no_zone_named_says_utc(ctx: ToolContextManager) -> None:
    cid = await ctx.save_tool_result("web_search", "old", input_fingerprint="q")
    await ctx.save_tool_result("web_search", "new", input_fingerprint="q")
    result = await _make_tools(ctx)["context_recall"].ainvoke({"context_id": cid})
    assert result.splitlines()[0].startswith("(Updated at ") and " UTC:" in result.splitlines()[0]


def test_a_zone_that_is_not_one_is_refused(ctx: ToolContextManager) -> None:
    with pytest.raises(ValueError, match="not a time zone"):
        load_workflow_tools(ctx, tz="Mars/Olympus")


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


@pytest.mark.asyncio
async def test_workflow_declare_says_what_it_replaced(ctx: ToolContextManager) -> None:
    tools = _make_tools(ctx)
    first = tools["workflow_declare"].invoke({"plan": "Move house", "steps": ["pack", "drive", "unpack"]})
    assert (
        first
        == "Workflow declared. No plan was there before.\n  Plan: Move house\n  Steps:\n  1. pack\n  2. drive\n  3. unpack"
    )
    ctx.advance_workflow_step()
    second = tools["workflow_declare"].invoke({"plan": "Stay put", "steps": ["unpack"]})
    assert second == (
        "Workflow declared. It replaced the plan 'Move house' (3 steps, 1 done, active).\n"
        "  Plan: Stay put\n  Steps:\n  1. unpack"
    )
