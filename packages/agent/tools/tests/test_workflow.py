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
    assert result == "Saved 'color' (string, 4 chars). Nothing was saved under that name before."
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
        "[NO CHANGE] variable_set: Variable limit reached (2). Delete one you no longer need with "
        "variable_delete, or save over one by reusing its name. Saved names: a, b."
    )
    assert tool_result_status(result) == "no_change"
    assert await ctx.get_variable("c") is None
    assert tool_result_status(await tools["variable_set"].ainvoke({"key": "a", "value": "9"})) == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["variable_set", "variable_get", "variable_delete"])
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


# -- What a set replaced or cut, the listing, and a delete ----------------------


@pytest.mark.asyncio
async def test_an_overwrite_says_the_value_it_replaced(ctx: ToolContextManager) -> None:
    tools = _make_tools(ctx)
    await tools["variable_set"].ainvoke({"key": "budget", "value": "400", "value_type": "number"})
    result = await tools["variable_set"].ainvoke({"key": "budget", "value": "450", "value_type": "number"})
    assert result == (
        "Saved 'budget' (number, 3 chars). It replaced the earlier value, which is gone (number, 3 chars): '400'"
    )
    assert (await ctx.get_variable("budget"))["value"] == "450"


@pytest.mark.asyncio
async def test_saving_the_same_value_again_says_so(ctx: ToolContextManager) -> None:
    tools = _make_tools(ctx)
    await tools["variable_set"].ainvoke({"key": "city", "value": "Paris"})
    result = await tools["variable_set"].ainvoke({"key": "city", "value": "Paris"})
    assert result == "Saved 'city' (string, 5 chars). It already held this value."


@pytest.mark.asyncio
async def test_a_long_replaced_value_is_shown_by_its_start_and_its_size(ctx: ToolContextManager) -> None:
    tools = _make_tools(ctx)
    await ctx.set_variable("notes", "x" * 250)
    result = await tools["variable_set"].ainvoke({"key": "notes", "value": "short"})
    assert result.endswith(f"(string, 250 chars): {'x' * 200!r} ... (50 more chars)")


@pytest.mark.asyncio
async def test_a_cut_value_says_how_much_was_kept_and_how_much_was_cut(collection: ContextItemCollection) -> None:
    ctx = ToolContextManager(collection, "00000000-0000-0000-0000-000000000001", "user1", var_max_chars=10)
    result = await _make_tools(ctx)["variable_set"].ainvoke({"key": "long", "value": "0123456789ABCDE"})
    assert result == (
        "Saved 'long' (string, 10 chars). It was cut: only the first 10 of 15 chars were kept, "
        "and the last 5 are not saved. Nothing was saved under that name before."
    )
    assert (await ctx.get_variable("long"))["value"] == "0123456789"


@pytest.mark.asyncio
async def test_a_value_that_fits_is_not_said_to_be_cut(collection: ContextItemCollection) -> None:
    """Good traffic past the cut report: a value exactly at the cap is whole."""
    ctx = ToolContextManager(collection, "00000000-0000-0000-0000-000000000001", "user1", var_max_chars=10)
    result = await _make_tools(ctx)["variable_set"].ainvoke({"key": "fits", "value": "0123456789"})
    assert result == "Saved 'fits' (string, 10 chars). Nothing was saved under that name before."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("value", "value_type", "problem"),
    [
        ("blue", "colour", "value_type 'colour' is not one of: string, number, json"),
        ("lots", "number", "value_type is number and 'lots' is not a number"),
    ],
)
async def test_a_value_that_is_not_its_kind_is_refused_and_nothing_is_saved(
    ctx: ToolContextManager, value: str, value_type: str, problem: str
) -> None:
    result = await _make_tools(ctx)["variable_set"].ainvoke({"key": "k", "value": value, "value_type": value_type})
    assert result == f"[TOOL ERROR] variable_set: {problem}. Nothing was saved."
    assert await ctx.get_variable("k") is None


@pytest.mark.asyncio
async def test_json_that_does_not_parse_is_refused(ctx: ToolContextManager) -> None:
    result = await _make_tools(ctx)["variable_set"].ainvoke({"key": "k", "value": "{oops", "value_type": "json"})
    assert result.startswith("[TOOL ERROR] variable_set: value_type is json and the value does not parse as JSON")
    assert await ctx.get_variable("k") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("value", "value_type"), [("-3.5e2", "number"), ('{"a": [1]}', "json"), ("anything", "string")]
)
async def test_a_value_of_its_kind_is_saved(ctx: ToolContextManager, value: str, value_type: str) -> None:
    """Good traffic past the kind check."""
    result = await _make_tools(ctx)["variable_set"].ainvoke({"key": "k", "value": value, "value_type": value_type})
    assert result.startswith(f"Saved 'k' ({value_type}, ")
    assert (await ctx.get_variable("k"))["value"] == value


@pytest.mark.asyncio
async def test_variable_list_shows_every_name_kind_size_and_start(ctx: ToolContextManager) -> None:
    tools = _make_tools(ctx)
    await ctx.set_variable("city", "Paris")
    await ctx.set_variable("budget", "400", "number")
    await ctx.set_variable("notes", "y" * 230)
    result = await tools["variable_list"].ainvoke({})
    assert result == (
        "3 of 50 variables this conversation may hold:\n"
        "- budget (number, 3 chars): '400'\n"
        "- city (string, 5 chars): 'Paris'\n"
        f"- notes (string, 230 chars): {'y' * 200!r} ... (30 more chars)"
    )


@pytest.mark.asyncio
async def test_variable_list_with_nothing_saved_says_so(ctx: ToolContextManager) -> None:
    result = await _make_tools(ctx)["variable_list"].ainvoke({})
    assert result == "No variables are saved in this conversation. It may hold 50."


@pytest.mark.asyncio
async def test_variable_delete_removes_it_and_says_what_it_held(ctx: ToolContextManager) -> None:
    tools = _make_tools(ctx)
    await ctx.set_variable("city", "Paris")
    await ctx.set_variable("budget", "400", "number")
    result = await tools["variable_delete"].ainvoke({"key": "city"})
    assert result == (
        "Deleted 'city' (string, 5 chars). It cannot be brought back. It held: 'Paris'. "
        "1 of 50 variables are saved now."
    )
    assert await ctx.get_variable("city") is None
    assert (await ctx.get_variable("budget"))["value"] == "400"


@pytest.mark.asyncio
async def test_variable_delete_of_a_name_not_saved_changes_nothing(ctx: ToolContextManager) -> None:
    tools = _make_tools(ctx)
    await ctx.set_variable("city", "Paris")
    result = await tools["variable_delete"].ainvoke({"key": "town"})
    assert result == "[NO CHANGE] variable_delete: no variable named 'town'. Saved names: city."
    assert tool_result_status(result) == "no_change"
    assert (await ctx.get_variable("city"))["value"] == "Paris"


@pytest.mark.asyncio
async def test_a_delete_frees_room_under_the_cap(collection: ContextItemCollection) -> None:
    ctx = ToolContextManager(collection, "00000000-0000-0000-0000-000000000001", "user1", var_limit=1)
    tools = _make_tools(ctx)
    await tools["variable_set"].ainvoke({"key": "a", "value": "1"})
    await tools["variable_delete"].ainvoke({"key": "a"})
    assert tool_result_status(await tools["variable_set"].ainvoke({"key": "b", "value": "2"})) == "ok"
