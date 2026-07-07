"""Tests for ToolContextManager (collection-backed)."""

from __future__ import annotations

import uuid

import pytest

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig

from threetears.agent.tools.collections import ContextItemCollection
from threetears.agent.tools.context import ToolContextManager

from testing_utils import FakePool, make_context_metadata, make_nats_mock


@pytest.fixture()
def l1_backend() -> SQLiteBackend:
    b = SQLiteBackend(db_name=f"test_mgr_{uuid.uuid4().hex[:8]}")
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
def config() -> DefaultCoreConfig:
    return DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")


@pytest.fixture()
def pool() -> FakePool:
    return FakePool()


@pytest.fixture()
def collection(registry: CollectionRegistry, config: DefaultCoreConfig, pool: FakePool) -> ContextItemCollection:
    nats = make_nats_mock()
    coll = ContextItemCollection(registry, config, nats_client=nats)
    coll.l3_pool = pool
    return coll


@pytest.fixture()
def ctx(collection: ContextItemCollection) -> ToolContextManager:
    return ToolContextManager(collection, "00000000-0000-0000-0000-000000000001", "user1")


# -- Construction guard --


def test_init_rejects_none_collection() -> None:
    """A ``None`` collection is a wiring bug, not a supported mode.

    ``load_context`` dereferences the collection unconditionally
    (``collection.find_by_conversation``), so a missing collection would
    otherwise surface as an opaque ``NoneType`` ``AttributeError`` mid-stream
    on the first load. Reject it at construction so a caller that fails to
    thread the context collection fails loudly and locally.
    """
    with pytest.raises(TypeError, match="collection"):
        ToolContextManager(None, "00000000-0000-0000-0000-000000000001", "user1")


# -- Variable tests --


@pytest.mark.asyncio
async def test_set_get_variable(ctx: ToolContextManager) -> None:
    await ctx.set_variable("name", "Alice")
    var = await ctx.get_variable("name")
    assert var is not None
    assert var["value"] == "Alice"
    assert var["value_type"] == "string"


@pytest.mark.asyncio
async def test_set_variable_limit(collection: ContextItemCollection) -> None:
    ctx = ToolContextManager(collection, "00000000-0000-0000-0000-000000000001", "user1", var_limit=2)
    await ctx.set_variable("a", "1")
    await ctx.set_variable("b", "2")
    with pytest.raises(ValueError, match="Variable limit reached"):
        await ctx.set_variable("c", "3")


@pytest.mark.asyncio
async def test_set_variable_truncates(collection: ContextItemCollection) -> None:
    ctx = ToolContextManager(collection, "00000000-0000-0000-0000-000000000001", "user1", var_max_chars=10)
    await ctx.set_variable("long", "x" * 100)
    var = await ctx.get_variable("long")
    assert var is not None
    assert len(var["value"]) == 10


@pytest.mark.asyncio
async def test_set_variable_upsert(collection: ContextItemCollection) -> None:
    ctx = ToolContextManager(collection, "00000000-0000-0000-0000-000000000001", "user1", var_limit=1)
    await ctx.set_variable("key", "v1")
    await ctx.set_variable("key", "v2")  # Should not raise
    var = await ctx.get_variable("key")
    assert var is not None
    assert var["value"] == "v2"


@pytest.mark.asyncio
async def test_delete_variable(ctx: ToolContextManager) -> None:
    await ctx.set_variable("x", "1")
    assert await ctx.delete_variable("x") is True
    assert await ctx.get_variable("x") is None


@pytest.mark.asyncio
async def test_delete_nonexistent(ctx: ToolContextManager) -> None:
    assert await ctx.delete_variable("nope") is False


@pytest.mark.asyncio
async def test_get_all_variables(ctx: ToolContextManager) -> None:
    await ctx.set_variable("a", "1")
    await ctx.set_variable("b", "2")
    all_vars = await ctx.get_all_variables()
    assert len(all_vars) == 2
    keys = {v["key"] for v in all_vars}
    assert keys == {"a", "b"}


# -- Tool result tests --


@pytest.mark.asyncio
async def test_save_tool_result(ctx: ToolContextManager) -> None:
    cid = await ctx.save_tool_result("calc", "42")
    item = await ctx.get_context_item(cid)
    assert item is not None
    assert item["content"] == "42"
    # Without an input fingerprint the key is made unique per call
    # (tool_name:context_id) so the v004 tool_result dedup index never
    # rejects the insert; the tool-name prefix is preserved.
    assert item["key"].startswith("calc:")


@pytest.mark.asyncio
async def test_save_tool_result_dedups_same_input_fingerprint(
    ctx: ToolContextManager,
) -> None:
    """Same tool + same input fingerprint upserts one row (dedup); a
    different fingerprint creates a distinct row."""
    cid1 = await ctx.save_tool_result(
        "web_search",
        "old results",
        input_fingerprint="query=moon",
    )
    cid2 = await ctx.save_tool_result(
        "web_search",
        "fresh results",
        input_fingerprint="query=moon",
    )
    # Same fingerprint -> same row, refreshed content.
    assert cid1 == cid2
    item = await ctx.get_context_item(cid2)
    assert item is not None
    assert item["content"] == "fresh results"
    tool_results = [i for i in ctx.items if i["context_type"] == "tool_result"]
    assert len(tool_results) == 1

    # Different fingerprint -> separate row.
    cid3 = await ctx.save_tool_result(
        "web_search",
        "mars results",
        input_fingerprint="query=mars",
    )
    assert cid3 != cid1
    tool_results = [i for i in ctx.items if i["context_type"] == "tool_result"]
    assert len(tool_results) == 2


@pytest.mark.asyncio
async def test_save_tool_result_observability_metadata(ctx: ToolContextManager) -> None:
    """``status`` / ``duration_ms`` / ``error`` are merged into metadata."""
    cid = await ctx.save_tool_result(
        "search",
        "found 5 results",
        status="completed",
        duration_ms=234,
    )
    item = await ctx.get_context_item(cid)
    assert item is not None
    assert item["metadata"]["status"] == "completed"
    assert item["metadata"]["duration_ms"] == 234


@pytest.mark.asyncio
async def test_save_tool_result_failed_with_error_falls_back_short_desc(
    ctx: ToolContextManager,
) -> None:
    """``status='failed'`` + ``error=...`` produces a FAILED short_desc."""
    cid = await ctx.save_tool_result(
        "search",
        "",
        status="failed",
        error="upstream timeout",
    )
    item = await ctx.get_context_item(cid)
    assert item is not None
    assert item["short_desc"].startswith("FAILED:")
    assert "upstream timeout" in item["short_desc"]
    assert item["metadata"]["status"] == "failed"
    assert item["metadata"]["error"] == "upstream timeout"


@pytest.mark.asyncio
async def test_save_tool_result_explicit_short_desc_wins_over_failure_fallback(
    ctx: ToolContextManager,
) -> None:
    """An explicit ``short_desc`` is not overridden by the failed-fallback."""
    cid = await ctx.save_tool_result(
        "search",
        "",
        short_desc="user-supplied summary",
        status="failed",
        error="upstream timeout",
    )
    item = await ctx.get_context_item(cid)
    assert item is not None
    assert item["short_desc"] == "user-supplied summary"


@pytest.mark.asyncio
async def test_save_tool_result_caller_metadata_takes_precedence(
    ctx: ToolContextManager,
) -> None:
    """Caller-supplied metadata keys are not overwritten by the kwargs."""
    cid = await ctx.save_tool_result(
        "search",
        "result",
        metadata={"status": "caller-wins"},
        status="completed",
    )
    item = await ctx.get_context_item(cid)
    assert item is not None
    assert item["metadata"]["status"] == "caller-wins"


@pytest.mark.asyncio
async def test_save_tool_result_error_truncated_to_1000(ctx: ToolContextManager) -> None:
    """Long error strings are truncated to 1000 chars in metadata."""
    huge = "x" * 5000
    cid = await ctx.save_tool_result(
        "search",
        "",
        status="failed",
        error=huge,
    )
    item = await ctx.get_context_item(cid)
    assert item is not None
    assert len(item["metadata"]["error"]) == 1000


@pytest.mark.asyncio
async def test_get_context_item_not_found(ctx: ToolContextManager) -> None:
    assert await ctx.get_context_item("nonexistent-id") is None


@pytest.mark.asyncio
async def test_get_context_item_updates_date_accessed(ctx: ToolContextManager) -> None:
    cid = await ctx.save_tool_result("calc", "42")
    item_before = await ctx.get_context_item(cid)
    assert item_before is not None
    accessed_before = item_before["date_accessed"]

    # Access again after a tiny delay
    item_after = await ctx.get_context_item(cid)
    assert item_after is not None
    assert item_after["date_accessed"] >= accessed_before


# -- Context building tests --


@pytest.mark.asyncio
async def test_build_conversation_context(ctx: ToolContextManager) -> None:
    await ctx.set_variable("name", "Alice")
    await ctx.save_tool_result("calc", "42")
    result = ctx.build_conversation_context()
    assert result is not None
    assert "name" in result
    assert "Alice" in result
    assert "42" in result
    assert "[Conversation Variables]" in result
    assert "[Tool Results]" in result


@pytest.mark.asyncio
async def test_build_conversation_context_empty(ctx: ToolContextManager) -> None:
    assert ctx.build_conversation_context() is None


# -- Media slot tests --


@pytest.mark.asyncio
async def test_register_media(ctx: ToolContextManager) -> None:
    await ctx.register_media("image_1", url="http://example.com/img.png")
    slots = ctx.get_slots()
    assert "image_1" in slots
    assert slots["image_1"]["metadata"]["url"] == "http://example.com/img.png"


@pytest.mark.asyncio
async def test_build_media_context(ctx: ToolContextManager) -> None:
    await ctx.register_media("image_1", url="http://example.com/img.png")
    result = ctx.build_media_context()
    assert result is not None
    assert "[Active Media Slots]" in result
    assert "image_1" in result


def test_build_media_context_empty(ctx: ToolContextManager) -> None:
    assert ctx.build_media_context() is None


# -- Workflow tests --


def test_workflow_lifecycle(ctx: ToolContextManager) -> None:
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


def test_has_active_workflow(ctx: ToolContextManager) -> None:
    assert ctx.has_active_workflow is False
    ctx.declare_workflow("plan", ["s1"])
    assert ctx.has_active_workflow is True
    ctx.complete_workflow()
    assert ctx.has_active_workflow is False


def test_workflow_state_none(ctx: ToolContextManager) -> None:
    assert ctx.workflow_state is None


# -- LRU eviction tests --


@pytest.mark.asyncio
async def test_result_limit_evicts_lru(collection: ContextItemCollection) -> None:
    ctx = ToolContextManager(collection, "00000000-0000-0000-0000-000000000001", "user1", result_limit=3)
    ids = []
    for i in range(5):
        cid = await ctx.save_tool_result(f"tool{i}", f"result{i}")
        ids.append(cid)

    # Only 3 should remain in the local projection
    tool_results = [i for i in ctx.items if i["context_type"] == "tool_result"]
    assert len(tool_results) == 3

    # Oldest two should be evicted
    assert await ctx.get_context_item(ids[0]) is None
    assert await ctx.get_context_item(ids[1]) is None
    # Newest three should be accessible
    assert await ctx.get_context_item(ids[2]) is not None


@pytest.mark.asyncio
async def test_result_limit_does_not_evict_variables(collection: ContextItemCollection) -> None:
    ctx = ToolContextManager(collection, "00000000-0000-0000-0000-000000000001", "user1", result_limit=2)
    await ctx.set_variable("name", "Alice")
    for i in range(3):
        await ctx.save_tool_result(f"tool{i}", f"result{i}")

    # Variable should still exist
    var = await ctx.get_variable("name")
    assert var is not None
    assert var["value"] == "Alice"

    # Only 2 tool results should remain
    tool_results = [i for i in ctx.items if i["context_type"] == "tool_result"]
    assert len(tool_results) == 2


@pytest.mark.asyncio
async def test_result_limit_lru_access_protects(collection: ContextItemCollection, pool: FakePool) -> None:
    ctx = ToolContextManager(collection, "00000000-0000-0000-0000-000000000001", "user1", result_limit=3)
    id_a = await ctx.save_tool_result("a", "result_a")
    id_b = await ctx.save_tool_result("b", "result_b")
    id_c = await ctx.save_tool_result("c", "result_c")

    # Access 'a' to update its date_accessed
    await ctx.get_context_item(id_a)

    # Add two more — should evict b and c, not a
    await ctx.save_tool_result("d", "result_d")
    await ctx.save_tool_result("e", "result_e")

    assert await ctx.get_context_item(id_a) is not None, "accessed item should survive"
    assert await ctx.get_context_item(id_b) is None, "untouched item should be evicted"
    assert await ctx.get_context_item(id_c) is None, "untouched item should be evicted"


# -- has_context --


@pytest.mark.asyncio
async def test_has_context(ctx: ToolContextManager) -> None:
    assert ctx.has_context is False
    await ctx.set_variable("x", "1")
    assert ctx.has_context is True


# -- load_context --


@pytest.mark.asyncio
async def test_load_context(ctx: ToolContextManager, pool: FakePool) -> None:
    # Save items via the manager
    await ctx.set_variable("name", "Alice")
    await ctx.save_tool_result("calc", "42")

    # Clear local state
    ctx.items = []
    assert ctx.has_context is False

    # Reload from collection
    await ctx.load_context()
    assert ctx.has_context is True
    assert len(ctx.items) == 2


# -- Arbitrary context item tests (save/get/delete by (context_type, key)) --


@pytest.mark.asyncio
async def test_save_and_get_item_by_type_and_key(ctx: ToolContextManager) -> None:
    """save_item_by_type_and_key then get_item_by_type_and_key round-trips."""
    cid = await ctx.save_item_by_type_and_key(
        context_type="workspace_pin",
        key="current",
        content="ws-uuid-content",
        short_desc="main",
        metadata={"workspace_name": "main"},
    )
    assert isinstance(cid, str)
    item = await ctx.get_item_by_type_and_key("workspace_pin", "current")
    assert item is not None
    assert item["context_type"] == "workspace_pin"
    assert item["key"] == "current"
    assert item["content"] == "ws-uuid-content"
    assert item["short_desc"] == "main"
    assert item["metadata"]["workspace_name"] == "main"


@pytest.mark.asyncio
async def test_get_item_by_type_and_key_missing_returns_none(
    ctx: ToolContextManager,
) -> None:
    """no matching item -> None (not raised)."""
    result = await ctx.get_item_by_type_and_key("workspace_pin", "current")
    assert result is None


@pytest.mark.asyncio
async def test_save_item_by_type_and_key_replaces_existing(
    ctx: ToolContextManager,
) -> None:
    """second save with same (type, key) replaces the first (single-item semantics)."""
    await ctx.save_item_by_type_and_key(
        context_type="workspace_pin",
        key="current",
        content="first",
    )
    await ctx.save_item_by_type_and_key(
        context_type="workspace_pin",
        key="current",
        content="second",
    )
    item = await ctx.get_item_by_type_and_key("workspace_pin", "current")
    assert item is not None
    assert item["content"] == "second"
    # only one item persists in the local projection
    matching = [i for i in ctx.items if i["context_type"] == "workspace_pin" and i["key"] == "current"]
    assert len(matching) == 1


@pytest.mark.asyncio
async def test_delete_item_by_type_and_key_returns_true_when_present(
    ctx: ToolContextManager,
) -> None:
    """delete returns True after removing an existing item."""
    await ctx.save_item_by_type_and_key(
        context_type="bookmark",
        key="favorite",
        content="x",
    )
    result = await ctx.delete_item_by_type_and_key("bookmark", "favorite")
    assert result is True
    assert await ctx.get_item_by_type_and_key("bookmark", "favorite") is None


@pytest.mark.asyncio
async def test_delete_item_by_type_and_key_is_idempotent(
    ctx: ToolContextManager,
) -> None:
    """delete returns False when no matching item; does not raise."""
    result = await ctx.delete_item_by_type_and_key("bookmark", "none")
    assert result is False


@pytest.mark.asyncio
async def test_save_item_by_type_and_key_scopes_by_type(
    ctx: ToolContextManager,
) -> None:
    """same key under different context_types co-exist without collision."""
    await ctx.save_item_by_type_and_key(
        context_type="workspace_pin",
        key="current",
        content="pin",
    )
    await ctx.save_item_by_type_and_key(
        context_type="bookmark",
        key="current",
        content="bm",
    )
    pin = await ctx.get_item_by_type_and_key("workspace_pin", "current")
    bm = await ctx.get_item_by_type_and_key("bookmark", "current")
    assert pin is not None and pin["content"] == "pin"
    assert bm is not None and bm["content"] == "bm"
