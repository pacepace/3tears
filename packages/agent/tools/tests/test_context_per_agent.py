"""A conversation several agents share: each agent's context manager reads and writes only its own rows."""

from __future__ import annotations

import uuid

import pytest

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig

from threetears.agent.tools.collections import ContextItemCollection
from threetears.agent.tools.context import ToolContextManager

from packages.agent.tools.tests.testing_utils import FakePool, make_context_metadata, make_nats_mock

CONVERSATION = uuid.UUID("00000000-0000-0000-0000-0000000000c1")
AGENT_A = uuid.UUID("00000000-0000-0000-0000-0000000000a1")
AGENT_B = uuid.UUID("00000000-0000-0000-0000-0000000000b1")


@pytest.fixture()
def collection() -> ContextItemCollection:
    l1 = SQLiteBackend(db_name=f"test_ctx_agents_{uuid.uuid4().hex[:8]}")
    l1.initialize(make_context_metadata())
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, kv_key_scope="test-principal")
    coll = ContextItemCollection(
        registry,
        DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""),
        nats_client=make_nats_mock(),
    )
    coll.l3_pool = FakePool()
    yield coll
    from threetears.core.testing import drain_and_shutdown_bridge

    drain_and_shutdown_bridge()
    l1.reset()


def _manager(collection: ContextItemCollection, agent_id: uuid.UUID | None, **kwargs: object) -> ToolContextManager:
    return ToolContextManager(collection, CONVERSATION, uuid.uuid4(), agent_id=agent_id, **kwargs)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_an_agent_reads_only_the_results_and_variables_its_own_turns_wrote(
    collection: ContextItemCollection,
) -> None:
    a = _manager(collection, AGENT_A)
    await a.save_tool_result("web_search", "A's result", input_fingerprint="q=a")
    await a.set_variable("plan", "A's plan")
    b = _manager(collection, AGENT_B)
    await b.save_tool_result("web_search", "B's result", input_fingerprint="q=b")

    b_again = _manager(collection, AGENT_B)
    await b_again.load_context()
    assert [item["content"] for item in b_again.items] == ["B's result"]
    assert "A's" not in (b_again.build_conversation_context() or "")
    a_again = _manager(collection, AGENT_A)
    await a_again.load_context()
    assert sorted(item["content"] for item in a_again.items) == ["A's plan", "A's result"]


@pytest.mark.asyncio
async def test_every_row_a_scoped_manager_writes_names_its_agent(collection: ContextItemCollection) -> None:
    b = _manager(collection, AGENT_B)
    await b.save_tool_result("web_search", "r1", input_fingerprint="q")
    await b.save_tool_result("calc", "r2")
    await b.set_variable("v", "x")
    await b.register_media("slot", media_id="m1")
    rows = list(collection.l3_pool.rows.values())  # type: ignore[union-attr]
    assert len(rows) == 4
    assert {row["agent_id"] for row in rows} == {AGENT_B}


@pytest.mark.asyncio
async def test_the_same_call_by_two_agents_is_two_rows(collection: ContextItemCollection) -> None:
    a = _manager(collection, AGENT_A)
    b = _manager(collection, AGENT_B)
    first = await a.save_tool_result("web_search", "A saw this", input_fingerprint="same")
    second = await b.save_tool_result("web_search", "B saw this", input_fingerprint="same")
    assert first != second
    contents = {row["agent_id"]: row["content"] for row in collection.l3_pool.rows.values()}  # type: ignore[union-attr]
    assert contents == {AGENT_A: "A saw this", AGENT_B: "B saw this"}
    # and each agent's own repeat still refreshes its own row
    again = await b.save_tool_result("web_search", "B saw it again", input_fingerprint="same")
    assert again == second
    assert len(collection.l3_pool.rows) == 2  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_a_manager_that_names_no_agent_reads_every_row_and_still_dedups(
    collection: ContextItemCollection,
) -> None:
    await _manager(collection, AGENT_A).save_tool_result("calc", "a", input_fingerprint="1")
    unscoped = _manager(collection, None)
    first = await unscoped.save_tool_result("calc", "n", input_fingerprint="2")
    assert await unscoped.save_tool_result("calc", "n again", input_fingerprint="2") == first
    reread = _manager(collection, None)
    await reread.load_context()
    assert sorted(item["content"] for item in reread.items) == ["a", "n again"]


@pytest.mark.asyncio
async def test_the_result_cap_is_each_agents_own(collection: ContextItemCollection) -> None:
    a = _manager(collection, AGENT_A, result_limit=2)
    for i in range(2):
        await a.save_tool_result("calc", f"a{i}")
    b = _manager(collection, AGENT_B, result_limit=2)
    for i in range(3):
        await b.save_tool_result("calc", f"b{i}")
    by_agent: dict[uuid.UUID, list[str]] = {}
    for row in collection.l3_pool.rows.values():  # type: ignore[union-attr]
        by_agent.setdefault(row["agent_id"], []).append(row["content"])
    # B's third result evicts B's oldest, never one of A's
    assert sorted(by_agent.get(AGENT_A, [])) == ["a0", "a1"]
    assert sorted(by_agent.get(AGENT_B, [])) == ["b1", "b2"]
