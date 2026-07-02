"""Tests for the context_recall builtin tool.

context_recall is the consumer half of the large-text offload mechanism:
given the ``<id>`` from a ``[ctx:<id>]`` handle it returns the full
stored content via the per-conversation
:class:`~threetears.agent.tools.context.ToolContextManager` resolved
from the active call scope. covers the happy path, the ``ctx:`` prefix,
the unknown-id miss, the no-scope / no-manager unavailable path, the
empty-input guard, and the default registration surface.
"""

from __future__ import annotations

import uuid

import pytest

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig

from threetears.agent.tools.builtin.context_recall import (
    ContextRecallTool,
    create_context_recall_tool,
)
from threetears.agent.tools.call_scope import ToolCallScope, enter_call_scope
from threetears.agent.tools.collections import ContextItemCollection
from threetears.agent.tools.context import ToolContextManager
from threetears.agent.tools.context_envelope import CallContext

from testing_utils import FakePool, make_context_metadata, make_nats_mock


@pytest.fixture()
def l1_backend() -> SQLiteBackend:
    b = SQLiteBackend(db_name=f"test_recall_{uuid.uuid4().hex[:8]}")
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
def collection(registry: CollectionRegistry, config: DefaultCoreConfig) -> ContextItemCollection:
    nats = make_nats_mock()
    coll = ContextItemCollection(registry, config, nats_client=nats)
    coll.l3_pool = FakePool()
    return coll


@pytest.fixture()
def manager(collection: ContextItemCollection) -> ToolContextManager:
    return ToolContextManager(collection, "00000000-0000-0000-0000-000000000001", "user1")


def _scope_for(manager: ToolContextManager) -> ToolCallScope:
    """build a call scope carrying ``manager`` for the dispatch."""
    return ToolCallScope(context=CallContext(), context_manager=manager)


class TestContextRecallExecute:
    """the tool's execute body against a real context manager."""

    @pytest.mark.asyncio
    async def test_recall_returns_stored_content(self, manager: ToolContextManager) -> None:
        """a saved tool result is returned in full by its context id."""
        full = "nmap scan output " * 500
        cid = await manager.save_tool_result("nmap", full)
        tool = ContextRecallTool()
        async with enter_call_scope(_scope_for(manager)):
            result = await tool.execute(context_id=cid)
        assert result.success is True
        assert result.content == full

    @pytest.mark.asyncio
    async def test_recall_accepts_ctx_prefix(self, manager: ToolContextManager) -> None:
        """a 'ctx:<id>' handle resolves to the same item as '<id>'."""
        full = "payload"
        cid = await manager.save_tool_result("nmap", full)
        tool = ContextRecallTool()
        async with enter_call_scope(_scope_for(manager)):
            result = await tool.execute(context_id=f"ctx:{cid}")
        assert result.success is True
        assert result.content == full

    @pytest.mark.asyncio
    async def test_recall_unknown_id_is_graceful_miss(self, manager: ToolContextManager) -> None:
        """an unknown id returns a clear not-found result, not an exception."""
        tool = ContextRecallTool()
        async with enter_call_scope(_scope_for(manager)):
            result = await tool.execute(context_id="0190abcd-0000-7000-8000-000000000000")
        assert result.success is False
        assert "not found" in result.content.lower()

    @pytest.mark.asyncio
    async def test_recall_without_scope_is_unavailable(self) -> None:
        """no call scope -> a clear unavailable result, not an exception."""
        tool = ContextRecallTool()
        result = await tool.execute(context_id="anything")
        assert result.success is False
        assert "unavailable" in result.content.lower()

    @pytest.mark.asyncio
    async def test_recall_without_manager_is_unavailable(self) -> None:
        """a scope carrying no context manager -> unavailable."""
        tool = ContextRecallTool()
        scope = ToolCallScope(context=CallContext(), context_manager=None)
        async with enter_call_scope(scope):
            result = await tool.execute(context_id="anything")
        assert result.success is False
        assert "unavailable" in result.content.lower()

    @pytest.mark.asyncio
    async def test_recall_empty_id_is_rejected(self, manager: ToolContextManager) -> None:
        """an empty context_id is rejected with a clear message."""
        tool = ContextRecallTool()
        async with enter_call_scope(_scope_for(manager)):
            result = await tool.execute(context_id="")
        assert result.success is False
        assert "context_id" in result.content


class TestContextRecallSchema:
    """name / version / schema surface."""

    def test_mcp_name(self) -> None:
        """canonical mcp name is threetears.context_recall."""
        assert ContextRecallTool().mcp_name() == "threetears.context_recall"

    def test_schema_requires_context_id(self) -> None:
        """the input schema requires context_id."""
        schema = ContextRecallTool().mcp_schema()
        assert "context_id" in schema.input_schema["required"]


class TestContextRecallRegistration:
    """context_recall is in the default builtin surface."""

    def test_in_standard_builtin_factories(self) -> None:
        """the zero-config factory map carries context_recall and builds it."""
        from threetears.agent.tools.builtin import STANDARD_BUILTIN_FACTORIES

        assert "threetears.context_recall" in STANDARD_BUILTIN_FACTORIES
        built = STANDARD_BUILTIN_FACTORIES["threetears.context_recall"]()
        assert isinstance(built, ContextRecallTool)

    def test_in_standard_tools_alias(self) -> None:
        """context_recall is part of the 'standard' group surface."""
        from threetears.agent.tools.aliases import STANDARD_TOOLS

        assert "threetears.context_recall" in STANDARD_TOOLS

    def test_factories_match_standard_tools(self) -> None:
        """the factory map and the standard alias stay in lockstep."""
        from threetears.agent.tools.aliases import STANDARD_TOOLS
        from threetears.agent.tools.builtin import STANDARD_BUILTIN_FACTORIES

        assert set(STANDARD_BUILTIN_FACTORIES.keys()) == set(STANDARD_TOOLS)

    def test_factory_produces_named_structured_tool(self) -> None:
        """create_context_recall_tool yields a StructuredTool with the canonical name."""
        tool = create_context_recall_tool({}, "recall stored content")
        assert tool.name == "threetears.context_recall"

    def test_register_builtins_includes_context_recall(self) -> None:
        """register_builtins wires context_recall onto the registry."""
        from threetears.agent.tools.builtin import register_builtins
        from threetears.agent.tools.registry import ToolRegistry

        reg = ToolRegistry()
        register_builtins(reg)
        assert "context_recall" in reg.list_types()


@pytest.mark.asyncio
async def test_offload_recall_round_trips_across_registries() -> None:
    """W1: offloaded content written via one manager is recallable via a
    SEPARATE manager that shares the same durable L3 store.

    simulates the SDK's two-registry architecture: the offloader writes
    through the agent-side ``ContextIntegration`` manager (registry A);
    the ``context_recall`` tool reads through the ToolServer
    ``context_factory`` manager (registry B). the two registries are
    distinct instances (distinct L1 SQLite backends) but resolve to the
    SAME durable L3 (here a shared :class:`FakePool`, the project's
    stateful L3 double).

    proves the round-trip logic: ``save_tool_result`` commits to L3
    synchronously (``upsert_tool_result`` -> direct ``l3_pool`` write),
    a fresh reader manager's ``load_context()`` pulls the row back from
    L3 (``find_by_conversation`` reads ``l3_pool`` directly), and
    ``get_context_item`` returns the full content. The production
    "two NatsProxy backends resolve to the same YugabyteDB" fact is
    config-level (same namespace + agent_id) and is validated on a live
    stack; this test pins the manager/collection logic above it.
    """
    from threetears.core._bridge import drain, shutdown

    shared_l3 = FakePool()
    conv = "00000000-0000-0000-0000-0000000000aa"
    cfg = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")

    backend_a = SQLiteBackend(db_name=f"rt_a_{uuid.uuid4().hex[:8]}")
    backend_a.initialize(make_context_metadata())
    backend_b = SQLiteBackend(db_name=f"rt_b_{uuid.uuid4().hex[:8]}")
    backend_b.initialize(make_context_metadata())
    try:
        reg_a = CollectionRegistry()
        reg_a.configure(l1_backend=backend_a)
        reg_b = CollectionRegistry()
        reg_b.configure(l1_backend=backend_b)

        coll_a = ContextItemCollection(reg_a, cfg, nats_client=make_nats_mock())
        coll_a.l3_pool = shared_l3
        coll_b = ContextItemCollection(reg_b, cfg, nats_client=make_nats_mock())
        coll_b.l3_pool = shared_l3

        writer = ToolContextManager(coll_a, conv, "user1")
        reader = ToolContextManager(coll_b, conv, "user1")

        full = "nmap scan output " * 2000
        cid = await writer.save_tool_result("nmap", full)

        # the reader has never observed this conversation; rehydrate from
        # the shared L3 the writer committed to.
        await reader.load_context()
        item = await reader.get_context_item(cid)
        assert item is not None
        assert item["content"] == full
    finally:
        drain()
        shutdown()
        backend_a.reset()
        backend_b.reset()
