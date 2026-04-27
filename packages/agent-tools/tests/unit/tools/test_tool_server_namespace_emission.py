"""tests for tool namespace emission on register / deregister.

namespace-task-01 phase 2 (tool-as-namespace) requires every
registered tool to materialize a ``platform.namespaces`` row of type
``tool`` so the unified rbac evaluator can resolve per-call
authorization at the Registry proxy. three-tier-task-01 phase F
routes that emission through :meth:`NamespaceCollection.save_entity`
on the agent-side three-tier stack (retiring the bespoke
``NamespaceEmitter`` Protocol + raw SQL). these tests cover:

- agent-owned tool emits a tool namespace with
  ``owner_agent_id=self._agent_id, customer_id=self._customer_id``
- platform built-in tool emits a tool namespace with both owner
  columns NULL
- missing ``namespace_collection`` suppresses emission (standalone
  devx / test scenario)
- a raising Collection on register surfaces the exception, not a
  silent drop
- deregister deletes the paired row through
  :meth:`NamespaceCollection.delete`
- a raising Collection on delete surfaces the exception
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid7

import pytest

from threetears.agent.tools.base_tool import (
    MCPToolDefinition,
    TearsTool,
    ToolResult,
)
from threetears.agent.tools.server import (
    ToolServer,
    _tool_namespace_id,
    _tool_namespace_name,
)


class _StubTool(TearsTool):
    """stub TearsTool for namespace emission tests."""

    def __init__(self, name: str = "test.ns_stub", version: str = "1.0") -> None:
        """initialize stub tool.

        :param name: namespaced tool name
        :ptype name: str
        :param version: version string
        :ptype version: str
        """
        self._name = name
        self._version = version

    async def execute(self, **kwargs: Any) -> ToolResult:
        """execute stub tool.

        :param kwargs: tool input parameters
        :ptype kwargs: Any
        :return: success result
        :rtype: ToolResult
        """
        return ToolResult(success=True, content=json.dumps(kwargs))

    def mcp_schema(self) -> MCPToolDefinition:
        """return stub MCP schema.

        :return: tool definition
        :rtype: MCPToolDefinition
        """
        return MCPToolDefinition(
            name=self._name,
            version=self._version,
            description="stub tool for namespace emission tests",
            input_schema={"type": "object", "properties": {}},
        )

    def mcp_name(self) -> str:
        """return namespaced tool name.

        :return: namespaced tool name
        :rtype: str
        """
        return self._name

    def mcp_version(self) -> str:
        """return tool version.

        :return: version string
        :rtype: str
        """
        return self._version


class _FakeNamespaceEntity:
    """mimics :class:`NamespaceEntity` for unit-test isolation.

    holds the data dict the Collection passes on construction so the
    test can inspect every field the ToolServer stamped on the entity
    without pulling the real :class:`aibots.hub.broker.namespaces`
    module into the agent-tools test graph.
    """

    def __init__(
        self,
        data: dict[str, Any],
        *,
        is_new: bool,
        collection: Any,
    ) -> None:
        """capture the entity payload for later assertions.

        :param data: field dict mint by the ToolServer
        :ptype data: dict[str, Any]
        :param is_new: ``True`` on insert path (unused here; kept to
            match the real :class:`BaseEntity` signature)
        :ptype is_new: bool
        :param collection: owning Collection reference (unused;
            mirrors the real signature)
        :ptype collection: Any
        """
        self.data = data
        self.is_new = is_new
        self.collection = collection


class _FakeNamespaceCollection:
    """captures ``save_entity`` / ``delete`` calls for test assertions.

    exposes :attr:`entity_class` so the production
    ``namespace_collection.entity_class(...)`` construction path in
    :meth:`ToolServer._emit_tool_namespace` resolves to
    :class:`_FakeNamespaceEntity`. ``save_entity`` / ``delete``
    respect the standard :class:`AsyncMock` semantics so callers can
    drive side effects for failure-path tests.
    """

    entity_class = _FakeNamespaceEntity

    def __init__(self) -> None:
        """wire async mocks for the two call surfaces under test."""
        self.save_entity = AsyncMock()
        self.delete = AsyncMock()


def _build_server(
    *,
    agent_id: Any = None,
    customer_id: Any = None,
    namespace_collection: Any = None,
) -> ToolServer:
    """build a ToolServer wired for namespace emission tests.

    :param agent_id: owning agent identity
    :ptype agent_id: Any
    :param customer_id: owning customer identity
    :ptype customer_id: Any
    :param namespace_collection: three-tier-task-01 phase F
        :class:`NamespaceCollection` stand-in
    :ptype namespace_collection: Any
    :return: fresh ToolServer
    :rtype: ToolServer
    """
    nc = MagicMock()
    nc.publish = AsyncMock()
    return ToolServer(
        nats_client=nc,
        namespace="aibots",
        agent_id=agent_id,
        customer_id=customer_id,
        namespace_collection=namespace_collection,
    )


class TestRegisterToolNamespaceEmission:
    """cover :meth:`ToolServer.register_tool` tool-namespace emission."""

    @pytest.mark.asyncio
    async def test_agent_owned_tool_emits_namespace_with_owner_and_customer(
        self,
    ) -> None:
        """agent-spun pod stamps both owner_agent_id and customer_id."""
        agent_id = uuid7()
        customer_id = uuid7()
        ns = _FakeNamespaceCollection()
        server = _build_server(
            agent_id=agent_id,
            customer_id=customer_id,
            namespace_collection=ns,
        )
        tool = _StubTool(name="aibots.calc", version="1.2.0")

        await server.register_tool(tool)

        ns.save_entity.assert_awaited_once()
        (entity,), _ = ns.save_entity.call_args
        assert isinstance(entity, _FakeNamespaceEntity)
        assert entity.is_new is True
        assert entity.data["name"] == _tool_namespace_name("aibots.calc", "1.2.0")
        assert entity.data["namespace_type"] == "tool"
        assert entity.data["owner_agent_id"] == agent_id
        assert entity.data["customer_id"] == customer_id
        assert entity.data["schema_name"] is None
        # natural-identity metadata: pre-sanitized mcp_name + mcp_version
        # so downstream pattern matching (hub access materializer +
        # registry authorizer canonical-name lookup) doesn't need to
        # reverse the sanitization in :func:`build_namespace_name`.
        assert entity.data["metadata"] == {
            "mcp_name": "aibots.calc",
            "mcp_version": "1.2.0",
        }
        assert entity.data["id"] == _tool_namespace_id(
            "aibots.calc",
            "1.2.0",
            agent_id,
        )

    @pytest.mark.asyncio
    async def test_platform_built_in_tool_emits_namespace_with_null_owner(
        self,
    ) -> None:
        """platform pod (no agent_id + no customer_id) emits NULL owners."""
        ns = _FakeNamespaceCollection()
        server = _build_server(
            agent_id=None,
            customer_id=None,
            namespace_collection=ns,
        )
        tool = _StubTool(name="platform.time.now", version="1.0.0")

        await server.register_tool(tool)

        (entity,), _ = ns.save_entity.call_args
        assert entity.data["name"] == _tool_namespace_name(
            "platform.time.now",
            "1.0.0",
        )
        assert entity.data["owner_agent_id"] is None
        assert entity.data["customer_id"] is None
        assert entity.data["id"] == _tool_namespace_id(
            "platform.time.now",
            "1.0.0",
            None,
        )

    @pytest.mark.asyncio
    async def test_missing_namespace_collection_suppresses_emission(self) -> None:
        """no namespace_collection means no emission (devx / standalone)."""
        server = _build_server(
            agent_id=uuid7(),
            customer_id=uuid7(),
            namespace_collection=None,
        )
        tool = _StubTool()
        # no assertion on a Collection; register should simply succeed
        await server.register_tool(tool)
        assert tool.mcp_name() in [k.split("@")[0] for k in server.tool_names]

    @pytest.mark.asyncio
    async def test_emit_failure_propagates(self) -> None:
        """a raising Collection surfaces the exception to the caller."""
        ns = _FakeNamespaceCollection()
        ns.save_entity.side_effect = RuntimeError("broker down")
        server = _build_server(
            agent_id=uuid7(),
            customer_id=uuid7(),
            namespace_collection=ns,
        )
        tool = _StubTool()

        with pytest.raises(RuntimeError, match="broker down"):
            await server.register_tool(tool)


class TestDeregisterToolNamespaceEmission:
    """cover :meth:`ToolServer.deregister_tool` tool-namespace deletion."""

    @pytest.mark.asyncio
    async def test_deregister_deletes_paired_namespace_row(self) -> None:
        """successful deregister issues the Collection delete call."""
        agent_id = uuid7()
        ns = _FakeNamespaceCollection()
        server = _build_server(
            agent_id=agent_id,
            customer_id=uuid7(),
            namespace_collection=ns,
        )
        tool = _StubTool(name="aibots.calc", version="1.2.0")
        await server.register_tool(tool)

        ns.delete.reset_mock()
        removed = await server.deregister_tool("aibots.calc")

        assert removed is True
        ns.delete.assert_awaited_once()
        (namespace_id,), _ = ns.delete.call_args
        assert namespace_id == _tool_namespace_id(
            "aibots.calc",
            "1.2.0",
            agent_id,
        )

    @pytest.mark.asyncio
    async def test_deregister_noop_does_not_emit_delete(self) -> None:
        """deregistering an unknown name skips the namespace delete."""
        ns = _FakeNamespaceCollection()
        server = _build_server(
            agent_id=uuid7(),
            customer_id=uuid7(),
            namespace_collection=ns,
        )
        removed = await server.deregister_tool("never-registered")
        assert removed is False
        ns.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_failure_propagates(self) -> None:
        """a raising Collection on delete surfaces the exception."""
        ns = _FakeNamespaceCollection()
        server = _build_server(
            agent_id=uuid7(),
            customer_id=uuid7(),
            namespace_collection=ns,
        )
        tool = _StubTool(name="aibots.calc", version="1.2.0")
        await server.register_tool(tool)

        ns.delete.side_effect = RuntimeError("delete failed")
        with pytest.raises(RuntimeError, match="delete failed"):
            await server.deregister_tool("aibots.calc")


class TestToolNamespaceNameHelper:
    """cover :func:`_tool_namespace_name` canonical name shape."""

    def test_shape_matches_documented_convention(self) -> None:
        """shape is ``tools.<sanitized-mcp>.<sanitized-version>``."""
        assert _tool_namespace_name("aibots.calc", "1.2.0") == "tools.aibots-calc.1-2-0"

    def test_version_dotted_segments_are_sanitized(self) -> None:
        """semver with dots collapses to hyphens inside each segment."""
        assert _tool_namespace_name("platform.x", "2.0.0-rc.1") == "tools.platform-x.2-0-0-rc-1"


class TestToolNamespaceIdHelper:
    """cover :func:`_tool_namespace_id` deterministic key derivation."""

    def test_same_triple_produces_same_id(self) -> None:
        """deterministic derivation makes replay idempotent via ON CONFLICT."""
        agent_id = uuid7()
        a = _tool_namespace_id("aibots.calc", "1.2.0", agent_id)
        b = _tool_namespace_id("aibots.calc", "1.2.0", agent_id)
        assert a == b

    def test_different_versions_produce_different_ids(self) -> None:
        """version pinning is preserved through the deterministic key."""
        agent_id = uuid7()
        a = _tool_namespace_id("aibots.calc", "1.2.0", agent_id)
        b = _tool_namespace_id("aibots.calc", "1.3.0", agent_id)
        assert a != b

    def test_platform_pod_uses_sentinel_owner(self) -> None:
        """None agent collapses onto a ``platform`` sentinel key."""
        a = _tool_namespace_id("platform.x", "1.0.0", None)
        b = _tool_namespace_id("platform.x", "1.0.0", None)
        assert a == b
