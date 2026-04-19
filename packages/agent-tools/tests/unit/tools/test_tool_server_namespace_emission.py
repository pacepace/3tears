"""tests for tool namespace emission on register / deregister.

namespace-task-01 phase 2 (tool-as-namespace): every registered tool
materializes a ``platform.namespaces`` row of type ``tool`` so the
unified rbac evaluator can resolve per-call authorization at the
Registry proxy. these tests cover:

- agent-owned tool emits a tool namespace with
  ``owner_agent_id=self._agent_id, customer_id=self._customer_id``
- platform built-in tool emits a tool namespace with both owner
  columns NULL
- missing ``l3_backend`` suppresses emission (test / standalone scenario)
- a raising backend on register surfaces the exception, not a silent drop
- deregister deletes the paired rows
- a raising backend on delete surfaces the exception
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
from threetears.agent.tools.server import ToolServer, _tool_namespace_name


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


def _build_server(
    *,
    agent_id: Any = None,
    customer_id: Any = None,
    l3_backend: Any = None,
) -> ToolServer:
    """build a ToolServer wired for namespace emission tests.

    :param agent_id: owning agent identity
    :ptype agent_id: Any
    :param customer_id: owning customer identity
    :ptype customer_id: Any
    :param l3_backend: namespace emitter
    :ptype l3_backend: Any
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
        l3_backend=l3_backend,
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
        l3 = MagicMock()
        l3.execute = AsyncMock()
        server = _build_server(
            agent_id=agent_id,
            customer_id=customer_id,
            l3_backend=l3,
        )
        tool = _StubTool(name="aibots.calc", version="1.2.0")

        await server.register_tool(tool)

        l3.execute.assert_awaited_once()
        args, kwargs = l3.execute.call_args
        # first arg is the SQL text; positional bindings follow
        assert "INSERT INTO platform.namespaces" in args[0]
        # bindings: namespace_id, name, owner_agent_id, customer_id, date_created, date_updated
        _namespace_id, name, owner_agent_id, binding_customer_id, *_rest = args[1:]
        assert name == _tool_namespace_name("aibots.calc", "1.2.0")
        assert owner_agent_id == agent_id
        assert binding_customer_id == customer_id
        assert kwargs.get("namespace") == "platform"

    @pytest.mark.asyncio
    async def test_platform_built_in_tool_emits_namespace_with_null_owner(
        self,
    ) -> None:
        """platform pod (no agent_id + no customer_id) emits NULL owners."""
        l3 = MagicMock()
        l3.execute = AsyncMock()
        server = _build_server(
            agent_id=None,
            customer_id=None,
            l3_backend=l3,
        )
        tool = _StubTool(name="platform.time.now", version="1.0.0")

        await server.register_tool(tool)

        args, _kwargs = l3.execute.call_args
        _namespace_id, name, owner_agent_id, customer_id, *_rest = args[1:]
        assert name == _tool_namespace_name("platform.time.now", "1.0.0")
        assert owner_agent_id is None
        assert customer_id is None

    @pytest.mark.asyncio
    async def test_missing_l3_backend_suppresses_emission(self) -> None:
        """no l3_backend means no emission (test / standalone scenarios)."""
        server = _build_server(agent_id=uuid7(), customer_id=uuid7(), l3_backend=None)
        tool = _StubTool()
        # no assertion on l3; register should simply succeed without wiring
        await server.register_tool(tool)
        assert tool.mcp_name() in [k.split("@")[0] for k in server.tool_names]

    @pytest.mark.asyncio
    async def test_emit_failure_propagates(self) -> None:
        """a raising backend surfaces the exception to the caller."""
        l3 = MagicMock()
        l3.execute = AsyncMock(side_effect=RuntimeError("broker down"))
        server = _build_server(
            agent_id=uuid7(),
            customer_id=uuid7(),
            l3_backend=l3,
        )
        tool = _StubTool()

        with pytest.raises(RuntimeError, match="broker down"):
            await server.register_tool(tool)


class TestDeregisterToolNamespaceEmission:
    """cover :meth:`ToolServer.deregister_tool` tool-namespace deletion."""

    @pytest.mark.asyncio
    async def test_deregister_deletes_paired_namespace_row(self) -> None:
        """successful deregister issues the delete statement."""
        l3 = MagicMock()
        l3.execute = AsyncMock()
        server = _build_server(
            agent_id=uuid7(),
            customer_id=uuid7(),
            l3_backend=l3,
        )
        tool = _StubTool(name="aibots.calc", version="1.2.0")
        await server.register_tool(tool)

        l3.execute.reset_mock()
        removed = await server.deregister_tool("aibots.calc")

        assert removed is True
        l3.execute.assert_awaited_once()
        args, kwargs = l3.execute.call_args
        assert "DELETE FROM platform.namespaces" in args[0]
        assert args[1] == "tool:aibots.calc:%"
        assert kwargs.get("namespace") == "platform"

    @pytest.mark.asyncio
    async def test_deregister_noop_does_not_emit_delete(self) -> None:
        """deregistering an unknown name skips the namespace delete."""
        l3 = MagicMock()
        l3.execute = AsyncMock()
        server = _build_server(
            agent_id=uuid7(),
            customer_id=uuid7(),
            l3_backend=l3,
        )
        removed = await server.deregister_tool("never-registered")
        assert removed is False
        l3.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_failure_propagates(self) -> None:
        """a raising backend on delete surfaces the exception."""
        l3 = MagicMock()
        l3.execute = AsyncMock(return_value=None)
        server = _build_server(
            agent_id=uuid7(),
            customer_id=uuid7(),
            l3_backend=l3,
        )
        tool = _StubTool(name="aibots.calc", version="1.2.0")
        await server.register_tool(tool)

        l3.execute.side_effect = RuntimeError("delete failed")
        with pytest.raises(RuntimeError, match="delete failed"):
            await server.deregister_tool("aibots.calc")


class TestToolNamespaceNameHelper:
    """cover :func:`_tool_namespace_name` canonical name shape."""

    def test_shape_matches_documented_convention(self) -> None:
        """shape is ``tool:<mcp_name>:<version>``."""
        assert _tool_namespace_name("aibots.calc", "1.2.0") == "tool:aibots.calc:1.2.0"

    def test_version_preserved_verbatim(self) -> None:
        """semver, pre-release suffix, and plain labels survive unchanged."""
        assert (
            _tool_namespace_name("platform.x", "2.0.0-rc.1")
            == "tool:platform.x:2.0.0-rc.1"
        )
