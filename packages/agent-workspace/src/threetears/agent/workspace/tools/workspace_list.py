"""``threetears.workspace.list`` -- enumerate workspaces via namespace discovery.

workspace-task-19 Phase 5 replaced the per-agent-schema SELECT with a
NATS request to the broker's workspace-discovery subject; namespace-
task-01 Phase 1 generalized that subject to
``{ns}.namespace.discover`` with a ``namespace_type`` filter on the
request. this tool now asks for ``namespace_type="workspace"``
explicitly.

discovery returns every namespace the caller can see in their customer
-- workspaces the calling agent owns plus workspaces granted to that
agent within the customer -- so cross-agent sharing surfaces naturally
in the list UI without any special-case branching here.

when discovery is unavailable (NATS not wired, broker down), the tool
surfaces the failure as errors-as-data per the TearsTool contract
rather than raising.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID, uuid7

from threetears.agent.tools.base_tool import (
    MCPToolDefinition,
    TearsTool,
    ToolResult,
)
from threetears.agent.tools.call_scope import current_scope
from threetears.observe import get_logger

from threetears.agent.workspace.discovery_client import (
    DiscoveryClientError,
    NamespaceDiscoveryClient,
)
from threetears.agent.workspace.factory import register_tool_builder

__all__ = [
    "WorkspaceListTool",
]

log = get_logger(__name__)


_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


class WorkspaceListTool(TearsTool):
    """list workspaces the caller can see via broker namespace discovery.

    returns a JSON-encoded array of
    ``{name, owner_agent_id, customer_id}`` entries, newest-update
    first. the ``name`` is the canonical namespace name
    (``workspace.<uuid>``) so subsequent tool calls can either quote the
    raw form or extract the uuid suffix and pass it as the workspace
    argument; tools accept either via :func:`_resolve_workspace`.
    """

    def __init__(
        self,
        discovery_client: NamespaceDiscoveryClient,
        agent_id: UUID,
    ) -> None:
        """
        binds tool to a discovery client and the owning agent.

        :param discovery_client: NATS client for ``namespace.discover``
        :ptype discovery_client: NamespaceDiscoveryClient
        :param agent_id: identifier of agent issuing discovery
        :ptype agent_id: UUID
        """
        self._discovery = discovery_client
        self._agent_id = agent_id

    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        issue ``namespace.discover`` for ``workspace`` type and return JSON.

        reads the caller's customer_id + user_id from the current
        :class:`ToolCallScope` so the broker can filter the discovery
        set to the caller's customer and honor user-scoped grants.
        missing scope (or missing customer on the scope) is treated as
        an unroutable call and surfaces as errors-as-data.

        :param kwargs: ignored, schema declares no inputs
        :ptype kwargs: Any
        :return: tool result with JSON array or error message
        :rtype: ToolResult
        """
        result: ToolResult
        scope = current_scope()
        customer_id: UUID | None = None if scope is None else scope.context.customer_id
        user_id: UUID | None = None if scope is None else scope.context.user_id
        correlation_id: UUID = (
            scope.context.correlation_id
            if scope is not None and scope.context.correlation_id is not None
            else uuid7()
        )
        try:
            if customer_id is None:
                result = ToolResult(
                    success=False,
                    content="",
                    error="workspace.list requires a customer_id on the call scope",
                )
            else:
                items = await self._discovery.discover(
                    correlation_id=correlation_id,
                    agent_id=self._agent_id,
                    customer_id=customer_id,
                    user_id=user_id,
                    namespace_type="workspace",
                )
                payload = [
                    {
                        "name": item.name,
                        "owner_agent_id": str(item.owner_agent_id),
                        "customer_id": str(item.customer_id),
                    }
                    for item in items
                ]
                result = ToolResult(success=True, content=json.dumps(payload))
        except DiscoveryClientError as exc:
            result = ToolResult(
                success=False,
                content="",
                error=f"list failed: {exc}",
            )
        except Exception as exc:
            log.exception("workspace_list failed: %s", exc)
            result = ToolResult(
                success=False,
                content="",
                error=f"list failed: {exc}",
            )
        return result

    def mcp_schema(self) -> MCPToolDefinition:
        """
        returns the MCP definition for this tool.

        pure: no side effects, safe for repeated discovery calls.

        :return: MCP-compatible tool definition
        :rtype: MCPToolDefinition
        """
        return MCPToolDefinition(
            name=self.mcp_name(),
            version=self.mcp_version(),
            description="list workspaces the caller can see (owned + granted)",
            input_schema=_INPUT_SCHEMA,
        )

    def mcp_name(self) -> str:
        """
        returns the namespaced tool name advertised to MCP clients.

        :return: tool name
        :rtype: str
        """
        return "threetears.workspace.list"

    def mcp_version(self) -> str:
        """
        returns the semver-compatible tool version.

        :return: version string
        :rtype: str
        """
        return "1.0"


def _build(**kwargs: Any) -> WorkspaceListTool:
    """
    constructs a :class:`WorkspaceListTool` from the factory dep bundle.

    consumes ``nats_client``, ``namespace``, and ``agent_id`` to build
    a :class:`NamespaceDiscoveryClient`; ignores the rest. registered
    with :mod:`threetears.agent.workspace.factory` on import so
    :func:`build_workspace_tools` emits this tool.

    :param kwargs: full factory dependency bundle
    :ptype kwargs: Any
    :return: constructed tool
    :rtype: WorkspaceListTool
    """
    client = NamespaceDiscoveryClient(
        nats_client=kwargs.get("nats_client"),
        namespace=kwargs.get("namespace") or "",
    )
    return WorkspaceListTool(
        discovery_client=client,
        agent_id=kwargs["agent_id"],
    )


register_tool_builder(_build)
