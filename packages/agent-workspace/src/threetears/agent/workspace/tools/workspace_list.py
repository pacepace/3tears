"""``threetears.workspace.list`` -- enumerate workspaces for an agent."""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from threetears.agent.tools.base_tool import (
    MCPToolDefinition,
    TearsTool,
    ToolResult,
)
from threetears.observe import get_logger

from threetears.agent.workspace.collections import WorkspaceCollection
from threetears.agent.workspace.factory import register_tool_builder

log = get_logger(__name__)


_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


class WorkspaceListTool(TearsTool):
    """list workspaces owned by the bound agent.

    returns a JSON-encoded array of ``{name, description, date_updated}``
    entries newest-update first. names are the LLM-facing handle for
    every other workspace tool; UUIDs never leave the implementation.
    """

    def __init__(
        self,
        workspace_collection: WorkspaceCollection,
        agent_id: UUID,
    ) -> None:
        """
        binds tool to a workspace collection scoped to one agent.

        :param workspace_collection: collection providing find_by_agent
        :ptype workspace_collection: WorkspaceCollection
        :param agent_id: identifier of agent whose workspaces to list
        :ptype agent_id: UUID
        """
        self._workspaces = workspace_collection
        self._agent_id = agent_id

    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        returns workspaces for the bound agent as JSON-encoded array.

        empty agent yields ``"[]"``; storage failures surface as errors-
        as-data via :class:`ToolResult` rather than raising.

        :param kwargs: ignored, schema declares no inputs
        :ptype kwargs: Any
        :return: tool result with JSON content or error message
        :rtype: ToolResult
        """
        result: ToolResult
        try:
            entities = await self._workspaces.find_by_agent(self._agent_id)
            payload = [
                {
                    "name": entity.name,
                    "description": entity.description or "",
                    "date_updated": entity.date_updated.isoformat(),
                }
                for entity in entities
            ]
            result = ToolResult(success=True, content=json.dumps(payload))
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
            description="list workspaces for the agent",
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

    consumes only ``workspace_collection`` and ``agent_id`` and ignores
    the rest. registered with :mod:`threetears.agent.workspace.factory`
    on import so :func:`build_workspace_tools` emits this tool.

    :param kwargs: full factory dependency bundle
    :ptype kwargs: Any
    :return: constructed tool
    :rtype: WorkspaceListTool
    """
    return WorkspaceListTool(
        workspace_collection=kwargs["workspace_collection"],
        agent_id=kwargs["agent_id"],
    )


register_tool_builder(_build)
