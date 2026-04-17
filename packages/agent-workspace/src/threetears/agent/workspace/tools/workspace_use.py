"""``threetears.workspace.use`` -- pin a workspace by name for the conversation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import UUID

from threetears.agent.tools.base_tool import (
    MCPToolDefinition,
    TearsTool,
    ToolResult,
)
from threetears.agent.tools.context import ToolContextManager
from threetears.observe import get_logger

from threetears.agent.workspace import pin
from threetears.agent.workspace.collections import WorkspaceCollection
from threetears.agent.workspace.factory import register_tool_builder

log = get_logger(__name__)


_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "human-readable workspace name to pin",
        },
    },
    "required": ["name"],
    "additionalProperties": False,
}


class WorkspaceUseTool(TearsTool):
    """pin a workspace by name to the current conversation.

    looks up ``(agent_id, name)`` once: on hit, writes the pin via
    :func:`pin.set_pin`; on miss, returns errors-as-data including the
    available workspace names so the caller can recover. resolution is
    single-shot to avoid pin-before-verify races.
    """

    def __init__(
        self,
        workspace_collection: WorkspaceCollection,
        agent_id: UUID,
        context_provider: Callable[[], ToolContextManager],
    ) -> None:
        """
        binds tool to a workspace collection and per-conversation context.

        :param workspace_collection: collection providing find_by_agent
            and find_by_agent_and_name
        :ptype workspace_collection: WorkspaceCollection
        :param agent_id: identifier of agent whose workspaces to query
        :ptype agent_id: UUID
        :param context_provider: zero-arg callable returning the current
            conversation's ToolContextManager
        :ptype context_provider: Callable[[], ToolContextManager]
        """
        self._workspaces = workspace_collection
        self._agent_id = agent_id
        self._context_provider = context_provider

    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        pins workspace ``name`` to the current conversation, or errors.

        algorithm:
          1. fetch workspace by ``(agent_id, name)``.
          2. on miss: list all available names and return errors-as-data.
          3. on hit: write pin via :func:`pin.set_pin` and return success.

        all failures arrive at the LLM as :class:`ToolResult` with
        ``success=False``; ``execute`` never raises.

        :param kwargs: must contain ``name``: workspace name to pin
        :ptype kwargs: Any
        :return: tool result reporting pin or error
        :rtype: ToolResult
        """
        name = kwargs.get("name", "")
        result: ToolResult
        try:
            found = await self._workspaces.find_by_agent_and_name(self._agent_id, name)
            if found is None:
                available = await self._workspaces.find_by_agent(self._agent_id)
                names = [entity.name for entity in available]
                result = ToolResult(
                    success=False,
                    content="",
                    error=f"workspace {name!r} not found; available: {names}",
                )
            else:
                await pin.set_pin(
                    self._context_provider(),
                    workspace_id=found.id,
                    workspace_name=name,
                    pinned_by_actor_id=self._agent_id,
                )
                result = ToolResult(
                    success=True,
                    content=f"pinned workspace {name!r}",
                )
        except Exception as exc:
            log.exception("workspace_use failed: %s", exc)
            result = ToolResult(
                success=False,
                content="",
                error=f"use failed: {exc}",
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
            description="pin a workspace by name for the current conversation",
            input_schema=_INPUT_SCHEMA,
        )

    def mcp_name(self) -> str:
        """
        returns the namespaced tool name advertised to MCP clients.

        :return: tool name
        :rtype: str
        """
        return "threetears.workspace.use"

    def mcp_version(self) -> str:
        """
        returns the semver-compatible tool version.

        :return: version string
        :rtype: str
        """
        return "1.0"


def _build(**kwargs: Any) -> WorkspaceUseTool:
    """
    constructs a :class:`WorkspaceUseTool` from the factory dep bundle.

    consumes ``workspace_collection``, ``agent_id``, and
    ``context_provider``; ignores the rest. registered with
    :mod:`threetears.agent.workspace.factory` on import so
    :func:`build_workspace_tools` emits this tool.

    :param kwargs: full factory dependency bundle
    :ptype kwargs: Any
    :return: constructed tool
    :rtype: WorkspaceUseTool
    """
    return WorkspaceUseTool(
        workspace_collection=kwargs["workspace_collection"],
        agent_id=kwargs["agent_id"],
        context_provider=kwargs["context_provider"],
    )


register_tool_builder(_build)
