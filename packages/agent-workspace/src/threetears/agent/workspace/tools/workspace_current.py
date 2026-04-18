"""``threetears.workspace.current`` -- report the conversation's pinned workspace."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from threetears.agent.tools.base_tool import (
    MCPToolDefinition,
    TearsTool,
    ToolResult,
)
from threetears.agent.tools.context import ToolContextManager
from threetears.observe import get_logger

from threetears.agent.workspace import pin
from threetears.agent.workspace.factory import register_tool_builder

__all__ = [
    "WorkspaceCurrentTool",
]

log = get_logger(__name__)


_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


class WorkspaceCurrentTool(TearsTool):
    """report which workspace is pinned to the current conversation.

    pin-only operation: reads :func:`pin.get_pin` and serializes the
    snapshot to JSON. when no pin is set, returns a structured response
    that includes a recovery hint pointing to ``workspace.use``.
    """

    def __init__(
        self,
        context_provider: Callable[[], ToolContextManager],
    ) -> None:
        """
        binds tool to a per-conversation context provider.

        :param context_provider: zero-arg callable returning the current
            conversation's ToolContextManager
        :ptype context_provider: Callable[[], ToolContextManager]
        """
        self._context_provider = context_provider

    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        returns the conversation's pin snapshot or a null-pin response.

        on hit: ``{"workspace_id", "workspace_name", "date_pinned"}``
        encoded as JSON. on miss: ``{"pin": null, "message": "..."}``
        with a hint to call ``workspace.use``. all failures arrive as
        :class:`ToolResult` with ``success=False`` and never raise.

        :param kwargs: ignored, schema declares no inputs
        :ptype kwargs: Any
        :return: tool result with JSON content or error message
        :rtype: ToolResult
        """
        result: ToolResult
        try:
            snapshot = await pin.get_pin(self._context_provider())
            if snapshot is None:
                payload: dict[str, Any] = {
                    "pin": None,
                    "message": "no workspace pinned; call workspace.use(name) to set",
                }
                result = ToolResult(success=True, content=json.dumps(payload))
            else:
                payload = {
                    "workspace_id": str(snapshot.workspace_id),
                    "workspace_name": snapshot.workspace_name,
                    "date_pinned": snapshot.date_pinned.isoformat(),
                }
                result = ToolResult(success=True, content=json.dumps(payload))
        except Exception as exc:
            log.exception("workspace_current failed: %s", exc)
            result = ToolResult(
                success=False,
                content="",
                error=f"current failed: {exc}",
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
            description="return the workspace pinned to the current conversation",
            input_schema=_INPUT_SCHEMA,
        )

    def mcp_name(self) -> str:
        """
        returns the namespaced tool name advertised to MCP clients.

        :return: tool name
        :rtype: str
        """
        return "threetears.workspace.current"

    def mcp_version(self) -> str:
        """
        returns the semver-compatible tool version.

        :return: version string
        :rtype: str
        """
        return "1.0"


def _build(**kwargs: Any) -> WorkspaceCurrentTool:
    """
    constructs a :class:`WorkspaceCurrentTool` from the factory dep bundle.

    consumes only ``context_provider`` and ignores the rest. registered
    with :mod:`threetears.agent.workspace.factory` on import so
    :func:`build_workspace_tools` emits this tool.

    :param kwargs: full factory dependency bundle
    :ptype kwargs: Any
    :return: constructed tool
    :rtype: WorkspaceCurrentTool
    """
    return WorkspaceCurrentTool(
        context_provider=kwargs["context_provider"],
    )


register_tool_builder(_build)
