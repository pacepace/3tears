"""``threetears.workspace.fs_read`` -- read a workspace file's content + sha256.

returns raw bytes decoded as UTF-8 when possible; otherwise emits base64
and sets ``is_binary=True`` in metadata so the LLM caller can detect a
binary file and choose an appropriate handler. sandbox read-enforcement
gates the call before any database read; the sha256 and version come
straight off the head-state row (never recomputed).
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from typing import Any
from uuid import UUID

from threetears.agent.tools.base_tool import (
    MCPToolDefinition,
    TearsTool,
    ToolResult,
)
from threetears.agent.tools.context import ToolContextManager
from threetears.core.security import SandboxDenied
from threetears.observe import get_logger

from threetears.agent.workspace.collections import (
    WorkspaceCollection,
    WorkspaceFileCollection,
)
from threetears.agent.workspace.factory import register_tool_builder
from threetears.agent.workspace.sandbox import WorkspaceSandbox
from threetears.agent.workspace.tools.helpers import (
    NoWorkspacePinned,
    WorkspaceNotFound,
    _resolve_workspace,
)

__all__ = [
    "FsReadTool",
]

log = get_logger(__name__)


_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "relative_path": {
            "type": "string",
            "description": "workspace-relative path of the file to read",
        },
        "workspace": {
            "type": "string",
            "description": "workspace name; defaults to pinned workspace",
        },
    },
    "required": ["relative_path"],
    "additionalProperties": False,
}


class FsReadTool(TearsTool):
    """read a workspace file's content, sha256, and version.

    resolves the workspace from the ``workspace`` kwarg or the current
    conversation pin, enforces sandbox read on the path, then fetches
    the head-state row. returns UTF-8 text when decodable, else base64
    with ``is_binary=True``.
    """

    def __init__(
        self,
        workspace_collection: WorkspaceCollection,
        workspace_file_collection: WorkspaceFileCollection,
        sandbox: WorkspaceSandbox,
        context_provider: Callable[[], ToolContextManager],
        agent_id: UUID,
    ) -> None:
        """
        binds tool to workspace + file collections, sandbox, context, agent.

        :param workspace_collection: collection providing workspace resolution
        :ptype workspace_collection: WorkspaceCollection
        :param workspace_file_collection: collection providing
            ``find_by_workspace_and_relative_path``
        :ptype workspace_file_collection: WorkspaceFileCollection
        :param sandbox: workspace sandbox enforcing read globs
        :ptype sandbox: WorkspaceSandbox
        :param context_provider: zero-arg callable returning conversation context
        :ptype context_provider: Callable[[], ToolContextManager]
        :param agent_id: identifier of agent owning workspace
        :ptype agent_id: UUID
        """
        self._workspaces = workspace_collection
        self._files = workspace_file_collection
        self._sandbox = sandbox
        self._context_provider = context_provider
        self._agent_id = agent_id

    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        read file at ``relative_path`` from resolved workspace.

        all failures arrive as :class:`ToolResult` with ``success=False``;
        this method never raises.

        :param kwargs: must include ``relative_path``; optional ``workspace``
        :ptype kwargs: Any
        :return: tool result with content (text or base64) and metadata
        :rtype: ToolResult
        """
        relative_path = kwargs.get("relative_path", "")
        workspace_arg = kwargs.get("workspace")

        result: ToolResult
        try:
            workspace = await _resolve_workspace(
                workspace_arg,
                self._context_provider(),
                self._workspaces,
                self._agent_id,
            )
            self._sandbox.enforce("read", relative_path)
            file_entity = await self._files.find_by_workspace_and_relative_path(workspace.id, relative_path)
            if file_entity is None:
                result = ToolResult(
                    success=False,
                    content="",
                    error=(f"file {relative_path!r} not found in workspace {workspace.name!r}"),
                )
            else:
                content_bytes = file_entity.content
                try:
                    decoded = content_bytes.decode("utf-8")
                    is_binary = False
                except UnicodeDecodeError:
                    decoded = base64.b64encode(content_bytes).decode("ascii")
                    is_binary = True
                result = ToolResult(
                    success=True,
                    content=decoded,
                    metadata={
                        "sha256": file_entity.sha256,
                        "version": file_entity.version,
                        "is_binary": is_binary,
                    },
                )
        except (WorkspaceNotFound, NoWorkspacePinned) as exc:
            result = ToolResult(success=False, content="", error=str(exc))
        except SandboxDenied as exc:
            result = ToolResult(success=False, content="", error=str(exc))
        except Exception as exc:
            log.exception("fs_read failed: %s", exc)
            result = ToolResult(
                success=False,
                content="",
                error=f"fs_read failed: {exc}",
            )
        return result

    def mcp_schema(self) -> MCPToolDefinition:
        """
        returns the MCP definition for this tool.

        :return: MCP-compatible tool definition
        :rtype: MCPToolDefinition
        """
        return MCPToolDefinition(
            name=self.mcp_name(),
            version=self.mcp_version(),
            description="read a workspace file's content and sha256",
            input_schema=_INPUT_SCHEMA,
        )

    def mcp_name(self) -> str:
        """
        returns the namespaced tool name advertised to MCP clients.

        :return: tool name
        :rtype: str
        """
        return "threetears.workspace.fs_read"

    def mcp_version(self) -> str:
        """
        returns the semver-compatible tool version.

        :return: version string
        :rtype: str
        """
        return "1.0"


def _build(**kwargs: Any) -> FsReadTool:
    """
    constructs an :class:`FsReadTool` from the factory dep bundle.

    consumes ``workspace_collection``, ``workspace_file_collection``,
    ``sandbox``, ``context_provider``, and ``agent_id``; ignores the
    rest. registered with :mod:`threetears.agent.workspace.factory` on
    import so :func:`build_workspace_tools` emits this tool.

    :param kwargs: full factory dependency bundle
    :ptype kwargs: Any
    :return: constructed tool
    :rtype: FsReadTool
    """
    return FsReadTool(
        workspace_collection=kwargs["workspace_collection"],
        workspace_file_collection=kwargs["workspace_file_collection"],
        sandbox=kwargs["sandbox"],
        context_provider=kwargs["context_provider"],
        agent_id=kwargs["agent_id"],
    )


register_tool_builder(_build)
