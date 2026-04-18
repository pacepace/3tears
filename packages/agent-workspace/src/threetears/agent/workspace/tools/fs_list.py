"""``threetears.workspace.fs_list`` -- list head-state files in a workspace.

returns a JSON array of ``{relative_path, sha256, version, date_updated}``
entries. an optional ``glob`` filters entries via
:meth:`pathlib.PurePosixPath.full_match` so ``**`` anchors recursively
and matches the semantics used by the YAML handler and sandbox. sandbox
read enforcement filters the result -- entries the sandbox denies are
omitted, making ``fs_list`` honor the agent's ``allow.read`` globs
naturally without surfacing a denial for the caller.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import PurePosixPath
from typing import Any
from uuid import UUID

from threetears.agent.tools.base_tool import (
    MCPToolDefinition,
    TearsTool,
    ToolResult,
)
from threetears.agent.tools.context import ToolContextManager
from threetears.core.security import SandboxDecision
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
    "FsListTool",
]

log = get_logger(__name__)


_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "glob": {
            "type": "string",
            "description": (
                "optional posix glob pattern; matches use PurePosixPath.full_match so ** anchors recursively"
            ),
        },
        "workspace": {
            "type": "string",
            "description": "workspace name; defaults to pinned workspace",
        },
    },
    "required": [],
    "additionalProperties": False,
}


class FsListTool(TearsTool):
    """list head-state files in a workspace, filtered by glob + sandbox.

    resolves the workspace from ``workspace`` kwarg or pin, fetches all
    head-state rows, optionally filters by a posix glob, drops rows the
    sandbox denies on ``read``, and returns a JSON list for the LLM.
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
        binds tool to collections, sandbox, conversation context, and agent.

        :param workspace_collection: collection providing workspace resolution
        :ptype workspace_collection: WorkspaceCollection
        :param workspace_file_collection: collection providing
            ``find_by_workspace``
        :ptype workspace_file_collection: WorkspaceFileCollection
        :param sandbox: workspace sandbox filtering result by read globs
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
        list workspace files; optional ``glob`` filters, sandbox-read filters.

        all failures arrive as :class:`ToolResult` with ``success=False``;
        this method never raises.

        :param kwargs: optional ``glob`` and ``workspace``
        :ptype kwargs: Any
        :return: tool result with JSON-encoded file-entry list
        :rtype: ToolResult
        """
        glob_pattern = kwargs.get("glob")
        workspace_arg = kwargs.get("workspace")

        result: ToolResult
        try:
            workspace = await _resolve_workspace(
                workspace_arg,
                self._context_provider(),
                self._workspaces,
                self._agent_id,
            )
            rows = await self._files.find_by_workspace(workspace.id)
            filtered = self._filter(rows, glob_pattern)
            entries = [
                {
                    "relative_path": f.relative_path,
                    "sha256": f.sha256,
                    "version": f.version,
                    "date_updated": f.date_updated.isoformat(),
                }
                for f in filtered
            ]
            result = ToolResult(
                success=True,
                content=json.dumps(entries),
                metadata={"count": len(entries)},
            )
        except (WorkspaceNotFound, NoWorkspacePinned) as exc:
            result = ToolResult(success=False, content="", error=str(exc))
        except Exception as exc:
            log.exception("fs_list failed: %s", exc)
            result = ToolResult(
                success=False,
                content="",
                error=f"fs_list failed: {exc}",
            )
        return result

    def _filter(self, rows: list[Any], glob_pattern: str | None) -> list[Any]:
        """
        apply optional glob and sandbox read check to head-state rows.

        sandbox filtering uses :meth:`check_relative_key` rather than
        :meth:`enforce` so a denied entry silently drops from the list
        instead of raising (the agent expectation is that ``list`` shows
        what it is allowed to see, not that ``list`` errors on first
        inaccessible entry).

        :param rows: head-state file entities from the collection
        :ptype rows: list[Any]
        :param glob_pattern: optional posix glob applied via full_match
        :ptype glob_pattern: str | None
        :return: filtered file list
        :rtype: list[Any]
        """
        result: list[Any] = []
        for row in rows:
            posix_path = PurePosixPath(row.relative_path)
            if glob_pattern is not None and not posix_path.full_match(glob_pattern):
                continue
            decision = self._sandbox.check_relative_key(row.relative_path, "read")
            if decision is SandboxDecision.DENY:
                continue
            result.append(row)
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
            description=(
                "list head-state files in a workspace, optionally filtered "
                "by posix glob and always filtered by sandbox read policy"
            ),
            input_schema=_INPUT_SCHEMA,
        )

    def mcp_name(self) -> str:
        """
        returns the namespaced tool name advertised to MCP clients.

        :return: tool name
        :rtype: str
        """
        return "threetears.workspace.fs_list"

    def mcp_version(self) -> str:
        """
        returns the semver-compatible tool version.

        :return: version string
        :rtype: str
        """
        return "1.0"


def _build(**kwargs: Any) -> FsListTool:
    """
    constructs an :class:`FsListTool` from the factory dep bundle.

    consumes ``workspace_collection``, ``workspace_file_collection``,
    ``sandbox``, ``context_provider``, and ``agent_id``; ignores the
    rest. registered with :mod:`threetears.agent.workspace.factory` on
    import so :func:`build_workspace_tools` emits this tool.

    :param kwargs: full factory dependency bundle
    :ptype kwargs: Any
    :return: constructed tool
    :rtype: FsListTool
    """
    return FsListTool(
        workspace_collection=kwargs["workspace_collection"],
        workspace_file_collection=kwargs["workspace_file_collection"],
        sandbox=kwargs["sandbox"],
        context_provider=kwargs["context_provider"],
        agent_id=kwargs["agent_id"],
    )


register_tool_builder(_build)
