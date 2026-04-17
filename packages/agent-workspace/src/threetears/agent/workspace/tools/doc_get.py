"""``threetears.workspace.doc_get`` -- structural read of a workspace document.

dispatches a :class:`threetears.core.serialization.FormatHandler` by file
extension (YAML today; JSON/TOML/.env when future handlers register),
loads the document tree, and returns either the whole dumped document or
the subtree addressed by an optional ``jsonpath`` expression. for files
without a registered handler -- plain text, SQL, binaries -- returns a
clean error that names the suffix and suggests ``fs_*`` tools.

this tool is read-only. sandbox ``read`` enforcement gates the call
before the database fetch so denied paths never touch the pool.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import UUID

from threetears.agent.tools.base_tool import (
    MCPToolDefinition,
    TearsTool,
    ToolResult,
)
from threetears.agent.tools.context import ToolContextManager
from threetears.core.security import SandboxDenied
from threetears.core.serialization import UnknownFormatError, handler_for
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

log = get_logger(__name__)


_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "relative_path": {
            "type": "string",
            "description": "workspace-relative path of the document to read",
        },
        "jsonpath": {
            "type": "string",
            "description": (
                "optional jsonpath expression; when supplied, returns the "
                "subtree or scalar at that path instead of the whole document"
            ),
        },
        "workspace": {
            "type": "string",
            "description": "workspace name; defaults to pinned workspace",
        },
    },
    "required": ["relative_path"],
    "additionalProperties": False,
}


class DocGetTool(TearsTool):
    """read a workspace document's parsed structure or a jsonpath subtree.

    resolves workspace, enforces sandbox read, fetches the head-state row,
    decodes UTF-8, then asks the format handler registered for the file's
    suffix to load the tree. when ``jsonpath`` is supplied, the handler
    resolves the expression and the result is serialized; otherwise the
    whole tree is dumped back through the handler so the caller sees the
    canonical format text. this tool never mutates the tree.
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
        return the document at ``relative_path`` or its subtree at ``jsonpath``.

        all failures arrive as :class:`ToolResult` with ``success=False``;
        this method never raises.

        :param kwargs: must include ``relative_path``; optional ``jsonpath``
            and ``workspace``
        :ptype kwargs: Any
        :return: tool result with serialized document or subtree, plus metadata
        :rtype: ToolResult
        """
        relative_path = kwargs.get("relative_path", "")
        jsonpath = kwargs.get("jsonpath")
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
            try:
                handler = handler_for(relative_path)
            except UnknownFormatError:
                suffix = Path(relative_path).suffix
                result = ToolResult(
                    success=False,
                    content="",
                    error=(
                        f"no FormatHandler for {suffix}; use fs_* tools for "
                        "this file type"
                    ),
                )
            else:
                file_entity = await self._files.find_by_workspace_and_relative_path(
                    workspace.id, relative_path
                )
                if file_entity is None:
                    result = ToolResult(
                        success=False,
                        content="",
                        error=(
                            f"file {relative_path!r} not found in workspace "
                            f"{workspace.name!r}"
                        ),
                    )
                else:
                    try:
                        text = file_entity.content.decode("utf-8")
                    except UnicodeDecodeError:
                        result = ToolResult(
                            success=False,
                            content="",
                            error="doc_get requires text file; got binary",
                        )
                    else:
                        tree = handler.load(text)
                        if jsonpath is not None and jsonpath != "":
                            subtree = handler.get(tree, jsonpath)
                            if subtree is None:
                                result = ToolResult(
                                    success=False,
                                    content="",
                                    error=(
                                        f"jsonpath {jsonpath!r} returned no matches"
                                    ),
                                )
                                payload = None
                            else:
                                payload = subtree
                        else:
                            payload = tree
                        if payload is not None:
                            serialized: str
                            if isinstance(payload, (dict, list)):
                                serialized = handler.dump(payload)
                            else:
                                serialized = json.dumps(payload, default=str)
                            result = ToolResult(
                                success=True,
                                content=serialized,
                                metadata={
                                    "sha256": file_entity.sha256,
                                    "format": handler.extensions[0],
                                },
                            )
        except (WorkspaceNotFound, NoWorkspacePinned) as exc:
            result = ToolResult(success=False, content="", error=str(exc))
        except SandboxDenied as exc:
            result = ToolResult(success=False, content="", error=str(exc))
        except Exception as exc:
            log.exception("doc_get failed: %s", exc)
            result = ToolResult(
                success=False,
                content="",
                error=f"doc_get failed: {exc}",
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
            description=(
                "read a workspace document's parsed structure, optionally "
                "narrowed to a jsonpath subtree; dispatches by file suffix "
                "through the FormatHandler registry"
            ),
            input_schema=_INPUT_SCHEMA,
        )

    def mcp_name(self) -> str:
        """
        returns the namespaced tool name advertised to MCP clients.

        :return: tool name
        :rtype: str
        """
        return "threetears.workspace.doc_get"

    def mcp_version(self) -> str:
        """
        returns the semver-compatible tool version.

        :return: version string
        :rtype: str
        """
        return "1.0"


def _build(**kwargs: Any) -> DocGetTool:
    """
    constructs a :class:`DocGetTool` from the factory dep bundle.

    consumes ``workspace_collection``, ``workspace_file_collection``,
    ``sandbox``, ``context_provider``, and ``agent_id``; ignores the
    rest. registered with :mod:`threetears.agent.workspace.factory` on
    import so :func:`build_workspace_tools` emits this tool.

    :param kwargs: full factory dependency bundle
    :ptype kwargs: Any
    :return: constructed tool
    :rtype: DocGetTool
    """
    return DocGetTool(
        workspace_collection=kwargs["workspace_collection"],
        workspace_file_collection=kwargs["workspace_file_collection"],
        sandbox=kwargs["sandbox"],
        context_provider=kwargs["context_provider"],
        agent_id=kwargs["agent_id"],
    )


register_tool_builder(_build)
