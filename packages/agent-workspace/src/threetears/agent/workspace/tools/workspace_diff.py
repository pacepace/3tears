"""``threetears.workspace.diff`` -- unified-diff between two refs of a file.

refs follow the ``_resolve_ref`` vocabulary: ``"head"``, integer version,
digit-string version, or checkpoint label. the ref-pair is resolved
inside a single acquired connection so both rows come from a consistent
read snapshot; the connection is not kept in a transaction (diff is
read-only). both blobs must decode as UTF-8 -- diff on binary is not
supported and returns a clean error text the LLM can act on.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable
from typing import Any
from uuid import UUID

from threetears.agent.acl import AclCache
from threetears.agent.tools.base_tool import (
    MCPToolDefinition,
    TearsTool,
    ToolResult,
)
from threetears.agent.tools.context import ToolContextManager
from threetears.core.security import SandboxDenied
from threetears.observe import get_logger

from threetears.agent.workspace.authorize import (
    WorkspaceAccessDenied,
)
from threetears.agent.workspace.collections import (
    WorkspaceCollection,
    WorkspaceFileCollection,
    WorkspaceFileVersionCollection,
)
from threetears.agent.workspace.factory import register_tool_builder
from threetears.agent.workspace.sandbox import WorkspaceSandbox
from threetears.agent.workspace.tools.helpers import (
    NoWorkspacePinned,
    WorkspaceNotFound,
    _resolve_ref,
    _resolve_workspace,
    authorize_workspace,
)

__all__ = [
    "WorkspaceDiffTool",
]

log = get_logger(__name__)


_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "relative_path": {
            "type": "string",
            "description": "workspace-relative path to diff",
        },
        "from_ref": {
            "type": ["string", "integer"],
            "description": ("origin ref: 'head', integer version, or checkpoint label"),
        },
        "to_ref": {
            "type": ["string", "integer"],
            "description": ("target ref: 'head', integer version, or checkpoint label"),
        },
        "workspace": {
            "type": "string",
            "description": "workspace name; defaults to pinned workspace",
        },
    },
    "required": ["relative_path", "from_ref", "to_ref"],
    "additionalProperties": False,
}


class WorkspaceDiffTool(TearsTool):
    """emit a unified diff between two refs of a workspace file.

    resolves workspace, enforces sandbox read, resolves both refs via
    :func:`_resolve_ref` under a single acquired connection, decodes
    both contents as UTF-8, and returns the plain :func:`difflib.
    unified_diff` text. no diff flags are applied (plain unified form
    is the least-surprise result for an LLM).
    """

    def __init__(
        self,
        workspace_collection: WorkspaceCollection,
        workspace_file_collection: WorkspaceFileCollection,
        workspace_file_version_collection: WorkspaceFileVersionCollection,
        sandbox: WorkspaceSandbox,
        context_provider: Callable[[], ToolContextManager],
        agent_id: UUID,
        db_pool: Any,
        acl_cache: AclCache,
    ) -> None:
        """
        binds tool to collections, sandbox, context, agent, and pool.

        :param workspace_collection: collection providing workspace resolution
        :ptype workspace_collection: WorkspaceCollection
        :param workspace_file_collection: accepted for dep symmetry; diff uses
            the journal collection and a pool connection directly
        :ptype workspace_file_collection: WorkspaceFileCollection
        :param workspace_file_version_collection: journal collection (retained
            for symmetry and future optimizations)
        :ptype workspace_file_version_collection: WorkspaceFileVersionCollection
        :param sandbox: workspace sandbox for per-path read enforcement
        :ptype sandbox: WorkspaceSandbox
        :param context_provider: zero-arg callable returning conversation context
        :ptype context_provider: Callable[[], ToolContextManager]
        :param agent_id: identifier of agent owning workspace
        :ptype agent_id: UUID
        :param db_pool: asyncpg pool supplying acquire for the read snapshot
        :ptype db_pool: Any
        """
        self._workspaces = workspace_collection
        self._files = workspace_file_collection
        self._versions = workspace_file_version_collection
        self._sandbox = sandbox
        self._context_provider = context_provider
        self._agent_id = agent_id
        self._db_pool = db_pool
        self._acl_cache = acl_cache

    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        produce unified diff between ``from_ref`` and ``to_ref``.

        all failures arrive as :class:`ToolResult` with ``success=False``;
        this method never raises.

        :param kwargs: must include ``relative_path``, ``from_ref``, and
            ``to_ref``; optional ``workspace``
        :ptype kwargs: Any
        :return: tool result with diff text or clean error
        :rtype: ToolResult
        """
        relative_path = kwargs.get("relative_path", "")
        from_ref = kwargs.get("from_ref")
        to_ref = kwargs.get("to_ref")
        workspace_arg = kwargs.get("workspace")

        # guard clause (entry-time input validation) per CLAUDE.md rule.
        if from_ref is None or to_ref is None:
            return ToolResult(
                success=False,
                content="",
                error="from_ref and to_ref are required",
            )

        result: ToolResult
        try:
            workspace = await _resolve_workspace(
                workspace_arg,
                self._context_provider(),
                self._workspaces,
                self._agent_id,
            )
            await authorize_workspace(
                workspace,
                "read",
                db_pool=self._db_pool,
                acl_cache=self._acl_cache,
            )
            self._sandbox.enforce("read", relative_path)
            # WS-ACL-06: thread namespace= so outside-tx reads resolve
            # against the owner agent's schema on grantee diffs.
            async with self._db_pool.acquire() as conn:
                from_row = await _resolve_ref(
                    conn,
                    workspace.id,
                    relative_path,
                    from_ref,
                    namespace_name=workspace.namespace_name,
                )
                to_row = await _resolve_ref(
                    conn,
                    workspace.id,
                    relative_path,
                    to_ref,
                    namespace_name=workspace.namespace_name,
                )
            if from_row is None:
                result = ToolResult(
                    success=False,
                    content="",
                    error=(f"ref {from_ref!r} not found for {relative_path!r}"),
                )
            elif to_row is None:
                result = ToolResult(
                    success=False,
                    content="",
                    error=(f"ref {to_ref!r} not found for {relative_path!r}"),
                )
            else:
                try:
                    from_text = from_row["content"].decode("utf-8")
                    to_text = to_row["content"].decode("utf-8")
                except UnicodeDecodeError:
                    result = ToolResult(
                        success=False,
                        content="",
                        error="diff only supported on text files",
                    )
                else:
                    diff_text = "".join(
                        difflib.unified_diff(
                            from_text.splitlines(keepends=True),
                            to_text.splitlines(keepends=True),
                            fromfile=f"{relative_path}@{from_ref}",
                            tofile=f"{relative_path}@{to_ref}",
                            n=3,
                        )
                    )
                    result = ToolResult(success=True, content=diff_text)
        except (WorkspaceNotFound, NoWorkspacePinned) as exc:
            result = ToolResult(success=False, content="", error=str(exc))
        except WorkspaceAccessDenied as exc:
            result = ToolResult(success=False, content="", error=str(exc))
        except SandboxDenied as exc:
            result = ToolResult(success=False, content="", error=str(exc))
        except Exception as exc:
            log.exception("workspace_diff failed: %s", exc)
            result = ToolResult(
                success=False,
                content="",
                error=f"diff failed: {exc}",
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
                "return a unified diff between two refs of a workspace file; "
                "refs may be 'head', integer version, or checkpoint label"
            ),
            input_schema=_INPUT_SCHEMA,
        )

    def mcp_name(self) -> str:
        """
        returns the namespaced tool name advertised to MCP clients.

        :return: tool name
        :rtype: str
        """
        return "threetears.workspace.diff"

    def mcp_version(self) -> str:
        """
        returns the semver-compatible tool version.

        :return: version string
        :rtype: str
        """
        return "1.0"


def _build(**kwargs: Any) -> WorkspaceDiffTool:
    """
    constructs a :class:`WorkspaceDiffTool` from the factory dep bundle.

    consumes workspace, file, and journal collections, ``sandbox``,
    ``context_provider``, ``agent_id``, and ``db_pool``; ignores the
    rest.

    :param kwargs: full factory dependency bundle
    :ptype kwargs: Any
    :return: constructed tool
    :rtype: WorkspaceDiffTool
    """
    return WorkspaceDiffTool(
        workspace_collection=kwargs["workspace_collection"],
        workspace_file_collection=kwargs["workspace_file_collection"],
        workspace_file_version_collection=kwargs["workspace_file_version_collection"],
        sandbox=kwargs["sandbox"],
        context_provider=kwargs["context_provider"],
        agent_id=kwargs["agent_id"],
        db_pool=kwargs["db_pool"],
        acl_cache=kwargs["acl_cache"],
    )


register_tool_builder(_build)
