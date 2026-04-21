"""``threetears.workspace.history`` -- list journal entries for a workspace.

returns a JSON array of journal-row metadata (relative_path, version,
action, label, actor, correlation, date_created, sha256, size_bytes),
newest-first, bounded by ``limit``. content blobs are deliberately
omitted so the payload stays small regardless of file size; callers
invoke ``threetears.workspace.diff`` when they need the actual bytes.

when ``relative_path`` is supplied, sandbox read-enforcement gates the
call (raise-on-deny) and the query narrows to that path. when it is
omitted, the workspace-wide journal is queried and sandbox
:meth:`check_relative_key` is applied per-row to silently drop denied
paths -- history is a "show me what i'm allowed to see" surface, not a
deny-first endpoint, matching ``fs_list`` semantics.
"""

from __future__ import annotations

import json
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
    _resolve_workspace,
    authorize_workspace,
    authorize_workspace_file,
)

__all__ = [
    "WorkspaceHistoryTool",
]

log = get_logger(__name__)


_DEFAULT_LIMIT = 50
_MAX_LIMIT = 500


_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "relative_path": {
            "type": "string",
            "description": (
                "optional workspace-relative path; when omitted, history spans every file in the workspace"
            ),
        },
        "limit": {
            "type": "integer",
            "minimum": 1,
            "maximum": _MAX_LIMIT,
            "description": ("maximum rows to return (newest first); default 50, max 500"),
        },
        "workspace": {
            "type": "string",
            "description": "workspace name; defaults to pinned workspace",
        },
    },
    "required": [],
    "additionalProperties": False,
}


class WorkspaceHistoryTool(TearsTool):
    """list journal entries for a workspace or a single path, newest-first.

    resolves workspace, optionally enforces sandbox read on a single
    path, then queries the append-only journal. returns metadata-only
    rows; the raw content blob is intentionally excluded so history
    responses remain bounded regardless of file size.
    """

    def __init__(
        self,
        workspace_collection: WorkspaceCollection,
        workspace_file_collection: WorkspaceFileCollection,
        workspace_file_version_collection: WorkspaceFileVersionCollection,
        sandbox: WorkspaceSandbox,
        context_provider: Callable[[], ToolContextManager],
        agent_id: UUID,
        acl_cache: AclCache,
        db_pool: Any = None,
    ) -> None:
        """
        binds tool to collections, sandbox, context, and owning agent.

        :param workspace_collection: collection providing workspace resolution
        :ptype workspace_collection: WorkspaceCollection
        :param workspace_file_collection: accepted for dep symmetry; history
            queries the journal collection directly
        :ptype workspace_file_collection: WorkspaceFileCollection
        :param workspace_file_version_collection: journal collection providing
            ``find_by_workspace`` and ``find_by_workspace_and_path``
        :ptype workspace_file_version_collection: WorkspaceFileVersionCollection
        :param sandbox: workspace sandbox for per-path read enforcement and
            result-filtering
        :ptype sandbox: WorkspaceSandbox
        :param context_provider: zero-arg callable returning conversation context
        :ptype context_provider: Callable[[], ToolContextManager]
        :param agent_id: identifier of agent owning workspace
        :ptype agent_id: UUID
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
        emit newest-first metadata rows for the workspace journal.

        all failures arrive as :class:`ToolResult` with ``success=False``;
        this method never raises. rows are serialized as a JSON array so
        the LLM caller can parse deterministically; ``size_bytes`` is
        computed from the in-row content bytes so the payload carries
        size without the blob itself.

        :param kwargs: optional ``relative_path`` (gates sandbox read),
            ``limit`` (default 50, max 500), ``workspace``
        :ptype kwargs: Any
        :return: tool result with JSON-encoded history entries
        :rtype: ToolResult
        """
        relative_path = kwargs.get("relative_path")
        limit_arg = kwargs.get("limit", _DEFAULT_LIMIT)
        workspace_arg = kwargs.get("workspace")

        result: ToolResult
        try:
            limit = self._clamp_limit(limit_arg)
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
            rows: list[Any]
            if relative_path is not None and relative_path != "":
                self._sandbox.validate_syntax(relative_path)
                await authorize_workspace_file(
                    workspace,
                    relative_path,
                    "read",
                    db_pool=None,
                    acl_cache=self._acl_cache,
                )
                rows = await self._versions.find_by_workspace_and_path(workspace.id, relative_path, limit)
            else:
                fetched = await self._versions.find_by_workspace(workspace.id, limit)
                rows = []
                for row in fetched:
                    # filter rows whose relative_path either fails syntactic
                    # validation or lacks a read-file glob matching the caller's
                    # rbac grants. namespace-task-01 phase 7 replaces the
                    # legacy ``sandbox.check_relative_key(...) is ALLOW``
                    # filter with the unified rbac gate; denied rows are
                    # silently dropped (matches the legacy filter's
                    # fail-quietly semantics, which this surface has always
                    # preserved because a partial result is better than
                    # failing the whole history call on one weird row).
                    try:
                        self._sandbox.validate_syntax(row.relative_path)
                        await authorize_workspace_file(
                            workspace,
                            row.relative_path,
                            "read",
                            db_pool=None,
                            acl_cache=self._acl_cache,
                        )
                    except (SandboxDenied, WorkspaceAccessDenied):
                        continue
                    rows.append(row)
            entries = [self._serialize_row(row) for row in rows]
            result = ToolResult(
                success=True,
                content=json.dumps(entries),
                metadata={"count": len(entries)},
            )
        except (WorkspaceNotFound, NoWorkspacePinned) as exc:
            result = ToolResult(success=False, content="", error=str(exc))
        except WorkspaceAccessDenied as exc:
            result = ToolResult(success=False, content="", error=str(exc))
        except SandboxDenied as exc:
            result = ToolResult(success=False, content="", error=str(exc))
        except Exception as exc:
            log.exception("workspace_history failed: %s", exc)
            result = ToolResult(
                success=False,
                content="",
                error=f"history failed: {exc}",
            )
        return result

    @staticmethod
    def _clamp_limit(raw: Any) -> int:
        """
        coerce ``limit`` kwarg into a bounded positive integer.

        :param raw: caller-supplied ``limit`` value (int or int-convertible)
        :ptype raw: Any
        :return: integer within ``[1, _MAX_LIMIT]``
        :rtype: int
        :raises ValueError: if value cannot be coerced to an int
        """
        value = int(raw)
        result: int
        if value < 1:
            result = 1
        elif value > _MAX_LIMIT:
            result = _MAX_LIMIT
        else:
            result = value
        return result

    @staticmethod
    def _serialize_row(row: Any) -> dict[str, Any]:
        """
        flatten a journal entity into a JSON-safe metadata dict (no content).

        :param row: journal entity from the version collection
        :ptype row: Any
        :return: metadata dict with string ids, iso date, size_bytes
        :rtype: dict[str, Any]
        """
        return {
            "relative_path": row.relative_path,
            "version": row.version,
            "action": row.action,
            "label": row.label,
            "actor_id": str(row.actor_id),
            "correlation_id": str(row.correlation_id),
            "date_created": row.date_created.isoformat(),
            "sha256": row.sha256,
            "size_bytes": len(row.content),
        }

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
                "list workspace-file journal entries newest-first, optionally "
                "narrowed to a single relative_path; returns metadata only "
                "(no content blob)"
            ),
            input_schema=_INPUT_SCHEMA,
        )

    def mcp_name(self) -> str:
        """
        returns the namespaced tool name advertised to MCP clients.

        :return: tool name
        :rtype: str
        """
        return "threetears.workspace.history"

    def mcp_version(self) -> str:
        """
        returns the semver-compatible tool version.

        :return: version string
        :rtype: str
        """
        return "1.0"


def _build(**kwargs: Any) -> WorkspaceHistoryTool:
    """
    constructs a :class:`WorkspaceHistoryTool` from the factory dep bundle.

    consumes workspace, file, and journal collections, ``sandbox``,
    ``context_provider``, and ``agent_id``; ignores the rest. registered
    with :mod:`threetears.agent.workspace.factory` on import so
    :func:`build_workspace_tools` emits this tool.

    :param kwargs: full factory dependency bundle
    :ptype kwargs: Any
    :return: constructed tool
    :rtype: WorkspaceHistoryTool
    """
    return WorkspaceHistoryTool(
        workspace_collection=kwargs["workspace_collection"],
        workspace_file_collection=kwargs["workspace_file_collection"],
        workspace_file_version_collection=kwargs["workspace_file_version_collection"],
        sandbox=kwargs["sandbox"],
        context_provider=kwargs["context_provider"],
        agent_id=kwargs["agent_id"],
        db_pool=kwargs.get("db_pool"),
        acl_cache=kwargs["acl_cache"],
    )


register_tool_builder(_build)
