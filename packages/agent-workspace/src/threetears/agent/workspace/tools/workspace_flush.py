"""``threetears.workspace.flush_to_disk`` -- project current L3 state onto disk.

the dual of :mod:`threetears.agent.workspace.tools.workspace_refresh`.
refresh pulls disk adds / diffs into L3 (disk -> L3); this tool
pushes the full current L3 head-state onto the sandboxed ``bind``
root (L3 -> disk).

flow C motivation: an agent that writes files via ``fs_write``,
``doc_set``, etc. mutates L3 but does not touch disk. downstream
consumers (a builder subprocess, a file-watching CI job, a human
inspector) expect the disk mirror to be current. the canonical
answer in the workspace design is :func:`materialize.bind`, an
async context manager that holds disk and L3 in sync via a watcher
plus capture-back. bind is long-lived by nature and does not fit a
single stateless tool call. ``flush_to_disk`` is the stateless
one-shot: call it when the agent wants disk to catch up with L3,
without opening a bind window.

semantics:

- every L3 head row is projected to disk via
  :func:`threetears.core.utils.atomic_write.atomic_write` so no
  observer ever sees a half-written file
- disk files that are NOT in L3 are LEFT ALONE (flush is additive,
  matching refresh-from-disk's directional symmetry); a deliberate
  disk-side delete still has to run through an explicit file-system
  delete, not the workspace tool surface
- validators are intentionally NOT re-dispatched: flush mirrors
  committed L3 state, and validators target the LLM-tool write
  surface where the decisions to persist are made

the tool is additive on purpose. a "clobber" mode that also deletes
disk-only files would be the natural next extension but requires a
separate design pass on deletion semantics.
"""

from __future__ import annotations

import asyncio
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
from threetears.core.utils.atomic_write import atomic_write
from threetears.observe import get_logger

from threetears.agent.workspace.authorize import (
    AclCacheLike,
    WorkspaceAccessDenied,
)
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
    authorize_workspace,
)

__all__ = [
    "WorkspaceFlushTool",
]

log = get_logger(__name__)


_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "workspace": {
            "type": "string",
            "description": "workspace name; defaults to pinned workspace",
        },
    },
    "required": [],
    "additionalProperties": False,
}


class WorkspaceFlushTool(TearsTool):
    """project every L3 head row for the workspace onto the sandboxed bind root.

    :meth:`execute` resolves the workspace via the explicit ``workspace``
    kwarg or the conversation pin, resolves the sandboxed ``bind`` root
    via the sandbox, and writes each L3 file through
    :func:`atomic_write` so crashed mid-writes cannot leave partial
    bytes. returns a count of files written.
    """

    def __init__(
        self,
        workspace_collection: WorkspaceCollection,
        workspace_file_collection: WorkspaceFileCollection,
        sandbox: WorkspaceSandbox,
        context_provider: Callable[[], ToolContextManager],
        agent_id: UUID,
        db_pool: Any = None,
        acl_cache: AclCacheLike | None = None,
    ) -> None:
        """capture collections, sandbox, context, and agent identity.

        :param workspace_collection: collection providing workspace resolution
        :ptype workspace_collection: WorkspaceCollection
        :param workspace_file_collection: collection providing
            ``find_by_workspace`` for the L3 head-state enumeration
        :ptype workspace_file_collection: WorkspaceFileCollection
        :param sandbox: workspace sandbox resolving the ``bind`` root
        :ptype sandbox: WorkspaceSandbox
        :param context_provider: zero-arg callable returning
            conversation context, used for pin resolution
        :ptype context_provider: Callable[[], ToolContextManager]
        :param agent_id: identifier of agent owning the workspace
        :ptype agent_id: UUID
        :return: None
        :rtype: None
        """
        self._workspaces = workspace_collection
        self._files = workspace_file_collection
        self._sandbox = sandbox
        self._context_provider = context_provider
        self._agent_id = agent_id
        self._db_pool = db_pool
        self._acl_cache = acl_cache

    async def execute(self, **kwargs: Any) -> ToolResult:
        """write every L3 head row to disk under the sandboxed bind root.

        all failures arrive as :class:`ToolResult` with ``success=False``;
        this method never raises.

        :param kwargs: optional ``workspace`` (else pinned)
        :ptype kwargs: Any
        :return: tool result reporting flushed-file count or error
        :rtype: ToolResult
        """
        workspace_arg = kwargs.get("workspace")
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
            try:
                disk_root = self._sandbox.resolve_fs_path(
                    workspace.name,
                    "bind",
                )
            except KeyError:
                result = ToolResult(
                    success=False,
                    content="",
                    error="bind_root not configured for this workspace",
                )
            else:
                n_written = await self._flush(
                    workspace=workspace,
                    disk_root=disk_root,
                )
                log.info(
                    "workspace.flush_to_disk.done",
                    extra={
                        "workspace_id": str(workspace.id),
                        "flushed_count": n_written,
                        "disk_root": str(disk_root),
                    },
                )
                result = ToolResult(
                    success=True,
                    content=f"flushed {n_written} files to disk",
                    metadata={"flushed_count": n_written},
                )
        except (WorkspaceNotFound, NoWorkspacePinned) as exc:
            result = ToolResult(success=False, content="", error=str(exc))
        except WorkspaceAccessDenied as exc:
            result = ToolResult(success=False, content="", error=str(exc))
        except Exception as exc:
            log.exception("workspace_flush failed: %s", exc)
            result = ToolResult(
                success=False,
                content="",
                error=f"flush failed: {exc}",
            )
        return result

    async def _flush(
        self,
        *,
        workspace: Any,
        disk_root: Path,
    ) -> int:
        """write every L3 head row for ``workspace`` to ``disk_root``.

        we skip files already matching on disk by comparing ``sha256``
        against the file bytes that currently exist; this turns the
        common "repeat flush with no L3 changes" case into a no-op
        and avoids ``atomic_write``'s rename-churn. directories along
        the path are created as needed.

        :param workspace: resolved live workspace entity
        :ptype workspace: Any
        :param disk_root: sandboxed ``bind`` root for the workspace
        :ptype disk_root: Path
        :return: count of files written (excluding sha-match skips)
        :rtype: int
        """
        head_rows = await self._files.find_by_workspace(workspace.id)
        disk_root.mkdir(parents=True, exist_ok=True)

        n_written = 0
        for row in head_rows:
            target = disk_root / row.relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            # sha-match optimization: if the on-disk bytes already
            # hash to the same value as the L3 row we skip the write
            # so idempotent flush calls do not churn the mtimes of
            # unchanged files.
            if await asyncio.to_thread(
                _disk_sha_matches,
                target,
                row.sha256,
            ):
                continue
            await atomic_write(target, row.content)
            n_written += 1
        return n_written

    def mcp_schema(self) -> MCPToolDefinition:
        """returns the MCP definition for this tool.

        :return: MCP-compatible tool definition
        :rtype: MCPToolDefinition
        """
        return MCPToolDefinition(
            name=self.mcp_name(),
            version=self.mcp_version(),
            description=(
                "project every L3 head row for the workspace onto the "
                "sandboxed bind root via atomic_write; idempotent on "
                "sha-match, additive (disk-only files left in place)"
            ),
            input_schema=_INPUT_SCHEMA,
        )

    def mcp_name(self) -> str:
        """returns the namespaced tool name advertised to MCP clients.

        :return: tool name
        :rtype: str
        """
        return "threetears.workspace.flush_to_disk"

    def mcp_version(self) -> str:
        """returns the semver-compatible tool version.

        :return: version string
        :rtype: str
        """
        return "1.0"


def _disk_sha_matches(path: Path, expected_sha: str) -> bool:
    """return True when the bytes at ``path`` hash to ``expected_sha``.

    runs on a worker thread via :func:`asyncio.to_thread` so the flush
    loop does not block the event loop on the read.

    :param path: absolute path to candidate on-disk file
    :ptype path: Path
    :param expected_sha: sha256 hex digest of the L3 head row
    :ptype expected_sha: str
    :return: True when the file exists and the bytes hash to the
        expected digest, False otherwise (including missing file)
    :rtype: bool
    """
    import hashlib

    if not path.is_file():
        return False
    try:
        data = path.read_bytes()
    except OSError:
        return False
    return hashlib.sha256(data).hexdigest() == expected_sha


def _build(**kwargs: Any) -> WorkspaceFlushTool:
    """construct a :class:`WorkspaceFlushTool` from the factory bundle.

    consumes ``workspace_collection``, ``workspace_file_collection``,
    ``sandbox``, ``context_provider``, and ``agent_id``; ignores the
    rest. registered with :mod:`threetears.agent.workspace.factory` so
    :func:`build_workspace_tools` emits this tool alongside the other
    workspace primitives.

    :param kwargs: full factory dependency bundle
    :ptype kwargs: Any
    :return: constructed tool
    :rtype: WorkspaceFlushTool
    """
    return WorkspaceFlushTool(
        workspace_collection=kwargs["workspace_collection"],
        workspace_file_collection=kwargs["workspace_file_collection"],
        sandbox=kwargs["sandbox"],
        context_provider=kwargs["context_provider"],
        agent_id=kwargs["agent_id"],
        db_pool=kwargs.get("db_pool"),
        acl_cache=kwargs.get("acl_cache"),
    )


register_tool_builder(_build)
