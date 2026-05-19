"""``threetears.workspace.refresh_from_disk`` -- additive disk -> L3 re-scan.

complements the live watcher that ``bind`` spawns: when the watcher
might miss a burst of external changes (a builder subprocess writes
many files at once, a bulk ``rsync``, or the agent simply wants to
force a re-scan after an external tool completes), the agent invokes
this tool to walk the sandboxed ``bind`` root and pull every
disk-present-but-L3-absent or disk-diffs-from-L3 file into L3.

additive only: files that live in L3 but no longer exist on disk are
LEFT ALONE. deletions flow through bind's watcher (while the window
is open) or bind's capture-back (on clean exit). a deliberate
agent-driven delete still goes through the ``fs_write`` -> transaction
path so validator and OCC semantics stay honest.

validator dispatch is intentionally bypassed: the refresh mirror
contract matches bind's, and validators target the LLM-tool write
surface instead. document + fs tools still gate their writes.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid7

from threetears.agent.acl import AclCache
from threetears.agent.tools.base_tool import (
    MCPToolDefinition,
    TearsTool,
    ToolResult,
)
from threetears.agent.tools.context import ToolContextManager
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
    _next_journal_version,
    _resolve_workspace,
    authorize_workspace,
)

__all__ = [
    "WorkspaceRefreshTool",
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


_INSERT_WORKSPACE_FILE_VERSION_SQL = """
INSERT INTO workspace_file_versions (
    version_id, workspace_id, relative_path, version, content,
    sha256, action, label, actor_id, correlation_id, date_created
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
"""

_UPSERT_WORKSPACE_FILE_SQL = """
INSERT INTO workspace_files (
    file_id, workspace_id, relative_path, content, sha256, version, date_updated
) VALUES ($1, $2, $3, $4, $5, $6, $7)
ON CONFLICT (workspace_id, relative_path) DO UPDATE SET
    content = EXCLUDED.content,
    sha256 = EXCLUDED.sha256,
    version = EXCLUDED.version,
    date_updated = EXCLUDED.date_updated
"""

_UPDATE_WORKSPACE_VERSION_SQL = """
UPDATE workspaces
SET current_version = GREATEST(current_version, $1),
    date_updated = $2
WHERE workspace_id = $3 AND agent_id = $4
"""


class WorkspaceRefreshTool(TearsTool):
    """rescan the sandboxed ``bind`` root and import disk additions / changes.

    resolves the workspace via the standard ``workspace`` kwarg or the
    current pin, resolves the sandboxed ``bind`` root, walks it via
    :meth:`pathlib.Path.rglob`, and writes any file that is
    disk-present-but-L3-absent (imported as ``create``) or
    disk-diffs-from-L3 (imported as ``update``). files only in L3 are
    left alone.
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
        """binds tool to collections, sandbox, context, and pool.

        :param workspace_collection: collection providing workspace resolution
        :ptype workspace_collection: WorkspaceCollection
        :param workspace_file_collection: collection providing
            ``find_by_workspace``
        :ptype workspace_file_collection: WorkspaceFileCollection
        :param workspace_file_version_collection: accepted for symmetry;
            transactional inserts go directly through the pool
        :ptype workspace_file_version_collection: WorkspaceFileVersionCollection
        :param sandbox: workspace sandbox resolving the ``bind`` root
        :ptype sandbox: WorkspaceSandbox
        :param context_provider: zero-arg callable returning conversation context
        :ptype context_provider: Callable[[], ToolContextManager]
        :param agent_id: identifier of agent owning the workspace
        :ptype agent_id: UUID
        :param db_pool: asyncpg pool supplying acquire + transaction
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
        """walk the bind root and pull adds / diffs into L3.

        all failures arrive as :class:`ToolResult` with ``success=False``;
        this method never raises.

        :param kwargs: optional ``workspace`` (else pinned)
        :ptype kwargs: Any
        :return: tool result reporting refreshed file count or error
        :rtype: ToolResult
        """
        workspace_arg = kwargs.get("workspace")
        correlation_id = uuid7()

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
                "write",
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
                n_imported = await self._refresh(
                    workspace=workspace,
                    disk_root=disk_root,
                    correlation_id=correlation_id,
                )
                log.info(
                    "workspace.refresh_from_disk.done",
                    extra={
                        "workspace_id": str(workspace.id),
                        "imported_count": n_imported,
                        "disk_root": str(disk_root),
                    },
                )
                result = ToolResult(
                    success=True,
                    content=f"refreshed {n_imported} files from disk",
                    metadata={"imported_count": n_imported},
                )
        except (WorkspaceNotFound, NoWorkspacePinned) as exc:
            result = ToolResult(success=False, content="", error=str(exc))
        except WorkspaceAccessDenied as exc:
            result = ToolResult(success=False, content="", error=str(exc))
        except Exception as exc:
            log.exception("workspace_refresh failed: %s", exc)
            result = ToolResult(
                success=False,
                content="",
                error=f"refresh failed: {exc}",
            )
        return result

    async def _refresh(
        self,
        *,
        workspace: Any,
        disk_root: Path,
        correlation_id: UUID,
    ) -> int:
        """walk disk, import adds / diffs, return count of files written.

        disk enumeration runs inside :func:`asyncio.to_thread` via
        :func:`_scan_disk_sync`, which matches the symlink-escape guard
        applied by :func:`_snapshot_disk_sync`. rows present in L3 but
        missing on disk are deliberately ignored; refresh is additive.

        :param workspace: resolved live workspace entity
        :ptype workspace: Any
        :param disk_root: sandboxed ``bind`` root for the workspace
        :ptype disk_root: Path
        :param correlation_id: correlation id stamped on journal rows
        :ptype correlation_id: UUID
        :return: count of files created or updated
        :rtype: int
        """
        disk_by_path = await asyncio.to_thread(_scan_disk_sync, disk_root)
        head_rows = await self._files.find_by_workspace(workspace.id)
        head_by_path: dict[str, str] = {row.relative_path: row.sha256 for row in head_rows}

        creates: list[tuple[str, bytes, str]] = []
        updates: list[tuple[str, bytes, str]] = []
        for rel, (content, sha) in disk_by_path.items():
            prior = head_by_path.get(rel)
            if prior is None:
                creates.append((rel, content, sha))
            elif prior != sha:
                updates.append((rel, content, sha))

        n_imported = 0
        if creates or updates:
            now = datetime.now(UTC)
            action_create: Literal["create"] = "create"
            action_update: Literal["update"] = "update"
            max_version = 0
            # WS-ACL-06: bind the tx to the workspace's namespace so
            # refresh writes land in the OWNER agent's schema on
            # grantee refreshes of shared workspaces.
            async with self._db_pool.acquire() as conn:
                async with conn.transaction(namespace=workspace.namespace_name):
                    for rel, content, sha in creates:
                        new_version = await _next_journal_version(
                            conn,
                            workspace.id,
                            rel,
                        )
                        if new_version > max_version:
                            max_version = new_version
                        await self._write_row(
                            conn=conn,
                            workspace_id=workspace.id,
                            relative_path=rel,
                            content=content,
                            sha=sha,
                            version=new_version,
                            action=action_create,
                            now=now,
                            correlation_id=correlation_id,
                        )
                    for rel, content, sha in updates:
                        new_version = await _next_journal_version(
                            conn,
                            workspace.id,
                            rel,
                        )
                        if new_version > max_version:
                            max_version = new_version
                        await self._write_row(
                            conn=conn,
                            workspace_id=workspace.id,
                            relative_path=rel,
                            content=content,
                            sha=sha,
                            version=new_version,
                            action=action_update,
                            now=now,
                            correlation_id=correlation_id,
                        )
                    await conn.execute(
                        _UPDATE_WORKSPACE_VERSION_SQL,
                        max_version,
                        now,
                        workspace.id,
                        workspace.agent_id,
                    )
            n_imported = len(creates) + len(updates)
        return n_imported

    async def _write_row(
        self,
        *,
        conn: Any,
        workspace_id: UUID,
        relative_path: str,
        content: bytes,
        sha: str,
        version: int,
        action: Literal["create", "update"],
        now: datetime,
        correlation_id: UUID,
    ) -> None:
        """insert one journal row + upsert one head row inside the caller's tx.

        :param conn: asyncpg connection enrolled in caller's transaction
        :ptype conn: Any
        :param workspace_id: parent workspace identifier
        :ptype workspace_id: UUID
        :param relative_path: workspace-relative path being imported
        :ptype relative_path: str
        :param content: file content bytes
        :ptype content: bytes
        :param sha: sha256 hex digest of ``content``
        :ptype sha: str
        :param version: new version number for this row
        :ptype version: int
        :param action: journal action verb recorded on the new version row
        :ptype action: Literal["create", "update"]
        :param now: timestamp to record on date_created + date_updated
        :ptype now: datetime
        :param correlation_id: correlation id stamped on journal row
        :ptype correlation_id: UUID
        :return: None
        :rtype: None
        """
        await conn.execute(
            _INSERT_WORKSPACE_FILE_VERSION_SQL,
            uuid7(),
            workspace_id,
            relative_path,
            version,
            content,
            sha,
            action,
            None,
            self._agent_id,
            correlation_id,
            now,
        )
        await conn.execute(
            _UPSERT_WORKSPACE_FILE_SQL,
            uuid7(),
            workspace_id,
            relative_path,
            content,
            sha,
            version,
            now,
        )

    def mcp_schema(self) -> MCPToolDefinition:
        """returns the MCP definition for this tool.

        :return: MCP-compatible tool definition
        :rtype: MCPToolDefinition
        """
        return MCPToolDefinition(
            name=self.mcp_name(),
            version=self.mcp_version(),
            description=(
                "additively import disk additions + diffs into L3 for the "
                "sandboxed bind root; disk-absent files remain in L3"
            ),
            input_schema=_INPUT_SCHEMA,
        )

    def mcp_name(self) -> str:
        """returns the namespaced tool name advertised to MCP clients.

        :return: tool name
        :rtype: str
        """
        return "threetears.workspace.refresh_from_disk"

    def mcp_version(self) -> str:
        """returns the semver-compatible tool version.

        :return: version string
        :rtype: str
        """
        return "1.0"


def _scan_disk_sync(disk_root: Path) -> dict[str, tuple[bytes, str]]:
    """sync walker: enumerate every file under ``disk_root`` with sha256.

    matches the symlink-escape guard used by :func:`_snapshot_disk_sync`
    so a planted symlink cannot pull ``/etc`` into L3 through refresh.
    sorted for deterministic test output; production callers ignore
    ordering.

    :param disk_root: absolute path to sandboxed bind root
    :ptype disk_root: Path
    :return: mapping from posix-style relative path to (bytes, sha256)
    :rtype: dict[str, tuple[bytes, str]]
    """
    out: dict[str, tuple[bytes, str]] = {}
    if not disk_root.is_dir():
        return out
    resolved_root = disk_root.resolve()
    for candidate in sorted(disk_root.rglob("*")):
        if not candidate.is_file():
            continue
        try:
            resolved_candidate = candidate.resolve()
            resolved_candidate.relative_to(resolved_root)
        except OSError, ValueError:
            log.warning(
                "workspace.refresh_from_disk.skip_escape",
                extra={
                    "extra_data": {
                        "candidate": str(candidate),
                        "disk_root": str(disk_root),
                    },
                },
            )
            continue
        rel = candidate.relative_to(disk_root).as_posix()
        data = candidate.read_bytes()
        out[rel] = (data, hashlib.sha256(data).hexdigest())
    return out


def _build(**kwargs: Any) -> WorkspaceRefreshTool:
    """constructs a :class:`WorkspaceRefreshTool` from the factory dep bundle.

    consumes ``workspace_collection``, ``workspace_file_collection``,
    ``workspace_file_version_collection``, ``sandbox``,
    ``context_provider``, ``agent_id``, and ``db_pool``; ignores the
    rest. registered with :mod:`threetears.agent.workspace.factory` on
    import so :func:`build_workspace_tools` emits this tool.

    :param kwargs: full factory dependency bundle
    :ptype kwargs: Any
    :return: constructed tool
    :rtype: WorkspaceRefreshTool
    """
    return WorkspaceRefreshTool(
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
