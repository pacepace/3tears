"""``threetears.workspace.reset`` -- restore a workspace to its template state.

reset is only valid on workspaces that were created from a template (the
template name is recorded in ``workspaces.template_name``). the tool
re-walks the template directory, computes a three-way set difference
against the current head-state files, and journals a single batch of
``revert`` / ``create`` / ``delete`` rows under one ``conn.transaction()``
so the head-state and journal advance atomically. all reverts share a
single new version number (``current_version + 1``) so callers can roll
the entire reset back as one history entry.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid7

from threetears.agent.tools.base_tool import (
    MCPToolDefinition,
    TearsTool,
    ToolResult,
)
from threetears.agent.tools.context import ToolContextManager
from threetears.observe import get_logger

from threetears.agent.workspace import audit, pin
from threetears.agent.workspace.authorize import (
    AclCacheLike,
    WorkspaceAccessDenied,
)
from threetears.agent.workspace.collections import (
    WorkspaceCollection,
    WorkspaceFileCollection,
    WorkspaceFileVersionCollection,
)
from threetears.agent.workspace.config import ValidatorEntry
from threetears.agent.workspace.factory import register_tool_builder
from threetears.agent.workspace.sandbox import WorkspaceSandbox
from threetears.agent.workspace.tools.helpers import (
    _resolve_validators,
    authorize_workspace,
    workspace_audit_identity,
)
from threetears.agent.workspace.validators import (
    WorkspaceValidationError,
    dispatch_validators,
)

__all__ = [
    "WorkspaceResetTool",
]

log = get_logger(__name__)


_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "workspace name to reset; defaults to pinned workspace",
        },
    },
    "required": [],
    "additionalProperties": False,
}


_INSERT_WORKSPACE_FILE_SQL = """
INSERT INTO workspace_files (
    id, workspace_id, relative_path, content, sha256, version, date_updated
) VALUES ($1, $2, $3, $4, $5, $6, $7)
ON CONFLICT (workspace_id, relative_path) DO UPDATE SET
    content = EXCLUDED.content,
    sha256 = EXCLUDED.sha256,
    version = EXCLUDED.version,
    date_updated = EXCLUDED.date_updated
"""

_DELETE_WORKSPACE_FILE_BY_PATH_SQL = "DELETE FROM workspace_files WHERE workspace_id = $1 AND relative_path = $2"

_INSERT_WORKSPACE_FILE_VERSION_SQL = """
INSERT INTO workspace_file_versions (
    id, workspace_id, relative_path, version, content,
    sha256, action, label, actor_id, correlation_id, date_created
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
"""

_UPDATE_WORKSPACE_VERSION_SQL = "UPDATE workspaces SET current_version = $1, date_updated = $2 WHERE id = $3"


class WorkspaceResetTool(TearsTool):
    """reset a workspace's files to its source template's contents.

    requires the workspace to have a ``template_name`` (set when the
    workspace was created via ``from_template``). diffs the template
    directory against the current head-state files, then journals one
    batch of revert/create/delete rows under a single transaction.
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
        nats_client: Any = None,
        namespace: str | None = None,
        validators: list[ValidatorEntry] | None = None,
        acl_cache: AclCacheLike | None = None,
    ) -> None:
        """
        binds tool to collections, sandbox, context, and pool.

        :param workspace_collection: collection providing find_by_agent_and_name
        :ptype workspace_collection: WorkspaceCollection
        :param workspace_file_collection: collection providing find_by_workspace
        :ptype workspace_file_collection: WorkspaceFileCollection
        :param workspace_file_version_collection: accepted for symmetry;
            transactional inserts go directly through the pool
        :ptype workspace_file_version_collection: WorkspaceFileVersionCollection
        :param sandbox: workspace sandbox enforcing template-read constraints
        :ptype sandbox: WorkspaceSandbox
        :param context_provider: zero-arg callable returning conversation context
        :ptype context_provider: Callable[[], ToolContextManager]
        :param agent_id: identifier of agent owning the workspace
        :ptype agent_id: UUID
        :param db_pool: asyncpg pool supplying acquire+transaction
        :ptype db_pool: Any
        :param nats_client: NATS client for audit publish; None skips audit
        :ptype nats_client: Any
        :param namespace: NATS subject namespace for audit subject
        :ptype namespace: str | None
        :param validators: per-pattern validator entries; every revert /
            create file coming off the template is validated before the
            batch INSERT / UPSERT runs. first failure aborts the reset
            transaction.
        :ptype validators: list[ValidatorEntry] | None
        """
        self._workspaces = workspace_collection
        self._files = workspace_file_collection
        self._versions = workspace_file_version_collection
        self._sandbox = sandbox
        self._context_provider = context_provider
        self._agent_id = agent_id
        self._db_pool = db_pool
        self._nats_client = nats_client
        self._namespace = namespace
        self._validators = validators
        self._acl_cache = acl_cache

    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        reset workspace ``name`` (or pinned) to its template state.

        when ``name`` is omitted, falls back to the conversation's pinned
        workspace; if neither is available, returns a clean error. all
        failures arrive as :class:`ToolResult` with ``success=False``;
        this method never raises.

        :param kwargs: optional ``name``; otherwise pinned workspace
        :ptype kwargs: Any
        :return: tool result reporting reset or error
        :rtype: ToolResult
        """
        name_arg = kwargs.get("name")

        correlation_id = uuid7()
        result: ToolResult
        try:
            name = await self._resolve_name(name_arg)
            workspace = await self._workspaces.find_by_agent_and_name(self._agent_id, name)
            if workspace is None or workspace.date_deleted is not None:
                result = ToolResult(
                    success=False,
                    content="",
                    error=f"workspace {name!r} not found",
                )
            elif workspace.template_name is None:
                result = ToolResult(
                    success=False,
                    content="",
                    error=(
                        f"workspace {name!r} has no template; reset is only "
                        "supported on workspaces created from a template"
                    ),
                )
            else:
                await authorize_workspace(
                    workspace,
                    "write",
                    db_pool=self._db_pool,
                    acl_cache=self._acl_cache,
                )
                template_files = await self._read_template_files(
                    workspace.template_name,
                )
                current_files = await self._files.find_by_workspace(
                    workspace.id,
                )
                n_changed = await self._apply_reset(
                    workspace_id=workspace.id,
                    namespace_name=workspace.namespace_name,
                    current_version=workspace.current_version,
                    template_files=template_files,
                    current_files=[(f.relative_path, f.content, f.sha256) for f in current_files],
                    correlation_id=correlation_id,
                )
                # defense-in-depth: swallow audit-side exceptions so a
                # hiccup there cannot corrupt the successful reset return.
                try:
                    if self._namespace is not None:
                        identity = workspace_audit_identity(workspace)
                        await audit.publish_workspace_event(
                            nats_client=self._nats_client,
                            namespace=self._namespace,
                            event_type="workspace.reset",
                            actor_user_id=identity.actor_user_id,
                            agent_id=self._agent_id,
                            calling_agent_id=identity.calling_agent_id,
                            owner_agent_id=identity.owner_agent_id,
                            customer_id=identity.customer_id,
                            namespace_id=identity.namespace_id,
                            resource_type="workspace",
                            resource_id=str(workspace.id),
                            action="reset",
                            details={
                                "template_name": workspace.template_name,
                                "files_changed": n_changed,
                            },
                            correlation_id=correlation_id,
                        )
                # NOSILENT: audit failure must never taint a successful reset
                except Exception as audit_exc:
                    log.exception(
                        "workspace_reset audit publish swallow caught: %s",
                        audit_exc,
                    )
                result = ToolResult(
                    success=True,
                    content=(
                        f"reset workspace {name!r} to template {workspace.template_name!r}; {n_changed} files affected"
                    ),
                )
        except _ResetError as exc:
            result = ToolResult(success=False, content="", error=str(exc))
        except WorkspaceAccessDenied as exc:
            result = ToolResult(success=False, content="", error=str(exc))
        except WorkspaceValidationError as exc:
            result = ToolResult(
                success=False,
                content="",
                error=f"validation failed: {exc.pattern} -> {exc.reason}",
            )
        except Exception as exc:
            log.exception("workspace_reset failed: %s", exc)
            result = ToolResult(
                success=False,
                content="",
                error=f"reset failed: {exc}",
            )
        return result

    async def _resolve_name(self, name_arg: str | None) -> str:
        """
        resolve workspace name from explicit arg or current pin.

        :param name_arg: explicit workspace name from kwargs
        :ptype name_arg: str | None
        :return: resolved workspace name
        :rtype: str
        :raises _ResetError: when neither argument nor pin is set
        """
        if name_arg:
            return name_arg
        snapshot = await pin.get_pin(self._context_provider())
        if snapshot is None:
            raise _ResetError("no workspace name provided and none pinned")
        return snapshot.workspace_name

    async def _read_template_files(
        self,
        template_name: str,
    ) -> list[tuple[str, bytes, str]]:
        """
        walk the named template directory and gate every file via sandbox.

        blocking filesystem walk + ``read_bytes`` is dispatched via
        :func:`asyncio.to_thread` so the event loop stays responsive on
        large templates. sandbox enforcement stays on the main loop.

        :param template_name: template directory name under templates root
        :ptype template_name: str
        :return: list of (relative_path, content_bytes, sha256_hex) triples
        :rtype: list[tuple[str, bytes, str]]
        """
        templates_root = self._sandbox.resolve_fs_path(template_name, "templates")
        candidates = await asyncio.to_thread(
            _collect_template_paths,
            templates_root,
        )
        for relative, _path in candidates:
            self._sandbox.enforce("read", relative)
        return await asyncio.to_thread(_read_template_bytes, candidates)

    async def _apply_reset(
        self,
        *,
        workspace_id: UUID,
        namespace_name: str,
        current_version: int,
        template_files: list[tuple[str, bytes, str]],
        current_files: list[tuple[str, bytes, str]],
        correlation_id: UUID,
    ) -> int:
        """
        apply the three-way diff in one transaction; return # files changed.

        :param workspace_id: identifier of workspace being reset
        :ptype workspace_id: UUID
        :param namespace_name: canonical workspace namespace name
            (``workspace.<uuid>``); threaded onto the tx via
            ``conn.transaction(namespace=...)`` so every statement
            lands in the owner agent's schema on grantee resets
        :ptype namespace_name: str
        :param current_version: workspace's current head version pointer
        :ptype current_version: int
        :param template_files: triples loaded from the template directory
        :ptype template_files: list[tuple[str, bytes, str]]
        :param current_files: triples loaded from the current head state
        :ptype current_files: list[tuple[str, bytes, str]]
        :return: count of files changed across revert/create/delete actions
        :rtype: int
        """
        new_version = current_version + 1
        now = datetime.now(UTC)
        template_by_path = {rel: (content, sha) for rel, content, sha in template_files}
        current_by_path = {rel: (content, sha) for rel, content, sha in current_files}

        revert_paths = set(template_by_path) & set(current_by_path)
        delete_paths = set(current_by_path) - set(template_by_path)
        create_paths = set(template_by_path) - set(current_by_path)
        n_changed = len(revert_paths) + len(delete_paths) + len(create_paths)

        # validator dispatch runs BEFORE the transaction opens so a
        # rejection leaves the workspace unmodified. deletes have no
        # content to validate -- revert / create both receive the
        # template bytes about to land in L3.
        if self._validators:
            for relative in sorted(revert_paths | create_paths):
                content, _sha = template_by_path[relative]
                dispatch_validators(self._validators, relative, content)

        # WS-ACL-06: bind the tx to the workspace's namespace so reset
        # writes land in the OWNER agent's schema on grantee resets.
        async with self._db_pool.acquire() as conn:
            async with conn.transaction(namespace=namespace_name):
                for relative in sorted(revert_paths):
                    content, sha = template_by_path[relative]
                    await self._upsert_file(conn, workspace_id, relative, content, sha, new_version, now)
                    await self._insert_journal(
                        conn,
                        workspace_id,
                        relative,
                        new_version,
                        content,
                        sha,
                        "revert",
                        now,
                        correlation_id,
                    )
                for relative in sorted(delete_paths):
                    await conn.execute(_DELETE_WORKSPACE_FILE_BY_PATH_SQL, workspace_id, relative)
                    await self._insert_journal(
                        conn,
                        workspace_id,
                        relative,
                        new_version,
                        b"",
                        hashlib.sha256(b"").hexdigest(),
                        "delete",
                        now,
                        correlation_id,
                    )
                for relative in sorted(create_paths):
                    content, sha = template_by_path[relative]
                    await self._upsert_file(conn, workspace_id, relative, content, sha, new_version, now)
                    await self._insert_journal(
                        conn,
                        workspace_id,
                        relative,
                        new_version,
                        content,
                        sha,
                        "create",
                        now,
                        correlation_id,
                    )
                await conn.execute(_UPDATE_WORKSPACE_VERSION_SQL, new_version, now, workspace_id)
        return n_changed

    async def _upsert_file(
        self,
        conn: Any,
        workspace_id: UUID,
        relative: str,
        content: bytes,
        sha: str,
        version: int,
        now: datetime,
    ) -> None:
        """
        insert or update a workspace_files head row for given relative path.

        :param conn: asyncpg connection bound inside the active transaction
        :ptype conn: Any
        :param workspace_id: identifier of parent workspace
        :ptype workspace_id: UUID
        :param relative: workspace-relative path
        :ptype relative: str
        :param content: file content bytes
        :ptype content: bytes
        :param sha: sha256 hex digest of content
        :ptype sha: str
        :param version: new version number for this row
        :ptype version: int
        :param now: timestamp to record on date_updated
        :ptype now: datetime
        """
        await conn.execute(
            _INSERT_WORKSPACE_FILE_SQL,
            uuid7(),
            workspace_id,
            relative,
            content,
            sha,
            version,
            now,
        )

    async def _insert_journal(
        self,
        conn: Any,
        workspace_id: UUID,
        relative: str,
        version: int,
        content: bytes,
        sha: str,
        action: str,
        now: datetime,
        correlation_id: UUID,
    ) -> None:
        """
        append a single workspace_file_versions journal row.

        :param conn: asyncpg connection bound inside the active transaction
        :ptype conn: Any
        :param workspace_id: identifier of parent workspace
        :ptype workspace_id: UUID
        :param relative: workspace-relative path
        :ptype relative: str
        :param version: version number for this journal entry
        :ptype version: int
        :param content: file content bytes captured at this version
        :ptype content: bytes
        :param sha: sha256 hex digest of content
        :ptype sha: str
        :param action: one of ``create``, ``revert``, ``delete``
        :ptype action: str
        :param now: timestamp to record on date_created
        :ptype now: datetime
        """
        await conn.execute(
            _INSERT_WORKSPACE_FILE_VERSION_SQL,
            uuid7(),
            workspace_id,
            relative,
            version,
            content,
            sha,
            action,
            None,
            self._agent_id,
            correlation_id,
            now,
        )

    def mcp_schema(self) -> MCPToolDefinition:
        """
        returns the MCP definition for this tool.

        :return: MCP-compatible tool definition
        :rtype: MCPToolDefinition
        """
        return MCPToolDefinition(
            name=self.mcp_name(),
            version=self.mcp_version(),
            description="reset a workspace's files to its source template state",
            input_schema=_INPUT_SCHEMA,
        )

    def mcp_name(self) -> str:
        """
        returns the namespaced tool name advertised to MCP clients.

        :return: tool name
        :rtype: str
        """
        return "threetears.workspace.reset"

    def mcp_version(self) -> str:
        """
        returns the semver-compatible tool version.

        :return: version string
        :rtype: str
        """
        return "1.0"


def _collect_template_paths(
    templates_root: Path,
) -> list[tuple[str, Path]]:
    """sync walker: enumerate template-root regular files as ``(relative, path)``.

    split from :meth:`WorkspaceResetTool._read_template_files` so the
    blocking walk runs inside :func:`asyncio.to_thread` while sandbox
    enforcement stays on the main event loop. sorted for deterministic
    validator / sandbox ordering.

    :param templates_root: sandbox-resolved template directory
    :ptype templates_root: Path
    :return: sorted list of ``(relative_path, full_path)`` tuples
    :rtype: list[tuple[str, Path]]
    """
    pairs: list[tuple[str, Path]] = []
    for path in sorted(templates_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(templates_root).as_posix()
        pairs.append((relative, path))
    return pairs


def _read_template_bytes(
    pairs: list[tuple[str, Path]],
) -> list[tuple[str, bytes, str]]:
    """sync reader: dereference ``(relative, path)`` pairs into byte triples.

    dispatched via :func:`asyncio.to_thread` so ``read_bytes`` + sha256
    runs off the event loop.

    :param pairs: ``(relative, full_path)`` tuples from walker
    :ptype pairs: list[tuple[str, Path]]
    :return: list of ``(relative_path, content_bytes, sha256_hex)`` triples
    :rtype: list[tuple[str, bytes, str]]
    """
    triples: list[tuple[str, bytes, str]] = []
    for relative, path in pairs:
        content = path.read_bytes()
        sha = hashlib.sha256(content).hexdigest()
        triples.append((relative, content, sha))
    return triples


class _ResetError(Exception):
    """internal sentinel for clean error-as-data branches inside execute."""


def _build(**kwargs: Any) -> WorkspaceResetTool:
    """
    constructs a :class:`WorkspaceResetTool` from the factory dep bundle.

    consumes ``workspace_collection``, ``workspace_file_collection``,
    ``workspace_file_version_collection``, ``sandbox``,
    ``context_provider``, ``agent_id``, and ``db_pool``; ignores the
    rest. registered with :mod:`threetears.agent.workspace.factory` on
    import so :func:`build_workspace_tools` emits this tool.

    :param kwargs: full factory dependency bundle
    :ptype kwargs: Any
    :return: constructed tool
    :rtype: WorkspaceResetTool
    """
    return WorkspaceResetTool(
        workspace_collection=kwargs["workspace_collection"],
        workspace_file_collection=kwargs["workspace_file_collection"],
        workspace_file_version_collection=kwargs["workspace_file_version_collection"],
        sandbox=kwargs["sandbox"],
        context_provider=kwargs["context_provider"],
        agent_id=kwargs["agent_id"],
        db_pool=kwargs["db_pool"],
        nats_client=kwargs.get("nats_client"),
        namespace=kwargs.get("namespace"),
        validators=_resolve_validators(kwargs),
        acl_cache=kwargs.get("acl_cache"),
    )


register_tool_builder(_build)
