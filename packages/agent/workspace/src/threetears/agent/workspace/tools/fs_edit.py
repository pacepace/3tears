"""``threetears.workspace.fs_edit`` -- Claude Code-style find/replace text edit.

exact-string (not regex) find-and-replace with whole-file atomic write.
replaces ALL occurrences of ``find`` (mirroring Claude Code Edit
semantics); the LLM disambiguates with unique find strings when single-
shot replacement is needed. sandbox write-enforcement gates the call
before any mutation; optimistic concurrency via ``expected_sha256``
rejects stale edits cleanly so the LLM can re-read and retry.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid7

from threetears.agent.acl import AclCache
from threetears.agent.audit import AuditEvent, publish_audit
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
from threetears.agent.workspace.config import ValidatorEntry
from threetears.agent.workspace.factory import register_tool_builder
from threetears.agent.workspace.sandbox import WorkspaceSandbox
from threetears.agent.workspace.tools.helpers import (
    NoWorkspacePinned,
    Sha256Mismatch,
    WorkspaceNotFound,
    _resolve_validators,
    _resolve_workspace,
    _write_file_atomic,
    authorize_workspace,
    authorize_workspace_file,
    workspace_audit_identity,
)
from threetears.agent.workspace.validators import WorkspaceValidationError

__all__ = [
    "FsEditTool",
]

log = get_logger(__name__)


_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "relative_path": {
            "type": "string",
            "description": "workspace-relative path",
        },
        "find": {
            "type": "string",
            "description": "Exact string (not regex).",
        },
        "replace": {"type": "string"},
        "expected_sha256": {
            "type": "string",
            "description": "Optimistic concurrency. Mismatch rejects without mutating.",
        },
        "workspace": {
            "type": "string",
            "description": "Defaults to pinned workspace.",
        },
    },
    "required": ["relative_path", "find", "replace"],
    "additionalProperties": False,
}


class FsEditTool(TearsTool):
    """find/replace a workspace text file atomically with OCC.

    resolves workspace, enforces sandbox write BEFORE any mutation,
    reads the current head-state row, decodes as UTF-8 (binary files are
    rejected), verifies the ``find`` string is non-empty and present,
    replaces all occurrences, then runs the three-row transaction
    through :func:`_write_file_atomic`. matches Claude Code Edit: all
    occurrences replaced per call, not just the first.
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
        nats_client: Any = None,
        namespace: str | None = None,
        validators: list[ValidatorEntry] | None = None,
    ) -> None:
        """
        binds tool to collections, sandbox, context, agent, and pool.

        :param workspace_collection: collection providing workspace resolution
        :ptype workspace_collection: WorkspaceCollection
        :param workspace_file_collection: head-state file collection
        :ptype workspace_file_collection: WorkspaceFileCollection
        :param workspace_file_version_collection: journal collection
        :ptype workspace_file_version_collection: WorkspaceFileVersionCollection
        :param sandbox: workspace sandbox enforcing write globs
        :ptype sandbox: WorkspaceSandbox
        :param context_provider: zero-arg callable returning conversation context
        :ptype context_provider: Callable[[], ToolContextManager]
        :param agent_id: identifier of agent owning workspace
        :ptype agent_id: UUID
        :param db_pool: asyncpg pool supplying acquire+transaction
        :ptype db_pool: Any
        :param nats_client: NATS client for audit publish; None skips audit
        :ptype nats_client: Any
        :param namespace: NATS subject namespace for audit subject
        :ptype namespace: str | None
        :param validators: per-pattern validator entries forwarded to
            :func:`_write_file_atomic` for every write; defaults to None
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
        replace every occurrence of ``find`` with ``replace`` in the file.

        all failures arrive as :class:`ToolResult` with ``success=False``;
        this method never raises.

        :param kwargs: must include ``relative_path``, ``find``, ``replace``;
            optional ``expected_sha256`` and ``workspace``
        :ptype kwargs: Any
        :return: tool result reporting edit summary or error
        :rtype: ToolResult
        """
        relative_path = kwargs.get("relative_path", "")
        find_str = kwargs.get("find", "")
        replace_str = kwargs.get("replace", "")
        expected_sha256 = kwargs.get("expected_sha256")
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
                "write",
                db_pool=self._db_pool,
                acl_cache=self._acl_cache,
            )
            self._sandbox.validate_syntax(relative_path)
            await authorize_workspace_file(
                workspace,
                relative_path,
                "write",
                db_pool=None,
                acl_cache=self._acl_cache,
            )
            if find_str == "":
                result = ToolResult(
                    success=False,
                    content="",
                    error="find string must not be empty",
                )
            else:
                existing = await self._files.find_by_workspace_and_relative_path(workspace.id, relative_path)
                if existing is None:
                    result = ToolResult(
                        success=False,
                        content="",
                        error=(f"file {relative_path!r} not found in workspace {workspace.name!r}"),
                    )
                else:
                    try:
                        text = existing.content.decode("utf-8")
                    except UnicodeDecodeError:
                        result = ToolResult(
                            success=False,
                            content="",
                            error=("fs_edit only supports text files; this file appears to be binary"),
                        )
                    else:
                        if find_str not in text:
                            result = ToolResult(
                                success=False,
                                content="",
                                error="find string not found in file",
                            )
                        else:
                            n_occurrences = text.count(find_str)
                            new_text = text.replace(find_str, replace_str)
                            new_bytes = new_text.encode("utf-8")
                            old_sha = existing.sha256
                            old_size = len(existing.content)
                            correlation_id = uuid7()
                            new_version, new_sha256 = await _write_file_atomic(
                                db_pool=self._db_pool,
                                workspace=workspace,
                                relative_path=relative_path,
                                content=new_bytes,
                                action="update",
                                actor_id=self._agent_id,
                                correlation_id=correlation_id,
                                expected_sha256=expected_sha256,
                                workspace_file_collection=self._files,
                                workspace_file_version_collection=self._versions,
                                workspace_collection=self._workspaces,
                                validators=self._validators,
                            )
                            # defense-in-depth audit publish: additive
                            # per-tool event on top of the baseline
                            # ``tool.call`` emitted by ToolServer.
                            try:
                                if self._namespace is not None:
                                    identity = workspace_audit_identity(workspace)
                                    event = AuditEvent(
                                        id=uuid7(),
                                        timestamp=datetime.now(UTC),
                                        event_type="workspace.fs_edit",
                                        actor_user_id=identity.actor_user_id,
                                        calling_agent_id=identity.calling_agent_id,
                                        owner_agent_id=identity.owner_agent_id,
                                        customer_id=identity.customer_id,
                                        resource_namespace_id=identity.namespace_id,
                                        resource_namespace_type="workspace_file",
                                        action="edit",
                                        outcome="success",
                                        correlation_id=correlation_id,
                                        details={
                                            "workspace_resource_id": (f"{workspace.id}/{relative_path}"),
                                            "bytes_before": old_size,
                                            "bytes_after": len(new_bytes),
                                            "sha256_before": old_sha,
                                            "sha256_after": new_sha256,
                                            "version": new_version,
                                            "occurrences": n_occurrences,
                                        },
                                    )
                                    await publish_audit(
                                        event,
                                        nats_client=self._nats_client,
                                        namespace=self._namespace,
                                    )
                            # NOSILENT: audit failure never taints edit
                            except Exception as audit_exc:
                                log.exception(
                                    "fs_edit audit publish swallow caught: %s",
                                    audit_exc,
                                )
                            result = ToolResult(
                                success=True,
                                content=(
                                    f"replaced {n_occurrences} occurrence(s); "
                                    f"sha256={new_sha256}, version={new_version}"
                                ),
                                metadata={
                                    "sha256": new_sha256,
                                    "version": new_version,
                                    "occurrences": n_occurrences,
                                },
                            )
        except Sha256Mismatch as exc:
            result = ToolResult(
                success=False,
                content="",
                error=(f"sha256 mismatch: expected {exc.expected!r}, current {exc.current!r}; re-read and retry"),
            )
        except WorkspaceValidationError as exc:
            result = ToolResult(
                success=False,
                content="",
                error=f"validation failed: {exc.pattern} -> {exc.reason}",
            )
        except (WorkspaceNotFound, NoWorkspacePinned) as exc:
            result = ToolResult(success=False, content="", error=str(exc))
        except WorkspaceAccessDenied as exc:
            result = ToolResult(success=False, content="", error=str(exc))
        except SandboxDenied as exc:
            result = ToolResult(success=False, content="", error=str(exc))
        except Exception as exc:
            log.exception("fs_edit failed: %s", exc)
            result = ToolResult(
                success=False,
                content="",
                error=f"fs_edit failed: {exc}",
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
                "replace every occurrence of find with replace in a "
                "workspace text file; atomic, with optimistic concurrency"
            ),
            input_schema=_INPUT_SCHEMA,
        )

    def mcp_name(self) -> str:
        """
        returns the namespaced tool name advertised to MCP clients.

        :return: tool name
        :rtype: str
        """
        return "threetears.workspace.fs_edit"

    def mcp_version(self) -> str:
        """
        returns the semver-compatible tool version.

        :return: version string
        :rtype: str
        """
        return "1.0"


def _build(**kwargs: Any) -> FsEditTool:
    """
    constructs an :class:`FsEditTool` from the factory dep bundle.

    consumes workspace, file, and journal collections, ``sandbox``,
    ``context_provider``, ``agent_id``, and ``db_pool``; ignores the
    rest. registered with :mod:`threetears.agent.workspace.factory` on
    import so :func:`build_workspace_tools` emits this tool.

    :param kwargs: full factory dependency bundle
    :ptype kwargs: Any
    :return: constructed tool
    :rtype: FsEditTool
    """
    return FsEditTool(
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
        acl_cache=kwargs["acl_cache"],
    )


register_tool_builder(_build)
