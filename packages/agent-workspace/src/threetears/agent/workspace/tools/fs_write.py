"""``threetears.workspace.fs_write`` -- atomic whole-file write with OCC.

content is accepted as text (UTF-8-encoded to bytes) or raw bytes when
the caller passes already-decoded content. ``expected_sha256`` enables
HTTP-If-Match-style optimistic concurrency: when supplied, the write is
rejected cleanly if another writer has advanced the head sha since the
caller last read. journal + head-state + workspace version pointer
advance in a single transaction via :func:`_write_file_atomic`; sandbox
write-enforcement gates the call before any mutation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal
from uuid import UUID, uuid7

from threetears.agent.tools.base_tool import (
    MCPToolDefinition,
    TearsTool,
    ToolResult,
)
from threetears.agent.tools.context import ToolContextManager
from threetears.core.security import SandboxDenied
from threetears.observe import get_logger

from threetears.agent.workspace import audit
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
)
from threetears.agent.workspace.validators import WorkspaceValidationError

log = get_logger(__name__)


_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "relative_path": {
            "type": "string",
            "description": "workspace-relative path to write",
        },
        "content": {
            "type": "string",
            "description": "file content as UTF-8 text",
        },
        "expected_sha256": {
            "type": "string",
            "description": (
                "prior sha256 caller expects on head row; mismatch rejects "
                "the write without mutating state (optimistic concurrency)"
            ),
        },
        "workspace": {
            "type": "string",
            "description": "workspace name; defaults to pinned workspace",
        },
    },
    "required": ["relative_path", "content"],
    "additionalProperties": False,
}


class FsWriteTool(TearsTool):
    """write a workspace file atomically with optimistic concurrency.

    resolves workspace, enforces sandbox write BEFORE any mutation,
    coerces ``content`` to bytes, determines journal action from whether
    a head row already exists, then runs the three-row transaction
    through :func:`_write_file_atomic`. OCC failure converts to a clean
    agent-visible error including the current sha so the LLM can re-read
    and retry.
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

    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        write ``content`` to ``relative_path`` in resolved workspace.

        all failures arrive as :class:`ToolResult` with ``success=False``;
        this method never raises. validator rejections surface as
        ``error="validation failed: {pattern} -> {reason}"`` so the LLM
        can self-correct against the schema.

        :param kwargs: must include ``relative_path`` and ``content``;
            optional ``expected_sha256`` and ``workspace``
        :ptype kwargs: Any
        :return: tool result reporting write summary or error
        :rtype: ToolResult
        """
        relative_path = kwargs.get("relative_path", "")
        raw_content = kwargs.get("content", "")
        expected_sha256 = kwargs.get("expected_sha256")
        workspace_arg = kwargs.get("workspace")

        result: ToolResult
        correlation_id = uuid7()
        try:
            workspace = await _resolve_workspace(
                workspace_arg,
                self._context_provider(),
                self._workspaces,
                self._agent_id,
            )
            self._sandbox.enforce("write", relative_path)
            content_bytes: bytes
            if isinstance(raw_content, bytes):
                content_bytes = raw_content
            else:
                content_bytes = str(raw_content).encode("utf-8")
            existing = await self._files.find_by_workspace_and_relative_path(
                workspace.id, relative_path
            )
            action: Literal["create", "update"] = (
                "update" if existing is not None else "create"
            )
            old_bytes: bytes = existing.content if existing is not None else b""
            old_sha: str | None = existing.sha256 if existing is not None else None
            new_version, new_sha256 = await _write_file_atomic(
                db_pool=self._db_pool,
                workspace=workspace,
                relative_path=relative_path,
                content=content_bytes,
                action=action,
                actor_id=self._agent_id,
                correlation_id=correlation_id,
                expected_sha256=expected_sha256,
                workspace_file_collection=self._files,
                workspace_file_version_collection=self._versions,
                workspace_collection=self._workspaces,
                validators=self._validators,
            )
            # defense-in-depth audit publish
            try:
                if self._namespace is not None:
                    await audit.publish_workspace_event(
                        nats_client=self._nats_client,
                        namespace=self._namespace,
                        event_type="workspace.fs_write",
                        actor_id=self._agent_id,
                        agent_id=self._agent_id,
                        resource_type="workspace_file",
                        resource_id=f"{workspace.id}/{relative_path}",
                        action="write",
                        details={
                            "bytes_before": len(old_bytes),
                            "bytes_after": len(content_bytes),
                            "sha256_before": old_sha,
                            "sha256_after": new_sha256,
                            "version": new_version,
                        },
                        correlation_id=correlation_id,
                    )
            # NOSILENT: audit failure must never taint a successful write
            except Exception as audit_exc:
                log.exception(
                    "fs_write audit publish swallow caught: %s", audit_exc,
                )
            result = ToolResult(
                success=True,
                content=(
                    f"wrote {len(content_bytes)} bytes; "
                    f"sha256={new_sha256}, version={new_version}"
                ),
                metadata={
                    "sha256": new_sha256,
                    "version": new_version,
                    "bytes_written": len(content_bytes),
                },
            )
        except Sha256Mismatch as exc:
            result = ToolResult(
                success=False,
                content="",
                error=(
                    f"sha256 mismatch: expected {exc.expected!r}, current "
                    f"{exc.current!r}; re-read and retry"
                ),
            )
        except WorkspaceValidationError as exc:
            result = ToolResult(
                success=False,
                content="",
                error=f"validation failed: {exc.pattern} -> {exc.reason}",
            )
        except (WorkspaceNotFound, NoWorkspacePinned) as exc:
            result = ToolResult(success=False, content="", error=str(exc))
        except SandboxDenied as exc:
            result = ToolResult(success=False, content="", error=str(exc))
        except Exception as exc:
            log.exception("fs_write failed: %s", exc)
            result = ToolResult(
                success=False,
                content="",
                error=f"fs_write failed: {exc}",
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
                "atomically write a workspace file with optimistic "
                "concurrency via expected_sha256"
            ),
            input_schema=_INPUT_SCHEMA,
        )

    def mcp_name(self) -> str:
        """
        returns the namespaced tool name advertised to MCP clients.

        :return: tool name
        :rtype: str
        """
        return "threetears.workspace.fs_write"

    def mcp_version(self) -> str:
        """
        returns the semver-compatible tool version.

        :return: version string
        :rtype: str
        """
        return "1.0"


def _build(**kwargs: Any) -> FsWriteTool:
    """
    constructs an :class:`FsWriteTool` from the factory dep bundle.

    consumes workspace, file, and journal collections, ``sandbox``,
    ``context_provider``, ``agent_id``, and ``db_pool``; ignores the
    rest. registered with :mod:`threetears.agent.workspace.factory` on
    import so :func:`build_workspace_tools` emits this tool.

    reads ``validators`` off the factory bundle when the caller passes
    an explicit ``validators`` kwarg, otherwise pulls
    ``config.validators`` when a ``config`` is provided. keeps bootstrap
    callers that haven't wired per-pattern schemas working with a clean
    ``None`` default.

    :param kwargs: full factory dependency bundle
    :ptype kwargs: Any
    :return: constructed tool
    :rtype: FsWriteTool
    """
    return FsWriteTool(
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
    )


register_tool_builder(_build)
