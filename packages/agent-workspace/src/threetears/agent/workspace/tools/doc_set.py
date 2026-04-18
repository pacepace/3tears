"""``threetears.workspace.doc_set`` -- structural write at a jsonpath.

loads the document through the registered :class:`FormatHandler`, mutates
the tree at ``jsonpath`` via ``handler.set``, dumps it back to text, and
writes atomically with optimistic concurrency via ``expected_sha256``.
comments, key order, anchors, and quote styles survive the round trip
because the handler uses a round-trip parser (YAML today via ruamel.yaml).

sandbox ``write`` enforcement gates the call before any DB read or write,
matching the enforce-then-act ordering enforced by shard 18's AST test.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import UUID, uuid7

from threetears.agent.tools.base_tool import (
    MCPToolDefinition,
    TearsTool,
    ToolResult,
)
from threetears.agent.tools.context import ToolContextManager
from threetears.core.security import SandboxDenied
from threetears.core.serialization import UnknownFormatError, handler_for
from threetears.observe import get_logger

from threetears.agent.workspace import audit
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
    NoWorkspacePinned,
    Sha256Mismatch,
    WorkspaceNotFound,
    _resolve_validators,
    _resolve_workspace,
    _write_file_atomic,
    authorize_workspace,
    workspace_audit_identity,
)
from threetears.agent.workspace.validators import WorkspaceValidationError

__all__ = [
    "DocSetTool",
]

log = get_logger(__name__)


_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "relative_path": {
            "type": "string",
            "description": "workspace-relative path of the document to mutate",
        },
        "jsonpath": {
            "type": "string",
            "description": "jsonpath expression identifying the target location",
        },
        "value": {
            "description": "value to set; any JSON type",
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
    "required": ["relative_path", "jsonpath", "value"],
    "additionalProperties": False,
}


class DocSetTool(TearsTool):
    """set a value at a jsonpath in a structured workspace document.

    resolves workspace, enforces sandbox write BEFORE any DB access,
    dispatches the :class:`FormatHandler` by file suffix, loads the tree,
    calls ``handler.set(tree, jsonpath, value)``, and writes the dumped
    text through :func:`_write_file_atomic` with OCC. comments and key
    order survive because the handler owns round-trip fidelity.
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
            :func:`_write_file_atomic`; validators see the post-dump
            bytes (same contract as fs_*) so structural vs byte-level
            tools share one validator surface
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
        mutate the document at ``jsonpath`` to ``value`` and write atomically.

        all failures arrive as :class:`ToolResult` with ``success=False``;
        this method never raises.

        :param kwargs: must include ``relative_path``, ``jsonpath``, and
            ``value``; optional ``expected_sha256`` and ``workspace``
        :ptype kwargs: Any
        :return: tool result summarizing the set or error
        :rtype: ToolResult
        """
        relative_path = kwargs.get("relative_path", "")
        jsonpath = kwargs.get("jsonpath", "")
        value = kwargs.get("value")
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
            self._sandbox.enforce("write", relative_path)
            try:
                handler = handler_for(relative_path)
            except UnknownFormatError as exc:
                result = ToolResult(
                    success=False,
                    content="",
                    error=(f"no FormatHandler for {relative_path!r}: {exc}; use fs_* tools for this file type"),
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
                            error="doc_set requires text file; got binary",
                        )
                    else:
                        tree = handler.load(text)
                        tree = handler.set(tree, jsonpath, value)
                        new_text = handler.dump(tree)
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
                        # defense-in-depth audit publish
                        try:
                            if self._namespace is not None:
                                identity = workspace_audit_identity(workspace)
                                await audit.publish_workspace_event(
                                    nats_client=self._nats_client,
                                    namespace=self._namespace,
                                    event_type="workspace.doc_set",
                                    actor_user_id=identity.actor_user_id,
                                    agent_id=self._agent_id,
                                    calling_agent_id=identity.calling_agent_id,
                                    owner_agent_id=identity.owner_agent_id,
                                    customer_id=identity.customer_id,
                                    namespace_id=identity.namespace_id,
                                    resource_type="workspace_file",
                                    resource_id=(f"{workspace.id}/{relative_path}"),
                                    action="set",
                                    details={
                                        "jsonpath": jsonpath,
                                        "value": value,
                                        "bytes_before": old_size,
                                        "bytes_after": len(new_bytes),
                                        "sha256_before": old_sha,
                                        "sha256_after": new_sha256,
                                        "version": new_version,
                                    },
                                    correlation_id=correlation_id,
                                )
                        # NOSILENT: audit failure never taints doc_set
                        except Exception as audit_exc:
                            log.exception(
                                "doc_set audit publish swallow caught: %s",
                                audit_exc,
                            )
                        result = ToolResult(
                            success=True,
                            content=(f"set {jsonpath} = {value!r}; sha256={new_sha256}, version={new_version}"),
                            metadata={
                                "sha256": new_sha256,
                                "version": new_version,
                                "bytes_written": len(new_bytes),
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
            log.exception("doc_set failed: %s", exc)
            result = ToolResult(
                success=False,
                content="",
                error=f"doc_set failed: {exc}",
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
                "set a value at a jsonpath in a structured workspace "
                "document, preserving comments and key order; atomic, with "
                "optimistic concurrency via expected_sha256"
            ),
            input_schema=_INPUT_SCHEMA,
        )

    def mcp_name(self) -> str:
        """
        returns the namespaced tool name advertised to MCP clients.

        :return: tool name
        :rtype: str
        """
        return "threetears.workspace.doc_set"

    def mcp_version(self) -> str:
        """
        returns the semver-compatible tool version.

        :return: version string
        :rtype: str
        """
        return "1.0"


def _build(**kwargs: Any) -> DocSetTool:
    """
    constructs a :class:`DocSetTool` from the factory dep bundle.

    consumes workspace, file, and journal collections, ``sandbox``,
    ``context_provider``, ``agent_id``, and ``db_pool``; ignores the
    rest. registered with :mod:`threetears.agent.workspace.factory` on
    import so :func:`build_workspace_tools` emits this tool.

    :param kwargs: full factory dependency bundle
    :ptype kwargs: Any
    :return: constructed tool
    :rtype: DocSetTool
    """
    return DocSetTool(
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
