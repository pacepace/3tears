"""``threetears.workspace.rollback_to`` -- revert files to a prior ref.

rolls back either a single file (when ``relative_path`` is supplied) or
every head file in the workspace to the content present at the supplied
``ref``. ``ref`` follows the ``_resolve_ref`` vocabulary (``"head"``,
integer, or checkpoint label).

design commitments:

- **sandbox-first fail-wholesale.** every file in the rollback set is
  passed through ``sandbox.enforce("write", path)`` BEFORE any call to
  ``_write_file_atomic``. if any file is denied, no writes occur --
  rollback is cleanly aborted via a ToolResult error. this pattern is
  load-bearing for the shard-18 AST test that asserts enforce-before-
  write ordering.
- **per-file transaction.** each file's rollback is delegated to
  :func:`_write_file_atomic`, which opens its own connection +
  transaction. the shard explicitly allows this: fail-wholesale is
  already guaranteed by the pre-enforce sweep above, and a
  cross-file transaction would require threading a connection through
  ``_write_file_atomic`` (not supported today). the per-file tx model
  keeps the journal + head-state + workspace-version triplet atomic
  per-file, which matches the OCC contract every other write tool
  follows.
- **no OCC.** rollback is explicit override; ``expected_sha256=None``.
- **revert action.** journal rows are labelled ``action='revert'`` so
  downstream history queries see the explicit rollback signal.
- **skip-on-miss.** when ``_resolve_ref`` returns None for a file (the
  file did not exist at that ref), rollback skips it silently. the
  caller receives an accurate ``n_changed`` count.
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
    WorkspaceNotFound,
    _resolve_ref,
    _resolve_validators,
    _resolve_workspace,
    _write_file_atomic,
    authorize_workspace,
)
from threetears.agent.workspace.validators import WorkspaceValidationError

__all__ = [
    "WorkspaceRollbackTool",
]

log = get_logger(__name__)


_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ref": {
            "type": ["string", "integer"],
            "description": ("target ref: 'head', integer version, or checkpoint label"),
        },
        "relative_path": {
            "type": "string",
            "description": (
                "optional single path to roll back; when omitted, every head file in the workspace is rolled back"
            ),
        },
        "workspace": {
            "type": "string",
            "description": "workspace name; defaults to pinned workspace",
        },
    },
    "required": ["ref"],
    "additionalProperties": False,
}


class WorkspaceRollbackTool(TearsTool):
    """revert workspace files to a prior ref via revert-action journal rows.

    rollback is a write-class operation: every file in the rollback set
    must pass ``sandbox.enforce("write", path)`` BEFORE any mutation
    happens. the two-phase pattern (enforce all, then write all) keeps
    a single denied path from leaving the workspace in a half-rolled-
    back state.
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
        :param workspace_file_collection: collection providing
            ``find_by_workspace`` and ``find_by_workspace_and_relative_path``
        :ptype workspace_file_collection: WorkspaceFileCollection
        :param workspace_file_version_collection: journal collection passed
            through to :func:`_write_file_atomic`
        :ptype workspace_file_version_collection: WorkspaceFileVersionCollection
        :param sandbox: workspace sandbox for per-path write enforcement
        :ptype sandbox: WorkspaceSandbox
        :param context_provider: zero-arg callable returning conversation context
        :ptype context_provider: Callable[[], ToolContextManager]
        :param agent_id: identifier of agent owning workspace; used as
            ``actor_id`` on revert journal rows
        :ptype agent_id: UUID
        :param db_pool: asyncpg pool supplying acquire for ref resolution
            and per-file ``_write_file_atomic`` transactions
        :ptype db_pool: Any
        :param nats_client: NATS client for audit publish; None skips audit
        :ptype nats_client: Any
        :param namespace: NATS subject namespace for audit subject
        :ptype namespace: str | None
        :param validators: per-pattern validator entries forwarded to
            :func:`_write_file_atomic`; rollback is still a content-write
            so validators run on each reverted file
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
        revert files in the rollback set to their content at ``ref``.

        phase 1: enumerate the rollback set. phase 2: sandbox-enforce
        write on every path -- any denial aborts the whole operation
        before any mutation. phase 3: for each file, resolve the target
        ref; skip if absent; else delegate to :func:`_write_file_atomic`
        with ``action='revert'``. all failures arrive as :class:`
        ToolResult` with ``success=False``.

        :param kwargs: must include ``ref``; optional ``relative_path``
            and ``workspace``
        :ptype kwargs: Any
        :return: tool result summarizing files rolled back
        :rtype: ToolResult
        """
        ref = kwargs.get("ref")
        relative_path = kwargs.get("relative_path")
        workspace_arg = kwargs.get("workspace")

        # guard clause (entry-time input validation) per CLAUDE.md rule.
        if ref is None:
            return ToolResult(
                success=False,
                content="",
                error="ref is required",
            )

        result: ToolResult
        correlation_id = uuid7()
        n_changed = 0
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
            rollback_set = await self._collect_rollback_set(workspace.id, relative_path)
            # phase 2: enforce write on every path BEFORE any mutation.
            # shard-18 AST test relies on this ordering.
            for path in rollback_set:
                self._sandbox.enforce("write", path)
            # phase 3: per-file resolve + atomic write.
            async with self._db_pool.acquire() as conn:
                for path in rollback_set:
                    target = await _resolve_ref(conn, workspace.id, path, ref)
                    if target is None:
                        continue
                    await _write_file_atomic(
                        db_pool=self._db_pool,
                        workspace=workspace,
                        relative_path=path,
                        content=target["content"],
                        action="revert",
                        actor_id=self._agent_id,
                        correlation_id=correlation_id,
                        expected_sha256=None,
                        workspace_file_collection=self._files,
                        workspace_file_version_collection=self._versions,
                        workspace_collection=self._workspaces,
                        validators=self._validators,
                    )
                    n_changed += 1
            # defense-in-depth audit publish: one event per rollback call
            try:
                if self._namespace is not None:
                    await audit.publish_workspace_event(
                        nats_client=self._nats_client,
                        namespace=self._namespace,
                        event_type="workspace.rollback_to",
                        actor_id=self._agent_id,
                        agent_id=self._agent_id,
                        resource_type="workspace",
                        resource_id=str(workspace.id),
                        action="rollback_to",
                        details={
                            "ref": str(ref),
                            "files_changed": n_changed,
                        },
                        correlation_id=correlation_id,
                    )
            # NOSILENT: audit failure never taints rollback
            except Exception as audit_exc:
                log.exception(
                    "workspace_rollback audit publish swallow caught: %s",
                    audit_exc,
                )
            result = ToolResult(
                success=True,
                content=(f"rolled back {n_changed} files to ref {ref!r}"),
                metadata={"n_changed": n_changed, "ref": ref},
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
            log.exception("workspace_rollback failed: %s", exc)
            result = ToolResult(
                success=False,
                content="",
                error=f"rollback_to failed: {exc}",
            )
        return result

    async def _collect_rollback_set(self, workspace_id: UUID, relative_path: str | None) -> list[str]:
        """
        enumerate the set of workspace-relative paths to roll back.

        single-file rollback: returns ``[relative_path]`` when the path
        has a head-state row, otherwise ``[]`` so the call becomes a
        no-op (rather than raising). whole-workspace rollback: returns
        every head-state file's relative_path in a stable order
        (pathwise) so tests observe deterministic enforce/write order.

        :param workspace_id: identifier of target workspace
        :ptype workspace_id: UUID
        :param relative_path: optional single path to roll back
        :ptype relative_path: str | None
        :return: ordered list of relative paths to roll back
        :rtype: list[str]
        """
        result: list[str]
        if relative_path is not None and relative_path != "":
            head = await self._files.find_by_workspace_and_relative_path(workspace_id, relative_path)
            result = [] if head is None else [head.relative_path]
        else:
            rows = await self._files.find_by_workspace(workspace_id)
            result = sorted(row.relative_path for row in rows)
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
                "revert file(s) in a workspace to their content at a prior "
                "ref ('head', integer version, or checkpoint label); emits "
                "revert-action journal rows"
            ),
            input_schema=_INPUT_SCHEMA,
        )

    def mcp_name(self) -> str:
        """
        returns the namespaced tool name advertised to MCP clients.

        :return: tool name
        :rtype: str
        """
        return "threetears.workspace.rollback_to"

    def mcp_version(self) -> str:
        """
        returns the semver-compatible tool version.

        :return: version string
        :rtype: str
        """
        return "1.0"


def _build(**kwargs: Any) -> WorkspaceRollbackTool:
    """
    constructs a :class:`WorkspaceRollbackTool` from the factory dep bundle.

    consumes workspace, file, and journal collections, ``sandbox``,
    ``context_provider``, ``agent_id``, and ``db_pool``.

    :param kwargs: full factory dependency bundle
    :ptype kwargs: Any
    :return: constructed tool
    :rtype: WorkspaceRollbackTool
    """
    return WorkspaceRollbackTool(
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
