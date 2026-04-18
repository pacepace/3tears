"""workspace access authorization helper.

every workspace tool calls :func:`authorize_workspace_access` at the
top of its ``execute`` method, immediately after resolving the target
:class:`~threetears.agent.workspace.entities.Workspace`. the helper
encodes the agent -> customer -> user scoping rules that keep a
cross-customer call from ever touching another customer's data, and
defers the same-customer grant check to the shared
:class:`AclCache` so that owner-agent writes, granted non-owner
reads, and denied non-owner reads all route through one codepath.

the helper itself does NO database IO: the namespace + grant lookups
live inside the cache (L1 -> L2 -> L3 as configured). this keeps the
authorization decision a single in-process function call on the hot
path and pushes all cache-warming / invalidation concerns to the cache
owner.

rules (in order):

1. **missing customer on scope** -- a tool call arriving without a
   ``customer_id`` on its envelope is unroutable; raise.
2. **cross-customer** -- a caller belonging to customer ``A`` can never
   touch a workspace owned by customer ``B``. categorical deny, never
   overridable by a grant (grants within the same customer only).
3. **owner short-circuit** -- when the calling agent owns the workspace
   AND the invoking user is the same user who created it, the owner
   path skips the cache entirely. no grant is required for an agent to
   touch the rows it owns under the user who authored them.
4. **delegated grant check** -- any other same-customer call requires
   a grant. the cache resolves user-scoped first, agent-wide second,
   and raises :class:`NamespaceAccessDeniedError` on no match. that
   error is re-raised as :class:`WorkspaceAccessDenied` at this layer
   so downstream tool code catches a single workspace-shaped exception
   type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol
from uuid import UUID

from threetears.agent.tools.call_scope import ToolCallScope
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.agent.tools.context_envelope import CallContext

__all__ = [
    "AclCacheLike",
    "WorkspaceAccessDenied",
    "WorkspaceLike",
    "authorize_workspace_access",
]

log = get_logger(__name__)

#: operations the helper distinguishes. ``"read"`` maps to ``select``
#: when delegating to the cache; ``"write"`` maps to ``upsert`` so the
#: cache rejects read-only grants. the workspace layer speaks in
#: intent-level verbs so callers do not have to know the cache's SQL
#: taxonomy.
_OPERATION_TO_CACHE: dict[str, str] = {
    "read": "select",
    "write": "upsert",
}


class WorkspaceAccessDenied(Exception):
    """raised when the current call is not allowed to touch the workspace.

    every tool's ``execute`` catches this (or lets it propagate as a
    tool-level error) so a single exception type covers every denial
    path: missing customer, cross-customer, missing grant, wrong access
    level. the message carries the specific reason so logs + tool
    responses surface actionable diagnostics.
    """


class WorkspaceLike(Protocol):
    """structural type for the workspace record the helper inspects.

    :class:`~threetears.agent.workspace.entities.Workspace` already
    exposes these attributes once workspace-task-19 Phase 1 lands; the
    helper types against the structural shape so it stays independent
    of the concrete entity class and so tests can substitute lightweight
    dataclasses.

    :ivar id: workspace UUID (also the matching namespace id)
    :ivar customer_id: owning customer UUID
    :ivar owner_agent_id: UUID of the agent that owns the physical rows
    :ivar created_by_user_id: UUID of the user who created the workspace
    :ivar namespace_name: canonical name of the workspace's namespace
        row; used to key the AclCache lookup
    """

    id: UUID
    customer_id: UUID
    owner_agent_id: UUID
    created_by_user_id: UUID
    namespace_name: str


class AclCacheLike(Protocol):
    """structural type for the shared ACL cache dependency.

    concrete implementation lives in ``aibots.hub.broker.acl.AclCache``;
    the workspace package types against the protocol so it takes no
    ``aibots`` import. any implementation that accepts the four kwargs
    and raises an exception on deny will satisfy the contract.
    """

    async def check_access(
        self,
        agent_id: UUID,
        namespace_name: str,
        operation: str,
        user_id: UUID | None = None,
    ) -> None:
        """raise on deny, return ``None`` on allow."""


async def authorize_workspace_access(
    scope: ToolCallScope,
    workspace: WorkspaceLike,
    operation: Literal["read", "write"],
    *,
    acl_cache: AclCacheLike,
) -> None:
    """authorize the current tool call's identity against ``workspace``.

    invoked at the top of every workspace tool's ``execute`` method,
    immediately after the tool resolves the target workspace. applies
    the rules documented on the module docstring (missing customer,
    cross-customer, owner short-circuit, delegated grant check) and
    either returns ``None`` to allow the caller to proceed or raises
    :class:`WorkspaceAccessDenied`.

    :param scope: live :class:`ToolCallScope` pushed by the tool
        server for this dispatch; ``scope.context`` carries
        ``customer_id`` / ``user_id`` / ``agent_id``
    :ptype scope: ToolCallScope
    :param workspace: workspace record being accessed; must expose
        ``customer_id``, ``owner_agent_id``, ``created_by_user_id``,
        ``namespace_name`` attributes (see :class:`WorkspaceLike`)
    :ptype workspace: WorkspaceLike
    :param operation: intent verb -- ``"read"`` for retrieval,
        ``"write"`` for mutation. mapped to ``select`` / ``upsert``
        when delegating to the cache
    :ptype operation: Literal["read", "write"]
    :param acl_cache: shared ACL cache instance exposing
        :meth:`check_access`; injected by the tool factory so the
        workspace package needs no ``aibots`` dependency
    :ptype acl_cache: AclCacheLike
    :return: nothing
    :rtype: None
    :raises WorkspaceAccessDenied: on any denial path (missing
        customer, cross-customer, no grant, wrong level, unknown
        operation)
    """
    # guard clauses (early raises); allowed per CLAUDE.md single-return rule.
    cache_operation = _OPERATION_TO_CACHE.get(operation)
    if cache_operation is None:
        raise WorkspaceAccessDenied(f"unknown workspace operation: {operation}")

    ctx = scope.context
    if ctx.customer_id is None:
        raise WorkspaceAccessDenied("missing customer_id on scope")

    if workspace.customer_id != ctx.customer_id:
        _log_denial(
            workspace=workspace,
            operation=operation,
            ctx=ctx,
            reason="cross-customer",
        )
        raise WorkspaceAccessDenied(
            "cross-customer access denied: "
            f"workspace customer={workspace.customer_id}, "
            f"caller customer={ctx.customer_id}",
        )

    # business logic: single return at the end.
    owner_path = (
        workspace.owner_agent_id == ctx.agent_id
        and workspace.created_by_user_id == ctx.user_id
    )
    grant_error: Exception | None = None

    if not owner_path:
        try:
            await acl_cache.check_access(
                agent_id=ctx.agent_id,  # type: ignore[arg-type]
                namespace_name=workspace.namespace_name,
                operation=cache_operation,
                user_id=ctx.user_id,
            )
        except Exception as exc:  # NamespaceAccessDeniedError or typed subclass
            grant_error = exc

    if grant_error is not None:
        _log_denial(
            workspace=workspace,
            operation=operation,
            ctx=ctx,
            reason=f"grant check failed: {grant_error}",
        )
        raise WorkspaceAccessDenied(f"grant check failed: {grant_error}")

    return None


def _log_denial(
    *,
    workspace: WorkspaceLike,
    operation: str,
    ctx: "CallContext",
    reason: str,
) -> None:
    """emit a structured INFO log line for a denied workspace access.

    :param workspace: workspace record that was targeted
    :ptype workspace: WorkspaceLike
    :param operation: intent verb from the caller
    :ptype operation: str
    :param ctx: call context carrying identity dimensions
    :ptype ctx: CallContext
    :param reason: short classification string (``"cross-customer"``,
        ``"grant check failed: ..."``, etc.)
    :ptype reason: str
    :return: nothing
    :rtype: None
    """
    log.info(
        "workspace access denied",
        extra={"extra_data": {
            "workspace_id": str(workspace.id),
            "namespace_name": workspace.namespace_name,
            "operation": operation,
            "caller_agent_id": (
                str(ctx.agent_id) if ctx.agent_id is not None else None
            ),
            "caller_user_id": (
                str(ctx.user_id) if ctx.user_id is not None else None
            ),
            "caller_customer_id": (
                str(ctx.customer_id) if ctx.customer_id is not None else None
            ),
            "reason": reason,
        }},
    )
