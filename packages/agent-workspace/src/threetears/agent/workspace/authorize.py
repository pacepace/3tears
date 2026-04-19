"""workspace access authorization helper.

every workspace tool calls :func:`authorize_workspace_access` at the
top of its ``execute`` method, immediately after resolving the target
:class:`~threetears.agent.workspace.entities.Workspace`. rbac-task-01
Phase 3 rewired the helper to delegate to the unified evaluator in
:mod:`threetears.agent.acl` — one source of truth for every
authorization decision across the broker, the agent runtime, and
discovery.

rules (in order):

1. **missing customer on scope** — a tool call arriving without a
   ``customer_id`` on its envelope is unroutable; raise.
2. **cross-customer** — a caller belonging to customer ``A`` can never
   touch a workspace owned by customer ``B``. categorical deny, never
   overridable by an assignment (assignments within the same customer
   only; the evaluator enforces this independently but we short-circuit
   here to keep the cheap check cheap).
3. **unified evaluator** — build an
   :class:`~threetears.agent.acl.EvaluationContext` carrying the
   workspace's :class:`~threetears.agent.acl.Namespace`, the intent
   action (``read`` / ``write``), and the caller's user + agent ids.
   :func:`~threetears.agent.acl.evaluate_decision` resolves owner
   short-circuit + group / assignment / role chain in one call. a
   deny lands here as :class:`WorkspaceAccessDenied`.

the helper intentionally does NOT re-implement owner-path short-
circuit: the evaluator already handles
``namespace.owner_agent_id == agent_id`` inside
:func:`~threetears.agent.acl.evaluator._resolve_side`. we keep the
cross-customer guard as a belt-and-suspenders check because it is
the one rule with zero cost to surface at the call site (no IO, no
evaluator round-trip) — and a tool arriving with a ``customer_id`` of
another customer is a programming error worth catching with a clear
message before the evaluator's generic deny.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol
from uuid import UUID

from threetears.agent.acl import (
    EvaluationContext,
    GrantLoader,
    MembershipLoader,
    Namespace as AclNamespace,
    evaluate_decision,
)
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


class WorkspaceAccessDenied(Exception):
    """raised when the current call is not allowed to touch the workspace.

    every tool's ``execute`` catches this (or lets it propagate as a
    tool-level error) so a single exception type covers every denial
    path: missing customer, cross-customer, no assignment, wrong
    action.
    """


class WorkspaceLike(Protocol):
    """structural type for the workspace record the helper inspects.

    :ivar id: workspace UUID (also the matching namespace id)
    :ivar customer_id: owning customer UUID
    :ivar owner_agent_id: UUID of the agent that owns the physical rows
    :ivar created_by_user_id: UUID of the user who created the workspace
    :ivar namespace_name: canonical name of the workspace's namespace
        row; kept on the Protocol for logging / back-compat callers
    """

    id: UUID
    customer_id: UUID
    owner_agent_id: UUID
    created_by_user_id: UUID
    namespace_name: str


class AclCacheLike(Protocol):
    """structural type for the broker-side acl gateway.

    rbac-task-01 Phase 3: the helper now speaks the unified evaluator
    surface directly. the protocol is kept to preserve the argument
    name on tool factories and because some agent-side callers supply
    a wrapper that holds the loaders; concrete implementations
    surface ``membership_loader`` and ``grant_loader`` for the
    evaluator call.
    """

    membership_loader: MembershipLoader
    grant_loader: GrantLoader


async def authorize_workspace_access(
    scope: ToolCallScope,
    workspace: WorkspaceLike,
    operation: Literal["read", "write"],
    *,
    acl_cache: AclCacheLike,
) -> None:
    """authorize the current tool call's identity against ``workspace``.

    builds an :class:`EvaluationContext` from the scope + workspace and
    calls :func:`~threetears.agent.acl.evaluate_decision` with the
    loaders the caller's acl_cache carries.

    :param scope: live :class:`ToolCallScope` pushed by the tool
        server for this dispatch; ``scope.context`` carries
        ``customer_id`` / ``user_id`` / ``agent_id``
    :ptype scope: ToolCallScope
    :param workspace: workspace record being accessed; must expose
        ``id``, ``customer_id``, ``owner_agent_id``,
        ``created_by_user_id``, ``namespace_name`` attributes (see
        :class:`WorkspaceLike`)
    :ptype workspace: WorkspaceLike
    :param operation: intent verb — ``"read"`` for retrieval,
        ``"write"`` for mutation
    :ptype operation: Literal["read", "write"]
    :param acl_cache: object exposing ``membership_loader`` +
        ``grant_loader`` (the shared :class:`~threetears.agent.acl.AclCache`
        gateway or a per-process equivalent)
    :ptype acl_cache: AclCacheLike
    :return: nothing
    :rtype: None
    :raises WorkspaceAccessDenied: on any denial path
    """
    if operation not in ("read", "write"):
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

    # business logic: single return.
    namespace = AclNamespace(
        id=workspace.id,
        customer_id=workspace.customer_id,
        namespace_type="workspace",
        owner_agent_id=workspace.owner_agent_id,
    )
    eval_ctx = EvaluationContext(
        namespace=namespace,
        action=operation,
        user_id=ctx.user_id,
        agent_id=ctx.agent_id,
    )
    decision = await evaluate_decision(
        eval_ctx,
        membership_loader=acl_cache.membership_loader,
        grant_loader=acl_cache.grant_loader,
    )

    grant_error: Exception | None = None
    if not decision:
        grant_error = WorkspaceAccessDenied(
            "evaluator denied access on namespace "
            f"{workspace.namespace_name}",
        )

    if grant_error is not None:
        _log_denial(
            workspace=workspace,
            operation=operation,
            ctx=ctx,
            reason=str(grant_error),
        )
        raise grant_error

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
    :param reason: short classification string
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
