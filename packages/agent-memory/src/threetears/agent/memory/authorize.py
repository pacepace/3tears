"""memory access authorization helper.

namespace-task-01 phase 3 moves memory reads, writes, and extracts
behind the unified rbac evaluator. every call into the memory
collection, retriever, extractor, or user-facing tool resolves the
per-(agent, customer) memory namespace, then evaluates the requested
action (``memory.read`` / ``memory.write`` / ``memory.extract``) for
the caller's ``(user_id, agent_id)`` pair.

owner short-circuit: when the calling agent owns the memory
namespace (agent-internal extraction path, retriever invoked from
the owning agent's runtime), the evaluator's built-in owner rule
short-circuits every action to allow without a grant. agent-internal
writes therefore never require assignment configuration.

user-side enforcement: when a user explicitly writes or reads
memories, the evaluator runs both sides. the user side must
contribute the action via an assignment on the ``memory`` namespace
(or a wider scope). a user without an assignment is denied with
:class:`MemoryAccessDenied` — distinct from "no memories" so
callers can surface the denial cleanly.

auto-assignment on first write: user-initiated writes that succeed
trigger :func:`ensure_memory_owner_assignment` so the user's
per-user group (``memory-owner:{user_id.hex}``) is bound to the
platform ``MemoryOwner`` role scoped to this memory namespace. this
is the write-time equivalent of the bootstrap translation phase 2
performed for tool access: declarative "the user who writes a
memory owns it" intent materialized into a concrete rbac row so the
evaluator can answer subsequent questions from cache.
"""

from __future__ import annotations

from typing import Awaitable, Callable, Literal
from uuid import UUID

from threetears.agent.acl import (
    EvaluationContext,
    GrantLoader,
    MembershipLoader,
    Namespace as AclNamespace,
    evaluate_decision,
)
from threetears.observe import get_logger

__all__ = [
    "ACTION_MEMORY_EXTRACT",
    "ACTION_MEMORY_READ",
    "ACTION_MEMORY_WRITE",
    "MEMORY_NAMESPACE_TYPE",
    "MEMORY_OWNER_GROUP_PREFIX",
    "MEMORY_OWNER_ROLE_NAME",
    "MemoryAccessDenied",
    "MemoryAuthorizerDependencies",
    "MemoryNamespaceRow",
    "MemoryNamespaceResolver",
    "MemoryOwnerAssignmentEnsurer",
    "authorize_memory_access",
    "memory_namespace_name",
]

log = get_logger(__name__)


#: namespace_type discriminator for memory rows in ``platform.namespaces``.
#: matches the closed-set admitted by the v018 CHECK constraint.
MEMORY_NAMESPACE_TYPE = "memory"


#: canonical action string for memory reads. evaluated against the
#: ``memory`` bucket of the caller's roles via
#: :meth:`~threetears.agent.acl.types.Role.actions_for`.
ACTION_MEMORY_READ = "memory.read"


#: canonical action string for user-initiated memory writes.
ACTION_MEMORY_WRITE = "memory.write"


#: canonical action string for agent-internal memory extraction.
#: distinct from ``memory.write`` so operators can grant / audit
#: "LLM emitted memories on the user's behalf" separately from
#: "user explicitly added a memory."
ACTION_MEMORY_EXTRACT = "memory.extract"


#: role name seeded by the hub v020 migration for the per-user
#: owner grant. the auto-assignment at first write binds the
#: per-user group to this role scoped to the memory namespace.
MEMORY_OWNER_ROLE_NAME = "MemoryOwner"


#: prefix for the auto-generated per-user memory-owner group name.
#: final shape is ``memory-owner:<user_id_hex>``; customer-scoped so
#: one group row per user per customer suffices.
MEMORY_OWNER_GROUP_PREFIX = "memory-owner"


class MemoryAccessDenied(Exception):
    """raised when the evaluator denies a memory access.

    carries the action and the caller's identity dimensions so
    downstream handlers can surface a precise denial reason instead
    of a silent empty-set fallback. the retriever, the collection,
    and the memory tools all catch-or-propagate this exception as
    the single denial type.
    """


def memory_namespace_name(agent_id: UUID, customer_id: UUID) -> str:
    """build the canonical memory namespace name for an (agent, customer) pair.

    shape: ``memory:<agent_id_hex[:8]>:<customer_id_hex[:8]>``. uses
    the first 8 hex chars of each UUID per the task shard
    convention; the uniqueness is carried by the full
    (namespace_type, owner_agent_id, customer_id) tuple on the row
    itself, so the short prefix is a human-readable display handle
    and not a uniqueness key.

    :param agent_id: owning agent UUID
    :ptype agent_id: UUID
    :param customer_id: owning customer UUID
    :ptype customer_id: UUID
    :return: canonical namespace name
    :rtype: str
    """
    return f"memory:{agent_id.hex[:8]}:{customer_id.hex[:8]}"


class MemoryNamespaceRow:
    """resolved ``platform.namespaces`` row fields for a memory namespace.

    mirrors the registry's ``ToolNamespaceRow`` shape so the
    evaluator round-trip pulls only the four fields it reads. the
    resolver constructs one of these from a cache hit or a freshly
    created row; the authorize helper forwards it straight into an
    :class:`~threetears.agent.acl.Namespace`.

    :ivar id: namespace UUID
    :ivar namespace_type: always ``"memory"`` for rows surfaced here
    :ivar owner_agent_id: owning agent UUID
    :ivar customer_id: owning customer UUID
    """

    __slots__ = ("id", "namespace_type", "owner_agent_id", "customer_id")

    def __init__(
        self,
        *,
        id: UUID,
        namespace_type: str,
        owner_agent_id: UUID,
        customer_id: UUID,
    ) -> None:
        """initialize a resolved memory namespace row.

        :param id: namespace UUID
        :ptype id: UUID
        :param namespace_type: namespace type (always ``"memory"``)
        :ptype namespace_type: str
        :param owner_agent_id: owning agent UUID
        :ptype owner_agent_id: UUID
        :param customer_id: owning customer UUID
        :ptype customer_id: UUID
        """
        self.id = id
        self.namespace_type = namespace_type
        self.owner_agent_id = owner_agent_id
        self.customer_id = customer_id


#: async callable resolving a memory namespace by (agent_id, customer_id).
#: returns a :class:`MemoryNamespaceRow` when the namespace exists
#: (creating it idempotently is the resolver's responsibility), or
#: ``None`` when creation is out-of-band and the row is absent.
MemoryNamespaceResolver = Callable[[UUID, UUID], Awaitable["MemoryNamespaceRow | None"]]


#: async callable ensuring the per-user MemoryOwner assignment exists
#: for a (user_id, memory_namespace_id) pair. invoked after a
#: successful user-initiated write. implementations are idempotent:
#: insert-if-absent on the group row, insert-if-absent on the
#: assignment row.
MemoryOwnerAssignmentEnsurer = Callable[[UUID, "MemoryNamespaceRow"], Awaitable[None]]


class MemoryAuthorizerDependencies:
    """bundle of the four callables the authorizer needs at call time.

    keeping the bundle as one value makes it cheap to pass through
    the collection / retriever / extractor / tool constructors
    without widening their signatures to four parameters each.

    :ivar membership_loader: actor -> memberships resolver
    :ivar grant_loader: groups -> assignments + roles resolver
    :ivar namespace_resolver: resolves (agent_id, customer_id) to a
        :class:`MemoryNamespaceRow` (creating idempotently)
    :ivar assignment_ensurer: ensures the per-user MemoryOwner
        assignment exists after a successful user-write
    """

    __slots__ = (
        "membership_loader",
        "grant_loader",
        "namespace_resolver",
        "assignment_ensurer",
    )

    def __init__(
        self,
        *,
        membership_loader: MembershipLoader,
        grant_loader: GrantLoader,
        namespace_resolver: MemoryNamespaceResolver,
        assignment_ensurer: MemoryOwnerAssignmentEnsurer,
    ) -> None:
        """initialize the dependency bundle.

        :param membership_loader: actor -> memberships resolver
        :ptype membership_loader: MembershipLoader
        :param grant_loader: groups -> assignments + roles resolver
        :ptype grant_loader: GrantLoader
        :param namespace_resolver: (agent, customer) -> namespace row
        :ptype namespace_resolver: MemoryNamespaceResolver
        :param assignment_ensurer: user memory-owner assignment ensurer
        :ptype assignment_ensurer: MemoryOwnerAssignmentEnsurer
        """
        self.membership_loader = membership_loader
        self.grant_loader = grant_loader
        self.namespace_resolver = namespace_resolver
        self.assignment_ensurer = assignment_ensurer


async def authorize_memory_access(
    *,
    action: Literal["memory.read", "memory.write", "memory.extract"],
    agent_id: UUID,
    customer_id: UUID,
    caller_user_id: UUID | None,
    caller_agent_id: UUID | None,
    deps: MemoryAuthorizerDependencies,
) -> MemoryNamespaceRow:
    """evaluate an action for a caller against the memory namespace.

    resolves the ``(agent_id, customer_id)`` memory namespace
    (creating idempotently if absent), then calls the unified
    evaluator with the action + caller identity. owner short-circuit
    is handled inside the evaluator; no separate code path here.

    :param action: canonical memory action string
    :ptype action: Literal["memory.read", "memory.write", "memory.extract"]
    :param agent_id: owning agent UUID (memory namespace owner)
    :ptype agent_id: UUID
    :param customer_id: owning customer UUID
    :ptype customer_id: UUID
    :param caller_user_id: invoking user UUID (``None`` for
        agent-only calls, e.g. extractor's owner path)
    :ptype caller_user_id: UUID | None
    :param caller_agent_id: invoking agent UUID (``None`` for
        admin-issued user-only calls); the owner short-circuit fires
        when this equals ``agent_id``
    :ptype caller_agent_id: UUID | None
    :param deps: authorizer dependency bundle
    :ptype deps: MemoryAuthorizerDependencies
    :return: resolved memory namespace row (callers often need the
        id for audit envelopes and subsequent assignment ensure)
    :rtype: MemoryNamespaceRow
    :raises MemoryAccessDenied: when the evaluator denies or when
        the namespace cannot be resolved
    """
    ns_row = await deps.namespace_resolver(agent_id, customer_id)
    if ns_row is None:
        log.info(
            "memory authorize: namespace unresolved",
            extra={
                "extra_data": {
                    "agent_id": str(agent_id),
                    "customer_id": str(customer_id),
                    "action": action,
                    "caller_user_id": (
                        str(caller_user_id) if caller_user_id else None
                    ),
                    "caller_agent_id": (
                        str(caller_agent_id) if caller_agent_id else None
                    ),
                }
            },
        )
        raise MemoryAccessDenied(
            f"memory namespace for agent={agent_id} customer={customer_id} "
            "could not be resolved",
        )

    evaluator_namespace = AclNamespace(
        id=ns_row.id,
        customer_id=ns_row.customer_id,
        namespace_type=ns_row.namespace_type,
        owner_agent_id=ns_row.owner_agent_id,
    )
    eval_ctx = EvaluationContext(
        namespace=evaluator_namespace,
        action=action,
        user_id=caller_user_id,
        agent_id=caller_agent_id,
    )
    decision = await evaluate_decision(
        eval_ctx,
        membership_loader=deps.membership_loader,
        grant_loader=deps.grant_loader,
    )

    if not decision:
        log.info(
            "memory access denied",
            extra={
                "extra_data": {
                    "action": action,
                    "namespace_id": str(ns_row.id),
                    "owner_agent_id": str(ns_row.owner_agent_id),
                    "customer_id": str(ns_row.customer_id),
                    "caller_user_id": (
                        str(caller_user_id) if caller_user_id else None
                    ),
                    "caller_agent_id": (
                        str(caller_agent_id) if caller_agent_id else None
                    ),
                }
            },
        )
        raise MemoryAccessDenied(
            f"evaluator denied {action} on memory namespace {ns_row.id}",
        )

    return ns_row
