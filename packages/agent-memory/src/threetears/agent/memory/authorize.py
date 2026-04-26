"""memory access authorization helper.

namespace-task-01 phase 3 moves memory reads, writes, and extracts
behind the unified rbac evaluator. every call into the memory
collection, retriever, extractor, or user-facing tool resolves the
per-(agent, customer) memory namespace, then evaluates the requested
action (``memory.read`` / ``memory.write`` / ``memory.extract``) for
the caller's ``(user_id, agent_id)`` pair.

three-tier-task-01 phase D retired the bespoke callable aliases
and the parallel value object that previously mirrored
:class:`aibots.hub.broker.namespaces.NamespaceEntity`. the authorizer
now takes a :class:`MemoryAuthorizerDependencies` bundle carrying
the Collections directly — every lookup call is a Collection
method call, every assignment-ensure call is a Collection method
call. downstream code never constructs bespoke callable adapters.

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

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import NAMESPACE_DNS, UUID, uuid5

from threetears.agent.acl import (
    AccessDenied,
    EvaluationContext,
    GrantLoader,
    MembershipLoader,
    Namespace as AclNamespace,
    evaluate_decision,
)
from threetears.core.namespaces import PLURAL_PREFIX_MEMORY, build_namespace_name
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
    "authorize_memory_access",
    "ensure_memory_owner_assignment",
    "memory_namespace_name",
]

log = get_logger(__name__)


#: namespace_type discriminator for memory rows in ``platform.namespaces``.
#: matches the closed-set admitted by the v018 CHECK constraint.
MEMORY_NAMESPACE_TYPE = "memory"


#: canonical action string for memory reads. evaluated against the
#: ``memory`` bucket of the caller's roles via
#: :meth:`~threetears.agent.acl.types.Role.actions_for`.
ACTION_MEMORY_READ: Literal["memory.read"] = "memory.read"


#: canonical action string for user-initiated memory writes.
ACTION_MEMORY_WRITE: Literal["memory.write"] = "memory.write"


#: canonical action string for agent-internal memory extraction.
#: distinct from ``memory.write`` so operators can grant / audit
#: "LLM emitted memories on the user's behalf" separately from
#: "user explicitly added a memory."
ACTION_MEMORY_EXTRACT: Literal["memory.extract"] = "memory.extract"


#: role name seeded by the hub v020 migration for the per-user
#: owner grant. the auto-assignment at first write binds the
#: per-user group to this role scoped to the memory namespace.
MEMORY_OWNER_ROLE_NAME = "MemoryOwner"


#: prefix for the auto-generated per-user memory-owner group name.
#: final shape is ``memory-owner:<user_id_hex>``; customer-scoped so
#: one group row per user per customer suffices.
MEMORY_OWNER_GROUP_PREFIX = "memory-owner"


class MemoryAccessDenied(AccessDenied):
    """raised when the evaluator denies a memory access.

    carries the action and the caller's identity dimensions so
    downstream handlers can surface a precise denial reason instead
    of a silent empty-set fallback. the retriever, the collection,
    and the memory tools all catch-or-propagate this exception as
    the single denial type.
    """


def memory_namespace_name(agent_id: UUID, customer_id: UUID) -> str:
    """build the canonical memory namespace name for an (agent, customer) pair.

    shape: ``memories.<agent_id_hex[:8]>.<customer_id_hex[:8]>`` per
    the canonical plural-prefix + dot-separator form pinned by
    :func:`threetears.core.namespaces.build_namespace_name`. uses the
    first 8 hex chars of each UUID per the task shard convention; the
    uniqueness is carried by the full
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
    return build_namespace_name(
        PLURAL_PREFIX_MEMORY,
        agent_id.hex[:8],
        customer_id.hex[:8],
    )


def memory_namespace_schema_name(agent_id: UUID, customer_id: UUID) -> str:
    """build the schema_name persisted on ``platform.namespaces`` rows.

    memory rows route through the shared agent database schema rather
    than a per-namespace Postgres schema; the row carries a stable
    synthetic schema string so SELECT queries joining namespaces on
    ``schema_name`` still match. shape mirrors
    :func:`memory_namespace_name` with a ``memory__`` prefix to keep
    the schema namespace disjoint from the display-name namespace.

    :param agent_id: owning agent UUID
    :ptype agent_id: UUID
    :param customer_id: owning customer UUID
    :ptype customer_id: UUID
    :return: schema name string
    :rtype: str
    """
    return f"memory__{agent_id.hex[:8]}__{customer_id.hex[:8]}"


class MemoryAuthorizerDependencies:
    """bundle of Collections + ACL loaders the authorizer uses at call time.

    three-tier-task-01 phase D shrank the bundle from four callables
    (two bespoke resolver / ensurer aliases plus the two ACL loaders)
    to the Collections themselves plus the ACL loaders. every former
    callable is now a direct Collection method on one of these
    handles. downstream call sites (retriever, extractor, collection,
    tools) carry a single bundle through their constructors so their
    signatures stay single-parameter.

    the Collections are typed ``Any`` at this layer because
    :class:`NamespaceCollection`, :class:`GroupCollection`,
    :class:`GroupMemberCollection`, :class:`RoleCollection`, and
    :class:`RoleAssignmentCollection` live in
    :mod:`aibots.hub.*` — a higher layer than ``agent-memory``.
    wiring code constructs the bundle with concrete Collection
    instances; this module only uses their documented method surface.

    :ivar acl_cache: shared :class:`threetears.agent.acl.AclCache`
        instance. reserved for future per-namespace decision caching;
        not read by :func:`authorize_memory_access` today (the
        evaluator hits loaders directly each call) but carried on
        the bundle so downstream retries and Phase E wiring have a
        single handle
    :ivar membership_loader: actor -> memberships resolver
    :ivar grant_loader: groups -> assignments + roles resolver
    :ivar namespace_collection: three-tier ``NamespaceCollection``
        used to resolve memory namespaces by
        ``(namespace_type, owner_agent_id, customer_id)``
    :ivar group_collection: three-tier ``GroupCollection`` used by
        :func:`ensure_memory_owner_assignment` for per-user group
        creation / lookup
    :ivar group_member_collection: three-tier ``GroupMemberCollection``
        used by :func:`ensure_memory_owner_assignment` for per-user
        group membership binding
    :ivar role_collection: three-tier ``RoleCollection`` used to look
        up the platform ``MemoryOwner`` role id via
        :meth:`list_builtin`
    :ivar role_assignment_collection: three-tier
        ``RoleAssignmentCollection`` used by
        :func:`ensure_memory_owner_assignment` via
        :meth:`ensure_group_role_assignment`
    """

    __slots__ = (
        "acl_cache",
        "membership_loader",
        "grant_loader",
        "namespace_collection",
        "group_collection",
        "group_member_collection",
        "role_collection",
        "role_assignment_collection",
    )

    def __init__(
        self,
        *,
        acl_cache: Any,
        membership_loader: MembershipLoader,
        grant_loader: GrantLoader,
        namespace_collection: Any,
        group_collection: Any,
        group_member_collection: Any,
        role_collection: Any,
        role_assignment_collection: Any,
    ) -> None:
        """initialize the dependency bundle.

        :param acl_cache: shared :class:`threetears.agent.acl.AclCache`
        :ptype acl_cache: Any
        :param membership_loader: actor -> memberships resolver
        :ptype membership_loader: MembershipLoader
        :param grant_loader: groups -> assignments + roles resolver
        :ptype grant_loader: GrantLoader
        :param namespace_collection: three-tier ``NamespaceCollection``
        :ptype namespace_collection: Any
        :param group_collection: three-tier ``GroupCollection``
        :ptype group_collection: Any
        :param group_member_collection: three-tier
            ``GroupMemberCollection``
        :ptype group_member_collection: Any
        :param role_collection: three-tier ``RoleCollection``
        :ptype role_collection: Any
        :param role_assignment_collection: three-tier
            ``RoleAssignmentCollection``
        :ptype role_assignment_collection: Any
        """
        self.acl_cache = acl_cache
        self.membership_loader = membership_loader
        self.grant_loader = grant_loader
        self.namespace_collection = namespace_collection
        self.group_collection = group_collection
        self.group_member_collection = group_member_collection
        self.role_collection = role_collection
        self.role_assignment_collection = role_assignment_collection


async def _resolve_or_create_memory_namespace(
    *,
    agent_id: UUID,
    customer_id: UUID,
    namespace_collection: Any,
) -> Any:
    """return the memory namespace entity for the (agent, customer) pair.

    looks up the triple ``(memory, agent_id, customer_id)`` via
    :meth:`NamespaceCollection.get_by_owner_and_customer`. when no
    row matches, constructs a new :class:`NamespaceEntity` with a
    deterministic :func:`uuid5` id keyed on the triple (so concurrent
    first-write racers converge on the same id without SELECT FOR
    UPDATE) and calls :meth:`save_entity`; the Collection's
    ``ON CONFLICT (id) DO UPDATE`` path makes the save idempotent
    under replay.

    raises :class:`MemoryAccessDenied` if the Collection yields
    ``None`` AND a save cannot create the row (configuration error:
    the Collection has no L3 pool bound, or the save raised a
    permission error). callers surface that as a cleanly-typed denial
    rather than a bare ``RuntimeError``.

    :param agent_id: owning agent UUID
    :ptype agent_id: UUID
    :param customer_id: owning customer UUID
    :ptype customer_id: UUID
    :param namespace_collection: three-tier ``NamespaceCollection``
    :ptype namespace_collection: Any
    :return: resolved namespace entity
    :rtype: Any
    :raises MemoryAccessDenied: when the namespace cannot be resolved
        or created
    """
    existing = await namespace_collection.get_by_owner_and_customer(
        namespace_type=MEMORY_NAMESPACE_TYPE,
        owner_agent_id=agent_id,
        customer_id=customer_id,
    )
    if existing is not None:
        return existing

    # deterministic id: same (agent, customer) -> same uuid5 so two
    # concurrent first-writes converge on the same namespace row via
    # ON CONFLICT (id) DO UPDATE. the uuid5 namespace/name scheme
    # matches the group-id + membership-id pattern used by the
    # memory-owner ensure path, keeping all derived ids uniformly
    # deterministic.
    new_id = uuid5(
        NAMESPACE_DNS,
        f"threetears.namespaces.memory.{agent_id.hex}.{customer_id.hex}",
    )
    now = datetime.now(UTC).replace(tzinfo=None)
    entity = namespace_collection.entity_class(
        {
            "id": new_id,
            "name": memory_namespace_name(agent_id, customer_id),
            "namespace_type": MEMORY_NAMESPACE_TYPE,
            "owner_agent_id": agent_id,
            "customer_id": customer_id,
            "schema_name": memory_namespace_schema_name(
                agent_id, customer_id,
            ),
            "metadata": {},
            "date_created": now,
            "date_updated": now,
        },
        is_new=True,
        collection=namespace_collection,
    )
    try:
        await namespace_collection.save_entity(entity)
    except Exception as exc:
        raise MemoryAccessDenied(
            f"memory namespace for agent={agent_id} customer={customer_id} "
            f"could not be created: {exc}",
        ) from exc

    # re-read to return the authoritative Collection-managed entity
    # (the freshly-constructed handle may not be bound into L1 if the
    # save path did not promote it). fall back to the freshly-
    # constructed entity if the re-read misses — that can happen when
    # the Collection's L3 round-trip lags the save.
    resolved = await namespace_collection.get_by_owner_and_customer(
        namespace_type=MEMORY_NAMESPACE_TYPE,
        owner_agent_id=agent_id,
        customer_id=customer_id,
    )
    return resolved if resolved is not None else entity


async def authorize_memory_access(
    *,
    action: Literal["memory.read", "memory.write", "memory.extract"],
    agent_id: UUID,
    customer_id: UUID,
    caller_user_id: UUID | None,
    caller_agent_id: UUID | None,
    deps: MemoryAuthorizerDependencies,
) -> Any:
    """evaluate an action for a caller against the memory namespace.

    resolves the ``(agent_id, customer_id)`` memory namespace
    (creating idempotently if absent via the Collection), then calls
    the unified evaluator with the action + caller identity. owner
    short-circuit is handled inside the evaluator; no separate code
    path here.

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
    :return: resolved memory namespace entity (callers often need the
        id for audit envelopes and subsequent assignment ensure)
    :rtype: Any
    :raises MemoryAccessDenied: when the evaluator denies or when
        the namespace cannot be resolved / created
    """
    ns_entity = await _resolve_or_create_memory_namespace(
        agent_id=agent_id,
        customer_id=customer_id,
        namespace_collection=deps.namespace_collection,
    )

    evaluator_namespace = AclNamespace(
        id=ns_entity.id,
        customer_id=ns_entity.customer_id,
        namespace_type=ns_entity.namespace_type,
        owner_agent_id=ns_entity.owner_agent_id,
    )
    eval_ctx = EvaluationContext(
        namespace=evaluator_namespace,
        action=action,
        user_id=caller_user_id,
        agent_id=caller_agent_id,
    )
    decision = await evaluate_decision(eval_ctx, cache=deps.acl_cache)

    if not decision:
        log.info(
            "memory access denied",
            extra={
                "extra_data": {
                    "action": action,
                    "namespace_id": str(ns_entity.id),
                    "owner_agent_id": str(ns_entity.owner_agent_id),
                    "customer_id": str(ns_entity.customer_id),
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
            f"evaluator denied {action} on memory namespace {ns_entity.id}",
        )

    return ns_entity


async def ensure_memory_owner_assignment(
    *,
    user_id: UUID,
    namespace: Any,
    deps: MemoryAuthorizerDependencies,
) -> None:
    """ensure the per-user MemoryOwner group + assignment rows exist.

    replaces the Phase-C bespoke ensurer callable. materializes three
    rows idempotently:

    1. ``platform.groups`` row named ``memory-owner:<user_id_hex>``
       with a deterministic :func:`uuid5` id keyed on
       ``(customer_id, user_id)``
    2. ``platform.group_members`` row binding ``user_id`` to the group
       with a deterministic :func:`uuid5` id keyed on
       ``(group_id, user_id)``
    3. ``platform.role_assignments`` row binding the group to the
       platform ``MemoryOwner`` role scoped to ``namespace.id`` via
       :meth:`RoleAssignmentCollection.ensure_group_role_assignment`

    logs + returns without raising when the platform ``MemoryOwner``
    role is not present in the builtin role catalog — operators see
    the wiring gap in logs, existing grants continue to work, and
    future reads by this user are denied (the intended fail-closed
    behavior before the role is seeded).

    idempotent by construction: every identifier derives from a
    :func:`uuid5` so concurrent racers resolve to the same rows; every
    Collection save path is an ON CONFLICT (id) DO UPDATE; the
    ensure_group_role_assignment helper is SELECT-then-INSERT by
    ``(group, role, scope)`` tuple.

    :param user_id: user UUID asked to be bound to the MemoryOwner
        grant
    :ptype user_id: UUID
    :param namespace: resolved memory :class:`NamespaceEntity` — the
        ensure call scopes the assignment to ``namespace.id``
    :ptype namespace: Any
    :param deps: authorizer dependency bundle carrying the
        Collections the ensurer needs
    :ptype deps: MemoryAuthorizerDependencies
    :return: nothing
    :rtype: None
    """
    builtin_roles = await deps.role_collection.list_builtin()
    owner_role_id: UUID | None = None
    for role in builtin_roles:
        if role.name == MEMORY_OWNER_ROLE_NAME:
            owner_role_id = role.id
            break
    if owner_role_id is None:
        log.warning(
            "memory owner assignment ensure skipped: role %s not "
            "present in builtin role catalog",
            MEMORY_OWNER_ROLE_NAME,
        )
        return None

    customer_id = namespace.customer_id
    group_id = uuid5(
        NAMESPACE_DNS,
        f"threetears.groups.{MEMORY_OWNER_GROUP_PREFIX}."
        f"{customer_id.hex}.{user_id.hex}",
    )
    group_name = f"{MEMORY_OWNER_GROUP_PREFIX}:{user_id.hex}"
    now = datetime.now(UTC).replace(tzinfo=None)

    existing_group = await deps.group_collection.get(group_id)
    if existing_group is None:
        group_entity = deps.group_collection.entity_class(
            {
                "id": group_id,
                "customer_id": customer_id,
                "name": group_name,
                "description": "auto (memory owner)",
                "date_created": now,
                "date_updated": now,
            },
            is_new=True,
            collection=deps.group_collection,
        )
        await deps.group_collection.save_entity(group_entity)

    membership_id = uuid5(
        NAMESPACE_DNS,
        f"threetears.group_members.{group_id.hex}.{user_id.hex}",
    )
    existing_member = await deps.group_member_collection.get(membership_id)
    if existing_member is None:
        member_entity = deps.group_member_collection.entity_class(
            {
                "id": membership_id,
                "group_id": group_id,
                "member_type": "user",
                "member_id": user_id,
                "customer_id": customer_id,
                "date_added": now,
            },
            is_new=True,
            collection=deps.group_member_collection,
        )
        await deps.group_member_collection.save_entity(member_entity)

    await deps.role_assignment_collection.ensure_group_role_assignment(
        group_id=group_id,
        role_id=owner_role_id,
        scope_type="namespace",
        scope_id=namespace.id,
    )
    # ensure_group_role_assignment returns the assignment id; callers
    # don't need it (ensure is fire-and-forget idempotency).
    return None
