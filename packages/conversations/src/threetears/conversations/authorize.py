"""conversation access authorization helper.

namespace-task-01 phase 10 moves conversation reads, writes, and
deletes behind the unified rbac evaluator. every admin / user-initiated
call that reads conversation history or writes (sends a message into)
a conversation resolves the per-(agent, customer) conversation
namespace, then evaluates the requested action
(``conversation.read`` / ``conversation.write`` /
``conversation.delete``) for the caller's ``(user_id, agent_id)`` pair.

the module mirrors :mod:`threetears.agent.memory.authorize` by design:
conversations and memories share the same namespace granularity — one
namespace per (agent, customer) with per-user owner grants
auto-materialized on first write — because that is the shape the
unified evaluator + admin surfaces were designed around. the singular
difference is the row-count semantics: memory has one row per
memory inside the namespace, conversation has one row per
conversation inside the namespace. per-row authorization is NOT the
goal; the namespace is the grantable resource.

owner short-circuit: when the calling agent owns the conversation
namespace (agent-internal reads — the runtime loading a conversation
row to append a message, extractors fetching the conversation for
memory extraction), the evaluator's built-in owner rule short-circuits
every action to allow without a grant. agent-internal writes
therefore never require assignment configuration.

user-side enforcement: when a user explicitly reads a conversation
(admin surfacing conversation history in the admin UI, a user
retrieving past conversations through a user-facing surface), the
evaluator runs the user side. the user side must contribute the action
via an assignment on the ``conversation`` namespace (or a wider scope).
a user without an assignment is denied with
:class:`ConversationAccessDenied` — distinct from "no conversations"
so callers can surface the denial cleanly.

auto-assignment on first write: user-initiated writes that succeed
trigger :func:`ensure_conversation_owner_assignment` so the user's
per-user group (``conversation-owner:{user_id.hex}``) is bound to the
platform ``ConversationOwner`` role scoped to this conversation
namespace. this is the write-time equivalent of phase 3's memory-owner
auto-assignment: declarative "the user who sends the first message
owns the conversation thread" intent materialized into a concrete rbac
row so the evaluator can answer subsequent questions from cache.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import NAMESPACE_DNS, UUID, uuid5

from threetears.agent.acl import (
    AccessDenied,
    AclCache,
    authorize_on_entity,
)
from threetears.core.namespaces import (
    PLURAL_PREFIX_CONVERSATION,
    build_namespace_name,
)
from threetears.observe import get_logger

__all__ = [
    "ACTION_CONVERSATION_DELETE",
    "ACTION_CONVERSATION_READ",
    "ACTION_CONVERSATION_WRITE",
    "CONVERSATION_NAMESPACE_TYPE",
    "CONVERSATION_OWNER_GROUP_PREFIX",
    "CONVERSATION_OWNER_ROLE_NAME",
    "ConversationAccessDenied",
    "ConversationAuthorizerDependencies",
    "authorize_conversation_access",
    "conversation_namespace_name",
    "ensure_conversation_owner_assignment",
]

log = get_logger(__name__)


#: namespace_type discriminator for conversation rows in
#: ``platform.namespaces``. matches the closed-set admitted by the
#: v042 CHECK constraint (which widens v037's 10-value set to 11).
CONVERSATION_NAMESPACE_TYPE = "conversation"


#: canonical action string for conversation reads. evaluated against
#: the ``conversation`` bucket of the caller's roles via
#: :meth:`~threetears.agent.acl.types.Role.actions_for`.
ACTION_CONVERSATION_READ: Literal["conversation.read"] = "conversation.read"


#: canonical action string for conversation writes (sending messages
#: into the conversation). distinct from ``conversation.read`` so
#: operators can grant "user may browse conversation history but may
#: not append new messages" without also granting write.
ACTION_CONVERSATION_WRITE: Literal["conversation.write"] = "conversation.write"


#: canonical action string for conversation deletes. rarely granted —
#: a user almost never owns delete on their own conversation (the
#: hub's GDPR flow deletes via the admin path). kept distinct so
#: operators can audit delete separately from write.
ACTION_CONVERSATION_DELETE: Literal["conversation.delete"] = "conversation.delete"


#: role name seeded by the hub v043 migration for the per-user
#: owner grant. the auto-assignment at first message-send binds the
#: per-user group to this role scoped to the conversation namespace.
CONVERSATION_OWNER_ROLE_NAME = "ConversationOwner"


#: prefix for the auto-generated per-user conversation-owner group
#: name. final shape is ``conversation-owner:<user_id_hex>``;
#: customer-scoped so one group row per user per customer suffices.
CONVERSATION_OWNER_GROUP_PREFIX = "conversation-owner"


class ConversationAccessDenied(AccessDenied):
    """raised when the evaluator denies a conversation access.

    carries the action and the caller's identity dimensions so
    downstream handlers can surface a precise denial reason instead of
    a silent empty-set fallback. the admin endpoint, the runtime
    loader, and any user-facing conversation tools all catch-or-
    propagate this exception as the single denial type.
    """


def conversation_namespace_name(agent_id: UUID, customer_id: UUID) -> str:
    """build the canonical conversation namespace name for an (agent, customer) pair.

    shape: ``conversations.<agent_id_hex[:8]>.<customer_id_hex[:8]>``
    per the canonical plural-prefix + dot-separator form pinned by
    :func:`threetears.core.namespaces.build_namespace_name`. uses the
    first 8 hex chars of each UUID per the task shard convention; the
    uniqueness is carried by the full
    (namespace_type, owner_agent_id, customer_id) tuple on the row
    itself, so the short prefix is a human-readable display handle and
    not a uniqueness key.

    :param agent_id: owning agent UUID
    :ptype agent_id: UUID
    :param customer_id: owning customer UUID
    :ptype customer_id: UUID
    :return: canonical namespace name
    :rtype: str
    """
    return build_namespace_name(
        PLURAL_PREFIX_CONVERSATION,
        agent_id.hex[:8],
        customer_id.hex[:8],
    )


def conversation_namespace_schema_name(
    agent_id: UUID, customer_id: UUID,
) -> str:
    """build the schema_name persisted on ``platform.namespaces`` rows.

    conversation rows route through the shared per-agent database
    schema (or the hub's ``platform.conversations`` table) rather than
    a per-namespace Postgres schema; the row carries a stable synthetic
    schema string so SELECT queries joining namespaces on
    ``schema_name`` still match. shape mirrors
    :func:`conversation_namespace_name` with a ``conversation__``
    prefix to keep the schema namespace disjoint from the display-name
    namespace.

    :param agent_id: owning agent UUID
    :ptype agent_id: UUID
    :param customer_id: owning customer UUID
    :ptype customer_id: UUID
    :return: schema name string
    :rtype: str
    """
    return f"conversation__{agent_id.hex[:8]}__{customer_id.hex[:8]}"


class ConversationAuthorizerDependencies:
    """bundle of Collections + ACL cache the authorizer uses at call time.

    mirrors :class:`threetears.agent.memory.authorize.MemoryAuthorizerDependencies`:
    acl-evaluator-helpers-migration retired the standalone
    ``membership_loader`` / ``grant_loader`` fields — the canonical
    primitive consumes loaders via the :class:`AclCache`. downstream
    call sites (admin endpoint, runtime loader, user-facing tools)
    carry a single bundle through their constructors so their
    signatures stay single-parameter.

    the Collections are typed ``Any`` at this layer because
    :class:`NamespaceCollection`, :class:`GroupCollection`,
    :class:`GroupMemberCollection`, :class:`RoleCollection`, and
    :class:`RoleAssignmentCollection` live in
    :mod:`aibots.hub.*` — a higher layer than ``conversations``.
    wiring code constructs the bundle with concrete Collection
    instances; this module only uses their documented method surface.

    :ivar acl_cache: shared :class:`AclCache` carrying loaders + ttl
        layers
    :ivar namespace_collection: three-tier ``NamespaceCollection``
        used to resolve conversation namespaces by
        ``(namespace_type, owner_agent_id, customer_id)``
    :ivar group_collection: three-tier ``GroupCollection`` used by
        :func:`ensure_conversation_owner_assignment` for per-user group
        creation / lookup
    :ivar group_member_collection: three-tier ``GroupMemberCollection``
        used by :func:`ensure_conversation_owner_assignment` for
        per-user group membership binding
    :ivar role_collection: three-tier ``RoleCollection`` used to look
        up the platform ``ConversationOwner`` role id via
        :meth:`list_builtin`
    :ivar role_assignment_collection: three-tier
        ``RoleAssignmentCollection`` used by
        :func:`ensure_conversation_owner_assignment` via
        :meth:`ensure_group_role_assignment`
    """

    __slots__ = (
        "acl_cache",
        "namespace_collection",
        "group_collection",
        "group_member_collection",
        "role_collection",
        "role_assignment_collection",
    )

    def __init__(
        self,
        *,
        acl_cache: AclCache,
        namespace_collection: Any,
        group_collection: Any,
        group_member_collection: Any,
        role_collection: Any,
        role_assignment_collection: Any,
    ) -> None:
        """initialize the dependency bundle.

        :param acl_cache: shared :class:`AclCache`
        :ptype acl_cache: AclCache
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
        self.namespace_collection = namespace_collection
        self.group_collection = group_collection
        self.group_member_collection = group_member_collection
        self.role_collection = role_collection
        self.role_assignment_collection = role_assignment_collection


async def _resolve_or_create_conversation_namespace(
    *,
    agent_id: UUID,
    customer_id: UUID,
    namespace_collection: Any,
) -> Any:
    """return the conversation namespace entity for the (agent, customer) pair.

    looks up the triple ``(conversation, agent_id, customer_id)`` via
    :meth:`NamespaceCollection.get_by_owner_and_customer`. when no row
    matches, constructs a new :class:`NamespaceEntity` with a
    deterministic :func:`uuid5` id keyed on the triple (so concurrent
    first-write racers converge on the same id without SELECT FOR
    UPDATE) and calls :meth:`save_entity`; the Collection's
    ``ON CONFLICT (id) DO UPDATE`` path makes the save idempotent under
    replay.

    raises :class:`ConversationAccessDenied` if the Collection yields
    ``None`` AND a save cannot create the row (configuration error: the
    Collection has no L3 pool bound, or the save raised a permission
    error). callers surface that as a cleanly-typed denial rather than
    a bare ``RuntimeError``.

    :param agent_id: owning agent UUID
    :ptype agent_id: UUID
    :param customer_id: owning customer UUID
    :ptype customer_id: UUID
    :param namespace_collection: three-tier ``NamespaceCollection``
    :ptype namespace_collection: Any
    :return: resolved namespace entity
    :rtype: Any
    :raises ConversationAccessDenied: when the namespace cannot be
        resolved or created
    """
    existing = await namespace_collection.get_by_owner_and_customer(
        namespace_type=CONVERSATION_NAMESPACE_TYPE,
        owner_agent_id=agent_id,
        customer_id=customer_id,
    )
    if existing is not None:
        return existing

    # deterministic id: same (agent, customer) -> same uuid5 so two
    # concurrent first-writes converge on the same namespace row via
    # ON CONFLICT (id) DO UPDATE. matches the v044 backfill id scheme.
    new_id = uuid5(
        NAMESPACE_DNS,
        f"threetears.namespaces.conversation.{agent_id.hex}.{customer_id.hex}",
    )
    now = datetime.now(UTC).replace(tzinfo=None)
    entity = namespace_collection.entity_class(
        {
            "id": new_id,
            "name": conversation_namespace_name(agent_id, customer_id),
            "namespace_type": CONVERSATION_NAMESPACE_TYPE,
            "owner_agent_id": agent_id,
            "customer_id": customer_id,
            "schema_name": conversation_namespace_schema_name(
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
        raise ConversationAccessDenied(
            f"conversation namespace for agent={agent_id} "
            f"customer={customer_id} could not be created: {exc}",
        ) from exc

    # re-read to return the authoritative Collection-managed entity.
    # the freshly-constructed handle may not be bound into L1 if the
    # save path did not promote it. fall back to the freshly-
    # constructed entity if the re-read misses — that can happen when
    # the Collection's L3 round-trip lags the save.
    resolved = await namespace_collection.get_by_owner_and_customer(
        namespace_type=CONVERSATION_NAMESPACE_TYPE,
        owner_agent_id=agent_id,
        customer_id=customer_id,
    )
    return resolved if resolved is not None else entity


async def authorize_conversation_access(
    *,
    action: Literal[
        "conversation.read", "conversation.write", "conversation.delete",
    ],
    agent_id: UUID,
    customer_id: UUID,
    caller_user_id: UUID | None,
    caller_agent_id: UUID | None,
    deps: ConversationAuthorizerDependencies,
) -> Any:
    """evaluate an action for a caller against the conversation namespace.

    resolves the ``(agent_id, customer_id)`` conversation namespace
    (creating idempotently if absent via the Collection), then calls
    the unified evaluator with the action + caller identity. owner
    short-circuit is handled inside the evaluator; no separate code
    path here.

    :param action: canonical conversation action string
    :ptype action: Literal["conversation.read", "conversation.write",
        "conversation.delete"]
    :param agent_id: owning agent UUID (conversation namespace owner)
    :ptype agent_id: UUID
    :param customer_id: owning customer UUID
    :ptype customer_id: UUID
    :param caller_user_id: invoking user UUID (``None`` for
        agent-only calls, e.g. runtime-internal read path)
    :ptype caller_user_id: UUID | None
    :param caller_agent_id: invoking agent UUID (``None`` for
        admin-issued user-only calls); the owner short-circuit fires
        when this equals ``agent_id``
    :ptype caller_agent_id: UUID | None
    :param deps: authorizer dependency bundle
    :ptype deps: ConversationAuthorizerDependencies
    :return: resolved conversation namespace entity (callers often
        need the id for audit envelopes and subsequent assignment
        ensure)
    :rtype: Any
    :raises ConversationAccessDenied: when the evaluator denies or
        when the namespace cannot be resolved / created
    """
    ns_entity = await _resolve_or_create_conversation_namespace(
        agent_id=agent_id,
        customer_id=customer_id,
        namespace_collection=deps.namespace_collection,
    )
    try:
        await authorize_on_entity(
            ns_entity=ns_entity,
            action=action,
            user_id=caller_user_id,
            agent_id=caller_agent_id,
            cache=deps.acl_cache,
            namespace_name=conversation_namespace_name(agent_id, customer_id),
        )
    except AccessDenied as exc:
        raise ConversationAccessDenied(
            f"evaluator denied {action} on conversation namespace "
            f"{ns_entity.id}",
        ) from exc
    return ns_entity


async def ensure_conversation_owner_assignment(
    *,
    user_id: UUID,
    namespace: Any,
    deps: ConversationAuthorizerDependencies,
) -> None:
    """ensure the per-user ConversationOwner group + assignment rows exist.

    materializes three rows idempotently on the first time a user
    sends a message in a conversation within the namespace:

    1. ``platform.groups`` row named
       ``conversation-owner:<user_id_hex>`` with a deterministic
       :func:`uuid5` id keyed on ``(customer_id, user_id)``
    2. ``platform.group_members`` row binding ``user_id`` to the
       group with a deterministic :func:`uuid5` id keyed on
       ``(group_id, user_id)``
    3. ``platform.role_assignments`` row binding the group to the
       platform ``ConversationOwner`` role scoped to ``namespace.id``
       via
       :meth:`RoleAssignmentCollection.ensure_group_role_assignment`

    logs + returns without raising when the platform
    ``ConversationOwner`` role is not present in the builtin role
    catalog — operators see the wiring gap in logs, existing grants
    continue to work, and future reads by this user are denied (the
    intended fail-closed behavior before the role is seeded).

    idempotent by construction: every identifier derives from a
    :func:`uuid5` so concurrent racers resolve to the same rows; every
    Collection save path is an ON CONFLICT (id) DO UPDATE; the
    ensure_group_role_assignment helper is SELECT-then-INSERT by
    ``(group, role, scope)`` tuple.

    :param user_id: user UUID asked to be bound to the
        ConversationOwner grant
    :ptype user_id: UUID
    :param namespace: resolved conversation :class:`NamespaceEntity` —
        the ensure call scopes the assignment to ``namespace.id``
    :ptype namespace: Any
    :param deps: authorizer dependency bundle carrying the Collections
        the ensurer needs
    :ptype deps: ConversationAuthorizerDependencies
    :return: nothing
    :rtype: None
    """
    builtin_roles = await deps.role_collection.list_builtin()
    owner_role_id: UUID | None = None
    for role in builtin_roles:
        if role.name == CONVERSATION_OWNER_ROLE_NAME:
            owner_role_id = role.id
            break
    if owner_role_id is None:
        log.warning(
            "conversation owner assignment ensure skipped: role %s not "
            "present in builtin role catalog",
            CONVERSATION_OWNER_ROLE_NAME,
        )
        return None

    customer_id = namespace.customer_id
    group_id = uuid5(
        NAMESPACE_DNS,
        f"threetears.groups.{CONVERSATION_OWNER_GROUP_PREFIX}."
        f"{customer_id.hex}.{user_id.hex}",
    )
    group_name = f"{CONVERSATION_OWNER_GROUP_PREFIX}:{user_id.hex}"
    now = datetime.now(UTC).replace(tzinfo=None)

    existing_group = await deps.group_collection.get(group_id)
    if existing_group is None:
        group_entity = deps.group_collection.entity_class(
            {
                "id": group_id,
                "customer_id": customer_id,
                "name": group_name,
                "description": "auto (conversation owner)",
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
