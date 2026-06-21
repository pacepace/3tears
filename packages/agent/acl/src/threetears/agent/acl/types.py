"""value types for the rbac evaluator and its trail-mode answer.

every type here is a frozen dataclass. the evaluator does not mutate
its inputs and never holds onto its outputs past return; immutability
keeps the audit trail safe to log, hash, and persist verbatim.

vocabulary:

- **actor** — a user or an agent. actors call into the system; they
  never appear on a grant row directly. instead they are members of
  one or more :class:`Group` and inherit grants via the group.
- **principal** — the only grant principal is :class:`Group`. one row
  on :class:`RoleAssignment` carries one ``group_id``. there is no
  parallel "user grant" or "agent grant" path.
- **scope** — what a single :class:`RoleAssignment` covers. three
  shapes (see :class:`ScopeType`): one specific namespace, every
  namespace of one type within one customer, or universal.
- **role** — named bundle of permissions, shaped
  ``{resource_type: [action, ...]}``. wildcard ``"*"`` is permitted
  for type-agnostic roles.
- **trail** — one chain of "this group, via this assignment, applies
  this role, contributing these actions." an :class:`EvaluationResult`
  contains zero or more trails; the operator reading the result sees
  every grant path that contributed to the answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping
from uuid import UUID

__all__ = [
    "ActorType",
    "EvaluationContext",
    "EvaluationResult",
    "Group",
    "GroupMembership",
    "LimitingSide",
    "MemberType",
    "Namespace",
    "Role",
    "RoleAssignment",
    "ScopeType",
    "Trail",
    "WILDCARD_RESOURCE_TYPE",
]


#: wildcard resource-type marker recognized inside :attr:`Role.permissions`.
#: a role with permissions ``{"*": ["read", "write"]}`` grants every
#: resource type the listed actions; lookups in :class:`Role.actions_for`
#: fall back to the wildcard bucket whenever no type-specific bucket
#: matches.
WILDCARD_RESOURCE_TYPE = "*"


class ActorType(StrEnum):
    """kind of actor whose access is being evaluated.

    :cvar USER: a human caller; resolved to the user-side groups
    :cvar AGENT: an agent caller; resolved to the agent-side groups
    """

    USER = "user"
    AGENT = "agent"


class MemberType(StrEnum):
    """kind of member inside a :class:`Group`.

    :cvar USER: human member; contributes to the user side of an
        intersection evaluation
    :cvar AGENT: agent member; contributes to the agent side of an
        intersection evaluation
    """

    USER = "user"
    AGENT = "agent"


class ScopeType(StrEnum):
    """shape of the scope a :class:`RoleAssignment` covers.

    :cvar NAMESPACE: assignment applies to one specific namespace,
        identified by :attr:`RoleAssignment.scope_namespace_id`.
    :cvar TYPE_CUSTOMER: assignment applies to every namespace of a
        given :attr:`RoleAssignment.scope_namespace_type` within
        :attr:`RoleAssignment.scope_customer_id`. covers "auditor on
        every workspace this customer has."
    :cvar ALL: assignment applies universally; only platform admins
        may create assignments at this scope.
    """

    NAMESPACE = "namespace"
    TYPE_CUSTOMER = "type_customer"
    ALL = "all"


class LimitingSide(StrEnum):
    """which side of an intersection is the smaller / capping set.

    :cvar USER: user-side action set is strictly smaller; the user is
        the cap. answers "i can do less than the agent can."
    :cvar AGENT: agent-side action set is strictly smaller; the agent
        is the cap. answers "the agent is read-only even though i'm
        an admin."
    :cvar EQUAL: both sides agree on the same action set; neither cap
        is reducing the other.
    :cvar NEITHER: at least one side is empty. the evaluation denied
        before any cap could apply.
    """

    USER = "user"
    AGENT = "agent"
    EQUAL = "equal"
    NEITHER = "neither"


@dataclass(frozen=True)
class Namespace:
    """target of an authorization check.

    a namespace row carries the customer it belongs to, the type it
    represents (``workspace``, ``agent``, ``shared``, ``system``, ...),
    and the agent that owns it. ownership short-circuits any grant
    lookup: an agent always has full access to namespaces it owns.

    :ivar id: namespace UUID; the primary key in ``platform.namespaces``
    :ivar customer_id: customer UUID this namespace belongs to, or
        ``None`` for a platform-scoped namespace (``customer_id IS NULL``)
    :ivar namespace_type: type discriminator string
        (``workspace``, ``agent``, ``shared``, ``system``, ...)
    :ivar owner_agent_id: UUID of the agent that owns the physical rows,
        or ``None`` for a namespace with no owning agent (datasource /
        customer / knowledge / ...)
    """

    id: UUID
    customer_id: UUID | None
    namespace_type: str
    owner_agent_id: UUID | None


@dataclass(frozen=True)
class Group:
    """the only grant principal.

    a group has zero or more members of type :class:`MemberType`. when
    ``customer_id`` is ``None`` the group is platform-scoped (admin-
    managed; members may span customers); otherwise the group is
    customer-scoped and every member's customer must match.

    identity is ``id`` (the group UUID) alone — the evaluator and every
    loader key on ``id``, never on ``name``. ``name`` is a human label
    (NOT unique); ``managed_key`` is the deterministic find-or-create
    handle a consuming app sets only on the groups it auto-manages.

    :ivar id: group UUID — the only identity
    :ivar name: human-readable label (NOT unique; what a UI shows)
    :ivar customer_id: owning customer UUID, or ``None`` for platform
        scope
    :ivar managed_key: optional deterministic handle a consuming app
        stamps on the groups it auto-manages (find-or-create key);
        ``None`` for user-created groups. unique-per-scope when present
        (per ``customer_id``); never shown to humans. the platform DDL
        owns the partial-unique index — agent-acl carries the column
        only
    """

    id: UUID
    name: str
    customer_id: UUID | None
    managed_key: str | None = None


@dataclass(frozen=True)
class GroupMembership:
    """a single ``(group, member)`` pair.

    materialized from ``platform.group_members``. the denormalized
    ``customer_id`` mirrors the member's customer at write time and
    lets the membership loader cache by customer without a join back
    to the actor table.

    :ivar group_id: group the member belongs to
    :ivar member_type: kind of member (user / agent)
    :ivar member_id: UUID of the user or agent
    :ivar customer_id: customer the member belongs to (NULL for a
        platform-scoped agent or for admin users not bound to one
        customer; the loader reflects whatever the actor row shows)
    """

    group_id: UUID
    member_type: MemberType
    member_id: UUID
    customer_id: UUID | None


@dataclass(frozen=True)
class Role:
    """named bundle of ``{resource_type: [action, ...]}`` permissions.

    permissions are stored as a JSONB column on ``platform.roles``;
    this dataclass mirrors the shape after deserialization. the
    wildcard resource-type ``"*"`` (see
    :data:`WILDCARD_RESOURCE_TYPE`) is permitted for type-agnostic
    roles like ``Reader``, ``Writer``, ``Admin``.

    :ivar id: role UUID
    :ivar name: role name (unique platform-wide)
    :ivar permissions: ``{resource_type: frozenset(actions)}`` mapping
    :ivar is_built_in: whether this role is shipped by the platform
        (true) or authored by a customer admin (always false today;
        the customer-authored path is reserved for future work)
    """

    id: UUID
    name: str
    permissions: Mapping[str, frozenset[str]]
    is_built_in: bool

    def actions_for(self, resource_type: str) -> frozenset[str]:
        """compute the action set this role grants for a resource type.

        resolution order:

        1. exact-match bucket ``permissions[resource_type]``.
        2. wildcard bucket ``permissions["*"]``.
        3. empty set when neither matched.

        the two buckets are unioned when both are present, which lets
        a role declare "every resource gets read, plus workspace gets
        write" with ``{"*": ["read"], "workspace": ["write"]}``. the
        result is a frozenset so callers can union and intersect
        across multiple roles without copying.

        :param resource_type: namespace type to look up
        :ptype resource_type: str
        :return: action set for the resource type (possibly empty)
        :rtype: frozenset[str]
        """
        type_specific = self.permissions.get(resource_type, frozenset())
        wildcard = self.permissions.get(WILDCARD_RESOURCE_TYPE, frozenset())
        result = type_specific | wildcard
        return result


@dataclass(frozen=True)
class RoleAssignment:
    """binds a :class:`Role` to a :class:`Group` on a scope.

    this is the only grant shape in the system. there is no parallel
    direct-user assignment row and no parallel direct-agent assignment
    row.

    field semantics by ``scope_type``:

    - :attr:`ScopeType.NAMESPACE` — ``scope_namespace_id`` set;
      ``scope_namespace_type`` and ``scope_customer_id`` ignored.
    - :attr:`ScopeType.TYPE_CUSTOMER` — ``scope_namespace_type`` and
      ``scope_customer_id`` set; ``scope_namespace_id`` is ``None``.
    - :attr:`ScopeType.ALL` — every scope_* field is ``None``;
      platform admins only may write rows of this shape.

    :ivar id: assignment UUID
    :ivar role_id: id of the :class:`Role` this assignment grants
    :ivar group_id: id of the :class:`Group` that receives the grant
    :ivar scope_type: kind of scope this assignment covers
    :ivar scope_namespace_id: namespace UUID for namespace-scope; else None
    :ivar scope_namespace_type: namespace type discriminator for
        type_customer-scope; else None
    :ivar scope_customer_id: customer UUID for type_customer-scope;
        else None
    """

    id: UUID
    role_id: UUID
    group_id: UUID
    scope_type: ScopeType
    scope_namespace_id: UUID | None
    scope_namespace_type: str | None
    scope_customer_id: UUID | None

    def covers(self, namespace: Namespace) -> bool:
        """true iff this assignment's scope covers the given namespace.

        scope coverage rules:

        - ``namespace`` scope: ids must match exactly.
        - ``type_customer`` scope: namespace_type must match AND
          customer_id must match. either side mismatch denies.
        - ``all`` scope: covers every namespace unconditionally.

        :param namespace: namespace under evaluation
        :ptype namespace: Namespace
        :return: whether this assignment's scope includes the namespace
        :rtype: bool
        """
        result = False
        if self.scope_type == ScopeType.ALL:
            result = True
        elif self.scope_type == ScopeType.NAMESPACE:
            result = self.scope_namespace_id == namespace.id
        elif self.scope_type == ScopeType.TYPE_CUSTOMER:
            result = (
                self.scope_namespace_type == namespace.namespace_type
                and self.scope_customer_id == namespace.customer_id
            )
        return result


@dataclass(frozen=True)
class EvaluationContext:
    """the question being asked of the evaluator.

    one of ``user_id`` or ``agent_id`` is required for a single-actor
    evaluation; both are required for an intersection evaluation
    (the production hot path). ``action`` is the action string the
    caller is checking for (e.g. ``"read"``, ``"write"``,
    ``"namespace.access"``).

    :ivar namespace: namespace under evaluation
    :ivar action: action string being checked
    :ivar user_id: invoking user UUID, or ``None`` for an agent-only
        evaluation
    :ivar agent_id: invoking agent UUID, or ``None`` for a user-only
        evaluation
    """

    namespace: Namespace
    action: str
    user_id: UUID | None = None
    agent_id: UUID | None = None


@dataclass(frozen=True)
class Trail:
    """one ``(group, assignment, role)`` chain that contributed to a result.

    every row of the introspection trail surfaces the exact data the
    evaluator walked: which group membership matched, which assignment
    that group has, which role that assignment refers to, and which
    actions the role contributed for the namespace's type. an
    :class:`EvaluationResult` may carry multiple trails (multiple grant
    paths produced overlapping action sets); each trail is independent.

    :ivar group: :class:`Group` whose membership applied
    :ivar assignment: :class:`RoleAssignment` covering the namespace
    :ivar role: :class:`Role` referenced by the assignment
    :ivar contributed_actions: actions the role contributed for the
        namespace's resource type (may be empty if the role grants no
        actions on this type — kept on the trail because the
        ``(group, assignment, role)`` triple still matched, the role
        just had nothing to add)
    """

    group: Group
    assignment: RoleAssignment
    role: Role
    contributed_actions: frozenset[str]


@dataclass(frozen=True)
class EvaluationResult:
    """the trail-mode answer the introspection api returns.

    fields populated for every evaluation:

    - :attr:`decision` — overall allow / deny.
    - :attr:`effective_actions` — final action set after intersection
      (or the single side when the evaluation was single-actor).

    fields populated for intersection evaluations only (both
    ``user_id`` and ``agent_id`` were set on the context):

    - :attr:`user_actions` — full set the user side contributed
      before intersection.
    - :attr:`agent_actions` — full set the agent side contributed
      before intersection.
    - :attr:`limiting_side` — which side is capping the effective
      access (or :attr:`LimitingSide.NEITHER` when one side was
      empty).
    - :attr:`user_trails` / :attr:`agent_trails` — every grant path
      contributing on each side, in stable order.

    fields populated for single-actor evaluations:

    - :attr:`trails` — every grant path on the one side that ran.
    - :attr:`user_actions` / :attr:`agent_actions` /
      :attr:`limiting_side` left at defaults (empty / NEITHER).

    :ivar decision: True for allow, False for deny
    :ivar effective_actions: final action set after intersection /
        single-side resolution
    :ivar trails: trails for the single-actor side that ran (empty
        for intersection evaluations; intersection evaluations use
        :attr:`user_trails` and :attr:`agent_trails` instead)
    :ivar user_actions: action set the user side contributed
        (intersection mode)
    :ivar agent_actions: action set the agent side contributed
        (intersection mode)
    :ivar limiting_side: which side is capping (intersection mode)
    :ivar user_trails: user-side grant paths (intersection mode)
    :ivar agent_trails: agent-side grant paths (intersection mode)
    :ivar agent_owner_short_circuited: True when the agent side
        short-circuited via :attr:`Namespace.owner_agent_id` match;
        no agent-side trails will be present in this case because
        ownership is not a grant
    """

    decision: bool
    effective_actions: frozenset[str]
    trails: tuple[Trail, ...] = field(default_factory=tuple)
    user_actions: frozenset[str] = field(default_factory=frozenset)
    agent_actions: frozenset[str] = field(default_factory=frozenset)
    limiting_side: LimitingSide = LimitingSide.NEITHER
    user_trails: tuple[Trail, ...] = field(default_factory=tuple)
    agent_trails: tuple[Trail, ...] = field(default_factory=tuple)
    agent_owner_short_circuited: bool = False
