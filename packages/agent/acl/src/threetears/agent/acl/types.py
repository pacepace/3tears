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
- **scope** — what a single :class:`RoleAssignment` covers. four
  shapes (see :class:`ScopeType`): one specific namespace, one
  namespace-name node and everything beneath it, every namespace of
  one type within one customer, or universal.
- **role** — named bundle of permissions, shaped
  ``{resource_type: [action, ...]}``. wildcard ``"*"`` is permitted
  for type-agnostic roles.
- **trail** — one chain of "this group, via this assignment, applies
  this role, contributing these actions." an :class:`EvaluationResult`
  contains zero or more trails; the operator reading the result sees
  every grant path that contributed to the answer.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final
from typing import Mapping
from uuid import UUID

from threetears.core.namespaces import namespace_contains

__all__ = [
    "ACTION_ARGUMENT_SEPARATOR",
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
    "action_acts_on_sql",
    "decode_role_permissions",
    "permissions_act_on",
]


#: wildcard resource-type marker recognized inside :attr:`Role.permissions`.
#: a role with permissions ``{"*": ["read", "write"]}`` grants every
#: resource type the listed actions; lookups in :class:`Role.actions_for`
#: fall back to the wildcard bucket whenever no type-specific bucket
#: matches.
WILDCARD_RESOURCE_TYPE = "*"

#: separator between the namespace type an action is on and its verb, as the platform spells its actions
#: (``memory.read``, ``customer.admin``, ``tool.call``).
_ACTION_TYPE_SEPARATOR = "."

#: separator between a parameterized action's stem and its argument (``read_file_matching:**/*.yaml``), as
#: :data:`~threetears.agent.acl.evaluator.READ_FILE_MATCHING_PREFIX` spells it. the argument is free text, so a ``.``
#: inside it names no type.
ACTION_ARGUMENT_SEPARATOR = ":"


def _action_namespace_type(action: str) -> str | None:
    """the namespace type ``action`` is an action ON, or None for a verb naming none.

    the ONE definition caller visibility counts a grant by (:func:`_action_acts_on`, :func:`action_acts_on_sql`). an
    action's stem is its text before the first :data:`ACTION_ARGUMENT_SEPARATOR` (the whole action when it holds
    none). a stem holding an :data:`_ACTION_TYPE_SEPARATOR` names the type before the first one (``customer.read`` ->
    ``customer``, ``datasource.read_matching:x.y`` -> ``datasource``); a stem holding none names no type -- a bare verb
    (``read``, ``call``) or a parameterized one (``read_file_matching:**/*.yaml``, the stem ``read_file_matching``),
    which acts on whatever type the bucket holding it is.

    WHY a name, not the bucket: a role's bucket is free text the evaluator matches by string, so
    ``{"customer": ["group.read"]}`` "acts" on a customer namespace by holding a non-empty bucket, yet the action is
    one on groups. counted by its bucket, such a role bound on a customer namespace opened every list of that
    customer (Pace's ruling 2026-11-20, option A). the platform's convention prefixes an action with its type
    (:mod:`threetears.agent.acl.catalog`); its shipped exceptions are the verbs naming no type, which this keeps.

    :param action: the action string, as a role's bucket holds it
    :ptype action: str
    :return: the type it names, or None
    :rtype: str | None
    """
    stem = action.split(ACTION_ARGUMENT_SEPARATOR, 1)[0]
    head, separator, _ = stem.partition(_ACTION_TYPE_SEPARATOR)
    return head if separator else None


def _action_acts_on(action: str, namespace_type: str) -> bool:
    """whether ``action`` is an action on ``namespace_type``'s own: it names that type, or names none
    (:func:`_action_namespace_type`).

    :param action: the action string
    :ptype action: str
    :param namespace_type: the namespace type
    :ptype namespace_type: str
    :return: whether it acts there
    :rtype: bool
    """
    named = _action_namespace_type(action)
    return named is None or named == namespace_type


def action_acts_on_sql(action_sql: str, type_sql: str) -> str:
    """:func:`_action_acts_on` spelled in SQL, over an action expression and a type expression.

    ``split_part(<action>, ':', 1)`` is the stem (the whole action when it holds no ``:``, as Python's
    ``split(':', 1)[0]``); ``strpos(<stem>, '.') = 0`` is a stem naming no type, and ``split_part(<stem>, '.', 1)`` the
    type it names. held equal to the Python rule on PostgreSQL and YugabyteDB by
    ``test_the_sql_and_python_action_rules_agree``.

    :param action_sql: SQL yielding the action text: a trusted column or alias, or a ``$N`` placeholder; never user
        input
    :ptype action_sql: str
    :param type_sql: SQL yielding the namespace type: a trusted column or a ``$N`` placeholder; never user input
    :ptype type_sql: str
    :return: boolean SQL expression
    :rtype: str
    """
    argument = ACTION_ARGUMENT_SEPARATOR.replace("'", "''")
    dot = _ACTION_TYPE_SEPARATOR.replace("'", "''")
    stem = f"split_part({action_sql}, '{argument}', 1)"
    return f"(strpos({stem}, '{dot}') = 0 OR split_part({stem}, '{dot}', 1) = {type_sql})"


def decode_role_permissions(raw: Any) -> dict[str, frozenset[str]]:
    """a role's stored ``permissions`` as ``{resource_type: frozenset(actions)}``: how every reader of a role row reads
    it (:class:`threetears.agent.acl.RoleCollection`, :func:`threetears.agent.acl.visible_customers`).

    asyncpg answers a JSONB column as a parsed ``dict`` (the JSONB codec registered) or as its raw ``str``, and a
    double-encoded write stores the object as a JSON string, which decodes to a ``str`` holding it: each lands here.
    A bucket that is not a list is no bucket.

    :param raw: the column's value as the driver answers it
    :ptype raw: Any
    :return: the permissions
    :rtype: dict[str, frozenset[str]]
    """
    parsed: dict[str, Any] = {}
    if isinstance(raw, dict):
        parsed = raw
    elif isinstance(raw, str) and raw:
        loaded = json.loads(raw)
        if isinstance(loaded, str) and loaded:
            loaded = json.loads(loaded)
        if isinstance(loaded, dict):
            parsed = loaded
    result: dict[str, frozenset[str]] = {}
    for resource_type, actions in parsed.items():
        if isinstance(actions, list):
            result[resource_type] = frozenset(str(a) for a in actions)
    return result


def _granted_actions(permissions: Mapping[str, frozenset[str]], resource_type: str) -> frozenset[str]:
    """the actions ``permissions`` grant on ``resource_type``: its own bucket and the :data:`WILDCARD_RESOURCE_TYPE`
    bucket, unioned (:meth:`Role.actions_for`, the one rule).

    :param permissions: a role's permissions
    :ptype permissions: Mapping[str, frozenset[str]]
    :param resource_type: the namespace type
    :ptype resource_type: str
    :return: the actions, possibly none
    :rtype: frozenset[str]
    """
    type_specific = permissions.get(resource_type, frozenset())
    wildcard = permissions.get(WILDCARD_RESOURCE_TYPE, frozenset())
    return frozenset(type_specific | wildcard)


def permissions_act_on(permissions: Mapping[str, frozenset[str]], namespace_type: str | None) -> bool:
    """whether a role with ``permissions`` grants an action OF ``namespace_type``'s own on it: among the actions its
    type bucket and the wildcard bucket grant (:func:`_granted_actions`, :meth:`Role.actions_for`'s union), one that
    :func:`_action_acts_on` the type. for None ("any type", a listing that names none), whether it holds an action of
    its own type anywhere: a bucket holding an action on the bucket's type, or a wildcard bucket holding any action
    (each acts on some type: the one it names, or every one).

    the rule caller visibility counts a grant by (:func:`threetears.agent.acl.visible_customers`, and
    :func:`threetears.agent.acl.caller_visible_customer_clause` spelling it in SQL). it is narrower than what the
    evaluator authorizes, deliberately: the evaluator answers one action on one namespace, and a role whose
    ``customer`` bucket holds ``group.read`` may read groups there; visibility answers which customers' rows of
    EVERY listed type a grant opens, and an action on groups is no membership in the customer (Pace's ruling
    2026-11-20, option A).

    WHY None asks for one action anywhere: a grant reaches a type only through a role that acts on it, and a listing of
    no type names none to check, so the weakest grant that is still a grant is the bar: a role with no action of its
    own type (``{}``, empty buckets, or only another type's actions) is no grant on any type, and opens nothing.

    :param permissions: a role's permissions
    :ptype permissions: Mapping[str, frozenset[str]]
    :param namespace_type: the type, or None for any
    :ptype namespace_type: str | None
    :return: whether it acts there
    :rtype: bool
    """
    if namespace_type is None:
        return any(
            bool(actions) if bucket == WILDCARD_RESOURCE_TYPE else any(_action_acts_on(a, bucket) for a in actions)
            for bucket, actions in permissions.items()
        )
    return any(_action_acts_on(action, namespace_type) for action in _granted_actions(permissions, namespace_type))


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
    :cvar GROUP: group member -- the CHILD group named by ``member_id``
        belongs to the parent :class:`Group`, so everything the parent
        is granted reaches the child's members transitively. resolution
        WALKS these edges at read, capped at
        :data:`MAX_GROUP_MEMBERSHIP_DEPTH`; nothing is expanded at
        write, which is what keeps cache invalidation per-group (see
        ``cache.py``'s membership layer)
    """

    USER = "user"
    AGENT = "agent"
    GROUP = "group"


#: the environment variable that sets the group nesting, read once at import. it counts LEVELS the way
#: a reader counts the chain on screen: the person plus each group in it, so
#: ``person -> customer-users:14th -> reports-report:all`` is 3 levels.
MAX_GROUP_NESTING_LEVELS_ENV: Final[str] = "THREETEARS_ACL_MAX_GROUP_NESTING_LEVELS"

#: the variable's earlier name, which counted membership HOPS (one fewer than levels). it never
#: shipped in a release; setting it refuses at import, naming the conversion, so a deployment that
#: still sets it cannot quietly get a nesting one level shallower than it meant.
RETIRED_MAX_GROUP_MEMBERSHIP_DEPTH_ENV: Final[str] = "THREETEARS_ACL_MAX_GROUP_MEMBERSHIP_DEPTH"

#: the levels when :data:`MAX_GROUP_NESTING_LEVELS_ENV` is unset: a person plus two groups
#: (person -> org group -> state group).
DEFAULT_MAX_GROUP_NESTING_LEVELS: Final[int] = 3

#: the shallowest a deployment may configure: a person plus the one group they are directly in.
MIN_CONFIGURABLE_GROUP_NESTING_LEVELS: Final[int] = 2

#: the deepest a deployment may configure. each level adds one nested ``EXISTS`` to the visibility
#: SQL and one read to a cold evaluator walk; 4 levels already multiplied the visibility and
#: discovery SQL's cost on YugabyteDB, so a value past this is a typo, not a decision.
MAX_CONFIGURABLE_GROUP_NESTING_LEVELS: Final[int] = 5


def resolve_max_group_nesting_levels(environ: Mapping[str, str]) -> int:
    """read the group nesting, in levels, from ``environ``, refusing a value the platform cannot honour.

    unset means :data:`DEFAULT_MAX_GROUP_NESTING_LEVELS`. anything else must be an integer from
    :data:`MIN_CONFIGURABLE_GROUP_NESTING_LEVELS` to :data:`MAX_CONFIGURABLE_GROUP_NESTING_LEVELS`.
    an invalid value raises rather than falling back: a deployment that asked for a nesting and
    silently got another would show memberships that grant nothing, which is the failure the depth
    cap's write-time refusal exists to prevent. so does
    :data:`RETIRED_MAX_GROUP_MEMBERSHIP_DEPTH_ENV` set at all, because it counted hops and read as
    levels it would be one short.

    :param environ: the environment to read, normally ``os.environ``
    :ptype environ: Mapping[str, str]
    :return: the levels: the person plus each group in the longest chain that resolves
    :rtype: int
    :raises ValueError: when the retired variable is set, or the variable is set to anything but an
        integer in range
    """
    retired = environ.get(RETIRED_MAX_GROUP_MEMBERSHIP_DEPTH_ENV)
    if retired is not None:
        hops = retired.strip()
        converted = str(int(hops) + 1) if hops.lstrip("-").isdigit() else "<old value + 1>"
        raise ValueError(
            f"{RETIRED_MAX_GROUP_MEMBERSHIP_DEPTH_ENV} is retired: it counted membership hops. unset it and set "
            f"{MAX_GROUP_NESTING_LEVELS_ENV}, which counts levels (the person plus each group): levels = old value + 1, "
            f"so {RETIRED_MAX_GROUP_MEMBERSHIP_DEPTH_ENV}={hops} becomes {MAX_GROUP_NESTING_LEVELS_ENV}={converted}"
        )
    raw = environ.get(MAX_GROUP_NESTING_LEVELS_ENV)
    levels = DEFAULT_MAX_GROUP_NESTING_LEVELS
    if raw is not None:
        try:
            levels = int(raw.strip())
        except ValueError:
            raise ValueError(f"{MAX_GROUP_NESTING_LEVELS_ENV}={raw!r} is not an integer") from None
        if not MIN_CONFIGURABLE_GROUP_NESTING_LEVELS <= levels <= MAX_CONFIGURABLE_GROUP_NESTING_LEVELS:
            raise ValueError(
                f"{MAX_GROUP_NESTING_LEVELS_ENV}={raw!r} is outside "
                f"{MIN_CONFIGURABLE_GROUP_NESTING_LEVELS}..{MAX_CONFIGURABLE_GROUP_NESTING_LEVELS}"
            )
    return levels


#: the configured nesting in levels -- the person plus each group in the longest chain that
#: resolves -- read once here from :data:`MAX_GROUP_NESTING_LEVELS_ENV` (default
#: :data:`DEFAULT_MAX_GROUP_NESTING_LEVELS`).
MAX_GROUP_NESTING_LEVELS: Final[int] = resolve_max_group_nesting_levels(os.environ)

#: how many membership hops resolution walks: an actor's DIRECT groups are depth 1, their parent
#: groups depth 2, and so on -- :data:`MAX_GROUP_NESTING_LEVELS` minus the person. configured, not
#: coded: every surface reads this one value -- the evaluator's walk, the cycle check, the generated
#: visibility SQL, and a consumer's own write-time cap and generated SQL -- so none can disagree
#: within a process. every process that evaluates (the hub, agent pods, the registry) must be given
#: the same setting, or one of them grants what another refuses.
MAX_GROUP_MEMBERSHIP_DEPTH: Final[int] = MAX_GROUP_NESTING_LEVELS - 1


class ScopeType(StrEnum):
    """shape of the scope a :class:`RoleAssignment` covers.

    :cvar NAMESPACE: assignment applies to one specific namespace,
        identified by :attr:`RoleAssignment.scope_namespace_id`.
    :cvar SUBTREE: assignment applies to the namespace-name node in
        :attr:`RoleAssignment.scope_namespace_name` and to every
        namespace beneath it. covers "caller on the whole
        ``tools.pentest`` family", which the other three cannot say:
        NAMESPACE reaches one row and TYPE_CUSTOMER reaches every tool
        namespace the customer has. the root is a NAME rather than an
        id because containment is a statement about the name space --
        ids are minted per row and carry no hierarchy, and a subtree
        root need not be a materialized row at all.
    :cvar TYPE_CUSTOMER: assignment applies to every namespace of a
        given :attr:`RoleAssignment.scope_namespace_type` within
        :attr:`RoleAssignment.scope_customer_id`. covers "auditor on
        every workspace this customer has."
    :cvar ALL: assignment applies universally; only platform admins
        may create assignments at this scope.
    """

    NAMESPACE = "namespace"
    SUBTREE = "subtree"
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
    and the namespace that owns it. ownership short-circuits any grant
    lookup: a principal always has full access to namespaces its own
    namespace owns.

    :ivar id: namespace UUID; the primary key in the hub's ``namespaces``
    :ivar customer_id: customer UUID this namespace belongs to, or
        ``None`` for a platform-scoped namespace (``customer_id IS NULL``)
    :ivar namespace_type: type discriminator string
        (``workspace``, ``agent``, ``shared``, ``system``, ...)
    :ivar owner_agent_id: UUID of the agent that owns the physical rows,
        or ``None`` for a namespace with no owning agent (datasource /
        customer / knowledge / ...). it says which AGENT owns the rows
        and is true wherever it is set; what it cannot say is that a
        namespace is owned by something which is not an agent, which is
        why it no longer decides authorization
    :ivar owner_namespace: canonical NAME of the namespace row that
        OWNS this one, or ``None`` when nothing owns it. this is the
        ownership key the evaluator's short-circuit reads. it names a
        namespace rather than an agent because a tool pod, a capability
        source and an in-hub component own namespaces too and none of
        them is an agent; one owner column covers all four without a
        column per kind. it is a NAME rather than a row id because the
        short-circuit must resolve with no i/o at all -- an agent pod's
        sandboxed L3 has no ``namespaces`` table, so the owner-path
        descriptors several packages build carry no row and could not
        supply an id. a name the caller can derive from its own
        identity is the only owner value answerable there. ``None``
        matches NOTHING -- see
        :func:`threetears.agent.acl.evaluator.evaluate_with_trail`
    :ivar name: canonical ``namespaces.name`` value, or
        ``None`` when the construction site had no name to supply (the
        workspace file-access path builds this value out of a workspace
        row rather than a namespace row). only
        :attr:`ScopeType.SUBTREE` reads it, and it reads it
        FAIL-CLOSED: a namespace with no name is covered by no subtree
        assignment, so a site that does not supply one narrows a
        subtree grant to nothing and can never widen one
    """

    id: UUID
    customer_id: UUID | None
    namespace_type: str
    owner_agent_id: UUID | None
    name: str | None = None
    owner_namespace: str | None = None


@dataclass(frozen=True)
class Group:
    """the only grant principal.

    a group has zero or more members of type :class:`MemberType`. when
    ``customer_id`` is ``None`` the group is platform-scoped (admin-
    managed; members may span customers); otherwise the group is
    customer-scoped and every member's customer must match.

    the evaluator and every loader key on ``id``. ``name`` is also an
    exact handle: the platform DDL makes it unique per customer, and
    unique across platform-scoped rows, which is what lets a consuming
    app ask about a group by the name a human typed.

    :ivar id: group UUID — exact, and stable across a rename
    :ivar name: group name; unique within the group's scope
    :ivar customer_id: owning customer UUID, or ``None`` for platform
        scope
    """

    id: UUID
    name: str
    customer_id: UUID | None


@dataclass(frozen=True)
class GroupMembership:
    """a single ``(group, member)`` pair.

    materialized from the hub's ``group_members``. the denormalized
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

    permissions are stored as a JSONB column on the hub's ``roles``;
    this dataclass mirrors the shape after deserialization. the
    wildcard resource-type ``"*"`` (see
    :data:`WILDCARD_RESOURCE_TYPE`) is permitted for type-agnostic
    roles like ``Reader``, ``Writer``, ``Admin``.

    a role is owned either by the platform (``customer_id is None``) or
    by exactly one customer. platform-owned roles are the shipped
    built-ins plus any operator-authored role a platform admin writes;
    customer-owned roles are authored by a customer admin out of that
    application's permission catalog (see
    :mod:`threetears.agent.acl.catalog`) and are resolvable ONLY inside
    the owning customer. name uniqueness follows ownership: unique
    across platform-owned rows, and unique per customer for
    customer-owned ones, so two customers may both have a "Field Manager".

    :ivar id: role UUID
    :ivar name: role name; unique within the role's ownership scope
    :ivar permissions: ``{resource_type: frozenset(actions)}`` mapping
    :ivar is_built_in: whether this role is shipped by the platform's
        seed migration. distinct from ownership: an operator-authored
        platform role is ``is_built_in=False`` with
        ``customer_id=None``
    :ivar customer_id: owning customer UUID for a customer-authored
        role, or ``None`` when the role is platform-owned. an
        assignment naming a customer-owned role contributes nothing
        against a namespace belonging to any other customer — the
        evaluator enforces that independently of the group wall, so a
        mis-scoped assignment row cannot leak the role
    """

    id: UUID
    name: str
    permissions: Mapping[str, frozenset[str]]
    is_built_in: bool
    customer_id: UUID | None = None

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
        return _granted_actions(self.permissions, resource_type)


@dataclass(frozen=True)
class RoleAssignment:
    """binds a :class:`Role` to a :class:`Group` on a scope.

    this is the only grant shape in the system. there is no parallel
    direct-user assignment row and no parallel direct-agent assignment
    row.

    field semantics by ``scope_type``:

    - :attr:`ScopeType.NAMESPACE` — ``scope_namespace_id`` set;
      ``scope_namespace_type`` and ``scope_customer_id`` ignored.
    - :attr:`ScopeType.SUBTREE` — ``scope_namespace_name`` set to the
      root node; every other scope_* field is ``None``. no customer is
      carried, because the cross-customer wall is enforced separately
      on the group's and the role's customer rather than by the scope.
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
    :ivar scope_namespace_name: root namespace-name node for
        subtree-scope; else None
    """

    id: UUID
    role_id: UUID
    group_id: UUID
    scope_type: ScopeType
    scope_namespace_id: UUID | None
    scope_namespace_type: str | None
    scope_customer_id: UUID | None
    scope_namespace_name: str | None = None

    def covers(self, namespace: Namespace) -> bool:
        """true iff this assignment's scope covers the given namespace.

        scope coverage rules:

        - ``namespace`` scope: ids must match exactly.
        - ``subtree`` scope: the namespace's ``name`` must be the
          assignment's root node or sit beneath it, decided by
          :func:`threetears.core.namespaces.namespace_contains` — the
          ONE containment implementation, segment-aware by
          construction, so a root of ``pentest`` reaches
          ``pentest.sqlmap`` and never ``pentestimposter.sqlmap``.
          a missing root or a missing name denies.
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
        elif self.scope_type == ScopeType.SUBTREE:
            result = (
                self.scope_namespace_name is not None
                and namespace.name is not None
                and namespace_contains(self.scope_namespace_name, namespace.name)
            )
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

    ``group_id`` asks a third, audit-shaped question: what does a
    direct member of this group reach? It is answered alone -- never
    alongside a user or an agent -- by the same walk a user member of
    the group gets: the group, its ancestors to
    :data:`MAX_GROUP_MEMBERSHIP_DEPTH`, and every wall past the
    member's own membership row. A caller recording what nesting a
    group changed asks it rather than re-deriving the walk, so the
    answer cannot drift from the evaluator's own.

    Two things the answer depends on that the caller owns:

    - it has no membership row to check, so the first wall -- a
      member's row must carry the namespace's customer -- cannot apply.
      For a customer-scoped group that changes nothing (the group's own
      wall covers it). For a PLATFORM-scoped group it is the reach of
      the widest possible member: a customer-scoped member reaches only
      its own customer's share of it.
    - the group's parent set is read through the membership cache like
      every other walk, so a caller asking right after it wrote a
      membership evicts that group's membership key first, or it is
      answered from the parent set before the write.

    :ivar namespace: namespace under evaluation
    :ivar action: action string being checked
    :ivar user_id: invoking user UUID, or ``None`` for an agent-only
        evaluation
    :ivar agent_id: invoking agent UUID, or ``None`` for a user-only
        evaluation
    :ivar group_id: group whose direct members are being evaluated,
        or ``None``; exclusive of ``user_id`` and ``agent_id``
    """

    namespace: Namespace
    action: str
    user_id: UUID | None = None
    agent_id: UUID | None = None
    group_id: UUID | None = None


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
    - :attr:`user_actions` for a user- or group-member evaluation, or
      :attr:`agent_actions` for an agent-only one -- the set the side
      that ran contributed; the other is left empty.
    - :attr:`limiting_side` left at :attr:`LimitingSide.NEITHER`.

    :ivar decision: True for allow, False for deny
    :ivar effective_actions: final action set after intersection /
        single-side resolution
    :ivar trails: trails for the single-actor side that ran (empty
        for intersection evaluations; intersection evaluations use
        :attr:`user_trails` and :attr:`agent_trails` instead)
    :ivar user_actions: action set the user side (or the group-member
        side) contributed
    :ivar agent_actions: action set the agent side contributed
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
