"""shared rbac evaluator + cache + collections for the 3tears platform.

this package is the single source of truth for "can actor do action
on namespace" decisions. the same code runs in the hub broker and
inside every agent pod, so authorization answers are byte-identical
across processes and one set of unit tests covers every caller.

public surface — evaluation:

- :func:`evaluate_decision` — fast yes/no path for the production
  hot path; cache-friendly when wired behind :class:`AclCache`.
- :func:`evaluate_with_trail` — verbose introspection / audit path
  returning the full :class:`EvaluationResult` with every
  contributing ``(group, assignment, role)`` chain.
- :class:`AclCache` — three-layer in-process ttl cache (membership,
  per-namespace assignments, per-type+customer assignments) with
  fine-grained invalidation hooks.
- value types :class:`Group`, :class:`GroupMembership`,
  :class:`Role`, :class:`RoleAssignment`, :class:`Namespace`,
  :class:`EvaluationContext`, :class:`EvaluationResult`,
  :class:`Trail`.
- enums :class:`ActorType`, :class:`MemberType`, :class:`ScopeType`,
  :class:`LimitingSide`.
- i/o protocols :class:`MembershipLoader` and :class:`GrantLoader`.

public surface — persistence:

- collections :class:`GroupCollection`,
  :class:`GroupMemberCollection`, :class:`RoleCollection`,
  :class:`RoleAssignmentCollection`,
  :class:`NamespaceCollection` — three-tier collections fronting
  the canonical rbac tables.
- entities :class:`GroupEntity`, :class:`GroupMemberEntity`,
  :class:`RoleEntity`, :class:`RoleAssignmentEntity`,
  :class:`NamespaceEntity`.
- loaders :class:`CollectionMembershipLoader`,
  :class:`CollectionGrantLoader` — concrete impls of the loader
  Protocols backed by the canonical Collections.
- invalidation models :class:`MembershipInvalidatePayload`,
  :class:`AssignmentInvalidatePayload`,
  :class:`RoleInvalidatePayload` — typed NATS payloads for
  cross-process cache invalidation.

callers wire concrete loaders against their persistence layer (or
use :class:`CollectionMembershipLoader` /
:class:`CollectionGrantLoader` against the canonical Collections)
and hand the loaders + a :class:`AclCache` instance to the evaluator
on every call. evaluation logic itself never opens a database
connection or publishes a NATS message.
"""

# Version derived from pyproject.toml so the metadata is the single
# source of truth -- a future release that bumps pyproject without
# updating ``__init__.py`` can't drift the runtime ``__version__``.
# The except guard handles the rare case where the package isn't
# installed via importlib.metadata (e.g. running directly from a
# checked-out source tree without ``uv sync``); the fallback keeps
# imports working but reports ``unknown`` rather than crashing.
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("3tears-agent-acl")
except _PackageNotFoundError:  # pragma: no cover - dev fallback
    __version__ = "unknown"

from threetears.agent.acl.audit_vocabulary import (
    RBAC_AUDIT_ACTIONS,
    RBAC_AUDIT_EVENT_TYPES,
    RBAC_AUDIT_RESOURCE_TYPES,
    RbacAuditAction,
    RbacAuditResourceType,
    RbacEventType,
)
from threetears.agent.acl.authorize import (
    AccessDenied,
    NamespaceNotFound,
    authorize,
    authorize_on_entity,
    authorize_with_trail,
)
from threetears.agent.acl.builtin_roles import (
    PLATFORM_BUILTIN_PRE_CHECK_TOOL_NAMES,
    PLATFORM_BUILTIN_TOOL_USER_ROLE_DESCRIPTION,
    PLATFORM_BUILTIN_TOOL_USER_ROLE_NAME,
    PLATFORM_BUILTIN_TOOL_USER_ROLE_PERMISSIONS,
    ensure_platform_builtin_tool_user_role,
)
from threetears.agent.acl.cache import (
    AclCache,
    ActorMembershipEntry,
    ActorMembershipKey,
    GroupNamespaceEntry,
    GroupNamespaceKey,
    GroupTypeCustomerEntry,
    GroupTypeCustomerKey,
)
from threetears.agent.acl.collections import (
    GroupCollection,
    GroupMemberCollection,
    NamespaceCollection,
    RoleAssignmentCollection,
    RoleCollection,
)
from threetears.agent.acl.entities import (
    GroupEntity,
    GroupMemberEntity,
    NamespaceEntity,
    RoleAssignmentEntity,
    RoleEntity,
)
from threetears.agent.acl.evaluator import (
    READ_FILE_MATCHING_PREFIX,
    WRITE_FILE_MATCHING_PREFIX,
    evaluate_decision,
    evaluate_file_access,
    evaluate_with_trail,
)
from threetears.agent.acl.invalidation import (
    AssignmentInvalidatePayload,
    MembershipInvalidatePayload,
    RoleInvalidatePayload,
)
from threetears.agent.acl.loader import GrantLoader, MembershipLoader
from threetears.agent.acl.loaders import (
    CollectionGrantLoader,
    CollectionMembershipLoader,
)
from threetears.agent.acl.query_visibility import (
    caller_visible_customer_clause,
    caller_visible_customers_query,
    customer_scope_visibility_clause,
    three_scope_visibility_clause,
)
from threetears.agent.acl.types import (
    ActorType,
    EvaluationContext,
    EvaluationResult,
    Group,
    GroupMembership,
    LimitingSide,
    MemberType,
    Namespace,
    Role,
    RoleAssignment,
    ScopeType,
    Trail,
    WILDCARD_RESOURCE_TYPE,
)

__all__ = [
    "AccessDenied",
    "AclCache",
    "ActorMembershipEntry",
    "ActorMembershipKey",
    "ActorType",
    "AssignmentInvalidatePayload",
    "CollectionGrantLoader",
    "CollectionMembershipLoader",
    "EvaluationContext",
    "EvaluationResult",
    "GrantLoader",
    "Group",
    "GroupCollection",
    "GroupEntity",
    "GroupMemberCollection",
    "GroupMemberEntity",
    "GroupMembership",
    "GroupNamespaceEntry",
    "GroupNamespaceKey",
    "GroupTypeCustomerEntry",
    "GroupTypeCustomerKey",
    "LimitingSide",
    "MemberType",
    "MembershipInvalidatePayload",
    "MembershipLoader",
    "Namespace",
    "NamespaceCollection",
    "NamespaceEntity",
    "NamespaceNotFound",
    "PLATFORM_BUILTIN_PRE_CHECK_TOOL_NAMES",
    "PLATFORM_BUILTIN_TOOL_USER_ROLE_DESCRIPTION",
    "PLATFORM_BUILTIN_TOOL_USER_ROLE_NAME",
    "PLATFORM_BUILTIN_TOOL_USER_ROLE_PERMISSIONS",
    "READ_FILE_MATCHING_PREFIX",
    "RBAC_AUDIT_ACTIONS",
    "RBAC_AUDIT_EVENT_TYPES",
    "RBAC_AUDIT_RESOURCE_TYPES",
    "RbacAuditAction",
    "RbacAuditResourceType",
    "RbacEventType",
    "Role",
    "RoleAssignment",
    "RoleAssignmentCollection",
    "RoleAssignmentEntity",
    "RoleCollection",
    "RoleEntity",
    "RoleInvalidatePayload",
    "ScopeType",
    "Trail",
    "WILDCARD_RESOURCE_TYPE",
    "WRITE_FILE_MATCHING_PREFIX",
    "authorize",
    "authorize_on_entity",
    "authorize_with_trail",
    "caller_visible_customer_clause",
    "caller_visible_customers_query",
    "customer_scope_visibility_clause",
    "ensure_platform_builtin_tool_user_role",
    "evaluate_decision",
    "evaluate_file_access",
    "evaluate_with_trail",
    "three_scope_visibility_clause",
]
