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
- :class:`AclCache` — three-layer in-process cache (membership,
  per-namespace assignments, per-type+customer assignments), evicted
  row by row from the access tables' write generations
  (:func:`~threetears.agent.acl.generation_follow.follow_access_tables`).
- value types :class:`Group`, :class:`GroupMembership`,
  :class:`Role`, :class:`RoleAssignment`, :class:`Namespace`,
  :class:`EvaluationContext`, :class:`EvaluationResult`,
  :class:`Trail`.
- enums :class:`ActorType`, :class:`MemberType`, :class:`ScopeType`,
  :class:`LimitingSide`.
- i/o protocols :class:`MembershipLoader` and :class:`GrantLoader`.

public surface — permission catalog (write path, not evaluation):

- :class:`PermissionCatalog` — the vocabulary of
  ``(resource_type, action)`` pairs applications have declared, built
  from :class:`ResourceTypeDescriptor` and :class:`ActionDescriptor`
  entries carrying their declaring application and their operator-facing
  labels.
- :func:`validate_permissions` — report every pair in a
  ``{resource_type: [action]}`` map that no entry declares;
  :func:`enforce_declared_permissions` is the raising form, and
  :class:`CatalogViolation` / :class:`CatalogViolationKind` /
  :class:`UndeclaredPermission` carry the detail.

nothing in that group participates in evaluation: a role evaluates
identically whether or not a catalog exists.

public surface — delegation ceiling (write path, not evaluation):

- :func:`resolve_held_permissions` — what a delegated admin
  demonstrably holds across their customer, answered by the evaluator
  and quantified over every namespace a grant could reach, returned as
  :class:`HeldPermissions` with the contributing trails.
- :func:`enforce_within_held_permissions` — refuse a permissions map
  that exceeds that ceiling, raising :class:`PermissionEscalation`
  carrying :class:`EscalationViolation` detail plus the trails;
  :func:`escalating_permissions` is the reporting form and
  :func:`held_actions_on` the primitive both are built from.

this is the check that lets role authoring and role assignment drop
below platform-admin without letting a customer admin hand out more than
they were given. it, too, changes no evaluation.

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
- :func:`evict_after_rbac_write` -- the rule every helper that WRITES
  a ``group_members`` / ``role_assignments`` / ``groups`` row while
  holding an :class:`AclCache` follows: evict what it wrote from that
  cache at once, so the writer's own next question is not answered
  from the entry the write just made wrong. every other process hears
  the write's row broadcast and its generation.
- :func:`register_rbac_l1_tables` -- the L1 SQLite mirror of the five
  rbac tables, GENERATED from the Collection schemas above. every
  process that evaluates locally needs it, and the hand-written
  copies drifted; see :mod:`threetears.agent.acl.tables`.
- :func:`row_scope_for_customer` -- the ``row_scope`` partition rule
  ``groups`` and ``namespaces`` share, stated once. both tables are
  keyed on the composite ``(row_scope, <entity>_id)``, so a caller
  addressing a row by ``get`` / ``delete`` / ``ensure`` needs the
  partition value as well as the id, and this is where that value
  comes from.

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
    INTERNAL_AUDIENCE,
    AccessDenied,
    ClaimsForAuthorization,
    ExternalAudienceNotSupported,
    ImpersonationCategory,
    NamespaceNotFound,
    authorize,
    authorize_from_claims,
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
from threetears.agent.acl.access_tables import ACCESS_TABLES, bind_acl_cache_to_access_tables
from threetears.agent.acl.caller_cache import (
    CallerAccessCache,
    CallerKey,
    CallerNamespaces,
    CallerNamespacesUnavailable,
    bind_caller_cache_to_access_tables,
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
from threetears.agent.acl.catalog import (
    ActionDescriptor,
    CatalogViolation,
    CatalogViolationKind,
    PermissionCatalog,
    ResourceTypeDescriptor,
    UndeclaredPermission,
    enforce_declared_permissions,
    validate_permissions,
)
from threetears.agent.acl.collections import (
    GroupCascade,
    GroupCollection,
    GroupMemberCollection,
    ImpersonationGateCollection,
    ImpersonationGateStatus,
    NamespaceCollection,
    NamespaceRescope,
    NamespaceRescopeNotAdvanced,
    NamespaceRescopeRefused,
    RoleAssignmentCollection,
    RoleCollection,
)
from threetears.agent.acl.delegation import (
    EscalationViolation,
    HeldPermissions,
    PermissionEscalation,
    enforce_within_held_permissions,
    escalating_permissions,
    held_actions_on,
    resolve_held_permissions,
)
from threetears.agent.acl.entities import (
    GroupEntity,
    GroupMemberEntity,
    ImpersonationGateEntity,
    NamespaceEntity,
    RoleAssignmentEntity,
    RoleEntity,
    row_scope_for_customer,
)
from threetears.agent.acl.evaluator import (
    READ_FILE_MATCHING_PREFIX,
    WRITE_FILE_MATCHING_PREFIX,
    evaluate_decision,
    evaluate_file_access,
    evaluate_with_trail,
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
from threetears.agent.acl.tables import (
    RBAC_L1_COLLECTIONS,
    RBAC_L1_TABLE_NAMES,
    register_rbac_l1_tables,
)
from threetears.agent.acl.types import (
    ActorType,
    EvaluationContext,
    EvaluationResult,
    Group,
    GroupMembership,
    LimitingSide,
    MAX_GROUP_MEMBERSHIP_DEPTH,
    MemberType,
    Namespace,
    Role,
    RoleAssignment,
    ScopeType,
    Trail,
    WILDCARD_RESOURCE_TYPE,
)
from threetears.agent.acl.write_eviction import evict_after_rbac_write

__all__ = [
    "INTERNAL_AUDIENCE",
    "AccessDenied",
    "ACCESS_TABLES",
    "AclCache",
    "ActorMembershipEntry",
    "ActorMembershipKey",
    "ActionDescriptor",
    "ActorType",
    "bind_acl_cache_to_access_tables",
    "bind_caller_cache_to_access_tables",
    "CallerAccessCache",
    "CallerKey",
    "CallerNamespaces",
    "CallerNamespacesUnavailable",
    "CatalogViolation",
    "CatalogViolationKind",
    "ClaimsForAuthorization",
    "CollectionGrantLoader",
    "CollectionMembershipLoader",
    "EscalationViolation",
    "EvaluationContext",
    "EvaluationResult",
    "ExternalAudienceNotSupported",
    "GrantLoader",
    "Group",
    "GroupCascade",
    "GroupCollection",
    "GroupEntity",
    "GroupMemberCollection",
    "GroupMemberEntity",
    "GroupMembership",
    "GroupNamespaceEntry",
    "GroupNamespaceKey",
    "GroupTypeCustomerEntry",
    "GroupTypeCustomerKey",
    "HeldPermissions",
    "ImpersonationCategory",
    "ImpersonationGateCollection",
    "ImpersonationGateEntity",
    "ImpersonationGateStatus",
    "LimitingSide",
    "MAX_GROUP_MEMBERSHIP_DEPTH",
    "MemberType",
    "MembershipLoader",
    "Namespace",
    "NamespaceCollection",
    "NamespaceEntity",
    "NamespaceNotFound",
    "NamespaceRescope",
    "NamespaceRescopeNotAdvanced",
    "NamespaceRescopeRefused",
    "PermissionCatalog",
    "PermissionEscalation",
    "PLATFORM_BUILTIN_PRE_CHECK_TOOL_NAMES",
    "PLATFORM_BUILTIN_TOOL_USER_ROLE_DESCRIPTION",
    "PLATFORM_BUILTIN_TOOL_USER_ROLE_NAME",
    "PLATFORM_BUILTIN_TOOL_USER_ROLE_PERMISSIONS",
    "READ_FILE_MATCHING_PREFIX",
    "RBAC_AUDIT_ACTIONS",
    "RBAC_L1_COLLECTIONS",
    "RBAC_L1_TABLE_NAMES",
    "RBAC_AUDIT_EVENT_TYPES",
    "RBAC_AUDIT_RESOURCE_TYPES",
    "RbacAuditAction",
    "RbacAuditResourceType",
    "RbacEventType",
    "ResourceTypeDescriptor",
    "Role",
    "RoleAssignment",
    "RoleAssignmentCollection",
    "RoleAssignmentEntity",
    "RoleCollection",
    "RoleEntity",
    "ScopeType",
    "Trail",
    "UndeclaredPermission",
    "WILDCARD_RESOURCE_TYPE",
    "WRITE_FILE_MATCHING_PREFIX",
    "authorize",
    "authorize_from_claims",
    "authorize_on_entity",
    "authorize_with_trail",
    "caller_visible_customer_clause",
    "caller_visible_customers_query",
    "customer_scope_visibility_clause",
    "enforce_declared_permissions",
    "enforce_within_held_permissions",
    "ensure_platform_builtin_tool_user_role",
    "escalating_permissions",
    "evaluate_decision",
    "evaluate_file_access",
    "evaluate_with_trail",
    "evict_after_rbac_write",
    "held_actions_on",
    "register_rbac_l1_tables",
    "resolve_held_permissions",
    "row_scope_for_customer",
    "three_scope_visibility_clause",
    "validate_permissions",
]
