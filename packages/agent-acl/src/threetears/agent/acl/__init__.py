"""shared rbac evaluator + cache for the 3tears platform.

this package is the single source of truth for "can actor do action
on namespace" decisions. the same code runs in the hub broker and
inside every agent pod, so authorization answers are byte-identical
across processes and one set of unit tests covers every caller.

public surface:

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

callers wire concrete loaders against their persistence layer and
hand the loaders + a :class:`AclCache` instance to the evaluator on
every call. this package never opens a database connection or
publishes a NATS message itself.
"""

__version__ = "0.1.0"

from threetears.agent.acl.cache import (
    AclCache,
    ActorMembershipEntry,
    ActorMembershipKey,
    GroupNamespaceEntry,
    GroupNamespaceKey,
    GroupTypeCustomerEntry,
    GroupTypeCustomerKey,
)
from threetears.agent.acl.evaluator import (
    evaluate_decision,
    evaluate_with_trail,
)
from threetears.agent.acl.loader import GrantLoader, MembershipLoader
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
    "AclCache",
    "ActorMembershipEntry",
    "ActorMembershipKey",
    "ActorType",
    "EvaluationContext",
    "EvaluationResult",
    "GrantLoader",
    "Group",
    "GroupMembership",
    "GroupNamespaceEntry",
    "GroupNamespaceKey",
    "GroupTypeCustomerEntry",
    "GroupTypeCustomerKey",
    "LimitingSide",
    "MemberType",
    "MembershipLoader",
    "Namespace",
    "Role",
    "RoleAssignment",
    "ScopeType",
    "Trail",
    "WILDCARD_RESOURCE_TYPE",
    "evaluate_decision",
    "evaluate_with_trail",
]
