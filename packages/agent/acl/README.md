# 3tears-agent-acl

Shared RBAC primitives for the 3tears platform: evaluator, cache,
canonical Collections, loader adapters, and NATS invalidation
payload models.

## Purpose

Single source of truth for "can actor do action on namespace" decisions
PLUS the persistence layer that feeds them. The same pure-Python
evaluator, canonical Collections, and loader adapters run in every
consuming application, so authorization answers are byte-identical
across processes and one set of unit tests covers every caller.

The Collections, loaders, and invalidation models are reusable across
any 3tears app. The package supersedes per-resource ACL code paths
(`namespace_grants`, workspace-specific checks, tool-access fnmatch
lists).

## Public API

Exports (see `src/threetears/agent/acl/__init__.py`):

### Evaluation

- `evaluate_decision(ctx, *, cache: AclCache) -> bool` -- fast yes/no
  hot path. The cache is consulted for membership and per-namespace
  contribution layers on every call, falling back to its loaders only
  on cache miss; production hit rate against repeated authz checks
  for the same `(actor, namespace)` is ~100% within the cache TTL.
- `evaluate_with_trail(ctx, *, cache) -> EvaluationResult` --
  introspection path returning every
  `(group, assignment, role) -> contributed_actions` chain plus
  `limiting_side` for user×agent intersection queries. Uses the same
  cache layers as `evaluate_decision`; trails are stored alongside
  the action set so successive decision-mode and trail-mode calls
  for the same actor + namespace serve from cache.
- `evaluate_file_access(*, namespace, user_id, agent_id, path,
  direction, cache) -> bool` -- workspace path-glob gate; same cache
  semantics.
- `authorize(*, namespace_collection, namespace_name, action,
  user_id, agent_id, cache) -> EvaluationResult` -- canonical
  authorization primitive every app's resource-typed wrapper
  is built on. Looks up the namespace by name, runs
  `evaluate_with_trail` through the cache, raises generic
  `AccessDenied` on deny / `NamespaceNotFound` on missing row.
- `authorize_with_trail` -- variant returning `(result, ns_entity)`
  for wrappers needing the entity.
- `AccessDenied` / `NamespaceNotFound` -- generic + namespace-miss
  exception classes; per-resource wrappers subclass `AccessDenied`
  to carry typed catching at endpoint code (e.g.
  `MemoryAccessDenied`, `DatasourceAccessDenied`).
- `AclCache` -- three-layer in-process TTL cache
  (`actor -> [GroupMembership]`, `(group_id, namespace_id) -> action_set
  + trails`, `(group_id, namespace_type, customer_id) -> action_set
  + trails`) with fine-grained invalidation hooks fired on
  group-membership / role / assignment change. The `ActorMembershipEntry`
  carries the full memberships tuple so the evaluator's
  cross-customer + member-type filter runs against cached state.

### Persistence

- `GroupCollection`, `GroupMemberCollection`, `RoleCollection`,
  `RoleAssignmentCollection`, `NamespaceCollection` -- three-tier
  `SchemaBackedCollection` subclasses fronting the canonical RBAC
  tables (`groups`, `group_members`, `roles`, `role_assignments`,
  `namespaces`). Schemas use canonical RBAC names with no
  deploy-specific schema prefix; the prefix is set on the L3 pool's
  `search_path`, not in the schema name on the Collection.
- `GroupEntity`, `GroupMemberEntity`, `RoleEntity`,
  `RoleAssignmentEntity`, `NamespaceEntity` -- `BaseEntity`
  subclasses; four use composite primary keys post-row_scope
  partitioning (`(row_scope, id)` for groups / role_assignments /
  namespaces; `(group_id, id)` for group_members).
- `CollectionMembershipLoader`, `CollectionGrantLoader` -- concrete
  loader adapters satisfying the `MembershipLoader` /
  `GrantLoader` Protocols, wired against the canonical Collections.

### Invalidation

- `MembershipInvalidatePayload`, `AssignmentInvalidatePayload`,
  `RoleInvalidatePayload` -- typed Pydantic models for the three
  `{ns}.acl.*.invalidate` NATS subjects. Wire format is single-source:
  every publisher (admin endpoints, agent self-mutations) and every
  subscriber (cache subscribers in any consuming app) speaks
  these models.

### Value types & protocols

- Value types: `Group`, `GroupMembership`, `Role`, `RoleAssignment`,
  `Namespace`, `EvaluationContext`, `EvaluationResult`, `Trail`.
- Enums: `ActorType`, `MemberType`, `ScopeType`, `LimitingSide`.
- I/O protocols: `GrantLoader`, `MembershipLoader` -- callers may
  implement these against any persistence layer; the canonical
  `Collection*Loader` adapters above are the reference impls.

## Consuming the package from a 3tears app

Each app constructs the canonical Collections against its own L3
pool (direct asyncpg in a server-style deployment, NATS-proxied L3 in
each agent pod) and wires the canonical loader adapters + cache.
Deployment-specific admin query shapes (dynamic `list_by_filter` /
per-cardinality counts / multi-table discovery JOINs) live on
consuming-app subclasses.

```python
from threetears.agent.acl import (
    AclCache,
    CollectionGrantLoader,
    CollectionMembershipLoader,
    EvaluationContext,
    GroupCollection,
    GroupMemberCollection,
    NamespaceCollection,
    RoleAssignmentCollection,
    RoleCollection,
    evaluate_decision,
)

# 1. construct Collections against your registry / L3 pool
group_collection = GroupCollection(registry=registry, config=core_config)
group_member_collection = GroupMemberCollection(registry=registry, config=core_config)
role_collection = RoleCollection(registry=registry, config=core_config)
role_assignment_collection = RoleAssignmentCollection(registry=registry, config=core_config)
namespace_collection = NamespaceCollection(registry=registry, config=core_config)

# 2. wire the canonical loaders
membership_loader = CollectionMembershipLoader(collection=group_member_collection)
grant_loader = CollectionGrantLoader(
    assignment_collection=role_assignment_collection,
    role_collection=role_collection,
    group_collection=group_collection,
)

# 3. build the cache
cache = AclCache(
    membership_loader=membership_loader,
    grant_loader=grant_loader,
    ttl_seconds=60,
)

# 4. evaluate
ctx = EvaluationContext(
    namespace=target_namespace,
    action="read",
    user_id=user_id,
    agent_id=agent_id,
)
allowed = await evaluate_decision(
    ctx,
    membership_loader=membership_loader,
    grant_loader=grant_loader,
)
```

Implicit ownership: if `namespace.owner_agent_id == ctx.agent_id`,
the agent side short-circuits to full permissions with no group
lookup or assignment query. Ownership is a property of the namespace
row, never a grant.
