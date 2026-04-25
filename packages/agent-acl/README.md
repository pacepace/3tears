# 3tears-agent-acl

Shared RBAC primitives for the 3tears platform: evaluator, cache,
canonical Collections, loader adapters, and NATS invalidation
payload models.

## Purpose

Single source of truth for "can actor do action on namespace" decisions
PLUS the persistence layer that feeds them. The same pure-Python
evaluator + canonical Collections + loader adapters run in the hub
broker, inside every agent pod, and inside any other 3tears app that
participates in the unified RBAC story, so authorization answers are
byte-identical across processes and one set of unit tests covers every
caller.

Produced by `rbac-task-01` (Groups / Roles / Assignments); generalized
by `acl-promotion-task-01` so the Collections + loaders + invalidation
models are reusable across any 3tears app, not just the aibots hub.
Supersedes the previous per-resource ACL code paths
(`namespace_grants`, workspace-specific checks, tool-access fnmatch
lists).

## Public API

Exports (see `src/threetears/agent/acl/__init__.py`):

### Evaluation

- `evaluate_decision(ctx) -> bool` — fast yes/no hot path;
  cache-friendly when wired behind `AclCache`. Used on every
  production call.
- `evaluate_with_trail(ctx) -> EvaluationResult` — verbose, uncached
  introspection path returning every
  `(group, assignment, role) → contributed_actions` chain plus
  `limiting_side` for user×agent intersection queries.
- `AclCache` — three-layer in-process TTL cache
  (`actor → [group_id]`, `(group_id, namespace_id) → action_set`,
  `(group_id, namespace_type, customer_id) → action_set`) with
  fine-grained invalidation hooks fired on group-membership / role /
  assignment change.

### Persistence

- `GroupCollection`, `GroupMemberCollection`, `RoleCollection`,
  `RoleAssignmentCollection`, `NamespaceCollection` — three-tier
  `SchemaBackedCollection` subclasses fronting the canonical RBAC
  tables (`groups`, `group_members`, `roles`, `role_assignments`,
  `namespaces`). Schemas use canonical RBAC names with no
  deploy-specific schema prefix; the prefix (e.g. `platform.` in the
  aibots hub deployment) is set on the L3 pool's `search_path`, not
  in the schema name on the Collection.
- `GroupEntity`, `GroupMemberEntity`, `RoleEntity`,
  `RoleAssignmentEntity`, `NamespaceEntity` — `BaseEntity`
  subclasses; four use composite primary keys post-row_scope
  partitioning (`(row_scope, id)` for groups / role_assignments /
  namespaces; `(group_id, id)` for group_members).
- `CollectionMembershipLoader`, `CollectionGrantLoader` — concrete
  loader adapters satisfying the `MembershipLoader` /
  `GrantLoader` Protocols, wired against the canonical Collections.

### Invalidation

- `MembershipInvalidatePayload`, `AssignmentInvalidatePayload`,
  `RoleInvalidatePayload` — typed Pydantic models for the three
  `{ns}.acl.*.invalidate` NATS subjects. Wire format is single-source:
  every publisher (admin endpoints, agent self-mutations) and every
  subscriber (hub broker, agent-pod cache, any other app) speaks
  these models.

### Value types & protocols

- Value types: `Group`, `GroupMembership`, `Role`, `RoleAssignment`,
  `Namespace`, `EvaluationContext`, `EvaluationResult`, `Trail`.
- Enums: `ActorType`, `MemberType`, `ScopeType`, `LimitingSide`.
- I/O protocols: `GrantLoader`, `MembershipLoader` — callers may
  implement these against any persistence layer; the canonical
  `Collection*Loader` adapters above are the reference impls.

## Consuming the package from a 3tears app

Each app constructs the canonical Collections against its own L3
pool (direct asyncpg in a hub-style deployment, NATS-proxied L3 in
each agent pod) and wires the canonical loader adapters + cache.
Hub-specific admin query shapes (dynamic `list_by_filter` /
per-cardinality counts / multi-table discovery JOINs) live on
deploying-app subclasses (e.g. `aibots.hub.rbac.collections.HubRoleAssignmentCollection`).

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

## Reference

- Initial promotion: `14-eng-ai-bot/docs/acl-promotion-task-01.md`
- Original RBAC design: `14-eng-ai-bot/docs/rbac-task-01-groups-roles-assignments.md`
- Hub-side admin extensions: `14-eng-ai-bot/src/aibots/hub/rbac/collections.py`
  (`HubGroupCollection` etc.), `14-eng-ai-bot/src/aibots/hub/broker/namespaces.py`
  (`HubNamespaceCollection`), `14-eng-ai-bot/src/aibots/hub/broker/acl.py`
  (`BrokerAclGateway`)
- Agent SDK consumer: `14-eng-ai-bot-agents/src/aibots_agents/runtime/three_tier_stack.py`
- Last commit (phase 1): 3tears `9bff571`; promotion: 3tears `abde333`,
  aibots `230d83e`, aibots-agents `a83226f`
