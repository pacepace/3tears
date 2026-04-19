# 3tears-agent-acl

Shared RBAC evaluator + cache for the 3tears platform.

## Purpose

Single source of truth for "can actor do action on namespace" decisions. The same pure-Python evaluator runs in the hub broker and inside every agent pod, so authorization answers are byte-identical across processes and one set of unit tests covers every caller. Produced by `rbac-task-01` (Groups / Roles / Assignments) in `14-eng-ai-bot/docs/rbac-task-01-groups-roles-assignments.md`; supersedes the previous per-resource ACL code paths (namespace_grants, workspace-specific checks, tool-access fnmatch lists).

The package is pure Python with no NATS or postgres dependencies. Concrete loaders against postgres (broker) or NATS-proxied L3 (agent pod) live in the calling repo; this package only defines the `GrantLoader` / `MembershipLoader` protocols and the evaluation logic.

## Public API

Exports (see `src/threetears/agent/acl/__init__.py`):

- `evaluate_decision(ctx) -> bool` — fast yes/no hot path; cache-friendly when wired behind `AclCache`. Used on every production call.
- `evaluate_with_trail(ctx) -> EvaluationResult` — verbose, uncached introspection path returning every `(group, assignment, role) → contributed_actions` chain plus `limiting_side` for user×agent intersection queries. Used by the `/admin/v1/access/*` introspection endpoints and by integration tests that assert on WHY a decision was reached, not only the yes/no.
- `AclCache` — three-layer in-process TTL cache (`actor → [group_id]`, `(group_id, namespace_id) → action_set`, `(group_id, namespace_type, customer_id) → action_set`) with fine-grained invalidation hooks fired on group-membership / role / assignment change.
- Value types: `Group`, `GroupMembership`, `Role`, `RoleAssignment`, `Namespace`, `EvaluationContext`, `EvaluationResult`, `Trail`.
- Enums: `ActorType`, `MemberType`, `ScopeType`, `LimitingSide`.
- I/O protocols: `GrantLoader`, `MembershipLoader` — callers implement these against their persistence layer.

## Minimal usage

```python
from threetears.agent.acl import (
    AclCache,
    EvaluationContext,
    evaluate_decision,
    evaluate_with_trail,
)

cache = AclCache(grant_loader=my_grant_loader, membership_loader=my_membership_loader)

ctx = EvaluationContext(
    user_id=user_id,
    agent_id=agent_id,
    customer_id=customer_id,
    namespace=target_namespace,
    action="read",
)

# hot path
allowed = await evaluate_decision(ctx, cache=cache)

# introspection path
result = await evaluate_with_trail(ctx, cache=cache)
# result.decision, result.effective_actions, result.user_trails,
# result.agent_trails, result.limiting_side
```

Implicit ownership: if `namespace.owner_agent_id == ctx.agent_id`, the agent side short-circuits to full permissions with no group lookup or assignment query. Ownership is a property of the namespace row, never a grant.

## Reference

- Shard: `14-eng-ai-bot/docs/rbac-task-01-groups-roles-assignments.md`
- Hot path wiring: `14-eng-ai-bot/src/aibots/hub/broker/acl.py`, `3tears/packages/agent-workspace/src/threetears/agent/workspace/authorize.py`
- Last commit (phase 1): 3tears `9bff571`; phase 3 wiring: 3tears `e194ffa`, aibots `2b3e0d5`
