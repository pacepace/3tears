# 3tears-agent-acl

Shared RBAC evaluator + cache for the 3tears platform.

This package owns:

- `Role`, `RoleAssignment`, `GroupMembership`, `Namespace` value types
- `EvaluationContext`, `EvaluationResult`, `Trail` — the inputs and outputs of an authorization check
- `evaluate_decision(...)` — fast, cache-friendly yes/no path used on every call
- `evaluate_with_trail(...)` — verbose, uncached path used by the introspection API and by integration tests
- `AclCache` — three-layer in-process cache (`actor → [group_id]`, `(group_id, namespace_id) → action_set`, `(group_id, namespace_type, customer_id) → action_set`) with TTL + invalidation hooks
- `GrantLoader` / `MembershipLoader` Protocols — the I/O boundary; concrete loaders live in the broker (postgres) and in the agent pod (NATS-proxied L3)

The package is pure Python with no NATS or postgres dependencies. The same evaluator runs in the hub broker and inside every agent pod so authorization decisions are byte-identical across processes.

See `docs/rbac-task-01-groups-roles-assignments.md` in the platform repo for the model and the rationale.
