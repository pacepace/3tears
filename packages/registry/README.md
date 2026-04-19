# 3tears Registry

MCP-compatible tool registry for the 3tears tool system. Routes tool calls between agents and tool pods via NATS request/reply.

Part of the [3tears](https://github.com/pacepace/3tears) framework.

## Components

- **`ToolCatalog`** — in-memory index of registered tool pods, backed by a NATS KV bucket for recovery across restarts.
- **`RegistrationHandler`** — subscribes to `{ns}.tools.register` and mutates the catalog.
- **`HeartbeatMonitor`** — sweeps pods whose heartbeats fell behind the timeout and evicts their endpoints.
- **`DiscoveryHandler`** — serves `{ns}.tools.discover` for pod-readiness polling.
- **`CallProxy`** — the hot path. Subscribes to `{ns}.tools.call`, authorizes via `AgentToolAuthorizer`, selects an endpoint via the configured `RoutingStrategy`, and forwards the call to the tool pod via NATS request/reply with identity + correlation carried through the `CallContext` envelope.

## Authorization (namespace-task-01 Phase 2)

Tool dispatch authorization lives behind the `AgentToolAuthorizer` protocol. Implementations receive the calling agent id, the invoking user id (from `CallContext.user_id`), and the fully qualified tool name, and return a boolean decision.

Production deployments wire `RbacEvaluatorAuthorizer` (in `threetears.registry.rbac_authorizer`) which delegates to the unified rbac evaluator from `threetears.agent.acl`:

- `ToolServer.register_tool` emits a `platform.namespaces` row of type `tool` on registration (name shape `tool:<mcp_name>:<version>`).
- The authorizer resolves the tool namespace via an injected `NamespaceByNameResolver` (caller-supplied — production wiring uses the hub's L3 pool; tests use in-memory fixtures).
- `evaluate_decision` resolves the two-sided grant chain: user side (groups the invoking user is in) intersected with agent side (groups the calling agent is in, short-circuited by namespace ownership). The decision is cached in `threetears.agent.acl.AclCache` with TTL + fine-grained invalidation.

Defense in depth: when `user_id=None` (tool dispatch without user identity) the authorizer denies unconditionally. When the namespace resolver returns `None` (tool registered but namespace row not yet visible) it denies — this catches registration races rather than defaulting to allow.

Platform-built-in tools land with `owner_agent_id=NULL, customer_id=NULL` — there is no implicit "anyone can call" behaviour for them. Grants are managed via explicit assignments on the platform-seeded `ToolCaller` role (same pattern as shared-type workspaces).

The legacy `KvAgentToolAuthorizer` (fnmatch on `access.tools` patterns read from NATS KV) and the `access_cache`-backed `ToolAccessAuthorizer` were retired in the same phase that introduced `RbacEvaluatorAuthorizer`; no dual-enforcement window, no back-compat aliases. The declarative `access.tools` expression on `agent.yaml` stays as operator-facing syntax and is translated to RBAC assignments at bootstrap (`aibots_agents.runtime.access_translation.translate_access_tools_to_assignments`).

## Dev-mode authorizers

`AllowAllAuthorizer` permits every dispatch unconditionally — enabled by `FOURTEENAIBOTS_REGISTRY_ALLOW_ALL_TOOLS=true`. Use only in local dev containers.

`DenyAllAuthorizer` refuses every dispatch — the standalone `python -m threetears.registry.server` entry point starts with this when `ALLOW_ALL_TOOLS` is unset. Production deployments construct `RbacEvaluatorAuthorizer` programmatically in the host application's startup (loaders depend on the hub-side DB pool).

## Standalone entry point

```bash
python -m threetears.registry
```

Reads `THREETEARS_NATS_URL` (defaults to `nats://localhost:4222`) and `FOURTEENAIBOTS_NATS_SUBJECT_NAMESPACE` (defaults to `aibots`). The standalone entry point is appropriate for dev / testing only — it cannot construct the rbac authorizer without a hub-side loader seam, so it defaults to deny-all.
