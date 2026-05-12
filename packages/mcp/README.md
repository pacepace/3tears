# 3tears-mcp

Shared MCP (Model Context Protocol) framework. Per-product MCP servers (metallm, hub, agent-admin) compose this framework instead of reimplementing stdio transport, JWT auth, error mapping, and per-tool RBAC.

## What's in here

| Module | Responsibility |
|---|---|
| `server` | `McpServer` -- wraps the official `mcp.server.Server`. Owns tool registration, RBAC gating before handler dispatch, structured error mapping per the MCP spec. |
| `tool` | `McpTool` dataclass (name, description, input_schema, required_permission, handler) and `register_tool` decorator. |
| `http_client` | `PlatformHttpClient` -- typed httpx client with JWT login + refresh-on-401. Used by both MCP server tool handlers (calling /api/v1/...) and CLI scripts (`debug-api.py`, `debug-token.py`, `settings-transfer.py`). One HTTP-client implementation, two transports. |
| `auth` | `Identity` dataclass + `IdentityProvider` Protocol + `EnvVarIdentityProvider` (v1 stdio impl). `Authorizer` Protocol + `LocalGrantAuthorizer` (default impl backed by `McpToolGrantCollection`). |
| `rbac` | `McpToolGrantCollection` -- `BaseCollection` over `mcp_tool_grants`. Exposes the in-memory grant cache that `LocalGrantAuthorizer` consults. |
| `migrations/` | `v01_create_mcp_tool_grants` -- platform-scope DDL. Consumers register via `MigrationRunner.register(epoch_pkg)` (same shape as `threetears.epoch`). |

## RBAC model

Per-tool, default-deny. Each `McpTool` declares a `required_permission` string (e.g. `"metallm.conversations.read"`, `"hub.audit.read"`). On every dispatch:

1. The framework calls `Authorizer.allows(identity, required_permission)`.
2. `LocalGrantAuthorizer` checks whether the caller's identity matches an active grant in `McpToolGrantCollection` for the requested permission.
3. If denied, the framework returns a structured MCP error to the client (not a Python exception in the response body).

The configured admin identity (env-var creds in v1) is **auto-granted in memory at server startup**. The grant is logged but NOT written to `mcp_tool_grants` — keeps the table truthful (only operator-added grants live there).

Grant changes propagate cross-pod via the **task-02** `mcp.rbac` epoch broadcast. `LocalGrantAuthorizer` subscribes to `Subjects.mcp_rbac_epoch()` via an `EpochListener`; on bump it reloads the grant cache from L3. Cold-start primes from `EpochClient.current(...)`. Missed broadcasts recover via the standard pull-on-stale path.

## Identity in v1

`EnvVarIdentityProvider` returns one fixed `Identity` for the lifetime of the server, derived from env-var credentials. v2 (HTTP transport + per-call bearer-token identity) plugs in by adding a `BearerTokenIdentityProvider`; the rest of the framework is unchanged. The `Authorizer.allows(identity, permission)` interface is unchanged between v1 and v2.

## Stdio transport discipline

Every byte on stdout/stderr from a stdio MCP server confuses the client. The framework configures logging to a file or to NATS; **no module under `threetears.mcp` should write to `sys.stdout` / `sys.stderr`**. An AST enforcement test guards this.

## Postgres backing

```
CREATE TABLE IF NOT EXISTS mcp_tool_grants (
    grant_id UUID PRIMARY KEY,
    principal_type TEXT NOT NULL,   -- 'user' | 'group' | 'role'
    principal_id UUID NOT NULL,
    tool_name TEXT NOT NULL,
    permission TEXT NOT NULL,
    date_created TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

Mutation paths (admin `POST /admin/mcp/grants`, `DELETE /admin/mcp/grants/{id}`) bump `Subjects.mcp_rbac_epoch()` after the row commit; sibling pods reload via the epoch listener.
