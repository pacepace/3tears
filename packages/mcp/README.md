# 3tears-mcp

Shared MCP (Model Context Protocol) framework. Per-product MCP servers compose this framework instead of reimplementing stdio transport, JWT auth, error mapping, and per-tool RBAC.

## Install

```bash
pip install 3tears-mcp           # server, tools, identity, PlatformHttpClient
pip install "3tears-mcp[grants]" # plus the Postgres-backed grant store
```

The base depends on the `mcp` SDK, `3tears-observe`, httpx, starlette and uvicorn only. Everything
below that needs the rest of the family -- `LocalGrantAuthorizer`, `rbac`, `migrations/` -- is the
`grants` extra. Bring your own `Authorizer` and the base is all you install.

## What's in here

| Module | Responsibility |
|---|---|
| `server` | `McpServer` -- wraps the official `mcp.server.Server`. Owns tool registration, RBAC gating before handler dispatch, structured error mapping per the MCP spec. `tool_error` builds a tool-execution error a handler returns. |
| `tool` | `McpTool` dataclass (name, description, input_schema, required_permission, handler, and optional output_schema, title, annotations), `McpToolAnnotations`, `ToolRegistry`, `ToolProvider`, and the `register_tool` decorator. |
| `http_server` | The Streamable-HTTP transport: `McpHttpTransport` for a host app to embed, `build_mcp_http_app` / `serve_mcp_http` to serve standalone, `current_bearer_token` for `BearerTokenIdentityProvider`. |
| `http_client` | `PlatformHttpClient` -- typed httpx client with JWT login + refresh-on-401. Used by both MCP server tool handlers (calling /api/v1/...) and CLI scripts. One HTTP-client implementation, two transports. |
| `auth` | `Identity` dataclass + `IdentityProvider` Protocol + `EnvVarIdentityProvider` (stdio impl). `Authorizer` Protocol + `LocalGrantAuthorizer` (default impl backed by `McpToolGrantCollection`). |
| `rbac` | `McpToolGrantCollection` -- `BaseCollection` over `mcp_tool_grants`. Exposes the in-memory grant cache that `LocalGrantAuthorizer` consults. |
| `migrations/` | `v01_create_mcp_tool_grants` -- platform-scope DDL. Consumers register via `MigrationRunner.register(epoch_pkg)` (same shape as `threetears.epoch`). |

## Tool metadata

`McpTool` (and `register_tool`) take three optional fields, listed by `tools/list` under their
protocol names:

```python
McpTool(
    name="runners.list",
    description="list the runners",
    input_schema={"type": "object", "properties": {}},
    required_permission="runners.read",
    handler=list_runners,
    output_schema={"type": "object", "properties": {"runners": {"type": "array"}}, "required": ["runners"]},
    title="List runners",
    annotations=McpToolAnnotations(read_only_hint=True, idempotent_hint=True),
)
```

| Field | Listed as | Notes |
|---|---|---|
| `output_schema` | `outputSchema` | A JSON Schema whose `type` is `"object"`, checked when the tool is constructed. While a tool declares one, every successful call must carry `structuredContent` that validates against it. Otherwise the server answers `OUTPUT_VALIDATION_ERROR` in place of the result and logs an error, because the bug is in the tool. Error results are exempt. |
| `title` | `title` | A human-readable display name. `name` stays the identifier the client calls. |
| `annotations` | `annotations` | `McpToolAnnotations(read_only_hint=, destructive_hint=, idempotent_hint=, open_world_hint=)`, listed as `readOnlyHint`, `destructiveHint`, `idempotentHint` and `openWorldHint`. A hint left `None` is omitted, so the client applies the spec's default for it (read-only `False`, destructive `True`, idempotent `False`, open-world `True`). Hints are advice to the client, not something the server enforces. |

## Results and tool errors

What a handler returns becomes the call's result:

| Handler returns | The client receives |
|---|---|
| a `dict` | `structuredContent` set to the dict, plus a text block holding its serialized JSON (the spec asks for both) |
| any other JSON-serializable value | the same, with the value wrapped as `{"result": value}` |
| `str`, `TextContent` or a list of `TextContent` | text content only, with no `structuredContent` |
| `tool_error({...})` | `isError: true`, with the object as `structuredContent` and as serialized JSON in a text block |
| an `mcp.types.CallToolResult` | that result, passed through unchanged |

`tool_error` is for the spec's *tool-execution error*: the tool ran, and the call failed in a way the
caller can read and act on:

```python
from threetears.mcp import tool_error

async def get_runner(runner_id: str) -> dict:
    runner = registry.find(runner_id)
    if runner is None:
        return tool_error({"code": "NOT_FOUND", "message": f"no runner {runner_id!r}", "retryable": False, "field": "runner_id"})
    return {"runner": runner.describe()}
```

A tool error is not logged, it is not mapped to one of the framework's codes, and `output_schema`
doesn't apply to it. The framework's own refusals (`UNKNOWN_TOOL`, `PERMISSION_DENIED`,
`IDENTITY_UNAVAILABLE`, `AUTHZ_ERROR`, `HANDLER_ERROR`, `OUTPUT_VALIDATION_ERROR`, `TOOLS_UNAVAILABLE`)
are also `isError: true`, but they carry `{"tool_name": ..., "error": {"code": ..., "message": ...}}` in
a text block and no `structuredContent`. A handler that raises still gets `HANDLER_ERROR`.

## A live tool set

`McpServer` takes either a fixed `registry=` or a `tool_provider=`. A provider is a callable that
returns the tools as they stand right now (any iterable of `McpTool`). The server calls it on every
`tools/list` and every `tools/call`, so a tool set that follows a live registry is served as it is at
each request: a tool added at runtime can be called, a removed one answers `UNKNOWN_TOOL`, and arguments
are checked against the input schema the tool has now.

```python
server = McpServer(
    name="coordinator",
    identity_provider=identity_provider,
    authorizer=authorizer,
    tool_provider=lambda: [op_to_tool(op) for op in ops.current()],
)
```

Tool names must be unique in each answer. If the provider raises, or returns two tools with one name, a
`tools/call` answers `TOOLS_UNAVAILABLE` and logs an error, and a `tools/list` fails with a JSON-RPC
error. The server does not send `notifications/tools/list_changed`, because a stateless HTTP transport
has no session to push it on. Clients see a changed set on their next `tools/list`.

## Embedding the HTTP transport in a host app

`build_http_app()` returns a standalone Starlette app that runs the SDK's session manager in its own
lifespan. Starlette does not run a mounted sub-app's lifespan, so a host serving MCP inside its own app
uses `build_http_transport()` instead. It adds the transport's routes and enters the transport's
lifespan in its own:

```python
await server.start()
transport = server.build_http_transport(path="/mcp")  # stateless=True, json_response=True by default

@contextlib.asynccontextmanager
async def lifespan(app: Starlette) -> AsyncIterator[None]:
    async with transport.lifespan():
        yield

app = Starlette(routes=[*own_routes, *transport.routes], lifespan=lifespan)
```

- `transport.routes` serves the endpoint at exactly `path`.
- `transport.endpoint` is the raw ASGI app, for a host's own `Route` or `Mount`. A
  `Mount("/mcp", app=transport.endpoint)` answers at `/mcp/`, and Starlette redirects `POST /mcp` there,
  so use `transport.routes` if the endpoint has to be exactly `/mcp`.
- A request that arrives while the lifespan isn't running gets a 503 that names
  `McpHttpTransport.lifespan()`. Without that, the SDK raises a bare "Task group is not initialized".
- A transport runs once, because the SDK's session manager is single-use. A host that restarts builds a
  new transport.

Bearer identity works the same way it does under `build_http_app()`. The endpoint reads the request's
`Authorization: Bearer` header into `current_bearer_token()`, which is the `token_source` to give
`BearerTokenIdentityProvider`.

## Protocol versions

The protocol versions are the SDK's. With the locked `mcp` 2.3.0, the HTTP transport serves the
handshake-era revisions (up to 2025-11-25) and the 2026-07-28 single-exchange protocol, including
`server/discover`. A 2.x `mcp.Client` in its default `auto` mode probes `server/discover` and settles
on 2026-07-28, and its calls go through the same dispatch, RBAC gate and bearer identity
(`tests/unit/test_http_transport_embedding.py` proves it). On mcp 1.x the transport speaks the
handshake only. 3tears adds nothing per version, so a newer SDK's versions are served once the lock
moves to it.

## RBAC model

Per-tool, default-deny. Each `McpTool` declares a `required_permission` string (e.g. `"conversations.read"`, `"audit.read"`). On every dispatch:

1. The framework calls `Authorizer.allows(identity, required_permission)`.
2. `LocalGrantAuthorizer` checks whether the caller's identity matches an active grant in `McpToolGrantCollection` for the requested permission.
3. If denied, the framework returns a structured MCP error to the client (not a Python exception in the response body).

The configured admin identity (env-var creds in the stdio impl) is **auto-granted in memory at server startup**. The grant is logged but NOT written to `mcp_tool_grants`. This keeps the table truthful (only operator-added grants live there).

Grant changes propagate cross-pod via the `mcp.rbac` epoch broadcast. `LocalGrantAuthorizer` subscribes to `Subjects.mcp_rbac_epoch()` via an `EpochListener`; on bump it reloads the grant cache from L3. Cold-start primes from `EpochClient.current(...)`. Missed broadcasts recover via the standard pull-on-stale path.

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
