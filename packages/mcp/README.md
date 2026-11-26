# 3tears-mcp

Shared MCP (Model Context Protocol) framework. Per-product MCP servers compose this framework instead of reimplementing stdio transport, JWT auth, error mapping, and per-tool RBAC.

## Install

```bash
pip install 3tears-mcp           # server, tools, identity, PlatformHttpClient
pip install "3tears-mcp[grants]" # plus the Postgres-backed grant store
```

The base depends on the `mcp` SDK (2.x; 1.x is not supported), `3tears-observe`, httpx, starlette and
uvicorn only. Everything below that needs the rest of the family -- `LocalGrantAuthorizer`, `rbac`,
`migrations/` -- is the `grants` extra. Bring your own `Authorizer` and the base is all you install.

## What's in here

| Module | Responsibility |
|---|---|
| `server` | `McpServer` -- wraps the official `mcp.server.Server`. Owns tool registration, RBAC gating before handler dispatch, structured error mapping per the MCP spec. `tool_error` builds a tool-execution error a handler returns. |
| `errors` | The error model: `ErrorCode`, `ErrorCodes` (a server's closed code set and the exceptions it maps), `McpToolError`, the built-in codes, `ERROR_META_KEY`. |
| `tool` | `McpTool` dataclass (name, description, input_schema, required_permission, handler, and optional output_schema, title, annotations), `McpToolAnnotations`, `ToolRegistry`, `ToolProvider`, and the `register_tool` decorator. |
| `context` | `CallContext`, the per-call context a handler asks for by signature: the progress token and reporter, a log channel to the client, and a cancellation signal. |
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
caller can read and act on. Its content is the tool's own (a JSON object, which also rides as
`structuredContent`, or text), and `code=` names the failure:

```python
from threetears.mcp import ErrorCode, ErrorCodes, McpServer, tool_error

NOT_FOUND = ErrorCode("NOT_FOUND", "no such runner")

async def get_runner(runner_id: str) -> dict:
    runner = registry.find(runner_id)
    if runner is None:
        return tool_error({"ok": False, "error": f"no runner {runner_id!r}"}, code=NOT_FOUND, argument="runner_id")
    return {"runner": runner.describe()}

server = McpServer(..., error_codes=ErrorCodes([NOT_FOUND]))
```

A tool error is not logged, and `output_schema` doesn't apply to it. A handler that raises still gets
`HANDLER_ERROR`, unless it raises `McpToolError(code, message, argument=...)` or an exception the server mapped.

## Error codes

Every failed `tools/call` carries its error on the protocol result, under `_meta["threetears.mcp/error"]`
(`ERROR_META_KEY`):

```json
{"code": "INVALID_ARGUMENTS", "message": "'name' is a required property", "retryable": false, "argument": "name"}
```

`retryable` says whether the same call, unchanged, can succeed later. `argument` names the argument at fault,
when one is. The content stays what it was: the framework's own refusals carry
`{"tool_name": ..., "error": {"code", "message", "retryable", "argument"?}}` in a text block (and no
`structuredContent`), an argument-validation failure keeps the SDK's `Input validation error: ...` text, and a
tool's failure keeps the tool's content. So a client that reads only the text is unaffected, and one that branches
on the code reads it from `_meta`.

The codes are a closed set, `McpServer.error_codes` (an `ErrorCodes`). The built-in ones:

| Code | Retryable | When |
|---|---|---|
| `UNKNOWN_TOOL` | no | no tool of that name is served |
| `TOOLS_UNAVAILABLE` | yes | the tool provider failed |
| `INVALID_ARGUMENTS` | no | the arguments do not match the input schema; `argument` names the one at fault |
| `IDENTITY_UNAVAILABLE` | no | the caller's identity could not be resolved |
| `PERMISSION_DENIED` | no | the caller may not call the tool (or may not have what it asked for) |
| `AUTHZ_ERROR` | yes | the authorizer failed, or its grant cache is reloading |
| `HANDLER_ERROR` | no | the handler raised something it did not name, or named a code the server never registered |
| `OUTPUT_VALIDATION_ERROR` | no | a successful result breaks the tool's `output_schema` |
| `TOOL_ERROR` | no | the tool returned a failure (`isError`) and named no code |
| `UNKNOWN_RESOURCE` | no | no resource or template matches a URI (resources, below) |
| `CONFLICT`, `MALFORMED_REQUEST`, `TOOL_AUTHORIZATION_UNAVAILABLE`, `TOOL_NOT_READY` | yes, no, yes, yes | a `ToolResult`'s codes from 3tears-agent-tools, with their meaning there |

A server adds its own with `ErrorCodes([...])` or `register(code)`. A code is a letter followed by letters, digits
and underscores; lower case is allowed, but not a code that differs from another only in case. `map_exception(
KeyError, NOT_FOUND)` answers a code when a handler raises that type (the most specific mapping wins). A failure
that names a code the server never registered answers `HANDLER_ERROR` and logs an error once per code, since the
server is missing a registration.

## The call context

A handler that wants to talk to the client while it runs declares a parameter annotated
`CallContext`. The server finds it from the handler's signature when the `McpTool` is built and passes
the call's context in it. A handler that declares none is called with its arguments only, as before.

```python
from threetears.mcp import CallContext

async def build(*, target: str, ctx: CallContext) -> dict:
    await ctx.report_progress(1, 3, "cloning")
    await ctx.log("info", {"phase": "clone", "target": target})
    ...
```

| Member | What it is |
|---|---|
| `progress_token` | The token the client sent to ask for progress, or `None`. |
| `report_progress(progress, total=None, message=None)` | Sends `notifications/progress` for this call. It sends nothing when the client sent no token. Each value must be finite and greater than the last, as the protocol requires; a value that isn't raises `ValueError`. |
| `log(level, data, *, logger=None)` | Sends a `notifications/message` entry to the client on this call's stream. On protocol 2026-07-28 and later it is sent only when the request opted in at that level or below (the `io.modelcontextprotocol/logLevel` key of its `_meta`). |
| `cancelled`, `wait_cancelled()` | Set when the call is cancelled: the client sent `notifications/cancelled` or went away. The handler itself sees the cancellation as `CancelledError` at its next `await`. The signal is for work the call started outside its own task, such as a worker thread or a task in a longer-lived group. `wait_cancelled()` returns only on cancellation; a call that ends normally never sets it, so whoever starts such a waiter stops it when the call returns. |
| `tool_name`, `request_id` | The tool being called and the call's protocol request id. |

The context holds no MCP SDK type, so a handler needs no SDK import to use it. Its parameter is
passed by keyword, and may be annotated `CallContext | None` (or `Optional[CallContext]`) with a
default of `None` to keep the handler callable without one; the server passes a context either way. A
tool whose handler declares two, a positional-only one, one named as an input property, or one that
names `CallContext` in any other form (`list[CallContext]`, `CallContext | int`) is refused when the
`McpTool` is built. An annotation that cannot be evaluated (a type imported only for type checking) is
read by the import statements of the handler's module: a name imported from `threetears.mcp`, an alias
of it included, or the qualified `threetears.mcp.CallContext`, is the call context, and the
`CallContext` that `threetears.core` exports is not. One that names nothing the module imports is
logged as a warning, once per tool and parameter, instead of passing it nothing in silence. An argument with the context's name, admitted by a
schema that allows extra properties, is refused as an input validation error before the handler runs.
A test of a handler builds its own `CallContext(tool_name=..., progress_token=..., progress_sender=...,
log_sender=...)` with senders that record what they are given.

### Progress, logging and cancellation over Streamable HTTP

Progress and log entries ride the call's own response stream, so they reach the client before the
call's result. Over Streamable HTTP that stream is SSE, which a transport answers with when it is built
with `json_response=False`. That works in stateless mode as well as stateful, on the handshake-era
protocol and on 2026-07-28:

```python
app = server.build_http_app(json_response=False)            # or build_http_transport / serve_http
```

| | `json_response=False` (SSE) | `json_response=True` (the default, plain JSON) |
|---|---|---|
| progress and log entries | sent on the call's stream, before the result | not sent: the response is the result alone. The server logs one warning, the first time a handler's entry is dropped, naming `json_response=False` |
| the client gives up on the call (cancels it, dropping its stream) | the handler is cancelled, and `ctx.cancelled` is set | in stateless mode the handler is not cancelled; it runs to its end |

`json_response=True` stays the default, so a client that reads a plain JSON body is unaffected. A
server whose tools report progress builds its transport with `json_response=False`; every MCP client
handles both, since the protocol requires it. `current_request_streams()` (in
`threetears.mcp.http_server`) says whether the request being served can carry them.

On 2026-07-28 the client chooses per request, as above. On the handshake-era protocols (up to
2025-11-25) the server declares the `logging` capability in its `initialize` result and serves
`logging/setLevel`, as the specification has a server that sends log entries do (owner-confirmed
2026-11-25):

| | handshake-era protocols (up to 2025-11-25) | 2026-07-28 |
|---|---|---|
| `logging` capability | declared | not declared (the protocol removed it) |
| which levels a call's log entries are sent at | every level, until the connection sends `logging/setLevel`; from then on that level and every more severe one, for the connection's later calls (a later `setLevel` replaces it) | the level the request opts in at, and every more severe one; none when it does not opt in |

A level is the connection's: another client's connection keeps every level. In stateless mode each
request is a connection of its own, so `logging/setLevel` is answered but leaves no level for a later
call to keep, and every level is sent as before. The SDK's server warns that the logging capability is
deprecated as of 2026-07-28; `McpServer` serves it on purpose for the handshake-era protocols and
builds its SDK server without that warning.

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
error. When the set changes, `await server.notify_tools_changed()` tells the clients that can be told
(stateful mode, below); every other client sees the changed set on its next `tools/list`.

## Stateful mode

Stateless is the default: each HTTP request is a connection of its own. `stateless=False` (on
`build_http_app`, `build_http_transport` and `serve_http`) serves stateful sessions on the handshake-era
protocols: the client keeps an `Mcp-Session-Id` across requests and a standalone stream the server can
push to.

```python
app = server.build_http_app(stateless=False, json_response=False)
...
await server.notify_tools_changed()            # notifications/tools/list_changed to every client that can take it
```

| | stateless (default) | stateful |
|---|---|---|
| the bearer | resolved for every message | resolved for every message: a token revoked while a session is open is refused (`IDENTITY_UNAVAILABLE`) on the session's next message, and no tool runs |
| `tools.listChanged` capability (handshake era) | not declared | declared |
| `notify_tools_changed()` | reaches no handshake-era client (it lists again on its next request) | sent on each open session's standalone stream |
| cancellation (`notifications/cancelled`, the client deleting its session, a dropped stream) | cancels the handler with an SSE transport (the progress table above) | cancels the handler, and `ctx.cancelled` is set |

On 2026-07-28 there are no sessions in either mode: a client asks for change notifications with a
`subscriptions/listen` request, whose response stream the server keeps open, and `notify_tools_changed()`
reaches each such stream too. stdio is one connection for the server's life, and is pushed to as a stateful
session is.

The bearer of each message is the one its own HTTP request carried: the SDK runs each message's handler in the
contextvars of the request that delivered it, so `current_bearer_token()` answers that message's token even on a
session opened by an earlier request (`tests/unit/test_stateful_mode.py` holds the SDK to it). A message without a
bearer is resolved as having none, never as an earlier message's caller.

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
`server/discover`. An `mcp.Client` in its default `auto` mode probes `server/discover` and settles
on 2026-07-28, and its calls go through the same dispatch, RBAC gate and bearer identity
(`tests/unit/test_http_transport_embedding.py` proves it). 3tears adds nothing per version, so a newer
SDK's versions are served once the lock moves to it.

## RBAC model

Per-tool, default-deny. Each `McpTool` declares a `required_permission` string (e.g. `"conversations.read"`, `"audit.read"`). On every dispatch:

1. The framework calls `Authorizer.allows(identity, required_permission)`.
2. `LocalGrantAuthorizer` checks whether the caller's identity matches an active grant in `McpToolGrantCollection` for the requested permission.
3. If denied, the framework returns a structured MCP error to the client (not a Python exception in the response body).

The configured admin identity (env-var creds in the stdio impl) is **auto-granted in memory at server startup**. The grant is logged but NOT written to `mcp_tool_grants`. This keeps the table truthful (only operator-added grants live there).

Grant changes propagate cross-pod via the `mcp.rbac` epoch broadcast. `LocalGrantAuthorizer` subscribes to `Subjects.mcp_rbac_epoch()` via an `EpochListener`; on bump it reloads the grant cache from L3. Cold-start primes from `EpochClient.current(...)`. Missed broadcasts recover via the standard pull-on-stale path.

### An authorizer that sees the call

`allows(identity, permission)` sees no arguments, so a gate that depends on one (a session the caller may not
reach, a target whose state refuses the call) cannot be written as it. An authorizer whose class also defines
`authorize_call(identity, call)` is given the whole call, as a `ToolCall(tool_name, arguments,
required_permission)`, in place of `allows`. It runs after the arguments pass the input schema and before the
handler, and answers `True`, `False` (`PERMISSION_DENIED`) or a `Refusal(code, message, argument=None)` to refuse
with a code of the server's set:

```python
from threetears.mcp import ErrorCode, ErrorCodes, Refusal, ToolCall

TAINTED = ErrorCode("tainted", "the session is tainted")

class SessionGate:
    async def allows(self, identity, permission):
        return True

    async def authorize_call(self, identity, call: ToolCall) -> bool | Refusal:
        session = call.arguments.get("session")
        if session in tainted_sessions:
            return Refusal(TAINTED, f"session {session!r} is tainted", argument="session")
        return True

    async def start(self): ...
    async def stop(self): ...

server = McpServer(..., authorizer=SessionGate(), error_codes=ErrorCodes([TAINTED]))
```

`call.arguments` is read-only. A refusal naming a code the server never registered answers `PERMISSION_DENIED`
and logs an error once per code. An authorizer that raises denies with `AUTHZ_ERROR`, as `allows` does. The
`CallAuthorizer` protocol states the shape; an authorizer with `allows` alone is called exactly as before.

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
