---
artifact: build-plan
version: 1
scope: mcp-face
branch: feat/mcp-face
depends_on: [mcp-call-context]
partition: "serial -- six chunks on one branch; each builds on the ones before it (the error model carries every later refusal, the stateful session registry carries resource subscriptions, the bridge needs the contract package)"
last_validated: 2026-11-26
---

# Build plan: an MCP face for one tool definition

Branch: `feat/mcp-face`, cut from `feat/mcp-and-l3-stores` (95bdd0c16), which carries the SDK 1.x drop, the
call context and progress and logging over Streamable HTTP. One PR into develop, area mcp (with agent-tools,
agent-audit and the new contract package), after `feat/mcp-and-l3-stores` merges.
Critic mode: one cumulative review at the end (one branch, one PR); an inner-loop review per chunk at most.

## Requirements Confidence

**Level:** Medium

**Why:** the goal is a consumer's stated need, written as its requirements (a closed error-code set on the
protocol result, MCP resources, an opt-in stateful mode that re-checks the bearer on every message, an
authorizer that sees the call, one tool definition serving both faces, a contract that installs without the
platform, an audit publish that reports failure). The shapes below are inferred; the decisions are not owner
rulings.

## Goals

Today a 3tears-mcp failure carries its code only as JSON text and an argument-validation failure carries none;
there are no resources; stateful mode resolves the bearer once, from the request that opened the session, and
nothing pushes `notifications/tools/list_changed`; an `Authorizer` sees only the permission; a `TearsTool`
cannot be served over MCP; the `TearsTool` contract installs NATS, Postgres drivers, LangGraph and LangChain;
and a failed audit publish is only logged.

Success, one item each:

- **U04 error model:** every failed `tools/call` carries `{code, message, retryable, argument?}` on the
  protocol result (`_meta["threetears.mcp/error"]`), from a closed set: the framework's own refusals each with
  their own code (an argument-validation failure included, naming the argument at fault), the agent-tools
  `ToolResult` codes with their meaning, and the codes a server registers. An exception-to-code registry maps
  what a handler raises.
- **U05 resources:** `resources/list`, `resources/read`, `resources/templates/list`, a required permission per
  resource, and `resources/subscribe` / `notifications/resources/updated` in stateful mode (and on 2026-07-28's
  `subscriptions/listen`), tested over HTTP.
- **U06 stateful mode:** opt-in stateful Streamable HTTP with `notifications/tools/list_changed`, resource
  subscriptions and request cancellation reaching the call context; the bearer identity is resolved for every
  message from that message's own request, and a revoked token is refused on the next message of an open
  session (test).
- **U07 an authorizer that sees the call:** an authorizer that asks for it is given the tool name, the
  arguments and the permission, before the handler runs, and may refuse with a code of its own; existing
  authorizers are unchanged.
- **U08 bridge:** a `TearsTool` is served by `McpServer` with its schema, its `ToolResult` mapped to the call
  result (its code on the protocol result), and the call context reachable from `execute` without changing
  `TearsTool.run`.
- **U09 the contract on its own:** `TearsTool`, `ToolResult`, `MCPToolDefinition` and the result codes install
  without NATS, a Postgres driver, LangGraph or LangChain, checked by a resolver test in CI.
- **U12 agent-audit:** a failed publish is reported to the caller and counted, not only logged.

Out of scope, said so: U10 (`face_api` without the hub) and U11 (L1 on a file) are dropped (U11 was replaced by
the SQLite L3 already on the base); U13 (progress through `TearsTool` and the registry) waits on an owner
question and is not built.

## Decisions taken (inferred; put to the owner in the PR body, not ruled)

- [DECISION: the error rides the result's `_meta` under `threetears.mcp/error`, and the framework's own text
  envelope gains `retryable` (and `argument`) beside `code` and `message` | a failed tool's content is the
  tool's to own (a consumer's `{ok: false, ...}` envelope stays its content unchanged), so the protocol code
  needs a place of its own on the result; `_meta` is the protocol's extension point and a client that reads
  only the text still sees the code. The key is a valid `_meta` prefix (`3tears` starts with a digit, so the
  label is `threetears.mcp`)]
- [DECISION: one closed set per server, `ErrorCodes`: the built-in codes (`UNKNOWN_TOOL`, `TOOLS_UNAVAILABLE`,
  `INVALID_ARGUMENTS`, `IDENTITY_UNAVAILABLE`, `PERMISSION_DENIED`, `AUTHZ_ERROR`, `HANDLER_ERROR`,
  `OUTPUT_VALIDATION_ERROR`, `TOOL_ERROR`, and agent-tools' `CONFLICT`, `MALFORMED_REQUEST`,
  `TOOL_AUTHORIZATION_UNAVAILABLE`, `TOOL_NOT_READY`) plus the server's registered ones; a code may be lower
  case (a consumer's existing codes are) but not a case-variant of another | the existing refusal names are
  kept, since clients match on them; `PERMISSION_DENIED` is one code for both meanings, as it is one condition.
  A failure naming a code the server never registered is a server bug: the server logs it once per code and
  answers `HANDLER_ERROR`, keeping the tool's content]
- [DECISION: a handler signals a coded failure by raising `McpToolError(code, message, argument=...)` or by
  returning `tool_error(..., code=...)`; an exception type mapped with `ErrorCodes.map_exception` answers its
  code; anything else is `HANDLER_ERROR` as before; a tool error that names no code is `TOOL_ERROR` | both
  styles exist in consumers today]
- [DECISION: an authorizer that sees the call implements `CallAuthorizer.authorize_call(identity, call)`,
  answering `True`, `False` (`PERMISSION_DENIED`) or a `Refusal(code, message, argument)`; the server calls it
  in place of `allows` when the authorizer has it | an existing `Authorizer` keeps its one method; a consumer's
  gate answers with its own code (a tainted session, an out-of-scope session), which `PERMISSION_DENIED` would
  lose]
- [DECISION: the bearer token of each message is read from the HTTP request that carried it (the SDK hands it
  to the handler as `ServerRequestContext.request`) and bound for the identity provider for that message |
  in stateful mode the session's task runs with the context of the request that opened it, so the
  request-scoped token was the first request's for the session's life; the per-message request is the
  transport's own record of who sent this message]
- [DECISION: the server keeps the handshake-era connections that can be pushed to (stateful HTTP sessions,
  stdio), each until its connection closes, reaching the SDK's per-connection `Connection` through
  `ServerSession._connection` in one module (`_mcp_sdk_internals.py`, exempted, checked against the installed
  SDK) | the SDK 2.3 exposes a connection's standalone channel and teardown stack only on `Connection`, which a
  lowlevel handler is not given (its own TODO, L54); without the teardown hook a registry of sessions would
  grow for the server's life]
- [DECISION: `McpServer.notify_tools_changed()`, `notify_resources_changed()` and
  `notify_resource_updated(uri)` push to every handshake-era connection that can receive it (a resource update
  only to those subscribed to the URI) and publish on the SDK's subscription bus for 2026-07-28
  `subscriptions/listen` streams | both eras are served by one server, and a 2.x client settles on 2026-07-28 by
  default; the capabilities declare `listChanged` and `subscribe` only where they are delivered (not in
  stateless HTTP on the handshake-era protocols)]
- [DECISION: `resources/list` lists only the resources the caller may read (its identity resolved, each
  distinct permission asked once) | a resource's URI can name data (a session); `tools/list` is unchanged and
  lists every tool, as before]
- [DECISION: resource failures are JSON-RPC errors (the protocol has no `isError` result for resources):
  `-32002` for an unknown URI as the specification says, and a server error carrying the same code set in its
  `data` (`IDENTITY_UNAVAILABLE`, `PERMISSION_DENIED`, `AUTHZ_ERROR`, `HANDLER_ERROR`) | one vocabulary across
  tools and resources]
- [DECISION: U09 is a new family package, `3tears-tool-contract` (import `threetears.tool_contract`), holding
  `TearsTool`, `ToolResult`, `MCPToolDefinition`, the result codes and the input coercion, with only
  `3tears-observe` and `3tears-tool-schema` as dependencies; `threetears.agent.tools.base_tool` and
  `.coercion` re-export the same objects, so every existing import keeps working and a tool written against
  either path is the same class | the lightest change that meets TRS-17: a light base for agent-tools with the
  rest as extras would break every existing install of it, and the existing dependency-free packages
  (`interfaces`, `tool-schema`) are held to stdlib-only, which the contract cannot keep (its coercion fallback
  reports through observe). `TearsTool.face_rest`'s type, `RestAffordance`, lives in core and is imported for
  type checking only, so the contract pulls no core at run time]
- [DECISION: the bridge lives in the contract package behind an `mcp` extra
  (`3tears-tool-contract[mcp]`, module `threetears.tool_contract.mcp_face`), not in 3tears-mcp | 3tears-mcp's
  hard-dependency floor is an owner ruling (2026-10-17); the bridge needs both sides, and putting it on the
  contract's side leaves that floor and the mesh-only installs of the contract untouched]
- [DECISION: the bridge refuses a tool whose `face_mcp` is `False`, takes the MCP name from the caller (default
  `mcp_name()`), and does not enforce `timeout_seconds` | `face_mcp` is the tool's own declaration of this
  reach; a dotted canonical name is rejected by some LLM tool APIs, so a local face names its tools itself; the
  timeout is documented as the caller's default, which the MCP client's own deadline is]
- [DECISION: the call context reaches `execute` as a keyword argument, when `execute` declares a parameter
  annotated `CallContext` (found as `McpTool` finds a handler's); `TearsTool.run` passes keys outside the input
  schema through unchanged | the same rule a plain handler has, no change to `run`; the parameter's default of
  `None` keeps the tool callable on the mesh, where no context exists]
- [DECISION: `publish_audit` returns `AuditPublishOutcome` (`PUBLISHED`, `SKIPPED`, `FAILED`), and every failure
  is counted in-process (`audit_publish_failure_count()`) and as the Prometheus counter
  `threetears_audit_publish_failures_total` | the caller learns of the failure on the call, and a server's own
  status reads the count; publish still never raises, so a tool call never depends on audit health]
- [DECISION: the resolver check is an enforcement test over `uv export --frozen --package ... --no-dev` (the
  lock's resolution of what an install pulls), for `3tears-tool-contract` and for `3tears-mcp`'s base | the
  same source `test_core_imports_without_arrow.py` reads; it needs no network, so the pool runs it]

## Chunk 01: the error model (U04)

- `packages/mcp/src/threetears/mcp/errors.py`: `ErrorCode`, `ErrorCodes`, `McpToolError`, the built-in codes,
  `ERROR_META_KEY`.
- `server.py`: `McpServer(error_codes=...)`; each refusal names its code; argument validation answers
  `INVALID_ARGUMENTS` with the argument at fault; an exception maps through the registry; a returned failure is
  checked against the set; `tool_error(..., code=, message=, argument=)`, and a text form.
- Tests: `packages/mcp/tests/unit/test_error_model.py` (over the SDK client).

## Chunk 02: an authorizer that sees the call (U07)

- `auth.py`: `ToolCall`, `Refusal`, `CallAuthorizer`; `server.py` calls `authorize_call` when present.
- Tests: `packages/mcp/tests/unit/test_call_authorizer.py`.

## Chunk 03: stateful mode (U06)

- `http_server.py`: the per-message bearer binding; whether the transport can push.
- `_mcp_sdk_internals.py`, `notifications.py` (the connections a server pushes to, the subscription bus),
  `server.py` (`notify_tools_changed`, capabilities, the listen handler).
- Tests: `packages/mcp/tests/unit/test_stateful_mode.py`: a revoked token refused on the next message of an
  open session and no tool run; a rotated token served; `tools/list_changed` in stateful HTTP and on 2026-07-28
  listen streams, none in stateless; cancellation (`notifications/cancelled`, the client deleting its session)
  reaching the call context.

## Chunk 04: resources (U05)

- `resources.py`: `McpResource`, `McpResourceTemplate`, `ResourceRegistry`, `ResourceProvider`,
  `ResourceContent`; `server.py`: list, read, templates, subscribe / unsubscribe, `notify_resource_updated`,
  `notify_resources_changed`.
- Tests: `packages/mcp/tests/unit/test_resources.py` over HTTP, both eras.

## Chunk 05: the contract on its own (U09)

- `packages/tool-contract/` (pyproject, README, LICENSE, `threetears.tool_contract`); agent-tools re-exports;
  registered where a member is (root pyproject, `.testament.toml` areas, `typecheck.sh`, README, dependency
  floors, the lock).
- `tests/enforcement/test_tool_contract_installs_light.py`: the resolver check.

## Chunk 06: the bridge (U08), and agent-audit (U12)

- `threetears.tool_contract.mcp_face.mcp_tool`; tests in `packages/tool-contract/tests/`.
- `packages/agent/audit/src/threetears/agent/audit/publish.py`: `AuditPublishOutcome`, the count; tests.

Docs change with the code in each chunk: the package READMEs and the CHANGELOG's Unreleased sections.

Done when: lint, `typecheck.sh` for each touched package, the tests of each touched package and their
importers, `scripts/test-mcp-floor.sh`, the enforcement suite; then once, after the cumulative Critic's fixes,
the full unit suite and the four integration suites on the pool.
