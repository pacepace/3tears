---
artifact: build-plan
version: 1
scope: mcp-call-context
branch: feat/mcp-progress-logging
depends_on: [mcp-drop-sdk1]
partition: "serial -- two stacked chunks in one package; chunk 02 tests what chunk 01 builds"
last_validated: 2026-11-25
---

# Build plan: 3tears-mcp call context, progress and client logging

Branches, stacked: `feat/mcp-call-context` (chunk 01, off `feat/mcp-drop-sdk1`) and
`feat/mcp-progress-logging` (chunk 02, off chunk 01). One PR each into develop, area mcp, merged in
that order after `feat/mcp-drop-sdk1`.
Critic mode: one review per chunk (each branch is reviewed and merged on its own).

## Requirements Confidence

**Level:** Medium

**Why:** the goal rests on a consumer's stated need (real progress for each phase of a long call, each
phase also logged, before the result, and stopping on cancellation); the shape of the surface is
inferred, and the decisions below are not owner rulings.

## Goals

A 3tears-mcp tool handler is called as `handler(**arguments)` and sees nothing of the request, so it
can neither report progress nor notice that the client gave up. A consumer serving long-running tools
through 3tears-mcp must send real `notifications/progress` for each phase of a call, each phase also
as a log message, before the result, and must stop its work when the call is cancelled.

Success:

- chunk 01: a handler that declares a `CallContext` parameter gets the progress token, a progress
  reporter, a client log channel and a cancellation signal; a handler that does not is called as
  before; both tested over a real MCP client.
- chunk 02: a real SDK client over Streamable HTTP receives `notifications/progress` and
  `notifications/message` before the call's result, in stateless and stateful modes, on the
  handshake-era protocol and on 2026-07-28; a transport that cannot carry them (plain-JSON
  responses) says so to the operator instead of dropping them silently.

Out of scope: `TearsTool`'s own progress (an open owner question), stateful mode (re-resolving the
bearer identity per message, `tools/list_changed`), and the error model.

## Decisions taken (inferred; put to the owner in the PR body, not ruled)

None of these is an owner ruling. Each describes the branch as it stands, and the PR body asks the
owner to rule on each; the last (the logging capability) is the one the README calls open.

- [DECISION: the context is found by the parameter's annotation, `CallContext`, not by a parameter
  name | the request was "detected by signature"; a fixed name would collide with tool arguments. An
  annotation that cannot be evaluated (a type imported only for type checking) is read by the import
  statements of the handler's module: a name imported from `threetears.mcp` (an alias included) or the
  qualified `threetears.mcp.CallContext` is the context, and `threetears.core`'s own `CallContext` (a
  tool dispatch's identity envelope) is not. `CallContext | None` (or `Optional[CallContext]`, or
  either inside `Annotated`) is the context too, and is passed one. Any other form that names it
  (`list[CallContext]`, `CallContext | int`) is refused at construction, since the handler would
  otherwise run without a context in silence. Each parameter's annotation is evaluated on its own, so
  one that fails does not hide another; one that cannot be evaluated, is not an input property and
  names nothing its module imports is logged as a warning, once per tool and parameter. The name
  `CallContext` is kept beside core's: reading by imports, not the name, tells the two apart]
- [DECISION: the context is a 3tears class with no SDK type in its surface; its senders are plain
  callables | consumer code must not need to import an SDK type to get these, and a handler's own
  test builds a context with recording senders]
- [DECISION: the cancellation signal is an `asyncio.Event` the dispatch sets when the handler's call is
  cancelled; the handler itself sees `CancelledError` as before | the SDK cancels the handler's scope
  on `notifications/cancelled` and on a dropped SSE stream, and does not expose its own
  `cancel_requested` event to a low-level handler. asyncio rather than anyio: the owner's ruling of
  2026-10-17 limits 3tears-mcp's hard dependencies, and 3tears runs on asyncio]
- [DECISION: `report_progress` refuses a value that is not finite or does not exceed the last |
  the protocol requires progress to increase; a handler bug surfaces in its own tests]
- [DECISION: log entries are sent as `notifications/message` through `send_notification`, gated by the
  SDK's own `allowed_log_levels` | the SDK's `send_log_message` warns (a `UserWarning`) on every call,
  since 2026-07-28 deprecates the logging capability, while the message itself remains; the gate keeps
  the 2026-07-28 per-request opt-in]
- [DECISION: the server does not declare the `logging` capability nor serve `logging/setLevel` | the
  SDK declares it only when an `on_set_logging_level` handler is passed, which it deprecates with a
  warning; the SDK client accepts `notifications/message` without it. OPEN for the owner: on the
  handshake-era protocols the specification has a server that sends log entries declare the
  capability, so a stricter client may drop them. The README states the current behaviour and says
  the point is open]

## Chunk 01: the call context (`feat/mcp-call-context`)

- `context.py`: `CallContext`, `LogLevel`.
- `tool.py`: `McpTool.context_parameter`, found at construction; refusals for two, positional-only, or
  an input property's name.
- `server.py`: the dispatch builds the context from the SDK's request context, refuses an argument
  named as the context, and sets the cancellation event when the call is cancelled.
- `packages/mcp/tests/unit/test_call_context.py`, `packages/mcp/tests/unit/conftest.py` (`loopback_mcp`, a real socket for the
  cancellation test); README section; CHANGELOG.
- The 2026-07-28 per-request log opt-in, which this chunk documents, is tested here with
  `mcp.Client`; chunk 02 adds the stream-ordering and transport-mode tests.
- The transport requirement is stated where a consumer adopting the context reads: progress and log
  entries need `json_response=False`, and the default delivers neither. The operator warning for the
  default is chunk 02's, and chunk 02 merges before any release is cut from develop; until then the
  CHANGELOG entry and the README state the requirement.

Done when: lint, `typecheck.sh mcp`, `test.sh mcp`, the enforcement suite pass; Critic run.

## Chunk 02: progress and logging over Streamable HTTP (`feat/mcp-progress-logging`)

- Wire tests with the SDK's client: progress and log entries arrive before the result, in stateless
  and stateful modes, on 2025-11-25 and 2026-07-28 (with and without the log opt-in), through
  `build_http_app` and an embedded `McpHttpTransport`.
- A transport built with `json_response=True` cannot carry them: the server logs one warning per
  server naming the fix the first time a handler's progress or log entry is dropped that way, in
  stateless and in stateful mode (the stateful case proves the request's transport reaches the
  session task the handler runs in).
- README: which transport options deliver progress, logging and cancellation, each row pinned by a
  test (including that a plain-JSON stateless call is not cancelled when its client stops waiting);
  CHANGELOG.
- Files: `packages/mcp/src/threetears/mcp/http_server.py`, `packages/mcp/src/threetears/mcp/server.py`,
  `packages/mcp/tests/unit/test_progress_and_logging.py`, `packages/mcp/tests/unit/conftest.py`.

Done when: lint, `typecheck.sh mcp`, `test.sh mcp`, the enforcement suite and the unit suite pass;
Critic run.
