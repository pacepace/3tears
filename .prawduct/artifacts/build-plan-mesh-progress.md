---
artifact: build-plan
version: 1
scope: mesh-progress
branch: feat/mesh-progress
depends_on: [mcp-face]
partition: "serial -- one chunk; the wire, the pod, the registry and the caller are one protocol and are tested together"
last_validated: 2026-11-28
---

# Build plan: a tool's progress and log entries travel over the mesh

Branch: `feat/mesh-progress`, cut from `feat/mcp-face` (1378d2e9a), merged back into it with a merge commit once
reviewed. It ships in the release that carries `feat/mcp-face`.
Critic mode: one review of the branch.

## Requirements Confidence

**Level:** Medium

**Why:** the goal is an owner ruling (2026-11-28, a consumer's design question Q1: progress reaches an orchestrator
over the mesh too, not only on a local connection). The wire shape below is inferred.

## Goals

Today a `TearsTool` served over MCP reaches its caller's progress token and log channel through the call context
(`feat/mcp-face`); the same tool called over the mesh (caller -> registry -> pod) is given no context, so a long call
reports nothing until its result. Success:

- a tool on a pod whose `execute` declares the call context, called over the mesh by a caller that asked for
  updates, reports progress and log entries that reach the caller, in order, before the result;
- a caller's MCP face turns them into `notifications/progress` for its own client's progress token and
  `notifications/message` at the levels its client takes, with one argument (`ToolCallClient.call(updates=ctx)`);
- nothing is subscribed, published or passed when the caller asked for nothing (no progress token, no log level);
- the tool never waits on the link: an update returns at once, progress coalesces to the latest, the rate is
  bounded on the pod and again at the registry, and a lost update loses only itself;
- the local MCP path is unchanged.

Out of scope, said so: the agent SDK's and the hub's caller side (they adopt `CallUpdates` in their own repos; until
then they send none and nothing changes for them), cancellation over the mesh (the pod sets the context's cancel
signal when the call is cut short on the pod; a caller's cancel does not cross the mesh today and is not added here).

## Decisions taken (inferred; put to the owner in the PR body, not ruled)

- [DECISION: two subject families, named like the durable result pair: `{ns}.tools.progress.{pod_id}.{call_id}`
  (pod -> registry, the pod publishes under its own id on a standing grant) and `{ns}.tools.updates.{caller}.{call_id}`
  (registry -> caller, the caller subscribes its own subtree) | the reply inbox takes one response (`resp.max` is 1)
  and dies with the receiving connection, so updates cannot ride it; a pod can forge into no other pod's call and no
  caller can read another's, for the reasons `tools.result` / `tools.reply` give. Core NATS, not JetStream: an update
  is best effort and worth nothing after its call]
- [DECISION: the request carries `updates: CallUpdates(subject, progress_token, log_level)` on both hops, omitted when
  unset; the registry refuses a caller subject that does not name the verified caller and mints a fresh pod subject per
  forward attempt | the same containment the delivery subject has; a fresh subject per attempt keeps a failed-over
  pod's late updates out of the call]
- [DECISION: on the pod the context is a `threetears.mcp.CallContext` passed to an `execute` that declares one, found
  as the MCP face finds it, only when the call asked for updates; its progress token is the caller's | one tool
  definition, one context type, both faces; a mesh call that asked for nothing is called exactly as before. Needs the
  `mcp` extra of 3tears-tool-contract on the pod; without it a tool cannot declare the context at all]
- [DECISION: each update carries a per-call `seq`; the caller drops one that does not exceed the last, and progress
  that does not exceed the last progress | ordering survives a relay or reconnect, and the protocol requires progress
  to increase]
- [DECISION: rate bound by a token bucket: the pod sends at most 10 a second (burst 10), coalescing progress to the
  latest and keeping at most 64 pending entries (the oldest dropped first); the registry relays at most 20 a second
  per call (burst 40) and drops the rest; an update over 16 KiB is not sent (a log entry's data is replaced by a note
  of its size) | a tenant's tool is not trusted to bound itself, and an honest one never meets the registry's bound]
- [DECISION: before the pod answers, it flushes what is pending for at most 1 s | progress after the result is
  ignored by an MCP client, so the last progress must leave first; the bound keeps a dead link from delaying the
  answer]
- [DECISION: `CallContext` gains `log_levels` (the levels the client takes; the server passes its rule's set) and
  `log` sends nothing outside it | the caller has to tell the pod which levels to send, and must not ask for entries
  its client refuses]

## Chunk 01: mesh progress

- `packages/nats/src/threetears/nats/subjects.py`: the two families and their wildcards.
- `packages/nats/src/threetears/nats/call_updates.py` (new): the ownership checks, the bounds, `UpdateBudget`.
- `packages/nats/src/threetears/nats/subject_permissions.py`: tool pod (publish own progress, subscribe own
  updates), agent pod (publish own in-process progress), registry (subscribe every pod's progress, publish every
  caller's updates).
- `packages/core/src/threetears/core/tool_wire/messages.py`: `CallUpdates`, `CallUpdate`, `CallRequest.updates`.
- `packages/registry/src/threetears/registry/proxy.py`: `ProxyCallRequest.updates`, the caller-subject check, the
  relay on both forward paths.
- `packages/registry/src/threetears/registry/client.py`: `ToolCallClient.call(updates=)`, `CallUpdateSink`.
- `packages/agent/tools/src/threetears/agent/tools/call_updates.py` (new) and `server.py`: the pod's publisher and
  the context passed to `execute`.
- `packages/tool-contract/src/threetears/tool_contract/mcp_face.py`: `mesh_context_parameter`, `mesh_call_context`.
- `packages/mcp/src/threetears/mcp/context.py`, `server.py`: `CallContext.log_levels`.
- Tests: `packages/nats/tests/unit/test_call_updates.py`, `packages/agent/tools/tests/unit/tools/test_call_updates.py`,
  `packages/registry/tests/unit/registry/test_mesh_progress.py` (caller -> real registry -> real pod over an
  in-memory bus, and a real MCP client in front of the caller), `packages/mcp/tests/unit` (log levels),
  `tests/enforcement/test_subject_grants_match_the_code.py` (the new sites), subject and permission unit tests.
- Docs: the agent-tools, registry and mcp READMEs where they describe the context and the client; CHANGELOG.

Done when: lint, typecheck of the touched packages, the touched packages' targeted tests and the enforcement gates
named above pass; Critic run; merged into `feat/mcp-face` with a merge commit.
