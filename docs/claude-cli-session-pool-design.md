# Claude CLI session pool — design

A subscription credential (a Claude long-lived token, `sk-ant-oat…`) is spent by driving the
bundled Claude Code CLI as a subprocess. `langchain-claude-code` opens a fresh
`ClaudeSDKClient` — a fresh subprocess — on **every** call. Measured on a production
deployment: ~2.4 s to start plus ~2.7 s per call, against ~600 ms for the same model over the
HTTP API, and a routing decision stretched to 4–19 s under load. Every conversation turn pays
it too.

This package pools the CLI: one subprocess per launch configuration, reused, and rewound to an
empty conversation between calls. Everything below was verified against the bundled CLI (claude-agent-sdk 0.2.116 and
0.2.118), not inferred from documentation. The extra bounds the SDK below 0.3 for that reason.

## What is fixed when a CLI starts, and what can change per call

| Setting | How it reaches the CLI | Per call? | Evidence |
|---|---|---|---|
| system prompt | as a launch flag, no: a second `initialize` returns before reading `systemPrompt`. As a named agent defined at launch, **yes**: `apply_flag_settings {"agent": name}` | yes, among the agents the CLI launched with | through a recording proxy, a switched agent's request (system blocks, messages, tools, `output_config`) was byte-identical to a CLI launched with that prompt; the switch held across the rewind; an agent defined after launch answered 'Agent "..." not found' |
| JSON schema | `--json-schema` launch flag | **no** | `apply_flag_settings {"jsonSchema": ...}` answered `{}` and the next call had no `structured_output` |
| model | `set_model` control request | yes | SDK method |
| bound tools (in-process MCP server) | `mcp_set_servers` control request | **yes** | swapped a live session's server in 3 ms; the model called the new tool |
| `max_turns` | `--max-turns` launch flag | per *query* | three queries at cap 2, each used 2 turns, none errored; the model forces 1 (below) |
| conversation | `rewind_conversation` control request to the call's first message | yes | planted a codeword, rewound, asked the model to quote its whole input: only the CLI's own `currentDate` reminder, 10 calls of 10 |
| `reconnect_mcp_server` | control request | — | **refused** for SDK servers ("SDK servers should be handled in print.ts") |

## The system prompt: stable part is switched, variable part travels

A caller's prompt usually changes every turn (retrieved memory, tool results, notices), and what a
CLI holds as its system prompt should not. Callers that cache prompts already mark the boundary: a
`SystemMessage` whose content is a list of blocks, the stable blocks carrying `cache_control`. The
chat model uses that boundary:

- blocks up to and including the **last** one carrying `cache_control` → the call's stable system
  prompt, which the pool switches to on the CLI as a named agent (below). It is not part of the
  session key;
- the blocks after it → prepended to the query, ahead of the conversation.

A bare-string system message has no marker, so all of it is stable. A CLI launches with no system
prompt and every stable prompt its key has seen defined as an agent, named from the prompt's
digest; each checkout switches to the call's prompt (see the table above and "Session key").

This also fixes a live defect. The base class did `str(msg.content)` on that list, so on a
subscription route the CLI received the Python **repr** of the blocks — literal `\n`
sequences and `cache_control` dicts inline — as the agent's persona.

Trade-off, stated: the variable part moves from the system role into the first user
message. It is the dynamic context the caller already separated from the persona, and on this
route the alternative is either no pooling or a repr.

## Tool calls are handed back, one model turn per call

A pooled session never runs a caller's tool. The subscription model forces `--max-turns 1`: the
model's tool uses -- parallel ones included -- are emitted, the CLI stops with `error_max_turns`
before a second model turn, and the tool calls come back as `AIMessage.tool_calls` for the caller's
graph to run (approval, shaping, its ledger). The handler the CLI calls in between answers with a
placeholder no turn reads. Verified live on the bundled CLI (2026-09-15): two parallel calls
emitted, both handlers called, `error_max_turns` after turn 2 of the CLI's count, and the same
session then answered an ordinary query with `success`.

One exception: a call that asks for a schema and gives the model no tool but the CLI's
own `StructuredOutput` launches with `--max-turns 6` and `MAX_STRUCTURED_OUTPUT_RETRIES=5`. The CLI
checks each `StructuredOutput` call against the schema and wants a turn of its own to retry a
mismatch; at one turn about a third of real structured calls ended on `error_max_turns` with no
answer. Such a call has no caller tool a second turn could run. So `max_turns` takes two values, and
both it and the environment are part of the key: a structured call with no tools never shares a
session with any other kind.

## Session key

A session is reusable for any call whose **launch-time** options match:

- a digest of the credential (the token never appears in a key or a path);
- `tools` (built-ins), `disallowed_tools`, `permission_mode`, `max_budget_usd`,
  `fallback_model`, and any caller-set `cwd`;
- `max_turns` and `env`, which differ for a structured call with no tools (above);
- every other launch-time field of the options, so nothing a caller sets can share a CLI
  launched without it.

Not in the key, because they are applied per checkout: `model` (`set_model`), the bound
tool server (`mcp_set_servers`), and a text system prompt (an agent switch). A CLI launches with
no system prompt and every prompt its key has seen defined as an agent named from the prompt's
digest; a call whose prompt a CLI lacks takes an idle CLI that has it, or starts one that defines
it and every earlier prompt (at most 32 per key, least recently used dropped), and at a cap an idle
CLI that lacks it makes room. `per_key` now defaults to the whole cap, since one key holds every
prompt. Measured on seven stage prompts, three stages at once, four sessions: 28 CLI starts and 24
evictions in 28 calls with the prompt in the key; 7 starts and 3 evictions with the switch, every
stage answering from its own prompt. Tool auto-approval is granted server-wide at launch
(`mcp__langchain-tools`), so a swapped-in tool set needs no relaunch.

A call carrying `resume` / `continue_conversation` wants the CLI's own stored session, which
isolation disables. It is not pooled: it gets a one-off client, as before.

## Isolation — every session, pooled or not

A CLI launched with no configuration of its own reads the host's. Measured:

| Launch | plugins | plugin commands | hook events | account connectors | no token |
|---|---|---|---|---|---|
| as shipped | 1 | 14 | 6 | Slack, Calendar, Drive, Gmail, ZoomInfo | uses the host's login |
| isolated | 0 | 0 | 0 | none | "Not logged in" |

`claude_cli_isolation(token)` supplies an empty `CLAUDE_CONFIG_DIR` per credential, an empty
`cwd`, `ENABLE_CLAUDEAI_MCP_SERVERS=false`, `--strict-mcp-config` and
`--no-session-persistence` (without which the CLI writes a transcript of every exchange to
disk).

## Lifecycle

- **Checkout** is exclusive. A call is complete only when its `ResultMessage` has been read.
- **Return:** the tool server is released, and the conversation is rewound to the call's first
  message: the call's messages go out through `LentClient`, which gives each a uuid, and the
  `rewind_conversation` control request cuts the conversation at that uuid. Then back to the idle
  set. The CLI grants a rewind to the FIRST message only while its server-side flag
  `tengu_rewind_first_message` is on (measured on: 2026-09-27, bundled CLI 2.1.207, subscription
  credential, isolated configuration); otherwise it answers `rewound: false`, "no preceding
  assistant". A refused, failed or timed-out rewind stops the session and starts a **spare** --
  a fresh CLI with the same launch options, started in the background and parked idle, one per key
  -- so the next call does not pay the start.
- **Never `/clear`.** It empties the conversation but leaves itself in it: a
  `<local-command-caveat>`, `<command-name>/clear</command-name>` and an empty
  `<local-command-stdout>`, which the next caller's model read as the person's latest input and
  answered (found live in 0.56.0; the reused session quoted those lines back on 10 calls of 10).
  Nothing else resets a live CLI without a visible turn: a new input `session_id` keeps the
  conversation (measured), and `end_session` ends the process.
- **Dispose, never re-pool,** on any error, timeout or cancellation mid-call: an abandoned
  stream is the one way a later borrower could read an earlier caller's answer.
- **Exhaustion:** past the per-key or process-wide cap, a call waits briefly for a session,
  then runs on a one-off isolated client. It is never refused, and it never moves to a
  different credential.
- **Orphans:** each CLI carries a marker in its environment naming the owning process (pid
  and start time, so a recycled pid cannot claim it). A startup sweep kills marked CLIs whose
  owner is gone. Disposal kills the process tree via `/proc`, not the process group: the SDK
  starts the CLI in the caller's own group, so `killpg` would signal the host process.
- **Idle TTL** plus a reaper task, started detached from any request's context.
- **At the process cap**, the longest-idle session under *another* key gives up its slot
  rather than the call running on its own CLI. Found live: four keys each left one idle
  session in a four-slot pool, and a fifth key's call was told "every session is busy" while
  none was. Busy sessions are never evicted.
- **Shutdown:** the host calls `close_claude_cli_pool()`.

The host owns the numbers (caps, TTL, timeouts) and the lifecycle hooks, because only the host
knows how many worker processes wrap the library. The package owns the mechanism.

## Verified live (dev container, a real subscription token, 2026-09-15)

| Check | Result |
|---|---|
| routing decision latency | 4–19 s before → 1.07–1.32 s (2.5 s on a cold key) |
| CLI starts | one per model call before → 5 across 12 turns, none after warm-up except on a real key change |
| isolation on every live CLI | `--strict-mcp-config`, `--no-session-persistence`, own `CLAUDE_CONFIG_DIR`, own cwd, `ENABLE_CLAUDEAI_MCP_SERVERS=false`; 0 transcripts on disk; no host configuration in any reply |
| tool swap on a pooled session | the agent called the date and fetch tools on reused sessions |
| Stop mid-answer | the session was disposed six seconds later, not re-pooled; the next message was served |
| clean shutdown | pool closed, 4 CLIs stopped, 0 left |
| `kill -9` of the owner | the CLI exited on its own (stdin EOF); the startup sweep is the backstop for one that hangs instead |
| distinct keys per conversation | ~4 at steady state: the conversation, the router, post-turn jobs, occasional tool-argument filling; a self-edited persona is a new key, correctly |

**Prompt caching is not regressed.** The same turns through this model, pooled and not:
same tool set, turns 2–4 read 12,251 cached tokens either way; with the tool set changing per
turn, the unpooled model read 0 on turn 2 and the pooled one 12,182. Caching is lost when the
*tool set* changes -- tool definitions lead the cached prefix -- not because of the pool. The
conversation history is never cross-turn cached on this route pooled or not, because the base
package flattens it into one user message.

Size: each CLI is ~260 MB resident (measured on four live ones), so the cap is a memory
budget, not just a concurrency limit.
