# agent-wake-03: Dispatch handler + `WakeTrigger` abstraction

> **REMOVED 2026-05-24:** the outbound delivery framework (delivery_target / delivery_config / DeliveryAdapter / email-SMTP adapter / delivery_status / delivery routing) was removed as an undesigned parallel abstraction. `dispatch_wake` no longer takes a `delivery_adapters` argument, there is no `DeliveryAdapter` Protocol, and there is no delivery-routing stage. Wake fires now ALWAYS deliver into the conversation via the handler callback; outbound delivery, if ever needed, will be a threetears.channels adapter. Inbound webhooks are unaffected. The text below is retained for history; the delivery-routing requirements (DISPATCH-03, the `delivery_adapters` arg of DISPATCH-04, the `delivery_status` field of DISPATCH-05, DISPATCH-16) are struck — do NOT rebuild them.

## 2026-05-19 revision deltas (apply BEFORE implementing)

Canonical source: `<metallm>/docs/long_running/PLACEMENT.md`.

**MAJOR scope reduction.** Pre-check executor framework + `no_agent` branch + multi-skill loading are all dropped.

**DROPPED:**
- Pre-check executor framework (`execute_http_get`, `execute_loki_query`, `execute_postgres_query`). Pre-checks are ordinary `TearsTool` subclasses surfaced via the attached skill's `tool_additions`. PLACEMENT §1.2.
- `no_agent` mode branch — `dispatch_wake` always invokes the handler callback. Skip-LLM via skill instructing `[SILENT]`. PLACEMENT §1.6.
- `prepared_context.pre_check_output` field — no pre-check stage.
- Multi-skill loading. Single skill resolution.

**ADDED:**
- `prepared_context.attached_skill: AgentSkillEntity | None` — SINGLE skill resolution from `trigger.skill_id` (and `webhook_subscription.default_skill_id` for webhook fires). PLACEMENT §1.1 / §1.3.

**WakeTrigger schema:** `skill_id: UUID | None` (not `skill_ids: tuple`).
**PreparedWakeContext schema:** `attached_skill: AgentSkillEntity | None`. No `pre_check_output`. Keep `context_blocks` for `context_from` chain.

**dispatch_wake flow (revised):**
1. Rate-limit check.
2. Conversation lookup.
3. Per-conv lock acquire (inline only).
4. `context_from` resolution → `prepared_context.context_blocks`.
5. Skill resolution → `prepared_context.attached_skill` (or None).
6. Invoke `handler_callback(trigger, prepared_context)` once. **Between tool-loop iterations inside the handler, the platform runs the pending-user-message detection query** (new per wake-yield, see below); populates `_user_message_pending` state.
7. Detect `[SILENT]` on result; set `wake_fires.display_suppressed`.
8. **Detect yield** (new per wake-yield): if handler returned `status='yielded'`, set `wake_fires.status='yielded'` and emit `3tears.agent_wake.fire.yielded` Loki event. Lock releases as normal.
9. ~~Delivery routing (`conversation` default — already placed by handler — or `email`).~~ ~~[REMOVED 2026-05-24]~~ — no delivery routing; handler always places the message in the conversation.
10. UPDATE `wake_fires` with final status + `actual_fired_at`.

**Wake-yield additions (cross-shard cooperation per `metallm/docs/long_running/shard-10-cooperative-yield.md`):**
- Between tool-loop iterations, run pending-user-message detection — `SELECT EXISTS (SELECT 1 FROM messages WHERE conversation_id = $1 AND source = 'user' AND date_created > $2 AND NOT EXISTS (... no assistant response after this user message ...))` with `$2 = wake_started_at`. Populates `_user_message_pending: bool` on the LangGraph state. Cost target: <5ms p99.
- When `_user_message_pending=True`, append a one-line hint to the per-iteration system prompt: *"A user message is waiting. If you can wrap up gracefully (call `wake_yield`), do so. Otherwise continue and they'll see your full response."*
- After each iteration, check `_yield_requested`. If True, exit the loop cleanly; return `HandlerCallbackResult(status='yielded', ...)`.
- The `WakeYieldTool` lives in shard-04. Gated to load only on wake-driven turns.
- `wake_fires.status` enum gains `'yielded'` value (schema delta in shard-01's 2026-05-19 revision deltas).

## Objective

Land the convergence point. Define the typed `WakeTrigger` /
`WakeDispatchResult` shapes, the `HandlerCallback` protocol, and the
`dispatch_wake(trigger, fire_id, pool, handler)` function that
implements:

- rate-limit check (delegates to shard 05's framework)
- conversation lookup (via `3tears-conversations`)
- per-conv lock check (`nats_distributed_lock` from shard 02; key = `conv.<id>.llm_active`)
- pre-check execution (this shard's executors)
- gate semantics (empty / `[SKIP]` → `skipped_gate`)
- `no_agent` direct-injection mode
- `context_from` resolution
- attached-skill loading
- invoking the product-supplied `HandlerCallback`
- `[SILENT]` suppression
- ~~delivery routing (`conversation` and/or `email`)~~ ~~[REMOVED 2026-05-24]~~
- final `wake_fires` row UPDATE

The handler is platform; everything inside is generic. The product
plugs in:

- The `HandlerCallback` (build prompt, inject message, capture response).
- ~~The `DeliveryAdapter` for non-default `delivery_target` values (e.g. product-specific SMTP wrapper for `email`).~~ ~~[REMOVED 2026-05-24]~~
- The `EncryptionService` for any decrypt operations (consumed but not
  owned by the wake package).

---

## Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| DISPATCH-01 | Define `WakeTrigger` and `WakeDispatchResult` as `@dataclass(frozen=True, kw_only=True)` in `packages/agent/wake/src/threetears/agent/wake/types.py`. | P0 |
| DISPATCH-02 | Define the `HandlerCallback` Protocol in `types.py`: `async def __call__(self, trigger: WakeTrigger, prepared_context: PreparedWakeContext, pool: Pool) -> HandlerCallbackResult`. | P0 |
| ~~DISPATCH-03~~ | ~~[REMOVED 2026-05-24]~~ ~~Define the `DeliveryAdapter` Protocol in `types.py`: `async def deliver(self, trigger, message_content, pool) -> DeliveryStatus`. Built-in `'conversation'` adapter is a no-op (message already in the conv). Product supplies `'email'` adapter.~~ Do NOT define a `DeliveryAdapter` Protocol. | ~~P0~~ |
| DISPATCH-04 | Implement `dispatch_wake(trigger, fire_id, pool, *, handler: HandlerCallback, wake_config: WakeConfig)` in `dispatch.py` (~~the `delivery_adapters: dict[str, DeliveryAdapter]` parameter is REMOVED 2026-05-24~~). The caller (tick or webhook receiver) has already INSERTed the initial `wake_fires` row with `status='dispatching'`; `dispatch_wake` UPDATES it to the final status before returning. | P0 |
| DISPATCH-05 | `dispatch_wake` enforces the same flow on EVERY exit path: UPDATE `wake_fires` with `status`, `error`, `duration_ms`, `pre_check_output`, ~~`delivery_status` (REMOVED 2026-05-24)~~, `display_suppressed` before returning. The caller relies on this contract — they don't re-UPDATE the row. | P0 |
| DISPATCH-06 | Pre-check execution: when `trigger.pre_check_type` is set, look up the registered executor and run it with bounded timeout (per-type from `wake_pre_check_types.config_jsonschema.timeout_seconds`, default 10s, max 30s). Cap output at 64KB with `[truncated: <total>B → 64KB]` suffix. | P0 |
| DISPATCH-07 | Gate semantics: pre-check output that is (a) empty / whitespace-only OR (b) starts with `[SKIP]` short-circuits — return `WakeDispatchResult(status='skipped_gate')` without invoking the handler, without inserting a wake message into the conversation. Schedule keeps polling (tick advances `next_fire_at` per cadence). | P0 |
| DISPATCH-08 | `no_agent` mode: when `trigger.no_agent=true` AND pre-check produced output, route the pre-check output directly to the delivery adapter chain; LLM never invoked. Handler callback NOT called in this path. | P0 |
| DISPATCH-09 | `context_from` resolution: when `trigger.context_from_schedule_id` is set, fetch the most-recent successful `wake_fires` row for that schedule, build a labeled context block (`Context from upstream schedule "{name}" (fired {when}):\n{content}\n---`), cap at 16KB combined with pre-check, and include in `PreparedWakeContext.context_blocks`. Missing upstream row = log warning, no block. Single-hop only (not transitive). | P0 |
| DISPATCH-10 | Skill ID resolution boundary: `dispatch_wake` is the ONLY code path that resolves `WakeTrigger.attached_skill_ids: tuple[UUID, ...]` into `AgentSkill` rows. Filters out `enabled=false` (logs warning per dropped attachment). Passes the resolved tuple into `PreparedWakeContext.attached_skills`. | P0 |
| DISPATCH-11 | Pre-check executor registry: three v1 executors registered in `dispatch.py` — `execute_http_get(config, wake_config, pool)`, `execute_loki_query(config, wake_config, pool)`, `execute_postgres_query(config, wake_config, pool)`. Each returns a `PreCheckResult(output: str, duration_ms: int, error: str | None)` dataclass. | P0 |
| DISPATCH-12 | HTTP GET executor enforces URL host allow-list from `wake_config.http_allowed_hosts: tuple[str, ...]` BEFORE the request — no SSRF. Allow-list is product-supplied via `WakeConfig` (shard 05). Empty allow-list = ALL hosts rejected (fail-closed). | P0 |
| DISPATCH-13 | Loki executor enforces "named query" — looks up `config['named_query']` in `wake_config.loki_named_queries: dict[str, str]`, renders with positional params, calls the Loki client (consumer-supplied via `WakeConfig.loki_client`). Inline LogQL rejected. Admin-only enforced at the agent-tool/REST layer (this shard trusts `WakeConfig` has already gated). | P0 |
| DISPATCH-14 | Postgres executor enforces "named query" — looks up `config['named_query']` in `wake_config.postgres_named_queries: dict[str, str]`, renders with positional params, executes against `pool` with `set local statement_timeout = '5s'`. Returns TSV-formatted rows. Inline SQL rejected. | P0 |
| DISPATCH-15 | `[SILENT]` suppression: when `HandlerCallbackResult.assistant_message_content` starts with `[SILENT]` (case-insensitive, optional trailing space/newline), set `wake_fires.display_suppressed=true`. The product's callback is responsible for stripping the prefix from the stored content and setting whatever its messages-table column needs to render as hidden (e.g. metallm's `messages.display='hidden'`); platform just records the flag. | P0 |
| ~~DISPATCH-16~~ | ~~[REMOVED 2026-05-24]~~ ~~Delivery routing: after the handler returns, inspect `trigger.delivery_target`. For `'conversation'` (default), the handler-callback has already placed the message in the conversation — no-op. For other values, invoke the matching adapter from `delivery_adapters: dict[str, DeliveryAdapter]`. Record outcome in `wake_fires.delivery_status`. Delivery failure does NOT block the response from landing in the conversation.~~ There is no delivery-routing stage; the handler callback always places the message in the conversation. Do NOT rebuild this. | ~~P0~~ |
| DISPATCH-17 | Pre-check failure handling: when an executor raises or times out, record error in `wake_fires.error` AND in `wake_fires.pre_check_output`. Default: STILL invoke the handler with a `(pre-check failed: <error>)` note in the context blocks. Override: per-schedule `pre_check_config.skip_llm_on_failure: true` makes pre-check failure short-circuit like a gate-skip. | P0 |

---

## Design Context

### Why `HandlerCallback` is a Protocol, not a function param

The wake feature's product surface is "I'll handle a wake event."
metallm's handler is `personality_node_wake_handler` (builds system
prompt, calls `inject_conversation_event`). A future aibot consumer's
handler might be entirely different (no LangGraph, different prompt
assembly). Same shape, different body.

Protocol shape:

```python
class HandlerCallback(Protocol):
    async def __call__(
        self,
        trigger: WakeTrigger,
        prepared_context: PreparedWakeContext,
        pool: Pool,
    ) -> HandlerCallbackResult: ...
```

`PreparedWakeContext` is the platform's job: pre-check output (when
non-gating), `context_from` block, attached `AgentSkill` rows (resolved
+ filtered), `fired_at`, `schedule_name`, etc. The handler reads
fields off the prepared context and assembles its product-specific
prompt + invocation.

`HandlerCallbackResult` carries back: `assistant_message_id: UUID` (the
message the handler wrote), `assistant_message_content: str` (for
`[SILENT]` detection + delivery rendering), `target_conversation_id:
UUID` (inline = same as parent; spawn = the new conv).

### Why the fire-row protocol is split between caller + dispatch_wake

The tick (shard 02) and the webhook receiver (shard 06) both INSERT
the initial `wake_fires` row with `status='dispatching'` BEFORE
invoking `dispatch_wake`. Rationale:

1. The INSERT is the "claim happened" anchor. If the dispatch crashes
   mid-flight, the row survives — the next investigation can see "we
   started, didn't finish."
2. The fire_id is passed in. `dispatch_wake` UPDATES the existing row
   to its final state; never INSERTs.
3. The caller's outer try/except is the safety net: if `dispatch_wake`
   raises an unhandled exception (handler bug crashed before UPDATE),
   the caller UPDATES the row to `status='failed'`.

This means `dispatch_wake` always finds the row it's updating; it
never has to wonder whether the row exists.

---

## Types specification

```python
# packages/agent/wake/src/threetears/agent/wake/types.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from asyncpg import Pool


@dataclass(frozen=True, kw_only=True)
class AgentSkill:
    """Resolved skill row — defined by the parallel skills redesign.

    This is the placeholder shape; the skills sibling owns the canonical
    definition. The wake package imports it from wherever skills lands.
    """
    skill_id: UUID
    name: str
    body: str
    enabled: bool


@dataclass(frozen=True, kw_only=True)
class WakeTrigger:
    schedule_id: UUID | None         # None for webhook-source dispatches
    user_id: UUID
    conversation_id: UUID
    fire_source: str                  # 'scheduled_tick' | 'manual_test' | 'external_event.webhook' | future vendor-suffixed values
    execution_mode: str               # 'inline' | 'spawn'
    schedule_type: str                # 'external_event' for webhook sources
    task_prompt: str | None
    schedule_name: str | None
    fired_at: datetime                # tick wall-clock at the moment the schedule was observed due
    no_agent: bool = False
    pre_check_type: str | None = None
    pre_check_config: dict[str, Any] = field(default_factory=dict)
    context_from_schedule_id: UUID | None = None
    # delivery_target / delivery_config REMOVED 2026-05-24 — no outbound delivery framework
    attached_skill_ids: tuple[UUID, ...] = ()


@dataclass(frozen=True, kw_only=True)
class PreparedWakeContext:
    """Built by dispatch_wake before invoking the HandlerCallback.

    The handler reads off this — it never re-fetches the schedule row,
    re-resolves skills, re-runs the pre-check, etc.
    """
    trigger: WakeTrigger
    pre_check_output: str | None     # None if no pre-check ran or pre-check gated
    context_blocks: tuple[str, ...]  # context_from block + (pre-check failed: ...) note when applicable
    attached_skills: tuple[AgentSkill, ...]


@dataclass(frozen=True, kw_only=True)
class HandlerCallbackResult:
    assistant_message_id: UUID
    assistant_message_content: str
    target_conversation_id: UUID     # spawn = new; inline = trigger.conversation_id


@dataclass(frozen=True, kw_only=True)
class WakeDispatchResult:
    status: str                       # 'fired' | 'skipped_busy' | 'skipped_gate' | 'failed' | 'rate_limited'
    target_conversation_id: UUID | None
    error: str | None = None
    should_expire_schedule: bool = False
    pre_check_output: str | None = None
    pre_check_duration_ms: int | None = None
    # delivery_status REMOVED 2026-05-24 — no outbound delivery framework
    display_suppressed: bool = False


@runtime_checkable
class HandlerCallback(Protocol):
    async def __call__(
        self,
        trigger: WakeTrigger,
        prepared_context: PreparedWakeContext,
        pool: Pool,
    ) -> HandlerCallbackResult:
        ...


# DeliveryAdapter Protocol REMOVED 2026-05-24 — the outbound delivery framework was
# removed as an undesigned parallel abstraction. Do NOT define this Protocol. The block
# below is retained for history only.
#
# @runtime_checkable
# class DeliveryAdapter(Protocol):
#     """Product-supplied adapter for non-'conversation' delivery_target values."""
#     async def deliver(
#         self,
#         trigger: WakeTrigger,
#         message_content: str,
#         pool: Pool,
#     ) -> str:
#         """Return one of: 'delivered' | 'delivered_*_failed' (target-specific)."""
#         ...


@dataclass(frozen=True, kw_only=True)
class PreCheckResult:
    output: str
    duration_ms: int
    error: str | None = None


@runtime_checkable
class PreCheckExecutor(Protocol):
    async def __call__(
        self,
        config: dict[str, Any],
        wake_config: "WakeConfig",
        pool: Pool,
    ) -> PreCheckResult:
        ...


# WakeConfig protocol lives in shard 05 (with the rate-limit framework)
```

---

## Dispatch flow

`dispatch_wake(trigger, fire_id, pool, *, handler, wake_config)` (~~`delivery_adapters` REMOVED 2026-05-24~~) — every step can short-circuit with a typed `WakeDispatchResult`; in EVERY exit path the function UPDATES the `wake_fires` row before returning:

```
1. Rate-limit check (shard 05's _check_rate_limit)
   ├── over cap → UPDATE wake_fires status='rate_limited'; return
   └── under cap → continue

2. Conversation lookup via 3tears-conversations
   ├── conv missing → UPDATE status='failed', should_expire_schedule=true; return
   └── found → continue

3. Per-conv lock check (inline mode only): nats_distributed_lock(nats, f"conv.{conv_id}.llm_active") with non-blocking acquire — if held, treat as "conv busy"
   ├── held → UPDATE status='skipped_busy'; return
   └── acquired → continue (release at end of dispatch)

4. Pre-check (if trigger.pre_check_type set)
   ├── execute_<type>(config, wake_config, pool)
   ├── output empty / starts with '[SKIP]' → UPDATE status='skipped_gate', pre_check_output=output; return
   ├── [REMOVED 2026-05-24] output non-empty + no_agent=true → route DIRECTLY to delivery_adapters — no longer exists (no delivery framework, no no_agent mode)
   ├── execute raises / times out + skip_llm_on_failure=true → UPDATE status='skipped_gate', error=<exc>; return
   └── execute raises / times out + skip_llm_on_failure=false → record error; continue (handler gets a "(pre-check failed)" context note)

5. Resolve context_from chain (if context_from_schedule_id set)
   ├── fetch most-recent successful wake_fires row for that schedule
   ├── build context block; truncate combined-with-pre-check at 16KB
   └── append to context_blocks

6. Resolve attached_skill_ids → AgentSkill rows; filter enabled=true
   └── log warning per dropped attachment

7. Build PreparedWakeContext(trigger, pre_check_output, context_blocks, attached_skills)

8. Invoke handler(trigger, prepared_context, pool)
   ├── handler raises → UPDATE status='failed', error=<exc>; return
   └── HandlerCallbackResult returned → continue

9. Detect [SILENT] prefix on handler_result.assistant_message_content
   └── set display_suppressed=true if matched

10. [REMOVED 2026-05-24] Delivery routing — no longer exists. The handler callback
    always places the message in the conversation; there is no delivery_target branch,
    no delivery_adapters dict, no delivery_status. Do NOT rebuild this step.

11. UPDATE wake_fires with final status='fired', target_conversation_id, pre_check_output, pre_check_duration_ms, display_suppressed, duration_ms; return WakeDispatchResult
    # (delivery_status REMOVED 2026-05-24)
```

The handler is the ONLY code path that runs steps 1-11. Tick (shard
02) and webhook receiver (shard 06) both follow the caller protocol
(INSERT initial row → invoke `dispatch_wake` → safety-net UPDATE on
unhandled exception). They don't reimplement any step above.

---

## Pre-check executor specifications

### `execute_http_get(config, wake_config, pool) -> PreCheckResult`

```
1. Validate URL host is in wake_config.http_allowed_hosts (exact-match; no glob v1)
   └── not allowed → PreCheckResult(output='', error='host not in allow-list: <host>')
2. asyncio.wait_for(aiohttp_session.get(url, headers=config.get('headers',{}),
                                       timeout=config.get('timeout_seconds', 10)),
                    timeout=config.get('timeout_seconds', 10) + 1)
3. status not 2xx → PreCheckResult(output='', error=f'http {status}')
4. read up to config.get('max_response_bytes', 65536) bytes
5. decode utf-8; truncate at 64KB with [truncated] suffix
6. return PreCheckResult(output=body, duration_ms=<elapsed>)
```

### `execute_loki_query(config, wake_config, pool) -> PreCheckResult`

```
1. named = config['named_query']
2. template = wake_config.loki_named_queries.get(named)
   └── missing → PreCheckResult(output='', error=f'named_query not found: {named}')
3. Render template with config.get('params', {}) — positional params, format-string-safe
4. Call wake_config.loki_client.query_range(rendered_query, range_minutes, limit)
5. Format matching lines newline-joined (or empty if zero matches)
6. Truncate at 64KB
7. return PreCheckResult(output=text, duration_ms=<elapsed>)
```

### `execute_postgres_query(config, wake_config, pool) -> PreCheckResult`

```
1. named = config['named_query']
2. template = wake_config.postgres_named_queries.get(named)
   └── missing → PreCheckResult(output='', error=f'named_query not found: {named}')
3. async with pool.acquire() as conn:
       await conn.execute("SET LOCAL statement_timeout = '5s'")
       rows = await conn.fetch(template, *config.get('params', []))
4. Format rows as TSV
5. Truncate at 64KB
6. return PreCheckResult(output=tsv, duration_ms=<elapsed>)
```

---

## Patterns to Follow

- Frozen dataclass shape: `3tears-conversations` `Conversation` entity for the immutable-after-final pattern.
- Protocol shape: `3tears-models` `BaseChatModel` pattern (LangChain-native Protocols).
- Per-conv lock: reuse `nats_distributed_lock` from shard 02 with key `conv.<conversation_id>.llm_active`. metallm's existing `BUCKET_LOCKS` bucket has this exact key shape; default `bucket_name='locks'` parameter handles it.
- `asyncio.wait_for` bounded execution: existing 3tears patterns in `threetears.models` for streaming utilities.
- Logging: `threetears.observe.get_logger(__name__)`, structured via `extra={"extra_data": {...}}`.

---

## Files to Create

- `packages/agent/wake/src/threetears/agent/wake/types.py` — All the dataclasses + Protocols above.
- `packages/agent/wake/src/threetears/agent/wake/dispatch.py` — `dispatch_wake(...)` + the pre-check executor registry.
- `packages/agent/wake/src/threetears/agent/wake/pre_check.py` — `execute_http_get`, `execute_loki_query`, `execute_postgres_query`.
- `packages/agent/wake/src/threetears/agent/wake/context.py` — `_resolve_context_from`, `_load_attached_skills`, `_build_prepared_context`.
- `packages/agent/wake/tests/unit/test_dispatch_flow.py` — table-driven tests covering every exit path with stubbed handler.
- `packages/agent/wake/tests/unit/test_pre_check_executors.py` — host allow-list rejection; named-query lookup; timeout handling; output truncation.
- `packages/agent/wake/tests/unit/test_silent_suppression.py` — `[SILENT]` prefix detection (case-insensitive, optional trailing).
- `packages/agent/wake/tests/integration/test_dispatch_e2e.py` — full dispatch flow against a real Postgres + stubbed handler that returns a known message; assert `wake_fires` row landed correctly.

---

## Implementation Notes

1. **`dispatch_wake` is the single update site for `wake_fires`.** No other code path UPDATEs the row (except the caller's outer safety-net try/except for unhandled exceptions). Enforce via a code-review rule + a drift-guard test (shard 05 includes this).

2. **`[SILENT]` prefix matching.** Compile once at module load:

   ```python
   import re
   _SILENT_PREFIX = re.compile(r"^\s*\[SILENT\]\s*", re.IGNORECASE)
   def is_silent(content: str) -> bool:
       return bool(_SILENT_PREFIX.match(content))
   ```

   The platform records the flag; the product callback is responsible for stripping the prefix from the stored message content + setting whatever message-table column it has for "hidden."

3. **`context_from` 16KB combined cap.** Pre-check output + context_from block share a 16KB budget. If pre-check is already at the budget, context_from is dropped with a logged warning. If both fit, both are included.

4. **`context_from` recursion = single-hop.** If schedule A's `context_from = B` and B's `context_from = C`, A gets B's output ONLY, not C's. Documented; not enforced at the DB layer.

5. **Per-conv lock key naming.** `conv.<conversation_id>.llm_active` — same key shape metallm's `inject_conversation_event` uses today. The wake handler acquires it BEFORE calling the handler callback; releases on `dispatch_wake` exit. Mutual exclusion with user-triggered turns.

   **Note:** the handler callback (e.g. metallm's) internally fires `asyncio.create_task` for the LLM call. The platform doesn't hold the per-conv lock for the LLM lifetime — it holds it for the INSERT-the-message phase, after which the LLM runs in a background task. This matches the existing `inject_conversation_event` semantics. The handler callback is responsible for any extended-lock semantics it needs.

6. **`PreparedWakeContext` is read-only.** Frozen dataclass. The handler does not mutate it; it reads the fields and assembles its prompt.

7. ~~**Delivery adapter registry.** `delivery_adapters` is a dict passed in by the consumer at `dispatch_wake` invocation time. Default contains only `{}` — `conversation` is not in the dict because it's a no-op. metallm injects `{'email': MetallmEmailDeliveryAdapter()}` at the wiring site.~~ ~~[REMOVED 2026-05-24]~~ — no outbound delivery framework; there is no `delivery_adapters` argument. The handler callback always places the message in the conversation.

8. **`wake_config` is the read-side of the platform's `WakeConfig` protocol (shard 05).** Carries: `http_allowed_hosts`, `loki_client`, `loki_named_queries`, `postgres_named_queries`, `max_fires_per_conv_per_day`, `max_fires_per_user_per_day`. The consumer (metallm) supplies an implementation backed by their `system_settings` table; tests supply an in-memory implementation.

9. **Error categorization.** `wake_fires.error` is human-readable. `wake_fires.status` is the machine-readable outcome class. Both populated on failure.

10. **Skill resolution.** Calls the skills package's `load_skills_by_ids(pool, skill_ids: tuple[UUID, ...]) -> tuple[AgentSkill, ...]` (defined by the parallel skills redesign, filtered `enabled=true`). If a skill_id doesn't exist (deleted), it's silently dropped with a logged warning. If a skill_id exists but `enabled=false`, dropped with a logged warning. The handler receives only enabled, existing skills.

11. **`asyncio.wait_for` for each pre-check executor.** Bounded execution. Timeout from `config.get('timeout_seconds', 10)`. Hard cap at 30s regardless of config — enforced inside the executor.

12. **`execute_postgres_query` runs against the same pool.** `SET LOCAL statement_timeout = '5s'` ensures runaway queries don't tie up a connection. The pool is the platform's connection pool, not a separate one.

---

## Anti-patterns

- DO NOT add a parallel `dispatch_webhook` function. The webhook receiver constructs a `WakeTrigger` and calls `dispatch_wake`. One handler, all paths.
- DO NOT skip the `wake_fires` UPDATE on any exit path. Tests assert this; the drift guard in shard 05 codifies it.
- DO NOT load skills inside the handler callback. Platform owns the resolution boundary (DISPATCH-10).
- DO NOT couple `WakeTrigger.fire_source` to a Literal. Free-form string — vendor-specific suffixes (`'external_event.webhook.github'`) extend naturally.
- DO NOT make `context_from` chains transitive. Single-hop. Pre-checks are bounded; chains shouldn't blow up.
- DO NOT skip the URL host allow-list. SSRF vector.
- DO NOT add a "store raw LogQL or SQL from the agent" pre-check shape. v1 is named-query only; admin-curated.
- DO NOT bury pre-check failure as "the wake didn't fire." Record error in `wake_fires.error` AND `wake_fires.pre_check_output`; default behavior is "invoke handler anyway with the error noted."
- DO NOT make the handler callback do the rate-limit check. The platform does it before invoking the callback.
- DO NOT acquire the per-conv lock from inside the handler callback. The platform handles lock lifecycle.

---

## Success Criteria

- [ ] All types (`WakeTrigger`, `WakeDispatchResult`, `HandlerCallback`, ~~`DeliveryAdapter`~~ ~~[REMOVED 2026-05-24]~~, `PreparedWakeContext`, `HandlerCallbackResult`, `PreCheckResult`, `PreCheckExecutor`) exist as documented.
- [ ] `dispatch_wake` correctly routes every documented exit path.
- [ ] Pre-check executors: host allow-list enforced; named-query lookup; timeout; output truncation.
- [ ] `[SILENT]` detection: case-insensitive, optional trailing space/newline; flag set correctly.
- [ ] `context_from` resolved single-hop; missing upstream row logs warning, no block.
- [ ] Skill resolution: enabled=false and missing IDs dropped with warning; resolved skills delivered in `position` order.
- [ ] `no_agent` mode skips the handler when pre-check produced output; delivery adapter still invoked.
- [ ] ~~`delivery_target='email'` adapter invoked when supplied; outcome recorded in `delivery_status`.~~ ~~[REMOVED 2026-05-24]~~ — no delivery framework.
- [ ] Conv-busy path: `skipped_busy` returned without inserting handler-driven changes.
- [ ] Conv-missing path: `should_expire_schedule=True` returned; status='failed'.
- [ ] `./scripts/check-all.sh` clean.

---

## Verification

```bash
cd /Users/pace/crypt/pub/dev-wsl/vscode/3tears/3tears
uv run --directory packages/agent/wake pytest tests/ -v
./scripts/check-all.sh
```
