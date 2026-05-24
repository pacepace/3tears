# agent-wake-04: Agent tools (schedule + webhook subscription CRUD) + webhook→trigger adapter

> **REMOVED 2026-05-24:** the outbound delivery framework (delivery_target / delivery_config / DeliveryAdapter / user_email_verified) was removed as an undesigned parallel abstraction. The schedule/subscription tools no longer accept `delivery_target` / `delivery_config` inputs, there is no `user_email_verified` construction-context flag, and `webhook_receive(...)` no longer takes a `delivery_adapters` argument. Wake fires now always deliver into the conversation; outbound delivery, if ever needed, will be a threetears.channels adapter. Inbound webhooks are unaffected. The text below retains these for history — TOOL-14 is struck; do NOT rebuild it.

## 2026-05-19 revision deltas (apply BEFORE implementing)

Canonical source: `<metallm>/docs/long_running/PLACEMENT.md`.

**Tool count: 14** (2026-05-19 evening: was 13 after dropping skill-attach/detach; gained `WakeYieldTool` per wake-yield optimization). PLACEMENT §1.1 + §8.5.1.

**DROPPED:**
- `WakeSkillAttachTool` — skill via `WakeScheduleCreateTool(..., skill_id=picked)`.
- `WakeSkillDetachTool` — `skill_id=None` via Update.

**ADDED:**
- `WakeScheduleUpdateTool` (explicit; was implicit). Accepts `skill_id`, `name`, `status`.
- `WebhookSubscriptionUpdateTool` (explicit). Accepts `default_skill_id`, `name`, `status`.

**MODIFIED:**
- `WakeScheduleCreateTool` input schema gains `skill_id: UUID | None` (ACL-validated).
- `WebhookSubscriptionCreateTool` input schema gains `default_skill_id: UUID | None`.

**Final tool set (13):**
1. `WakeScheduleCreateTool` (with `skill_id`)
2. `WakeScheduleUpdateTool` (with `skill_id`)
3. `WakeScheduleListTool`
4. `WakeSchedulePauseTool`
5. `WakeScheduleResumeTool`
6. `WakeScheduleDeleteTool`
7. `WebhookSubscriptionCreateTool` (with `default_skill_id`)
8. `WebhookSubscriptionUpdateTool` (with `default_skill_id`)
9. `WebhookSubscriptionListTool`
10. `WebhookSubscriptionPauseTool`
11. `WebhookSubscriptionResumeTool`
12. `WebhookSubscriptionDeleteTool`
13. `WebhookSubscriptionRotateSecretTool`
14. **`WakeYieldTool`** (NEW per wake-yield, see `metallm/docs/long_running/shard-10-cooperative-yield.md` YIELD-03). Gated to load ONLY on wake-driven turns (closure check on `_active_wake_fire_id` state). No input arguments. Returns confirmation. Sets `_yield_requested=True`. **Net tool count = 14.**

**Drop from input schemas:** `pre_check_type`, `pre_check_config`, `no_agent` fields.

**ACL validation on `skill_id` / `default_skill_id`:** verify user has ACL grant on the referenced skill (or it's a registry-sourced tool-skill the user has ACL for). Reject with clear error if not.

**Webhook adapter:** receiver sets `trigger.skill_id = subscription.default_skill_id` when constructing the `WakeTrigger`.

## Objective

Land the agent-tool surface: seven `TearsTool` subclasses for managing
wake schedules + skill attachments, plus six more for managing webhook
subscriptions, plus the adapter glue that converts a verified webhook
payload (from shard 06's `WebhookReceiver`) into a `WakeTrigger` and
calls `dispatch_wake`.

Builds on `3tears-agent-tools`' `TearsTool` base + `MCPToolDefinition`.
Tool descriptions are terse (≤6 short lines) per the existing
`3tears-agent-tools` convention. Per-type config validators live here
(reused by the product's REST surface — they import from this package).

---

## Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| TOOL-01 | Seven schedule-management `TearsTool` subclasses in `packages/agent/wake/src/threetears/agent/wake/tools/`: `WakeScheduleCreateTool`, `WakeScheduleListTool`, `WakeSchedulePauseTool`, `WakeScheduleResumeTool`, `WakeScheduleDeleteTool`, `WakeSkillAttachTool`, `WakeSkillDetachTool`. Each subclasses `threetears.agent.tools.TearsTool`. | P0 |
| TOOL-02 | Six webhook-subscription `TearsTool` subclasses: `WebhookSubscriptionCreateTool`, `WebhookSubscriptionListTool`, `WebhookSubscriptionPauseTool`, `WebhookSubscriptionResumeTool`, `WebhookSubscriptionDeleteTool`, `WebhookSubscriptionRotateSecretTool`. | P0 |
| TOOL-03 | `WakeScheduleCreateTool` accepts typed shortcuts + cron escape-hatch (`daily_at`, `every_n_hours`, `random_within_window`, `one_shot_at`, `cron`, `relative_delay`). Validates per-type config app-side via `_validate_schedule_config(schedule_type, config)`. | P0 |
| TOOL-04 | `WakeScheduleCreateTool` computes `next_fire_at` at create-time using `_compute_next_fire_at` from shard 02. | P0 |
| TOOL-05 | Tools scope to `conversation_id` bound at construction time. The conversation_id is NOT a settable input on any tool — defense against cross-conv writes. | P0 |
| TOOL-06 | `WakeScheduleCreateTool` enforces per-conversation max active schedules (default 10). Reads cap from `WakeConfig.max_schedules_per_conversation`. Reject with clear `[TOOL ERROR]` when at cap. | P0 |
| TOOL-07 | Tool descriptions ≤6 short lines per existing `3tears-agent-tools` convention. Explicit about scope, schedule types, execution_mode, rendering. | P0 |
| TOOL-08 | `WakeScheduleDeleteTool` is a hard delete (FK cascade). `WakeSchedulePauseTool` sets `status='paused'` (reversible). | P0 |
| TOOL-09 | `WakeScheduleCreateTool` + `WebhookSubscriptionCreateTool` MUST NOT be loaded when the agent is running on a wake-triggered or webhook-triggered turn. The DETECTION mechanism for "this turn was wake-triggered" lives in the consuming product (different products have different message-source signals); the platform exposes the tools, the product decides when to include them in the loaded set. Documented as the consumer's responsibility. | P0 |
| TOOL-10 | Pre-check semantic validation: `http_get` URL host validated against `wake_config.http_allowed_hosts` (rejects otherwise). `loki_query` / `postgres_query` reject for non-admin users (admin status comes from the consumer's authorization context — the platform tool takes `is_admin: bool` as a construction-time parameter, not an input). | P0 |
| TOOL-11 | Per-type config validator function exports: `_validate_schedule_config`, `_validate_pre_check_config`, `_validate_context_from_chain` — importable by the product's REST router to reuse the validators. | P0 |
| TOOL-12 | Cycle detection on `context_from_schedule_id` at create/update time: BFS the chain from target until NULL, revisit `this_schedule_id` (reject as cycle), or max-depth 8 (reject as too-deep). Same-conversation enforced (cross-conv target rejected). | P0 |
| TOOL-13 | `attached_skill_ids` ownership check: every skill_id passed in must reference an `agent_skills` row owned by the same `user_id`. Rejection with clear per-ID error. | P0 |
| ~~TOOL-14~~ | ~~[REMOVED 2026-05-24]~~ ~~`delivery_target='email'` requires a `user_email_verified: bool` flag set on the tool's construction context (platform doesn't reach into a product's users table). Consumer supplies. Rejection with clear error.~~ No delivery framework; do NOT rebuild this. | ~~P0~~ |
| TOOL-15 | Webhook receiver adapter: `webhook_to_wake_trigger(subscription, payload, encryption_service, pool) -> WakeTrigger` builds the trigger from a verified subscription + decoded payload. Renders `task_prompt_template` via `jinja2.sandbox.SandboxedEnvironment` with `{event: <payload>}` as the only variable. | P0 |
| TOOL-16 | `webhook_receive(subscription_id, payload_bytes, signature_header, source_ip, pool, encryption_service, dispatch_callback, wake_config)` (~~the `delivery_adapters` arg is REMOVED 2026-05-24~~) — top-level entry point invoked by shard 06's `WebhookReceiver`. Looks up subscription, verifies HMAC, checks `allowed_source_pattern`, rate-limits, INSERTs the `wake_fires` row with `status='dispatching'`, calls `dispatch_wake(...)`. Returns a result object with HTTP status + fire_id. | P0 |

---

## Tool surfaces (12 tools)

### `WakeScheduleCreateTool`

Input schema (Pydantic):

```python
class WakeScheduleCreateInput(BaseModel):
    schedule_type: Literal["daily_at", "every_n_hours", "random_within_window", "one_shot_at", "cron", "relative_delay"]
    schedule_config: dict[str, Any]
    execution_mode: Literal["inline", "spawn"] = "inline"
    task_prompt: str | None = None
    name: str | None = None
    no_agent: bool = False
    pre_check_type: str | None = None
    pre_check_config: dict[str, Any] = Field(default_factory=dict)
    context_from_schedule_id: str | None = None  # short or full UUID
    # delivery_target / delivery_config REMOVED 2026-05-24 — no outbound delivery framework
    attached_skill_ids: list[str] = Field(default_factory=list)
```

Description (≤6 lines):

```
Schedule a wake in THIS conversation. You'll be woken and run through the same loop as for user messages.

- schedule_type + schedule_config — WHEN
- execution_mode — 'inline' (here) vs 'spawn' (new conv, EMPTY context, write task_prompt self-contained)
- pre_check_type + no_agent — cheap watchdogs without LLM cost
- context_from_schedule_id — chain off another wake's output
- attached_skill_ids — load skills before running

Returns [schedule:<id>]. Max 10 ACTIVE schedules per conversation (separate from the 24-fires-per-day rate cap). Tip: write task_prompt self-contained — future-you may not have the context that made this worth scheduling.
```

### `WakeScheduleListTool`

No input fields. Returns one line per schedule:

```
[schedule:<id>] · <name-or-untitled> · <schedule_type> · next: <next_fire_local> · mode: <execution_mode> · <status>
```

### `WakeSchedulePauseTool` / `WakeScheduleResumeTool` / `WakeScheduleDeleteTool`

Input:

```python
class WakeScheduleIdInput(BaseModel):
    schedule_id: str  # short or full UUID; resolved by _resolve_schedule_id
```

Descriptions:

```
# Pause
Pause a wake schedule. It stops firing until you wake_schedule_resume it. Status goes 'active' -> 'paused'.

# Resume
Resume a paused wake schedule. Recomputes next_fire_at from now using the schedule's type/config and goes 'paused' -> 'active'.

# Delete
Delete a wake schedule permanently. Fire history (wake_fires) cascades — gone too. Use pause instead if you might want it back.
```

### `WakeSkillAttachTool` / `WakeSkillDetachTool`

Input:

```python
class WakeSkillAttachInput(BaseModel):
    schedule_id: str
    skill_id: str
    position: int | None = None  # defaults to end

class WakeSkillDetachInput(BaseModel):
    schedule_id: str
    skill_id: str
```

### `WebhookSubscriptionCreateTool` (+ list / pause / resume / delete / rotate_secret)

Shape mirrors the schedule tools. The `Create` returns `[webhook:<id>] secret=<hex>` — the secret is shown ONCE (platform stores Fernet-encrypted ciphertext; consumer never gets the plaintext on subsequent GETs except via `RotateSecret`).

```python
class WebhookSubscriptionCreateInput(BaseModel):
    name: str | None = None
    task_prompt_template: str  # Jinja2 sandbox; {{event}} is the entire payload
    execution_mode: Literal["inline", "spawn"] = "inline"
    # delivery_target / delivery_config REMOVED 2026-05-24 — no outbound delivery framework
    attached_skill_ids: list[str] = Field(default_factory=list)
    allowed_source_pattern: str | None = None  # regex against source IP
```

---

## Webhook receiver adapter

```python
# packages/agent/wake/src/threetears/agent/wake/webhook_adapter.py
from __future__ import annotations

import hmac
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any
from uuid import UUID

from asyncpg import Pool
from jinja2.sandbox import SandboxedEnvironment
from uuid_utils import uuid7

from threetears.observe import get_logger

from threetears.agent.wake.collections import (
    WakeFireCollection,
    WebhookSubscriptionCollection,
)
from threetears.agent.wake.dispatch import dispatch_wake
from threetears.agent.wake.types import (
    # DeliveryAdapter REMOVED 2026-05-24 — no outbound delivery framework
    HandlerCallback,
    WakeTrigger,
)
# Imported from shard 05's wake_config module
from threetears.agent.wake.config import WakeConfig

log = get_logger(__name__)
_jinja_env = SandboxedEnvironment(autoescape=False)


class WebhookReceiveResult:
    status_code: int       # 202 / 401 / 403 / 404 / 429
    fire_id: UUID | None
    message: str           # human-readable for diagnostics


async def webhook_receive(
    *,
    subscription_id: UUID,
    payload_bytes: bytes,
    signature_header: str | None,
    source_ip: str | None,
    pool: Pool,
    encryption_service: Any,         # consumer-supplied EncryptionService Protocol
    handler: HandlerCallback,
    wake_config: WakeConfig,
    # delivery_adapters: dict[str, DeliveryAdapter] REMOVED 2026-05-24 — no outbound delivery framework
) -> WebhookReceiveResult:
    """Verify, rate-limit, and dispatch an inbound webhook.

    Returns a WebhookReceiveResult with HTTP status and the fire_id when accepted.
    """
    subs = WebhookSubscriptionCollection(pool)
    sub = await subs.get_active(subscription_id)
    if sub is None:
        return WebhookReceiveResult(404, None, "subscription not found or paused")

    # HMAC verification
    if not signature_header:
        return WebhookReceiveResult(401, None, "missing signature header")
    secret = sub.decrypt_secret(encryption_service)
    expected = "sha256=" + hmac.new(secret.encode(), payload_bytes, sha256).hexdigest()
    if not hmac.compare_digest(expected, signature_header):
        return WebhookReceiveResult(401, None, "invalid signature")

    # Source IP allow-list check
    if sub.allowed_source_pattern is not None and source_ip is not None:
        import re
        if not re.match(sub.allowed_source_pattern, source_ip):
            return WebhookReceiveResult(403, None, "source IP not allowed")

    # Per-subscription rate-limit
    fires = WakeFireCollection(pool)
    count = await fires.count_in_window_for_subscription(
        subscription_id=subscription_id,
        window_seconds=3600,
    )
    if count >= wake_config.max_webhook_fires_per_subscription_per_hour:
        return WebhookReceiveResult(429, None, "rate limit exceeded")

    # Render template
    try:
        payload = _decode_payload(payload_bytes)
        template = _jinja_env.from_string(sub.task_prompt_template or "")
        rendered = template.render(event=payload)
    except Exception as exc:
        log.warning("webhook template render failed", extra={"extra_data": {"error": str(exc)}})
        return WebhookReceiveResult(400, None, f"template render error: {exc}")

    # Construct trigger
    trigger = WakeTrigger(
        schedule_id=None,
        user_id=sub.user_id,
        conversation_id=sub.conversation_id,
        fire_source="external_event.webhook",
        execution_mode=sub.execution_mode,
        schedule_type="external_event",
        task_prompt=rendered,
        schedule_name=sub.name,
        fired_at=datetime.now(UTC),
        # delivery_target / delivery_config REMOVED 2026-05-24 — no outbound delivery framework
        attached_skill_ids=await subs.load_attached_skill_ids(subscription_id),
    )

    fire_id = uuid7()
    await fires.create_dispatching(
        fire_id=fire_id,
        schedule_id=None,
        webhook_subscription_id=subscription_id,
        conversation_id=sub.conversation_id,
        fired_at=trigger.fired_at,
        fire_source=trigger.fire_source,
        execution_mode=trigger.execution_mode,
        # delivery_target_resolved REMOVED 2026-05-24 — no outbound delivery framework
    )

    try:
        await dispatch_wake(
            trigger,
            fire_id,
            pool,
            handler=handler,
            wake_config=wake_config,
            # delivery_adapters REMOVED 2026-05-24 — no outbound delivery framework
        )
    except Exception as exc:
        log.exception("webhook dispatch failed for subscription %s", subscription_id)
        await fires.finalize_failed(fire_id, error=str(exc), duration_ms=None)
        return WebhookReceiveResult(500, fire_id, "dispatch failed")

    return WebhookReceiveResult(202, fire_id, "dispatched")
```

The `WebhookReceiver` framework in shard 06 calls `webhook_receive(...)`
inside its HTTP handler. Shard 06 owns the routing-and-HTTP-status
plumbing; this shard owns the verify → construct trigger → dispatch
flow.

---

## Patterns to Follow

- `TearsTool` subclass shape: `3tears-agent-tools` `packages/agent/tools/src/threetears/agent/tools/builtin/*.py` examples.
- Pydantic input schema with `Literal` typing: existing `3tears-agent-tools` tool examples.
- `MCPToolDefinition` shape: `3tears-agent-tools.base_tool.MCPToolDefinition`.
- Validator-import-reuse from REST: `_validate_*` functions importable as module-level symbols.
- `_resolve_schedule_id` pattern: short or full UUID resolution (existing metallm `_resolve_message_id` pattern from `conversation_tool_loaders.py` — adapted for `agent_wake_schedules`).

---

## Files to Create

- `packages/agent/wake/src/threetears/agent/wake/tools/__init__.py` — exports all 14 tool classes + validators.
- `packages/agent/wake/src/threetears/agent/wake/tools/schedule_tools.py` — Seven schedule tools.
- `packages/agent/wake/src/threetears/agent/wake/tools/webhook_tools.py` — Six webhook subscription tools.
- `packages/agent/wake/src/threetears/agent/wake/tools/validators.py` — `_validate_schedule_config`, `_validate_pre_check_config`, `_validate_context_from_chain` (importable by product REST).
- `packages/agent/wake/src/threetears/agent/wake/tools/resolve.py` — `_resolve_schedule_id`, `_resolve_subscription_id`, `_resolve_skill_id`.
- `packages/agent/wake/src/threetears/agent/wake/webhook_adapter.py` — `webhook_receive` + `WebhookReceiveResult`.
- `packages/agent/wake/tests/unit/test_validators.py` — every schedule_type + every pre_check_type validation case.
- `packages/agent/wake/tests/unit/test_tool_descriptions.py` — assert each tool description is ≤6 lines.
- `packages/agent/wake/tests/integration/test_tools_e2e.py` — create → list → pause → resume → delete cycle; cap-of-10 enforcement; cross-conv scoping.
- `packages/agent/wake/tests/integration/test_webhook_receive.py` — valid signature → 202 + fire_id; invalid signature → 401; rate limit → 429; bad template → 400.

---

## Implementation Notes

1. **Tool construction context.** Each tool is constructed with: `(pool, user_id, conversation_id, wake_config, is_admin)`. None of these are user-settable inputs on any tool. Multiple tools share the construction context; build a `WakeToolContext` dataclass for clean injection.

2. **Description discipline.** Each tool's description ≤6 short lines, ≤1 line per field. The platform's tools follow the existing `3tears-agent-tools` convention. Drift guard: `test_tool_descriptions.py` asserts line count.

3. **Cap-of-10 check on create.** Single COUNT query before insert: `SELECT COUNT(*) FROM agent_wake_schedules WHERE conversation_id = $1 AND status IN ('active', 'paused')`. If `>= wake_config.max_schedules_per_conversation`, error.

4. **`_resolve_schedule_id` accepts short or full UUID.** Short form: `[schedule:019e...]` — matches the existing metallm `_resolve_message_id` shape. Scoped to the active conversation; cross-conv resolution rejected.

5. **`_validate_schedule_config` per-type branches.** Pure function returning a human-readable error string or `None`:

   ```python
   def _validate_schedule_config(schedule_type: str, config: dict) -> str | None:
       match schedule_type:
           case "daily_at": ...
           case "every_n_hours": ...
           case "random_within_window": ...
           case "one_shot_at": ...
           case "cron": ...
           case "relative_delay": ...
       return None
   ```

   Each branch returns `"<schedule_type> requires '<field>' ...; got: {...}"` on invalid.

6. **TZ defaulting.** When `tz` is omitted in `daily_at` / `random_within_window`, fall back to the construction-context's `user_timezone` (consumer-supplied). Resolve once at create-time + store in `schedule_config`.

7. **`one_shot_at` past-time guard.** Reject creates with `fire_at_iso <= now`.

8. **Webhook secret storage.** The `WebhookSubscriptionCreateTool` calls `WebhookSubscriptionCollection.create_with_secret(plaintext_secret=secrets.token_hex(32), encryption_service=...)`. Returns the entity + the plaintext secret in a tuple. The tool returns the plaintext in the result string ONCE; no other code path retrieves it.

9. **`task_prompt_template` validation.** Parse via `SandboxedEnvironment.parse(template)` at create-time to catch syntax errors. Render-time errors (e.g. `event.field` access on a payload missing the field) surface at receive-time and fail with 400.

10. **`allowed_source_pattern` is a compiled regex.** Validated at create-time via `re.compile(pattern)`; rejected if invalid.

11. **`webhook_receive` is async-throughout.** No blocking I/O. HMAC `compare_digest` is constant-time + cheap; not a concern.

12. **Recursive cron-create disable detection is product-side.** The platform exposes the tools; the consumer decides when to include `WakeScheduleCreateTool` + `WebhookSubscriptionCreateTool` in the loaded set for a given turn. metallm's detection (query `messages.source` on the most recent non-assistant row) is metallm-specific. Documented in the product's tool-loading site, not here.

---

## Anti-patterns

- DO NOT accept arbitrary cron expressions without validating via `CronTrigger.from_crontab`. APScheduler raises opaquely at tick time otherwise.
- DO NOT silently accept unknown keys in `schedule_config`. Strict shape check.
- DO NOT expose `conversation_id` as a settable input on any tool. Construction-bound only.
- DO NOT add a "snooze for N minutes" tool. Out of scope for v1.
- DO NOT compute `next_fire_at` inside the tool body — call `_compute_next_fire_at` from shard 02's pure helper.
- DO NOT skip the `task_prompt` length cap (~4000 chars).
- DO NOT add `is_admin` as a user-settable input. Construction-time only — the platform receives `is_admin` from the consumer's auth context.
- DO NOT load the receiver's HTTP routing into this shard. shard 06 owns the `WebhookReceiver` framework + the routing; this shard owns the `webhook_receive` adapter function.
- DO NOT recompute the HMAC outside of `webhook_receive`. Single verification point; constant-time compare.

---

## Success Criteria

- [ ] All 13 tools subclass `TearsTool` correctly with valid `MCPToolDefinition`.
- [ ] All tool descriptions pass the ≤6-line drift guard.
- [ ] Per-type config validators reject every documented invalid case with clear field-level errors.
- [ ] `WakeScheduleCreateTool` creates a row + returns `[schedule:<id>]`; `next_fire_at` set correctly.
- [ ] Cap-of-10 enforced on create.
- [ ] `WakeSchedulePauseTool` → status='paused'; `Resume` → status='active' + recomputed `next_fire_at`.
- [ ] Cycle detection rejects `A → B → A` and `A → A`.
- [ ] `attached_skill_ids` ownership check rejects skills owned by other users.
- [ ] ~~`delivery_target='email'` rejected for unverified email.~~ ~~[REMOVED 2026-05-24]~~ — no delivery framework.
- [ ] `webhook_receive` returns 202 + fire_id for valid signature; 401 for invalid; 429 for over-rate.
- [ ] `webhook_receive` records a `wake_fires` row + invokes `dispatch_wake` correctly.
- [ ] `./scripts/check-all.sh` clean.
