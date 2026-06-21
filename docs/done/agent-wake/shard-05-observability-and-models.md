# agent-wake-05: Observability + rate-limit framework + Pydantic models

> **REMOVED 2026-05-24:** the outbound delivery framework was removed as an undesigned parallel abstraction. The `3tears_agent_wake_delivery_total` counter (OBS-07), the `delivery_failed` failure reason, the `max_email_per_recipient_per_hour` `WakeConfig` field + its default (OBS-15 / OBS-16), and the `delivery_target` / `delivery_config` / `delivery_target_resolved` / `delivery_status` Pydantic model fields are GONE. Wake fires now always deliver into the conversation; outbound delivery, if ever needed, will be a threetears.channels adapter. Inbound webhooks are unaffected. The text below retains these for history — the struck requirements must NOT be rebuilt.

## 2026-05-19 revision deltas (apply BEFORE implementing)

Canonical source: `<metallm>/docs/long_running/PLACEMENT.md`.

**Counter updates:**
- DROP `3tears_agent_wake_pre_check_total` — pre-checks are tool calls.
- DROP `3tears_agent_wake_no_agent_total` — `no_agent` mode is gone.
- KEEP `3tears_agent_wake_fires_total{status, schedule_type, execution_mode}`. Status enum: `'fired'`, `'fired_silent'`, **`'yielded'`** (NEW per wake-yield), `'skipped_busy'`, `'skipped_rate_limit'`, `'skipped_cap'`, `'skipped_no_handler'`, `'failed'`. (No `'skipped_gate'`.)
- ADD `3tears_agent_wake_drift_seconds` histogram — `actual_fired_at - scheduled_fire_at`.
- ADD `3tears_agent_wake_yield_duration_seconds` histogram — `actual_fired_at - wake_started_at` for yielded fires.

**`_check_rate_limit` framework gains:**
- `_check_active_schedule_cap(trigger, pool, config)` — `count(*) WHERE conversation_id = ? AND status = 'active' <= config.max_schedules_per_conversation`. Returns `'skipped_cap'` reason. PLACEMENT §1.9.

**Pydantic model updates:**
- `CreateWakeScheduleRequest` — gains `skill_id: UUID | None`. Drops `no_agent`, `pre_check_type`, `pre_check_config`.
- ADD `UpdateWakeScheduleRequest` (new model). Same fields all optional.
- `CreateWebhookSubscriptionRequest` — gains `default_skill_id: UUID | None`. Drops pre-check fields.
- ADD `UpdateWebhookSubscriptionRequest` (new model).
- `WakeScheduleResponse` — drops pre-check / no_agent; adds `skill_id`, `missed_fire_policy`.
- `WakeFireResponse` — drops `pre_check_output`; adds `actual_fired_at`, `scheduled_fire_at`.
- DROP `WakePreCheckTypeResponse` entirely.

**Loki event names:**
- DROP `3tears.agent_wake.pre_check.executed`.
- DROP `3tears.agent_wake.no_agent.fired`.
- KEEP `3tears.agent_wake.fire.silent`.
- ADD `3tears.agent_wake.fire.drift` (when `actual - scheduled > 60s`).
- ADD `3tears.agent_wake.fire.yielded` (when Saoirse called `wake_yield`). Carries `wake_fire_id`, `schedule_id`, `conversation_id`, `actual_fired_at`, `wake_duration_ms`. No body content.

## Objective

Three platform-side concerns colocated because they share the
`WakeConfig` protocol surface:

1. Prometheus counter / histogram registrations (with the
   bounded-cardinality discipline preserved).
2. Loki structured event names + payload shapes.
3. The `_check_rate_limit` framework + `WakeConfig` protocol.
4. The Pydantic request/response models (shared by the consumer's
   REST router).

Consumers (metallm) implement `WakeConfig` by reading their own
`system_settings`. Consumers consume the Pydantic models by importing
them in their FastAPI router.

This shard renumbers the original metallm shard-08 concerns: the
mechanism is platform; the policy values are product. Grafana
dashboards stay product-side (different consumers want different
panels).

---

## Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| OBS-01 | New module `packages/agent/wake/src/threetears/agent/wake/metrics.py` exporting Prometheus counter / histogram instances. | P0 |
| OBS-02 | Counter `3tears_agent_wake_fires_total{status, schedule_type, execution_mode}` — 4 statuses × 6 types × 2 modes = 48 series max. | P0 |
| OBS-03 | Counter `3tears_agent_wake_failures_total{reason}` — bounded set of reasons (`conv_deleted`, `rate_limited`, `pre_check_failed`, `handler_exception`, ~~`delivery_failed`~~ ~~[REMOVED 2026-05-24]~~, etc.). | P0 |
| OBS-04 | Histogram `3tears_agent_wake_tick_duration_seconds` (no labels — single histogram). | P0 |
| OBS-05 | Counter `3tears_agent_wake_pre_check_total{pre_check_type, outcome}` — outcome in `output | empty | skip_prefix | failed`. | P0 |
| OBS-06 | Histogram `3tears_agent_wake_pre_check_duration_seconds{pre_check_type}`. | P0 |
| ~~OBS-07~~ | ~~[REMOVED 2026-05-24]~~ ~~Counter `3tears_agent_wake_delivery_total{target, status}` — target in `conversation | email | future-values`; status in `delivered | failed | suppressed_silent`.~~ No delivery framework; do NOT add this counter. | ~~P0~~ |
| OBS-08 | Counter `3tears_agent_wake_skill_load_total{outcome}` — outcome in `loaded | skipped_invalid | skipped_missing | skipped_disabled`. | P0 |
| OBS-09 | Counter `3tears_agent_wake_webhook_received_total{outcome}` — outcome in `accepted | auth_failed | rate_limited | bad_template | source_rejected | not_found`. | P0 |
| OBS-10 | NO labels including `conversation_id`, `user_id`, `schedule_id`, `subscription_id`. Unbounded cardinality. Enforced via a drift-guard test. | P0 |
| OBS-11 | Structured Loki event names follow `3tears.agent_wake.<event>` schema: `tick.started`, `tick.completed`, `fire.dispatched`, `fire.skipped_busy`, `fire.skipped_gate`, `fire.failed`, `fire.rate_limited`, `fire.silent`, `webhook.received`, `webhook.auth_failed`. | P0 |
| OBS-12 | Loki event payloads include `schedule_id` / `conversation_id` / `user_id` / `schedule_type` / `execution_mode` / `fire_source` for grepability. NEVER include `task_prompt` content (PII risk). | P0 |
| OBS-13 | New module `packages/agent/wake/src/threetears/agent/wake/rate_limit.py` exporting `_check_rate_limit(trigger, pool, wake_config) -> bool`. Returns True when both per-conv and per-user counts are under cap. | P0 |
| OBS-14 | Rate-limit query semantics: count `wake_fires` rows in the last 24h with `status='fired'` (rate_limited rows do NOT count toward the cap because their `next_fire_at` already bumped past the window). Per-user count joins through both `agent_wake_schedules` AND `webhook_subscriptions` to cover both fire sources. | P0 |
| OBS-15 | `WakeConfig` Protocol in `packages/agent/wake/src/threetears/agent/wake/config.py` carrying: `max_fires_per_conv_per_day`, `max_fires_per_user_per_day`, ~~`max_email_per_recipient_per_hour`~~ ~~[REMOVED 2026-05-24]~~, `max_webhook_fires_per_subscription_per_hour`, `max_schedules_per_conversation`, `http_allowed_hosts`, `loki_client`, `loki_named_queries`, `postgres_named_queries`. | P0 |
| OBS-16 | Default values: per-conv 24, per-user 100, ~~per-email 5~~ ~~[REMOVED 2026-05-24]~~, per-webhook-sub 60, max-schedules 10. These are the DEFAULT VALUES (consumer can override via their `WakeConfig` impl). | P0 |
| OBS-17 | Pydantic request/response models: `CreateWakeScheduleRequest`, `UpdateWakeScheduleRequest`, `WakeScheduleResponse`, `WakeScheduleListResponse`, `WakeFireResponse`, `WakeFireListResponse`, `CreateWebhookSubscriptionRequest`, `UpdateWebhookSubscriptionRequest`, `WebhookSubscriptionResponse`, `WebhookSubscriptionListResponse`, `WakePreCheckTypeResponse`, `WakePreCheckTypeListResponse`. | P0 |
| OBS-18 | Pydantic models live in `packages/agent/wake/src/threetears/agent/wake/api_models.py` so they can be imported by any consuming product's FastAPI router without dragging in the full collection layer. | P0 |
| OBS-19 | Rate-limited tick handling: when cap is hit on a scheduled-tick fire, insert ONE `wake_fires` row with `status='rate_limited'` AND UPDATE the schedule's `next_fire_at` to the window rollover time. The schedule resumes naturally at the next window. Schedule is NOT paused. | P0 |
| OBS-20 | Webhook rate-limited: receiver returns 429 with `Retry-After` header pointing to the window rollover. | P0 |

---

## `WakeConfig` Protocol

```python
# packages/agent/wake/src/threetears/agent/wake/config.py
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class WakeConfig(Protocol):
    """Read-side configuration the consumer supplies to dispatch_wake.

    Implementations typically read from the consumer's system_settings table.
    Pure read protocol — no mutation methods. Cached in the consumer (suggested
    60s TTL); admin updates propagate within a tick.
    """

    @property
    def max_fires_per_conv_per_day(self) -> int: ...

    @property
    def max_fires_per_user_per_day(self) -> int: ...

    # max_email_per_recipient_per_hour REMOVED 2026-05-24 — no outbound delivery framework

    @property
    def max_webhook_fires_per_subscription_per_hour(self) -> int: ...

    @property
    def max_schedules_per_conversation(self) -> int: ...

    @property
    def http_allowed_hosts(self) -> tuple[str, ...]: ...

    @property
    def loki_client(self) -> Any | None:
        """A consumer-supplied Loki client with a query_range(query, since, until, limit) async method."""
        ...

    @property
    def loki_named_queries(self) -> dict[str, str]: ...

    @property
    def postgres_named_queries(self) -> dict[str, str]: ...


DEFAULT_MAX_FIRES_PER_CONV_PER_DAY = 24
DEFAULT_MAX_FIRES_PER_USER_PER_DAY = 100
# DEFAULT_MAX_EMAIL_PER_RECIPIENT_PER_HOUR REMOVED 2026-05-24 — no outbound delivery framework
DEFAULT_MAX_WEBHOOK_FIRES_PER_SUBSCRIPTION_PER_HOUR = 60
DEFAULT_MAX_SCHEDULES_PER_CONVERSATION = 10
```

---

## `_check_rate_limit` framework

```python
# packages/agent/wake/src/threetears/agent/wake/rate_limit.py
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from asyncpg import Pool

from threetears.observe import get_logger

from threetears.agent.wake.config import WakeConfig
from threetears.agent.wake.types import WakeTrigger

log = get_logger(__name__)


async def _check_rate_limit(
    trigger: WakeTrigger,
    pool: Pool,
    config: WakeConfig,
) -> bool:
    """Return True when both per-conv and per-user counts are under cap.

    Counts only status='fired' rows in the last 24h. rate_limited rows do
    not count toward the cap because their next_fire_at already advanced
    past the window when first throttled.
    """
    since = datetime.now(UTC) - timedelta(hours=24)
    conv_count = await pool.fetchval(
        """
        SELECT COUNT(*) FROM wake_fires
        WHERE conversation_id = $1 AND fired_at > $2 AND status = 'fired'
        """,
        trigger.conversation_id,
        since,
    )
    if conv_count >= config.max_fires_per_conv_per_day:
        return False

    # Per-user: cover both schedule-source and webhook-source fires
    user_count = await pool.fetchval(
        """
        SELECT
            (SELECT COUNT(*) FROM wake_fires wf
             JOIN agent_wake_schedules ws ON wf.schedule_id = ws.schedule_id
             WHERE ws.user_id = $1 AND wf.fired_at > $2 AND wf.status = 'fired')
          + (SELECT COUNT(*) FROM wake_fires wf
             JOIN webhook_subscriptions ws ON wf.webhook_subscription_id = ws.subscription_id
             WHERE ws.user_id = $1 AND wf.fired_at > $2 AND wf.status = 'fired')
          AS total
        """,
        trigger.user_id,
        since,
    )
    return user_count < config.max_fires_per_user_per_day
```

`dispatch_wake` calls `_check_rate_limit` as its first step (per shard
03 step 1). When False, `dispatch_wake` UPDATEs the `wake_fires` row to
`status='rate_limited'` and (for scheduled-tick sources) the caller
(tick) bumps the schedule's `next_fire_at` to the window rollover.

---

## Pydantic models

```python
# packages/agent/wake/src/threetears/agent/wake/api_models.py
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class CreateWakeScheduleRequest(BaseModel):
    schedule_type: Literal["daily_at", "every_n_hours", "random_within_window", "one_shot_at", "cron", "relative_delay"]
    schedule_config: dict[str, Any]
    execution_mode: Literal["inline", "spawn"] = "inline"
    task_prompt: str | None = None
    name: str | None = None
    no_agent: bool = False
    pre_check_type: str | None = None
    pre_check_config: dict[str, Any] = Field(default_factory=dict)
    context_from_schedule_id: str | None = None
    # delivery_target / delivery_config REMOVED 2026-05-24 — no outbound delivery framework
    attached_skill_ids: list[str] = Field(default_factory=list)


class UpdateWakeScheduleRequest(BaseModel):
    status: Literal["active", "paused"] | None = None  # 'expired' is server-set only
    name: str | None = None
    task_prompt: str | None = None


class WakeScheduleResponse(BaseModel):
    schedule_id: str
    conversation_id: str
    user_id: str
    schedule_type: str
    schedule_config: dict[str, Any]
    task_prompt: str | None
    execution_mode: str
    status: str
    next_fire_at: str | None        # ISO-8601 UTC
    last_fired_at: str | None
    name: str | None
    no_agent: bool
    pre_check_type: str | None
    pre_check_config: dict[str, Any]
    context_from_schedule_id: str | None
    # delivery_target / delivery_config REMOVED 2026-05-24 — no outbound delivery framework
    attached_skill_ids: list[str]    # in position order
    date_created: str
    date_updated: str


class WakeScheduleListResponse(BaseModel):
    schedules: list[WakeScheduleResponse]
    total_count: int


class WakeFireResponse(BaseModel):
    fire_id: str
    schedule_id: str | None
    webhook_subscription_id: str | None
    conversation_id: str
    target_conversation_id: str | None
    fired_at: str
    fire_source: str
    execution_mode: str
    status: str
    error: str | None
    duration_ms: int | None
    pre_check_output: str | None
    pre_check_duration_ms: int | None
    # delivery_target_resolved / delivery_status REMOVED 2026-05-24 — no outbound delivery framework
    display_suppressed: bool


class WakeFireListResponse(BaseModel):
    fires: list[WakeFireResponse]
    total_count: int


class CreateWebhookSubscriptionRequest(BaseModel):
    name: str | None = None
    task_prompt_template: str
    execution_mode: Literal["inline", "spawn"] = "inline"
    # delivery_target / delivery_config REMOVED 2026-05-24 — no outbound delivery framework
    attached_skill_ids: list[str] = Field(default_factory=list)
    allowed_source_pattern: str | None = None


class UpdateWebhookSubscriptionRequest(BaseModel):
    status: Literal["active", "paused"] | None = None
    name: str | None = None
    task_prompt_template: str | None = None


class WebhookSubscriptionResponse(BaseModel):
    subscription_id: str
    conversation_id: str
    user_id: str
    name: str | None
    execution_mode: str
    status: str
    task_prompt_template: str | None
    # delivery_target / delivery_config REMOVED 2026-05-24 — no outbound delivery framework
    verification_scheme: str
    allowed_source_pattern: str | None
    last_fired_at: str | None
    attached_skill_ids: list[str]
    date_created: str
    date_updated: str
    # secret_plaintext only present on Create + RotateSecret responses,
    # not on the base shape — see CreateWebhookSubscriptionResponse below


class CreateWebhookSubscriptionResponse(WebhookSubscriptionResponse):
    secret_plaintext: str  # display-once


class WebhookSubscriptionListResponse(BaseModel):
    subscriptions: list[WebhookSubscriptionResponse]  # no secret_plaintext on listings
    total_count: int


class WakePreCheckTypeResponse(BaseModel):
    type_id: str
    display_name: str
    description: str | None
    config_jsonschema: dict[str, Any]
    requires_admin: bool


class WakePreCheckTypeListResponse(BaseModel):
    types: list[WakePreCheckTypeResponse]
```

---

## Patterns to Follow

- Prometheus metric registration site: existing `3tears-models` `UsageTracker` instrument pattern. Single source of truth per metric.
- Loki event logging: `threetears.observe.get_logger(__name__).info("event.name", extra={"extra_data": {...}})`.
- Protocol with `@runtime_checkable`: existing `3tears-agent-tools` Protocol shapes.
- Pydantic v2 patterns: existing `3tears-models` request/response model shapes.

---

## Files to Create

- `packages/agent/wake/src/threetears/agent/wake/metrics.py` — Prometheus counter / histogram registrations.
- `packages/agent/wake/src/threetears/agent/wake/rate_limit.py` — `_check_rate_limit(trigger, pool, config)`.
- `packages/agent/wake/src/threetears/agent/wake/config.py` — `WakeConfig` Protocol + DEFAULT_ constants.
- `packages/agent/wake/src/threetears/agent/wake/api_models.py` — all Pydantic models.
- `packages/agent/wake/src/threetears/agent/wake/events.py` — Loki event name constants (string constants — `EVENT_FIRE_DISPATCHED = "3tears.agent_wake.fire.dispatched"`, etc.) to prevent typo drift.
- `packages/agent/wake/tests/unit/test_metrics_cardinality.py` — drift guard: assert no metric labelnames include `conversation_id`, `user_id`, `schedule_id`, `subscription_id`.
- `packages/agent/wake/tests/unit/test_rate_limit.py` — rate-limit logic with mocked counts; per-conv at cap; per-user at cap; both under cap.
- `packages/agent/wake/tests/unit/test_api_models.py` — Pydantic model round-trip + validation tests.

---

## Implementation Notes

1. **Counter naming convention.** `3tears_agent_wake_*` prefix. metallm's existing dashboards reference `metallm_wake_*` and need a one-time rename — this is captured in metallm's shard 08 (residue).

2. **Bounded-label discipline.**
   - `status` in `{fired, skipped_busy, skipped_gate, failed, rate_limited}` → 5 values.
   - `schedule_type` in `{daily_at, every_n_hours, random_within_window, one_shot_at, cron, relative_delay, external_event}` → 7 values.
   - `execution_mode` in `{inline, spawn}` → 2 values.
   - Max cardinality of `3tears_agent_wake_fires_total` = 70 series. Acceptable.

3. **Loki event name discipline.** All events under `3tears.agent_wake.<area>.<event>`. The `events.py` module exports them as constants; tests assert grep-ability.

4. **`task_prompt` NEVER in logs.** PII risk; conversation messages are the source of truth for what was said. Drift guard test: AST grep for any `_logger.*` call site in `agent/wake/` that includes `task_prompt` in `extra_data`.

5. **Rate-limit query uses indexes.** `idx_wake_fires_conv_time` + `idx_wake_fires_conv_time_status` cover both the per-conv count and the per-user count's two subqueries. Verify with EXPLAIN ANALYZE in an integration test.

6. **Window-rollover logic for rate-limited fires.** When rate-limited, the tick sets `next_fire_at` to either:
   - `now + 1h` (a "reasonable retry" — try once an hour to see if quota has freed),
   - OR the schedule's normal cadence — whichever is later.

   This prevents the schedule from being stuck "polling every 60s and rate-limiting every poll." Simple impl: `next_fire_at = max(normal_next_fire, now + 1h)`.

7. **`_check_rate_limit` is called inside `dispatch_wake`.** Step 1 per shard 03. The function takes the `WakeConfig` impl and the asyncpg pool; no other dependencies.

8. **Pydantic model placement.** Models in `api_models.py` are import-safe without dragging in `asyncpg` or `nats`. The product's FastAPI router imports from `threetears.agent.wake.api_models` cleanly.

9. **`config_jsonschema` rendering on `WakePreCheckTypeResponse`.** The jsonb column is rendered as a Python dict in the response — Pydantic serializes to JSON automatically. The shape is the JSON Schema document, not a re-rendered form schema.

10. ~~**Email rate-limit per recipient.** Lives in the email delivery adapter (product-side). The platform exports the default value via `WakeConfig.max_email_per_recipient_per_hour`; the product's email adapter reads it and applies the limit.~~ ~~[REMOVED 2026-05-24]~~ — no outbound delivery framework; do NOT build email rate-limiting.

---

## Anti-patterns

- DO NOT add `conversation_id` / `user_id` / `schedule_id` / `subscription_id` as Prometheus labels. Drift-guard test enforces this.
- DO NOT log `task_prompt` content. Drift-guard test enforces this.
- DO NOT couple the rate-limit query to a specific NATS KV cache. Two COUNT queries per fire; indexed; cheap enough.
- DO NOT pause schedules when rate-limit hits. Window-rollover bump only.
- DO NOT bundle Prometheus dashboards into this package. Consumers ship their own dashboards.
- DO NOT inline the `WakeConfig` impl into the package as a default. Protocol only; consumers supply.

---

## Success Criteria

- [ ] All Prometheus counters / histograms registered with the correct labelnames.
- [ ] Cardinality drift-guard test passes.
- [ ] `_check_rate_limit` returns False at cap, True under; both per-conv and per-user paths tested.
- [ ] Rate-limit query plan uses the indexes from shard 01 (verified via EXPLAIN ANALYZE).
- [ ] Loki event name constants exist; structured emission via `_logger.info(constant, extra={"extra_data": {...}})`.
- [ ] `task_prompt` drift-guard test passes.
- [ ] Pydantic models round-trip cleanly.
- [ ] `WakeConfig` Protocol importable without dragging in transport dependencies.
- [ ] `./scripts/check-all.sh` clean.
