# agent-wake — long-running agent foundation

> **Design-decision log:** see `metallm/docs/long_running/PLACEMENT.md` Section 1 for the canonical decision log. Key changes captured below.

## Why this release exists

3tears agents today only act when a user types something. Every existing agent built on this platform — metallm/Saoirse, future aibots, any multi-step agent — inherits that limitation. This release ships the long-running-agent foundation as a platform capability.

## Prerequisite

The [`agent-tools-eligibility`](../agent-tools-eligibility/shard-01-tool-eligibility-flags.md) shard MUST ship first (or bundled in the same release). It adds `tool_eligible` + `skill_eligible` flags to `TearsTool` — without those flags, the pre-check tools (which this package's wakes will surface via attached skills) cannot be registered correctly.

## What this release ships

- **`3tears-agent-wake`** — new agent capability package providing:
  - The typed `WakeTrigger` abstraction.
  - The `agent_wake_schedules` schema with a NULLABLE `skill_id` FK (one skill per wake max, per PLACEMENT §1.1).
  - The `wake_fires` history table.
  - The `webhook_subscriptions` table with a NULLABLE `default_skill_id` FK.
  - The tick engine + `_compute_next_fire_at`.
  - The `dispatch_wake(trigger, fire_id, pool, *, handler, wake_config, delivery_adapters)` convergence point.
  - The agent tools for schedule + webhook subscription CRUD (no skill attach/detach tools — skill is set via `_create`/`_update`).
  - Per-conv / per-user rate-limit mechanism (`_check_rate_limit`) reading caps from a product-supplied `WakeConfig` protocol.

- **`3tears-nats` `nats_distributed_lock`** — lifts metallm's `scheduler_lock` into the platform NATS package as a first-class primitive.

- **`3tears-channels` `WebhookReceiver`** — generic HMAC-verified inbound webhook receiver. Verifies signatures, looks up subscriptions, renders payload templates, constructs `WakeTrigger`, calls `dispatch_wake`.

The first consumer is metallm (long_running). Future consumers get the same capability as a single dependency.

## What this release does NOT ship (2026-05-19 revision)

- **No `wake_pre_check_types` table.** Pre-checks are ordinary `TearsTool` subclasses (`http_get`, `loki_query`, `postgres_query`) registered in `3tears-agent-tools` with `tool_eligible=False, skill_eligible=True`. A wake's attached skill surfaces them via `tool_additions`. The LLM calls them via its normal tool loop. PLACEMENT §1.2.

- **No pre-check executor framework in `dispatch_wake`.** `dispatch_wake` no longer has an "execute pre-check, capture output, decide whether to fire LLM" stage. The agent's normal tool loop handles tool calls; the agent's response can start with `[SILENT]` to suppress display. PLACEMENT §1.2.

- **No `no_agent` mode.** The "skip the LLM, deliver pre-check output as the message" pattern is replaced by a skill with `prompt_mode='replace'` and a body instructing the model to emit `[SILENT]` when nothing's changed (or to emit structured findings only). PLACEMENT §1.6.

- **No `wake_schedule_skill_attachments` junction table.** One skill per wake max — single nullable FK column on `agent_wake_schedules`. PLACEMENT §1.1 / §1.3.

- **No `webhook_subscription_skill_attachments` junction table.** Subscriptions carry a single nullable `default_skill_id` FK.

- **No `wake_skill_attach_tool` / `wake_skill_detach_tool` agent tools.** Skill attachment is via `wake_schedule_create(..., skill_id=picked)` and `wake_schedule_update(..., skill_id=picked)`.

## Locked design decisions

These are pre-resolved per the design conversations on 2026-05-19. Shards reference them; do NOT re-litigate.

| Decision | Locked answer | Source |
|---|---|---|
| Package home for wake runtime | New `3tears-agent-wake` at `packages/agent/wake/`. | |
| Schedules-are-rows | Source of truth in `agent_wake_schedules`. APScheduler in-memory tick only. | |
| One convergence point | Every wake source constructs `WakeTrigger` + calls `dispatch_wake`. | |
| Handler-callback pattern | `dispatch_wake` invokes a product-supplied `HandlerCallback`. metallm's runs `personality_node`; future products supply their own. | |
| Tick interval | 60 seconds. Product registers the APScheduler job; platform provides the pure-async tick body. | |
| Distributed lock | `3tears-nats.nats_distributed_lock` — first-class primitive. Replaces metallm's `scheduler_lock`. | |
| Concurrency model | Serial per conversation (per-conv NATS lock). Missed-fire policy default `'coalesce'` (configurable per-schedule field). Drift = fire-late, record actual_fired_at. | PLACEMENT §1.3 / §1.7 / §1.8 |
| One skill per wake | Nullable `skill_id` FK on `agent_wake_schedules`. No junction tables. | PLACEMENT §1.1 / §1.3 |
| Pre-check tools | Ordinary `TearsTool` subclasses with `tool_eligible=False, skill_eligible=True` in `3tears-agent-tools`. The wake's attached skill surfaces them. | PLACEMENT §1.2 / §1.6 |
| `[SILENT]` suppression | LLM response starting with `[SILENT]` stored but `wake_fires.display_suppressed=true` + product-side `messages.display='hidden'`. | PLACEMENT §1.4 |
| `context_from` chains | Single-hop, same-conversation only. | |
| Delivery routing | `delivery_target` enum starts with `conversation` (default) + `email`. Email rate-limited per-recipient (5/hr default). `DeliveryAdapter` protocol platform; concrete adapters product. | |
| Skill table location | `3tears-agent-skills.agent_skills` (sibling package; ships in same release or earlier). FK in `agent_wake_schedules.skill_id`. | |
| Webhook receiver host | `3tears-channels`. Per-subscription rate-limit defaults product-supplied. | |
| Per-conv / per-user rate-limit | Mechanism platform (`_check_rate_limit`); cap values product-supplied via `WakeConfig`. | |
| Per-conv active-schedule cap | Mechanism platform; default 10 (product-configurable). | PLACEMENT §1.9 |
| Counter naming | `3tears_agent_wake_*` prefix (platform-emitted). | PLACEMENT §1.15 |
| Pydantic API models | Platform-defined in `3tears-agent-wake.api_models`; consumers import. | PLACEMENT §1.16 |

## Shard sequence (revised)

Shards are sequential. Each one depends on the previous.

| # | Shard | What it does |
|---|---|---|
| 01 | [Schema + collections](shard-01-schema-and-collections.md) | `agent_wake_schedules` (with nullable `skill_id` FK), `wake_fires`, `webhook_subscriptions` (with nullable `default_skill_id` FK). **No `wake_pre_check_types`, no junction tables.** Entity classes + Collections + agent-scope migration registration. |
| 02 | [Tick engine + distributed lock](shard-02-tick-and-distributed-lock.md) | (a) Lift `nats_distributed_lock` into `3tears-nats`. (b) `wake_tick_job` body + `_compute_next_fire_at` + `WakeScheduleCollection.claim_and_reschedule` + missed-fire policy handling. |
| 03 | [Dispatch handler + WakeTrigger](shard-03-dispatch-handler.md) | `WakeTrigger`, `WakeDispatchResult`, `HandlerCallback` protocol, `dispatch_wake(trigger, fire_id, pool, handler, wake_config, delivery_adapters)` with: rate-limit, conv lookup, skill resolution (SINGLE), context_from resolution, [SILENT] suppression detection, delivery routing. **No pre-check executor framework.** |
| 04 | [Agent tools + webhook adapter](shard-04-agent-tools-and-webhook-adapter.md) | 13 `TearsTool` subclasses: six wake-schedule (`Create` with `skill_id` param, `Update` with `skill_id`, `List`, `Pause`, `Resume`, `Delete`) + seven webhook-subscription (`Create` with `default_skill_id`, `Update`, `List`, `Pause`, `Resume`, `Delete`, `RotateSecret`). **No `wake_skill_attach`/`wake_skill_detach`.** Plus `WebhookReceiver → WakeTrigger` adapter glue. |
| 05 | [Observability + rate-limit + Pydantic models](shard-05-observability-and-models.md) | Prometheus counters with `3tears_agent_wake_*` prefix. Loki event names. `_check_rate_limit` + per-conv active-schedule cap framework. `WakeConfig` protocol. Pydantic request/response models. |
| 06 | [Channels webhook receiver framework](shard-06-channels-webhook-receiver.md) | `3tears-channels`-side: `WebhookReceiver` (HMAC-SHA256 verification, per-subscription rate-limit, payload templating with sandboxed Jinja2, subscription CRUD primitive). |

**Cross-cutting addition (wake-yield, 2026-05-19 evening per Saoirse-review optimization):**

The wake-yield cooperative-interrupt feature touches FOUR of the shards above:
- **shard-01 (schema + collections):** `wake_fires.status` CHECK constraint enum gains `'yielded'` value.
- **shard-03 (dispatch-handler):** pending-user-message detection + system-prompt hint + yield-handling at iteration boundaries.
- **shard-04 (agent tools + webhook adapter):** new 14th `TearsTool` — `WakeYieldTool`, gated to load only on wake-driven turns.
- **shard-05 (observability):** `'yielded'` status enum value + Loki event + new Prometheus histogram (`3tears_agent_wake_yield_duration_seconds`).

Each shard's 2026-05-19 revision deltas section documents the wake-yield additions. Canonical end-to-end design: `metallm/docs/long_running/shard-10-cooperative-yield.md` (metallm-side counterpart shard).

## Migration dependency graph

```
3tears-conversations (existing) ──┐
                                  │
3tears-agent-skills (sibling) ───┐│
                                 ││
                                 ▼▼
                       3tears-agent-wake (this release)
                                 │
                                 │   depends_on:
                                 │   - conversations (FK conversation_id)
                                 │   - agent_skills (FK skill_id, default_skill_id)
                                 │
                                 ▼
                       3tears-channels (this release)
                                 │
                                 │   WebhookReceiver invokes agent_wake.dispatch_wake
                                 ▼
                       metallm (consumer)
```

`3tears-agent-wake.migrations.__init__.py::register(runner)`:

```python
pkg = PackageMigrations(
    name="agent_wake",
    scope=MigrationScope.AGENT,
    depends_on=("conversations", "agent_skills"),
)
```

## Verification of the whole release

1. **Pass `./scripts/check-all.sh`** — lint + typecheck + tests across all touched packages.
2. **Pass the metallm bump dry-run** — apply the new 3tears release to a metallm dev environment.
3. **No new Prometheus labels with unbounded cardinality** — `conversation_id` / `user_id` / `schedule_id` NOT labels. Enforcement test in shard 05.
4. **Smoke test against real Postgres + NATS** — apply migrations, create a one-shot wake, run the tick, assert fire row + handler callback received a populated `WakeTrigger` with `skill_id` either set or null.
5. **Webhook smoke** — POST to receiver with valid HMAC signature; assert 202 + fire row + handler invoked.

## Scope of platform vs. product

This release stops at the abstraction boundary. Specifically NOT in scope here:

- **The personality_node integration.** metallm's `HandlerCallback` lives in metallm's repo.
- **The FastAPI router files.** metallm exposes wake endpoints via its own router.
- **React UI.** metallm's frontend.
- **System prompt assembly.** metallm's `build_wake_awareness_block` lives in metallm. The `PreparedWakeContext` dataclass that drives it is platform.
- **URL allow-list / named-query contents.** Mechanism in `3tears-agent-tools` (the tools themselves read product-supplied config); values configured by the consuming product.
- **Per-conv / per-user cap values.** Mechanism platform; values supplied by the product's `WakeConfig`.
- **Grafana dashboards.** Platform emits counters; product builds dashboard.
- **Email SMTP wrapper.** Framework platform; concrete adapter product.
- **Skill-body rendering / per-turn composition.** Handled by `3tears-agent-skills.rendering.compose_turn_context`. This package just resolves `skill_id` to an `AgentSkillEntity` and hands it to the consumer via `PreparedWakeContext.attached_skill`.
