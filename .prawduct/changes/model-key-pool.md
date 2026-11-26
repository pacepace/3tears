---
schema: 1
type: feature
scope: model-key-pool
date: 2026-11-26
title: Model key pool
---
A chat model can run on a pool of 1..n API keys for one provider (anthropic, openai or openrouter), each inside an
operator-set budget per billing period. The pool stays on its active key until that key is exhausted, revoked,
forbidden, tripped or low on budget, then moves it, durably, for every process sharing the pool. A rate limit means
retry on the same key; it never moves the key. A key forbidden for one model is skipped by that model's calls and
never moves the active key, which every model shares. One key is a normal pool. Requirements: `docs/model-key-pool-requirements.md`; operator guide: `docs/model-key-pool.md`.

- `KeyPool` and `KeySpec` in `threetears.models`: a redacted `Secret`, a label, a USD budget, an optional priority and
  provider account per key. Sticky selection, with priority pull-back when priorities differ, behind a selection
  seam. Spend is reserved before each call and settled after it, so concurrent calls can't pass a key's budget.
- Per-provider classifiers read what a failure says about the key: exhausted (until the reset the provider named, or
  the period's end), revoked, forbidden, rate-limited, a request too large for the key, or not the key's. They tell a
  used-up quota from a rate limit by the body, since providers send both as a 429. An account-scoped verdict reaches
  every key in that account. An unknown error never burns a key.
- `NoKeyAvailableError`, a `ModelProviderError`, names every key's label and state and the earliest time one could
  serve again. `await pool.status()` gives the same facts for a health endpoint, and `pool.reset(label)` clears a
  key's state.
- Stores: `InMemoryKeyPoolStore` for one process, and `BudgetLedger` in `threetears.core.coordination` for a pool
  shared across processes. The ledger keeps spend in two new coordination tables (migration v004), written to L3
  before each call returns, and refuses a registry with no L2 or no L3. A store failure fails the call closed.
- `cost_of_usage` is the one place usage is priced: cache reads and writes at their own rates, each falling back to
  the input rate, plus `cost_per_request`. The tracker, `UsageAccumulator` and the pool all use it.
- `estimate_input_tokens` is public. `create_chat_model` takes `breaker_registry=`, which keeps the per-model
  rate-limit breaker that `breaker=` drops. `CircuitBreaker.seconds_until_admit()` and
  `CircuitBreakerRegistry.peek()` read a breaker without side effects. The provider factories take
  `unretried_key_errors=`, a rule (the pool's `refuses_retry`) naming the responses that reach the caller without the
  SDK first retrying them on the same key: a 401, 402 or 403, a 429 whose body reports a quota, and a 429 asking a
  wait past the pool's `rate_limit_exhaustion_wait`. A rate-limit 429 keeps the SDK's retries.
- `create_chat_model(model_id, key_pool=pool)` builds the pooled model, a `PooledChatModel`: one model per key, with
  every re-binding (`bind_tools`, `with_structured_output`, `with_retry` and the rest) applied to each. A call
  reserves its estimate on the active key, fails over to the next key when the provider refuses that key, and
  settles the cost the provider reported; with no reported usage the estimate stays as spend. Each key is tried once
  per call. A rate limit is retried on the same key -- by the SDK for anthropic and openai, by the pool twice for
  openrouter, whose SDK never retries a 429 -- then reaches the caller unchanged. A rate limit asking a wait past
  `rate_limit_exhaustion_wait` (120 s) counts as exhaustion until it ends. OpenRouter's `weight_exceeds_budget` 402
  skips the key for that call only. Usage records carry the serving key's label as `credential_label`
  (`llm.credential_label` on spans).
- `KeyPoolMetrics` in `threetears.models.key_pool`, behind the `prometheus` extra, exports each key's spend,
  reservations, room, budget, state and active flag, and counts moves, failovers, refusals for room, orphaned
  reservations, `NoKeyAvailableError` raises and store failures.

## Consumer note
Install changes in `3tears-models`:

- `anthropic`'s floor moves from 0.116 to 1.12. The unretried mode's middleware is verified against 1.12. Upgrade a consumer
  that pins `anthropic` below it.
- `openai` (`>=3.26`) and `httpx` (`>=0.27`) are now direct dependencies of models. Both arrived transitively before;
  a consumer pinning either below those floors fails to resolve.

`threetears_llm_cost_usd_total`, and the tracker's `cost_usd`, now price cache reads and writes at the model's own
cache rates instead of the full input rate, and add the model's `cost_per_request` once per call. For cache-heavy
Anthropic traffic the counter drops, since cache reads cost far less than input. Traffic that writes the cache more
than it reads it can rise, where a cache-write price above the input rate is registered. Dashboards and alerts on
the counter will step at the upgrade; the new values are the correct ones. `UsageAccumulator` totals change the same
way.

The key pool itself is opt-in: no existing `create_chat_model(api_key=...)` call changes behaviour. `api_key` is now
optional, and exactly one of `api_key` and `key_pool` is required. A consumer adopting the pool:

- With `BudgetLedger`, calls the model through `ainvoke`, `astream` and `abatch`. The sync paths run the pool on a
  private event loop, which the ledger's loop-bound NATS and database clients don't support. Sync calls suit
  `InMemoryKeyPoolStore`.
- Applies the coordination migrations through v004 for `BudgetLedger` (or declares `LEDGER_TABLE_SCHEMAS` where its
  pod can't run DDL).
- For metrics, builds one `KeyPoolMetrics` per Prometheus registry, passes it as `KeyPool(observer=metrics)`, and
  calls `await metrics.refresh(pool)` from its health check or a periodic task, since the gauges move only then.
