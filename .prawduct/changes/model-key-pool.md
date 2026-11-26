---
schema: 1
type: feature
scope: model-key-pool
date: 2026-11-26
title: Model key pool
---
A chat model can run on a pool of 1..n API keys for one provider (anthropic, openai or openrouter), each inside an
operator-set budget per billing period. The pool stays on its active key until that key is exhausted, revoked,
forbidden, rate-limited, tripped or low on budget, then moves it, durably, for every process sharing the pool. One key
is a normal pool. Requirements: `docs/model-key-pool-requirements.md`; operator guide: `docs/model-key-pool.md`.

- `KeyPool` and `KeySpec` in `threetears.models`: a redacted `Secret`, a label, a USD budget, an optional priority and
  provider account per key. Sticky selection, with priority pull-back when priorities differ, behind a selection
  seam. Spend is reserved before each call and settled after it, so concurrent calls can't pass a key's budget.
- Per-provider classifiers read what a failure says about the key: exhausted, revoked, forbidden, rate-limited until
  a time, or not the key's. An account-scoped verdict reaches every key in that account. An unknown error never burns
  a key.
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
  `unretried_key_errors=`, so a 401, 402, 403 or 429 reaches the caller without the SDK first retrying it on the
  same key.
- `create_chat_model(model_id, key_pool=pool)` builds the pooled model. <!-- chunk 07: filled at integration -->

## Consumer note
Install changes in `3tears-models`:

- `anthropic`'s floor moves from 0.116 to 1.12. The 429-unretried mode is verified against 1.12. Upgrade a consumer
  that pins `anthropic` below it.
- `openai` (`>=3.26`) and `httpx` (`>=0.27`) are now direct dependencies of models. Both arrived transitively before;
  a consumer pinning either below those floors fails to resolve.

`threetears_llm_cost_usd_total`, and the tracker's `cost_usd`, now price cache reads and writes at the model's own
cache rates instead of the full input rate, and add the model's `cost_per_request` once per call. For cache-heavy
Anthropic traffic the counter drops, since cache reads cost far less than input. Traffic that writes the cache more
than it reads it can rise, where a cache-write price above the input rate is registered. Dashboards and alerts on
the counter will step at the upgrade; the new values are the correct ones. `UsageAccumulator` totals change the same
way.

The key pool itself is opt-in: no existing `create_chat_model(api_key=...)` call changes behaviour. A consumer adopting
`BudgetLedger` applies the coordination migrations through v004 (or declares `LEDGER_TABLE_SCHEMAS` where its pod
can't run DDL).
