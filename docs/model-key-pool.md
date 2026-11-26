# Model key pool

A key pool runs a chat model on 1..n API keys for one provider, each inside an operator-set budget per billing
period. Every call starts on the pool's **active key**. That key stays active until it can't serve, and then the pool
moves it to the next key. The move is recorded in the pool's store, so every process sharing the pool moves with it.

Use a pool whenever a consumer runs model calls on API keys it budgets. One key is a normal pool, with the same
budgeting, states and errors as ten. Requirements and the reasoning behind them:
[model-key-pool-requirements.md](model-key-pool-requirements.md).

**Warnings:**

1. **Processes that share a pool must share one store.** The default store is in-memory, so each process budgets
   alone, and together they can spend several times a key's budget. Pass every process the same
   [`BudgetLedger`](#choosing-a-store), and call the model through its [async methods](#sync-or-async).
2. **The in-memory store forgets on restart.** A restarted script starts every key at zero spend for the period.
3. **A new secret under an old label starts that label at zero spend for the period.** The label's whole budget is
   available again. See [Replacing a key](#replacing-a-key).

## Contents

- [Building a pool](#building-a-pool)
- [How spend is counted](#how-spend-is-counted)
- [Rate limits](#rate-limits)
- [Choosing a store](#choosing-a-store)
- [Sync or async](#sync-or-async)
- [Reading `status()`](#reading-status)
- [Key states](#key-states)
- [Resetting and replacing a key](#resetting-and-replacing-a-key)
- [`NoKeyAvailableError`](#nokeyavailableerror)
- [Log lines](#log-lines)
- [Metrics](#metrics)

## Building a pool

Build the pool from the consumer's own secrets, then pass it to `create_chat_model` in place of `api_key`. 3tears
never reads a consumer's configuration, and ships no model prices, so register the model's input and output prices
first: from the consumer's model registry, or with an override as below.

```python
from decimal import Decimal

from threetears.models import (
    CapabilityOverride,
    KeyPool,
    KeySpec,
    Secret,
    create_chat_model,
    register_capability_override,
)

register_capability_override(  # USD per token, at the consumer's own prices
    model_id,
    CapabilityOverride(cost_per_input_token=Decimal("0.000003"), cost_per_output_token=Decimal("0.000015")),
)
pool = KeyPool(
    "anthropic",
    "chat",
    [
        KeySpec(Secret(settings.primary_key), label="primary", budget=Decimal("500")),
        KeySpec(Secret(settings.overflow_key), label="overflow", budget=Decimal("200")),
    ],
)
model = create_chat_model(model_id, key_pool=pool)
```

The pool's provider (`anthropic`, `openai` or `openrouter`) must be the one `create_chat_model` resolves for the
model: its `provider=`, or the capabilities registry's. Building a pool with no keys, a repeated label, a repeated
secret or a Claude subscription token (`sk-ant-oat…`) raises `ValueError`. Every process that builds a pool with the
same provider and name on one store shares its spend, state and active key.

`create_chat_model` refuses, before building anything:

- `ValueError` for both or neither of `api_key` and `key_pool`, a pool of another provider, or `key_pool` with
  `breaker`, `breaker_registry` or `unretried_key_errors`. Each key's breakers come from the pool's
  `breaker_registry`, which the pool reads key health from, and every key's model runs with `unretried_key_errors`
  so the pool sees a key's refusal the first time;
- `UnpricedModelError` for a model with no input or output price registered, or not registered at all: the pool
  can't budget its calls.

The pooled model is one model per key, built up front. `bind_tools`, `with_structured_output`, `with_retry` and
every other re-binding apply to each key's model, so a call runs the same binding whichever key serves it. Each
usage record carries the serving key's label as `credential_label`, which reaches spans as `llm.credential_label`.

Each `KeySpec` carries:

| Field | Meaning |
|---|---|
| `secret` | the credential, in a `Secret`, whose `str`, `repr` and every format show `<redacted>` |
| `label` | the key's name everywhere: logs, metrics, `status()`, errors. 1 to 32 letters, digits, `_`, `.` or `-`, unique in the pool |
| `budget` | a `Decimal` above zero: what the key may spend in one billing period |
| `currency` | `"USD"`, the only accepted value, since every model price is in USD. Any other is refused, not converted |
| `priority` | an `int`, lower preferred; `None` (the default) ranks after every set priority |
| `account` | the provider account the key's credit and rate limits belong to; `None` (the default) makes the key its own account |

**Priority.** Leave priorities equal (the default) and the pool stays on its current key until that key is forced
off, then moves on in configuration order, wrapping around. It doesn't move back early. Set distinct priorities when
one key should carry the traffic and another serve only while it can't. The pool then returns to the most-preferred
key as soon as that key can serve again.

**Account.** Give keys that share one provider account the same `account` tag. Credit is held per account, not per
key. With the tag, an exhausted key holds every key in its account at once. Without it, the pool fails over into the
same empty account, one failed call per key.

The pool's own settings, all keyword arguments to `KeyPool`:

| Setting | Default | Meaning |
|---|---|---|
| `anchor_day` | `1` | the day of the month each billing period starts on, 1 to 28, at 00:00 UTC |
| `min_headroom` | `Decimal("0.01")` | the fraction of a key's budget below which the pool moves off the key. It is read from settled spend alone, so a burst of open reservations never marks a key low |
| `rate_limit_exhaustion_wait` | 120 s | the longest wait a rate limit may ask for and still be one. A rate limit whose `retry-after` (or reset header) is further off counts as exhaustion until then: the call moves to the next key instead of waiting. See [Rate limits](#rate-limits) |
| `reservation_margin` | 10 min | how long past a call's deadline its reservation is kept before it counts as orphaned. A pooled model's call has no deadline, so this runs from when the call started: a pooled call still running after it, such as a long stream, is settled by the sweep at its full reservation and logged and counted as an orphan. Its own settle then corrects the spend, but the orphan line and count stay. Raise the margin above your longest call |
| `refresh_interval` | 5 s | how stale this process's copy of the pool's record may get before a call reads the store again |
| `sweep_interval` | 1 min | how often, at most, the pool settles orphaned reservations |
| `observer` | `None` | receives the pool's events, such as a [`KeyPoolMetrics`](#metrics). The pool counts them in `pool.counters` either way |
| `breaker_registry` | a new one per pool | where each key's circuit breakers live; the pool reads key health from it |
| `classifier`, `selection` | the provider's classifier, `StickySelection` | the seams that read a failed call and choose the active key |
| `clock`, `sleep` | UTC wall clock, `asyncio.sleep` | the pool's time source, and the wait before the pool retries a rate limit itself |

A period with anchor day 1 is a calendar month, named like `2026-11`. Any other anchor names the period by the date it
starts, like `2026-11-15`. Spend is kept per period, so a new period starts at zero with no job to run. A rollover
also clears `exhausted` and returns the pool to its most-preferred key.

## How spend is counted

The pool reserves before each call, and settles after it:

1. **Reserve.** The pool holds an estimate on the key: the prompt's estimated input tokens at the input price, plus
   the call's `max_tokens` (or the model's `max_output_tokens`) at the output price, plus any per-request price. The
   reservation succeeds only if `spent + reserved + estimate` fits in the budget. Concurrent calls can't pass the
   budget together, in one process or many.
2. **Settle.** After a successful call, the reservation is replaced by the cost of the usage the provider reported,
   priced by `cost_of_usage`, the function the usage tracker prices with. Cache reads and writes are at their own
   prices. When the provider reports no usage, the estimate stays as spend.
3. **On failure**, a call the provider can't have billed (a 4xx refusal before any answer, a connection that never
   reached it, or the key's breaker refusing it) is released. One it may have billed, such as a 5xx or a stream that
   failed partway, is settled at the full estimate. A failure after the provider reported usage, such as a structured
   answer that fails to parse, settles at the reported cost. A cancelled call, or a stream its caller stopped reading,
   settles at the reported cost where every model run reported usage, and otherwise at no less than the estimate.
4. **Orphans.** A reservation whose process crashed is settled at its full estimate once it expires (the call's
   deadline plus `reservation_margin`), and counted as orphaned.

Because reservations are estimates, a key can end a period slightly over its budget. The provider still enforces its
own credit limit, and the pool turns that refusal into the `exhausted` state.

A call that sets no `max_tokens` (on `create_chat_model`, `bind` or the call) for a model that declares no
`max_output_tokens` raises `NoOutputLimitError` before any network call: the pool can't bound its cost. An unpriced
model is refused earlier, by `create_chat_model`.

A caller with its own transport, such as a batch client, reserves with `await pool.reserve(amount)` and closes the
lease it gets back itself, reading a failure with `pool.classify(exc)`. A rate limit that caller retries on the same
key; `await pool.wait_to_retry(verdict, retries, label=lease.label)` waits and says whether to, for a provider whose
SDK doesn't retry.

## Rate limits

A rate limit is not a quota. A rate limit means retry, on the same key; a used-up quota means move to the next key.
Each provider answers both with a 429 (OpenRouter with a 402 or 403 for its budgets too), and the pool tells them
apart by the body:

| Provider | Exhaustion: moves to the next key | Rate limit: retried on the same key |
|---|---|---|
| anthropic | a 429 whose `error.details.error_code` is `enforced_spend_limit_reached` or `contract_spend_limit_reached`; a 402 `billing_error`; a 400 saying the credit balance is too low or a spend limit was reached | every other 429, `ramp_limit_exceeded` included |
| openai | a 429 whose `error.code` (or `type`) is `insufficient_quota`, `credit_balance_exhausted`, `organization_spend_limit_exceeded`, `project_spend_limit_exceeded`, `organization_usage_limit_exceeded` or `billing_hard_limit_reached` | every other 429: `rate_limit_exceeded`, `slow_down`, type `rate_limit_error` |
| openrouter | a 402 `openrouter_key_limit` or `openrouter_credits`; a 403 whose `limit_source` is a guardrail, workspace or key budget, held until its `resets_at` | a 429; a 402 `openrouter_in_flight_budget` that carries `Retry-After` |

A rate limit never moves the active key, and the pool records nothing about it: no key state, no move, no failover.

- **Anthropic and OpenAI.** The key's SDK retries the rate limit itself, as an `api_key=` model's does, with the
  caller's `max_retries` and honouring `retry-after`. The pool only stops it retrying a response that is the key's: a
  401, 402 or 403, a 429 whose body reports a quota, and a 429 whose wait passes `rate_limit_exhaustion_wait`. To
  tell, the pool's hook reads the body of a 429 and of no other response. A 429 whose body isn't the provider's JSON
  stays a rate limit.
- **OpenRouter.** Its SDK never retries a 429, so the pool retries the call on the same key, twice, each after the
  `retry-after` (or `X-RateLimit-Reset`) the provider gave, or a short backoff when it gave none. No wait is longer
  than `rate_limit_exhaustion_wait`. These retries are not failovers. Each one logs an INFO line naming the key, the
  retry and the wait ([Log lines](#log-lines)), so a slow call's waits are on record; nothing else is.

After those retries, the provider's rate-limit error reaches the caller unchanged, its reservation released.

**A long wait counts as exhaustion.** A rate limit asking a wait past `rate_limit_exhaustion_wait` (120 s by default)
is a cap, not a rate: the call moves to the next key, and the old key reads `exhausted` until the wait ends. This is a
guard against provider daily caps, not a case expected often.

**A request too large for the key.** OpenRouter's 402 `openrouter_credits` with `reason: weight_exceeds_budget` means
one request costs more than the key's whole budget. The call skips that key, as a call too large for a key's room
does, and the key stays `ok` and active.

**Breakers still count rate limits, but never move a key.** Each key's model keeps the per-model rate-limit breaker
an `api_key=` model has. When repeated rate limits open it, the call fails fast with `CircuitOpenError`, as it would
with `api_key=`. The pool doesn't move off the key or try another, since every key shares the same rates. Only the
credential breaker, which 5xx and connection failures open, makes a key `tripped`.

## Choosing a store

| Store | Use for | Survives |
|---|---|---|
| `InMemoryKeyPoolStore` (the default) | one process: a script, a CLI, a single worker | nothing; a restart starts at zero spend |
| `BudgetLedger` from `threetears.core.coordination` | several processes sharing a pool | a process restart and a NATS wipe |

`BudgetLedger` keeps spend and state in two coordination tables in L3, the database, with NATS KV (L2) as the
compare-and-swap fence in front of it. Every reserve and settle is written to the database before it returns.

```python
from threetears.core.coordination import BudgetLedger

pool = KeyPool("anthropic", "chat", keys, BudgetLedger(registry, purpose="model-key-pool"))
```

To run it:

1. Configure the `CollectionRegistry` with both an L2 client and an L3 pool. The ledger raises `ValueError` without
   either: without L2 two processes could admit the same room, and without L3 a broker restart loses every key's
   spend.
2. Apply the coordination migrations, through v004, which creates `coordination_ledger_accounts` and
   `coordination_documents`: `threetears.core.coordination.migrations.register(runner)`. A pod whose broker refuses
   DDL declares `LEDGER_TABLE_SCHEMAS` from `threetears.core.coordination.tables` in its data section instead.
3. Give every process that shares the pool the same provider, pool name and ledger `purpose`. A different purpose is
   a different ledger.

If the store fails, the call fails closed with `KeyPoolStoreError`, with the cause chained. It is logged at ERROR and
counted, and the pool never falls back to unbudgeted calls. The one exception is a settle that fails after the
provider answered: the caller still gets the answer, and the open reservation settles at its full amount when it
expires.

## Sync or async

With `BudgetLedger`, call the model through `ainvoke`, `astream` and `abatch`. The ledger's NATS and database clients
belong to the event loop that opened them, and only the async paths run the pool on the caller's loop.

The sync paths (`invoke`, `stream`, `batch`) run the pool on a private event loop in a daemon thread, whether or not
the calling thread has a loop of its own. That suits `InMemoryKeyPoolStore`, which guards itself with a thread lock.

## Reading `status()`

`await pool.status()` reads the store afresh and returns a `KeyPoolStatus` for a health endpoint. It carries no
secret and no fingerprint.

| Field | Meaning |
|---|---|
| `pool`, `provider` | the pool's name and provider |
| `period` | the current billing period's id |
| `active` | the active key's label. The pool's first use in a process, `status()` included, registers its keys and chooses one, so a first `status()` already shows it; `None` only when the store holds no active key |
| `keys` | one `KeyStatus` per key, in preference order |

Each `KeyStatus`:

| Field | Meaning |
|---|---|
| `label` | the key |
| `state` | `ok`, or the worst reason it can't serve (see [Key states](#key-states)) |
| `until` | when `state` ends, where that is known: the reset the provider named for an exhaustion, a breaker's recovery, the period's rollover |
| `spent` | money settled this period, orphans included |
| `reserved` | money held by calls in flight, plus any orphans not yet swept |
| `remaining` | `budget - spent - reserved`, never below zero |
| `budget` | the budget this process configures |
| `active` | whether it is the active key |
| `forbidden_models` | models the key is forbidden for, when it is forbidden per model |

`state` describes the key as a whole. A key forbidden for one model reads `ok` and lists that model in
`forbidden_models`.

The sweep that settles orphans runs on the next call, at most once a minute, so a crashed process's reservations sit
in `reserved` until then.

## Key states

| State | Means | Ends when | What to do |
|---|---|---|---|
| `ok` | the key can serve | | nothing |
| `exhausted` | the provider says the credit, spend limit or quota is used up, for the key or its whole account; or a rate limit asked a wait past `rate_limit_exhaustion_wait` | the reset the provider named passes (an OpenRouter budget's `resets_at`, a long wait's end), else the period rolls over; or the label gets a new secret, or an operator resets it | add credit or raise the provider's spend limit, then reset the key |
| `revoked` | the provider no longer accepts the key (401) | the label gets a new secret, or an operator resets it | replace the key; reset only if the revocation was a mistake that's now undone |
| `forbidden` | the key may not use the model, or any model (403) | the label gets a new secret, or an operator resets it | grant the key access at the provider, then reset it |
| `tripped` | the key's circuit breaker is open in this process, after repeated failures | the breaker closes | usually a provider outage; wait. Breaker state is per process and never stored |
| `no_room` | the key's budget less its settled spend is below `min_headroom`. Open reservations don't count: a key full of in-flight calls is skipped call by call and stays `ok` | the period rolls over, or an operator resets it | the key has spent its budget: raise the budget in configuration, redeploy, then reset the key |

`exhausted`, `revoked` and `forbidden` live in the store, so a state one process learns, every process sees. A rate
limit is no state at all ([Rate limits](#rate-limits)). A call too large for a key that still has room above
`min_headroom` skips that key for the one call, and the key stays `ok`.

## Resetting and replacing a key

**Reset** a key once its cause is fixed:

```python
await pool.reset("primary")
```

It clears `exhausted`, `revoked`, `forbidden` and `no_room` from the store, for every secret the
label has held, so every process sees the change. It also resets the key's breakers, in the calling process only;
other processes' breakers close on their own. Spend is untouched. With distinct priorities, a reset key the pool
prefers takes over again. An unknown label raises `ValueError`.

### Replacing a key

Put the new secret under the same label in configuration and redeploy. The pool tells secrets apart by a stable
fingerprint, so the label starts over: fresh state, and zero spend for the period. A process still running the old
secret keeps the old secret's state and spend until it restarts, so a rolling deploy never resets either secret's
spend mid-rollout. The store remembers the last four secrets per label.

**Changing a budget** needs only a configuration change and a redeploy. The budget is checked on every reservation,
so it applies at once. While processes disagree about a budget, each logs a WARNING naming the label and both amounts.

**Removing a key** is a configuration change too. If the stored active key is no longer configured, the pool moves
to its next key.

## `NoKeyAvailableError`

The pool raises `NoKeyAvailableError` when no key can serve a call, or every key that could has been tried. When there
is nothing to try, it raises before any network call. It is a `ModelProviderError`, so `is_provider_error()` is true
for it, and `friendly_api_error()` shows a user "No anthropic key can take this request right now. Please retry
later or contact an administrator."

Its message names the provider, every key and its state, and the earliest time any key could serve again:

```
anthropic call failed (no_key_available): no key in pool 'chat' can serve model '<model>' (primary: exhausted; overflow: tripped until 2026-11-26T12:00:30+00:00); the earliest a key could serve again: 2026-11-26T12:00:30+00:00
```

A key's state there is one of `exhausted`, `revoked`, `forbidden`, `tripped until T`,
`no room: R of B` (remaining of budget), or `tried`: this call already tried the key. The earliest time is `unknown`
when no key is known to come back, such as every key revoked.

For an `exhausted` key the earliest time is the reset the provider named, or else the pool's next period boundary,
when the pool clears the state. The boundary is not a promise from the provider: a used-up prepaid credit stays used
up until someone adds credit. A `no room` key comes back at the period boundary when the call is larger than its
budget less its settled spend; when only other calls' open reservations kept the call off it, it comes back as they
settle, and the error names no time for it.

Each key is tried at most once per call. A rate limit never raises `NoKeyAvailableError`: it is retried on the same
key, then the provider's own error reaches the caller ([Rate limits](#rate-limits)). Once a stream has handed the
caller a chunk, nothing is retried.

The same facts are on the error as attributes: `pool`, `model_id`, `keys` (each a `KeyStatus`) and `earliest`.

## Log lines

The pool logs under `threetears.models.key_pool.pool`. Each line names keys by label only, and carries the pool,
labels and reason as structured fields too.

| Level | Line | When |
|---|---|---|
| WARNING | `key pool 'chat' moved its active key from 'primary' to 'overflow': exhausted` | a move off a key that won't serve again this period without an operator: `exhausted`, `revoked`, `forbidden` |
| INFO | the same line, ending `tripped` or `low_room` | a move off a key that will come back on its own. A rate limit moves nothing and logs no move |
| INFO | the same line, ending `pull_back`, `period_rollover`, `not_configured` or `operator_reset` | a move back to a preferred key, the period's restart on the most-preferred key, a move off a stored active key this process doesn't configure, or a move to a key an operator reset |
| WARNING | a line naming the key and its new state | a key becomes unusable (`exhausted`, `revoked`, `forbidden`) with no key to move to, so no move line is logged |
| ERROR | `key pool 'chat': no key in pool 'chat' can serve … (3 more not logged in the last 60 s)` | a `NoKeyAvailableError`; at most one line per pool per minute, counting the ones it skipped |
| ERROR | `key pool 'chat': the store's reserve failed (KvError); the call fails closed` | a store failure, with the cause attached |
| WARNING | `key pool 'chat': settled 2 orphaned reservation(s) on key 'primary' at their full amount` | the sweep found reservations a crashed call never closed, or a call still running past `reservation_margin` |
| INFO | `key pool 'chat': key 'primary' was rate limited; retry 1 of 2 on the same key in 17.0 s` | the pool is about to wait out a rate limit its provider's SDK doesn't retry (OpenRouter's), then retry on the same key |
| WARNING | `key pool 'chat': the call on key 'primary' was served but its settlement failed; …` | logged by `threetears.models.pooled`: the answer was returned, and the reservation settles at its full amount when it expires |
| WARNING | `key pool 'chat': key 'primary' was last configured with a budget of 500 USD; this process configures 400` | processes disagree about a budget |
| INFO | `key pool 'chat': key 'primary' holds a new secret; it starts from fresh state and zero spend this period` | a label's secret changed |
| INFO | `key pool 'chat': an operator reset key 'primary'` | `pool.reset()` ran |

A move is logged once, by the process that made it. A process that loses the race adopts the winner's key and logs
nothing.

## Metrics

The pool's Prometheus metrics need the `prometheus` extra (`3tears-models[prometheus]`). Every label is a pool name, a
key label from configuration, or a value this package defines, so the number of series is bounded by configuration.

Build one `KeyPoolMetrics` per Prometheus registry and pass it to every pool it should count. Counters move as the
pool's events happen. Gauges move only when you refresh them, from a health check or a periodic task:

```python
from threetears.models.key_pool import KeyPoolMetrics, KeyPoolStatus

metrics = KeyPoolMetrics()  # prometheus_client's default registry; registry= for another
pool = KeyPool("anthropic", "chat", keys, ledger, observer=metrics)

async def health() -> KeyPoolStatus:
    return await metrics.refresh(pool)  # reads pool.status() and writes it into the gauges
```

`refresh` returns the same snapshot `status()` does, and starts the pool's counters at zero so a rate over them is
defined before their first event. If the store fails, it raises `KeyPoolStoreError` and the gauges keep the last
snapshot. Without the extra, building `KeyPoolMetrics` raises `ImportError`.

| Metric | Labels | Value |
|---|---|---|
| `threetears_key_pool_spent_usd` | `pool`, `key` | `spent` from `status()` |
| `threetears_key_pool_reserved_usd` | `pool`, `key` | `reserved` |
| `threetears_key_pool_remaining_usd` | `pool`, `key` | `remaining` |
| `threetears_key_pool_budget_usd` | `pool`, `key` | `budget`, as this process configures it |
| `threetears_key_pool_key_state` | `pool`, `key`, `state` | 1 for the key's [state](#key-states), 0 for each other: `ok`, `exhausted`, `revoked`, `forbidden`, `tripped`, `no_room` |
| `threetears_key_pool_key_active` | `pool`, `key` | 1 for the active key, else 0 |
| `threetears_key_pool_moves_total` | `pool`, `from`, `to`, `reason` | moves of the active key, by the reasons in [Log lines](#log-lines) |
| `threetears_key_pool_failovers_total` | `pool`, `reason` | failovers inside one call: the call ran again on another key after its key failed. `reason` is `exhausted`, `revoked`, `forbidden`, `tripped`, or `low_room` for a request too large for the key. A rate limit is never one |
| `threetears_key_pool_room_refusals_total` | `pool`, `key` | reservations the key refused for room, so the call went to another key |
| `threetears_key_pool_orphaned_reservations_total` | `pool`, `key` | reservations the sweep settled at their full amount |
| `threetears_key_pool_no_key_available_total` | `pool` | `NoKeyAvailableError` raises |
| `threetears_key_pool_store_failures_total` | `pool`, `operation` | store calls that failed, by store method |

The gauges are the first six rows; the rest are counters. A move changes the active key for every call, and a
failover is one call trying its next key, so the two counts needn't match.

Without the extra, `pool.counters` holds the same counts in memory: moves by from, to and reason; failovers by
reason; refusals for room and orphans by key; store failures by operation; and `NoKeyAvailableError` raises.
