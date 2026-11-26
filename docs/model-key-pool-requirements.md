# Model key pool: requirements

**Status:** Requirements, owner-reviewed 2026-11-26 (Q1 and Q2 ruled; Q3 takes its recommendation; Q4 superseded by the
owner's later ruling the same day that separates rate limiting from quota exhaustion, `.prawduct/owner-rulings.md`).
Every requirement that ruling changed carries an "Amended 2026-11-26" note.
**Package:** `3tears-models` (`threetears.models`), plus one durable store whose home is open question Q1.
**Source:** Hearsay's request (owner-approved 2026-11-25), as modified by the owner's three directions:
provider-agnostic, 1..n keys, sticky rotation. Where they disagree, the owner's directions win. §5 lists
every change made to Hearsay's request.
**Read against:** 3tears at `4a8d99dec`. Every claim about existing code cites a file under
`packages/models/src/threetears/models/` unless it names another path.

## Summary

A **key pool** is a set of 1..n credentials for **one** provider. Each credential has an operator label, a
budget for each billing period, and an optional priority. A chat model built from the pool runs every call on
the pool's **active key**. That key stays active until it can't serve: it is exhausted, revoked, forbidden,
tripped, or too low on budget. The pool then moves the active key to the next eligible key. The move is a state
change, recorded in a store that every process sharing the pool reads. Later calls start on the new key, so they
don't begin with a failure on the dead one. A rate limit is not one of these: it is retried on the same key, and
never moves it (amended 2026-11-26).

Spend is reserved before each call and settled after it, so concurrent calls can't take a key past its budget,
whether they run in one process or in several. Provider-specific behaviour sits behind a per-provider
classifier: how a provider says "out of credit", "revoked" or "rate-limited", and how its usage is priced. When
no key can serve a call, the pool raises one typed error that names every key's label and state. It never
returns `None`.

---

## 1. Purpose and scope

**Purpose.** Let a 3tears consumer run its model calls on several keys, with these properties:

- each key stays inside an operator-set budget;
- an exhausted or revoked key is moved off once, durably, not rediscovered by every call;
- a rate-limited call is retried on the same key, and a rate limit never moves a key (amended 2026-11-26);
- a consumer with one key gets the same budgeting, state and errors as a consumer with ten.

**In scope (v1):**

- Chat models built by `create_chat_model` for the providers that have a chat factory today: `anthropic`
  (HTTP API keys), `openai` and `openrouter` (`factory.py:359-365`).
- Every call shape those models take: `invoke`/`ainvoke`, `stream`/`astream`, `batch`/`abatch`, `bind_tools`,
  `with_structured_output`, and `bind(**structured_output_kwargs(...))`.

**Out of scope, with reasons:**

| Excluded | Why |
|---|---|
| Provider batch APIs (Anthropic Message Batches, OpenAI Batch) | `threetears.models` has no batch client today; nothing under `packages/models/src` submits or polls a batch. A batch is a different lifecycle from a chat call: submit, poll for up to a day, then settle each result. It is not a pool feature. v1 does make the reservation surface public (KP-E6), so a later batch client can reserve on and bill to one key without changing the pool. That is the seam Hearsay's batch case needs. |
| Reconciling with a provider's own usage or cost API | Not needed for correctness. The pool's budget is the pool's own ceiling; the provider still enforces the real credit limit, and the classifier turns that signal into the *exhausted* state (KP-C2), so drift corrects itself at the cost of one failed call. A reconciler would also need admin credentials, which differ by provider (Anthropic's cost reporting takes an admin key, not an API key). Worth building later, as an optional job, if drift proves material. |
| Claude subscription tokens (`sk-ant-oat…`, the Claude CLI route, `providers/anthropic.py:253`) | A subscription has no per-call price to budget, and its limits come back as free-text `ModelRateLimitError.resets` (`errors.py:117`), not as a status code with a retry-after. A pool of tokens could use the same sticky rotation later; v1 refuses them when a pool is built. |
| Embeddings, Whisper, image providers | `create_embedding_model` attaches no callbacks (`factory.py`, its docstring), so there is no usage to settle against. Add when embeddings gain usage tracking. |
| Amazon Bedrock, Vertex and other providers without a chat factory | There is no factory to pool. KP-A1 keeps the credential opaque, so a future SigV4 credential fits without changing the pool. Adding a provider means adding a classifier (KP-C1), not changing the core. |
| Fallback across providers or models | A pool is one provider's keys. Model fallback is a separate concern. |
| Reading secrets | The caller passes secrets in. 3tears never reads a consumer's configuration. |

## 2. What exists today, and the gaps

| Piece | Where | Covers | Gap for a pool |
|---|---|---|---|
| `create_chat_model(model_id, *, api_key, ...)` | `factory.py:196` | anthropic, openai, openrouter | Takes one key, fixed when the model is built. |
| Callbacks that survive `bind_tools` and `with_structured_output` | `factory.py:56`, `_CallbacksKeptBinding` | all three | A pool needs the same re-binding applied to each key's model. |
| Per-credential breaker | `circuit_breaker.py`, `CircuitBreakerRegistry.get(provider, credential=, model=)` | any provider | The default registry is module-private (`factory.py:53`), so a pool can't read a key's breaker state through the front door. Passing `breaker=` explicitly also turns off the per-model rate-limit breaker (`factory.py:183`). `check()` (`circuit_breaker.py:286`) moves OPEN to HALF_OPEN and admits a probe, so it can't double as a read-only eligibility test. The credential fingerprint uses a random key per registry (`circuit_breaker.py:726-738`), so it can't be shared across processes. |
| Error classification | `errors.py` | `friendly_api_error` handles Anthropic statuses; `is_rate_limit` covers any SDK 429 plus OpenRouter's mid-stream 429 | **No "out of credit" predicate exists.** `friendly_api_error` only shows the provider's own message for 400/402/403 (`errors.py:353`), which is how "credit balance is too low" reaches a user. Nothing reads `retry-after`, nothing recognises a revoked key as a key state, and OpenAI SDK errors fall through to the generic message. |
| Usage tracking | `tracking.py`: `UsageTracker`, `UsageTrackingCallback`, `UsageAuditSink`/`UsageCounterSink`; `usage.py`: `UsageAccumulator` | anything reporting LangChain usage | Cost is `input × in + output × out` (`tracking.py:1106`). Cache reads and writes are charged at the full input rate, though `ModelCapabilities` carries `cost_per_cache_read_token`/`_write_token` (`capabilities.py:253-254`), and `cost_per_request` is ignored. Sinks run fire-and-forget (`tracking.py:777`), so they can't be the settlement path. A failed call records nothing. |
| Prices | `capabilities.py` cost fields, filled by consumers (`registry_loader.py`); `price_lookup.py` sources | USD only | 3tears bakes in no prices, so a model with no registered price can't be budgeted. |
| Token estimate | `tracking._estimate_tokens` (`tracking.py:391`) | provider-agnostic | Private. A reservation needs a public estimate. |
| Shared, durable state | `threetears.core.coordination` (`DistributedCounter`, `WindowedCounter`), `threetears.nats.kv.KvBucketLike` (CAS `update`) | | `3tears-models` depends on neither core nor nats (`packages/models/pyproject.toml`). Every NATS KV bucket is memory-backed by rule, and durability belongs to L3 (`docs/design-durable-coordination.md`). |

## 3. Terms

- **Key.** One credential plus its config: `label`, `budget`, `priority`, `account`.
- **Account.** The provider-side scope a key's credit and rate limits belong to. Anthropic, for example, holds
  credit and rate limits per organisation (and per workspace where a spend limit is set), not per key.
- **Active key.** The key the pool sends new calls to.
- **Moving** the active key is durable: it is recorded in the store. **Skipping** a key affects one call only.
- **Reservation.** Budget held for one in-flight call. **Settlement** replaces it with the call's actual cost.
- **Room.** `budget − spent − reserved` for the current period.
- **Settled room.** `budget − spent` for the current period: room before open reservations. Within a period it only
  shrinks.
- **Eligible.** A key is eligible when its state is `ok`, its breaker isn't OPEN, and it has room.

## 4. Requirements

Every requirement must be testable without a network (KP-L1).

### A. Configuration

- **KP-A1.** A pool is built from a provider name, a pool name, and 1..n key specs. Each spec carries:
  - a secret (opaque to the pool, held in a type whose `str`/`repr` is redacted);
  - a `label` (unique within the pool, safe for metric labels: short, bounded charset);
  - a `budget`: an amount plus an ISO currency code;
  - an optional integer `priority` (lower means preferred);
  - an optional `account` tag (default: each key is its own account).

  Building with zero keys, duplicate labels, duplicate secrets, or a subscription token raises `ValueError`.
- **KP-A2.** One key is a normal pool. Nothing branches on `n == 1`. A one-key pool goes through the same
  reservation, state, metrics and errors as a larger one.
- **KP-A3.** The budget currency must match the currency of the prices the pool uses. Every price in
  `ModelCapabilities` is USD today, so a non-USD budget is refused at build time, not converted.
- **KP-A4.** Building a pooled model for a model with no input or output price registered raises a typed error.
  The pool can't budget a call it can't price, and treating that call as free would be a silent zero.
- **KP-A5.** Budgets come from configuration at build time, and the store holds only spend and state. The store
  records the budget each process last configured. A process configured with a different budget for the same
  `(pool, label)` logs a WARNING naming the label and both amounts.

### B. Secrecy

- **KP-B1.** No secret appears in any log line, metric label or value, span attribute, exception message,
  `repr`, store record, or `status()` output. The same holds for any LangChain callback's `serialized` payload
  the pool controls. Keys are named by `label` only.
- **KP-B2.** To detect that a label now holds a different secret, the store keeps a **stable** fingerprint: a
  truncated, non-reversible digest, the same in every process. The breaker registry's per-process random
  fingerprint can't serve this (`circuit_breaker.py:738`). The fingerprint is never logged or exported. When a
  label's fingerprint changes, that label starts from fresh state and zero spend for the period.
- **KP-B3.** One test plants a sentinel secret, drives every state and error path, and asserts the sentinel
  appears nowhere in captured logs, exception `str`/`repr`, metric exposition, span attributes or store
  contents.

### C. Key states and per-provider classification

- **KP-C1.** Provider knowledge lives behind one classifier per provider. The pool core contains no status codes,
  error types or header names. Given what an attempt raised, the classifier returns:
  - a **verdict**: `exhausted(until?)`, `revoked`, `forbidden`, `rate_limited(until?)`, `oversized`, or
    `not_key_related`. `rate_limited` is retried, never recorded; `oversized` (one request larger than the key may
    spend at all) skips the key for that call only, never recorded;
  - a **scope** for the verdict: this key, this key and model, or the key's whole account;
  - whether the provider **may have billed** the attempt.

  An exception the classifier doesn't recognise is `not_key_related`. An unknown error never burns a key.

  The same classifier also decides, from a raw response (status, headers, and a 429's body), whether the provider
  SDK may retry it on the same key (KP-H4), so the SDK hook and the verdict share one rule. *Amended 2026-11-26 (owner: a rate limit means retry; a quota means move to the next key).*
- **KP-C2.** v1 ships classifiers for `anthropic`, `openai` and `openrouter`. Each rule is pinned by a recorded
  response fixture. The intended mapping, to be confirmed against each provider's documentation when the
  classifier is built:

  | Signal | Verdict |
  |---|---|
  | Anthropic 402 `billing_error`; a 400 whose body says the credit balance is too low or that an organisation or workspace spend limit was reached; a 429 whose `error.details.error_code` is `enforced_spend_limit_reached` or `contract_spend_limit_reached` | `exhausted` (account scope) |
  | OpenAI 429 whose `error.code` (or `type`) is `insufficient_quota`, `credit_balance_exhausted`, `organization_spend_limit_exceeded`, `project_spend_limit_exceeded`, `organization_usage_limit_exceeded` or `billing_hard_limit_reached` | `exhausted` (account scope) |
  | OpenRouter 402 `openrouter_key_limit` (key scope), `openrouter_credits` without `weight_exceeds_budget` (account scope), or naming no limit (key scope) | `exhausted` |
  | OpenRouter 403 whose `limit_source` is `openrouter_guardrail_budget`, `openrouter_workspace_budget` or `openrouter_key_limit` | `exhausted` until its `resets_at`; key scope for `metadata.scope` `key_limit` or `api_key`, account scope otherwise |
  | OpenRouter 402 `openrouter_credits` with `reason: weight_exceeds_budget` | `oversized` (this call skips the key) |
  | 401, or the provider's authentication error type | `revoked` (key scope) |
  | 403, or the provider's permission error type, except as below | `forbidden` (key and model scope) |
  | Any other 429: Anthropic `rate_limit_error` with any other code (`ramp_limit_exceeded` included); OpenAI `rate_limit_exceeded`, `slow_down`, type `rate_limit_error`; OpenRouter 429, its mid-stream `(code: 429)` (`errors.py:375`), and its 402 `openrouter_in_flight_budget` carrying `Retry-After` | `rate_limited`, until the provider's `retry-after` (or its reset header: Anthropic's and OpenAI's of a used-up limit, OpenRouter's `X-RateLimit-Reset` in epoch milliseconds) |
  | OpenRouter 403 whose `error_type` is `content_policy_violation` or `refusal`, or that carries moderation metadata; 5xx, 529, timeouts, connection errors, other 4xx | `not_key_related` |

  Message matching is a last resort, used only where a provider gives no type or code for the case. Each such
  rule names its fixture. A Claude Code workspace's spend message is not matched. *Amended 2026-11-26 (owner: a rate limit means retry; a quota means move to the next key).* The exhaustion codes
  were checked against each provider's documentation that day.
- **KP-C3.** A rate limit is not held. A rate limit whose wait (`retry-after`, `retry-after-ms`, or a documented reset
  header) runs past a per-pool threshold (`rate_limit_exhaustion_wait`, default 120 s) counts as exhaustion: the
  call moves to the next key, and the old key is `exhausted` until the wait ends, on the whole key (a key's state is
  not kept per model). It guards against provider daily caps; the owner doubts the case is real, so it stays small.
  *Amended 2026-11-26 (owner: a rate limit means retry; a quota means move to the next key).* It replaces the rate-limit hold, its 30 s floor and its doubling, which had no case left.
- **KP-C4.** State lifetimes:

  | State | Ends when |
  |---|---|
  | `exhausted` | the reset the provider named passes (an OpenRouter budget's `resets_at`, a long rate-limit wait's end), else the period rolls over; or the label's fingerprint changes, or an operator resets it |
  | `revoked`, `forbidden` | the label's fingerprint changes, or an operator resets it |
  | breaker `tripped` | the breaker closes; this state is process-local and read from the breaker |

  `exhausted`, `revoked` and `forbidden` live in the shared store, so a fact one process learns about a key, every
  process sees. Breaker state stays process-local, as it is today. There is no `rate_limited` state. *Amended 2026-11-26 (owner: a rate limit means retry; a quota means move to the next key).*
- **KP-C5.** A verdict with account scope applies to every key sharing that `account`. Rate limits and credit
  shared across one Anthropic organisation are exactly this case; without it, the pool would fail over from
  one key to another key that shares the same exhausted credit. (Rate limits shared across an organisation no
  longer need this: a rate limit records nothing. Amended 2026-11-26.)
- **KP-C6.** The pool reads each key's credential breaker state without side effects; it doesn't read the
  per-model rate-limit breaker, because a rate limit never moves a key (KP-H4; amended 2026-11-26). It never
  calls `check()` to decide eligibility, because `check()` admits a probe. Per-key models keep both the credential breaker and the
  per-model rate-limit breaker that `create_chat_model` wires today. Meeting both rules needs a public way to
  give the factory a `CircuitBreakerRegistry` and to read state from it. Nothing outside `factory.py` may bind
  to `_DEFAULT_BREAKER_REGISTRY` (CLAUDE.md, leading underscores).

### D. Sticky rotation

- **KP-D1.** A pool has one active key per pool name, held in the store. Each process caches it and refreshes it
  whenever the store reports a change, or within 5 s otherwise. Selecting a key doesn't read every key's state
  on every call.
- **KP-D2.** A call starts on the active key. The active key **moves** only when it becomes ineligible for every
  model, judged on whole-key states alone:
  - its state turns `exhausted`, `revoked` or `forbidden` for every model; a rate limit never moves it
    (amended 2026-11-26);
  - its breaker is OPEN;
  - its settled room falls below `min_headroom` (default 1% of its budget). Open reservations are pessimistic
    estimates that settle within the call, so they never mark a key low for the period; a call they crowd off a key
    skips it, as in KP-D3 (amended 2026-11-26).

  A key `forbidden` for one model only is recorded, and every call for that model **skips** it, as in KP-D3, using
  the next eligible key without moving the active key. Calls for other models stay on it. The active key is shared
  by every model the pool serves: if one model's forbid moved it, the next call for another model would pull it
  back (KP-D5), so a pool serving both would move on every alternation, each move costing a room snapshot, a record
  write, a log line and a count, and breaking KP-G6's success path. With equal priorities it would instead move every
  model's traffic onto the next key's budget. *Amended 2026-11-26 (owner: a per-model forbid skips for the call and
  never moves or pulls back the active key).*
- **KP-D3.** A call whose reservation doesn't fit on a key whose settled room is still above `min_headroom` **skips**
  that key for this call only. The active key stays put, so one large call doesn't strand the room that small
  calls can still use. A call the provider refuses as larger than the key may spend at all (OpenRouter's
  `weight_exceeds_budget`) skips the key the same way, and nothing is marked exhausted. *Amended 2026-11-26 (owner: a rate limit means retry; a quota means move to the next key).*
- **KP-D4.** The next key is the first eligible key after the current one, ordered by `(priority, config
  order)` and wrapping around. Two processes moving the pointer at the same time agree: the move is a
  compare-and-set from the old key, and the loser adopts the winner's choice.
- **KP-D5.** **Returning to an earlier key.**
  - With **distinct priorities**, the pool pulls traffic back to the most-preferred eligible key as soon as it
    sees that key eligible again: a period rollover, an exhaustion's named reset passing, its breaker closing, an
    operator reset, or a fingerprint change (amended 2026-11-26: a rate limit holds nothing to pass). Priority means "use B only while A can't serve."
    Eligibility here is the key's whole-key state, as in KP-D2: a forbid for one model never moved the key, so it
    never pulls one back (amended 2026-11-26).
  - With **equal priorities**, which is the default, the pool doesn't pull back. It stays on the current key
    until that key is forced off, then moves on in order.
  - A **period rollover** always resets the active key to the most-preferred key (the first configured key when
    priorities are equal).
- **KP-D6.** Sticky is the only v1 policy, and the selection code is a seam that a second policy can plug into
  (see Q3). "Least spent" is not provided, for three reasons:
  - It spreads every call across every key, which gives up provider prompt-cache locality. Anthropic's cache
    is scoped to the organisation or workspace, so a cache entry written on one key's account is never read
    on another's.
  - It exhausts every key at the same moment, which leaves no headroom.
  - It makes every process touch every key's ledger, which multiplies store contention.

  Spreading load only helps a caller bound by rate limits whose keys sit in separate accounts. Sticky rotation
  doesn't rotate on a rate limit (amended 2026-11-26), so such a caller is Q3's case.
- **KP-D7.** Each move logs once:
  - at WARNING for `exhausted`, `revoked` and `forbidden`;
  - at INFO for low room and a breaker trip. A rate limit moves nothing and logs no move (amended 2026-11-26).

  The log line names the pool, both labels and the reason, and the move is counted (KP-K1).

### E. Budget: reservation and settlement

- **KP-E1.** Before each attempt, the pool reserves an estimate on the chosen key: estimated input tokens priced
  at the input rate, plus the call's `max_tokens` priced at the output rate. If `max_tokens` isn't set, the
  model's `max_output_tokens` from its capabilities is used. A call with neither is refused with a typed error
  rather than reserved at zero. The input estimate is a public, local function (no network); today's private
  `_estimate_tokens` is the starting point.
- **KP-E2.** The reservation is atomic and conditional in the store: it succeeds only if `spent + reserved +
  amount ≤ budget`. Concurrent calls, in one process or in many processes on one store, never together take a
  key past its budget. The test is N concurrent reservations against a budget that fits fewer than N.
- **KP-E3.** After a successful attempt, the pool settles the reservation to the actual cost of the usage the
  model reported. That cost includes cache reads and writes at `cost_per_cache_read_token` and
  `cost_per_cache_write_token`, each falling back to the input rate when unset, plus `cost_per_request`.
  Settlement runs inline before the call returns. It doesn't ride on the tracker's fire-and-forget sinks. When
  usage is `"unavailable"`, the reservation is kept as spend; it is never settled to zero.
- **KP-E4.** For a failed attempt, the classifier decides:
  - if the provider can't have billed (refused before any response, such as 401, 402, 429 or a connection
    failure), the reservation is released;
  - if it may have billed (for example, a stream that failed partway), the reservation is settled at its full
    amount.
- **KP-E5.** Every reservation carries an id and an expiry (default: the call's deadline plus a margin). A
  reservation left by a crashed process is settled at its full amount when it expires, and counted as
  orphaned. Counting it as spend is the conservative choice.
- **KP-E6.** Reserving and settling is public API: `reserve(estimate) → lease`, then `lease.settle(cost)` or
  `lease.release()`. Production code needs it: a non-LangChain caller, and a later batch client that reserves on
  one key and settles per result. It is not exposed only for tests.

### F. Billing period

- **KP-F1.** By default, a period is a UTC calendar month. An anchor day from 1 to 28 is configurable; the 28
  limit means no month ever lacks the anchor. Spend is keyed by `(pool, label, period id)`, so a rollover needs
  no job: the first reservation in a new period finds zero.
- **KP-F2.** A reservation settles into the period it was made in, even if the call ends after the rollover.
- **KP-F3.** The clock is injectable, following the `clock=` precedent in `CircuitBreaker`, so rollover is
  tested at the boundary without waiting.

### G. Shared store

- **KP-G1.** The store is a Protocol in `threetears.models` with plain types only, so an implementation outside
  the package can satisfy it structurally. It provides:
  - an atomic conditional reserve;
  - settle and release;
  - expiry sweeps for orphaned reservations;
  - compare-and-set on the active key and on key state;
  - operator reset;
  - a snapshot for `status()`.
- **KP-G2.** `threetears.models` ships an in-memory store. It is the production choice for a single-process
  consumer such as a script or CLI, and the store the unit tests use. It is not a test-only API.
- **KP-G3.** A durable store follows the family's shape: L3 (PostgreSQL or Yugabyte, through a collection or
  `DataStore`) holds the truth, and NATS KV, which is memory-only by rule, is at most a fence in front of it, as
  `WindowedCounter` uses it. Spend survives a process restart and a NATS wipe. Its home is Q1.
- **KP-G4.** Two pool instances on one store, in one process or two, see each other's spend, reservations,
  states and active key.
- **KP-G5.** If the store is unreachable, the call fails closed: a typed `KeyPoolStoreError` carrying the cause,
  logged and counted. The pool doesn't fall back to unbudgeted calls.
- **KP-G6.** The success path costs at most two store operations per call: reserve and settle. Other operations
  happen only on moves, state changes and sweeps. A call that skips a key forbidden for its model is on the success
  path: the skip reads the cached record and writes nothing (amended 2026-11-26).

### H. Failover inside one call

- **KP-H1.** On a key-moving verdict (KP-C2: `exhausted`, `revoked`, `forbidden`), the pool records the state,
  moves the active key if the failed key was the active one and can no longer serve any model (a forbid for one
  model moves nothing, KP-D2), and retries the call on the next eligible key,
  transparently to the caller. An `oversized` verdict retries on the next key and records nothing.
- **KP-H2.** Each key is tried at most once per call. A key skipped for room (KP-D3), or refused as `oversized`,
  counts as tried. Retries of a rate limit on the same key (KP-H4) are not tries of another key. *Amended 2026-11-26 (owner: a rate limit means retry; a quota means move to the next key).* It
  removes the exception that let a rate-limited key serve once more after the pool's wait.
- **KP-H3.** A `not_key_related` failure is not retried on another key. It propagates unchanged, with today's
  breaker behaviour, so 5xx and connection errors behave as they do now.
- **KP-H4.** A key-moving failure reaches the pool on its first occurrence. The SDK must not first sleep on it
  and retry on the same key (the Anthropic SDK retries every 429, honouring `retry-after` uncapped). A rate limit is
  the reverse: it is retried on the same key, honouring `retry-after`, and then reaches the caller unchanged, with
  its reservation released and nothing about the key changed.
  - Anthropic and OpenAI: the SDK retries it with the caller's `max_retries`. The pool's hook marks
    `x-should-retry: false` only on a 401, 402 or 403, a 429 whose body reports a quota, and a 429 whose wait passes
    the KP-C3 threshold. It reads the body of a 429 only, never a 2xx stream; a body that doesn't parse stays
    retriable.
  - OpenRouter, whose SDK never retries a 429: the pool retries the call on the same key twice, honouring
    `retry-after` up to the KP-C3 threshold, through its injected sleep and clock. These are not failovers.

  Retries of `not_key_related` failures on the same key keep today's count. Each key's per-model rate-limit breaker
  (KP-C6) still counts rate limits, as on an `api_key=` model, and an open one fast-fails the call with
  `CircuitOpenError`, which reaches the caller unchanged. It neither blocks the key in selection nor fails the call
  over to another key: every key shares the same rates. Only the credential breaker's trip makes a key `tripped`. *Amended 2026-11-26 (owner: a rate limit means retry; a quota means move to the next key).*
- **KP-H5.** Once any streamed output has reached the caller, the pool never retries. It records the key's
  state, settles per KP-E4, and raises the provider's error.

### I. Errors

- **KP-I1.** When no key is eligible, or every eligible key has been tried, the pool raises `NoKeyAvailableError`.
  It names each key's label and state (`exhausted`, `revoked`, `forbidden`, `tripped until T`, `no room: R of B`,
  `tried`) and the earliest time any key could serve again. A rate limit never raises it: the provider's own error
  reaches the caller (KP-H4; amended 2026-11-26). When there is nothing to try, it is raised
  before any network call.
- **KP-I2.** `NoKeyAvailableError` is a `ModelProviderError`, so `is_provider_error()` (`errors.py:420`) is true
  for it and `friendly_api_error()` gives a fixed message. Every raise is logged at ERROR and counted. The log may
  be deduplicated per pool to one line per 60 s, provided the line carries the count it suppressed; the counter
  never is.
- **KP-I3.** The new error types carry no secret (KP-B1). The CLAUDE.md exception rules apply throughout: a
  broad handler re-raises or calls `contain()`, and there are no soft-fail paths.

### J. Integration

- **KP-J1.** The entry point stays `create_chat_model(model_id, key_pool=pool, purpose=…, ...)`, which takes
  exactly one of `api_key` or `key_pool`; passing both or neither raises `ValueError`. That matches the
  factory's "one public entry point per model class" (`factory.py`, module docstring). The pool's provider must
  match the provider resolved for `model_id`.
- **KP-J2.** The model it returns behaves as `create_chat_model`'s model does today. `bind_tools`,
  `with_structured_output`, `bind(**structured_output_kwargs(...))`, `with_config` and `with_retry` all return a
  pooled runnable that applies the same re-binding to whichever key's model serves the call. The tracker and
  breaker callbacks run on every path, as `_CallbacksKeptBinding` guarantees today.
- **KP-J3.** Each attempt fires the callbacks of the key it ran on:
  - the usage tracker records successes;
  - the breaker callback is that key's breaker;
  - callbacks the caller adds (such as `UsageAccumulator`, or callbacks in invocation config) see each attempt's
    start and its end or error.
- **KP-J4.** `UsageRecord` gains an optional `credential_label`. It flows to spans and sinks, so an audit sink
  can attribute spend per key, and it is never a label on the locked `threetears_llm_*` Prometheus instruments.
  `usage_scope` attribution is unchanged.
- **KP-J5.** The pool's settlement and the tracker's `cost_usd` use one cost function (see Q2).

### K. Observability

- **KP-K1.** Per `(pool, key label)`:
  - gauges: period spend, reserved, remaining, budget, state, and whether the key is active;
  - counters: moves by `from`, `to` and `reason`; failovers inside a call by `reason`; calls refused for room;
    orphaned reservations. No state, reason or failover names a rate limit: a rate limit is retried, not counted as
    a key event (amended 2026-11-26);
  - per pool: a counter of `NoKeyAvailableError` raises.

  Labels are bounded by configuration. The metrics sit behind the existing `prometheus` extra.
- **KP-K2.** `pool.status()` returns a snapshot for health endpoints: each key's label, state, `until`, spend,
  reserved, remaining, and the active key. It contains no secrets. (Amended 2026-11-26: no per-model rate-limit
  list, since a rate limit is no state.)

### L. Tests (no network)

- **KP-L1.** Unit tests in `packages/models/tests/unit`, using the in-memory store, an injected clock, and fake
  chat models that raise recorded provider errors. Every fake declares its parity
  (`tests/enforcement/test_fake_protocol_parity.py`). Coverage:
  - sticky selection;
  - priority pull-back after each trigger in KP-D5, and no pull-back when priorities are equal;
  - skip versus move (KP-D3);
  - each classifier rule, against its fixture;
  - account-scoped verdicts;
  - failover for each verdict; at most once per key; no retry after partial stream output;
  - a rate limit retried on the same key (by the SDK, or twice by the pool for OpenRouter) and then raised, with no
    state or move; the long-wait threshold; the hook reading a 429's body only (amended 2026-11-26);
  - concurrent reservations never exceeding the budget;
  - settlement including cache reads and writes; reservation release versus full settlement on failure;
  - orphan expiry; period rollover and the anchor day;
  - a one-key pool going through every path above;
  - `NoKeyAvailableError` contents, its log line and its counter;
  - store-down failing closed;
  - secrecy (KP-B3);
  - `bind_tools` and structured output on a pooled model.
- **KP-L2.** The durable store's tests run in the integration suite (`./scripts/test-integration.sh`):
  - spend surviving a restart (a new pool on the same store);
  - two pools on one store sharing spend, state and the active key;
  - concurrent reservations from two processes.

### M. Operator docs

- **KP-M1.** A page in `docs/` covering:
  - building a pool from the consumer's own secrets;
  - choosing `account`, priority and `min_headroom`;
  - what each state means and what to do about it;
  - resetting a key, and replacing a key under the same label;
  - reading the metrics and `status()`.

## 5. Changes from Hearsay's request

| Hearsay asked | This document | Why |
|---|---|---|
| Anthropic key pooling | Provider-agnostic, with per-provider classifiers; v1 covers anthropic, openai and openrouter | Owner's direction |
| N keys | 1..n keys, with one key not treated as a special case | Owner's direction |
| Per-call selection, "least spent" by default, round-robin or priority selectable | Sticky active key that moves durably, priority pull-back, no least-spent policy (KP-D) | Owner's direction. Least-spent costs cache locality and synchronises exhaustion (KP-D6) |
| Monthly credit in USD | A budget in a currency, refused if it doesn't match the price currency (USD today) | Owner's "in a currency"; 3tears prices are USD only |
| `errors.py` "already classifies credit balance too low" | It doesn't: it only shows the provider's message. The classifier is new work | Checked against `errors.py:353` |
| "Credit balance too low" makes the key exhausted | Also covers 402, spend-limit 400s, the spend-cap 429, OpenAI `insufficient_quota` and OpenRouter 402, and spreads to every key in the same `account` | Credit and limits are per provider account, not per key; without this the pool fails over into the same empty account |
| 401/403 make the key revoked until the pool is rebuilt | 401 means revoked; 403 means forbidden for that key and model. Both last until the label's secret changes or an operator resets the key, not until a rebuild | State is shared and durable, so a rebuild in one process clears nothing. A 403 is usually a permission on a resource, not a dead key |
| 429: hold for retry-after | A rate limit is retried on the same key and never held or moved; only a used-up quota (told apart by the body) moves, and a wait past 120 s counts as one (KP-C3, KP-H4) | Owner, 2026-11-26: "rate limit means retry. Quota exhaustion means move to next key" |
| 5xx and connection errors keep SDK and breaker behaviour | Kept, and so does a rate limit; the SDK must not retry a key-moving error (401-403, a quota 429) on the same key first (KP-H4) | Otherwise every call sleeps on a used-up key before failing over |
| Short keyed fingerprint "like the breaker registry's" | A stable fingerprint, never exported | The breaker's fingerprint is random per process (`circuit_breaker.py:738`) and can't be shared |
| Store "e.g. DataStore/PostgreSQL-backed" | Protocol in models; durable store on L3 with an optional L2 fence | `docs/design-durable-coordination.md`: NATS KV is memory-only, L3 is durable |
| Settle to "the actual usage the tracker records" | Settle inline from reported usage, with one shared cost function that prices cache reads and writes | Tracker sinks are fire-and-forget (`tracking.py:777`), and its cost ignores cache prices (`tracking.py:1106`) |
| Message Batches "valuable" | Out of v1; the public reserve/settle lease (KP-E6) is the seam | No batch client exists in 3tears; a batch's lifecycle is not a chat call's |
| Reconciliation not required | Agreed, and not recommended for v1 | §1 |

## 6. Open questions for the owner

- **Q1. Where does the durable store live?** `3tears-models` depends on neither `3tears` (core) nor
  `3tears-nats`. There are two options:
  - (a) a generic budget-ledger primitive in `threetears.core.coordination`, beside `WindowedCounter`, that meets
    the models Protocol structurally and needs no dependency on models;
  - (b) a new `3tears-models` extra that depends on core.

  Recommendation: (a). It is reusable beyond models and keeps the models base dependencies unchanged.
  **Ruled 2026-11-26: (a), a core primitive.**
- **Q2. Should the tracker's `cost_usd` start pricing cache reads and writes too (KP-J5)?** It is correct, but it
  changes the values behind `threetears_llm_cost_usd_total` on existing dashboards: lower for cache-heavy
  Anthropic traffic. Recommendation: yes, in the same release, with a release note.
  **Ruled 2026-11-26: yes, same release, with a release note.**
- **Q3. Should a second policy, "spread", ship in v1?** "Spread" means round-robin over eligible keys, weighted by
  room. Recommendation: no, until a consumer shows it is bound by rate limits across separate accounts.
  **Not separately ruled; v1 follows the recommendation (no "spread").**
- **Q4. When every key is only rate-limited, should a call wait for the earliest `until`, up to a cap, instead of
  raising at once?** A one-key pool otherwise loses the SDK's current retry-after wait (KP-H4).
  Recommendation: yes, with a per-pool cap that defaults to 0 (raise at once).
  **Ruled 2026-11-26: yes, a per-pool cap defaulting to 0.** **Superseded 2026-11-26** by the owner's later ruling
  that a rate limit means retry: a rate limit no longer takes a key out, so no call is left waiting on held keys,
  and the SDK's own `retry-after` wait is kept (KP-H4). The cap is removed.

The change is an API addition to the family, so it ships in a MINOR bump
(`tests/enforcement/test_api_growth_requires_a_minor_bump.py`), with every `3tears*` package moved in lockstep.
