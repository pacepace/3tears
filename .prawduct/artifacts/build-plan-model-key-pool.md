---
artifact: build-plan
version: 2
scope: model-key-pool
branch: feat/model-key-pool
base: develop
working-branches: []
last_validated: 2026-11-26
depends_on:
  - artifact: api-contract
  - artifact: architecture
  - artifact: security-model
  - artifact: observability-strategy
governed_by:
  - artifact: project-preferences
    dispositions:
      - "leading underscores are a stability contract (no test binds to a private name; no public API only for a test) → conforms: the lease API (KP-E6) and the in-memory store (KP-G2) have production callers; the breaker registry seam (KP-C6) replaces the need to touch `_DEFAULT_BREAKER_REGISTRY`"
      - "a failure reaches the caller or is recorded (docs/exceptions.md) → conforms: KP-G5 fails closed, KP-I1 raises, unknown errors are not_key_related and propagate"
      - "test fakes declare parity → conforms: every fake chat model and fake store carries `# parity-with:` or subclasses"
      - "lockstep family versions; API growth ships in a MINOR → conforms: no version bump in this work (work PRs never bump); the release carrying it is a minor"
      - "delegation and delegate verification (owner 2026-11-22) → conforms: see partition"
      - "inner-loop verification on testament (owner 2026-11-22) → conforms"
partition: "wave 1 delegated, 4 delegates in isolated worktrees (01 cost, 02 breaker seam, 03 classifiers, 04 store protocol), main session integrates; 05 (core ledger) delegated after 04 lands, in parallel with 06 built by the main session; 07 and 08 serial in the main session, since both edit the pooled model and factory"
---

# Build plan: model key pool

Requirements: `docs/model-key-pool-requirements.md` (owner-reviewed 2026-11-26). Requirement ids (KP-xx) below are that
document's. This plan says how to build it, and in what order.

## Goals

**Response taken:** Proceed, citing it. The requirements doc states what the pool is for, what success is and what is
out of scope, and the owner ruled its open questions on 2026-11-26 (`.prawduct/owner-rulings.md`).

**Product:** Hearsay, and any 3tears consumer, runs its model calls on 1..n keys for one provider, each inside an
operator-set budget. A dead or used-up key is moved off once, durably, for every process sharing the pool; a rate
limit is retried on the same key and moves nothing (owner, 2026-11-26). The
owner's three directions shape it: provider-agnostic, 1..n keys with no special case for one, sticky rotation.

**Architecture:** what must hold:
- `threetears.models` keeps its base dependencies. It depends on neither core nor nats. The durable store is a core
  primitive that satisfies a models Protocol structurally (Q1).
- The pool core holds no provider knowledge (KP-C1).
- The success path costs at most two store operations (KP-G6).
- `create_chat_model` stays the one entry point (KP-J1).

What must stay possible:
- a "spread" policy plugging into the selection seam (Q3);
- a batch client reserving on a lease (KP-E6);
- more providers, through a new classifier only.

**Constraints:**
- No network in unit tests (KP-L1).
- No secret anywhere observable (KP-B).
- CLAUDE.md's exception, underscore, fake-parity and attribution rules.
- No version bump on this branch.

**Tradeoffs accepted:**
- Reservations are estimates, so a key can end a period slightly over budget once usage settles above the
  estimate. Settlement records the truth, and the next reservation sees it. Revisit if overshoot is material.
- The core ledger requires an L2 fence, and it persists to L3 synchronously on every reserve and settle. That costs a
  database write per call, which buys a NATS wipe that loses no spend. Revisit if the write shows up in latency.

**Decisions made before building (inferred; flag at the boundary):**
- [DECISION: money is `decimal.Decimal`, stored as a string | budgets are compared by value, and float error near the
  budget line would let a reservation through that the sum says doesn't fit | owner can veto]
- [DECISION: the store Protocol is written in generic ledger terms, not pool terms, so core can satisfy it without
  pool vocabulary. It has two parts:
  - **accounts:** `reserve`, `settle`, `release`, `sweep_expired` and `snapshot`, keyed by an opaque account string the
    pool builds from `(pool, label, period)`;
  - **records:** `read_record` and `compare_and_set_record`, a versioned JSON document the pool keeps its active key,
    key states, fingerprints and configured budgets in.

  Why: Q1's "meets the models Protocol structurally" needs a Protocol core can implement without models' types. Plain
  types only (str, Decimal, dict, int, float, bool). | owner can veto]
- [DECISION: an account row holds its open reservations as a map in the row, and is compare-and-swapped as one unit.
  So "fits under the budget" is checked atomically with no second table. Open reservations are bounded by the calls in
  flight on one key. | owner can veto]
- [DECISION: the core ledger uses a new coordination table `coordination_ledger_accounts`, with a synchronous L3
  write policy and the L2 order columns. Records go in a second table, `coordination_documents`. Both come from one
  new migration in the coordination migrations package. The ledger refuses a registry with no L2
  (`require_l2_fence`), as the exactly-once primitives do, because the compare-and-swap is the guarantee. | owner can
  veto]
- [DECISION: the work ships as two PRs from this branch. Part 1 is chunks 01-05, the foundations, each useful on its
  own. Part 2 is chunks 06-08, the pool itself. Each part ends in its own `cumulative` review. | Why: one review over
  the whole diff spans several risk surfaces and well over 12 judgeable files, which the roster rule says one pass
  can't hold | owner can veto] *Superseded mid-build by one PR; see chunk 06's integration line below. The review-size
  concern stays: the boundary review runs the three-reviewer roster, scoped by risk surface.*

**Decisions made mid-build:** *(appended while building, one line each)*
- Chunk 01: `UsageAccumulator` prices through `cost_of_usage` too, per call, so a turn's total agrees with the tracker
  (settled by chunk 01's "one place usage is priced"; Critic rev-20261126T015246Z-257c15f6).
- Chunk 01: the text-level token estimate stays private, and the price Protocol is private `_Rates`. Each had no
  production caller outside its module (CLAUDE.md, no public API only for a test).
- Chunks 01 and 04: `test_public_api_has_a_caller` is red at these chunk boundaries for the names chunk 06 calls:
  `UnpricedModelError` (the KP-A4 build refusal), `estimate_input_tokens` (the KP-E1 reservation estimate), and chunk
  04's store and period names. Chunk 06 must leave it green, or ledger a name with a reason. That is the fix for
  blocking findings rev-20261126T020832Z-d8caf915 and rev-20261126T021046Z-3fb55497 R-1 (the same red, chunk 04's
  names added), which ride chunk 06's commit; one verify pass after chunk 06 resolves both.
- Chunk 06 delegated (opus, isolated worktree) instead of built by the main session: the main session's context was
  spent integrating wave 1, and a fresh builder with the merged APIs in its brief builds it better. The main session
  still integrates and governs it. Partition otherwise unchanged.
- Chunk 03: `KeyErrorClassifier`, `VerdictScope` and `classifier_for` join the caller-test red; chunk 06's pool is
  their caller (rev-20261126T022714Z-c4a71301 R-1 rides chunk 06 with the earlier two).
- Chunk 03: `httpx` declared as a base dependency of models; `classify.py` imports it on the base path (same review, R-2).
- Chunk 03: `providers/_langchain_anthropic_internals.py` overrides langchain-anthropic's private `_client_params`
  through `super()`, the one route to hand the SDK client a `Middleware`. Ruff's SLF001 does not flag `super()._x`, so
  `_underscore_exemptions.txt` would carry a stale line; the module docstring records the override and the version it
  is verified against.
- Chunk 06 integrated (merge 251d99ed0). It gives callers to every earlier red name but `estimate_input_tokens`.
  The red now holds `KeyPool`, `KeyPoolLease`, `reservation_estimate` and `estimate_input_tokens`. Chunk 07's
  pooled model is their production caller (it takes the pool, reserves a lease per attempt and prices the estimate),
  so the red, and the open caller findings, ride chunk 07. That retires the two-PR split above: Part 1 alone would
  ship public names with no caller and fail CI's caller gate. The branch ships as one PR, with one `cumulative` review
  at chunk 08, which also matches the maintainer's preference for fewer, larger PRs (main session, 2026-11-26). `ReservationAlreadyOpen` and `ReservationSnapshot` have no
  caller here (third-party stores raise the one, and snapshot readers read the other), so they are ledgered.
- Chunk 07 delegated (opus, isolated worktree) from the fix commit after 251d99ed0, for the same reason as chunk 06.
  The main session integrates it, records the suite, then runs one `verify-resolutions` pass over chunks 06 and 07 and
  the fix commit: the open caller blocker pins every other inner review to the whole branch.
- Chunk 08's consumer note must say: anthropic's floor moves from 0.116 to 1.12, and openai and httpx are now direct
  dependencies of models.
- Chunk 08: `KeyPoolMetrics` gauges refresh by an explicit `await metrics.refresh(pool)`, which returns the snapshot, not
  by a scrape-time callback: `status()` is a coroutine over the store, a scrape is synchronous on the `/metrics` thread
  (deadlock where that is the pool's loop), and a store failure must raise to a caller, not vanish in a scrape.
- Chunk 08: `KeyPoolObserver` has no failover event, so KP-K1's failovers-by-reason counter is not built; chunk 07's
  integration adds the event to the Protocol, `KeyPoolCounters` and `KeyPoolMetrics`.
- Chunk 08: store failures are exported too (`operation` label, bounded by the store's methods), since the Protocol
  reports them; key state is a 0/1 gauge per state (`TRIED` excluded: never in a snapshot).
- Chunk 08: `KeyPoolMetrics` without the extra raises `ImportError` naming it (an explicit build, as the claude-cli
  and voyageai extras do); instruments are built once per registry through `BuildOnce`, current only while a weak
  reference to the registry lives, because a dead registry's id is reused.
- Chunk 08: `KeyPoolMetrics` has no src caller (the pool only reports to an observer it is handed), so
  `_public_api_consumers.txt` names Hearsay. Owner to confirm at the boundary, as for `BudgetLedger`.
- Chunk 08: the failover event is built (it retires the "no failover event" line above): `KeyPoolObserver.on_failover`, `KeyPool.record_failover`, and `threetears_key_pool_failovers_total{pool, reason}`. Its reason is `MoveReason`, not `VerdictKind`, because it can name `tripped` (a breaker refusing after the key was chosen) and it shares the moves counter's axis.
- Chunk 08: a failover is counted when the call's next attempt runs on another key. A retry on the same key after its hold, a refusal that ends in `NoKeyAvailableError`, and a partial-stream failure are not failovers. `refresh` starts the five failover reasons and each store method at zero. *Rate-limited reason superseded 2026-11-26; see the rate-limit lines below.*
- Chunk 08 (operator check): a move onto a key an operator just reset carries `MoveReason.OPERATOR_RESET`, at INFO. A key made exhausted, revoked or forbidden with no move off it is logged at WARNING. The `NoKeyAvailableError` dedup suppresses only a line whose key states repeat the last logged line; the suppressed count is kept.
- Chunk 08: the KP-B3 sweep (`test_pooled_chat_secrecy.py`) found no leak. It found a served call whose settle failed raising `KeyPoolStoreError`, because the `finally` settled again. Such a lease now stays open for the sweep, and the answer is returned.
- Chunk 05: the ledger tables live in `LEDGER_TABLE_SCHEMAS`, not `COORDINATION_TABLE_SCHEMAS`, because v002 alters
  every table in that tuple and would fail where it runs before v004 creates them. Money and JSON are TEXT: no float
  before the budget compare, and JSONB would reorder the reservations the sweep reports in order.
- Chunk 05: the ledger refuses a registry with no L3 (main session, "do it right"). Spend on L2 alone is lost to a
  broker restart, and no ledger deployment runs without a database, unlike identity-edge's throttles.
- Chunk 07: the pooled model (`threetears.models.pooled.PooledChatModel`) holds every key's model built up front, and a
  re-binding (any method returning a runnable, plus `bind`, `with_config`, `with_retry`, `with_types`) is applied to
  each, so `with_retry` retries inside one key's attempt, as it does on an `api_key=` model.
- Chunk 07: the sync paths drive the async pool on one private daemon-thread event loop. A store bound to one event
  loop (the core ledger's NATS and database clients) must be driven through the async paths; the docstring says so.
- Chunk 07: Q4 over KP-H2: a key refused for a rate limit stays eligible for that call once more, so the pool's wait
  can serve it after its hold (a one-key pool keeps the SDK's retry-after wait); with the default cap of 0 the call
  raises at once. Every other verdict marks the key tried. [DECISION | owner can veto] *Superseded 2026-11-26: the owner
  separated rate limiting from quota exhaustion, so the exception and the Q4 wait are removed (lines below).*
- Chunk 07: settlement uses only provider-reported usage; "estimated" or "unavailable" usage keeps the reservation as
  spend. A not-key-related failure after reported usage (a structured answer that fails to parse) settles the
  reported cost. A settle that fails after a served call is logged and the answer returned: the open reservation
  settles in full at expiry. A breaker that refuses after the pool chose its key releases the lease and counts as tried.
- Chunk 07: a reservation carries no deadline (LangChain calls have none); it expires `reservation_margin` after the
  call starts, and a late settle is accepted. Any chunk handed to a stream's caller, an empty first one included,
  counts as output for KP-H5. `breaker`, `breaker_registry` and `unretried_key_errors` are refused with `key_pool`.
- Chunk 05: `BudgetLedger` has no caller in this repo's src (models can't import core), so
  `_public_api_consumers.txt` names Hearsay, whose wiring follows the release. Owner to confirm at the boundary.
- Rate-limit change (owner, 2026-11-26): a rate limit is no key state and no move reason. Removed outright, no shims:
  the hold with its floor and doubling, `rate_limited until T`, `MoveReason.RATE_LIMITED`, `KeyState.RATE_LIMITED`,
  `KeyStatus.rate_limited_models`, `max_rate_limit_wait` and the chunk 07 one-more-attempt exception.
- Rate-limit change: the classifier keeps a `rate_limited` verdict (the pool needs it to retry), which no longer moves
  the key. `Verdict.moves_key` is exhausted, revoked or forbidden; `tries_another_key` adds `oversized`.
- Rate-limit change: the SDK hook's rule is the classifier's `refuses_retry(status, headers, body, now, exhaustion_wait)`,
  reached through `KeyPool.refuses_retry`, which the factory hands each key's model as `unretried_key_errors`. The
  provider factories' `unretried_key_errors` changes from a bool to that rule (`None` = the SDK's own policy):
  injection, because a provider module importing `key_pool.classify` closes an import cycle (classify imports the
  providers' names). `KEY_MOVING_STATUSES` is gone.
- Rate-limit change: the classifiers share their two halves by composition (module-level per-provider readers passed
  to `_refuses_retry` and `_long_wait_exhausts`), not a base class with private hooks, which the underscore contract
  forbids overriding.
- Rate-limit change: the long-wait reading lives in the classifier (`classify(exhaustion_wait=)`), so the hook and the
  verdict compare the same `until`. A long wait from a per-model (`KEY_AND_MODEL`) rate limit is held on the whole
  key, since key state is not kept per model. A wait of exactly the threshold is still a rate limit.
- Rate-limit change: an exhausted record is `{"period", "until"}`. With a provider-named `until` it holds until then,
  across a rollover; without one, to the period's end as before. The record layout stays 1: nothing was released.
- Rate-limit change: OpenRouter's pool-side retries are `KeyPool.wait_to_retry(verdict, retries)`: two, through the
  pool's sleep, waiting the provider's `until` capped at the threshold, else 0.5 s then 1 s (the Stainless SDKs'
  starting backoff). Whether to retry is the classifier's `sdk_retries_rate_limits`, so the core names no provider.
- Rate-limit change: an `oversized` refusal (OpenRouter `weight_exceeds_budget`) counts as a `low_room` failover: it
  is a room refusal the provider reported, on the failover counter's existing axis. The failover reasons are
  exhausted, revoked, forbidden, tripped, low_room.
- Rate-limit change (main session, from the owner's ruling): each key's per-model rate-limit breaker still counts rate
  limits (KP-C6), but the pool no longer reads it in selection, and its `CircuitOpenError` reaches the caller with no
  failover. A rate limit never moves the key; only the credential breaker's trip makes a key `tripped`.
- Rate-limit change: `_Call.abandoned`'s docstring now says what the code does: a cancelled call settles at the
  reported cost when every model run reported usage, else at no less than its reservation.

**Level:** Medium. The pool logic is well specified. Two foreign surfaces carry the risk, and their verify-api steps
are what raise the level:
- each SDK's exception shapes and retry hooks (chunk 03);
- `BaseCollection.l2_cas_mutate` with a synchronous policy (chunk 05).

## Verification strategy

Unit tests use the in-memory store, an injected clock, and fake chat models that raise the classifiers' recorded
provider errors. The core ledger is proven in the integration suite on the testament pool:
- spend survives a restart;
- two pools share state;
- concurrent reservations from two processes.

After chunk 07, a script in the scratchpad builds a two-key pool against a fake transport. It drives one key to
exhausted, and its `status()` and log lines are read as an operator would read them.

## Build chunks

*The rate-limit parts of chunks 03, 04, 06 and 07 below (the 429-unretried mode, holds with a floor and doubling, the
Q4 wait) were replaced on 2026-11-26 by the owner's ruling that a rate limit means retry; the decisions above say
what replaced them. The chunks are kept as built.*

### Chunk 01: One cost function, and a public token estimate

- **Description:** `cost_of_usage(capabilities, usage)` prices input, output, cache reads and cache writes, plus
  `cost_per_request`, returning a `Decimal` (KP-E3, KP-J5). Cache reads and writes fall back to the input rate when
  unset. The tracker's `cost_usd` moves onto it (Q2, ruled: same release, with a release note). A public, local
  `estimate_input_tokens(messages)` is promoted from `tracking._estimate_tokens`, which production now needs (KP-E1).
- **Depends on:** none
- **Deliverables:** the cost function and token estimate in `threetears.models`, exported; the tracker uses the cost
  function; a release-note line in the change fragment saying `threetears_llm_cost_usd_total` drops for cache-heavy
  Anthropic traffic.
- **Tests:** unit: each price component; fallback to the input rate; `cost_per_request`; tracker `cost_usd` on a
  cache-heavy usage record; estimate parity with the old private function on its existing cases.
- **Acceptance criteria:** the tracker's existing tests pass, with any expected value that charged cache at the input
  rate updated and the reason named in the commit. The cost function is the only place usage is priced.
- **Done when:**
  1. Acceptance criteria met and tests pass
  2. `/prawduct:critic` run and blocking findings resolved
  3. Committed

### Chunk 02: A breaker registry the factory accepts, and a read with no side effects

- **Description:** `create_chat_model` and its callback builder take an optional public `breaker_registry=`. It keeps
  both the credential breaker and the per-model rate-limit breaker, unlike `breaker=`, which drops the second (KP-C6).
  `CircuitBreaker` gains a public, side-effect-free read of its state, so the pool can test eligibility without
  `check()` admitting a probe. The default stays the module's own registry, and nothing outside `factory.py` touches it.
- **Depends on:** none
- **Deliverables:** the `breaker_registry=` parameter; a public state read on `CircuitBreaker` (and on the registry by
  `(provider, credential, model)` if the pool needs it); docstrings.
- **Tests:** unit: a model built with a given registry records into that registry's credential and rate-limit
  breakers; the read never moves OPEN to HALF_OPEN; passing both `breaker=` and `breaker_registry=` raises
  `ValueError`.
- **Acceptance criteria:** existing factory and breaker tests unchanged and green.
- **Done when:**
  1. Acceptance criteria met and tests pass
  2. `/prawduct:critic` run and blocking findings resolved
  3. Committed

### Chunk 03: Per-provider classifiers, and key-moving errors reaching the caller unretried

- **Description:** A `Verdict` type: `exhausted`, `revoked`, `forbidden`, `rate_limited(until)` or `not_key_related`.
  Each carries a scope (key, key and model, account) and a `may_have_billed` flag. A classifier Protocol, with v1
  classifiers for anthropic, openai and openrouter, follows KP-C2's table, each rule pinned by a recorded response
  fixture. `retry-after` is parsed as seconds or an HTTP date. A missing one leaves `until` to the pool's floor and
  doubling (KP-C3).

  The provider factories gain a way to build a model whose SDK does not retry 429, so a key-moving failure reaches
  the pool on its first occurrence. Retries of 5xx and connection errors keep today's count (KP-H4).
- **Depends on:** none
- **Foreign API:** anthropic, openai and openrouter SDK exception types, response bodies and retry policy
- **Deliverables:** a classify module and the three classifiers; recorded fixtures under the models tests; a
  429-unretried mode in the three provider factories.
- **Tests:** unit: every KP-C2 row against its fixture, for each provider; an unknown exception is `not_key_related`;
  `retry-after` in both forms; OpenRouter's mid-stream 429; a model in 429-unretried mode surfaces a 429 after one
  attempt and still retries a 5xx the default number of times (driven through a fake HTTP transport, no network).
- **Acceptance criteria:** no status code, error type or header name appears outside the classifiers and the
  provider modules.
- **Done when:**
  0. verify-api: read the installed SDKs' exception classes, error-body shapes and retry code
     (`anthropic._base_client`, `openai._base_client`, the openrouter SDK). Record what was read in the chunk's commit
     message. Build fixtures from real error-body shapes in SDK source or provider docs, and name the source of each.
  1. Acceptance criteria met and tests pass
  2. `/prawduct:critic` run and blocking findings resolved
  3. Committed

### Chunk 04: The store Protocol, billing periods, and the in-memory store

- **Description:** The `KeyPoolStore` Protocol in generic ledger terms (Goals, decisions):
  - `reserve(account, reservation_id, amount, budget, expires_at)` succeeds only when `spent + reserved + amount ≤
    budget`, atomically (KP-E2);
  - `settle(account, reservation_id, amount)` and `release(account, reservation_id)`;
  - `sweep_expired(now)` settles expired reservations at their full amount and reports them as orphaned (KP-E5);
  - `snapshot(accounts)`;
  - `read_record(name)` returns `(version, document)`, and `compare_and_set_record(name, expected_version, document)`
    swaps it.

  The billing period: a UTC month with an anchor day from 1 to 28, a period id, and an injectable clock (KP-F). A
  reservation settles into the period it was made in (KP-F2), because the account string carries the period.

  `InMemoryKeyPoolStore` is the single-process production store (KP-G2): asyncio-safe and plain-typed.
- **Depends on:** none
- **Persisted format (lock-in):** the account and record shapes are what the durable store persists. The questions the
  data must answer:
  - what has this key spent this period;
  - what is reserved, and by which calls;
  - which reservations are orphaned;
  - what is the active key;
  - what state is each key in, and until when;
  - which fingerprint and budget each label was last configured with;
  - when was each key last rate-limited, and how long was the hold (for doubling).
- **Deliverables:** the Protocol, the period helpers and the in-memory store, in `threetears.models`.
- **Tests:** unit: N concurrent reservations against a budget that fits fewer than N; settle above and below the
  reservation; release; double settle and unknown ids are refused with a typed error; expiry sweep; rollover at the
  anchor day, including the 28th and a month end; compare-and-set conflict returns the winner's document.
- **Acceptance criteria:** the Protocol uses plain types only; a structural `isinstance` check against a
  `runtime_checkable` Protocol passes for the in-memory store.
- **Done when:**
  1. Acceptance criteria met and tests pass
  2. `/prawduct:critic` run and blocking findings resolved
  3. Committed

### Chunk 05: The durable ledger in core

- **Description:** `BudgetLedger` in `threetears.core.coordination`, beside `WindowedCounter` (Q1, ruled). It is built
  on the new coordination tables (Goals, decisions), with an L2 compare-and-swap fence and a synchronous L3 write on
  every mutation. A NATS wipe loses no spend (KP-G3). It refuses a registry with no L2. Storage failures propagate as
  they are; the pool wraps them as `KeyPoolStoreError` (KP-G5). It satisfies chunk 04's Protocol structurally, with no
  import of `threetears.models`.
- **Depends on:** Chunk 04 (the Protocol's exact signatures)
- **Foreign API:** `BaseCollection.l2_cas_mutate` with `l3_write_policy = "synchronous"` (in-repo, but a contract this
  chunk leans on)
- **Deliverables:** the ledger and its tables; a migration; exported names; `docs/design-durable-coordination.md` gains
  the ledger's row in its tables section.
- **Tests:**
  - unit: core, with the registry fixtures core's coordination tests already use;
  - integration (KP-L2, `./scripts/test-integration.sh core`): spend survives a new ledger on the same database after
    an L2 wipe; two ledgers share accounts and records; concurrent reservations from two processes never exceed the
    budget;
  - a conformance test, wherever the workspace lets one package's test see both, that core's ledger satisfies the
    models Protocol.
- **Acceptance criteria:** integration suite green on the pool, with every skip named; models' `pyproject.toml`
  dependencies unchanged.
- **Done when:**
  0. verify-api: read `BaseCollection.l2_cas_mutate` and the synchronous persist path, and confirm the won result is
     in L3 before the call returns
  1. Acceptance criteria met and tests pass
  2. `/prawduct:critic` run and blocking findings resolved (the one `cumulative` review is chunk 08's, over the branch)
  3. Committed

### Chunk 06: The pool core

- **Description:** `KeyPool` and `KeySpec`:
  - a redacted `Secret`, labels, budgets, priority and account; build-time refusals (KP-A1 to A5);
  - stable fingerprints (KP-B2);
  - the state machine and its lifetimes (KP-C4);
  - account-scoped verdicts (KP-C5);
  - holds with a floor and doubling (KP-C3);
  - sticky selection with a cached active key, refreshed on change or within 5 s (KP-D1, D2);
  - skip versus move (KP-D3);
  - ordered moves by compare-and-set (KP-D4);
  - priority pull-back (KP-D5), with a selection seam for a later policy (KP-D6);
  - move logging (KP-D7);
  - the public `reserve(estimate) → lease` with `settle` and `release` (KP-E6), and reservation expiry (KP-E5);
  - `NoKeyAvailableError` with its contents and its deduplicated log (KP-I);
  - `status()` (KP-K2);
  - operator reset.
- **Depends on:** Chunks 03 and 04
- **Deliverables:** the pool module(s) and their public exports.
- **Tests:** unit, from KP-L1's list: everything except the model-integration items, including a one-key pool through
  every path.
- **Exposed API:** `threetears.models` key pool (`KeyPool`, `KeySpec`, `Secret`, `KeyPoolLease`, `NoKeyAvailableError`,
  `KeyPoolStoreError`, `KeyPoolStore`, `InMemoryKeyPoolStore`)
- **Acceptance criteria:** the pool core imports nothing provider-specific.
- **Done when:**
  1. Acceptance criteria met and tests pass
  2. `/prawduct:critic` run and blocking findings resolved
  3. Committed

### Chunk 07: The pooled chat model

- **Description:** `create_chat_model(model_id, key_pool=pool, ...)` takes exactly one of `api_key` or `key_pool`. The
  pool's provider must match (KP-J1). A pooled runnable builds and caches one model per key:
  - each with that key's breakers from the pool's registry (chunk 02);
  - each in 429-unretried mode (chunk 03).

  Re-binding (`bind_tools`, `with_structured_output`, `bind`, `with_config`, `with_retry`) returns a pooled runnable
  that applies the same re-binding to whichever key serves the call (KP-J2).

  Each attempt:
  1. reserves its estimate (KP-E1);
  2. runs with that key's callbacks (KP-J3);
  3. settles from reported usage through chunk 01's cost function, or settles or releases on failure per the
     classifier (KP-E3, E4).

  Failover follows KP-H: at most once per key, `not_key_related` propagates unchanged, and nothing is retried after
  streamed output. When every key is only rate-limited, the call waits for the earliest `until` up to a per-pool cap,
  default 0 (Q4, ruled). `UsageRecord.credential_label` flows to spans and sinks, and it is not a Prometheus label
  (KP-J4). Subscription tokens are refused at build (§1).
- **Depends on:** Chunks 01, 02, 03 and 06
- **Exposed API:** `create_chat_model(key_pool=...)`
- **Deliverables:** the pooled runnable; the factory parameter; the `UsageRecord` field.
- **Tests:** unit: invoke, ainvoke, stream, astream, batch and abatch on a pooled model; `bind_tools` and structured
  output on a pooled model (the re-binding reaches the serving key's model); failover per verdict; no retry after a
  partial stream; tracker and breaker callbacks fire on the serving key; the Q4 wait with cap 0 and a positive cap
  under an injected clock; both or neither of `api_key` and `key_pool` raises.
- **Acceptance criteria:** an existing `create_chat_model(api_key=...)` caller sees no behaviour change; the existing
  factory suite is green.
- **Done when:**
  1. Acceptance criteria met and tests pass
  2. `/prawduct:critic` run and blocking findings resolved
  3. Committed

### Chunk 08: Metrics, the secrecy sweep, and the operator page

- **Description:** KP-K1's gauges and counters, behind the `prometheus` extra, with labels bounded by configuration.
  KP-B3's sentinel test drives every state and error path and searches logs, exception text, metric exposition, span
  attributes and store contents for the secret. The KP-M1 operator page goes in `docs/`. The API contract artifact
  records the new surface. The change fragment gets its narrative and consumer note, including Q2's release note.
- **Depends on:** Chunk 07
- **Deliverables:** the metrics; the sentinel test; new `docs/model-key-pool.md`; `.prawduct/artifacts/api-contract.md`
  updated; the fragment filled.
- **Tests:** unit: each metric moves on its event; the sentinel test.
- **Acceptance criteria:** `./scripts/check-all.sh` green, including the integration suite (`-rs`, each skip named).
- **Type:** cumulative-final
- **Done when:**
  1. Acceptance criteria met and tests pass
  2. Committed, then `/prawduct:critic cumulative` run over the branch and blocking findings resolved
