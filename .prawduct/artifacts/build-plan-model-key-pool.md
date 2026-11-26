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
operator-set budget. A dead or rate-limited key is moved off once, durably, for every process sharing the pool. The
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
  can't hold | owner can veto]

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
- Chunk 05: the ledger tables live in `LEDGER_TABLE_SCHEMAS`, not `COORDINATION_TABLE_SCHEMAS`, because v002 alters
  every table in that tuple and would fail where it runs before v004 creates them. Money and JSON are TEXT: no float
  before the budget compare, and JSONB would reorder the reservations the sweep reports in order.
- Chunk 05: the ledger refuses a registry with no L3 (main session, "do it right"). Spend on L2 alone is lost to a
  broker restart, and no ledger deployment runs without a database, unlike identity-edge's throttles.
- Chunk 05: `BudgetLedger` has no caller in this repo's src (models can't import core), so
  `_public_api_consumers.txt` names Hearsay, whose wiring follows the release. Owner to confirm at the boundary.

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
  2. Committed, then `/prawduct:critic cumulative` run over Part 1 and blocking findings resolved
- **Critic mode:** cumulative

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
  2. Committed, then `/prawduct:critic cumulative` run over Part 2 and blocking findings resolved
