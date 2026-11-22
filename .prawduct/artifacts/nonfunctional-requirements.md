---
artifact: nonfunctional-requirements
version: 1
depends_on:
  - artifact: product-brief
last_validated: null
---

# Non-Functional Requirements

## Summary

3tears is an alpha library family (`README.md`). Its design promises local synchronous reads, cross-pod coherence through NATS, and no lost requests when a NATS server rolls. The repo states almost no numeric targets. Every number below is a measurement or a default in code or docs, not a requirement. `.prawduct/project-state.yaml` leaves all four `nonfunctional` fields null.

## Performance

The design promises, and how each is tested:

| Promise | Source | How it is tested |
|---|---|---|
| L1 is SQLite in the process: reads are local and synchronous (`users[user_id, "name_display"]`) | `README.md`, `packages/core/README.md` | Behaviour tests in `packages/core/tests`; no latency bound is asserted |
| Reads fall through L1, L2, L3 and promote on a miss | `README.md` | Same |
| Bulk work stays off the event loop: a commit that settles thousands of rows evicts L1 on a worker thread | `packages/core/tests/unit/collections/test_l1_eviction_off_loop.py` | `loop_timing` marker, wall-clock lateness of the loop |
| An L2-first write commits in one NATS epoch record; it waits for L3 by default | `docs/l2-first-operations.md` | Behaviour tests; cost measured by script (below) |

Wall-clock tests carry the `loop_timing` marker. They run serially on CI's own runner and stay off the testament pool, because beside other batches a bound on the code becomes a bound on the runner's load (`.testament.toml`, `scripts/ci-unit-tests.sh`). `threetears.core.testing.loop_lateness` discounts lateness that the host's own idle loop shared (`packages/core/tests/unit/testing/test_loop_lateness.py`).

Write cost of a declared write generation, measured by `scripts/measure-epoch-write-cost.py` (`docs/runbook-epoch-write-cost.md`). No test lane runs it; it is run by hand before a release that touches the epoch path.

| Writers | Variant | p50 ms | p99 ms | Saves/s |
|---|---|---|---|---|
| 1 | undeclared | 1.51 | 4.75 | 563 |
| 1 | declared | 2.66 | 5.06 | 351 |
| 8 | undeclared | 4.03 | 6.72 | 1894 |
| 8 | declared | 5.99 | 82.65 | 554 |
| 32 | undeclared | 15.63 | 47.55 | 1727 |
| 32 | declared | 45.19 | 312.50 | 445 |

These ran on an Apple M4 under Docker Desktop at one commit (`docs/runbook-epoch-write-cost.md`). They are evidence, not a target. The runbook states a decision rule: declaring is cheap while `failed` stays 0 and CAS losses stay under 1 per write.

Other limits in code or config:

- Default per-test timeout is 600 s (`pyproject.toml`, `timeout = 600`).
- The L2-first worker pass interval `every` is the only tuning knob and bounds how long a non-waiting write stays unsaved (`docs/l2-first-operations.md`).

No target exists for L1 read latency, L2 round trip, or L3 write latency.

## Scalability

The target is many pods sharing one L2 (NATS KV) and one L3 (`.prawduct/project-state.yaml`, topology `client-server`). The repo states no pod count, row count or throughput goal.

| Concern | What the repo says | Source |
|---|---|---|
| Pods read locally | Pods never read L3 on an L2-first collection | `docs/l2-first-operations.md` |
| One epoch key per declared table | Throughput of a declared table is bounded by that key: about 450 to 550 saves/s measured, against 1,700 to 1,900 undeclared | `docs/runbook-epoch-write-cost.md` |
| Many concurrent writers on one table | Not acceptable today for tens of writers: some saves exhaust 30 CAS rounds after the row committed | `docs/runbook-epoch-write-cost.md` |
| Change log fills | Gauge `threetears_epoch_changes_fill_ratio`; warn at 0.75; past 1.0 writes that need a record fail | `docs/l2-first-operations.md` |
| Large L1 copies | One host's DuckDB copy is about 360 MB; snapshot design bounds memory | `docs/design-scoped-snapshot.md` |
| One NATS L2 for a single pod | Optional with PostgreSQL on one pod; required for more than one pod or a git L3 | `README.md` |

## Consistency and coherence

| Guarantee | Mechanism | Source |
|---|---|---|
| A write reaches L3, L1 and L2 on `save()` under `ALWAYS` | Write-through | `packages/core/README.md` |
| Deferred flush only for named tables | `ON_CHECKPOINT`, `ON_SCHEDULE`, `ON_SHUTDOWN` on tables in `collection_flush_tables`; a deferred write not yet in L3 can be lost | `packages/core/README.md` |
| Lost update is refused | Optimistic lock on `date_updated`; raises `ConcurrentModificationError` | `packages/core/README.md` |
| A missed broadcast drops the table instead of serving stale rows | Write generation, followed by every pod | `packages/core/README.md`, `docs/epoch-task-06-collection-generations-by-default.md` |
| After NATS loses data, pods keep serving what they hold; a key they lack raises `L2LostError` | One leased worker restores the change log and rows from L3 | `docs/l2-first-operations.md` |
| One holder at a time | `KVLease`: TTL-bounded mutex, renewed on a schedule, `LeaseLost` once lost | `packages/core/src/threetears/core/coordination/lease.py` |
| Order within a run on NATS | A run whose order is its meaning takes a `PublishPin` | `docs/nats-credential-renewal.md` |

The L1 staleness statement is in conflict inside the repo: see Open questions.

## Availability and reliability

No uptime target exists. 3tears is a library; availability belongs to the host. The repo does state failure behaviour.

| Event | Behaviour | Proof |
|---|---|---|
| NATS server enters lame-duck | The client moves to a successor connection through `renew_connection`; old subscriptions leave their queue group after 1 s | `docs/nats-credential-renewal.md`; `packages/nats/tests/integration/test_lame_duck_handover_live.py` |
| Same event, measured | Pod moved about 90 ms after the notice; no request or subscribed message lost on a two-node 2.14.2 cluster | `docs/nats-credential-renewal.md` |
| NATS user JWT expires | Default TTL is one day (backstop); renewal is make-before-break about once a day per pod | `docs/nats-credential-renewal.md`; `packages/nats/tests/integration/test_credential_renewal_live.py` |
| A principal must lose access | Kick through the system account; the auth callout refuses the reconnect. A lost kick keeps the credential until expiry | `docs/nats-credential-renewal.md` |
| Lease holder stops renewing | The per-key TTL expires the lease | `packages/core/src/threetears/core/coordination/lease.py` |
| L3 refuses a committed non-waiting write | Counter `threetears_l2_first_l3_refused_total`; alert on any increase | `docs/l2-first-operations.md` |
| Failure in a handler | Reaches the caller or is recorded through `contain()`; never neither | `docs/exceptions.md`, `tests/enforcement/test_no_silent_swallow.py` |

## Compatibility

| Surface | Floor or rule | Tested by |
|---|---|---|
| Python | `>=3.14` | `packages/core/pyproject.toml`; `pyproject.toml` (`python_version = "3.14"`) |
| OpenTelemetry | `>=1.40` for api and sdk in the workspace; `observe` declares its own floor | `.github/workflows/ci.yml` step "Test observe at its declared OpenTelemetry floor"; `scripts/test-otel-floor.sh`; `tests/enforcement/test_otel_floor.py` |
| MCP SDK | `mcp>=1.27,<3`: both 1.x and 2.x supported; lock pins 2.x | `scripts/test-mcp-v1.sh` (mcp 1.30.0, fastmcp 3.4.8), run by `scripts/check-all.sh` |
| Family versions | All `3tears*` packages share one version; sibling bounds `>=M.m.0,<M.(m+1).0` | `tests/enforcement/test_intra_family_version_bounds.py` |
| API growth | An added intra-family API ships in a minor bump | `tests/enforcement/test_api_growth_requires_a_minor_bump.py` |
| Public API | Alpha: can change between minors until 1.0.0; consumers pin the whole family | `README.md`, `CLAUDE.md` |
| Web packages | npm `@3tears/web-*` carry the family version | `pyproject.toml`, `.github/workflows/web.yml` |

## Cost visibility

3tears does not set an LLM budget. It makes spend visible to the host.

- `UsageTracker` records one row per LLM call; `UsageAccumulator` totals a turn (`packages/models/src/threetears/models/usage.py`).
- Vendor prices are not baked in. A registry of price sources prices per 1M tokens: OpenRouter's public listing is the catch-all, Voyage is scraped (`packages/models/src/threetears/models/price_lookup.py`).
- Matching is conservative: a near model never gets a price. A wrong price is judged worse than none, so an admin enters it by hand (same file).
- Prompt caching of the system prompt on Anthropic models, with cache read and write counters normalised for the host (`docs/prompt-caching.md`).

Infrastructure cost: the testament pool stops a batch at 30 min (`.testament.toml`, `batch_timeout_s`) and the pool job at 75 min (`docs/testing.md`). No hosting or LLM budget is stated in the repo.

## Accessibility

Not applicable as a product requirement: `has_human_interface` is false and the `web/` packages are building blocks for host apps (`.prawduct/project-state.yaml`). `web/README.md` and the web package READMEs state no accessibility rule or test.

## Candidate norms (unratified)

- Every `3tears*` intra-family dependency carries a lockstep bound: `tests/enforcement/test_intra_family_version_bounds.py`.
- An intra-family API addition needs a minor bump: `tests/enforcement/test_api_growth_requires_a_minor_bump.py`.
- A broad exception handler re-raises or calls `contain()`: `tests/enforcement/test_no_silent_swallow.py`.
- The OpenTelemetry floor is tested at its declared version: `tests/enforcement/test_otel_floor.py`.
- Wall-clock tests carry `loop_timing` and run serially: `tests/enforcement/test_ci_unit_shards.py`, `scripts/ci-unit-tests.sh`.
- No attribution to an AI agent in commits: `scripts/check-attribution.sh` (run by CI).
- A leading underscore binds only inside its module: `tests/enforcement/_underscore_exemptions.txt`.

## Open questions

1. L1 staleness. `packages/core/README.md` says nothing in L1 ages, while `docs/epoch-task-05-l1-max-age-lazy-expiry.md` describes a lazy max-age. Which holds today? Is there a stated bound on how long a pod may serve a stale row after a missed broadcast?
2. Is there any target for L1 read, L2 round trip, or L3 write latency, or is local synchronous read a qualitative promise only?
3. What pod count and write concurrency per table should the epoch path serve? The measured ceiling is hundreds of saves/s per declared table.
4. Is a failed declared save (30 CAS rounds exhausted, row committed) acceptable at any rate? The runbook calls any value above 0 a correctness problem.
5. Does the repo want an uptime or recovery target for L2 loss, or does it stay the host's concern?
6. Does `web/` need an accessibility stance for `@3tears/web-pwa`?
7. Is there a spend ceiling for the live-model test lane (`scripts/test-live-claude-cli.sh`)? None is stated in the repo.
