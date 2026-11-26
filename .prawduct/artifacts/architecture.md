---
artifact: architecture
version: 1
depends_on:
  - artifact: product-brief
  - artifact: data-model
  - artifact: security-model
  - artifact: nonfunctional-requirements
last_validated: null
---

# Architecture

## Summary

3tears is a family of Python libraries that a host application imports into its own processes (`docs/integration-guide.md` §1). It ships no daemon of its own for the data layer. The topology lives at runtime: many host pods share one L2 (NATS JetStream KV) and one L3 (PostgreSQL, YugabyteDB or a git working tree) through the library (`README.md`; `.prawduct/project-state.yaml` `multi_process_distributed: client-server`). A few packages do ship runnable processes: the tool registry, tool pods, the L2-first worker and the scrape sidecar.

Terms used below:

| Term | Meaning | Source |
|---|---|---|
| L1 | In-process, in-memory SQLite cache, one per process, rebuilt from L3 on restart | `docs/integration-guide.md` §3 |
| L2 | NATS JetStream KV, shared by every pod; keeps pod caches coherent | `README.md` |
| L3 | The durable source of truth: an asyncpg-style pool, `NatsProxyL3Backend`, or a git working tree | `docs/integration-guide.md` §3 |
| Host | The application that embeds 3tears and owns connections, config, secrets and lifecycle | `docs/integration-guide.md` §2 |
| Pod | One host process; many run the same code | `README.md` |
| Principal | The identity a process authenticates to NATS as (`Principal` enum: agent pod, tool pod, registry, hub, gateway, channel adapter, agent router, dataset executor) | `packages/nats/src/threetears/nats/subject_permissions.py` |
| Epoch | A strictly monotonic generation number per NATS subject, broadcast best-effort and caught up by polling | `packages/epoch/README.md` |
| Write generation | A table's epoch, advanced on every committed write and followed by pods | `packages/epoch/README.md` |
| Family | Every `3tears*` PyPI package and every `@3tears/web-*` npm package, released at one version | `docs/releasing.md` |

## Overview & Topology

### Runtimes

| Runtime | Ships in | Where it runs | Lifecycle |
|---|---|---|---|
| Host pod with 3tears embedded (L1 inside) | every package; wiring in `threetears.core` | host's servers | always-on, many replicas |
| NATS server with JetStream (L2, pub/sub, request/reply) | host-provided | host's cluster | always-on |
| PostgreSQL or YugabyteDB (L3) | host-provided | host's cluster | always-on |
| Git working tree (alternative L3) | `threetears.core.backends.git` | host's filesystem; one writer per repository under a NATS lease | per host |
| L3 broker answering `<ns>.l3.*` | host-provided; only the client ships (`docs/integration-guide.md` §13) | host's cluster | always-on |
| Tool registry | `3tears-registry` (`python -m threetears.registry`) | host's cluster | always-on |
| Tool pod | `3tears-agent-tools` (`ToolServer`, `ToolServerBootstrap`) | host's cluster, or in an agent's process | always-on, or follows an owner pid |
| L2-first worker | `run_l2_first_worker` in `threetears.core.collections.l2_first` | a host process holding a lease | always-on, one claimed holder |
| Scheduled-job tick pump | `3tears-scheduled-jobs` (`scheduled_tick_job`) | host pods, cross-pod locked | scheduled |
| nodriver sidecar (HTTP, port 8088) | `packages/scrape/sidecar` (AGPL, own version) | container beside the scrape caller | always-on |
| S3-compatible object store | host-provided; client in `3tears-object-store` | host's cloud | always-on |
| Browser app using `@3tears/web-pwa` | `web/packages/pwa` | person's browser | user-launched |
| Host platform services (hub, gateway, auth callout) | host repos; `docker-bake.hcl` builds them from sibling repos | host's cluster | always-on |

### Sketch

```text
  browser (@3tears/web-pwa)          external APIs (models, search, Slack, Discord)
          |                                       ^
          v                                       |
  +----------------- host pod (many) --------------------+
  |  host code -> threetears.* libraries                  |
  |  L1 SQLite (in-process)                               |
  +-----+-------------------+--------------------+--------+
        | KV, pub/sub,      | SQL (direct pool)  | HTTP :8088
        | request/reply     |                    v
        v                   |            nodriver sidecar
  +-----------+             |
  | NATS (L2) |<--- registry, tool pods, hub, gateway, L3 broker
  +-----+-----+             |
        | <ns>.l3.*         v
        +----------> L3 broker ----> PostgreSQL / YugabyteDB (L3)
```

### Why this shape

- Apps run on many pods, and their state scatters without a shared layer: caches drift and keeping pods in step becomes a second job (`README.md`).
- Reads stay local and synchronous through L1; writes stay coherent through L2 and land in L3 (`README.md`).
- Sandboxed worker pods hold no database credentials, so their L3 goes through a NATS proxy to a broker that holds them (`docs/integration-guide.md` §3, §8.3).
- Every tier past L3 is optional. A single pod runs L1 plus L3 and adds NATS when it scales out (`docs/integration-guide.md` §5, principle 2).

## Components & Responsibilities

| Component | Purpose | Owned state | Must never | Source |
|---|---|---|---|---|
| `CollectionRegistry` | The one dependency-injection seam for all three tiers; drives cross-pod invalidation | per-process tier wiring, table bindings | wire L2 without a `kv_key_scope` (raises `L2ScopeNotConfiguredError`) | `docs/integration-guide.md` §4, §8.2 |
| `BaseCollection` / `DataStore` | Per-table three-tier gateway; change-tracked entities with optimistic locking | the table's rows, cached per tier | treat L1 as durable | `docs/integration-guide.md` §3–4 |
| `NatsClient` | The single wrapper over nats-py: reconnect, subject namespacing, KV helpers | the connection; streams and buckets it declared | be bypassed; no other package imports `nats` | `packages/nats/README.md`; `tests/enforcement/test_nats_wrapper_usage.py` |
| `3tears-epoch` | Generation-stamped reload signals for config caches and table write generations | the `{ns}-epochs` KV counters; durable `config_epochs` rows for tile epochs | let a pod advance an epoch (pods hold the bucket read-only) | `packages/epoch/README.md` |
| Registry | Admits tool copies from verified publishers, authorizes calls, routes to a tool pod | tool catalog (KV-backed), heartbeats | admit a publisher from the manifest's own claims | `packages/registry/README.md` |
| Tool pod (`ToolServer`) | Serves tools over NATS; stamps a baseline audit event per call | none durable | write `namespaces` rows (the platform emitter is the sole writer) | `packages/agent/tools/README.md` |
| L2-first worker | Saves committed L2-first epochs to L3; refills L2 from L3 after NATS loses data | saver position per table | run without holding its lease | `docs/l2-first-operations.md` |
| `3tears-agent-audit` | One envelope and a fire-and-forget publish for every domain | none on a hub deployment; its own table when no hub exists | taint a tool's response on publish failure | `packages/agent/audit/README.md` |
| `3tears-observe` | Loggers with `NullHandler`, `@traced`, `contain()` | context vars per task | install handlers, formatters or exporters | `docs/observability.md` |
| `3tears-interfaces` | Shared Protocols and vocabulary with no dependencies | none | import anything beyond the stdlib and itself | `packages/interfaces/README.md` |
| nodriver sidecar | Renders pages in a real browser; HITL sessions | browser sessions, session tokens | authenticate identity (it honours a bearer capability per session) | `docs/sidecar-http-contract.md` §4 |
| `3tears-evals` | Eval engine behind host-filled ports | none of its own; the host supplies the store | depend on any other 3tears package | `README.md`; `docs/family-convergence.md` header |

The package graph layers on `3tears-observe` and `3tears-interfaces`, which declare no family dependencies. `3tears-nats` depends only on observe. Core depends on observe and nats. The agent packages and integrations build on core (each `pyproject.toml`). `3tears-evals` and `3tears-tool-schema` declare no family dependencies (`packages/evals/pyproject.toml`, `packages/tool-schema/pyproject.toml`).

## Communication & Boundaries

| Channel | Transport | Direction | Sync | Contract |
|---|---|---|---|---|
| Collection reads and writes to L2 | NATS KV, bucket `{namespace}-collections`, keys `{scope}.{table}.{body}` | pod ↔ NATS | async | `docs/integration-guide.md` §8.2 |
| Cross-pod invalidation | pub/sub on `threetears.cache.invalidate` (no namespace prefix), `CacheInvalidationMessage` | writer → every pod | async, broadcast | `packages/nats/src/threetears/nats/subjects.py` |
| Epoch bumps | pub/sub `EpochBumpMessage`, counter in `{ns}-epochs` KV | hub/gateway → pods | async, best-effort | `packages/epoch/README.md` |
| Proxied L3 | request/reply `<ns>.l3.query`, `l3.batch`, `l3.tx.*`, `l3.truncate`, `l3.load_files` | pod → broker | sync request | `docs/integration-guide.md` §3 |
| Direct L3 | asyncpg-style pool | pod → database | async | `docs/integration-guide.md` §3 |
| Tool registration and heartbeats | `{ns}.tools.register` request/reply; `{ns}.tools.heartbeat.*` publish | tool pod → registry | mixed | `packages/core/src/threetears/core/tool_wire/messages.py` |
| Tool calls | `{ns}.tools.call` (`ProxyCallRequest`) then registry → pod (`CallRequest`) | agent or pod → registry → tool pod | sync request | `packages/registry/README.md` |
| Audit | publish on `{ns}.audit.>`; JetStream consumer persists | every producer → hub or own persister | async, at-least-once | `packages/agent/audit/README.md` |
| Owner-routed request and byte pipe | `forward` / `serve_owner`; `open_pipe` with `PIPE_PROTOCOL_VERSION` | any pod → key owner | sync, streamed | `packages/nats/README.md`; `packages/nats/src/threetears/nats/pipe_wire.py` |
| Sidecar | HTTP `/v1/*` on port 8088 | scrape caller → sidecar | sync | `docs/sidecar-http-contract.md` |

The contract details for each channel are in `api-contract.md`.

### Trust boundaries

- A subject proves nothing about its sender. Every NATS receiver declares how it authenticates what arrives (`tests/enforcement/test_subscribers_declare_sender_posture.py`).
- Each principal's subject grants equal the subjects its code uses, in both directions (`tests/enforcement/test_subject_grants_match_the_code.py`).
- The L2 key scope is the isolation boundary between principals; `kv_key_scope_for` is its only producer (`tests/enforcement/test_l2_scope_discipline.py`).
- The registry verifies publishers from identity tokens (`ToolPodAuthenticator`) and refuses an unidentified registry build (`RegistryIdentityUnavailableError`) (`packages/registry/README.md`).
- Known residual: a pod that can write a value and read it back can put those bytes on any subject in its account; the receiver is the defence (`docs/design-durable-coordination.md`, last section).

What each crossing requires is for `security-model.md`.

## Data Ownership & Consistency

| State | Writer | Readers | Consistency | Source |
|---|---|---|---|---|
| Collection rows (default) | any pod, through its collection | every pod | L3 is truth; L1/L2 are disposable; a write goes to L3, promotes, then broadcasts invalidation | `docs/integration-guide.md` §3, §5 |
| L2-first collection rows | any pod with the grants; commit point is the NATS epoch record | pods read L2 only | the saver makes L3 match; a write that does not wait may be lost on NATS data loss | `docs/l2-first-operations.md` |
| `{namespace}-collections` bucket config | one declaring identity | every other process binds | bind refuses to create, so a missing declarer fails wiring | `docs/integration-guide.md` §12 Step 3 |
| Epoch counters | hub and gateway | agent pod, tool pod, registry (read only) | push for speed, pull for correctness | `packages/epoch/README.md` |
| Tool `namespaces` rows | the platform-side namespace emitter (sole writer) | registry authorizer | upsert by deterministic `uuid5` | `packages/agent/tools/README.md` |
| Audit rows | the hub's consumer, or `start_audit_persister` without a hub | admin queries | at-least-once; the envelope `id` deduplicates | `packages/agent/audit/README.md` |
| Coordination state (revocations, counters, claims, redemptions) | primitives in `threetears.core.coordination` | every pod | revocations synchronous to L3; counters and claims L2 CAS with write-behind to L3 | `docs/design-durable-coordination.md` |
| Schema migrations | one one-shot task per schema | — | database-wide DDL lock; idempotent, `(version, package)`-keyed | `docs/integration-guide.md` §9, §12 Step 2 |

- Concurrent edits of one entity raise `ConcurrentModificationError` on a `date_updated` mismatch. The host chooses retry or merge (`docs/integration-guide.md` §5, principle 6).
- Write-behind counters lose at most the increments since the last flush on a NATS wipe (`docs/design-durable-coordination.md`).
- Every NATS KV bucket is memory-backed; durability belongs to L3 (`docs/design-durable-coordination.md`; `packages/core/tests/enforcement/test_kv_buckets_are_memory_only.py`).

## Failure Modes & Resilience

| What fails | Effect | Recovery | Source |
|---|---|---|---|
| L2 unhealthy | reads degrade to L1 plus L3; programming errors still surface | automatic | `docs/integration-guide.md` §5, principle 4 |
| NATS restart wipes streams or buckets | the declaring `NatsClient` recreates them, empty, after every reconnect | owner refills (`on_restored`, `PersistedCopyBucket`) | `packages/nats/README.md` |
| NATS loses L2-first data | pods serve what they hold; missing keys and writes raise `L2LostError` | worker refills L2 from the L3 epoch log | `docs/l2-first-operations.md` |
| Epoch counter replaced by broker restart | broadcasts alone miss it | `on_reset` callback plus `catchup_tick` | `packages/epoch/README.md` |
| Nonce bucket wiped | `ReplayGuard` refuses artifacts issued before the bucket's creation, plus tolerance (5s, 60s or 0s by signer) | window passes; services `bind()` at startup | `docs/design-durable-coordination.md` |
| Missing KV grant | the request blocks to its deadline and times out; the log names the bucket | add the grant | `packages/epoch/README.md` |
| Registry host dependency fails | answered with a code (`PUBLISHER_VERIFICATION_UNAVAILABLE`, `CATALOG_UNAVAILABLE`, `TOOL_AUTHORIZATION_UNAVAILABLE`, `REGISTRY_INTERNAL_ERROR`) | pod heartbeat re-offers its manifest | `packages/registry/README.md` |
| Tool pod shutdown fails | logged once at ERROR; process exits 70 | orchestrator restarts | `packages/agent/tools/README.md` |
| Audit publish fails | logged at WARN; the tool's response is untouched | none | `packages/agent/audit/README.md` |
| Subscriber callback raises | republished to `{ns}.deadletter.{path}` by default | operator | `packages/nats/README.md` |
| Epoch change log fills | past 1.0 every write needing a record fails; warn at 0.75 | raise `max_bytes` or find the writer | `docs/l2-first-operations.md` |
| Shutdown without `close_collections()` | the last write-behind interval is lost | none | `docs/design-durable-coordination.md` |

Every handler either re-raises or records the failure through `contain()` (`docs/exceptions.md`). NATS reconnect is unbounded at runtime and bounded at startup (`packages/nats/README.md`). Shutdown order is: stop the invalidation listener, drain NATS, close the pool, with a grace window at least the NATS drain timeout (`docs/integration-guide.md` §12 Step 6).

## Deployment & Version Skew

- **Deploy units.** Every Python package and every web package releases at one version from one tag (`docs/releasing.md`; `pyproject.toml` `[tool.threetears-release]`). The sidecar carries its own version outside the lockstep (`docs/sidecar-http-contract.md` §3).
- **Skew inside a process** is refused at install. Intra-family dependencies are bounded to one minor line, so pip cannot resolve a mixed family (`tests/enforcement/test_intra_family_version_bounds.py`).
- **Skew between processes** is expected. Pods, the registry, brokers and the hub upgrade on their own schedules (expand, migrate, contract; owner ruling 2026-10-07, `packages/core/src/threetears/core/wire.py`). Wire models read across a deployment boundary ignore unknown fields (`tests/enforcement/test_wire_models_are_tolerant_readers.py`).
- **Grants outlive code** for older releases still running, each dated with its reason (`tests/enforcement/test_subject_grants_match_the_code.py`).
- **Order of operations.** Upgrade the L3 broker before any pod sends `l3.truncate`, `l3.tx.copy` or `l3.load_files` (`docs/integration-guide.md` §3). Apply migrations once per schema before pods use the tables (`docs/integration-guide.md` §12 Step 2).
- **Releases.** Bump, PR to `develop`, PR `develop` to `main`, push a tag; the tag push publishes PyPI then npm (`docs/releasing.md`).

## Scaling Model

- L1 scales with pod count for free (`docs/integration-guide.md`, Scaling notes).
- L2 is the shared layer that makes horizontal scaling cache-coherent (same source).
- L3 is the durability and throughput ceiling (same source).
- Singletons: the L2-first worker under a lease (`docs/l2-first-operations.md`); the scheduled-job tick pump under a cross-pod lock (`packages/scheduled-jobs/README.md`); migrations as one task per schema; one writer per git repository (`docs/integration-guide.md` §3).
- Known bottleneck: a table's write generation is one key every writer compare-and-swaps. Heavy concurrent writing contends on it, and after 30 lost rounds the writer raises `GenerationUnavailableError` (`docs/design-durable-coordination.md`, "What one generation per table costs").
- Signals: `threetears_epoch_changes_fill_ratio`, `collection_epoch_tables_behind`, `threetears_l2_first_refills_total` (`docs/l2-first-operations.md`).

Scale targets belong in `nonfunctional-requirements.md`; none are recorded yet (`.prawduct/project-state.yaml` `nonfunctional`).

## Cross-Cutting Runtime Concerns

- **Correlation.** `threetears.observe` keeps one context whose entries ride every log line and span (`docs/observability.md`). Tool calls carry `correlation_id` in `CallContext` across the registry hop, and audit events share it per request (`packages/agent/tools/README.md`).
- **Time.** A PoP signer must agree with the registry to within 5s; DPoP signers get 60s (`docs/design-durable-coordination.md`). `timestamptz` columns store aware UTC (`docs/integration-guide.md` §9).
- **Configuration.** The host configures; 3tears reads a small set of `THREETEARS_*` variables as fallbacks and tuning (`docs/integration-guide.md` §10). Cache reloads after config writes propagate by epoch (`packages/epoch/README.md`).
- **Secrets.** The host injects DSNs and NATS credentials (`docs/integration-guide.md` §12 Step 4). NATS user JWTs live a day and are renewed make-before-break; access is cut by a server kick (`docs/nats-credential-renewal.md`). The registry's identity comes from a host-supplied provider factory (`packages/registry/README.md`).
- **Environment parity.** A dev box runs L1 plus L3 and skips NATS (`docs/integration-guide.md` §12). Testcontainer fixtures ship in `3tears[testing]` (§8.1). The integration suite runs on the testament pool with Postgres, NATS and S3 as child processes (`docs/testing.md`, "The integration suite").

## Decision Log

| Decision | Alternatives | Why | Source |
|---|---|---|---|
| Ship libraries; the host owns services and lifecycle | a deployed service | "library instruments, host configures" | `docs/integration-guide.md` §1–2 |
| L3 is the only required tier | require NATS | single pods start small and scale out later | `docs/integration-guide.md` §5 |
| Lockstep family version with minor-line bounds | unbounded siblings | mixed installs broke at runtime; pip blamed the wrong package | `docs/releasing.md` |
| Every NATS KV bucket memory-backed; no exemptions | file-backed buckets | file storage gives one replica, no backups, no schema | `docs/design-durable-coordination.md` |
| Tolerant readers on cross-deployable wire models | `extra="forbid"` everywhere | strict models turn every additive field into a lockstep release | `packages/core/src/threetears/core/wire.py` (2026-10-07) |
| Tool wire messages live in core | beside the pod or the registry | both ends parse them without depending on each other | `packages/core/src/threetears/core/tool_wire/messages.py` |
| `3tears-evals` depends on no 3tears package | build on core | ports let a host fill them with its own systems | `docs/family-convergence.md` header (2026-10-11) |
| No function-level imports in serving code | lazy imports | a lazy import runs mid-request and holds the event loop | `tests/enforcement/test_no_function_level_imports.py` (2026-10-24) |
| Sidecar versioned outside the lockstep | family version | it can signal a break without a framework release | `docs/sidecar-http-contract.md` §3 |
| Day-long NATS JWT plus kick | short TTL renewals | each renewal was a handover with its own risk | `docs/nats-credential-renewal.md` (2026-09-30) |

## Direction

Ratified 2026-11-26 from the test-backed candidates below, which the owner approved in bulk once
duplicates were removed ("Yes to all recommendations"); each Why names its gate or the ruling
behind it.

- **Every NATS KV bucket is memory-backed; durability belongs to L3.**
  Why: L2 is a cache, and the rule carries no exemption mechanism. Gate:
  `packages/core/tests/enforcement/test_kv_buckets_are_memory_only.py`.
- **Every package reaches NATS through `threetears.nats.NatsClient`; only `3tears-nats` imports
  `nats`.**
  Why: the wrapper holds reconnect policy, subject namespacing and the KV helpers; a direct import
  opts out of all three. Gate: `tests/enforcement/test_nats_wrapper_usage.py`.
- **`kv_key_scope_for` is the only producer of an L2 key scope.**
  Why: the scope is the isolation boundary between principals, and an interpolated name through a
  non-injective sanitizer can collide two principals into one scope. Gate:
  `tests/enforcement/test_l2_scope_discipline.py`.
- **A KV bucket opens only through the reconcile primitive in `kv.py`.**
  Why: a direct opener gets none of the reconciliation and silently carries another caller's
  configuration (a second opener once asked for a different TTL against a live bucket). Gate:
  `tests/enforcement/test_kv_bucket_open_discipline.py`.
- **A KV grant names the bucket its opener actually creates.**
  Why: the grant string becomes real JWT permissions, and a wrong name fails silently in both
  directions. Gate: `tests/enforcement/test_kv_bucket_grant_naming.py`.
- **Each principal's subject grants equal the subjects its code receives and publishes on, in
  both directions.**
  Why: a refused subscribe is a subscription that receives nothing (a silently stale cache), and a
  grant with no site outlives the code that needed it. Gate:
  `tests/enforcement/test_subject_grants_match_the_code.py`.
- **Every NATS receiver declares how it authenticates the sender of what it receives.**
  Why: a subject proves nothing about its sender; any connection can have its bytes delivered on
  any subject. Gate: `tests/enforcement/test_subscribers_declare_sender_posture.py`.
- **No code bypasses epochs or degrades quietly.**
  Why: an epoch is its change set, and every pod applies exactly those rows; a whole-table drop, a
  swallowed L2 failure, a TTL standing in for an epoch or a raw L3 write breaks that. Gate:
  `tests/enforcement/test_epoch_discipline.py`.
- **Every collection with a `TableSchema` hands back, from every tier, exactly what was written.**
  Why: a tier that changes a type or loses precision serves the changed value on every read it
  answers. Gate: `tests/enforcement/test_tier_parity.py`.
- **Every persisted entity id is a UUIDv7, never a uuid4.**
  Why: cursor-paged queries order by the primary key, and a uuid4 makes them skip rows or return
  them out of creation order. Gate: `tests/enforcement/test_uuidv7_enforcement.py`.
- **A package imports only what its own `pyproject.toml` declares.**
  Why: the workspace masks undeclared dependencies until a standalone install of one package
  fails in a consumer. Gates: `tests/enforcement/test_imports_declared.py`,
  `tests/enforcement/test_dependency_alignment.py`.

- **Pricing is the host's: the engine reports tokens and models, the host injects prices, and
  3tears ships only a swappable dev default.**
  Why: a consumer may have special deals, BYOK, subscriptions or local models; a rate table in the
  engine would be wrong for them. Owner 2026-10-28 (`.prawduct/owner-rulings.md`).

## Candidate norms (unratified)

- Every NATS KV bucket is memory-backed — `packages/core/tests/enforcement/test_kv_buckets_are_memory_only.py`
- No package imports `nats` except `3tears-nats` — `tests/enforcement/test_nats_wrapper_usage.py`
- `kv_key_scope_for` is the only producer of an L2 key scope — `tests/enforcement/test_l2_scope_discipline.py`
- KV buckets open only through the reconcile primitive — `tests/enforcement/test_kv_bucket_open_discipline.py`
- Subject grants match the code in both directions — `tests/enforcement/test_subject_grants_match_the_code.py`
- Every NATS receiver declares its sender posture — `tests/enforcement/test_subscribers_declare_sender_posture.py`
- No code bypasses epochs or degrades quietly — `tests/enforcement/test_epoch_discipline.py`
- Every collection tier hands back what was written — `tests/enforcement/test_tier_parity.py`
- Persisted entity ids are UUIDv7 — `tests/enforcement/test_uuidv7_enforcement.py`
- A package imports only what its own `pyproject.toml` declares — `tests/enforcement/test_imports_declared.py`, `tests/enforcement/test_dependency_alignment.py`
- Broad handlers re-raise or `contain()` — `tests/enforcement/test_no_silent_swallow.py`

## Open questions

- The integration guide says to declare `{namespace}-collections` on file storage so it survives a broker restart (`docs/integration-guide.md` §3, §12 Step 3). The memory-only rule says every bucket is memory-backed (`docs/design-durable-coordination.md`). The gate scans 3tears source, and the host declares this bucket. Which storage does the collections bucket use today?
- How long must an older pod keep working against newer services? `docs/family-convergence.md` §5 lists "define a supported-version window so consumer skew is bounded" as a proposal. No recorded window was found; the subject-grant ledger dates each kept grant instead.
- The L3 broker, hub and gateway live outside this repo. Who owns the `l3.*` request bodies as a contract when a field changes? Today they are pinned in `packages/core/tests/unit/backends/` and the CHANGELOG (`docs/integration-guide.md` §3).
- No scale, latency or uptime targets are recorded (`.prawduct/project-state.yaml` `nonfunctional`). The only measured numbers are one developer-machine run (`docs/design-durable-coordination.md`, "Hot-path cost").
- `packages/media-contracts/` holds only `__pycache__` leftovers and no tracked files; the package was folded into `3tears-interfaces` (commit `12365eb696`). Should the directory be deleted?
