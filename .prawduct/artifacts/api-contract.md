---
artifact: api-contract
version: 1
depends_on:
  - artifact: product-brief
  - artifact: data-model
  - artifact: security-model
  - artifact: architecture
last_validated: null
---

# API Contract

## Summary

3tears exposes three programmatic surfaces to external consumers (`.prawduct/project-state.yaml` `exposes_programmatic_interface: library, external`):

| Surface | Published as | Canonical contract |
|---|---|---|
| Python library | `3tears*` on PyPI, import root `threetears.*` | each module's `__all__` (or its public top-level names), read by `threetears.enforcement.release.surface`; typed, mypy strict (`CLAUDE.md` Conventions) |
| NATS wire protocols | the same packages; spoken between separately deployed processes | Pydantic models in `threetears.core.tool_wire`, `threetears.registry.proxy` and peers; subject paths from `threetears.nats.Subjects` |
| Browser packages | `@3tears/web-*` on npm | each package's `exports` map and checked-in `types/*.d.ts` (`web/packages/pwa/package.json`; `web/package.json` `check`) |

One family version covers all three (`docs/releasing.md`). Two smaller surfaces sit beside them: a few CLIs, and the scrape sidecar's HTTP API, which carries its own version (`docs/sidecar-http-contract.md`). Topology and trust boundaries are in `architecture.md`.

## Overview & Surface Type

- **Library.** Public means: a module path with no `_`-prefixed component, and a name in its `__all__`, or a public top-level name it defines when it has no `__all__` (`packages/enforcement/src/threetears/enforcement/release/surface.py`). A leading underscore is never public (`CLAUDE.md`, "Leading underscores are a stability contract").
- **Wire.** A wire model is any Pydantic model a receiver validates from another deployable's bytes. The inventory of those is `_WIRE_MODELS` in `tests/enforcement/test_wire_models_are_tolerant_readers.py`.
- **npm.** `@3tears/web-pwa` is plain ES modules with no dependencies; React is an optional peer (`web/packages/pwa/README.md`; `web/packages/pwa/package.json`).
- **Consumers.** External. Host apps named in the docs include Discodon, metallm, scriob and the aibots platform (`docs/family-convergence.md`; `docker-bake.hcl`).

## Operations

### Library entry points

| Package | Main operations | Source |
|---|---|---|
| `threetears.core` | `CollectionRegistry.configure`, `DataStore` (`create_table`, `query`, `execute`, `run_migrations`), `BaseEntity.save`/`reload`, `BaseCollection.get`/`delete`/`invalidate_cache`, `MigrationRunner`, `KVLease` | `docs/integration-guide.md` §4, §15 |
| `threetears.nats` | `NatsClient.connect`/`publish`/`subscribe_typed`/`request`, `ensure_kv_bucket`, `Subjects` | `packages/nats/README.md` |
| `threetears.epoch` | `EpochClient.bump`/`advance_to`/`versions`, `EpochListener.subscribe`, `catchup_tick` | `packages/epoch/README.md` |
| `threetears.observe` | `get_logger`, `traced`, `set_context`, `contain`, `contained`, `cancel_and_wait` | `docs/observability.md`; `docs/exceptions.md` |
| `threetears.agent.tools` | `TearsTool`, `ToolResult`, `ToolServer`, `ToolRouter`, `ToolExecutor`, `parse_document` | `packages/agent/tools/README.md` |
| `threetears.registry` | `ToolCatalog`, `CallProxy`, `ToolCallClient`, `RbacEvaluatorAuthorizer` | `packages/registry/README.md` |
| `threetears.agent.audit` | `AuditEvent`, `publish_audit` | `packages/agent/audit/README.md` |
| `threetears.models` key pool | `KeyPool`, `KeySpec`, `Secret`, `KeyPool.status`/`reset`/`reserve`, `create_chat_model(model_id, key_pool=)` | `docs/model-key-pool.md`; [Model key pool surface](#model-key-pool-surface) |
| Every other package | the tables in `README.md` "Packages" and each package README | `README.md` |

### Model key pool surface

Added on `feat/model-key-pool`, unreleased; it ships in a minor (`docs/model-key-pool-requirements.md`, the closing
line). Operator guide: `docs/model-key-pool.md`.

| Where | Public names | For |
|---|---|---|
| `threetears.models` | `KeyPool`, `KeyPoolLease`, `KeySpec`, `Secret`, `KeyPoolStore`, `InMemoryKeyPoolStore`, `KeyPoolStoreError`, `NoKeyAvailableError`, `NoOutputLimitError`, `reservation_estimate` | building a pool, reserving on it, and its errors |
| `threetears.models.key_pool` | the names above, plus `KeyPoolStatus`, `KeyStatus`, `KeyState`; `KeyPoolObserver`, `KeyPoolCounters`, `MoveReason`; `AccountSnapshot`, `ReservationSnapshot`, `ReservationNotFound`, `ReservationAlreadyOpen`; `SelectionPolicy`, `StickySelection`, `KeyCandidate`; `KeyErrorClassifier`, `Verdict`, `VerdictKind`, `VerdictScope`, `classifier_for`; `period_bounds`, `period_id` | `status()`; the event hook metrics export through; the store Protocol's plain types; the policy and classifier seams; billing periods |
| `threetears.models` (`pricing`) | `cost_of_usage`, `TokenRates`, `UnpricedModelError` | the one place usage is priced: cache reads and writes at their own rates, falling back to the input rate, plus `cost_per_request` |
| `threetears.models` (`tracking`) | `estimate_input_tokens` | a local, network-free prompt estimate for a reservation |
| `threetears.models.errors` | `NO_KEY_AVAILABLE_REASON` | the `reason` `NoKeyAvailableError` carries, which `friendly_api_error()` maps to a fixed message |
| `threetears.models` (`UsageTracker.make_callback`, `UsageTrackingCallback`, `UsageAccumulator`) | new keyword arguments `cost_per_cache_read_token`, `cost_per_cache_write_token`, `cost_per_request` | the tracker and accumulator price through `cost_of_usage` |
| `create_chat_model` | new keyword argument `breaker_registry=`; passing it with `breaker=` raises `ValueError` | the breaker registry seam: keeps both the credential and the per-model rate-limit breaker, unlike `breaker=` |
| `threetears.models.circuit_breaker` | `CircuitBreaker.seconds_until_admit()`, `CircuitBreakerRegistry.peek()` | reading a breaker with no side effects: never admits a probe, and `peek` never creates a breaker or marks one used |
| `threetears.models.providers.anthropic`, `.openai`, `.openrouter` | `create_anthropic_chat`, `create_openai_chat`, `create_openrouter_chat` gain `unretried_key_errors=False` | `True` makes 401, 402, 403 and 429 reach the caller on their first occurrence; 5xx and connection retries keep `max_retries`. On OpenRouter it changes nothing, since that SDK never retries those. Refused on a Claude subscription token |
| `threetears.core.coordination` | `BudgetLedger` | the durable store: satisfies `KeyPoolStore` structurally, with no import of `threetears.models`; refuses a registry with no L2 or no L3 |
| `threetears.core.coordination.tables` | `CoordinationLedgerAccountsCollection`, `CoordinationDocumentsCollection`, `LEDGER_TABLE_SCHEMAS` | the ledger's two tables, created by coordination migration v004 |
| `create_chat_model` | `key_pool=`; `api_key` is now optional, and exactly one of `api_key` and `key_pool` is required (`ValueError` otherwise). With `key_pool`: `ValueError` for a pool of another provider or for `breaker`, `breaker_registry` or `unretried_key_errors`; `UnpricedModelError` for a model with no input or output price | the pooled chat model (KP-J1): reserves each call's estimate on the serving key, fails over between keys, settles the reported cost. Typed `BaseChatModel`; the object is a `PooledChatModel` |
| `threetears.models.pooled` | `PooledChatModel` | what `create_chat_model(key_pool=)` returns: a `Runnable` holding one model per key. Every re-binding (`bind_tools`, `with_structured_output`, `bind`, `with_config`, `with_retry`, `with_types`) applies to each key's model. The async paths run the pool on the caller's loop; the sync paths on a private loop, so a loop-bound store such as `BudgetLedger` needs the async paths |
| `threetears.models` (`tracking`) | `UsageRecord.credential_label`; `credential_label=` on `UsageTracker.make_callback` and `UsageTrackingCallback` | the label of the key a pooled call ran on, `None` for an `api_key=` model. Reaches spans as `llm.credential_label` and sinks; never a Prometheus label |
| `threetears.models.key_pool` | `KeyPoolMetrics` | the Prometheus exporter (KP-K1), behind the `prometheus` extra (`ImportError` without it): pass as `KeyPool(observer=...)`; counters move on events, gauges on `await metrics.refresh(pool)`, which returns the `status()` snapshot. Instruments in `docs/model-key-pool.md#metrics` |

Safety: reads are safe. `save()` is guarded by optimistic locking and raises on a stale `date_updated` (`docs/integration-guide.md` §4). Migrations are idempotent and keyed by `(version, package)` (§9).

### Wire operations

| Subject | Request → reply | Idempotency | Source |
|---|---|---|---|
| `{ns}.tools.register` | `RegistrationManifest` → `RegistrationResponse` (with `refused_tools`) | re-offered by every heartbeat | `packages/core/src/threetears/core/tool_wire/messages.py`; `packages/registry/README.md` |
| `{ns}.tools.heartbeat.{pod}` | `HeartbeatMessage`, publish only | yes | same |
| `{ns}.tools.discover` | readiness polling | yes | `packages/registry/README.md` |
| `{ns}.tools.call` | `ProxyCallRequest` → `ProxyCallResponse` | no; a `TOOL_TIMEOUT` may have reached the tool (`packages/registry/src/threetears/registry/proxy.py`) | `packages/registry/README.md` |
| registry → tool pod | `CallRequest` → `CallResponse`, or `CallAccepted` then a delivered result | no | `packages/core/src/threetears/core/tool_wire/messages.py` |
| `{ns}.audit.{event_type}` | `AuditEvent`, publish only | yes, by envelope `id` | `packages/agent/audit/src/threetears/agent/audit/envelope.py` |
| `threetears.cache.invalidate` | `CacheInvalidationMessage`, broadcast | yes | `packages/nats/src/threetears/nats/subjects.py` |
| epoch subjects | `EpochBumpMessage` (`subject_path`, `epoch`, `payload`), broadcast | yes; monotonic | `packages/epoch/README.md` |
| `<ns>.l3.query`, `l3.batch`, `l3.tx.*`, `l3.truncate`, `l3.load_files` | SQL request → rows or row count plus `generations` | per subject | `docs/integration-guide.md` §3 |

### CLIs

| Command | Contract | Source |
|---|---|---|
| `python -m threetears.registry` | reads `THREETEARS_NATS_URL`, `THREETEARS_NATS_SUBJECT_NAMESPACE`; refuses to build without an identity provider | `packages/registry/README.md` |
| Tool pod via `ToolServerBootstrap.run()` | exits 70 on a failed shutdown, 78 on bad config | `packages/agent/tools/README.md` |
| `threetears-release` | the release tool behind `./scripts/bump-version.sh` | `packages/enforcement/pyproject.toml`; `docs/releasing.md` |

## Inputs & Outputs

- **Entities and schemas.** `TableDef`, `ColumnDef`, `IndexDef`, `ForeignKeyDef` are validated Pydantic models. Identifiers match `^[a-z][a-z0-9_]*$`; column types are a fixed set (`docs/integration-guide.md` §9). Entity shapes belong in `data-model.md`.
- **Config.** `CoreConfig` is a `runtime_checkable` Protocol, so a host's own settings object satisfies it (`docs/integration-guide.md` §10).
- **Ports.** Where a package needs something from its host it asks for a small Protocol, and the host supplies an adapter (`README.md`, "Adopting a package"). Shared Protocols live in `threetears.interfaces` (`packages/interfaces/README.md`).
- **Wire identity.** Per-call identity rides one nested `CallContext` (`conversation_id`, `user_id`, `customer_id`, `correlation_id`, `agent_id`), each optional (`packages/core/src/threetears/core/tool_wire/context.py`).
- **Pagination.** Cursor-based: `Keyset`, `Page`, `encode_cursor`, `decode_cursor`, `CursorError` from `threetears.core` (`docs/integration-guide.md` §15). UUIDv7 ids keep primary-key cursors in creation order (`tests/enforcement/test_uuidv7_enforcement.py`).
- **Size bounds.** `PayloadTooLargeError` and `KvValueTooLargeError` (`packages/nats/src/threetears/nats/errors.py`); `l3.load_files` requests stay under `LOAD_FILES_MAX_WAIT_SECONDS` (`docs/integration-guide.md` §3).

## Error Model

`design_decisions.api_error_model_approach` is not recorded (`.prawduct/project-state.yaml`). The code does the following today.

| Surface | Shape | Vocabulary | Source |
|---|---|---|---|
| Library | exceptions; a failure reaches the caller or is recorded through `contain()` | per-package hierarchies, e.g. `ConcurrentModificationError`, `DataLayerUnavailableError`, `L2LostError`, `L2ScopeNotConfiguredError`; `NatsClientError` with `KvError`, `RequestTimeoutError` and peers | `docs/exceptions.md`; `packages/core/src/threetears/core/exceptions.py`; `packages/nats/src/threetears/nats/errors.py` |
| Missing extra | `ImportError` naming the extra | — | `packages/core/README.md` |
| Model key pool | exceptions; the pool fails closed and never returns `None` | `NoKeyAvailableError` (a `ModelProviderError`, reason `no_key_available`, naming every key's label and state); `KeyPoolStoreError`, with its subclasses `ReservationNotFound` (also a `LookupError`) and `ReservationAlreadyOpen` (also a `ValueError`); `UnpricedModelError` and `NoOutputLimitError` (both `ValueError`). `BudgetLedger` raises by builtin base (`LookupError`, `ValueError`, `TypeError`) and lets storage failures propagate for the pool to wrap | `packages/models/src/threetears/models/key_pool/errors.py`; `key_pool/store.py`; `packages/core/src/threetears/core/coordination/budget_ledger.py` |
| Tool result | `ToolResult.error_code`, copied to `CallResponse` and `ProxyCallResponse` | closed `TOOL_RESULT_ERROR_CODES` (`CONFLICT`, `MALFORMED_REQUEST`, `TOOL_NOT_READY`); upper case; `ValueError` at construction otherwise | `packages/agent/tools/README.md` |
| Registration | `RegistrationResponse.error_code`, `RefusedTool.code` | `RefusalCode` StrEnum | `packages/registry/src/threetears/registry/ownership.py` |
| Call path | `ProxyCallResponse.error_code`; `ToolCallError.error_code` on the client | codes such as `TOOL_TIMEOUT`, `TOOL_UNAVAILABLE`, `TOOL_NOT_AUTHORIZED`, `TOOL_AUTHORIZATION_UNAVAILABLE`, `REGISTRY_INTERNAL_ERROR` | `packages/registry/src/threetears/registry/proxy.py`; `packages/registry/README.md` |
| WebSocket | `{"type": "error", "code": ..., "message": ...}` | e.g. `UNAUTHENTICATED`, closed 1008 | `packages/channels/README.md` |
| Sidecar HTTP | `{"error": {"code", "message"}}`; FastAPI `{"detail": ...}` on 422 | open; an unknown code falls back on the status | `docs/sidecar-http-contract.md` §2 |

Security checks fail closed: an input they cannot classify is refused (`docs/exceptions.md`).

## Versioning

Recorded decision (`.prawduct/project-state.yaml` `design_decisions.api_versioning_approach`, 2026-11-22, status active):

- **Scheme.** One family version for every `3tears*` package, lockstep (`docs/releasing.md`). The npm packages carry the same version (`tests/enforcement/test_web_packages_follow_the_family_version.py`).
- **Bounds.** Every intra-family dependency is `>=X.Y.0,<X.Y+1.0` (`tests/enforcement/test_intra_family_version_bounds.py`).
- **Bump rule.** A public API addition ships in a minor; everything else in a patch; a major only on the owner's word (`tests/enforcement/test_api_growth_requires_a_minor_bump.py`; `packages/enforcement/src/threetears/enforcement/release/growth.py`).
- **Granularity.** The whole family. Consumers pin every package to one exact version (`README.md` "Versioning").
- **Runtime version.** Each `__version__` is read from installed metadata (`tests/enforcement/test_runtime_version_is_not_hardcoded.py`).

Versions on other surfaces:

| Surface | Handle | Source |
|---|---|---|
| NATS wire models | no per-message version; compatibility comes from tolerant readers | `packages/core/src/threetears/core/wire.py` |
| Byte pipe | `PIPE_PROTOCOL_VERSION` and `MIN_PIPE_PROTOCOL_VERSION`, both 1 | `packages/nats/src/threetears/nats/pipe_wire.py` |
| Sidecar HTTP | `/v1` path; bumped only by a breaking change; own package version | `docs/sidecar-http-contract.md` §3 |
| `3tears-evals` | per-name tiers in `packages/evals/api/surface.json`; moves recorded in `MIGRATIONS.toml` | `packages/evals/MIGRATIONS.toml` |

The growth gate cannot see a new field on an existing model, a route without a decorator, a NATS subject, or a method added to a private class a public function returns (`growth.py`).

## Deprecation & Compatibility

- **Library, family-wide.** Alpha: until `1.0.0` the public API can change between minor versions (`README.md` "Status"). No deprecation window beyond that is recorded (`.prawduct/owner-rulings.md`, 2026-11-22).
- **`3tears-evals`.** `host-contract` names move only with a deprecation cycle: the old name works beside the new one for one minor release and is dropped in the next. `stable` names may move in any release with a record. `provisional` names promise nothing (`packages/evals/MIGRATIONS.toml`).
- **Wire.** Senders and receivers upgrade on their own schedules: expand, migrate, contract (owner ruling 2026-10-07, `packages/core/src/threetears/core/wire.py`). Receivers ignore unknown fields and refuse named authority-steering fields. A grant the current code stopped using stays, dated, until no supported release uses it (`tests/enforcement/test_subject_grants_match_the_code.py`). A broker that predates an `l3.*` subject answers it with an error, so the broker upgrades first (`docs/integration-guide.md` §3).
- **Tool error codes.** A new code ships together with its face in the platform's error map (`packages/agent/tools/README.md`).
- **Sidecar.** Adding a route, a response field, an optional request field or an error code needs no bump. Unknown request fields are ignored (`docs/sidecar-http-contract.md` §3).

Retention: none family-wide before 1.0.0; removal may ship in any minor release (README Status). `3tears-evals` host-contract names keep a one-minor deprecation cycle (packages/evals/MIGRATIONS.toml).

## Surface Inventory & Stability Tiers

Tier meanings here: `stable` means no breaking change inside one minor line, and a minor may break it until 1.0.0 (`README.md` "Status"). `experimental` means the package says its API is still changing. Every `_`-prefixed module or name is internal and outside this list (`CLAUDE.md`).

- `threetears.core` — stable
- `threetears.nats` — stable
- `threetears.observe` — stable
- `threetears.epoch` — stable
- `threetears.interfaces` — stable
- `threetears.tool_schema` — stable
- `threetears.conversations` — stable
- `threetears.datasources` — stable
- `threetears.object_store` — stable
- `threetears.backup` — stable
- `threetears.geo` — stable
- `threetears.iam` — stable
- `threetears.registry` — stable
- `threetears.scheduled_jobs` — stable
- `threetears.langgraph` — stable
- `threetears.mcp` — stable
- `threetears.models` — stable
- `threetears.models.key_pool` — stable (new, unreleased)
- `threetears.channels` — stable
- `threetears.search` — stable
- `threetears.scrape` — stable
- `threetears.enforcement` — stable
- `threetears.agent.tools` — stable
- `threetears.agent.memory` — stable
- `threetears.agent.knowledge` — stable
- `threetears.agent.skills` — stable
- `threetears.agent.identity` — stable
- `threetears.agent.intention` — stable
- `threetears.agent.workspace` — stable
- `threetears.agent.acl` — stable
- `threetears.agent.audit` — stable
- `threetears.agent.wake` — stable
- `threetears.evals` (host-contract and stable names) — stable
- `threetears.evals` (provisional names) — experimental
- `{ns}.tools.*` wire (`RegistrationManifest`, `CallRequest`, `CallResponse`, `ProxyCallRequest`, `ProxyCallResponse`, `HeartbeatMessage`) — stable
- `{ns}.audit.*` wire (`AuditEvent`) — stable
- `threetears.cache.invalidate` wire (`CacheInvalidationMessage`) — stable
- epoch wire (`EpochBumpMessage`) — stable
- `<ns>.l3.*` wire — stable
- `@3tears/web-pwa` exports (`.`, `./worker`, `./workbox`, `./react`, `./vite`, `./shell-worker.js`, `./safe-area.css`) — stable
- sidecar `/v1/*` HTTP — stable

## Conventions

- **Timestamps.** `timestamptz` stores aware UTC; `timestamp` is naive (`docs/integration-guide.md` §9).
- **Ids.** Persisted entity ids are UUIDv7 (`tests/enforcement/test_uuidv7_enforcement.py`). Tool namespace ids are deterministic `uuid5` (`packages/agent/tools/README.md`).
- **Codes.** Error and refusal codes are upper-case strings; registry codes are a `StrEnum` (`packages/agent/tools/README.md`; `ownership.py`).
- **Subjects.** Typed `Subject` objects from the `Subjects` factory; callers never format subject strings (`packages/nats/README.md`).
- **Publish.** `publish` takes a `BaseModel`; raw bytes go through `publish_raw` (`packages/nats/README.md`).
- **Logging.** Each module logs through `get_logger(__name__)` with a `NullHandler` under the `threetears` hierarchy (`docs/observability.md`).
- **Async.** The data API is async end to end (`docs/integration-guide.md` §7).
- **Optional extras.** The extra's library is imported at module top under `try/except ImportError`, and code that needs it raises naming the extra (`docs/family-convergence.md` §5; `tests/enforcement/test_no_function_level_imports.py`).

## Security

Authentication and authorization belong to `security-model.md`. At the API boundary:

- Tolerant wire models refuse fields that would let a caller name its own authority (`refused_fields`), e.g. `customer_id` on `AuditAnonymizeRequest` (`packages/core/src/threetears/core/wire.py`; `packages/agent/audit/src/threetears/agent/audit/erasure.py`).
- The registry denies an agent's dispatch with no `user_id` and a tool whose namespace row is not yet visible (`packages/registry/README.md`).
- Every JWS verifier pins the algorithm in two stages (`tests/enforcement/test_jwt_alg_pinning.py`).
- `@traced(record_args=True)` redacts sensitive parameter names (`docs/observability.md`).
- `l3.load_files` fetches only presigned `https` URLs on the caller's export bucket and prefix (`docs/integration-guide.md` §3).
- A sidecar session token is a bearer capability over one session and carries no identity (`docs/sidecar-http-contract.md` §4).

## Conditional Patterns

- **Long-running calls.** A tool pod answers `CallAccepted` at once and publishes the result to a delivery subject later, so the registry can tell a busy pod from a gone one (`packages/core/src/threetears/core/tool_wire/messages.py`).
- **Concurrency.** Optimistic locking on `date_updated` raises `ConcurrentModificationError`; the host owns retry (`docs/integration-guide.md` §5).
- **Events.** Audit is at-least-once with `id` dedup (`envelope.py`). Epoch broadcasts are best-effort with catch-up polling (`packages/epoch/README.md`). Failed subscriber callbacks go to `{ns}.deadletter.{path}` (`packages/nats/README.md`).
- **Correlation.** `CallContext.correlation_id` crosses the registry hop, and log context rides every line and span (`context.py`; `docs/observability.md`).

## Direction

Ratified 2026-11-26 from declarations the owner already made; each Why cites where.

- **Every `3tears*` package releases at one version, and every intra-family dependency carries
  the bound `>=X.Y.0,<X.(Y+1).0` matching the declaring package's own version.** A bare sibling
  name in a `dependencies` list is a bug, not a shorthand.
  Why: unbounded siblings let pip resolve a mixed family: an install that builds clean and breaks
  at runtime, or a resolution failure that names the wrong package (both incidents in
  `docs/releasing.md`). `CLAUDE.md`: "This is a hard rule, not a style preference."; owner
  2026-11-22, "Yes to the API versioning policy" (`.prawduct/owner-rulings.md`).
- **An intra-family API addition ships in a MINOR bump.**
  Why: the minor-line bound above is what tells a consumer which siblings carry a name; an
  addition in a patch release breaks that promise. `CLAUDE.md` § The package family versions in
  lockstep; owner 2026-11-22, "Yes to the API versioning policy" (`.prawduct/owner-rulings.md`).
- **A leading underscore is a stability contract: nothing outside the defining class or module
  binds to it, in `src/` or in `tests/`, and no public API exists only for a test.** A test
  reaches an object through its constructor, its fixture or its behaviour. Inline
  `# noqa: SLF001`, per-file SLF001 ignores for tests, and aliases like `_helper = helper` are
  banned. Reaching into a third-party library's private members is confined to one module per
  library, recorded in `tests/enforcement/_underscore_exemptions.txt` with a specific rationale.
  Why: an underscore name is implementation detail; a binding from outside its owner turns every
  internal change into a break. Owner ruling 2026-10-01 (`CLAUDE.md` § Leading underscores are a
  stability contract).

- **A name added to a module's public surface has a caller in some package's `src` other than
  its own module, or names its outside caller in `_public_api_consumers.txt`.**
  Why: no public API exists only for a test; a name with no production caller is surface a
  release must carry for nobody (`CLAUDE.md` § Leading underscores). Gate:
  `tests/enforcement/test_public_api_has_a_caller.py`. Owner 2026-11-26 bulk ratification.
- **A wire model a separately deployed receiver reads is a `TolerantWireModel`, and only those
  models are.**
  Why: components and consumers upgrade on their own schedules (expand / migrate / contract,
  owner ruling 2026-10-07), so a receiver must ignore a field a newer sender adds while still
  refusing the fields that steer authority. Gate:
  `tests/enforcement/test_wire_models_are_tolerant_readers.py`. Owner 2026-11-26 bulk ratification.
- **The `web/` packages (`@3tears/*`) carry the Python family's version, and the release tool
  lists every one of them.**
  Why: one tag releases PyPI and npm together, so a package the release tool does not list would
  ship at a stale version. Gate: `tests/enforcement/test_web_packages_follow_the_family_version.py`.
  Owner 2026-11-26 bulk ratification.
- **A package's runtime `__version__` is read from installed metadata, never written.**
  Why: a release that bumps `pyproject.toml` without touching `__init__.py` cannot drift the
  runtime version. Gate: `tests/enforcement/test_runtime_version_is_not_hardcoded.py`. Owner
  2026-11-26 bulk ratification.
- **Managed model ids come only from `threetears.models.defaults`; no source or test pins a raw
  managed id.**
  Why: a model rev is then one edit in one module. Gate:
  `tests/enforcement/test_no_hardcoded_model_literals.py`. Owner 2026-11-26 bulk ratification.

## Candidate norms (unratified)

- Intra-family dependencies carry a minor-line bound — `tests/enforcement/test_intra_family_version_bounds.py`
- Public API growth requires a minor bump — `tests/enforcement/test_api_growth_requires_a_minor_bump.py`
- A new public name has a caller in some package's `src` — `tests/enforcement/test_public_api_has_a_caller.py`
- Nothing binds a private name outside its owner — `tests/enforcement/test_private_binding.py`, `tests/enforcement/test_underscore_access.py`
- Cross-deployable wire models are tolerant readers, and only those — `tests/enforcement/test_wire_models_are_tolerant_readers.py`
- Web packages follow the family version — `tests/enforcement/test_web_packages_follow_the_family_version.py`
- `__version__` is read from metadata — `tests/enforcement/test_runtime_version_is_not_hardcoded.py`
- Managed model ids come only from `threetears.models.defaults` — `tests/enforcement/test_no_hardcoded_model_literals.py`
- Logger calls pass no structlog keywords — `tests/enforcement/test_logger_call_kwargs.py`

## Open questions

- No error-model decision is recorded (`api_error_model_approach: null`). The tool-result vocabulary is closed and the sidecar's is open (`packages/agent/tools/README.md`; `docs/sidecar-http-contract.md` §2). Should the family record one rule?
- Wire changes are outside the growth gate (`growth.py`, "What it cannot see"). Should a new wire field or subject also require a minor bump?
- `AuditEvent` subclasses `TolerantWireModel` (`extra="ignore"`), while its module docstring and `packages/agent/audit/README.md` say `extra='forbid'`. Which is the intended contract?
- Should the family adopt a deprecation window like evals' host-contract cycle, or stay with the alpha notice until 1.0.0?
- Is the call-path code set on `ProxyCallResponse.error_code` closed, like `TOOL_RESULT_ERROR_CODES`, or open? No declared vocabulary for it was found in `proxy.py`.
- Is each package listed `stable` above meant to be? Only evals declares its own tiers; the others inherit the family's alpha notice.
