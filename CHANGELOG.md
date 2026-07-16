# Changelog

All notable changes to the 3tears platform packages are recorded here.
This project follows semantic versioning across all 21 workspace
packages (bumped in lock-step).

## v0.17.1 -- 2026-07-16

**Two additions that were written the same day as v0.17.0 but were left unmerged on
feature branches and missed that release. Both land here instead.**

- **`ToolRelevanceIndex` + the `tool_search` meta-tool (`packages/agent/tools`,
  `relevance.py`).** Embeds and ranks a tool catalog against the current turn's query,
  returning the top-k most relevant tools with an LRU cache keyed on the catalog
  identity; a `tool_search` `BaseTool` wrapper lets a model reach anything filtered out
  of the initial top-k on demand. Falls back to the full, unfiltered catalog on any
  embedding failure or when ranking exceeds a configurable latency ceiling -- a
  degraded turn is never worse than today's full-catalog behavior. This is the
  platform primitive metallm's own dynamic tool-relevance selection consumes.
- **`acting_as_principal_id` on `AuditEvent`** (`packages/agent/audit`, `envelope.py`).
  `14-eng-ai-bot-identity`'s impersonation flow (`identity.impersonation.start`/`stop`)
  needs to record both the impersonation TARGET (`actor_user_id`, whose session it is)
  and the ADMIN actually driving it. Previously that producer carried the admin
  identity in `details["acting_as_principal_id"]` -- works on the wire, but isn't a
  typed, Hub-queryable column. Additive only: optional, defaults to `None`, every
  existing producer unaffected.

## v0.17.0 -- 2026-07-15

**Support for `14-eng-ai-bot-identity`, the platform's new NATS-native multi-tenant
identity broker.** Four additions to `packages/core` and `packages/agent/acl`, built and
landed across identity-core's own build (chunks 03/05/13), consumed there via a
temporary local-path override while this release was pending:

- **`jwk_thumbprint()` (`packages/core`, `security/identity_token.py`) now accepts
  `EllipticCurvePublicKey`, not just Ed25519.** Extends the RFC 7638 thumbprint to the
  EC required-member set (`crv`, `kty`, `x`, `y`, via PyJWT's `ECAlgorithm.to_jwk`) --
  needed for DPoP proof validation binding a P-256 client key. The existing Ed25519
  branch is unchanged, verified byte-identical against a pinned vector.
- **`RevocationGuard` (`packages/core`, `coordination/replay_guard.py`), a new sibling to
  `ReplayGuard`.** Where `ReplayGuard.record_unique`'s presence-only sentinel fits a
  single-use nonce or an exact `jti`/`sid` revocation, a `sub` (principal) or
  `customer_id` (tenant) revocation needs a value comparison, not membership: record a
  `revoked_at` timestamp per key, then `is_revoked_before(key, moment=...)`. Fail-closed
  on KV transport failure, same durability posture as `ReplayGuard`.
- **`WindowedCounter` (`packages/core`, `coordination/windowed_counter.py`), a new
  generic throttle primitive.** A windowed attempt counter over a NATS JetStream KV
  bucket (`record_attempt`/`count`/`is_over_threshold`) for a "how many times in the
  last N seconds" shape neither `ReplayGuard` nor `RevocationGuard` express. Fail-open
  vs. fail-closed is a constructor-level caller choice (`fail_open: bool`, default
  `False`), since a throttle counter doesn't always sit on a hard security boundary.
- **`authorize_from_claims` + the impersonation gate schema (`packages/agent/acl`).** A
  claims-aware authorization entry point layering an impersonation deny-list overlay
  on top of the existing `authorize()`: denies unconditionally when
  `act_reason == "impersonation"` and the caller names a sensitive
  `ImpersonationCategory`, otherwise defers as normal. `ImpersonationGateCollection`/
  `ImpersonationGateEntity` add the per-tenant `disabled|requested|enabled` + TTL gate
  schema, with read-time TTL self-revert. Real Hub-side wiring (a live NATS responder
  persisting this collection against Postgres) is not part of this release --
  identity-core's own test suite proves the wire contract against a local fake double.

## v0.16.1 -- 2026-07-15

**Real token-level streaming for the Claude Max subscription backend.** `ClaudeCodeChatModel._astream`
(`langchain-claude-code` 0.1.0) requests `include_partial_messages=True` from the Claude Agent SDK --
which makes the subprocess emit granular `StreamEvent` text deltas -- but the method only ever
consumed the terminal, whole-block `AssistantMessage`, silently dropping every delta. A subscription
turn arrived as one or two large lumps instead of a real token stream.

- **`_SubscriptionChatModel._astream`** (`_claude_cli.py`, alongside the existing `_build_options` /
  `_wrap_langchain_tool` overrides for other upstream gaps in this same package) now consumes
  `StreamEvent` text deltas and yields each one immediately as it arrives, tracked per content-block
  index so the terminal `AssistantMessage` never re-yields (and thereby doubles) text a delta already
  streamed. A block that produces no `StreamEvent` at all (older CLI build, future SDK regression)
  still gets its text emitted whole from the `AssistantMessage` -- strictly additive, never worse than
  before.
- Verified against a real Claude Max subscription session: a response streamed in 13 chunks over
  ~10.6s (visible incremental delivery), versus 1-2 chunks arriving all at once under the prior
  behavior.

## v0.14.1 -- 2026-07-06

**Refreshing NATS connect credentials + per-key tool-pod identity.** A connection's auth
credential is no longer a single string captured at connect and re-presented (stale) on
every reconnect — it is a PROVIDER re-invoked each (re)connect, so a short-lived self-minted
identity token is re-minted fresh and the connection never wedges when the credential
expires mid-session.

- **`NatsClient.connect(auth_token=...)` is now a token PROVIDER** (`Callable[[], str]`,
  invoked by nats-py on every (re)connect) rather than a static `str`. Static-credential
  services wrap their token in `static_token_provider`; self-minting principals pass a
  provider backed by `IdentityMinter`.
- **`IdentityMinter`** (`threetears.core.security`) — holds a custody Ed25519 key and
  self-mints short-lived EdDSA identity JWTs (the stateful counterpart to the pure
  `sign_identity_token`), for a pod/agent/tool-pod to present as its connect credential.
- **`NatsClient.is_healthy`** reports `False` when a connection is stuck in a persistent
  Authorization-Violation reconnect loop (a rejected credential the forever-reconnect rides
  forever without closing), so a `/healthz` keyed on it lets k8s restart the pod.
- **Tool-pod per-key identity, both auth layers.** `ToolServer` accepts an `auth_token`
  provider so a tool pod self-mints its connect JWT for the NATS auth-callout, and carries
  the same JWT on its registration manifest. The registry `ToolPodAuthenticator.verify_pod`
  now takes the RAW JWT (was a token hash); `RegistrationHandler` verifies token-bearing
  manifests and admits tokenless (agent-owned in-process) pods.

**Cross-worker cancellation primitives (additive).** A WS-streaming consumer that runs turns
as fire-and-forget tasks now has a platform primitive to stop one — locally or on whichever
worker holds it — instead of hand-rolling a registry + NATS routing. Purely additive; no
existing signature changed.

- **`threetears.core.KeyedTaskRegistry`** — a per-worker registry of cancellable
  `asyncio.Task`s keyed by `UUID` (`register`/`pop`/`get`/`discard`, identity-guarded). Keeps
  a fire-and-forget task's handle reachable so it can be cancelled by key. `pop` is
  pop-before-cancel (a redelivered cancel is a clean no-op).
- **`threetears.nats.CrossWorkerCanceller` + `TaskCancelEnvelope`** — wraps a
  `KeyedTaskRegistry`; `request_cancel(key, payload)` cancels the task locally when this
  worker owns it, else publishes on a consumer-supplied broadcast `Subject` so the OWNING
  worker cancels. On cancel it invokes a consumer `on_cancel(key, payload)` callback with an
  **opaque** payload — the primitive knows nothing about locks/frames/checkpoints; all product
  semantics stay in the callback and the cancelled coroutine's own `finally`. `registry` is a
  required constructor arg so `threetears.nats` keeps `threetears.core` a type-only dependency.
  (Mirrors the `channels` `RoomFanout` publish-one/act-on-receive-per-pod pattern, specialised
  to cancellation. First consumer: metallm's Stop button.)

**Fixes (enforcement debt on this branch).**
- `IdentityMinter` per-mint session id now uses `uuid7` (time-ordered), satisfying the
  uuidv7 enforcement.
- The provider `NameTranslatingChatMixin`'s `_name_reverse_map` type hint is now
  `TYPE_CHECKING`-guarded, so the (pydantic-required) per-subclass `PrivateAttr` declarations
  are no longer flagged as shadowing a base private. No runtime change.

## v0.13.11 -- 2026-07-02

The **scope-and-objects** framework family: the huge-object offload backend and
the general engagement re-authorization seam that pentest scan scope is built on.

- **Object offload (Path-2).** A streaming S3/MinIO `ObjectStore` with scope-first
  keys (`object-store`); a langgraph offload seam that streams large tool results
  to the store and threads an `ObjectHandle` through the graph, plus tool-authored
  offload summaries via `content_and_artifact` (`langgraph`); the pod-side produce
  seam and a `build_s3_object_store` secret-ref-resolving wiring helper
  (`agent-tools`); the object-catalog NATS subjects — `hub_object_commit` /
  `hub_object_resolve` — and `list_entries` (`nats`, `object-store`); a general
  report tool that renders to the store and a general deliver tool that resolves an
  object id to a presigned URL, both pod-side and identity-token authed against the
  verified tenant (`agent-tools`); and a `BIGINT_TYPE` column tag for int8 columns
  (`core`).
- **Engagement scope (ES-1/ES-5).** The `hub_engagement_scope` NATS subject and the
  pod-side engagement re-authorization resolver + scope-injection seam (`nats`,
  `agent-tools`), and an `engagement_provider` carried on `BootstrapContext` so the
  runtime can auto-stamp the active engagement onto outgoing tool calls
  (`langgraph`). The framework treats `target_type` as an opaque string; the pentest
  domain interprets it.

## v0.13.10 -- 2026-06-29

Fixes the platform-wide **1-hour agent cliff**: every long-lived agent pod went
dead ~1h after boot because the auth-callout's NATS user JWT (default 3600s TTL)
expired while connected, and the NATS server's auth `-ERR` is routed by nats-py
straight to a terminal `close` that bypasses `_attempt_reconnect` — so
forever-reconnect (the network-drop path) never recovers it, and host daemons have
no k8s liveness net.

### Added

- **`NatsClient.reconnect()`** (`threetears.nats`) — a force-reconnect primitive
  that drives nats-py's own `_attempt_reconnect` on a still-connected client
  (synthesizing a `StaleConnectionError` through `_process_op_err`), so the
  transport cycles and the server auth handshake re-runs — under decentralized
  auth this re-runs the Hub auth-callout, minting a **fresh user JWT with full
  TTL** — while subscriptions replay under their original `sid` and the same
  underlying client object is reused (consumers holding `.raw` stay valid).
  Raises on an already-closed client; no-ops when not currently connected (so it
  never trips `_process_op_err`'s connection-closing else-branch).

This is the primitive the SDK uses for **proactive NATS-JWT re-auth**: forcing a
reconnect a margin *before* expiry, while the current JWT is still valid, so the
connection never reaches the terminal auth-expiry close. (The SDK-side re-auth
loop, the Hub's TTL-in-handshake reporting, and the env-overridable TTL knob ride
on this primitive and live in the consumer repos.)

### Fixed

- Three pre-existing over-strict third-party-stub `mypy` errors in
  `threetears.nats.client` (the gate-excluded `nats` package is now mypy-clean):
  the wrapper's float `flush()` timeout (nats-py annotates `int` but waits via
  `asyncio.wait_for`, which accepts float) and `_subscribe_internal`'s `Subject |
  str` subject (it already coerces `str`).

## v0.13.9 -- 2026-06-28

Platform-wide authentication lands and is **enforced**. The NATS bus is
fail-closed (an anonymous connect is rejected); every tool call carries a
Hub-issued, cryptographically-bound caller identity; and RBAC evaluates the
**verified** identity rather than a self-asserted envelope field. Shipped
enforce-only — no warn rung, no `no_auth_user`. Also: first-class
human-in-the-loop interrupts, an `engagement_id` identity dimension, and a
Kubernetes-resilience pass across the NATS + identity-verifier layer.

### Added — platform auth (A: NATS connection auth)

- **Auth-callout connection auth.** A connecting agent/tool pod presents its
  bootstrap token; the Hub's auth-callout responder resolves the principal and
  mints a **least-privilege, per-principal NATS user JWT** (`threetears.nats`:
  `user_jwt`, `auth_callout`, `subject_permissions`). Each principal's pub/sub
  allow-list is scoped to its own identity-bound subjects + reply inbox — no bare
  `>` wildcards, no cross-tenant KV/stream reach.
- Ships **enforce-only**: `no_auth_user` removed, anonymous connect rejected
  (`Authorization Violation`); platform services authenticate with per-service
  static users.

### Added — platform auth (B: identity tokens + crypto binding)

- **Hub-issued identity tokens** — EdDSA/Ed25519 JWS, alg-pinned, published via
  JWKS over NATS request/reply, minted at the bootstrap handshake and attached to
  every outgoing `CallContext` (`threetears.core.security`: `identity_token`,
  `jwks_provider`).
- **Verify-and-re-stamp at the registry proxy AND the tool pod** — the verified
  agent/user/customer overwrite the envelope, so RBAC authorizes the verified
  identity, never a self-asserted one. Fail-closed.
- **Crypto binding (DPoP-style).** A per-pod proof-of-possession key binds each
  call (`cnf` + `ath` + body-hash + single-use nonce, replay-guarded); the proxy
  mints a body-bound `proxy_assertion` (Ed25519 JWS, `aud=pod_id`) the tool pod
  verifies (`threetears.core.security`: `pop`, `proxy_assertion`, `replay_guard`).

### Added — HITL + identity

- **Human-in-the-loop interrupt surfacing** in `threetears.langgraph` streaming:
  a LangGraph `interrupt()` emits a `StreamInterruptEvent` terminal and stashes
  `__interrupt__` instead of an empty end, so an approval gate can pause and
  resume via `Command(resume=)`. Additive — uninterrupted graphs end as before.
- **`engagement_id`** promoted to a first-class typed `CallContext` field.

### Changed — Kubernetes resilience

- **Identity-token refresh lifecycle** — pods re-handshake before expiry reusing
  the pop key (cnf intact), so tool-calling survives past the token TTL.
- Forever-retry startup for critical bindings (never flip ready with a dead
  handler); honest liveness/readiness (real `ping()` + a `jwks_warmed` gate);
  effectively-infinite NATS reconnect; reactive JWKS self-heal on a kid-miss;
  guarded background loops. Built for undefined start order, N replicas, and pod
  movement, not a later resilience pass.

### Fixed

- `timezone_converter` resolves `"now"` itself instead of requiring the caller to
  supply the current datetime — the tool carries the value, the caller never
  infers it.
- Three registry proxy tests import their shared dispatch helper relatively, so
  the canonical full-suite run (`pytest packages/ tests/`) collects cleanly, not
  only per-package.

## v0.13.8 -- 2026-06-24

On cancel (e.g. a datasource tool-call timeout) the Redshift driver aborted the
query by closing the client connection — but closing the **client** socket does
not kill the running **server-side** Redshift query. A real abandoned query ran
on the cluster for **7.4 hours**, leaking a connection-pool slot the whole time
and re-exhausting the small pool faster than it could drain, which silently
stopped an agent from answering.

### Fixed — `3tears-datasources` — `RedshiftDriver` cancellation

- **The driver now captures each connection's `pg_backend_pid()` at open and, on
  cancel, issues `pg_terminate_backend(<pid>)` from a fresh short-lived
  connection** before closing/evicting the poisoned connection. Closing the
  client socket alone left the query running server-side; terminating the backend
  actually stops it. (The DB user need not be a superuser — `pg_terminate_backend`
  on one's own session works where `CANCEL` does not.)
- **Best-effort and non-fatal throughout.** The pid read at open is best-effort
  (a failure only degrades the server-side cancel; the connection stays usable).
  The terminate runs in a worker thread under `wait_for`, logs on success,
  logs + bumps the existing `cancellation.failed` counter on failure, and never
  raises — the client-socket close + evict path runs regardless.
- Pairs with consumers capping each datasource's `query_timeout_seconds` at its
  tool-call timeout: that bounds queries that **respect** `statement_timeout`;
  this terminates the ones that **wedge past** it.

## v0.13.7 -- 2026-06-23

NATS is the **L2** tier in 3tears — ephemeral, with durability riding JetStream
R3 replication plus the consumer's real L3 (git/DB). The JetStream helpers,
however, defaulted to **file** storage, so any consumer running against a
deliberately memory-only NATS deployment failed at first KV/stream creation with
`10047 insufficient storage resources available` (it surfaced as a 500 on the
first collections L2 access — presence join, entry read).

### Fixed — `3tears-nats` / `3tears` — JetStream storage now defaults to memory

- **`NatsClient.kv_bucket` and `NatsClient.ensure_jetstream_stream` now default
  `storage="memory"`** (was `"file"`); `NatsKvBucket.__init__` matches. `"file"`
  remains available as a deliberate, explicit opt-in for the rare object that
  genuinely needs on-disk durability.
- **`core.cache.NatsKvClient` no longer forces the `collections` bucket to
  `file`** — it now uses the `BucketConfig` memory default. This is the bucket
  whose file-backed creation failed on a memory-only account.
- Net effect: a consumer on a memory-only NATS (no file store, `max_file: 0`)
  works out of the box; nothing has to opt into memory. File storage is now the
  conscious exception, matching the L2 contract.

## v0.13.6 -- 2026-06-23

Closes a permanent-staleness race in the cross-pod config-epoch machinery
that any consumer loading local state before subscribing could hit -- it
surfaced as a gateway serving a model catalog that contradicted the admin
API, and the same shape sat latent in the MCP grant cache.

### Fixed — `3tears-epoch` — `threetears.epoch.listener`

- **`EpochListener.subscribe` gains an optional `primed_epoch` parameter so a
  consumer that loaded local state before subscribing can never go permanently
  stale.** `subscribe` primed its per-subject last-seen by reading
  `EpochClient.current()` at subscribe time. A consumer that loads local state (a
  model catalog, a grant cache) and only then subscribes therefore primed
  last-seen to whatever epoch had committed by subscribe time — which can be
  AHEAD of the epoch the loaded state actually reflects. A bump landing in the
  load→subscribe window then pins last-seen past the loaded state, the periodic
  `catch_up` sees `current == last_seen` and never fires, and the consumer serves
  stale state forever with no recovery path. The fix is additive and
  backward-compatible: pass `primed_epoch` = the epoch the loaded state reflects
  (read `current()` BEFORE the load, then load, then subscribe). last-seen is then
  never ahead of the loaded state, so any bump at or after the load is detected
  (broadcast or `catch_up`); worst case is one redundant reload, never permanent
  staleness. Omitting `primed_epoch` preserves the prior `current()`-at-subscribe
  behaviour — correct only when no state was loaded against an earlier epoch.

### Fixed — `3tears-mcp` — `threetears.mcp.auth`

- **`LocalGrantAuthorizer.start` reads the rbac epoch BEFORE reloading the grant
  cache and primes the listener to it.** `start` reloaded the grant cache and
  then subscribed, so a `mcp.rbac` bump committing in that window pinned the
  listener's last-seen past the freshly-loaded grants and the catch-up tick
  (`current == last_seen`) never recovered it — the authorizer could serve a
  permanently-stale grant set, making default-deny RBAC decisions on revoked or
  stale grants. It now reads `current()` before the reload and passes it as
  `primed_epoch`, mirroring the gateway catalog fix. Also asserts the listener is
  non-None in the catch-up loop (only ever spawned under epoch mode), closing a
  latent `union-attr`.

## v0.13.5 -- 2026-06-22

Closes the remaining gaps that surfaced while converging a host app's bespoke
tool loop onto `build_tool_agent`: wire-side tool-name translation leaking
through every non-`astream` chat surface, the agent node force-hoisting a
system message a caller had already assembled, a hook emitter able to abort a
turn, and `SqlL3Backend` dropping namespace + `customer_scope` when it wraps a
scope-aware transport.

### Fixed — `3tears-core` — `threetears.core.backends`

- **`SqlL3Backend` now forwards `namespace` + transport kwargs (e.g.
  `NatsProxyL3Backend`'s `customer_scope`) to a scope-aware wrapped pool instead
  of dropping them.** The collection registry wraps any non-`DurableStore` L3
  transport in `SqlL3Backend` to add the structured CRUD layer — including
  `NatsProxyL3Backend`, which is a raw-SQL-over-NATS transport with no
  `fetch_one`/`upsert`. But `SqlL3Backend`'s raw-SQL methods
  (`fetch`/`fetchrow`/`fetchval`/`execute`) were written to wrap a bare asyncpg
  pool: they silently dropped `namespace` and had no `customer_scope` parameter.
  So an agent-SDK scoped/RBAC read through a wrapped `NatsProxyL3Backend` either
  raised `TypeError: unexpected keyword argument 'customer_scope'` or lost
  namespace scoping; ordinary collection ops survived only via a default-namespace
  fallback. The wrapper now detects a scope-aware pool via an `accepts_scoped_reads`
  capability marker (an identity check, **not** `isinstance` — `NatsProxyL3Backend`
  omits `fetchval` and so does not satisfy the `L3Backend` protocol structurally,
  which would make an `isinstance` gate silently fail) and forwards `namespace` plus
  any extra kwargs generically via `**kwargs`, staying ignorant of NATS-specific
  concepts. A bare asyncpg pool lacks the marker, so its behaviour is unchanged.
  Pre-existing since the per-call `customer_scope` channel landed; it affected the
  agent SDK (knowledge retrieval + RBAC visibility through the proxy), not the hub
  (which passes a raw asyncpg pool that `SqlL3Backend` is built to wrap).

### Fixed — `3tears-models` — `threetears.models.providers`

- **`_NameTranslatingChat{OpenRouter,Anthropic}` now un-mangle tool-call names on
  every public surface — `ainvoke` / `invoke` and `agenerate` / `generate` — not
  just `astream` / `_agenerate`.** The wrappers reverse-translate names from the
  underscored wire form (forced by strict provider validators) back to the
  canonical dotted form. That happened in the public `astream` override and in
  `_agenerate`, but `BaseChatModel.ainvoke` aggregates from the PROTECTED
  `_astream` (not `_agenerate`) whenever `_should_stream()` is true — i.e. a
  streaming callback is attached, as when running under an `astream_events` tap —
  so both overrides were bypassed and underscored names reached consumers whose
  tool-dispatch maps key on the dotted canonical name, causing silent tool-call
  misses. Both providers now override the public `ainvoke` / `invoke` to
  un-mangle the returned message (mirrors the `astream` strategy; overriding
  `_astream` directly would drop `on_chat_model_stream` callbacks), and override
  the batch `agenerate` / `generate` chokepoint (which `abatch` and direct
  callers route through, and which also aggregates from `_astream` under
  streaming) to reverse-translate every generation. All passes are idempotent
  with `_agenerate`'s translation (`reverse_translate_message` keys on the
  underscored wire name, so a second pass is a no-op). Regression tests on both
  providers force the `_astream` aggregation path (`stream=True`) and assert the
  dotted name on each surface.

### Added — `3tears-langgraph` — `threetears.langgraph.nodes`

- **`agent_node` honours a `preassembled_messages` flag.** A host app that has
  already assembled the full message list — system prompt, history, and a
  trailing post-history injection in a deliberate position — was having its
  ordering rewritten by the node's default system-message hoist/merge. With the
  flag set, the node passes the messages through untouched (no hoist, no
  `str()` coercion, no merge), letting the caller own prompt assembly while still
  using the converged loop. Default behaviour is unchanged.

### Fixed — `3tears-langgraph` — `threetears.langgraph.hooks`

- **`_ComposedToolNodeHook` no longer lets a hook emitter abort a turn.** The
  composer now wraps `on_tool_start` / `on_tool_end` / `on_heartbeat` emitter
  dispatch in a guard that logs and swallows exceptions (a dispatch with no run
  context, a transient event-bus error) instead of propagating them out of the
  tool node and crashing the turn. `GraphBubbleUp` (LangGraph's control-flow
  signal for interrupts/commands) is re-raised first so the guard never
  swallows legitimate graph control flow.

## v0.13.4 -- 2026-06-22

Adds the op-log stream-head read a consumer needs to reconcile its
committed-through cursor against an external record when the two diverge.

### Added — `3tears-nats` — `threetears.nats.oplog`

- **`OpLog.last_seq() -> int`** — the stream's current head sequence (one `stream_info()`
  read, O(1)): the seq the next `append` will follow, i.e. the value `expected_last_seq`
  must equal. A consumer that derives the op-log head from an external record (e.g. a git
  `Op-Seq` commit trailer) can clamp to this when the two diverge — a reset/fresh stream
  sitting behind an ahead-of-it external record — instead of wedging on a CAS that can never
  match the shorter stream.

## v0.13.3 -- 2026-06-21

Completes the conversation-folder relationship (referential integrity + helpers),
hardens the write-buffer flush against orphaned writes, and adds a per-dimension
column-coverage probe to datasources.

### Added — `3tears-conversations` — `threetears.conversations`

- **Folder referential integrity (migration v009).** A `UNIQUE(folder_id)` on `folders`
  plus an FK `conversations.folder_id → folders.folder_id` **ON DELETE SET NULL`, so
  deleting a folder auto-unfiles its conversations at the DB level (no consumer can
  forget the unfile). `ConversationsCollection.clear_folder(agent_id, folder_id)` — the
  cache-coherent unfile-all (routes each conversation through `save_entity` so L1/L2 are
  invalidated, vs a raw L3 UPDATE) — and `count_by_folder(agent_id, folder_id)`, the cheap
  per-folder count peer of `find_by_folder`.

### Fixed — `3tears` (core) — `threetears.core.collections.flush`

- **Orphaned writes no longer poison the atomic batch.** `flush_pending` now partitions the
  drained buffer by retry count: only never-failed writes (`retries == 0`) enter the atomic
  batch; any write that has already failed (`retries > 0`) — e.g. an orphan whose FK parent
  was deleted and will never return — routes straight to the per-entity loop. Previously one
  un-satisfiable write aborted the whole transaction every cycle until it exhausted its
  ~100-retry budget, forcing per-entity fallback for ALL co-buffered writes. The per-entity
  safety net + FK-aware re-enqueue is unchanged.

### Added — `3tears-datasources` — `threetears.datasources`

- **Per-dimension column-value coverage.** `Driver.column_value_coverage_by_dimension(schema,
  table, dimension_column, columns)` — the grouped sibling of `column_value_coverage`: one
  `GROUP BY` pass reporting non-null/non-zero coverage per numeric column per dimension value,
  so a caller can see a column loaded for some dimension values but all-zero for others (the
  partial-coverage case the whole-table probe can't see). Concrete on the `Driver` ABC
  (portable SQL routed through `fetch`), so every backend inherits it.

## v0.13.2 -- 2026-06-21

Conversation folders (a reusable grouping primitive lifted from metallm) and a Redshift
connection-concurrency cap. Additive across `3tears-conversations` and `3tears-datasources`.

### Added -- `3tears-conversations` -- `threetears.conversations`

- **Folder system** — `Folder` entity + `FolderCollection`: an app-agnostic, mutable, per-owner
  named container that groups conversations, lifted from metallm's product-side feature so any
  3tears consumer reuses one canonical entity. Scoped per `(agent_id, folder_id)` with a `name` and
  a free-form `metadata` JSONB (app presentation: color/icon/sort_order). Adds the `folders` table
  and a nullable `conversations.folder_id` (migration v008).

### Changed -- `3tears-datasources` -- `threetears.datasources`

- **Cap simultaneously-open Redshift connections.** A burst of N concurrent `fetch()` could open N
  connections past the warehouse user's `CONNECTION LIMIT` even after the 0.13.1 bounded-cache fix.
  An `asyncio.Semaphore` sized to `connection_cache_size`, acquired before opening, now bounds
  concurrently-open connections to the cache size (the executor still bounds concurrent work).

## v0.13.1 -- 2026-06-21

Patch: size the Redshift warm-connection cache as a bounded pool so concurrent
queries reuse warm connections instead of overshooting a tight per-user Redshift
CONNECTION LIMIT.

### Fixed -- `3tears-datasources` -- `threetears.datasources`

- Redshift warm-connection cache is now a bounded pool. `executor_max_workers`
  previously defaulted to 10 while `connection_cache_size` defaulted to 3, so
  concurrent queries past the cache opened a fresh connection every time — which
  overshoots a tight per-user Redshift CONNECTION LIMIT and fails with "too many
  connections" (the cache never acted as a pool). Now `executor_max_workers`
  defaults to 5 and `connection_cache_size` defaults to `executor_max_workers`
  (cache == workers) via a model validator, so queries past the bound queue on the
  executor and reuse warm connections rather than opening doomed ones. Set both
  per datasource to the user's connection limit.

## v0.13.0 -- 2026-06-21

### Changed — `3tears` (core) — BREAKING

- **Neutral L3 store seam (`collections-task-06`).** The collection framework's L3
  (durable) tier extension points were renamed to be storage-agnostic so a non-SQL
  backend (e.g. a git working tree) can be an L3: `fetch_from_postgres` →
  `fetch_from_store`, `save_to_postgres` → `save_to_store`, `delete_from_postgres` →
  `delete_from_store`, `persist_to_postgres` → `persist_to_store`. Behavior unchanged;
  `SchemaBackedCollection` generates the new names. `l3_pool`/`get_l3_pool` and
  `serialize`/`deserialize` are unchanged. **Consumers: see
  `docs/migrating-to-l3-store-seam.md`.**

- **Atomic write-buffer flush (`collections-task-06` L3B-04) — BREAKING for hand-rolled
  overrides.** `flush_pending` now persists a toposorted batch inside ONE backend
  transaction (degrading to the per-entity loop when the backend has no usable
  `transaction()`). To carry the transaction handle, `BaseCollection.persist_to_store`
  calls `save_to_store(data, *, conn=...)`. The base + `SchemaBackedCollection` signatures
  already accept `conn`, so the schema-backed collections are unaffected — but any
  **hand-rolled `save_to_store` override** with the old signature now raises
  `TypeError: ... unexpected keyword argument 'conn'` on every flush. Fix by migrating the
  collection to `SchemaBackedCollection` (preferred) or threading `conn` through the
  override. **Consumers: see `docs/migrating-to-l3-store-seam.md`.**

### Fixed — `3tears` (core)

- **L2 serde now round-trips `NUMERIC` columns as `Decimal`.** The schema-backed
  L2 (JSON) codec handled `UUID`/`datetime`/`bytes` but not `Decimal`, so
  serializing any row with a `NUMERIC_TYPE` value (a money/metric column) raised
  `TypeError: Object of type Decimal is not JSON serializable` — and the decode
  side had no `NUMERIC` branch either, so a value would have come back as a bare
  string/number rather than `Decimal`. `json_default` now emits `Decimal` as its
  exact decimal string and `decode_l2_value` rehydrates a `NUMERIC` column back to
  `Decimal` (via `Decimal(str(value))`, tolerating a float/int producer too — app
  code that computes a cost as a `float` round-trips losslessly without the
  binary-float expansion `Decimal(float)` would introduce). Surfaced by metallm
  migrating its cost-bearing collections onto `SchemaBackedCollection`.

### Changed — `3tears-nats` — BREAKING

- **`deadletter_on_error` → `deadletter_on_failure`.** The `NatsClient.subscribe` /
  `subscribe_typed` parameter (and the subscribe log field of the same name) was renamed so
  benign config field names no longer read as errors in log/alert greps. Behavior is
  identical — it still controls whether a callback/validation failure republishes to
  `{ns}.deadletter.{subject}`. Update any call site that passes the keyword explicitly
  (`grep -rn deadletter_on_error`); callers relying on the `True` default need no change.

### Added — `3tears` (core)

- `threetears.core.backends.L3Backend` (raw-SQL transport) and `DurableStore` (SQL-free
  structured ops: `fetch_one`/`upsert`/`delete`/`scan`) protocols; `SqlL3Backend`
  implementing both over an asyncpg pool; `DurableStoreCollection` (a collection whose L3
  tier is a `DurableStore` — the base a git-backed collection subclasses); and
  `parse_rowcount`, the one framework-owned asyncpg status-tag parser.

### Added — `3tears-scheduled-jobs` (new package)

- The generic, multipod-safe scheduled-jobs core, generalized from
  agent-wake's tick machinery onto a payload-agnostic, consumer-neutral
  surface. `threetears.scheduled_jobs.tick` — the pure-async tick engine
  body a consumer's scheduler (e.g. APScheduler) invokes per interval;
  it enumerates due jobs via `ScheduleStore.list_due_for_tick` (a
  deliberate `__SPANS_PARTITIONS__` cross-partition scan) and claims each
  via an optimistic-CAS on `next_fire_at = expected_next_fire`, so two
  ticks across pods can never double-fire one job.
- `threetears.scheduled_jobs.reschedule` — the next-fire computation
  (interval / one-shot / terminal), with `coalesce` / `catch_up`
  missed-fire policies.
- Store protocols (`ScheduleStore` / `FireStore` / `DueSchedule`) the
  tick engine talks to, plus a default three-tier store keyed on an
  opaque `kind` (TEXT) + `payload` (JSONB): the `scheduled_jobs` +
  `job_fires` tables (partition column `partition_key`, composite PKs,
  `ON DELETE CASCADE` fire history), `ScheduledJobCollection` /
  `JobFireCollection`, and the v001 migration. The platform never
  inspects `kind` / `payload`.
- `config` (tick limits / policy defaults), `events` (lifecycle event
  names), and `metrics` (the `threetears_scheduled_jobs_` Prometheus
  instruments — fires / failures / tick-duration / drift — with the
  forbidden-label cardinality guard preserved). `prometheus_client`
  stays an optional extra; the emitter no-ops gracefully when absent.

### Changed — `3tears-agent-wake` — BREAKING

- **Tick engine delegates to `3tears-scheduled-jobs` (S-2).** The cross-pod tick
  pump (lock acquire/degrade-open, due-scan, optimistic-CAS claim, per-fire
  isolation, drift) and the reschedule math now live ONCE in the generic
  scheduled-jobs core; `threetears.agent.wake.tick` is a thin adapter over it.
  The wake-facing contract is UNCHANGED: `wake_tick_job(pool, nats_client,
  dispatch_callback)`, the wake-shaped `DispatchCallback`, `WakeTrigger`,
  `WakeDispatchResult`, the schedule/fire schema, the richer `FireStatus`, and
  the webhook / `[SILENT]` handling all stay put. The cross-pod lock key stays
  `"agent_wake_tick"`.
- **Removed `threetears.agent.wake.reschedule`** (and its private
  `_compute_next_fire_at`). The identical math is now public at
  `threetears.scheduled_jobs.compute_next_fire_at` — same positional signature.
- **Dropped the direct `3tears-nats` dependency** (added `3tears-scheduled-jobs`).
  The cross-pod lock now belongs to the scheduled-jobs core; wake reaches NATS
  only transitively. No code change for consumers that pass a `nats_client`
  through `wake_tick_job` (still typed `Any`).
- **Tick Prometheus metrics moved to the `threetears_scheduled_jobs_*` family.**
  The per-fire / drift / tick-duration counters the tick used to emit on the
  `threetears_agent_wake_*` instruments now come from the scheduled-jobs emitter;
  the CAS-miss failure reason changed `conv_busy` → `claim_lost`, and the
  per-fire `execution_mode` label is no longer on the tick fire counter. The
  genuinely wake-specific `threetears_agent_wake_yield_duration_seconds` is
  preserved (re-emitted by the adapter). Webhook / rate-limit / schedule-cap
  metrics are unchanged. The `EVENT_FIRE_SKIPPED_BUSY` log's `extra_data` keys
  changed (`conversation_id`/`fire_source`/`execution_mode` → `job_id`/
  `partition_key`). **Operators: update dashboards/alerts that key on the old
  `agent_wake` tick metrics or the `conv_busy` reason.**
- **Consumers: see `docs/migrating-agent-wake-to-scheduled-jobs.md`.**

### Added — `3tears-nats`

- **Owner-routed request forward** (`threetears.nats.forward` / `serve_owner`) — a
  generic, payload-agnostic primitive for "send a request to whichever pod currently
  *serves* a key, and get its reply." It is the messaging half of a single-writer
  pattern; it does NOT elect a leader (a separate `nats_distributed_lock` / `KVLease`
  decides who serves — the consumer ties them together). `serve_owner(nats, key,
  handler)` is an async context manager a consumer runs *while it holds the key*: it
  subscribes the key's forward subject in a **queue group keyed by that subject**, so a
  brief two-owner overlap during lease handoff still dispatches each request to exactly
  one owner. `forward(nats, key, payload, *, timeout) -> bytes` requests that subject and
  returns the owner's reply bytes. Payload + reply are opaque `bytes`.
- **Typed forward errors.** No current owner (no subscriber / timeout in the handoff
  window) raises `NoOwnerError`; an owner whose handler *raised* surfaces to the caller as
  `ForwardedHandlerError` carrying the original exception's **type name + message** (so a
  consumer can map a forwarded failure back onto its own typed exception). Both subclass
  `ForwardError` (← `NatsClientError`). Wire framing is a 1-byte tag (`0x00` ok + verbatim
  reply bytes; `0x01` err + UTF-8 JSON `{type, message}`), keeping an empty/arbitrary
  handler reply unambiguous from an error frame.
- **`Subjects.forward(key)`** — derives `{ns}.forward.{sha256hex(key)}`; the key is hashed
  (not `_sanitize`-mapped) so arbitrary app keys carrying `.`/spaces/`*`/`>` map to a
  subject-safe, collision-resistant, deterministic token, mirroring `Subjects.room`.
## v0.12.3 -- 2026-06-20

Broker isolation, NATS durability, prompt-cache cost accounting, and Redshift
hardening. Additive across `3tears`, `3tears-models`, `3tears-nats`, and
`3tears-langgraph`; one fixed-path change to the Redshift connection lifecycle.

### Added -- `3tears-models` -- `threetears.models`

- `UsageRecord` carries `cache_read_tokens` and `cache_creation_tokens`, surfaced
  as `llm.cache_read_tokens` / `llm.cache_creation_tokens` OTel span attributes,
  so prompt-cache hits and writes are tracked per call.
- `registry_loader` populates `cost_per_cache_read_token` and
  `cost_per_cache_write_token` from the capabilities registry, so cache-aware
  cost can be computed downstream instead of billing cached input at the full
  input rate.

### Added -- `3tears-nats` -- `threetears.nats`

- Bounded redelivery + dead-letter in the durable consumer factory
  (`resilience-task-01` RES-01-01/03): a message that fails past its redelivery
  budget is routed to a dead-letter subject instead of redelivering forever.
- Agent-deregister subject on `Subjects`, so a pod can announce its teardown.

### Added -- `3tears` -- `threetears.core`

- Per-call `customer_scope` channel on `NatsProxyL3Backend` reads -- the proxy
  carries the caller's customer scope per read rather than per connection, the
  substrate for conversation-scoped RBAC pool reads (broker isolation).
- Centralize JSONB through native binding under the codec (`collections-task-04`,
  Option B): a single binding path for JSONB columns, plus an enforcement drift
  guard (`test_jsonb_native_binding`) so a new column cannot silently bypass it.

### Added -- `3tears-langgraph` -- `threetears.langgraph`

- Turn-level keepalive on `StreamingResponse` (`long-response-task-01` LRT-02):
  a long single response emits periodic keepalives so the stream does not idle
  out mid-generation.

### Added / Fixed -- `3tears-datasources` -- `threetears.datasources`

- Per-column value-coverage probe: classifies a column as unloaded when every
  value is zero across the table -- the `UNLOADED_COLUMN` source the hub mirrors
  into datasource read results -- with driver-coverage tests.
- Redshift: re-apply `search_path` on every connection acquisition, so a pooled
  connection no longer serves a stale path left by a prior caller.

### Changed -- `3tears` -- `threetears.enforcement`

- The fake-parity walker accepts an inline `# parity-exempt: <rationale>` marker
  (matching the cache/underscore exemption style), removing the line-shift
  fragility of the prior line-numbered exemption file.

## v0.12.2 -- 2026-06-17

Additive: add the documented-schema digest entity + collection to
`3tears-datasources` — the materialized, by-pk schema/concept summary the hub
publishes per datasource and agent pods read at conversation start (the
foundation for schema priming). No behavior change to existing datasource
collections.

### Added — `3tears-datasources` — `threetears.datasources`

- `DataSourceSchemaDigest` entity + `DataSourceSchemaDigestCollection`, a
  three-tier collection keyed by `datasource_id` for a by-pk hot-L1 read with
  L2/L3 fallback and cross-pod invalidation. The table has no `id` column, so
  `primary_key_column = "datasource_id"` (the `BaseCollection` default would
  emit `WHERE id = ?` / `ON CONFLICT (id)` and break every by-pk read +
  invalidation). One row per datasource; the `tables` projection is JSONB.

### Fixed — `3tears-datasources` — `threetears.datasources`

- JSONB write double-encode: a pre-`json.dumps`'d string bound as `::jsonb` was
  re-encoded by the text-format jsonb codec into a scalar. Digest writes now
  text-cast (`::text::jsonb`) so the value lands as a real JSONB array. Covered
  by a real-L1 round-trip test (no-codec test pools gave a false green).

## v0.12.1 -- 2026-06-16

Patch: stop the OpenRouter wrapper from logging streaming tool-call
continuations as junk tool names. Every DeepSeek tool turn produced a
per-chunk WARNING storm (`dropped invalid_tool_calls entry with junk name:
None`) that buried real signal; the dropped entries were harmless to tool
arguments (the chunk merge re-derives from `tool_call_chunks`), but the
noise was severe. No behavior change to tool dispatch.

### Fixed — `3tears-models` — `threetears.models`

- `filter_invalid_tool_calls` now treats a nameless `invalid_tool_calls`
  entry (`name` None / absent / empty) as a normal streaming-continuation
  fragment — kept, never logged. Only a concrete, undispatchable name claim
  is rejected: a non-empty string failing the canonical tool-name regex
  (the genuine junk case, e.g. a quote-garbage name leaked from XML-shaped
  tool-call text) or a non-string / non-dict value. Genuine-junk rejection
  is unchanged. Verified by a local A/B on a real DeepSeek-over-OpenRouter
  tool turn: 12 `junk name: None` warnings before, 0 after.

## v0.12.0 -- 2026-06-15

Durable channel-answer delivery and native Slack rendering. A finished
agent answer is published to a durable JetStream subject and delivered
out-of-band, so an answer that takes minutes — or completes while the
channel adapter is restarting — is delivered, never lost. Agent markdown
now renders into native Slack Block Kit instead of arriving as raw text.

### Added — `3tears-channels` — `threetears.channels`

- `markdown_to_slack_blocks` — converts GitHub-flavored markdown into native
  Slack Block Kit: mrkdwn emphasis/links, `header` blocks, native `table`
  blocks (numeric columns right-aligned), code fences, and dividers, bounded
  to Slack's per-message limits. `SlackAdapter` now always renders answers as
  blocks with a plain-text fallback, and `post_message` delivers a finished
  answer out-of-band on the bot token.
- `ChannelDeliveryMessage` — the durable channel-delivery envelope, with a
  NATS-KV-valid `dedup_key` making at-least-once delivery post at-most-once.

### Added — `3tears-nats` — `threetears.nats`

- JetStream durable-delivery helpers on `NatsClient`: `ensure_jetstream_stream`
  (create-or-reconcile), `jetstream_publish` (PubAck-awaited), and
  `jetstream_subscribe_durable` (manual-ack consumer).
- `Subjects.channels_deliver` / `channels_deliver_wildcard` — the
  `{ns}.channels.deliver.{channel_type}` delivery subject family.

## v0.11.0 -- 2026-06-13

The governed-knowledge layer: agents answer data questions with curated,
scoped business knowledge instead of guessing. Concepts (a business term →
its data binding) and playbook entries (procedures) merge across the
platform / customer / user scope ladder; datasources are shareable across
customers with origin lineage; the model registry becomes a single source
of truth.

### Added — `3tears` (core) — `threetears.knowledge`

- Governed-knowledge merge: `merge_concept_views` / `merge_entry_views`
  resolve the three-scope shadow ladder (user > customer > platform, D4),
  flag ambiguity when same-name definitions compete with no declared shadow
  (D5), and honour the `always_inject` invariant (KNW-25). One shared
  `resolve_shadow_chains` walk, so the hub eval fingerprint and a live SDK
  turn agree byte-for-byte on the effective view.
- `ConceptSnapshot.datasource_table_ref` + `build_table_ref` — a concept's
  bound table renders as its agent-usable `schema.table` name (one source
  of truth for the format), never the raw `datasource_table_id` UUID the
  agent has no tool to resolve.
- `EntryEnforcement` constraint on playbook-entry snapshots; draft-command
  wire models + tool `BootstrapContext` for the correction-harvest surface.
- `repoint_user_rows` + `MemoryRepointResult` — the user-merge repoint
  primitives (`threetears.agent.memory`, `threetears.conversations`).

### Added — `3tears-agent-acl`

- Shared caller-visibility SQL: `three_scope_visibility_clause` +
  `customer_scope_visibility_clause` — one copy of the security SQL that
  admits a row iff it passes the platform/customer/user read rule. Every
  RBAC-scoped list composes it; no per-row Python visibility filter.

### Added — `3tears-datasources`

- Platform-sharing: a flat datasource PK, visibility, and origin lineage
  (`origin_datasource_id`) so a customer datasource inherits a
  platform-shared datasource's schema docs + governed knowledge.

### Added — `3tears-models`

- Single source of truth for model ids + capabilities, with a no-literal
  guard that keeps stale model strings out of the codebase.

### Added — `3tears-nats`

- `hub_channel_installs` subject so the Slack adapter fetches its active
  installs over NATS (sandboxed; no DB credentials cross the wire).

### Fixed

- `threetears.langgraph` — `NOSTREAM_TAG` + `replace_content` keep internal
  model calls out of the user-facing stream; the bound-model cache degrades
  gracefully on an unhashable model.
- `threetears.knowledge` — `EntryEnforcement.canonical_sql` is truly
  optional; hardened the core by-pk read + langgraph injection.

## v0.10.5 -- 2026-06-03

A reusable keyset (seek) paginator in `threetears.core` for paging large,
append-heavy ordered lists without `LIMIT`/`OFFSET` drift.

### Added — `3tears` (core)

- `threetears.core.pagination` — a shared cursor-pagination primitive. `Keyset`
  builds the `ORDER BY` clause and the composite row-value seek predicate
  (`(a, b) < ($1::text::t1, $2::text::t2)`) for a sort key + direction;
  `encode_cursor`/`decode_cursor` give an opaque, URL-safe base64-JSON cursor;
  `Keyset.page` trims the `+1` sentinel and emits the next cursor. The caller
  owns the SQL (columns are a trusted allow-list, never user input). Replaces
  ad-hoc `OFFSET` (which skips/repeats rows as the list grows under you) and
  hand-rolled "list-since" cursors. Exported from `threetears.core`:
  `Keyset`, `Page`, `CursorError`, `encode_cursor`, `decode_cursor`.
- Cursor values round-trip through JSON, so non-native key types (`datetime`,
  `UUID`, `Decimal`) serialize to strings; the keyset binds them as `text` and
  casts (`$1::text::timestamptz`) so drivers like asyncpg accept the string and
  Postgres parses it — the paginator pages by a timestamp key, the common case.

## v0.10.4 -- 2026-06-03

Single-node NATS resilience: the platform now survives a NATS restart on
ephemeral JetStream storage instead of silently losing the wake heartbeat.

### Fixed — `3tears-agent-wake`

- `wake_tick_job` degrades open when the cross-pod lock cannot be acquired
  (`KvError` -- the bucket/stream is gone after a NATS restart on ephemeral
  storage -- distinct from `LockHeld`): the tick body runs anyway, since
  per-schedule mutual exclusion is the Postgres optimistic-CAS in
  `WakeScheduleCollection.claim_and_reschedule`, not the lock. A NATS wipe no
  longer silences the wake heartbeat for hours until a process restart. Worst
  case under a NATS outage: every pod runs the due-scan and contends on the
  CAS (the handled `SKIPPED_BUSY` path) -- no double-fires, no data loss.

### Fixed — `3tears-nats`

- `NatsKvBucket` self-heals a vanished stream/bucket. A single-node NATS
  restart on ephemeral JetStream storage wipes every stream and KV bucket;
  the client caches bucket handles, so every op then failed forever
  (`nats: no response from stream`) until the process restarted. The bucket
  now retains its open config and, on a transport failure (not KeyNotFound /
  CAS-conflict), re-opens once -- recreating the bucket when `create_if_missing`
  -- and retries the op. The handle heals in place, so the client bucket cache
  needs no flush; a second failure surfaces as `KvError` as before.

## v0.10.3 -- 2026-06-02

Three platform features consumed by metallm: a per-schedule wake
conversation-history switch, conversation-search date filters, and
tool-result dedup (the foundation for bounding agent context bloat).
Plus a cron-scheduling correctness fix.

### Added — `3tears-agent-wake`

- `agent_wake_schedules.include_conversation_history` (BOOLEAN NOT NULL
  DEFAULT true, migration v006): per-schedule switch for whether a fire
  carries the conversation's recent history into the wake's LLM context.
  Threaded through the entity, collection, `WakeTrigger`, tick, the
  create/update/response API models, and the `wake_schedule_create` /
  `wake_schedule_update` tools. Independent of the attached skill's
  `prompt_mode` (persona) — the two compose.

### Fixed — `3tears-agent-wake`

- `CronTrigger.from_crontab` no longer adopts the host's local timezone:
  fire times are stored/compared in UTC, so a non-UTC host fired cron
  schedules at the wrong wall-clock instant. Now pinned to `_tz(config)`
  (UTC by default), matching every other schedule type.

### Added — `3tears-conversations`

- `ConversationsCollection.search` gains `date_field` (`"created"` |
  `"updated"`, allow-listed to a real column — never interpolated, raises
  `ValueError` otherwise) plus inclusive `date_after` / `date_before`
  bounds.

### Added — `3tears-agent-tools`

- Tool results dedup on `(tool, input)`: `ContextItemCollection`
  `upsert_tool_result` (sharing the extracted `_upsert_keyed` codepath
  with `upsert_variable`) on a new `ix_context_items_tool_result_key`
  partial-unique index (migration v004, non-destructive legacy-key
  suffix first). `context.save_tool_result(input_fingerprint=)` keys
  `tool_name + ':' + sha256(input)` and upserts; the shared
  `make_tool_result_dedup_key` lets storage and lookup agree (consumed by
  metallm's per-tool TTL result reuse).

## v0.10.2 -- 2026-06-01

Single-feature release on top of v0.10.1. `DatasourceConfig` now
threads `allowed_schemas` onto the connection's `search_path` at
open time so agents can write unqualified table names in their SQL
instead of fully qualifying every reference. Closes the Hub-side
pairing of the long-standing "agent must qualify every table" UX
papercut.

### Added — `3tears-datasources`

- `RedshiftConnectionConfig`, `PostgresConnectionConfig`, and
  `YugabyteConnectionConfig` carry a new `allowed_schemas: list[str]`
  field (default `[]` means "leave the backend default in place").
- Shared helpers `build_search_path_value` /
  `build_set_search_path_sql` in
  `threetears.datasources.drivers._util` with identifier-quoting
  for adversarial schema names.
- Redshift driver issues `SET search_path TO "<schemas>"` via
  `cursor.execute` after the existing `SET statement_timeout` block
  on every connection open.
- asyncpg driver passes `server_settings={"search_path": "..."}`
  through `create_pool`, landing the value in the pgwire STARTUP
  packet so it survives `DISCARD ALL` reset on pool release. An
  `init=` callback would NOT — that was the trip-wire surfaced by
  the live testcontainer pass.
- Coverage: 8 new unit tests (4 per driver), 4 new live integration
  tests against Redshift and the asyncpg testcontainer.

## v0.10.1 -- 2026-05-29

Single-fix release on top of v0.10.0. `RedshiftDriver` now runs
`ROLLBACK` on a query error before returning the connection to its
cache so a single bad SELECT no longer poisons the cached session
for the rest of the consumer's conversation.

### Fixed — `3tears-datasources`

- `RedshiftDriver._acquire_and_run` catches the query exception,
  runs `conn.rollback()` through the existing sync bridge, and
  releases the rolled-back connection back to the cache. Cancel
  path stays as-is (`asyncio.CancelledError` is `BaseException`-
  rooted and propagates through the dedicated `_on_cancel`
  callback, not double-handled here). If the rollback itself
  raises, the connection is evicted instead of released and a
  WARNING is logged; the ORIGINAL query exception is what
  propagates to callers in every branch. Coverage: three new
  unit tests (mocked-cursor positive / rollback-failure / two-
  fetch end-to-end) plus one new live integration test against
  `central-reporting` gated on `OTS_REDSHIFT_PASSWORD`.

  Background: `redshift_connector` uses the DB-API default of
  `autocommit=False`. A failed statement leaves the connection's
  implicit transaction in `aborted` state and the server then
  rejects every subsequent statement on that connection with
  `25P02: current transaction is aborted, commands ignored until
  end of transaction block` until an explicit `ROLLBACK` runs.
  Without the rollback, the agent's tool loop on a typo'd SELECT
  spins through its recursion budget retrying because every retry
  inherits the same poisoned cached connection.

## v0.10.0 -- 2026-05-23

The long-running-agent foundation release. Three new platform features
land in lock-step: a tool-eligibility flag pair on the existing
`3tears-agent-tools` base class, a brand-new `3tears-agent-skills`
package for procedural memory, and a brand-new `3tears-agent-wake`
package for scheduled + webhook-triggered fires. Two existing packages
gain supporting capabilities: `3tears-nats` exposes a distributed-lock
primitive lifted from metallm; `3tears-channels` ships a generic
`WebhookReceiver` framework with a pluggable verifier registry.

The first consumer is metallm's long_running + skills work (separate
release on the metallm side that pins this 3tears version).

### Added — `3tears-agent-tools`

- `TearsTool.tool_eligible: bool = True` and `TearsTool.skill_eligible:
  bool = False` class attributes decouple "is this tool in the agent's
  default tool surface?" from "is this tool discoverable in the skills
  catalog?". The defaults preserve pre-v0.10.0 behaviour for every
  existing tool. Subclasses opt-in to the new visibility states.
- New `agent_tools_platform` PLATFORM-scope migration adds
  `tool_eligible` + `skill_eligible` BOOLEAN columns to `namespaces`
  with `DEFAULT TRUE` / `DEFAULT FALSE` so existing rows keep their
  pre-shard semantics.
- `ToolNamespaceEmitter` / `ToolServer.publish_registration` stamps the
  flags onto the namespace row and emits a structured WARNING when a
  tool registers with both flags False (would be invisible to every
  agent surface).
- `agent-acl.NamespaceCollection` gains
  `list_tool_namespaces_for_actor(...)` (default surface =
  `tool_eligible=True` ∩ ACL) and
  `list_skill_eligible_tool_namespaces(...)` (skills catalog UNION
  source). Eligibility filters AFTER ACL — eligibility decides
  VISIBILITY; ACL decides AUTHORIZATION.
- `agent-acl.builtin_roles` ships the `PlatformBuiltinToolUser` role
  definition + canonical pre-check `mcp_name` list (`http_get`,
  `loki_query`, `postgres_query`) + idempotent
  `ensure_platform_builtin_tool_user_role` bootstrap helper. The
  deploying app seeds the `role_assignments` rows post-registration
  (per-version namespace UUIDs only exist after `ToolNamespaceEmitter`
  runs).

### Added — `3tears-agent-skills` (new package)

- `agent_skills` + `agent_skill_invocations` tables (partition column
  `agent_id`, composite PK + standalone UNIQUE on bare id for
  cross-package FKs). FTS-maintained `search_vector` (weighted A/B/C
  over `name || trigger_keywords || body`) for `skill_list` query
  filtering — NOT for auto-load (auto-load via classifier is
  explicitly out of scope per the v1 design).
- `AgentSkillCollection` + `AgentSkillInvocationCollection` with the
  full method surface (find_by_name_for_user, list_for_user, bump_use_count,
  increment_outcome_counts, record, list_for_skill, set_message_id,
  set_outcome).
- Seven `TearsTool` factories: `skill_create`, `skill_list`,
  `skill_get`, `skill_update`, `skill_delete`, `skill_invoke`,
  `skill_introspect` (the last returns the minimal-token shape for
  cheap discovery). Per-user cap of 200 prose skills; ACL probe on
  every tool name in `tool_additions`; first-invoke-wins enforcement
  on `skill_invoke` (with consumer-supplied state probe + setter
  Callable hooks).
- `compose_turn_context(active_skill, base_system_prompt,
  base_tool_names, *, acl_permits) -> ComposedTurnContext` — pure
  per-turn composition function. `prompt_mode='additive'` appends body
  to base prompt; `prompt_mode='replace'` substitutes (consumer
  layers per-user additions like NSFW / jailbreak on top in either
  mode). `tool_additions` ACL-gated; `tool_restrictions` subtractive
  without ACL check. One skill per turn maximum (no multi-skill
  composition).
- `SkillRegistryClient` Protocol decouples the package from
  `3tears-agent-acl` / `3tears-agent-tools` dependencies — consumers
  wire concrete bindings via three small Callable hooks
  (`conversation_id_resolver`, `active_skill_probe`,
  `active_skill_setter`) + a three-method Protocol surface
  (`acl_permits`, `list_skill_eligible_tools`, `get_tool_introspect`).

### Added — `3tears-agent-wake` (new package)

- `agent_wake_schedules`, `wake_fires`, `webhook_subscriptions` tables
  (partition column `conversation_id`; nullable `skill_id` FK on
  schedules; nullable `default_skill_id` FK on webhook subscriptions —
  single skill per wake / per subscription per the v1 design;
  `webhook_subscriptions.endpoint_secret_ciphertext` BYTEA Fernet-
  encrypted, decrypted via `EncryptionService` Protocol). All
  migrations idempotent; cross-package FKs land via post-creation
  guarded ALTER blocks.
- `wake_tick_job(pool, nats_client, dispatch_callback, *, wake_config)`
  — pure-async tick body the consumer's APScheduler
  `IntervalTrigger(seconds=60)` job invokes. Atomic CAS claim per
  schedule via `WakeScheduleCollection.claim_and_reschedule` (two
  ticks cannot fire the same schedule). Missed-fire policies
  `'coalesce'` (default) and `'catch_up'`; drift-recording via
  `wake_fires.scheduled_fire_at` + `wake_fires.actual_fired_at`.
  Per-fire skip emits `EVENT_FIRE_SKIPPED_BUSY`. Wake-yield
  cooperative-interrupt support via `wake_fires.status='yielded'` +
  yield-duration histogram.
- `_compute_next_fire_at(schedule, now)` covers all seven schedule
  types (cron / daily_at / one_shot / random_window /
  relative_delay / interval + the existing). DST-correct via stdlib
  `zoneinfo` (spring-forward + fall-back integration tests pinned).
- `dispatch_wake(trigger, fire_id, pool, *, handler, wake_config,
  delivery_adapters)` — sole entry point every wake source flows
  through (tick + webhook). Resolves attached skill (single-skill
  per PLACEMENT §1.3); resolves `context_from` single-hop
  same-conversation chain with 16KB truncation; invokes the consumer's
  `HandlerCallback`; detects `[SILENT]` prefix on response
  (case-insensitive, whitespace-tolerant); routes delivery to each
  target via the supplied `DeliveryAdapter` Protocol mapping
  (silent fires skip delivery; raised adapter exceptions caught +
  logged WARNING, fire still marked success because the LLM produced
  output). `_check_rate_limit` enforced at step 1 (per-conv per-day +
  per-user per-day; per-subscription per-hour on the webhook path).
- Fourteen `TearsTool` factories: six wake-schedule
  (`wake_schedule_create` / `_update` / `_list` / `_pause` / `_resume`
  / `_delete`) + seven webhook-subscription
  (`webhook_subscription_create` / `_update` / `_list` / `_pause` /
  `_resume` / `_delete` / `_rotate_secret`) + `wake_yield` (gated to
  load only on wake-driven turns via `is_wake_turn()` closure). Skill
  attachment is via the create/update `skill_id` parameter — no
  separate `wake_skill_attach` / `wake_skill_detach` tools. Detach
  semantics use explicit `detach_skill: bool = False` /
  `detach_default_skill: bool = False` / `clear_name: bool = False`
  fields because LangChain `@tool` cannot distinguish "field absent"
  from "explicit null".
- Per-conversation active-schedule cap (`WakeConfig.
  max_schedules_per_conversation = 10` default per PLACEMENT §1.9).
  App-side cycle detection on `context_from_schedule_id` (single-hop
  same-conversation; max-depth 10 defense-in-depth). ACL probe on
  every `skill_id` attached to a wake.
- `WakeConfig` Protocol + `DEFAULT_WAKE_CONFIG` constant — product
  supplies caps, URL allow-lists, named-query registries; platform
  honours.
- Prometheus instruments (prefix `threetears_agent_wake_*` — the
  documented `3tears_agent_wake_*` prefix is rewritten by
  `prometheus_client` because identifiers must match
  `[a-zA-Z_][a-zA-Z0-9_]*`): fires/failures/tick-duration counters,
  drift/yield-duration histograms, rate-limit/cap-rejection counters,
  webhook-received counter, delivery counter. No unbounded-cardinality
  labels (`conversation_id` / `user_id` / `schedule_id` /
  `subscription_id` / `agent_id` / `fire_id` are FORBIDDEN as
  labels). Enforcement test pinned at
  `tests/unit/test_metrics_cardinality.py`.
- Loki event-name constants (`EVENT_TICK_STARTED`, `EVENT_FIRE_*`,
  `EVENT_DELIVERY_*`, `EVENT_WEBHOOK_*`).
- Pydantic v2 request/response models in `api_models` for the wake
  REST surface (consumers import; metallm pins in shard-09 of the
  metallm long_running release). All models declare
  `extra='forbid'`; `pre_check_type` / `no_agent` /
  `pre_check_output` round-trip rejected (anti-patterns per
  PLACEMENT §1.2).

### Added — `3tears-nats`

- `nats_distributed_lock(client, key, *, ttl, heartbeat_interval,
  holder_id) -> AsyncContextManager` lifted from metallm's
  `scheduler_lock`. Atomic NATS KV `bucket.create()` claim; background
  heartbeat task refreshes lease before TTL; raises `LockHeld` on
  conflict; auto-expires on holder crash. Constant-time bucket-TTL
  mismatch check raises `ValueError` rather than silently inheriting
  the first caller's TTL.

### Added — `3tears-channels`

- `WebhookReceiver` framework (optional `[webhook]` extra; depends on
  `fastapi` + `3tears-agent-wake`). `register_verifier(scheme,
  callable)` lets vendor-specific schemes (GitHub `X-Hub-Signature-
  256`, Stripe `Stripe-Signature`, etc.) plug in. Default scheme
  `generic_hmac_sha256` ships with `verify_generic_hmac_sha256`
  (constant-time `hmac.compare_digest`). HTTP status mapping
  202 / 400 / 401 / 403 / 404 / 413 / 429 (with `Retry-After: 60`) /
  500. 1 MiB payload cap enforced BEFORE subscription lookup +
  secret decryption (closes cost-attack vector on unverified
  payloads).
- `verify_generic_hmac_sha256` + `compute_generic_hmac_sha256_signature`
  live at `threetears.agent.wake.hmac_util` (one shared
  implementation; both channels' receiver and agent-wake's adapter
  import from there).
- `webhook_subscriptions.verification_scheme` CHECK constraint opened
  in v005 migration (was hardcoded to the single
  `generic_hmac_sha256` literal; now `~ '^[a-z0-9_]+$' AND length
  BETWEEN 1 AND 64`). Registered schemes are validated at
  receiver-handle time (unknown → 400) since the DB cannot consult
  the live in-process registry.

### Notes

- All 18 workspace packages bumped to 0.10.0 in lock-step (the
  `3tears-agent-skills` + `3tears-agent-wake` packages are new in
  this release; the other 16 keep their existing surfaces with
  the additions documented above).
- Test count: 6,564 unit + 201 integration, all green.
  No new "ours-side" test warnings — the only remaining 67
  warnings are upstream (langgraph `LangChainPendingDeprecationWarning`
  + langchain_core `asyncio.iscoroutinefunction` deprecation).
- Migration ordering: `agent-skills` migrations (v001 + v002) land
  before `agent-wake` migrations (`depends_on=("conversations",
  "agent_skills")` enforces the topological order via the canonical
  `MigrationRunner`). The `agent-tools` PLATFORM-scope migration
  for the eligibility columns runs once at hub startup against the
  shared schema.
- Cross-package dep direction: `channels` → `agent-wake` (via the
  `[webhook]` extra) is the only new directional edge. `agent-wake`
  → `agent-skills` (single-skill resolution from
  `AgentSkillCollection`). No circular imports. The `nats`
  distributed-lock primitive is consumed by `agent-wake` (the tick
  body) and by metallm's existing backup job (which becomes a
  re-export when metallm pins this release).
- Backwards compatibility: NO breaking changes. The two new
  `TearsTool` flags default to the pre-v0.10.0 behavior.
  Migration v005 in `agent-wake` opens a previously-stricter
  CHECK constraint (additive); no schema breaks. All new tables
  and columns are additive. Existing consumers continue to work.

## v0.9.1 -- 2026-05-23

### Changed

- **`3tears-datasources` — pluggable secret resolution (Path A).**
  Datasource credentials are no longer named by an env var
  (`password_env` / `credentials_json_env`). They now carry a
  `scheme://locator` *reference* in `password_ref` /
  `credentials_json_ref`, resolved at driver-creation time (Hub-side,
  scoped to one datasource) by a pluggable backend in the new
  `threetears.datasources.secrets` module. The secret value never
  lives in agent.yaml, never lands plaintext in the Hub DB, and never
  sits in a long-lived process variable — it is only ever held inside
  a `SecretStr` and unwrapped at the last moment when handed to the
  backend lib. Shipped backends:
    - `env://NAME` — read process env var `NAME` (the devx backend;
      devx mounts the agent project `.env` into the Hub container so
      every datasource credential resolves on a fresh stack with no
      per-secret hand-listing).
    - `k8s://rel/path` — read a projected-Secret file under
      `AIBOTS_DATASOURCE_SECRETS_DIR` (default `/var/run/secrets/aibots`);
      the prod shape (k8s `Secret` as a volume).
  `vault://`, `aws-secretsmanager://` and `gcp-sm://` are registered
  but raise a clear "not implemented" error so the scheme surface is
  stable for config authors today. Config validators call
  `validate_ref` at load time (shape/scheme check, no env/fs touch);
  resolution stays a use-time concern. This is a hard rename with no
  backwards-compatibility shim.
- **`3tears-datasources` realigned to the monorepo lockstep version.**
  The package had been on an independent `0.1.x` line; it now versions
  with every other workspace package (`0.9.1`). Its README "Versioning
  policy" and CHANGELOG were rewritten accordingly.

### Notes

- Patch bump: the only behavioural change is internal to
  `3tears-datasources` (the credential-reference rename + resolver).
  No other package's public API changed.
- All 17 workspace packages bumped to 0.9.1 in lock-step (the
  `3tears-datasources` package joined the lockstep this release).
- The platform Docker image stamp tracks this tag (`v0.9.1`); the
  devx compose now injects the whole agent `.env` into the Hub
  container generically, retiring the per-secret passthrough.

## v0.9.0 -- 2026-05-20

### Added

- `threetears.models.chunk_merging.merge_chunks` -- canonical merge of
  streamed `AIMessageChunk` lists into a single `AIMessage`. Wraps
  LangChain's `AIMessageChunk.__add__` for the merge, finalizes to a
  concrete `AIMessage`, and preserves `invalid_tool_calls` for
  downstream recovery. Replaces inline duplicates across consumers
  (metallm personality node, 14-eng-ai-bot router,
  14-eng-ai-bot-agents tool loop).
- `threetears.models.chunk_parsing.parse_chunk` -- canonical extractor
  of `(text, reasoning)` per streamed chunk. Covers all three
  observed shapes (OpenAI / OpenRouter string content, Anthropic-direct
  list-of-blocks, OpenRouter / OpenAI reasoning models'
  `additional_kwargs["reasoning_content"]`) and mixed cases. Pure,
  no-I/O hot-path helper.
- `threetears.models.tool_name_validation` -- canonical tool-name
  validator (`is_valid_tool_name`, `validate_tool_name`,
  `filter_invalid_tool_calls`, `ToolNameValidationError`). Pins the
  3tears tool-name regex (`^[a-zA-Z0-9_.-]{1,64}$`) covering every
  observed provider validator plus the dotted canonical form.

### Fixed

- Closes the metallm 2026-05-19 prod incident (conv
  `019e3e26-9870-7a03-8f04-8cc6a4f5f418`) where a misbehaving
  model response surfaced a tool-call name with an embedded
  XML-attribute fragment (`memory_recall" name="memory_recall`).
  The junk name reached metallm's dispatch layer through the
  chat-model wrapper unfiltered and was persisted as an
  unrecoverable invocation. The OpenRouter and Anthropic provider
  wrappers now call `filter_invalid_tool_calls` on every streamed
  chunk and every `_agenerate` result, dropping junk entries with
  one `WARNING` log per drop (name truncated to 80 chars). This
  blocks `function.name` junk from reaching downstream dispatch in
  any 3tears consumer.

### Notes

- v0.9.0 is a minor bump because it establishes new wrapper-layer
  contracts that downstream consumers can rely on: clean tool
  names guaranteed at the chat-model boundary, plus the canonical
  chunk-parsing / chunk-merging utilities. Bugfix patch would have
  been wrong given the new public API surface.
- All 16 workspace packages bumped to 0.9.0 in lock-step.
- No backwards-incompatible changes. Existing consumers that
  inline their own chunk parsing / merging continue to work; the
  new utilities are opt-in.
