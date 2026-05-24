# Changelog

All notable changes to the 3tears platform packages are recorded here.
This project follows semantic versioning across all 17 workspace
packages (bumped in lock-step).

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
