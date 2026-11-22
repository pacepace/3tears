---
artifact: data-model
version: 1
depends_on: []
last_validated: null
---

# Data Model

## Summary

3tears keeps data in entities and collections over three tiers. L1 is SQLite inside the process. L2 is NATS KV, shared by every pod. L3 is PostgreSQL (or a YugabyteDB/PostGIS variant), or a git working tree (`README.md`, `packages/core/README.md`).

L3 is the source of truth. L1 and L2 are caches that a write or a generation signal can always drop (`packages/core/README.md`, "L1 Staleness"; `packages/core/src/threetears/core/collections/generation.py`).

Each package owns its tables and ships its own migrations. One runner applies them, one version number per package, with a ledger in `_schema_migrations` (`docs/how-to-add-a-migration.md`, `packages/core/src/threetears/core/data/migrations/runner.py`).

The table families are below, with the invariants the code and the enforcement tests hold.

## Terms

| Term | Meaning | Source |
|---|---|---|
| Entity | A thin cache proxy for one row. Field data lives in L1. It tracks changes and reads through to the tiers | `packages/core/src/threetears/core/entities/base.py` |
| Collection | A table seen through the three tiers. It implements fetch, save and delete against L3 | `packages/core/README.md` |
| `SchemaBackedCollection` | A collection generated from a declared `TableSchema` | `packages/core/src/threetears/core/collections/schema_backed.py` |
| Partition column | The one column per table that every read and write must filter on | `docs/partition-column-pattern.md` |
| Write generation | A per-table counter, advanced after each committed write | `packages/core/src/threetears/core/collections/generation.py` |
| Epoch | A generation-stamped number broadcast over NATS so pods reload caches | `packages/epoch/README.md` |
| Scope | `AGENT` (a per-agent schema) or `PLATFORM` (the shared schema) | `packages/core/src/threetears/core/data/migrations/scope.py` |

## Tier model

| Tier | Store | Access | Role |
|---|---|---|---|
| L1 | SQLite in WAL mode (DuckDB also exists) | Sync, in-process | Entity attribute reads and writes |
| L2 | NATS KV bucket `{namespace}-collections` | Async, shared | Cross-pod cache and invalidation |
| L3 | PostgreSQL, or a git working tree | Async | Durable truth |

Source: `packages/core/README.md`, `packages/core/src/threetears/core/cache/`, `packages/core/src/threetears/core/backends/`.

### Reads

A read goes L1, then L2, then L3 and promotes on a miss. `users[id]` does this synchronously through a background loop. `await users.get(id)` does it asynchronously and returns `None` when the row exists in no tier (`packages/core/README.md`).

### Writes

`save_entity` writes L3 first. L2 then takes the committed row as a compare-and-swap at the revision read before the L3 write. L1 caches the row only when L2 took it and no other save of the key overlapped (`packages/core/src/threetears/core/collections/base.py`, `save_entity` docstring).

When the compare-and-swap is refused, the key is deleted from L2 and L1, and the next read takes whichever row L3 committed last. The saved entity holds the stored row itself and never depends on L1's copy (same docstring).

With `conn=`, the row is written inside the caller's transaction and no tier caches it. `CallerTransaction` evicts the key from L1 and L2 and broadcasts after commit or rollback (same docstring, `packages/core/src/threetears/core/collections/caller_transaction.py`).

### Flush strategies

| Strategy | Behavior |
|---|---|
| `ALWAYS` | Every save writes L3 at once |
| `ON_CHECKPOINT` | Buffers in L1 and L2, flushes on `flush_pending()` |
| `ON_SCHEDULE` | As `ON_CHECKPOINT`, with a timer |
| `ON_SHUTDOWN` | Flushes at application shutdown |

Only tables named in `collection_flush_tables` may defer. Every other table writes at once (`packages/core/README.md`, `packages/core/src/threetears/core/collections/flush.py`).

The deferred buffer is an L1 table, `write_buffer`. A flush persists a toposorted batch in one transaction when the backend has `transaction()`, and falls back to a per-entity loop otherwise (`docs/migrating-to-l3-store-seam.md`, "L3B-04").

A failed deferred write is retried. After `_MAX_FLUSH_RETRIES` (10) it is dropped from the buffer with a logged permanent failure. Foreign-key failures get a larger budget (`packages/core/src/threetears/core/collections/flush.py`).

### Optimistic locking

A schema's `cas_column` (conventionally `date_updated`) is the fence. An update runs `WHERE <cas_column> = <original>`. Zero rows changed raises `ConcurrentModificationError` (`packages/core/README.md`, `packages/core/src/threetears/core/collections/schema_backed.py`, `packages/core/src/threetears/core/exceptions.py`).

### Coherence across pods

Three mechanisms keep L1 and L2 honest.

- **Row broadcast.** A write publishes an invalidation envelope naming the row ids. Peers evict those rows (`packages/core/src/threetears/core/collections/base.py` module docstring; `docs/epoch-task-04-pending-invalidation-queue.md` for why no local queue exists).
- **Write generation.** A table that declares `write_generation` gets a generation advanced after each committed write. A pod that missed a broadcast reads the epoch change record of each missed advance and evicts exactly those rows. It never drops the whole table for a gap (`packages/core/src/threetears/core/collections/generation.py`).
- **Write marks.** A row trigger inserts a mark into the write-marks table in the same transaction as the write. A `WriteMarkReconciler` advances the generation from those marks, so a commit whose own advance was lost is still seen. `TRUNCATE` fires no row trigger and leaves no mark (`packages/core/src/threetears/core/collections/write_marks.py`).

Nothing cached in L1 ages. A row stays until a write that reaches it is heard. A cache that needs a bound goes on the epoch system, not on a timer (`packages/core/README.md`, "L1 Staleness").

### Absences

A collection that sets `negative_cache_max_age` caches "this key exists in no tier". Each absence is stamped with the table's generation read before the L3 lookup. An absence whose stamp is not the current generation is not trusted. Absence markers live in the L1 table `collection_absent_markers`, never in L2 (`packages/core/src/threetears/core/collections/base.py`, `generation.py`).

### L2-first collections

A collection that declares `l2_first` commits a write in NATS. The commit is one epoch change record carrying the row. A saver then makes L3 match, and the L3 epoch log (`threetears_epoch_log`) holds the records (`packages/core/src/threetears/core/collections/l2_first.py`, `docs/l2-first-operations.md`).

| Mode | Declaration | Behavior |
|---|---|---|
| Waiting (default) | `L2_FIRST` | The write returns after L3 saved it. An L3 refusal reaches the caller and the L2 write is withdrawn |
| Not waiting | `L2First(wait_for_l3=False, refetchable_from=...)` | The write returns at the commit. An L3 refusal is an ERROR and is counted. Meant for data its source can resend |

When NATS loses data, a lookup raises `L2LostError` and one leased worker refills L2 from L3. A write that did not wait and had not reached L3 is gone until its source resends it (`docs/l2-first-operations.md`).

### Without L2 or with a git L3

Without NATS, the same collection writes straight to L3. One pod on PostgreSQL does not need NATS. More than one pod, or a git L3, does (`README.md`, "Install").

A git L3 stores rows as files, with one commit per change set and one writer per repository. It implements the SQL-free `DurableStore` protocol, and `DurableStoreCollection` runs on it (`packages/core/src/threetears/core/backends/git/__init__.py`, `docs/migrating-to-l3-store-seam.md`).

### L2 keys and buckets

Every L2 key is `{scope}.{table}.{body}`. The scope is the NATS principal the process authenticates as. `configure()` raises `L2ScopeNotConfiguredError` when an L2 client has no scope (`packages/core/README.md`, "Configure the Registry").

Every KV bucket in the repo is memory-backed. Durability belongs to L3 (`packages/core/tests/enforcement/test_kv_buckets_are_memory_only.py`).

## Entities

Each table's columns live in its package's migrations and `TableSchema` declarations. This artifact lists the families, not every column. A table whose schema is declared in the same package as its entity is marked "schema" below.

### Core

| Table | Purpose | Source |
|---|---|---|
| `_schema_migrations` | Migration ledger: `(version, package)` key, description, `date_applied` | `packages/core/src/threetears/core/data/migrations/runner.py` |
| `coordination_counters` | Windowed attempt counts, write-behind | `packages/core/src/threetears/core/coordination/tables.py` |
| `coordination_claims` | Idempotency claims and their stored outcome, write-behind | same |
| `coordination_revocations` | Standing revocations, synchronous and negative-cached | same |
| `coordination_redemptions` | Single-use redemption ledger, synchronous | same |
| collection write marks | One row per committed write on a switched-on table, drained by the reconciler | `packages/core/src/threetears/core/collections/write_marks.py`, `coordination/migrations/v003_create_collection_write_marks.py` |
| `threetears_epoch_log` | L3 copy of L2-first epoch records, in a schema the host names | `docs/l2-first-operations.md` |

The coordination tables key rows by `(purpose, key)`, so one collection serves every counter or ledger in a process (`coordination/tables.py`).

### Per package

| Package | Tables | Purpose | Scope | Source |
|---|---|---|---|---|
| `3tears-conversations` | `conversations`, `folders` | One row per conversation: owner scope, channel reference, status, summary, timestamps. Folders group conversations | AGENT | `packages/conversations/README.md`, `.../migrations/v008_create_folders_and_conversation_folder_id.py` |
| `3tears-agent-memory` | `memories`, `memory_chunks`, `media`, `media_content`, `conversation_memory_refs`, `memory_consolidations` | Extracted memories and their searchable chunks, attached media, per-conversation surfacing, and gist-from-source provenance edges | AGENT | `packages/agent/memory/src/threetears/agent/memory/collections.py`, `.../migrations/v026_create_memory_consolidations.py` |
| `3tears-agent-tools` | `context_items`; columns on `namespaces` | Per-conversation live tool context. The platform migrations add tool and face flags to `namespaces` | AGENT; PLATFORM | `.../tools/migrations/v001_create_context_items_table.py`, `.../tools/platform_migrations/__init__.py` |
| `3tears-agent-workspace` | `workspaces`, `workspace_files`, `workspace_file_versions` | Workspaces, their files, and file versions | AGENT | `.../workspace/migrations/v001_create_workspace_tables.py` |
| `3tears-agent-skills` | `agent_skills`, `agent_skill_invocations` | Skill definitions and one row per skill load | AGENT | `.../skills/migrations/v001_create_agent_skills.py`, `v002_create_agent_skill_invocations.py` |
| `3tears-agent-wake` | `agent_wake_schedules`, `wake_fires`, `webhook_subscriptions`, `webhook_refusals` | Schedules, fire history, inbound webhook subscriptions, refused webhooks | AGENT | `.../wake/migrations/v001` to `v003`, `v010_create_webhook_refusals.py` |
| `3tears-agent-identity` | `identity_versions` | Versioned identity blocks, a parent-pointer chain per block | AGENT | `packages/agent/identity/README.md`, `.../migrations/v001_create_identity_versions.py` |
| `3tears-agent-intention` | `intentions` | Standing wants with an embedding and a status enum | AGENT | `.../intention/migrations/v001_create_intentions_table.py` |
| `3tears-agent-knowledge` | `concepts`, `playbook_entries` (read through collections over the platform tables) | Governed knowledge the agent retrieves | PLATFORM (tables owned elsewhere) | `packages/agent/knowledge/src/threetears/agent/knowledge/collections.py` |
| `3tears-agent-acl` | `namespaces`, `groups`, `group_members`, `roles`, `role_assignments` (schema also declares `impersonation_gates`) | The RBAC graph the evaluator reads | PLATFORM | `.../acl/migrations/v001_create_acl_tables.py`, `.../acl/collections.py` |
| `3tears-agent-audit` | `audit_events` | The single audit table: one envelope, indexed | PLATFORM | `.../audit/migrations/v001_create_audit_events.py` |
| `3tears-epoch` | `config_epochs` | Durable epoch rows for epochs whose value escapes the cluster, such as tile versions | PLATFORM | `packages/epoch/README.md`, `.../epoch/migrations/v001_create_config_epochs.py` |
| `3tears-langgraph` | `checkpoints`, `checkpoint_writes` | LangGraph checkpoint persistence, string ids and BYTEA payloads | AGENT | `.../langgraph/migrations/v001_create_checkpoint_tables.py` |
| `3tears-mcp` | `mcp_tool_grants` | Principal-to-tool grants | PLATFORM | `.../mcp/migrations/v001_create_mcp_tool_grants.py` |
| `3tears-scheduled-jobs` | `scheduled_jobs`, `job_fires` | Default store for the cross-pod tick loop | PLATFORM | `.../scheduled_jobs/migrations/v001_create_scheduled_jobs.py` |
| `3tears-scrape` | `scrape_targets`, `scrape_recipes`, `scrape_extractions`, `scrape_target_health` | Targets, reusable extraction recipes, extractions, fetch health | PLATFORM | `packages/scrape/src/threetears/scrape/migrations.py` |
| `3tears-datasources` | `datasources`, `datasource_tables`, `datasource_columns`, `datasource_relations`, `datasource_schema_digests`, `table_templates` | Datasource catalog and discovered schema | schema | `packages/datasources/src/threetears/datasources/collections.py` |
| `3tears-evals` | `documents` | Store the engine's report documents | host-owned | `packages/evals/src/threetears/evals/analysis/report_doc_store.py` |

`3tears-core`'s `DataStore` also creates dynamic per-agent tables from a `TableDef` (`packages/core/src/threetears/core/data/store.py`, `data/schema.py`).

## Relationships

| Relationship | Cardinality and rule | Source |
|---|---|---|
| Memory to chunk | One to many, cascade on memory delete | `packages/agent/memory/src/threetears/agent/memory/entities.py` |
| Memory to media and media content | Composite foreign keys keyed on `(agent_id, ...)` | `.../memory/migrations/v009_media_composite_fk.py` to `v012_memories_media_composite_fk.py` |
| Gist to source memory | Many source rows to one gist, one edge row per pair | `.../memory/migrations/v026_create_memory_consolidations.py` |
| Conversation to memory | `conversation_memory_refs` tracks which items surfaced in which conversation | `docs/partition-column-pattern.md` |
| Skill to invocation | Composite foreign key `(agent_id, skill_id)`, cascade on skill delete | `.../skills/migrations/v002_create_agent_skill_invocations.py` |
| Skill to wake schedule | Nullable `skill_id` on the schedule references `agent_skills(skill_id)` | `.../wake/migrations/v001_create_agent_wake_schedules.py` |
| Schedule or webhook to fire | A fire has at most one of `schedule_id` and `webhook_subscription_id` | `.../wake/migrations/v002_create_wake_fires.py` |
| Scheduled job to fire | `job_fires.job_id` references the bare `job_id` | `.../scheduled_jobs/migrations/v001_create_scheduled_jobs.py` |
| Conversation to its owners | Memory, tools, workspaces and wake key off `conversation_id`. `conversations` is the root of the agent-schema dependency graph | `packages/conversations/README.md` |

Each of these tables also carries a standalone `UNIQUE` on its bare id, so a cross-package foreign key can reference it without knowing the partition (`skills v001`, `wake v001`, `scheduled-jobs v001`).

## State machines

| Entity | States | Source |
|---|---|---|
| Identity version | `proposed`, `active`, `superseded`, `rejected`. Exactly one `active` per `(agent_id, customer_id, user_id, block_key, variant)`. Propose, consent, apply, with non-destructive rollback | `packages/agent/identity/README.md`, `.../identity/migrations/v001_create_identity_versions.py` |
| Intention | `open`, `asked`, `granted`, `dropped` | `.../intention/migrations/v001_create_intentions_table.py` |
| Wake fire | `success`, `silent`, `yielded`, `skipped`, `failed` at v002. Later versions add `dispatching` and a checked-quiet status | `.../wake/migrations/v002_create_wake_fires.py`, `v004_add_dispatching_status.py`, `v008_checked_quiet_status.py` |

Transition rules beyond these lists are not stated in the files read. See Open questions.

## Constraints

- **Partition column.** Exactly one column per table may declare `partition=True`. It must be in the primary key and is forced immutable (`docs/partition-column-pattern.md`, `schema_backed.py` `TableSchema.__post_init__`).
- **Partition methods.** Every public method on a partitioned collection takes the partition column, is decorated `@spans_partitions`, or is listed in `partition_exempt_methods`. A violation fails at import with `PartitionEnforcementError` (`docs/partition-column-pattern.md`).
- **Cross-partition reads.** The fan-out argument must be a non-empty `tuple`. A `list` or `()` raises `TypeError`. ACL stays in the service layer (`docs/partition-column-pattern.md`).
- **Partition columns by table.** `agent_id` for memories, media, conversations and workspaces. `conversation_id` for `context_items`, `conversation_memory_refs` and the wake tables. `workspace_id` for workspace files and versions (`docs/partition-column-pattern.md`, wake v001). `scheduled_jobs` uses `partition_key` (scheduled-jobs v001).
- **Immutable columns.** Immutable columns are written on INSERT and excluded from UPDATE (`schema_backed.py`, `Column.immutable`). A `TSVECTOR` column must be `immutable=True`, since a trigger maintains it.
- **Datetimes.** Every datetime column is `DATETIMETZ_TYPE` (`TIMESTAMPTZ`, aware UTC). `DATETIME_TYPE` is deleted (`packages/core/tests/enforcement/test_no_datetime_type_columns.py`, `test_column_type_alignment.py`).
- **JSONB.** A `jsonb` column is bound as a native Python object. `json.dumps` plus `::jsonb` in `save_to_store` double-encodes and fails the build (`packages/core/tests/enforcement/test_jsonb_native_binding.py`).
- **Write generation.** A collection's write-generation switch and its `TableSchema` declaration must agree (`tests/enforcement/test_write_generation_is_declared_on_the_schema.py`; `PackageMigrations.mark_write_generation` raises `ValueError` on a mismatch).
- **Negative caches.** A collection that caches absences writes L3 only through `save_entity` or `l2_cas_mutate` (`packages/core/tests/enforcement/test_negative_cache_write_paths.py`).
- **Composite keys.** Composite-key call sites pass a tuple, and arity matches unpacking (`tests/enforcement/test_composite_pk_call_sites_pass_a_tuple.py`, `test_collection_pk_arity_matches_unpacking.py`).
- **Ledger identity.** A recorded migration description that differs from the build's raises `LedgerMismatchError` (`runner.py`).

## Migrations

### Versioning

- Each package registers one `PackageMigrations(name, scope, depends_on)`. Versions are the integers passed to `pkg.version(N)`, and the filename prefix is informational (`docs/how-to-add-a-migration.md`).
- The runner orders packages topologically by `depends_on` (Kahn's algorithm) and applies pending versions per scope. `apply_package` does not resolve dependencies (`runner.py`, `docs/how-to-add-a-migration.md`).
- `_schema_migrations` holds `(version, package, description, date_applied)` with primary key `(version, package)`. The runner verifies descriptions against the build's, because the description is the only evidence of which migration a number was (`runner.py`).
- A writing run holds a database-wide DDL lock from before the ledger read until after the last row. It runs on one pinned connection and takes one migration per database at a time, because concurrent index builds hang each other in YugabyteDB (`runner.py`, `packages/core/src/threetears/core/ddl/ddl_lock.py`).
- `downgrade_for_scope`, `stamp_version` and `preview_for_scope` exist. Drift detection compares live `information_schema` to the DDL the migrations would issue (`runner.py`, `data/migrations/drift.py`).
- The runner gives each table a package marks with `mark_write_generation` its write-mark trigger at the end of every run (`data/migrations/registry.py`).

### Version depth by package

| Package | Versions | Package | Versions |
|---|---|---|---|
| memory | 32 | tools | 5 (+3 platform) |
| conversations | 11 | skills | 5 |
| wake | 10 | workspace | 5 |
| identity | 3 | coordination | 3 |
| epoch | 2 | acl, audit, intention, mcp, langgraph, scheduled-jobs | 1 |

Counted from the `migrations/` directories under `packages/`.

### Rules for a migration

From `docs/how-to-add-a-migration.md`:

- Idempotent statements only (`IF NOT EXISTS`, `ADD COLUMN IF NOT EXISTS`).
- No schema qualification in agent migrations. The L3 broker sets `search_path`.
- One body per file. No autogeneration. No seeding another package's tables.
- Applied migrations are immutable. A change is a new version.

### Enforced rules

The walker in `packages/core/src/threetears/core/data/migrations/enforcement.py` runs over every package's migrations (`packages/core/tests/enforcement/test_migration_yugabyte_safety.py`).

| Rule | Severity | What it flags |
|---|---|---|
| M-1 | Critical | A `DO` block mixing DDL and DML |
| M-2 | High | A backfill `UPDATE` with no replay-guard predicate |
| M-3 | Medium | `CREATE` or `ADD COLUMN` without `IF NOT EXISTS`, `DROP` without `IF EXISTS` |
| M-4 | High | A multi-column primary key with no `UNIQUE` on the preserved id |
| M-5 | High | `TRUNCATE TABLE` |

Exemptions live in `packages/core/tests/enforcement/_migration_exemptions.txt`, each with a rationale. `MIGRATION_ENFORCEMENT_MODE` selects `report` or `strict` (`enforcement.py`).

## Access paths

| Read | Found by | Source |
|---|---|---|
| One row by key | L1, then L2 key `{scope}.{table}.{body}`, then L3 primary key | `packages/core/README.md` |
| Own-partition list | Partition column in the predicate, enforced by AST walker | `packages/core/tests/enforcement/test_partition_column_enforcement.py` |
| Cross-partition list | `ANY($1::uuid[])` over a resolved tuple, bypassing the by-key caches | `docs/partition-column-pattern.md` |
| Visibility lists (knowledge) | SQL-side RBAC visibility clause, bypassing by-key caches | `packages/agent/knowledge/src/threetears/agent/knowledge/collections.py` |
| Hybrid memory search | Full-text, vector (HNSW) and recency over `memory_chunks` | `README.md`, `.../memory/migrations/v022_add_hnsw_gin_indexes.py`, `v029_memory_chunks_heading_weighted_search.py` |
| Whole tables on a pod | Scoped snapshot in L2 (Arrow IPC chunks), loaded into DuckDB L1 | `docs/design-scoped-snapshot.md` |
| Large binaries | S3-compatible object store, outside the entity tiers | `README.md` |

The knowledge collections omit the `embedding` column from their projection so a proxy pool never decodes a pgvector value on the per-turn list path (`knowledge/collections.py`).

A scoped snapshot is one Arrow chunk per scope and table, one pointer per scope, and a compare-and-set pointer that never moves to a lower epoch. Every row must carry a scope (`docs/design-scoped-snapshot.md`).

Expected volumes are not stated in the sources read. See Open questions.

## Data lifecycle

### Erasure in the audit trail

An audit record is never deleted by erasure and no id on it changes. Every erasure path routes the record's content through one rule (`packages/agent/audit/README.md`):

- `anonymize_details` keeps every key. It keeps the value under a key on `SAFE_DETAIL_KEYS` and replaces the whole value under any other key with `"[anonymized]"`. It is pure and idempotent.
- `anonymize_ip` returns `None`. An address is removed, not truncated.
- A key nobody classified is masked, so it fails safe. `tests/enforcement/test_audit_details_keys_are_classified.py` holds producers to the classification.
- An agent asks the hub to anonymize its own rows through `request_audit_anonymization` on `{ns}.hub.audit.anonymize`. The hub reads the agent from the verified identity token and touches only that agent's rows.
- A deployment with no hub runs `anonymize_audit_rows` against its own table.
- `prune_audit_events` is batched retention by age.

### Other lifecycles

| Data | Rule | Source |
|---|---|---|
| Memory | Hard delete only. `is_deleted` and `date_deleted` were removed in v018. Chunks cascade. Consolidated sources stay recallable | `.../memory/entities.py`, `.../memory/dream.py` |
| Identity versions | Superseded, never lost. Rollback is non-destructive | `packages/agent/identity/README.md` |
| Workspace | `date_deleted` column added in v002 | `.../workspace/migrations/v002_add_date_deleted_column.py` |
| Conversation memory refs | Dangling refs are tolerated. The table carries no foreign key to `conversations`, so a later conversation delete leaves them | `.../memory/dream.py`, `.../memory/entities.py` |
| L2 and L1 | Dropped by broadcast, generation catch-up or failed compare-and-swap. L2 buckets are memory-only | `generation.py`, `test_kv_buckets_are_memory_only.py` |
| Epoch counters | Ephemeral epochs live in a memory KV bucket. A tile epoch is the exception and keeps a `config_epochs` row | `packages/epoch/README.md` |
| Superseded snapshot chunks | Deleted by the bucket's declarer, in bounded batches | `docs/design-scoped-snapshot.md` |
| Write marks | Drained and deleted by the reconciler | `write_marks.py` |

## Candidate norms (unratified)

- Every datetime column is `TIMESTAMPTZ` (`packages/core/tests/enforcement/test_no_datetime_type_columns.py`).
- A partitioned table's SQL names its partition column (`packages/core/tests/enforcement/test_partition_column_enforcement.py`).
- Migrations are idempotent, replay-guarded, and free of mixed DDL and DML (`packages/core/tests/enforcement/test_migration_yugabyte_safety.py`).
- Every KV bucket is memory-backed (`packages/core/tests/enforcement/test_kv_buckets_are_memory_only.py`).
- jsonb values are bound natively, never pre-encoded (`packages/core/tests/enforcement/test_jsonb_native_binding.py`).
- A switched-on write generation is declared on the schema (`tests/enforcement/test_write_generation_is_declared_on_the_schema.py`).
- A collection that caches absences writes L3 only through the framework (`packages/core/tests/enforcement/test_negative_cache_write_paths.py`).
- Audit `details` keys are classified (`tests/enforcement/test_audit_details_keys_are_classified.py`).
- Every package family member's version bound tracks the family version (`tests/enforcement/test_intra_family_version_bounds.py`).

## Open questions

1. The core README says writes "flow down (L1 -> L2 -> L3)" and also "L3 -> L1 -> L2". `save_entity`'s docstring supports L3 first. Is the first phrasing a stale description?
2. Which migration scope owns `conversations` against the platform's `namespaces`? `conversations` registers as AGENT scope, while `namespaces` (ACL) is PLATFORM, and tools alter `namespaces` from an AGENT-scope-adjacent package. The cross-scope foreign key story is not written down in the files read.
3. `docs/how-to-add-a-migration.md` cites `14-eng-ai-bot` hub files and `migrations-task-01-canonical-runner.md`, which now lives at `docs/done/migrations-task-01-canonical-runner.md`. Is the hub's runner builder the only place packages are wired?
4. Several enforcement tests default to `report` mode (`MIGRATION_ENFORCEMENT_MODE`, `PARTITION_ENFORCEMENT_MODE`). Which mode CI runs is not stated in the files read.
5. Expected row volumes and retention periods are not stated for any table except what `prune_audit_events` takes as a parameter.
6. State-transition rules for wake fires, intentions and identity versions (who may move a row from which state) are in package code not read here. Only the state lists are cited.
7. `docs/epoch-task-06-collection-generations-by-default.md` says stage 4 (flip the default so every collection carries a write generation) waits on a measurement. Which tables outside the access tables, knowledge and datasources are switched on is not confirmed.
8. `impersonation_gates` is declared as a `TableSchema` in `acl/collections.py`, but the ACL v001 migration lists five tables. Which migration creates it is not confirmed.
9. `3tears-evals` `documents` appears as a `CREATE TABLE` in the evals package. The host owns the store, and the table's ownership and migration path were not read.
10. `.prawduct/artifacts/` has no product brief. `depends_on` is empty until one exists.
