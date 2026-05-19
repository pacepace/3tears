# v0.8.0-task-03: Enrich every 3tears-owned TableSchema with FK / Index / Enum / Vector / ServerDefault declarations

## Objective

Add the new optional fields (foreign keys, indexes, enum_type, vector_dim, server_default, TSVECTOR_TYPE) to every `TableSchema` declaration in 3tears packages so that `to_sqlalchemy_table` (shard 02) produces fully-shaped SQLAlchemy Tables matching the existing v0.7.5 hand-written factory output.

This shard depends on shards 01 + 02. Do not start until both have landed.

---

## Locked design decisions (from README.md)

All API choices are pre-resolved. This shard adds existing-knowledge declarations to schemas — no new design.

---

## Scope: 7 critical + 9 hygiene

A grep run at design time enumerates **16 production TableSchemas** in 3tears (excluding tests and `schema_backed.py`'s own example). Of these, **7 are registered on metallm's `sa_metadata`** and thus participate in the v0.8.0 Alembic-parity release gate (shard 05). The remaining 9 are not registered in metallm but still benefit from enrichment (downstream 3tears consumers, future host applications, internal hygiene).

### Critical-for-parity (7 — MUST enrich for shard 05 to pass)

These tables ARE registered in metallm `api/src/data/models.py` via factory calls. Enrichment + factory collapse (shard 04) directly affects metallm Alembic auto-gen.

| Table | Collection | File |
|---|---|---|
| `memories` | `MemoriesCollection` | `packages/agent/memory/src/threetears/agent/memory/collections.py:711` |
| `media` | `MediaCollection` | `packages/agent/memory/src/threetears/agent/memory/collections.py:1696` |
| `media_content` | `MediaContentCollection` | `packages/agent/memory/src/threetears/agent/memory/collections.py:1745` |
| `memory_chunks` | `MemoryChunkCollection` | `packages/agent/memory/src/threetears/agent/memory/collections.py:2209` |
| `conversation_memory_refs` | `MemoryRefsCollection` | `packages/agent/memory/src/threetears/agent/memory/collections.py:2945` |
| `context_items` | `ContextItemCollection` | `packages/agent/tools/src/threetears/agent/tools/collections.py:90` |
| `mcp_tool_grants` | (factory only — no Collection-level TableSchema today) | `packages/mcp/src/threetears/mcp/rbac.py` |

### Hygiene-only (9 — enrich for completeness but DOES NOT affect metallm parity gate)

These tables exist in 3tears but are NOT in metallm's `sa_metadata`. Verified by `grep -rn "<CollectionName>" api/src/` returning zero hits for each. The hygiene-only enrichment improves the schema declarations for any future downstream consumer.

| Table | Collection | File | Why hygiene-only |
|---|---|---|---|
| `workspaces` | `WorkspacesCollection` | `packages/agent/workspace/src/threetears/agent/workspace/collections.py:58` | not used by metallm |
| `workspace_files` | `WorkspaceFilesCollection` | same package | not used by metallm |
| `workspace_file_versions` | `WorkspaceFileVersionsCollection` | same package | not used by metallm |
| `groups` | `GroupsCollection` | `packages/agent/acl/src/threetears/agent/acl/collections.py` | platform-managed; not in metallm sa_metadata |
| `group_members` | `GroupMembersCollection` | same package | same |
| `roles` | `RolesCollection` | same package | same |
| `role_assignments` | `RoleAssignmentsCollection` | same package | same |
| `namespaces` | `NamespacesCollection` | same package | same |
| `conversations` | `ConversationsCollection` (3tears) | `packages/conversations/src/threetears/conversations/collection.py` | metallm uses its own `api/src/data/collections/conversations.py` instead |

Implementing subagent: **prioritize the 7 critical tables first** and ship the shard 05 parity gate green. Then enrich the 9 hygiene tables in the same PR — they're easier than the critical ones (smaller column sets, fewer FKs) and the parity tests in shard 02 only need to cover the 6 factory tables (no factory exists for mcp_tool_grants today, but it's registered as a Table on metallm sa_metadata via the `mcp_tool_grants_table` factory in 3tears MCP package — confirm the factory location with `grep -rn "mcp_tool_grants_table" packages/mcp/`).

## Files to Modify

Locate each TableSchema declaration via the table above and enrich in-place. No new files. No new packages.

## Files to NOT Modify

## Files to NOT Modify

- `packages/agent/memory/src/threetears/agent/memory/collections.py` factories (`memories_table`, `media_table`, etc.) — DO NOT touch them in this shard. Their collapse happens in shard 04, AFTER schemas are enriched and the parity tests from shard 02 confirm structural equivalence.
- `packages/core/src/threetears/core/collections/schema_backed.py` — already done in shards 01+02.

---

## Research

The authoritative source for FK / Index / Enum / Default information per table is:
1. **The existing v0.7.5 factories** in `packages/agent/memory/src/threetears/agent/memory/collections.py` and `packages/agent/tools/src/threetears/agent/tools/collections.py`. These hand-write the SQLAlchemy declarations; copy the structural details into the corresponding TableSchema.
2. **The 3tears migration files** in each package's `migrations/v*.py`. These create the actual indexes + FKs in prod.
3. **The metallm Postgres schema dump** (already done in the v0.14.7 work, see prod `information_schema.columns`) for tables that exist in metallm — confirms exactly what's deployed.

When the migration files and the v0.7.5 factory disagree, the migration files are authoritative for prod. Cross-check by running the metallm alembic + 3tears migrations against a fresh testcontainer and inspecting `information_schema.table_constraints` / `pg_indexes`.

---

## Patterns to Follow

Worked example: enriching `MemoriesCollection.schema` from its current shape to the v0.8.0 enriched shape.

### Before (current v0.7.5):

```python
class MemoriesCollection(SchemaBackedCollection[MemoryEntity]):
    primary_key_column: str | tuple[str, ...] = ("agent_id", "memory_id")
    schema = TableSchema(
        name="memories",
        primary_key=("agent_id", "memory_id"),
        columns=[
            Column("memory_id", UUID_TYPE),
            Column("agent_id", UUID_TYPE, partition=True),
            Column("customer_id", UUID_TYPE),
            Column("user_id", UUID_TYPE),
            Column("conversation_id", UUID_TYPE, immutable=True),
            Column("message_id_source", UUID_TYPE, nullable=True),
            Column("type_memory", STRING_TYPE),
            Column("content", STRING_TYPE),
            Column("summary", STRING_TYPE, nullable=True),
            Column("embedding", VECTOR_TYPE, nullable=True),
            Column("alias", STRING_TYPE, nullable=True),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
            Column("date_updated", DATETIMETZ_TYPE, nullable=True),
        ],
        cas_column="date_updated",
    )
```

### After (v0.8.0 enriched):

```python
class MemoriesCollection(SchemaBackedCollection[MemoryEntity]):
    primary_key_column: str | tuple[str, ...] = ("agent_id", "memory_id")
    schema = TableSchema(
        name="memories",
        primary_key=("agent_id", "memory_id"),
        columns=[
            Column("memory_id", UUID_TYPE),
            Column("agent_id", UUID_TYPE, partition=True),
            Column("customer_id", UUID_TYPE),
            Column("user_id", UUID_TYPE,
                   foreign_key=("users", "user_id")),
            Column("conversation_id", UUID_TYPE, immutable=True),
            Column("message_id_source", UUID_TYPE, nullable=True,
                   foreign_key=("messages", "message_id")),
            Column("type_memory", ENUM_TYPE,
                   enum_type=("preference", "fact", "decision",
                              "topical_context", "relational_context"),
                   enum_name="memory_type"),
            Column("content", STRING_TYPE),
            Column("summary", STRING_TYPE, nullable=True),
            Column("embedding", VECTOR_TYPE, vector_dim=1024,
                   nullable=True),
            Column("search_vector", TSVECTOR_TYPE, nullable=True,
                   immutable=True),
            Column("alias", STRING_TYPE, nullable=True),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
            Column("date_updated", DATETIMETZ_TYPE, nullable=True),
        ],
        cas_column="date_updated",
        indexes=(
            Index("ix_memories_user_date", "user_id", "date_created"),
        ),
    )
```

Key changes from the worked example (these are the patterns to apply across every schema):

1. **Single-column FKs**: `Column(..., foreign_key=("ref_table", "ref_col"))`. Source: the v0.7.5 factory's `SAColumn(..., ForeignKey("ref_table.ref_col"))` lines.
2. **Composite FKs**: not present in `memories` but present in `media_content` / `memory_chunks`. Use `foreign_keys=(ForeignKey(local_cols, ref_table, ref_cols, on_delete=...),)` at TableSchema level.
3. **ENUM columns**: `Column(col_name, ENUM_TYPE, enum_type=(...), enum_name="...")`. Source: the v0.7.5 factory's `SAEnum(...)` declarations.
4. **VECTOR columns**: `Column(col_name, VECTOR_TYPE, vector_dim=1024)`. Source: the v0.7.5 factory's `Vector(_MEMORY_VECTOR_DIM)` calls.
5. **TSVECTOR columns**: `Column(col_name, TSVECTOR_TYPE, nullable=True, immutable=True)`. Mark `immutable=True` because the value comes from the Postgres trigger, not Collection writes. This is a behavioural change — UPDATE generators will now exclude search_vector from `SET` clauses. Verify no existing test relies on `search_vector` appearing in UPDATE output.
6. **Server defaults**: `Column(col_name, ..., server_default="image")`. Source: factory's `SAColumn(..., server_default=...)` lines.
7. **Indexes**: `indexes=(Index("name", "col1", "col2", unique=..., where="..."),)` at TableSchema level. Source: factory's `Index(...)` table-args lines.

---

## Per-table enrichment checklists

Per table, derive from the v0.7.5 factory + cross-check against the migration files. The list below is the expected enrichment for each table; the implementing subagent should verify each against the factory and migration files before applying.

### `memories` (worked example above) — CRITICAL
- single-col FKs: `user_id → users.user_id`, `message_id_source → messages.message_id`
- enum: `type_memory` with 5 values + name `memory_type`
- vector: `embedding` with `vector_dim=1024`
- TSVECTOR: `search_vector` (add the column to the TableSchema, currently missing)
- indexes:
  - `ix_memories_user_date(user_id, date_created)` — from v0.7.5 factory
  - `ix_memories_user_alias(agent_id, user_id, alias) WHERE alias IS NOT NULL UNIQUE` — currently lives in metallm alembic 088 only, NOT in the v0.7.5 factory (factory comment at lines 261-267 says "metallm-deploy-specific"). v0.8.0 declares this index in 3tears: the partial-unique semantic is intrinsic to the `alias` column contract (per-user uniqueness within an agent partition), not a deployment choice. Declare it in 3tears so the parity gate stays clean.

### `media` — CRITICAL
- single-col FKs: `user_id → users.user_id`, `cloud_connection_id → cloud_connections.cloud_connection_id` (on_delete `SET NULL`)
- composite FK: `(agent_id, memory_id) → memories(agent_id, memory_id)` on_delete `CASCADE`
- server_defaults: `metadata_json="{}"`, `media_category="image"`, `extraction_status="none"`
- indexes: `ix_media_user_date(user_id, date_created)`, `ix_media_mime_type(mime_type)`, `ix_media_memory_id(memory_id)`, `uq_media_cloud_connection_file(cloud_connection_id, cloud_file_id) UNIQUE`
- Add to TableSchema columns: s3_key, mime_type, size_bytes, source, generation_prompt, thumbnail_s3_key, cloud_file_id, cloud_file_url (currently missing from TableSchema, present in v0.7.5 factory)

### `media_content` — CRITICAL
- single-col FKs: `user_id → users.user_id`, `model_id → models.model_id`, `provider_id → providers.provider_id`
- composite FK: `(agent_id, media_id) → media(agent_id, media_id)` on_delete `CASCADE`
- vector: `embedding` with `vector_dim=1024`
- TSVECTOR: `search_vector`
- numeric: `cost` with `precision=12, scale=8` — uses `NUMERIC_TYPE` (added in shard 01)
- indexes: `ix_media_content_media_type(media_id, content_type)`, `ix_media_content_user(user_id)`
- Add to TableSchema columns: model_name, provider_name, token_count_prompt, token_count_completion, cost (NUMERIC_TYPE with precision/scale) — currently missing

### `memory_chunks` — CRITICAL
- single-col FK: `user_id → users.user_id`
- composite FK: `(agent_id, memory_id) → memories(agent_id, memory_id)` on_delete `CASCADE`
- vector: `embedding` with `vector_dim=1024`
- TSVECTOR: `search_vector`
- indexes: `ix_memory_chunks_memory(memory_id, chunk_index)`, `ix_memory_chunks_user(user_id)`
- Add to TableSchema columns: `summary`, `heading_context`, `page_number`, `token_count`, `message_id_start`, `message_id_end` if not present

### `conversation_memory_refs` — CRITICAL
- columns already in TableSchema, no FKs, no indexes per the v0.7.5 factory
- consult `packages/agent/memory/src/threetears/agent/memory/migrations/v002_create_conversation_memory_refs.py` for any indexes the migration creates that the factory omits. Apply if present.

### `context_items` — CRITICAL
- consult `packages/agent/tools/src/threetears/agent/tools/migrations/v001_create_context_items_table.py` for the canonical FK + index list (the v0.7.5 factory at `collections.py:45` is intentionally minimal and does NOT enumerate the indexes that the migration creates).
- Likely indexes (verify against the migration file): `idx_ctx_conversation`, `idx_ctx_conversation_type`, plus the partial unique implied by `upsert_variable`'s `ON CONFLICT (conversation_id, key) WHERE context_type = 'variable'` clause. Declare each one explicitly in the TableSchema.
- FKs: verify against the migration; `conversation_id` likely references `conversations` but may not be declared as an FK if the conversations table is migrated separately.

### `mcp_tool_grants` — CRITICAL
- TableSchema source: `packages/mcp/src/threetears/mcp/rbac.py` (look for the `class ...Collection` declaration)
- Factory: `packages/mcp/src/threetears/mcp/<file>.py` — locate via `grep -rn "mcp_tool_grants_table" packages/mcp/`
- Migration: `packages/mcp/src/threetears/mcp/migrations/v001_create_mcp_tool_grants.py`. Declares two indexes: `idx_mcp_tool_grants_principal(principal_id, permission)`, `idx_mcp_tool_grants_tool(tool_name)`
- FKs + columns: derive from the migration file
- This is the only one of the 7 critical tables that may NOT have a pre-existing v0.7.5-style factory in the same shape — the implementing subagent should compare what the metallm side registers (`api/src/data/models.py:1129` imports `mcp_tool_grants_table as _register_mcp_tool_grants_table`) against what enrichment produces.

### `workspaces` — HYGIENE
- TableSchema source: `packages/agent/workspace/src/threetears/agent/workspace/collections.py:58`
- Migration: `packages/agent/workspace/src/threetears/agent/workspace/migrations/v001_create_workspace_tables.py`. Indexes include `idx_workspace_files_workspace(workspace_id)`. Declare per-table.

### `workspace_files` + `workspace_file_versions` — HYGIENE
- Same migration file as `workspaces`. Apply same enrichment pattern. The v001 migration declares `idx_workspace_file_versions_history` — assign to the right TableSchema by reading the migration.

### `groups` / `group_members` / `roles` / `role_assignments` / `namespaces` — HYGIENE
- TableSchema source: `packages/agent/acl/src/threetears/agent/acl/collections.py`
- These tables are PLATFORM-MANAGED (not created by 3tears migrations). DDL lives externally. The TableSchema declarations exist for Collection ops only.
- For hygiene enrichment: read the columns declared in `collections.py`, infer FK relationships from the column names + the test fixture DDL at `packages/agent/workspace/tests/integration/test_cross_agent_workspace.py:137+` (which CREATEs these tables for integration tests).
- No metallm parity gate impact; do not block on these.

### `conversations` (3tears version) — HYGIENE
- TableSchema source: `packages/conversations/src/threetears/conversations/collection.py`
- Migration: `packages/conversations/src/threetears/conversations/migrations/v001_create_conversations_table.py` declares indexes `idx_conv_user(user_id, date_created)`, `idx_conv_customer(customer_id, date_created)`, `idx_conv_status(status)`. Plus the v005 migration adds the `search_vector` GIN index.
- This is the 3tears version; metallm has its OWN ConversationsCollection in `api/src/data/collections/conversations.py`. Enriching the 3tears version does NOT affect metallm.

## Common reference: migration files vs factories

When the migration file and the v0.7.5 factory disagree, the migration file is **authoritative for prod**. The implementing subagent MUST consult the migration files for every table (not just the factory) because the v0.7.5 factories were hand-written from imperfect knowledge of the migration state — explicit caveats in their own comments (e.g., the memories factory's "metallm-deploy-specific" note about `ix_memories_user_alias`) indicate known gaps.

For the metallm parity gate, the implementing subagent should also verify against prod schema (via `./scripts/db-query.sh sql /tmp/check.sql` in the metallm dev DB which has all migrations applied) before declaring enrichment complete on the 7 critical tables.

---

## Verification per-table

After enriching each schema, the parity tests from shard 02 should still pass — the enriched TableSchema's `to_sqlalchemy_table` output must structurally match the existing v0.7.5 factory's output for the same table. If a parity test fails after shard 03 enrichment, the enriched schema is wrong; fix the schema, not the test.

Whole-shard verification:
```bash
cd /Users/pace/crypt/pub/dev-wsl/vscode/3tears/3tears
uv run pytest packages/core/tests/unit/collections/test_to_sqlalchemy_table.py -v  # all 6 parity tests must pass
uv run pytest packages/ tests/ -m "not integration" -q
uv run ruff check . && uv run ruff format . --check
uv run mypy --explicit-package-bases -p threetears.core -p threetears.agent.memory -p threetears.agent.tools
```

---

## Anti-patterns

- DO NOT add columns to a TableSchema that don't exist in prod. Run the metallm dev DB schema inspection (`./scripts/db-query.sh sql /tmp/inspect-tables.sql` from the metallm v0.14.7 work) to verify each column truly exists before declaring it.
- DO NOT change behaviour of existing Collection methods in this shard. Pure schema-declaration additions only.
- DO NOT touch the v0.7.5 factory bodies. Their collapse is shard 04.
- DO NOT modify Alembic migration files. They're history; this shard is declaration-only.
- AVOID guessing at FK / index details. Cross-reference v0.7.5 factory + migrations + prod schema before declaring.

---

## TSVECTOR write-path audit (REQUIRED)

Declaring `Column("search_vector", TSVECTOR_TYPE, nullable=True, immutable=True)` changes Collection write behaviour. Subagent MUST audit and verify before considering this shard complete:

1. **Read the SQL generator implementations** in `packages/core/src/threetears/core/collections/schema_backed.py`: `_build_insert_sql`, `_build_upsert_sql` (the `ON CONFLICT DO UPDATE SET <mutable>` path), and `_build_cas_update_sql`.

2. **Confirm semantics:**
   - INSERT column list: TSVECTOR columns SHOULD be excluded (Postgres trigger will populate; the row insert must not provide a value or the trigger may behave unexpectedly). Verify the generator excludes immutable-but-trigger-maintained columns from INSERT, OR explicitly omits the search_vector column via some other mechanism today. Document what you find.
   - UPDATE SET clause: TSVECTOR columns MUST be excluded — that's the `immutable=True` semantics. Verify the existing generator handles this correctly.
   - CAS path: TSVECTOR is never the CAS column, so no special handling needed; but verify the existing CAS generator also respects immutable.

3. **Add new explicit tests** even if no existing test breaks:
   - `test_immutable_tsvector_excluded_from_update_set` — generate an UPDATE SQL string from a TableSchema with TSVECTOR column, assert the column name does NOT appear in SET clause
   - `test_immutable_tsvector_in_insert_column_list` — generate an INSERT SQL string, document and assert the actual behaviour (whether TSVECTOR is included or excluded; correct behaviour depends on what Postgres expects vs the trigger)
   - `test_tsvector_save_entity_against_real_postgres` — integration test that inserts via Collection.save_entity and confirms the trigger populates search_vector (in pgvector/pgvector:pg16 testcontainer)

4. **If any existing Collection test (in `packages/agent/memory/tests/`, etc.) reads or writes search_vector explicitly**, document the test + decide whether it needs updating. The expected case is "no test cares about search_vector" — but the audit MUST confirm.

## Success Criteria

- [ ] Each of the **7 critical** TableSchemas enriched with `foreign_keys` / `indexes` / column-level annotations matching prod schema (verified via `./scripts/db-query.sh` on metallm dev DB)
- [ ] `memories` TableSchema includes the `ix_memories_user_alias` partial-unique index (relocated from metallm alembic 088)
- [ ] `media_content` TableSchema uses `NUMERIC_TYPE` for `cost` column (added in shard 01)
- [ ] All three search_vector columns declared as `Column(name, TSVECTOR_TYPE, nullable=True, immutable=True)`
- [ ] TSVECTOR write-path audit complete: SQL generators inspected, new tests added, any existing test that referenced search_vector documented
- [ ] Each of the **9 hygiene** TableSchemas enriched best-effort against the package's own migration files (no metallm parity requirement)
- [ ] All 6 parity tests in `test_to_sqlalchemy_table.py` pass after enrichment (NB: tests now compare against hand-written reference Tables per shard 04, NOT against the v0.7.5 factory output — see shard 04)
- [ ] Full CI suite passes
- [ ] Ruff + mypy clean
- [ ] Brief commit message enumerating each enriched table

---

## Verification

```bash
cd /Users/pace/crypt/pub/dev-wsl/vscode/3tears/3tears
uv run pytest packages/core/tests/unit/collections/test_to_sqlalchemy_table.py -v
uv run pytest packages/ tests/ -m "not integration" -q
uv run ruff check . && uv run ruff format . --check
uv run mypy --explicit-package-bases -p threetears.core -p threetears.agent.memory -p threetears.agent.tools
```

All four MUST exit 0.

---

## Enforcement Test Suggestions

- [ ] Drift risk: future column adds to a 3tears-owned table land in the alembic migration but not in the TableSchema. Suggested test: compare each TableSchema's column-name set against a structural snapshot of the prod schema (e.g. via a fixture that runs all migrations against a testcontainer and dumps `information_schema.columns`). Useful guard against the v0.14.6-class divergence trap. Flag for review.
- [ ] Drift risk: the same parity-test pattern from shard 02 should run continuously in CI to catch future enrichment drift. Already covered by shard 02 tests; no new test needed.
