# 3tears v0.8.0 — Enriched TableSchema (one declaration, full SQLAlchemy semantics)

## Why this release exists

v0.7.5 shipped four hand-written SQLAlchemy `Table` factories in
`packages/agent/memory/src/threetears/agent/memory/collections.py`
(`memories_table`, `media_table`, `media_content_table`,
`memory_chunks_table`) to close the cross-repo schema-divergence trap
that broke metallm v0.14.6 on prod. Those factories duplicate the
column lists already declared in each Collection's `TableSchema` —
the divergence trap moved from "between repos" to "within one file,"
but the duplication itself is still there. Every column add to a
3tears-owned table still requires updating two declarations in the
same file.

The v0.7.5 release notes name this explicitly:

> A future 3tears v0.8.0 enriches TableSchema with native FK / Index /
> Enum / Vector-dim / Generated-column semantics so the factory
> implementations can collapse against a single declaration. Tracked
> separately as task shards.

v0.8.0 IS that follow-up. Six shards in this directory describe the
work.

## Locked design decisions (from the design discussion)

These decisions are pre-resolved. Shards reference them; do NOT
re-litigate inside an implementation shard.

| Decision | Locked answer |
|---|---|
| FK declaration shape | Two-shape: `Column.foreign_key=("table", "col")` for single + `TableSchema.foreign_keys=[ForeignKey(...)]` for composite |
| Enum constraints | `Column(..., enum_type=("preference", "fact", ...), enum_name="memory_type")` |
| Trigger-maintained columns (TSVECTOR) | Column-only declaration; trigger DDL stays in alembic |
| Migration scope | Every 3tears-owned `TableSchema`, in one release |
| Index API shape | `Index("name", "a", "b", unique=True, where="...")` — factory function returning frozen `IndexDef` dataclass |
| Composite FK type | `ForeignKey(local_cols, ref_table, ref_cols, on_delete=...)` — same factory pattern as `Index` |
| Vector dimension | `Column.vector_dim: int | None` per-column |
| Server defaults | `Column.server_default: str | None`, raw Postgres expression |
| Numeric type | `NUMERIC_TYPE = "numeric"` tag + `Column.precision: int | None` + `Column.scale: int | None` per-column (needed for `media_content.cost = Numeric(12, 8)`) |
| TSVECTOR type | `TSVECTOR_TYPE = "tsvector"` tag. Declared with `nullable=True, immutable=True` so Collection UPDATE generators exclude it (trigger maintains the column server-side). Trigger DDL stays in alembic |
| `on_delete` allowed values | `Literal["CASCADE", "SET NULL", "RESTRICT", "NO ACTION"]` |
| `on_update` | Not modeled in v0.8.0 (vanishingly rare in our codebase) |
| Naming | Index + FK names are required (no auto-generation) |
| Module location | All new types stay in `schema_backed.py` |
| Backward compat | All new fields default to `None` / empty list. Pre-v0.8.0 declarations keep working |
| CHECK constraints beyond enums | Not for v0.8.0 |
| Postgres enum vs CHECK | Prod uses real `CREATE TYPE ... AS ENUM(...)` (verified via `pg_type` query). `SAEnum(...)` default `native_enum=True` matches prod — DO NOT pass `native_enum=False` |
| Partition-column coercion | The existing partition-column-to-immutable rebuild in `TableSchema.__post_init__` MUST be extended to pass through all new v0.8.0 Column fields (otherwise a `Column("agent_id", partition=True, foreign_key=...)` would silently drop the FK) |

## Shard sequence

Shards are sequential. Each one depends on the previous.

| # | Shard | What it does |
|---|---|---|
| 01 | [API extension](shard-01-tableschema-api-extension.md) | Add `Index`, `ForeignKey`, enriched `Column` / `TableSchema` fields. Pure data-model additions. Validators. Unit tests. |
| 02 | [`to_sqlalchemy_table` impl](shard-02-to-sqlalchemy-table.md) | Convert enriched `TableSchema` → fully-shaped SQLAlchemy `Table`. Type-tag mapper, FK / Index / Enum / Vector translation. Roundtrip tests. |
| 03 | [Cross-package enrichment](shard-03-enrich-existing-schemas.md) | Add `foreign_keys` / `indexes` / `enum_type` / `vector_dim` / `server_default` to every 3tears-owned `TableSchema` declaration. |
| 04 | [Factory collapse](shard-04-factory-collapse.md) | Replace hand-written SQLAlchemy column lists in the six existing factories (`memories_table`, `media_table`, `media_content_table`, `memory_chunks_table`, `conversation_memory_refs_table`, `context_items_table`) with one-line calls to `to_sqlalchemy_table`. |
| 05 | [metallm v0.15.0 coordination](shard-05-metallm-bump.md) | Bump metallm's 3tears pin to v0.8.0. Run Alembic auto-gen parity check. Verify zero phantom migrations. metallm v0.15.0 release. |
| 06 | [Migration guide + release notes](shard-06-docs-and-release-notes.md) | Hand-written 3tears v0.8.0 GitHub Release notes. Migration guide for downstream consumers. |

## How shards work

Each shard is self-contained per `metallm/docs/TASK_TEMPLATE.md`. A
subagent should be able to pick up any shard and complete it without
asking questions (assuming previous shards in the sequence are done).

Code in shards is **prescriptive** (specific values that must match,
exact field signatures, exact mapping tables) rather than
**suggestive** (full implementations). The implementing subagent
writes the actual code.

## Verification of the whole release

Beyond per-shard success criteria, the v0.8.0 release as a whole must
produce **zero phantom migrations** from metallm Alembic auto-gen
against the seven tables registered on metallm's `sa_metadata` from
3tears: `memories`, `media`, `media_content`, `memory_chunks`,
`conversation_memory_refs`, `context_items`, `mcp_tool_grants`.

Shard 05 is the gate. "Zero phantom migrations" means:

- No `op.add_column` / `op.drop_column` against the seven tables
- No `op.create_index` / `op.drop_index` (including partial-unique
  indexes like `ix_memories_user_alias`) against the seven tables
- No `op.create_foreign_key` / `op.drop_constraint` against the
  seven tables
- No `op.alter_column` for type / nullable changes on the seven
  tables

Permitted noise (does NOT count as a phantom migration):

- `op.alter_column(server_default=...)` on TSVECTOR columns, where
  Postgres reports back a `server_default` for trigger-maintained
  columns that doesn't match the declaration. Document any specific
  occurrence in the shard 05 verification output but do not require
  the diff to be empty.

If auto-generate produces any non-permitted op on the seven tables,
the release is not ready. Other tables (workspace, ACL, etc.) are
out of scope for the parity gate because they are not registered on
metallm's `sa_metadata`.

## Scope of enrichment vs. parity

The 7 metallm-registered tables MUST be enriched to pass the parity
gate. The remaining 9 3tears-owned TableSchemas (workspaces,
workspace_files, workspace_file_versions, groups, group_members,
roles, role_assignments, namespaces, conversations) are enriched in
v0.8.0 as **good hygiene for downstream consumers** but do not affect
metallm's release gate. Shard 03 lists them explicitly with
appropriate priority labels.
