# Migration Helpers + Yugabyte-Safe Patterns

Canonical helpers for authoring migrations against YugabyteDB. Five free
async functions in `threetears.core.data.migrations.helpers`, plus an
AST walker enforcing yugabyte-safe shape across every migration file.

## The v054 Footgun

YugabyteDB's transactional DDL semantics differ from PostgreSQL: DDL
statements (`ALTER TABLE`, `CREATE TABLE`, etc.) auto-commit even
inside an explicit `BEGIN ... COMMIT` block. As a consequence, mixing
DDL and DML inside a single PL/pgSQL `DO` block causes the DML to
operate against a stale schema/snapshot.

```sql
-- BROKEN on YugabyteDB: the UPDATE silently no-ops because the ALTER
-- auto-committed and the DML ran against a stale snapshot.
DO $$
BEGIN
    ALTER TABLE namespaces ADD COLUMN row_scope VARCHAR(8) DEFAULT 'customer';
    UPDATE namespaces SET row_scope = 'platform' WHERE customer_id IS NULL;
END $$;
```

The yugabyte-safe shape splits each operation into its own transaction:

```python
await store.execute("ALTER TABLE namespaces ADD COLUMN IF NOT EXISTS row_scope ...")
await store.execute("UPDATE namespaces SET row_scope = 'platform' WHERE ...")
await store.execute("ALTER TABLE namespaces ADD CONSTRAINT ... CHECK ...")
```

The helpers in this module guarantee the split at construction time so
a migration author cannot reintroduce the bug by accident.

## Helpers

### `add_column_with_backfill`

Adds a column, optionally backfills it, and (optionally) installs a
CHECK gate. Splits ADD COLUMN, UPDATE backfill, and CHECK across
separate execute calls. Idempotent via `IF NOT EXISTS` + replay-guard
predicate.

```python
await add_column_with_backfill(
    store,
    table="namespaces",
    column="row_scope",
    column_type="VARCHAR(8)",
    default="'customer'",
    not_null=True,
    backfill_value_sql="'platform'",
    backfill_predicate="customer_id IS NULL",
)
```

### `add_check_constraint`

Adds a CHECK constraint with an `information_schema` existence probe.
Single execute (DDL only).

```python
await add_check_constraint(
    store,
    table="namespaces",
    constraint_name="namespaces_row_scope_ck",
    expression="row_scope IN ('platform', 'customer')",
)
```

### `replace_check_constraint`

Replaces an existing CHECK constraint via `pg_get_constraintdef`
compare-then-swap. Single execute. Pass `engine_normalized_def` for
OID-stable replay (the engine canonicalises CHECK expressions; a
literal target_def DROP+ADDs every replay).

```python
await replace_check_constraint(
    store,
    table="namespaces",
    constraint_name="namespaces_namespace_type_ck",
    new_expression="namespace_type IN ('agent', 'shared', ...)",
    engine_normalized_def=(
        "CHECK ((namespace_type)::text = ANY "
        "(ARRAY['agent'::character varying, ...]::text[]))"
    ),
)
```

### `replace_primary_key`

Encodes the v054 PK-swap dance: drop inbound FKs, drop old PK, add new
composite PK, add UNIQUE on the preserved id column, recreate inbound
FKs with original ON DELETE clauses. Single execute (DDL only --
nothing the yugabyte DDL/DML separation rule cares about).

```python
await replace_primary_key(
    store,
    table="agents",
    new_columns=("customer_id", "id"),
    inbound_fks=(
        InboundFk(
            source_table="channel_configs",
            constraint_name="channel_configs_agent_id_fkey",
            source_column="agent_id",
            on_delete="",
        ),
    ),
)
```

### `add_partition_column`

Convenience composing `add_column_with_backfill` + `add_check_constraint`
for the partition-column-add pattern.

```python
await add_partition_column(
    store,
    table="namespaces",
    column="row_scope",
    column_type="VARCHAR(8)",
    default="'customer'",
    backfill_value_sql="'platform'",
    backfill_predicate="customer_id IS NULL",
    add_check_allowed_values=("platform", "customer"),
)
```

### `add_index`

Thin `CREATE INDEX IF NOT EXISTS` wrapper. Supports `unique=True` and
partial indexes via `where=`.

```python
await add_index(
    store,
    table="namespaces",
    name="idx_namespaces_active",
    columns=("name",),
    where="status = 'active'",
)
```

## AST Walker

`threetears.core.data.migrations.enforcement` walks every `*.py` under
the configured migration directories and applies five rules:

- **M-1 (CRITICAL)**: DO blocks containing both DDL and DML
- **M-2 (HIGH)**: backfill UPDATE without replay-guard predicate
- **M-3 (MEDIUM)**: non-idempotent DDL lacking IF NOT EXISTS / IF EXISTS
- **M-4 (HIGH)**: composite PRIMARY KEY ADD without sibling UNIQUE
- **M-5 (HIGH)**: TRUNCATE TABLE in migrations (pre-GA exemptions
  with rationale allowed)

Mode controlled by `MIGRATION_ENFORCEMENT_MODE` env var. Defaults to
`strict` (sub-task 7 flip). Set to `report` during cleanup windows.

Each repo carries a per-repo invocation in
`tests/enforcement/test_migration_yugabyte_safety.py` and an exemption
file at `tests/enforcement/_migration_exemptions.txt`. Exemption file
format:

```
<file_path>:<rule_name> # rationale: <specific reason>
```

Blank rationales ("internal access needed", "tests need this") are
rejected by the walker meta-test.

## When to Use Helpers vs Raw Execute

**Use a helper** when:
- You're adding a column with a backfill (the v054 footgun shape)
- You're adding or replacing a CHECK constraint
- You're swapping a primary key (composite PK + UNIQUE preservation)
- You're adding an index

**Use raw `store.execute`** when:
- The operation is a single DDL statement that doesn't fit any helper
  shape (e.g. `ALTER COLUMN ... SET NOT NULL`)
- You need a one-off pattern the helpers don't (yet) cover; consider
  promoting the pattern to a helper if it surfaces twice

The helpers are intentionally conservative: each one solves one
specific shape. Promoting a new helper requires the pattern to surface
in at least two migrations.
