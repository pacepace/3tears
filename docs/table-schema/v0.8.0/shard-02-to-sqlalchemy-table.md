# v0.8.0-task-02: Implement `TableSchema.to_sqlalchemy_table(metadata)`

## Objective

Convert an enriched `TableSchema` (as defined by shard 01) into a fully-shaped SQLAlchemy `Table` registered on the caller's `MetaData`. The output Table must be **structurally identical** to the hand-written tables in the existing v0.7.5 factories (FKs, indexes, enum CHECK constraints, vector dimensions, server defaults, partition-PK shape) so that downstream Alembic auto-gen produces zero phantom migrations.

This shard depends on shard 01. Do not start until shard 01 lands.

---

## Locked design decisions (from README.md)

All API choices are pre-resolved. This shard implements the conversion, not the API.

---

## Files to Modify

- `packages/core/src/threetears/core/collections/schema_backed.py` — add `TableSchema.to_sqlalchemy_table()` method + a private `_TYPE_MAPPING` table

## Files to Create

- `packages/core/tests/unit/collections/test_to_sqlalchemy_table.py` — roundtrip + parity tests

---

## Patterns to Follow

The four existing v0.7.5 factories are the parity reference:
- `packages/agent/memory/src/threetears/agent/memory/collections.py:memories_table` (and three siblings)
- `packages/agent/memory/src/threetears/agent/memory/collections.py:conversation_memory_refs_table`
- `packages/agent/tools/src/threetears/agent/tools/collections.py:context_items_table`

`to_sqlalchemy_table` output for the corresponding `TableSchema` (once enriched in shard 03) must produce the same `Table` shape these hand-written factories produce today.

---

## Method signature

```python
class TableSchema:
    def to_sqlalchemy_table(self, metadata: "sa.MetaData") -> "sa.Table":
        """Register this schema's canonical SQLAlchemy ``Table`` on ``metadata``.

        Idempotent: if a table with ``self.name`` already exists on the
        passed metadata, returns the existing Table without re-registering.

        :param metadata: SQLAlchemy metadata to attach the table to
        :return: the registered Table
        """
        ...
```

Imports of SQLAlchemy types stay inside the method body (or behind a `TYPE_CHECKING` guard at module top) so `schema_backed.py` doesn't carry a hard SQLAlchemy import for code paths that never touch it.

---

## Prescriptive type mapping

Module-level mapping from 3tears tag → SQLAlchemy type constructor. Lazy-imports `pgvector` like the v0.7.5 factories do.

```python
# Inside the method (or at module scope behind TYPE_CHECKING):
from sqlalchemy import Boolean, DateTime, Enum as SAEnum, Integer, Numeric, Text
from sqlalchemy.dialects.postgresql import BYTEA, JSONB, TSVECTOR, UUID as PgUUID

_TYPE_MAPPING: dict[str, Callable[[Column], Any]] = {
    UUID_TYPE: lambda col: PgUUID(as_uuid=True),
    STRING_TYPE: lambda col: Text(),
    DATETIMETZ_TYPE: lambda col: DateTime(timezone=True),
    JSONB_TYPE: lambda col: JSONB(),
    BYTES_TYPE: lambda col: BYTEA(),
    INT_TYPE: lambda col: Integer(),
    BOOL_TYPE: lambda col: Boolean(),
    VECTOR_TYPE: lambda col: _require_pgvector()(col.vector_dim),
    NUMERIC_TYPE: lambda col: Numeric(col.precision, col.scale),
    TSVECTOR_TYPE: lambda col: TSVECTOR(),
    ENUM_TYPE: lambda col: SAEnum(
        *col.enum_type,
        name=col.enum_name,
        create_constraint=True,
    ),
}
```

Notes:

- VECTOR_TYPE / NUMERIC_TYPE entries assume `vector_dim` / `precision` / `scale` are set (shard 01 validators enforce).
- `_require_pgvector()` already exists in `packages/agent/memory/src/threetears/agent/memory/collections.py`; this shard should refactor it to live in `schema_backed.py` (or import + re-export) so both `to_sqlalchemy_table` and the agent-memory factories share one source. **Do not duplicate the helper.**
- **ENUM_TYPE uses SAEnum's default `native_enum=True`** — emits `CREATE TYPE name AS ENUM(...)` and uses the type on the column. This matches prod reality: every enum-column in our prod schema (memories.type_memory, conversations.status, messages.role, users.role, etc.) IS a real Postgres enum type registered in `pg_type`, NOT a VARCHAR + CHECK constraint. Verified at design time via `SELECT typname FROM pg_type WHERE typcategory='E'` on the dev DB (which has all migrations applied; equivalent to prod). **DO NOT pass `native_enum=False`** — that would emit VARCHAR + CHECK and Alembic auto-gen would want to drop the real enum types on every run.
- TSVECTOR_TYPE + NUMERIC_TYPE are added by shard 01 (along with their validators). This shard maps them to SQLAlchemy types.

---

## Conversion behavior

### Columns

For each `Column` in `self.columns`:

1. Build the SQLAlchemy type via `_TYPE_MAPPING[col.column_type](col)`.
2. Compute primary-key flag: `col.name in self.pk_columns`.
3. Compute nullable: `col.nullable` (existing field).
4. Construct kwargs:
   - `primary_key=True` if PK
   - `nullable=False` if PK, else `col.nullable`
   - `server_default=col.server_default` if non-None
5. If `col.foreign_key` is set, wrap the column with an inline `sa.ForeignKey`:
   ```python
   sa.Column(
       col.name,
       sa_type,
       sa.ForeignKey(f"{col.foreign_key[0]}.{col.foreign_key[1]}"),
       primary_key=...,
       nullable=...,
       server_default=...,
   )
   ```
6. Otherwise just `sa.Column(col.name, sa_type, **kwargs)`.

### Composite FKs

For each `fk` in `self.foreign_keys`:
```python
sa.ForeignKeyConstraint(
    list(fk.local_cols),
    [f"{fk.ref_table}.{ref_col}" for ref_col in fk.ref_cols],
    ondelete=fk.on_delete if fk.on_delete != "NO ACTION" else None,
)
```

Note: SQLAlchemy treats `ondelete=None` as "NO ACTION" (the SQL default). Passing the string would emit `ON DELETE NO ACTION` redundantly. Match the hand-written factories' behavior — they don't emit redundant `NO ACTION`.

### Indexes

For each `idx` in `self.indexes`:
```python
sa.Index(
    idx.name,
    *idx.columns,
    unique=idx.unique,
    postgresql_where=sa.text(idx.where) if idx.where else None,
)
```

### Final assembly

```python
def to_sqlalchemy_table(self, metadata):
    if self.name in metadata.tables:
        return metadata.tables[self.name]

    sa_cols = [self._build_sa_column(col) for col in self.columns]
    sa_fks = [self._build_sa_foreign_key_constraint(fk) for fk in self.foreign_keys]
    sa_indexes = [self._build_sa_index(idx) for idx in self.indexes]

    return sa.Table(
        self.name,
        metadata,
        *sa_cols,
        *sa_fks,
        *sa_indexes,
    )
```

---

## Parity tests (the most important tests in this shard)

The bar: feed a hand-enriched `TableSchema` matching the existing v0.7.5 factory output, call `to_sqlalchemy_table`, and assert the result is structurally identical to what the factory produces.

Test structure (one test per existing factory):

```python
def test_parity_memories_table():
    # Hand-write the TableSchema enriched with all v0.7.5 details.
    enriched_schema = TableSchema(
        name="memories",
        primary_key=("agent_id", "memory_id"),
        columns=[
            Column("memory_id", UUID_TYPE),
            Column("agent_id", UUID_TYPE, partition=True),
            Column("customer_id", UUID_TYPE, nullable=False),
            Column("user_id", UUID_TYPE, foreign_key=("users", "user_id")),
            Column("conversation_id", UUID_TYPE, nullable=False),
            Column("message_id_source", UUID_TYPE, nullable=True,
                   foreign_key=("messages", "message_id")),
            Column("type_memory", ENUM_TYPE,
                   enum_type=("preference", "fact", "decision",
                              "topical_context", "relational_context"),
                   enum_name="memory_type"),
            Column("content", STRING_TYPE),
            Column("summary", STRING_TYPE, nullable=True),
            Column("embedding", VECTOR_TYPE, vector_dim=1024, nullable=True),
            Column("search_vector", TSVECTOR_TYPE, nullable=True, immutable=True),
            Column("alias", STRING_TYPE, nullable=True),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
            Column("date_updated", DATETIMETZ_TYPE, nullable=True),
        ],
        indexes=(
            Index("ix_memories_user_date", "user_id", "date_created"),
        ),
    )

    # Build the SQLAlchemy table two ways and assert structural equality.
    m1 = sa.MetaData()
    via_factory = memories_table(m1)  # existing v0.7.5 factory

    m2 = sa.MetaData()
    via_to_sqla = enriched_schema.to_sqlalchemy_table(m2)

    _assert_tables_equivalent(via_factory, via_to_sqla)
```

Equivalence helper `_assert_tables_equivalent` compares the following structural fields. Read each Index's `postgresql_where` via `dialect_kwargs` (NOT `dialect_options`, which is internal-state and has versioned shape):

```python
def _index_signature(idx: sa.Index) -> tuple[str, frozenset[str], bool, str | None]:
    """Return a structural signature for an Index."""
    where = idx.dialect_kwargs.get("postgresql_where")
    return (
        idx.name,
        frozenset(c.name for c in idx.columns),
        bool(idx.unique),
        str(where.compile(compile_kwargs={"literal_binds": True})) if where is not None else None,
    )

def _fk_signature(constraint) -> tuple[tuple[str, ...], str, tuple[str, ...], str | None]:
    """Return a structural signature for a ForeignKeyConstraint."""
    local_cols = tuple(c.name if hasattr(c, 'name') else c for c in constraint.column_keys)
    ref_table = constraint.elements[0].column.table.name
    ref_cols = tuple(el.column.name for el in constraint.elements)
    return (local_cols, ref_table, ref_cols, constraint.ondelete)
```

The helper compares:
- `set(col.name for col in t.columns)` — column-name sets match
- For each column: `type(col.type).__name__` (type class), `col.primary_key`, `col.nullable`, `col.server_default.arg` if non-None
- `set(col.name for col in t.primary_key.columns)` — PK column sets match
- `set(_index_signature(i) for i in t.indexes)` — index signatures match (catches name, columns, unique, partial WHERE)
- `set(_fk_signature(c) for c in t.constraints if isinstance(c, sa.ForeignKeyConstraint))` — table-level FK signatures match
- For each column with an inline FK: `(col.name, fk.target_fullname, fk.ondelete)` matches across both Tables

**Critical correctness note for composite FKs:** The v0.7.5 factories document composite FKs (`media_content.(agent_id, media_id) → media`, `memory_chunks.(agent_id, memory_id) → memories`) in COMMENTS ONLY — they do not actually emit `ForeignKeyConstraint(...)` in the factory body. So a naive "parity vs factory" check will trivially pass even if enrichment forgets composite FKs. The parity tests MUST include EXPLICIT per-table assertions for composite FKs by reading the prod schema as ground truth, not just by comparing to the factory:

```python
def test_parity_media_content_table():
    enriched = TableSchema(...)
    m = sa.MetaData()
    t = enriched.to_sqlalchemy_table(m)
    _assert_tables_equivalent(media_content_table(sa.MetaData()), t)
    # Additionally: assert composite FK that the factory doesn't emit:
    composite_fks = [
        c for c in t.constraints
        if isinstance(c, sa.ForeignKeyConstraint)
        and len(c.column_keys) > 1
    ]
    assert any(
        tuple(c.column_keys) == ("agent_id", "media_id") and
        c.elements[0].column.table.name == "media" and
        c.ondelete == "CASCADE"
        for c in composite_fks
    ), "composite FK (agent_id, media_id) → media missing"
```

Apply the same "factory parity + ground-truth augmentation" pattern to `memory_chunks` (composite FK to memories).
- ForeignKeyConstraint comparison: list of `(local_cols, ref_table, ref_cols, ondelete)` tuples

Required parity tests:

- `test_parity_memories_table`
- `test_parity_media_table`
- `test_parity_media_content_table`
- `test_parity_memory_chunks_table`
- `test_parity_conversation_memory_refs_table`
- `test_parity_context_items_table`

All six must produce structurally equivalent Tables.

---

## Unit tests (small isolated cases)

In addition to parity tests:

- `test_to_sqla_idempotent` — calling `schema.to_sqlalchemy_table(metadata)` twice on the same metadata returns the existing Table on second call (no duplicate-table error)
- `test_to_sqla_composite_pk` — schema with `primary_key=("a", "b")` produces Table where both columns are `primary_key=True`
- `test_to_sqla_single_pk` — schema with `primary_key="a"` produces Table where only `a` is PK
- `test_to_sqla_vector_dim_propagates` — schema with `Column("emb", VECTOR_TYPE, vector_dim=768)` produces SQLAlchemy `Vector(768)` (assuming pgvector available)
- `test_to_sqla_enum_creates_check_constraint` — schema with ENUM_TYPE column produces a SQLAlchemy `Enum` with `create_constraint=True`
- `test_to_sqla_single_fk_inline` — schema with `Column(..., foreign_key=("users", "user_id"))` produces a column with the FK inline
- `test_to_sqla_composite_fk_table_level` — schema with `foreign_keys=(ForeignKey(("a", "b"), "t", ("a", "b"), on_delete="CASCADE"),)` produces a `ForeignKeyConstraint` at table level with `ondelete="CASCADE"`
- `test_to_sqla_partial_index_postgresql_where` — schema with `Index(..., where="x IS NOT NULL")` produces `postgresql_where=sa.text("x IS NOT NULL")`
- `test_to_sqla_server_default_propagates` — schema with `Column(..., server_default="image")` produces a column with `server_default` set
- `test_to_sqla_pgvector_unavailable_raises_at_call` — when `pgvector` cannot be imported and a schema has a VECTOR_TYPE column, `to_sqlalchemy_table` raises `ImportError` at call time (mirrors the v0.7.5 factory behavior; do NOT short-circuit silently)
- `test_to_sqla_numeric_precision_scale` — schema with `Column("cost", NUMERIC_TYPE, precision=12, scale=8, nullable=True)` produces SQLAlchemy `Numeric(12, 8)`
- `test_to_sqla_tsvector_immutable` — schema with `Column("search_vector", TSVECTOR_TYPE, nullable=True, immutable=True)` produces a `TSVECTOR` column. The immutability is a Collection-side concern enforced by the existing UPDATE generator using the immutable flag; this test just confirms the SQLAlchemy type round-trips.

---

## Anti-patterns

- DO NOT silently skip a column if its type tag isn't in `_TYPE_MAPPING`. Raise `KeyError` with the column name + bad tag so unknown tags fail loudly.
- DO NOT add SQLAlchemy as an unconditional top-of-module import in `schema_backed.py`. Lazy-load inside the method or behind `TYPE_CHECKING` so non-Alembic / non-SQLAlchemy consumers don't pay the import cost.
- DO NOT duplicate the `_require_pgvector` helper from the agent-memory module. Refactor it to live in `schema_backed.py` so there is one canonical place.
- DO NOT model trigger DDL (TSVECTOR generation). Out of scope — alembic owns triggers.
- DO NOT add `naming_convention` to `MetaData` here. Explicit names per design.
- AVOID assertions on `__repr__` output in parity tests — SQLAlchemy's repr drifts across versions. Compare structural fields explicitly.

---

## Success Criteria

- [ ] `TableSchema.to_sqlalchemy_table(metadata: sa.MetaData) -> sa.Table` implemented
- [ ] `_TYPE_MAPPING` populated with all 11 type tags (UUID, STRING, DATETIMETZ, JSONB, BYTES, INT, BOOL, VECTOR, ENUM, NUMERIC, TSVECTOR)
- [ ] `_require_pgvector` helper refactored to a single location in `schema_backed.py`; v0.7.5 factory file imports from there
- [ ] All 6 parity tests pass (memories, media, media_content, memory_chunks, conversation_memory_refs, context_items) — using the tightened helper that reads `dialect_kwargs` not `dialect_options`
- [ ] media_content + memory_chunks parity tests include explicit composite-FK assertions (factory comments → real assertions; see "Critical correctness note for composite FKs" above)
- [ ] All listed unit tests pass
- [ ] Method is idempotent: second call on same metadata returns existing Table
- [ ] Full CI suite passes: `uv run pytest packages/ tests/ -m "not integration"`
- [ ] Ruff clean
- [ ] Mypy clean

---

## Verification

```bash
cd /Users/pace/crypt/pub/dev-wsl/vscode/3tears/3tears
uv run pytest packages/core/tests/unit/collections/test_to_sqlalchemy_table.py -v
uv run pytest packages/ tests/ -m "not integration" -q
uv run ruff check . && uv run ruff format . --check
uv run mypy --explicit-package-bases -p threetears.core -p threetears.agent.memory -p threetears.agent.tools
```

---

## Enforcement Test Suggestions

- [ ] Drift risk: new type tags added to `schema_backed.py` (e.g. a hypothetical `BIGINT_TYPE`) without a corresponding entry in `_TYPE_MAPPING`. Suggested test: enforce that every `*_TYPE = "..."` module-level string constant has a key in `_TYPE_MAPPING`. Useful guard. Flag for review.
