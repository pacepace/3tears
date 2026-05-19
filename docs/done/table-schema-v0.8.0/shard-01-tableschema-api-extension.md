# v0.8.0-task-01: Extend TableSchema with FK / Index / Enum / Vector / ServerDefault declarations

## Objective

Add the optional declarations that let `TableSchema` express the full SQLAlchemy semantics of the 3tears-owned tables (foreign keys, indexes, enum CHECK constraints, vector dimensions, server defaults). Pure data-model additions — no behaviour changes for existing schemas, no `to_sqlalchemy_table` implementation yet (that's shard 02). All new fields default to `None` / empty list so v0.7.x declarations continue to work unchanged.

---

## Locked design decisions (from README.md)

This shard MUST NOT re-litigate any of these. They are pre-resolved.

- Two-shape FK: `Column.foreign_key=("table", "col")` for single-column + `TableSchema.foreign_keys=[ForeignKey(...)]` for composite
- Enum: `Column(..., enum_type=("a", "b", ...), enum_name="...")`. Prod uses real `CREATE TYPE ... AS ENUM(...)` (verified via `pg_type`) — SAEnum default `native_enum=True` is correct, do NOT pass `native_enum=False`.
- Trigger-maintained columns: add `TSVECTOR_TYPE = "tsvector"` tag; declared with `nullable=True, immutable=True` so Collection UPDATE generators skip them
- `Index("name", "a", "b", unique=True, where="...")` — factory function returning frozen `IndexDef` dataclass
- `ForeignKey(local_cols, ref_table, ref_cols, on_delete=...)` — factory function returning frozen `ForeignKeyDef` dataclass
- `Column.vector_dim: int | None` per-column
- `Column.server_default: str | None` raw Postgres expression
- `NUMERIC_TYPE = "numeric"` tag + `Column.precision: int | None` + `Column.scale: int | None` per-column (needed for `media_content.cost = Numeric(12, 8)`)
- `on_delete` literal values: `"CASCADE" | "SET NULL" | "RESTRICT" | "NO ACTION"`
- No `on_update`
- Names on Index + FK are REQUIRED (no auto-generation)
- All new types live in `schema_backed.py`
- The existing partition-column coercion in `TableSchema.__post_init__` MUST be extended to pass through new Column fields (otherwise a `Column(..., partition=True, foreign_key=...)` silently drops the FK)

---

## Files to Modify

- `packages/core/src/threetears/core/collections/schema_backed.py` — add new dataclasses + factory functions + extend `Column` and `TableSchema`
- `packages/core/tests/unit/collections/test_schema_backed.py` (or wherever the existing `TableSchema` unit tests live; locate via `grep -rn "TableSchema(" packages/core/tests/` before creating new files)

## Files to NOT Modify

- Any existing `TableSchema(...)` declaration in `packages/*/src/threetears/*/collections.py`. Those changes belong to shard 03.
- Any `to_sqlalchemy_table` impl. That's shard 02.

---

## Prescriptive types

### New module-level type tags

Add three new type tags alongside the existing `UUID_TYPE` / `STRING_TYPE` / etc. constants:

```python
ENUM_TYPE = "enum"
NUMERIC_TYPE = "numeric"
TSVECTOR_TYPE = "tsvector"
```

The tags exist so the SQL generator + `to_sqlalchemy_table` can dispatch on them (shard 02 uses them).

- `ENUM_TYPE`: prod uses real `CREATE TYPE ... AS ENUM(...)` for `memory_type`, `conversation_status`, `message_role`, `user_role`, `visibility_mode`, `participation_role`, `tool_invocation_status`. Verified via `pg_type` query at design time (see README.md "Postgres enum vs CHECK" decision).
- `NUMERIC_TYPE`: required for `media_content.cost` which is `Numeric(12, 8)` in the v0.7.5 factory. No existing tag covers fixed-precision decimal.
- `TSVECTOR_TYPE`: required for the three `search_vector` columns (memories, media_content, memory_chunks). Trigger-maintained server-side; declared with `nullable=True, immutable=True` so Collection UPDATE generators exclude it from SET clauses.

### New `OnDelete` Literal

```python
from typing import Literal

OnDelete = Literal["CASCADE", "SET NULL", "RESTRICT", "NO ACTION"]
```

### New `ForeignKeyDef` dataclass

```python
@dataclass(frozen=True)
class ForeignKeyDef:
    local_cols: tuple[str, ...]
    ref_table: str
    ref_cols: tuple[str, ...]
    on_delete: OnDelete = "NO ACTION"
```

Validation in `__post_init__`:
- `local_cols` and `ref_cols` must be the same length (1-tuple = single FK; same-length tuples = composite)
- Both tuples non-empty
- `on_delete` is one of the allowed literals (Pydantic / `typing` doesn't enforce at runtime, so check explicitly)

### New `IndexDef` dataclass

```python
@dataclass(frozen=True)
class IndexDef:
    name: str
    columns: tuple[str, ...]
    unique: bool = False
    where: str | None = None
```

Validation:
- `name` non-empty
- `columns` non-empty

### `ForeignKey` factory function

Lives at module level, returns a `ForeignKeyDef`.

```python
def ForeignKey(
    local_cols: tuple[str, ...] | str,
    ref_table: str,
    ref_cols: tuple[str, ...] | str,
    *,
    on_delete: OnDelete = "NO ACTION",
) -> ForeignKeyDef:
    ...
```

Behavior:
- If `local_cols` or `ref_cols` is a bare `str` (not a tuple), coerce to a 1-tuple. This is a convenience for the common single-column case so callers can write `ForeignKey("user_id", "users", "user_id")` without typing `("user_id",)` twice.
- Returns `ForeignKeyDef(local_cols=..., ref_table=..., ref_cols=..., on_delete=...)`.

### `Index` factory function

Lives at module level, returns an `IndexDef`.

```python
def Index(
    name: str,
    *columns: str,
    unique: bool = False,
    where: str | None = None,
) -> IndexDef:
    ...
```

Varargs for columns reads cleanly at call sites:
```python
Index("ix_memories_user_date", "user_id", "date_created")
Index("ix_memories_user_alias", "agent_id", "user_id", "alias",
      unique=True, where="alias IS NOT NULL")
```

Returns `IndexDef(name=name, columns=tuple(columns), unique=unique, where=where)`.

### Extended `Column` dataclass

The existing fields stay unchanged. Add seven new optional fields:

```python
@dataclass(frozen=True)
class Column:
    # ---existing fields (DO NOT modify):---
    name: str
    column_type: str
    immutable: bool = False
    nullable: bool = False
    partition: bool = False
    # ---new in v0.8.0:---
    foreign_key: tuple[str, str] | None = None
    enum_type: tuple[str, ...] | None = None
    enum_name: str | None = None
    vector_dim: int | None = None
    server_default: str | None = None
    precision: int | None = None    # for NUMERIC_TYPE columns
    scale: int | None = None        # for NUMERIC_TYPE columns
```

Note: the existing `Column` is `frozen=True`. Adding fields to a frozen dataclass is non-breaking (additions go after existing fields with defaults).

### Extended `TableSchema` dataclass

Add two optional fields:

```python
@dataclass(frozen=True)
class TableSchema:
    # ---existing fields (DO NOT modify):---
    name: str
    primary_key: str | tuple[str, ...]
    columns: list[Column]
    cas_column: str | None = None
    on_conflict: OnConflict = "update"
    # ---new in v0.8.0:---
    foreign_keys: tuple[ForeignKeyDef, ...] = ()
    indexes: tuple[IndexDef, ...] = ()
```

Use `tuple` not `list` to keep the frozen-dataclass equality + hashability stable.

---

## Validation in `__post_init__`

The existing `TableSchema.__post_init__` already validates primary-key columns + partition columns. EXTEND it to also validate the new fields:

### `Column` `__post_init__`

Add validators for new field combinations:

1. **`foreign_key` × `column_type`**: foreign_key may be set on any column_type. No restriction.

2. **`enum_type` requires `column_type == ENUM_TYPE` and `enum_name`**:
   ```
   if self.enum_type is not None:
       if self.column_type != ENUM_TYPE:
           raise ValueError(
               f"Column(name={self.name!r}): enum_type only valid with column_type=ENUM_TYPE; "
               f"got column_type={self.column_type!r}"
           )
       if not self.enum_name:
           raise ValueError(
               f"Column(name={self.name!r}): enum_type requires enum_name to be set"
           )
       if len(self.enum_type) == 0:
           raise ValueError(
               f"Column(name={self.name!r}): enum_type must be a non-empty tuple"
           )
   ```

3. **`ENUM_TYPE` requires `enum_type`** (inverse of above):
   ```
   if self.column_type == ENUM_TYPE and self.enum_type is None:
       raise ValueError(
           f"Column(name={self.name!r}): column_type=ENUM_TYPE requires enum_type to be set"
       )
   ```

4. **`vector_dim` requires `column_type == VECTOR_TYPE`**:
   ```
   if self.vector_dim is not None and self.column_type != VECTOR_TYPE:
       raise ValueError(
           f"Column(name={self.name!r}): vector_dim only valid with column_type=VECTOR_TYPE; "
           f"got column_type={self.column_type!r}"
       )
   ```

5. **`VECTOR_TYPE` requires `vector_dim`** (inverse):
   ```
   if self.column_type == VECTOR_TYPE and self.vector_dim is None:
       raise ValueError(
           f"Column(name={self.name!r}): column_type=VECTOR_TYPE requires vector_dim to be set"
       )
   ```

6. **`foreign_key` tuple shape**: must be exactly `(table_name, column_name)` — both strings, both non-empty.

7. **`server_default`** has no cross-field constraints — any string accepted.

8. **`precision` / `scale` require `column_type == NUMERIC_TYPE`**:
   ```
   if (self.precision is not None or self.scale is not None) and self.column_type != NUMERIC_TYPE:
       raise ValueError(
           f"Column(name={self.name!r}): precision/scale only valid with column_type=NUMERIC_TYPE; "
           f"got column_type={self.column_type!r}"
       )
   ```

9. **`NUMERIC_TYPE` requires both `precision` and `scale`**:
   ```
   if self.column_type == NUMERIC_TYPE and (self.precision is None or self.scale is None):
       raise ValueError(
           f"Column(name={self.name!r}): column_type=NUMERIC_TYPE requires both precision and scale to be set"
       )
   ```

10. **`TSVECTOR_TYPE` columns should be declared `immutable=True`** (defensive warning to catch mis-declarations):
    ```
    if self.column_type == TSVECTOR_TYPE and not self.immutable:
        raise ValueError(
            f"Column(name={self.name!r}): column_type=TSVECTOR_TYPE columns are trigger-maintained "
            f"server-side and must be declared immutable=True to exclude them from Collection UPDATE generators"
        )
    ```

### Partition-column coercion — REQUIRED FIX

The existing `TableSchema.__post_init__` rebuilds partition columns to force `immutable=True` (current `schema_backed.py` lines 232–239 approximately):

```python
# Current code:
for col in self.columns:
    if col.partition:
        partition_cols.append(col.name)
        if not col.immutable:
            col = Column(  # noqa: PLW2901
                name=col.name,
                column_type=col.column_type,
                immutable=True,
                nullable=col.nullable,
                partition=True,
            )
    coerced_columns.append(col)
```

This call ONLY passes the v0.7.x fields. After this shard lands, any `Column("agent_id", UUID_TYPE, partition=True, foreign_key=("agents", "agent_id"))` would have the `foreign_key` field SILENTLY DROPPED during construction. Same problem for `enum_type`, `enum_name`, `vector_dim`, `server_default`, `precision`, `scale`.

EXTEND the coercion to pass through every new field:

```python
# Updated code:
for col in self.columns:
    if col.partition:
        partition_cols.append(col.name)
        if not col.immutable:
            col = Column(  # noqa: PLW2901
                name=col.name,
                column_type=col.column_type,
                immutable=True,
                nullable=col.nullable,
                partition=True,
                foreign_key=col.foreign_key,
                enum_type=col.enum_type,
                enum_name=col.enum_name,
                vector_dim=col.vector_dim,
                server_default=col.server_default,
                precision=col.precision,
                scale=col.scale,
            )
    coerced_columns.append(col)
```

Required test for this fix:

- `test_partition_column_preserves_v080_fields_through_coercion` — declare `Column("agent_id", UUID_TYPE, partition=True, foreign_key=("agents", "agent_id"), server_default="...", vector_dim=None)` (mix of fields) and assert the post-coercion column on the schema still has `foreign_key=("agents", "agent_id")` and the server_default. Without the fix, this test fails because the coercion drops the fields.

### `TableSchema` `__post_init__`

Existing validators stay. ADD the new ones below.

**Validator placement note:** The new validators reference `self._by_name`, which is constructed at approximately line 254 in current source via `by_name = {c.name: c for c in coerced_columns}` followed by `object.__setattr__(self, "_by_name", by_name)` at approximately line 269. The new validators MUST run AFTER `_by_name` is installed (after the partition-coercion + by-name construction) but BEFORE the function returns. Place the new validator block immediately before the final closing of `__post_init__` (after line 269 in current source).

1. **`foreign_keys[i].local_cols` must reference columns declared on this schema**:
   ```
   for fk in self.foreign_keys:
       for local_col in fk.local_cols:
           if local_col not in self._by_name:
               raise ValueError(
                   f"TableSchema(name={self.name!r}): foreign_key references "
                   f"local_col={local_col!r} which is not declared in columns"
               )
   ```

2. **`indexes[i].columns` must reference columns declared on this schema**:
   ```
   for idx in self.indexes:
       for col in idx.columns:
           if col not in self._by_name:
               raise ValueError(
                   f"TableSchema(name={self.name!r}): index name={idx.name!r} references "
                   f"col={col!r} which is not declared in columns"
               )
   ```

3. **Index names + FK names should be unique within this TableSchema** (to catch typos that produce two indexes with the same name):
   ```
   index_names = [i.name for i in self.indexes]
   if len(index_names) != len(set(index_names)):
       raise ValueError(
           f"TableSchema(name={self.name!r}): duplicate index names"
       )
   ```

---

## Public exports

Add to `__all__` in `schema_backed.py`:

- `ENUM_TYPE`
- `NUMERIC_TYPE`
- `TSVECTOR_TYPE`
- `OnDelete`
- `ForeignKeyDef`
- `IndexDef`
- `ForeignKey`
- `Index`

DO NOT remove anything from existing `__all__`.

---

## Unit tests

Create or extend the existing `TableSchema` unit tests. Required test cases:

### `Column` validators

- `test_column_enum_type_requires_enum_column_type` — `Column("x", STRING_TYPE, enum_type=("a", "b"), enum_name="foo")` raises `ValueError`
- `test_column_enum_column_type_requires_enum_values` — `Column("x", ENUM_TYPE)` raises `ValueError`
- `test_column_enum_column_type_requires_enum_name` — `Column("x", ENUM_TYPE, enum_type=("a",))` raises `ValueError`
- `test_column_enum_type_empty_tuple_rejected` — `Column("x", ENUM_TYPE, enum_type=(), enum_name="foo")` raises `ValueError`
- `test_column_vector_dim_requires_vector_column_type` — `Column("x", STRING_TYPE, vector_dim=1024)` raises `ValueError`
- `test_column_vector_column_type_requires_dim` — `Column("x", VECTOR_TYPE)` raises `ValueError`
- `test_column_numeric_requires_precision_and_scale` — `Column("x", NUMERIC_TYPE)`, `Column("x", NUMERIC_TYPE, precision=12)`, `Column("x", NUMERIC_TYPE, scale=8)` each raise `ValueError`
- `test_column_precision_scale_require_numeric_type` — `Column("x", INT_TYPE, precision=12, scale=8)` raises `ValueError`
- `test_column_tsvector_requires_immutable` — `Column("x", TSVECTOR_TYPE, nullable=True)` raises `ValueError` (immutable=True is required)
- `test_column_tsvector_with_immutable_constructs` — `Column("x", TSVECTOR_TYPE, nullable=True, immutable=True)` constructs cleanly
- `test_column_foreign_key_passes_through` — `Column("user_id", UUID_TYPE, foreign_key=("users", "user_id"))` constructs cleanly + retains the field
- `test_column_server_default_passes_through` — `Column("x", STRING_TYPE, server_default="image")` constructs cleanly

### `ForeignKey` factory

- `test_foreign_key_single_column_via_strings` — `ForeignKey("user_id", "users", "user_id")` returns FK with `local_cols=("user_id",)` and `ref_cols=("user_id",)`
- `test_foreign_key_composite_via_tuples` — `ForeignKey(("agent_id", "memory_id"), "memories", ("agent_id", "memory_id"), on_delete="CASCADE")` returns FK with both 2-tuples + correct `on_delete`
- `test_foreign_key_mismatched_lengths_raises` — `ForeignKey(("a",), "t", ("a", "b"))` raises `ValueError`
- `test_foreign_key_invalid_on_delete_raises` — `ForeignKey("user_id", "users", "user_id", on_delete="BOGUS")` raises `ValueError`

### `Index` factory

- `test_index_basic` — `Index("ix_x", "a", "b")` returns `IndexDef(name="ix_x", columns=("a", "b"), unique=False, where=None)`
- `test_index_unique_partial` — `Index("ix_x", "a", unique=True, where="a IS NOT NULL")` returns `IndexDef` with the WHERE clause + unique flag
- `test_index_no_columns_raises` — `Index("ix_x")` raises `ValueError`

### `TableSchema` validators

- `test_table_schema_foreign_keys_must_reference_declared_columns` — adding an FK whose `local_cols` aren't in `columns` raises `ValueError`
- `test_table_schema_indexes_must_reference_declared_columns` — adding an Index whose `columns` aren't in `columns` raises `ValueError`
- `test_table_schema_duplicate_index_names_rejected`
- `test_table_schema_default_values_for_v080_fields` — constructing `TableSchema(...)` without `foreign_keys` or `indexes` gives `foreign_keys=()`, `indexes=()`

### Backward compatibility

- `test_existing_table_schema_declaration_works_unchanged` — pick one existing TableSchema from the codebase (e.g. `MemoriesCollection.schema`) and assert that it still constructs without errors after the API extension. This guards against accidentally making any of the new fields non-optional.

---

## Anti-patterns

- DO NOT modify existing `TableSchema(...)` declarations in `packages/*/src/threetears/*/collections.py`. Existing declarations must continue to work unchanged. Shard 03 handles enrichment.
- DO NOT implement `to_sqlalchemy_table` here. That's shard 02. This shard is data-model only.
- DO NOT add `on_update` to `ForeignKeyDef`. Not in scope for v0.8.0.
- DO NOT auto-generate index or FK names. They must be explicit per locked design.
- DO NOT add CHECK constraint support beyond enums. Not in scope.
- DO NOT split `schema_backed.py` into multiple modules. All new types stay in this file.
- AVOID changing the order of existing `Column` / `TableSchema` fields. Adding new fields after existing ones with defaults is non-breaking; reordering is breaking.

---

## Success Criteria

- [ ] `ENUM_TYPE`, `NUMERIC_TYPE`, `TSVECTOR_TYPE`, `OnDelete`, `ForeignKeyDef`, `IndexDef`, `ForeignKey`, `Index` defined in `schema_backed.py` and exported
- [ ] `Column` gains seven optional fields: `foreign_key`, `enum_type`, `enum_name`, `vector_dim`, `server_default`, `precision`, `scale` — all default to `None`
- [ ] `TableSchema` gains two optional fields: `foreign_keys`, `indexes` — both default to empty tuple
- [ ] `Column.__post_init__` validates the ten cross-field constraints listed above (including precision/scale, TSVECTOR immutable requirement)
- [ ] `TableSchema.__post_init__` extends partition-column coercion to pass through all v0.8.0 fields (no silent field drop)
- [ ] `TableSchema.__post_init__` validates that FK / Index column references resolve to declared columns (placed after `_by_name` is constructed)
- [ ] All existing `TableSchema(...)` declarations across the codebase continue to construct without modification
- [ ] All listed unit tests pass (including the partition-coercion regression test)
- [ ] Full CI suite passes: `uv run pytest packages/ tests/ -m "not integration"`
- [ ] Ruff clean: `uv run ruff check .` + `uv run ruff format . --check`
- [ ] Mypy clean: `uv run mypy --explicit-package-bases -p threetears.core -p threetears.agent.memory -p threetears.agent.tools`

---

## Verification

```bash
cd /Users/pace/crypt/pub/dev-wsl/vscode/3tears/3tears
uv run pytest packages/core/tests/unit/collections/ -v -q
uv run pytest packages/ tests/ -m "not integration" -q
uv run ruff check . && uv run ruff format . --check
uv run mypy --explicit-package-bases -p threetears.core -p threetears.agent.memory -p threetears.agent.tools
```

All four commands MUST exit 0.

---

## Enforcement Test Suggestions

Consider whether enforcement tests would catch future drift:

- [ ] Drift risk: a future developer adds a new `column_type` tag (e.g. `BIGINT_TYPE`) and forgets to extend the `Column.__post_init__` cross-field validators. Suggested test: enforce that every value of `column_type` either has no special cross-field requirements OR is explicitly listed in a `_TYPE_VALIDATOR_REGISTRY` (visible in code). Probably overkill — flag as an option, do not implement without approval.
