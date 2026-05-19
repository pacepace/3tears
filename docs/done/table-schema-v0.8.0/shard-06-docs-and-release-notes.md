# v0.8.0-task-06: Migration guide + 3tears v0.8.0 release notes

## Objective

Write the hand-curated GitHub Release notes for 3tears v0.8.0 and the migration guide for downstream 3tears consumers (anyone using `TableSchema` declarations in their own code). The release notes are the artifact deployers and code-readers see; the migration guide is what downstream consumers read before bumping their pin.

This shard depends on shards 01-05 landing. It can run in parallel with shard 05's metallm release work.

---

## Locked design decisions

No new design. This shard is documentation only.

---

## Files to Create / Modify

In the 3tears repo:

- `docs/table-schema/v0.8.0/migration-guide.md` — NEW. Step-by-step for downstream consumers bumping from v0.7.x to v0.8.0.
- 3tears v0.8.0 GitHub Release notes — written into the GitHub Release after tagging, via `gh release edit v0.8.0 --notes-file <path>`. Stage the markdown in `/tmp/3tears-v080-release-notes-tsc<timestamp>.md` per the project convention for ephemeral release-note files.

---

## Migration guide content

The migration guide answers: "I have a downstream package that declares TableSchemas and uses 3tears Collections. What changes when I bump to v0.8.0?"

Required sections:

### 1. Summary (1-2 paragraphs)

What changed: TableSchema gains FK / Index / Enum / Vector / ServerDefault declarations + `to_sqlalchemy_table(metadata)` method. Hand-written SQLAlchemy `Table` factory functions can collapse into one-line delegations. Single source of truth.

What didn't change: existing `TableSchema(...)` declarations keep working. Collection methods unchanged. No breaking API.

### 2. Why this release

Brief story of the v0.14.6 metallm incident (memory_add broken on prod because of L1 cache divergence with the metallm side's duplicate ORM declarations). v0.7.5 closed the cross-repo gap with hand-written factories; v0.8.0 closes the within-3tears gap with `to_sqlalchemy_table`.

### 3. New API surface

Quick reference. Code snippets from the locked design decisions:

```python
from threetears.core.collections.schema_backed import (
    Column,
    ForeignKey,
    Index,
    TableSchema,
    UUID_TYPE, STRING_TYPE, ENUM_TYPE, VECTOR_TYPE, TSVECTOR_TYPE,
    NUMERIC_TYPE,  # added in v0.8.0
    # ... other type tags
)

schema = TableSchema(
    name="my_table",
    primary_key=("agent_id", "id"),
    columns=[
        Column("id", UUID_TYPE),
        Column("agent_id", UUID_TYPE, partition=True),
        # Single-column FK shorthand:
        Column("user_id", UUID_TYPE,
               foreign_key=("users", "user_id")),
        # Real Postgres enum with CHECK constraint:
        Column("status", ENUM_TYPE,
               enum_type=("draft", "active", "archived"),
               enum_name="my_status"),
        # Vector with explicit dim:
        Column("embedding", VECTOR_TYPE, vector_dim=1024,
               nullable=True),
        # Fixed-precision decimal:
        Column("cost", NUMERIC_TYPE,
               precision=12, scale=8, nullable=True),
        # Server default:
        Column("metadata_json", JSONB_TYPE,
               nullable=False, server_default="{}"),
        # Trigger-maintained tsvector (immutable required so
        # Collection UPDATE generators skip it):
        Column("search_vector", TSVECTOR_TYPE,
               nullable=True, immutable=True),
        Column("date_created", DATETIMETZ_TYPE, immutable=True),
    ],
    # Composite FK at table level:
    foreign_keys=(
        ForeignKey(("agent_id", "other_id"),
                   "other_table", ("agent_id", "id"),
                   on_delete="CASCADE"),
    ),
    indexes=(
        Index("ix_my_table_user", "user_id"),
        Index("ix_my_table_search_vector", "search_vector",
              where="search_vector IS NOT NULL"),
    ),
)
```

### 4. Step-by-step: enriching your existing TableSchemas

Numbered checklist matching the worked example in shard 03:

1. Open the TableSchema declaration
2. For each column with an FK in your migrations, add `foreign_key=("ref_table", "ref_col")` on the Column declaration
3. For each enum-style column, change `STRING_TYPE` to `ENUM_TYPE` + add `enum_type=(...)` + `enum_name="..."`
4. For each vector column, add `vector_dim=<int>` (matching the embedding provider's dim)
5. For each `search_vector` column (if any), declare it as `Column(col, TSVECTOR_TYPE, nullable=True, immutable=True)` — the trigger that populates it stays in your alembic migrations unchanged
6. For each column with a `server_default` in your migrations, add `server_default="<expression>"`
7. For each composite FK in your migrations, add a `ForeignKey(...)` to `TableSchema.foreign_keys`
8. For each index in your migrations, add an `Index(...)` to `TableSchema.indexes`
9. If your host application uses Alembic auto-generate against your TableSchemas, run `alembic revision --autogenerate -m "v0.8.0 parity check"` and verify the output is empty. Fix any divergences before shipping.

### 5. Collapsing your hand-written `Table` factories (optional)

If your downstream package has its own `<table>_table(metadata)` factory function (matching the v0.7.5 pattern), you can collapse the body to a one-liner after enriching the TableSchema:

```python
# Before:
def my_table(metadata: MetaData) -> Table:
    if "my_table" in metadata.tables:
        return metadata.tables["my_table"]
    return Table("my_table", metadata, ...)  # 20+ lines

# After:
def my_table(metadata: MetaData) -> Table:
    return MyCollection.schema.to_sqlalchemy_table(metadata)
```

### 6. What breaks (and what doesn't)

- **Nothing breaks** by default. All new TableSchema fields are optional with sensible defaults.
- **Behaviour change**: TSVECTOR columns declared with `immutable=True` are excluded from Collection UPDATE generators (their SET clause). If your code relies on TSVECTOR being in UPDATE statements (unusual — typically the trigger handles it), audit before shipping.

### 7. Reference: full design rationale

Point to `docs/table-schema/v0.8.0/README.md` for the design context + decisions table.

---

## Release notes content

Follow the pattern established by recent 3tears releases (v0.7.3, v0.7.4, v0.7.5 — read those for tone and structure). Required sections:

- **Headline**: one sentence, e.g., "Enriched TableSchema — one declaration, full SQLAlchemy semantics"
- **Why this release exists**: v0.7.5 closed the cross-repo trap with hand-written factories; v0.8.0 closes the within-3tears trap by enriching TableSchema itself. Single source of truth.
- **What changes** (the meat):
  - New API surface (FK / Index / Enum / Vector / ServerDefault / TSVECTOR)
  - `to_sqlalchemy_table(metadata)` method on TableSchema
  - All 3tears-owned schemas enriched (table or list summarizing changes per package)
  - Hand-written factories collapsed to one-line delegations
- **Compatibility**: no breaking changes for existing downstream consumers; new fields are additive
- **Verification**: test counts (post-shard-04 baseline), ruff/mypy clean, parity tests cover the 6 factory-replacement cases
- **Companion**: metallm v0.15.0 (link to release once published)
- **What's next**: optional follow-up items (CI-side autogen parity check as enforcement, etc.)
- **The lessons, archived**: 2-3 bullet points capturing what we learned from the v0.14.6 incident and the subsequent v0.7.5 + v0.8.0 work. Match the tone of the v0.14.6/v0.14.7 release notes.

---

## Anti-patterns

- DO NOT use auto-generated GitHub release stubs (`## What's Changed` + PR list). Pace's standing rule: real release notes with substantive content. Auto-stubs are not acceptable.
- DO NOT publish the release notes before the tag is pushed and `release.yml` (or the equivalent) runs green.
- DO NOT include code that "should work in theory" without verifying — every code snippet in the migration guide should be a slight variant of something that actually exists in the codebase (memories TableSchema, media TableSchema, etc.).
- AVOID overpromising "single source of truth" without the qualifier — note that triggers still live in alembic. Be honest about scope.

---

## Success Criteria

- [ ] `docs/table-schema/v0.8.0/migration-guide.md` created with all 7 sections listed above
- [ ] Release notes drafted in `/tmp/3tears-v080-release-notes-tsc<timestamp>.md`
- [ ] After tagging 3tears v0.8.0: `gh release edit v0.8.0 --notes-file <path>` publishes the substantive notes
- [ ] Release notes follow the established tone + structure from v0.7.3 / v0.7.4 / v0.7.5
- [ ] Migration guide includes a working code snippet for at least one enrichment scenario (single FK, composite FK, enum, vector, tsvector, server default, index)
- [ ] Once metallm v0.15.0 ships (shard 05), update the release notes' companion section to link the actual metallm release URL

---

## Verification

```bash
# Manual review:
less docs/table-schema/v0.8.0/migration-guide.md      # all 7 sections present
gh release view v0.8.0                                  # body matches the drafted markdown
```

---

## Enforcement Test Suggestions

None. This is documentation; the artifact IS the deliverable.
