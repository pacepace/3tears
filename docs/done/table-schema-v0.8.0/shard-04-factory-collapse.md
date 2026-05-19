# v0.8.0-task-04: Collapse hand-written SQLAlchemy factories into one-line calls to `to_sqlalchemy_table`

## Objective

Replace the hand-written SQLAlchemy `Table(...)` declarations inside the six existing v0.7.x factory functions with one-line calls to `<schema>.to_sqlalchemy_table(metadata)`. Each factory body shrinks from ~30 lines of SQLAlchemy column declarations to ~3 lines (idempotency check + factory call). Schema declarations after shard 03 become the single source of truth.

This shard depends on shards 01, 02, AND 03. Do not start until all three have landed.

---

## Locked design decisions (from README.md)

The factory functions stay as-is in their public API. Their *bodies* change.

---

## Files to Modify

- `packages/agent/memory/src/threetears/agent/memory/collections.py`:
  - `memories_table(metadata)`
  - `media_table(metadata)`
  - `media_content_table(metadata)`
  - `memory_chunks_table(metadata)`
  - `conversation_memory_refs_table(metadata)`
- `packages/agent/tools/src/threetears/agent/tools/collections.py`:
  - `context_items_table(metadata)`

Any other 3tears factory functions of the same shape (search via `grep -rn "def.*_table(metadata: MetaData)" packages/`) — collapse them too.

## Files to NOT Modify

- The corresponding `TableSchema` declarations. Those were enriched in shard 03 and should not change here.
- Any host application that calls the factories (e.g. metallm `api/src/data/models.py`). Public factory signatures stay the same.

---

## Worked example

### Before (v0.7.5 hand-written):

```python
def memories_table(metadata: MetaData) -> Table:
    """Register the ``memories`` table on the given SA metadata (v0.7.5)."""
    if "memories" in metadata.tables:
        return metadata.tables["memories"]
    vector_cls = _require_pgvector()
    return Table(
        "memories",
        metadata,
        SAColumn("memory_id", PgUUID(as_uuid=True), primary_key=True, nullable=False),
        SAColumn("agent_id", PgUUID(as_uuid=True), primary_key=True, nullable=False),
        SAColumn("customer_id", PgUUID(as_uuid=True), nullable=False),
        SAColumn(
            "user_id",
            PgUUID(as_uuid=True),
            ForeignKey("users.user_id"),
            nullable=False,
        ),
        # ... 10 more SAColumn declarations ...
        Index("ix_memories_user_date", "user_id", "date_created"),
    )
```

### After (v0.8.0 collapsed):

```python
def memories_table(metadata: MetaData) -> Table:
    """Register the ``memories`` table on the given SA metadata.

    v0.8.0: schema declaration is now the single source of truth. This
    factory is a thin idempotency wrapper around
    ``MemoriesCollection.schema.to_sqlalchemy_table(metadata)``.
    """
    return MemoriesCollection.schema.to_sqlalchemy_table(metadata)
```

(The `to_sqlalchemy_table` method handles the `if name in metadata.tables: return` short-circuit internally per shard 02 — the factory doesn't need to repeat it.)

---

## Per-factory collapse

Apply the same transformation to all six factories. Each one becomes:

```python
def <table>_table(metadata: MetaData) -> Table:
    """[Same docstring as v0.7.5, updated to note this is now a thin wrapper]"""
    return <Collection>.schema.to_sqlalchemy_table(metadata)
```

The factory's *purpose* (one-call registration on metadata) stays the same. The *implementation* delegates to the schema.

---

## Module-level cleanup

After all factories collapse, the following imports may become unused in `packages/agent/memory/src/threetears/agent/memory/collections.py`:

- `sqlalchemy.Column as SAColumn` — DELETE if unused
- `sqlalchemy.{ForeignKey, Index, Integer, Numeric, Text}` — DELETE the specific ones not used by anything else in the file (some may still be needed)
- `sqlalchemy.dialects.postgresql.{JSONB, TSVECTOR}` — DELETE if unused
- `sqlalchemy.Enum as SAEnum` — DELETE if unused
- `_require_pgvector` — if shard 02 moved this helper into `schema_backed.py`, DELETE the local copy here and update remaining callers (if any) to import from `schema_backed.py`

Use `ruff check` to find unused imports after collapse. The check is enforcement; act on it.

Same cleanup pass applies to `packages/agent/tools/src/threetears/agent/tools/collections.py`.

---

## Testing — REQUIRED parity-test rewrite (NOT optional)

Pre-shard-04 state: the parity tests in `test_to_sqlalchemy_table.py` build a Table two ways — once via the v0.7.5 factory (hand-written SQLAlchemy declarations), once via `to_sqlalchemy_table` against the enriched TableSchema — and assert structural equivalence. They catch any drift between the framework's auto-generation and the factory's ground truth.

Post-shard-04 state IF the tests aren't rewritten: both sides of the comparison call `to_sqlalchemy_table` (because the factories now delegate to it). The tests compare `to_sqlalchemy_table` output against itself — they trivially pass and provide ZERO regression protection. The framework could silently break and the tests wouldn't notice.

**REQUIRED in this shard:** before collapsing each factory, capture its current hand-written SQLAlchemy declarations into a **reference fixture function** in the test module. Then update the parity test to compare against the reference fixture, not against the factory.

### Concrete steps per factory

For each of the six factories (`memories_table`, `media_table`, `media_content_table`, `memory_chunks_table`, `conversation_memory_refs_table`, `context_items_table`):

1. **Before collapsing the factory body**, copy the entire v0.7.5 hand-written body into a new function in the parity test file:

   ```python
   # In packages/core/tests/unit/collections/test_to_sqlalchemy_table.py:

   def _reference_memories_table(metadata: sa.MetaData) -> sa.Table:
       """Hand-written reference Table for parity testing.

       Frozen v0.7.5 shape. DO NOT modify this function when prod
       schema changes -- this is the regression-protection fixture
       for ``to_sqlalchemy_table`` behavior, not a tracking schema.
       If prod schema legitimately changes, add a NEW reference
       function (e.g. ``_reference_memories_table_v090``) and run
       parity against the new one in addition to the old.
       """
       if "memories" in metadata.tables:
           return metadata.tables["memories"]
       vector_cls = _require_pgvector()  # import from schema_backed.py
       return sa.Table(
           "memories",
           metadata,
           # ... full v0.7.5 hand-written declaration ...
       )
   ```

   Note that the reference also needs to include the v0.7.5 caveats that the original factory had (e.g., `ix_memories_user_alias` documented as "metallm-deploy-specific" — but the parity test should EXPLICITLY assert the index IS present, because shard 03 declared it in the TableSchema; if the reference function omits it, the parity test fails until the reference is updated to match the v0.8.0 ground truth).

2. **Update the parity test** to compare against `_reference_memories_table` instead of `memories_table`:

   ```python
   def test_parity_memories_table():
       # ...enriched_schema construction (unchanged)...
       m1 = sa.MetaData()
       via_reference = _reference_memories_table(m1)  # CHANGED
       m2 = sa.MetaData()
       via_to_sqla = enriched_schema.to_sqlalchemy_table(m2)
       _assert_tables_equivalent(via_reference, via_to_sqla)
   ```

3. **Then** collapse the factory body.

### Why hand-written reference instead of factory output

Two reasons:

1. **Regression protection.** The reference function is FROZEN at v0.7.5 shape. If a future change to `to_sqlalchemy_table` breaks the parity test, that's catching real drift in the framework — the kind v0.14.6 had between two different declarations.
2. **Bug coexistence.** The v0.7.5 factories had bugs (missing composite FKs documented in comments only, etc.). The reference should encode the v0.7.5 KNOWN SHAPE plus any explicit augmentations from shard 02's parity-test design (composite FKs added explicitly in test assertions). If the reference matches what the factory ACTUALLY emitted, parity catches if `to_sqlalchemy_table` regresses to that buggy shape; if the reference matches the AUGMENTED v0.8.0 shape, parity catches if `to_sqlalchemy_table` regresses below that. Per shard 02 design, use the AUGMENTED shape.

### Whole-suite verification

```bash
cd /Users/pace/crypt/pub/dev-wsl/vscode/3tears/3tears
uv run pytest packages/ tests/ -m "not integration" -q
```

After this shard, the parity tests should continue to fail-meaningfully if `to_sqlalchemy_table` regresses. Confirm by temporarily breaking `to_sqlalchemy_table` in a small way (e.g., return without the index) and observing the parity test fail. Revert the temporary break before committing.

---

## Anti-patterns

- DO NOT change a factory's public signature. Host applications call `memories_table(metadata)` etc.; the call site stays unchanged.
- DO NOT delete the factory functions entirely. They remain the canonical registration entry point for host applications. Inlining the `.to_sqlalchemy_table(metadata)` call at every call site is worse — the factory name is the discoverability hook.
- DO NOT leave dead imports in the module. Run `ruff check` to catch them.
- DO NOT modify the corresponding TableSchema declarations. Schemas were finalized in shard 03; this shard is implementation collapse.
- AVOID leaving the v0.7.5 hand-written body as a comment in the file. Git history has it; the code shouldn't.

---

## Success Criteria

- [ ] All six factory functions reduced to a single delegating call to `<Collection>.schema.to_sqlalchemy_table(metadata)`
- [ ] Unused imports removed from both `collections.py` files (`ruff check` clean)
- [ ] **Six hand-written `_reference_<table>_table(metadata)` fixture functions** added to `test_to_sqlalchemy_table.py`, capturing the v0.8.0 canonical shape (v0.7.5 + composite FKs + ix_memories_user_alias per shard 03)
- [ ] All 6 parity tests updated to compare against the hand-written reference, NOT against the factory output
- [ ] Parity-test regression confirmed by temporarily breaking `to_sqlalchemy_table` and observing the test fail; break reverted before commit
- [ ] Full CI suite passes
- [ ] Mypy clean
- [ ] Each factory's docstring updated to note it's now a thin wrapper, with reference to the source-of-truth schema

---

## Verification

```bash
cd /Users/pace/crypt/pub/dev-wsl/vscode/3tears/3tears
uv run pytest packages/ tests/ -m "not integration" -q
uv run ruff check . && uv run ruff format . --check
uv run mypy --explicit-package-bases -p threetears.core -p threetears.agent.memory -p threetears.agent.tools
```

---

## Enforcement Test Suggestions

- [ ] Drift risk: a future developer adds a new factory function with hand-written SAColumn declarations instead of delegating to `to_sqlalchemy_table`. Suggested test: enforce that every `def *_table(metadata: MetaData) -> Table:` body contains a call to `.to_sqlalchemy_table(`. Catches the anti-pattern of falling back to hand-writing. Useful guard. Flag for review.
