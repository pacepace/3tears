# collections-task-01: Schema-Backed BaseCollection

**Status:** Ready for implementation
**Scope:** `3tears/packages/core/src/threetears/core/collections/`, every subclass across agent-tools / agent-workspace / agent-memory. `(3tears)` label.

---

## Objective

Every `BaseCollection` subclass reimplements roughly 80% of the same boilerplate: `_fetch_from_postgres(id)` (SELECT by PK), `_save_to_postgres(data, ts?)` (upsert), `_delete_from_postgres(id)` (DELETE), `_serialize`/`_deserialize` (JSON round-trip). The domain-specific part is the column list. The rest is duplicated structure that drifts between files — this session's JSONB-decode bug touched `ContextItemCollection` specifically because it hand-rolled the read path; the other collections have the same pattern and are one PR away from needing the same fix.

Collapse the duplication: introduce `SchemaBackedCollection` that takes a table name + a column list as a config and implements the four `_*_from_postgres` methods mechanically. Domain-specific collections become declarations, not implementations.

---

## Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| COLL-01 | `SchemaBackedCollection[EntityT]` subclass of `BaseCollection[EntityT]` taking a schema descriptor (table name + columns + primary key column + JSONB columns list). Implements `_fetch_from_postgres`, `_save_to_postgres`, `_delete_from_postgres`, `_serialize`, `_deserialize` generically. | P0 |
| COLL-02 | Every existing collection (`WorkspaceCollection`, `WorkspaceFileCollection`, `WorkspaceFileVersionCollection`, `ContextItemCollection`, `ConversationCollection`, `MemoryCollection`) converts from hand-written to schema-declared. Domain extras (e.g. `find_by_agent_and_name`) stay hand-written; the shared CRUD does not. | P0 |
| COLL-03 | JSONB columns handled uniformly: on read, JSON-decode from string to dict; on write, JSON-encode from dict to string with `::jsonb` cast. This removes the class of bug that hit `ContextItemCollection` this session. | P0 |
| COLL-04 | UUID / datetime / bytes coercions happen in one place (`_normalize_value(column_spec, value)`) so every collection handles the NATS-proxy wire types identically. | P0 |
| COLL-05 | `_save_to_postgres` returns the affected row count parsed from the asyncpg tag — no more hand-rolled `int(result.split()[-1])` per file. | P0 |
| COLL-06 | Schema descriptor can be generated from a pydantic model or SQLAlchemy table OR hand-written; the declaration is lightweight either way. | P1 |
| COLL-07 | Full test parity: every existing collection test passes against the new schema-backed implementation without modification to the assertions (only the collection's construction call changes). | P0 |

---

## Design Context

Current shape (excerpt from `WorkspaceCollection`):

```python
async def _save_to_postgres(self, data, original_timestamp=None):
    result = await self._postgres_pool.execute(
        """INSERT INTO workspaces (id, agent_id, name, ...)
           VALUES ($1, $2, $3, ...)
           ON CONFLICT (id) DO UPDATE SET ...""",
        data["id"], data["agent_id"], data["name"], ...,
    )
    return int(result.split()[-1])
```

One of those per collection. Each one has its own typo potential, its own missing-column bug potential, its own ON-CONFLICT list to keep aligned with the INSERT list. The new shape:

```python
class WorkspaceCollection(SchemaBackedCollection[WorkspaceEntity]):
    schema = TableSchema(
        name="workspaces",
        primary_key="id",
        columns=[
            Column("id", UUID_TYPE),
            Column("agent_id", UUID_TYPE),
            Column("customer_id", UUID_TYPE),  # post rbac
            Column("name", STRING_TYPE),
            ...
            Column("metadata", JSONB_TYPE),
        ],
    )
    entity_class = WorkspaceEntity
```

No `_save_to_postgres`, no `_fetch_from_postgres`, no `_delete_from_postgres` — the base class handles them from the schema. Domain-specific query methods (`find_by_agent_and_name`, `find_by_workspace`) stay as they are — those aren't the CRUD, they're the domain surface.

### What `SchemaBackedCollection._save_to_postgres` has to build

An upsert. From the schema:
- `INSERT INTO {table} ({columns}) VALUES ({$1..$N}) ON CONFLICT ({pk}) DO UPDATE SET {non_pk_columns} = EXCLUDED.{same}`
- Parameters pass through `_serialize_value(column_spec, data[column.name])` so UUID/datetime/JSON handling is uniform.
- Returns `int(tag.split()[-1])` with the asyncpg-tag-shape we already pin on the proxy.

### What stays bespoke

- Domain-specific SELECT methods (`find_by_agent`, `find_by_workspace`, etc.) — those are the actual collection surface callers use.
- Composite queries (joins, WHERE EXISTS, etc.) — not CRUD boilerplate; they stay hand-written.
- Cache invalidation heuristics, if any — per-domain.

---

## Files to Create / Modify

### Create

- `3tears/packages/core/src/threetears/core/collections/schema_backed.py` — `SchemaBackedCollection`, `TableSchema`, `Column`, type registry.
- `3tears/packages/core/tests/unit/collections/test_schema_backed.py` — CRUD round-trip tests against a fake pool, JSONB handling, UUID coercion, datetime timezone preservation.

### Modify

- `3tears/packages/agent-workspace/src/threetears/agent/workspace/collections.py` — three collections convert to schema-backed.
- `3tears/packages/agent-tools/src/threetears/agent/tools/collections.py` — `ContextItemCollection` converts; domain methods stay.
- `3tears/packages/agent-memory/src/threetears/agent/memory/collections.py` — `MemoryCollection` + `ConversationCollection` convert.

### Retire

- Hand-rolled `_save_to_postgres` / `_fetch_from_postgres` / `_delete_from_postgres` methods across the three converted collection modules.

---

## Implementation Notes

1. Build `SchemaBackedCollection` with `TableSchema` + `Column` types. Test against a fake pool that records executed SQL. Validate the generated upsert SQL matches the hand-rolled version byte-for-byte.
2. Convert one collection (suggest `ContextItemCollection` — smallest, already-broken — or `MemoryCollection` — simplest schema). Verify its existing test suite passes.
3. Convert remaining collections one at a time with test re-run between each.
4. The `TableSchema` declaration can be written to be compatible with SQLAlchemy's `Table` if we want to share with the migration runner (migrations-task-01). Optional shared source of truth; not required by this task.

---

## Anti-patterns

- **DO NOT** try to generalize the domain-specific query methods (`find_by_agent_and_name` etc.) into the generic base. That's the part that's legitimately per-collection; collapsing it would push too much domain logic into the base class. CRUD is generalizable; queries aren't.
- **DO NOT** put SQL in the entity class. The collection owns SQL; entities are value objects.
- **DO NOT** silently skip columns that aren't in the data dict on `_save_to_postgres`. An upsert with a missing column is usually a bug; fail loudly with a clear error message.
- **DO NOT** auto-generate column lists from entity classes by introspection. Explicit schema declaration is the readable contract; introspection makes refactors invisible to reviewers.

---

## Success Criteria

- [ ] `SchemaBackedCollection` exists with unit tests covering CRUD round-trip, JSONB handling, UUID/datetime coercion, error cases.
- [ ] Six existing collections converted; their existing test suites pass without modification to assertions.
- [ ] CRUD code duplication across the three modules drops to near zero (measured: lines-of-code in the converted files).
- [ ] The JSONB-decode bug that hit `ContextItemCollection` this session is structurally prevented for every collection.
- [ ] mypy + ruff clean.

---

## Verification

```bash
uv run --directory 3tears/packages/core pytest tests/unit/collections/ -v
uv run --directory 3tears/packages/agent-tools pytest tests/ -v
uv run --directory 3tears/packages/agent-workspace pytest tests/ -v
uv run --directory 3tears/packages/agent-memory pytest tests/ -v
```
