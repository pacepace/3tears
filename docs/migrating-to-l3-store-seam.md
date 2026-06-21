# Migrating to the neutral L3 store seam (`collections-task-06`)

**What changed:** the collection framework's L3 (durable) tier seam stopped pretending the
store is Postgres. The four override points were renamed to be storage-agnostic, so a non-SQL
backend (a git working tree) can be an L3. **Behavior is unchanged** — this is a rename plus
additive new capability.

This branch (`feature/scriob-foundation`) ships it for scriob. **Other consumers
(`metallm`, `14-eng-ai-bot`, `14-eng-ai-bot-agents`, the `agent-wake` worktree) update when
they upgrade to this 3tears version.** Here is the complete change list.

## Breaking — rename these (mechanical, no behavior change)

On every `BaseCollection` / `SchemaBackedCollection` subclass override **and** every direct
caller:

| Old | New |
|-----|-----|
| `fetch_from_postgres`   | `fetch_from_store`   |
| `save_to_postgres`      | `save_to_store`      |
| `delete_from_postgres`  | `delete_from_store`  |
| `persist_to_postgres`   | `persist_to_store`   |

One sweep per repo:

```bash
grep -rlE 'fetch_from_postgres|save_to_postgres|delete_from_postgres|persist_to_postgres' --include='*.py' \
  | xargs sed -i '' \
      -e 's/fetch_from_postgres/fetch_from_store/g' \
      -e 's/save_to_postgres/save_to_store/g' \
      -e 's/delete_from_postgres/delete_from_store/g' \
      -e 's/persist_to_postgres/persist_to_store/g'
# verify: the grep above returns empty afterwards.
```

`SchemaBackedCollection` (the ~307 schema-driven inheritors) generates the new names
automatically — those subclasses need **no** change unless they *override* a seam method
or *call* it directly.

**Not renamed (no change):** `l3_pool` / `get_l3_pool`, `serialize` / `deserialize`, and the
`execute() -> str` status-tag contract all stay as-is.

### metallm enforcement test
`metallm api/tests/enforcement/test_schema_agreement.py` asserts `node.name != "save_to_postgres"`
(~line 131). Retarget it to `"save_to_store"`. Keep the empty-extraction `pytest.fail` guard.
`14-eng-ai-pentest-kit`'s `test_collection_contracts.py` is name-agnostic — no edit, confirm green.

## Additive — new, opt-in (nothing to change unless you want them)

- **`L3Backend` Protocol** (`threetears.core.backends.L3Backend`) — formalizes the raw-SQL
  transport (`fetch`/`fetchrow`/`execute`/`execute_batch`/`acquire`/`transaction`).
  `NatsProxyL3Backend` and the new `SqlL3Backend` conform.
- **`DurableStore` Protocol** (`threetears.core.backends.DurableStore`) — the **SQL-free**
  structured ops (`fetch_one`/`upsert`/`delete`/`scan`). The seam a non-SQL backend (e.g. a
  `GitL3Backend`) implements.
- **`SqlL3Backend`** — the default backend over an asyncpg pool; implements both protocols.
- **`DurableStoreCollection`** (`threetears.core.collections.DurableStoreCollection`) — a base
  whose L3 tier is a `DurableStore`; subclass it to back a collection with a non-SQL durable
  store and keep the full L1/L2 cache + cross-pod invalidation machinery.
- **`parse_rowcount`** (`threetears.core.backends.parse_rowcount`) — the one framework-owned
  asyncpg status-tag parser. Optionally replace local `int(result.split()[-1])` idioms with it.

## Verify after upgrading

```bash
grep -rEn 'fetch_from_postgres|save_to_postgres|delete_from_postgres|persist_to_postgres' --include='*.py'  # empty
./scripts/check-all.sh   # lint + mypy --strict + tests, exit 0
```

## Internal completion (no consumer action)

`collections-task-06` is complete — the items below are internal and do **not** change the
consumer contract above:

- **L3B-02** — `l3_pool` is typed `L3Backend | None`; the registry wraps a raw transport
  (asyncpg pool / `NatsProxyL3Backend`) in `SqlL3Backend` so the resolved handle always
  exposes the `DurableStore` ops (a backend already satisfying `DurableStore`, e.g. a
  `GitL3Backend`, passes through unwrapped).
- **L3B-03** — `SchemaBackedCollection`'s CRUD now routes through the structured `DurableStore`
  seam; the schema-aware SQL generation + value codec live in `SqlL3Backend` (driven by a
  registered `TableSchema`), so the SQL-backed and non-SQL (git) collections share one seam.
  Behavior is byte-identical (same projection casts, jsonb/vector/tsvector handling, composite
  PK, server-default dropping, CAS fencing, conflict modes).
- **L3B-04** — `flush_pending` persists a toposorted batch in one transaction when the backend
  exposes a usable `transaction()`, degrading to the per-entity loop (with the unchanged
  FK-deferral classification) otherwise.
