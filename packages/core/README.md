# 3tears Core

Three-tier caching library for Python applications. Provides collections (L1 SQLite -> L2 NATS KV -> L3 PostgreSQL) with subscript access, entity proxy objects, and configurable flush strategies.

## Architecture

```
L1 (SQLite, in-process, sync)  ->  L2 (NATS KV, shared, async)  ->  L3 (PostgreSQL, persistent, async)
```

- **L1**: In-memory SQLite via WAL mode. Sync access. Used by entity attribute reads/writes.
- **L2**: NATS KV shared cache. Async. Cross-pod consistency for multi-instance deployments.
- **L3**: PostgreSQL (or PostGIS, YugabyteDB, etc.). Async. Source of truth.

Reads promote up the stack (L3 miss -> L2 miss -> L1 hit on next access). Writes flow down (L1 -> L2 -> L3, with optional deferred flush).

## Quick Start

### 1. Configure the Registry

```python
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.cache.sqlite import SQLiteBackend

# Create and configure
l1 = SQLiteBackend("my_app_cache")
l1.initialize(sa_metadata)  # SQLAlchemy metadata with your table definitions

registry = CollectionRegistry()
registry.configure(
    l1_backend=l1,          # SQLiteBackend instance
    l2_client=nats_client,  # NATS client (optional, None to skip L2)
    l3_pool=postgres_pool,  # asyncpg pool
)
```

### 2. Per-Collection Pool Overrides

Different collections can use different databases:

```python
# Default: all collections use YugabyteDB
registry.configure(l3_pool=yugabyte_pool)

# Override: geo collection uses PostGIS
registry.configure()  # keep defaults
# When creating the collection, register with override:
geo_collection = GeoCollection(registry, config, nats_client, write_buffer)
registry.register(geo_collection, l3_pool=postgis_pool)
```

### 3. Define a Collection

```python
from threetears.core.collections.base import BaseCollection
from threetears.core.entities.base import BaseEntity

class UserEntity(BaseEntity):
    primary_key_field = "user_id"

class UsersCollection(BaseCollection[UserEntity]):
    primary_key_column = "user_id"

    @property
    def table_name(self) -> str:
        return "users"

    @property
    def entity_class(self) -> type[UserEntity]:
        return UserEntity

    async def _fetch_from_postgres(self, entity_id):
        row = await self.l3_pool.fetchrow(
            "SELECT * FROM users WHERE user_id = $1", entity_id
        )
        return dict(row) if row else None

    async def _save_to_postgres(self, data, original_timestamp=None):
        # INSERT or UPDATE with optimistic locking
        ...

    async def _delete_from_postgres(self, entity_id):
        await self.l3_pool.execute(
            "DELETE FROM users WHERE user_id = $1", entity_id
        )

    def _serialize(self, data):
        return json.dumps(data, default=str).encode()

    def _deserialize(self, data):
        return json.loads(data)
```

### 4. Create Collection Instances

```python
from threetears.core.collections.flush import WriteBuffer

write_buffer = WriteBuffer()
users = UsersCollection(registry, config, nats_client, write_buffer)
```

The `config` parameter must satisfy the `CoreConfig` protocol:

```python
class CoreConfig(Protocol):
    collection_flush: str           # "ALWAYS", "ON_CHECKPOINT", "ON_SCHEDULE", "ON_SHUTDOWN"
    collection_flush_interval: int  # seconds between scheduled flushes
    collection_flush_tables: str    # comma-separated table names eligible for deferred flush
```

## Access Patterns

### Subscript Access (sync, transparent pull-through)

Subscript access is the primary API. On L1 miss, data is transparently pulled through L2/L3 via a background event loop — no `await` needed, no `ensure()` required:

```python
# Read entity — pulls through L2/L3 automatically on L1 miss
entity = users[user_id]

# Read single field
name = users[user_id, "name_display"]

# Write single field (writes to L1, tracks for flush)
users[user_id, "name_display"] = "New Name"

# Write full entity data (writes dict to L1)
users[user_id] = {"user_id": user_id, "name_display": "New Name", ...}

# Check if entity is in L1 (does NOT pull through — L1 only)
if user_id in users:
    entity = users[user_id]
```

`__getitem__` raises `KeyError` only if the entity doesn't exist in any tier. The L1 fast path is ~microseconds; an L1 miss with pull-through adds ~50-200us bridge overhead plus the actual L2/L3 I/O time.

For hot-path code where you want to avoid the sync-async bridge overhead on first access, you can pre-warm L1:

```python
await users.ensure(user_id)  # async: pre-warms L1
entity = users[user_id]       # guaranteed L1 hit, no bridge needed
```

### Async Operations

```python
# Three-tier read: L1 -> L2 -> L3, promotes on miss. Returns None if not found.
entity = await users.get(user_id)

# Create a new entity (not persisted until save)
entity = users.create({"user_id": uuid7(), "name_display": "Alice", ...})

# Save through three-tier write path (L3 -> L1 -> L2)
await users.save_entity(entity)
# Or via entity directly:
await entity.save()

# Reload from L3 (discards local changes)
await entity.reload()

# Delete from all tiers
await users.delete(user_id)

# Invalidate L1 + L2 (force next read to hit L3)
await users.invalidate_cache(user_id)
```

### Entity Attribute Access

Entities are thin cache proxies. Field data lives in L1, not in the entity object.

```python
entity = await users.get(user_id)

# Read (checks entity._changes first, then L1 cache)
print(entity.name_display)

# Write (writes to L1 + tracks change)
entity.name_display = "Updated Name"

# Check dirty state
entity.is_dirty  # True after modification
entity.is_new    # True if created via collection.create()

# Get all changes
entity.get_changes()  # {"name_display": "Updated Name"}

# Export full entity data from L1
entity.to_dict()

# Persist
await entity.save()
```

## Flush Strategies

Controls when deferred writes reach L3 (PostgreSQL):

| Strategy | Behavior |
|---|---|
| `ALWAYS` | Every `save_entity()` writes to L3 immediately |
| `ON_CHECKPOINT` | Writes buffer to L1 + L2; flushes to L3 on explicit `flush_pending()` call |
| `ON_SCHEDULE` | Same as ON_CHECKPOINT but with timer-based auto-flush |
| `ON_SHUTDOWN` | Writes buffer; flushes to L3 on application shutdown |

Only tables listed in `collection_flush_tables` are eligible for deferred writes. All other tables always write immediately regardless of strategy.

## Optimistic Locking

Collections use `date_updated` for optimistic locking. When saving an existing entity, the `_save_to_postgres` implementation should check:

```sql
UPDATE users SET ... WHERE user_id = $1 AND date_updated = $2
```

If `rows_affected == 0` for an UPDATE, `BaseCollection.save_entity()` raises `ConcurrentModificationError`.

## Subclassing Guide

**BaseEntity**: Set `primary_key_field` to your PK column name. Add computed properties as needed. Do NOT store data in instance attributes — all data lives in L1.

**BaseCollection**: Set `primary_key_column`. Implement the 5 abstract methods: `_fetch_from_postgres`, `_save_to_postgres`, `_delete_from_postgres`, `_serialize`, `_deserialize`. Use `self.l3_pool` for database access. Add domain-specific query methods (e.g., `find_by_email`).
