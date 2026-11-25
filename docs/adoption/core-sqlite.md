# 3tears: the SQLite L3

`threetears.core.backends.sqlite_file.SqliteL3Backend` -- one SQLite file as the durable tier (L3), for a
deployment of one process on one machine. Part of [`core`](core.md), an L3 beside `SqlL3Backend` (Postgres or
YugabyteDB, directly or through the NATS proxy) and the [git store](core-git.md).

## Choosing an L3

| L3 | Use it when | Collections over it |
|---|---|---|
| `SqlL3Backend` (Postgres / YugabyteDB) | many pods, many writers, data a platform shares; migrations, indexes, SQL | `SchemaBackedCollection`, `DataStore`, `DurableStoreCollection`; every feature (change notes, scope epochs, delta files, Arrow copies, partition loads) |
| `SqliteL3Backend` (a file) | one process on one machine, no database server and no NATS: a local tool's state (queues, ledgers, watches) that must survive a restart | `SchemaBackedCollection` and `DurableStoreCollection`: save, read, compare-and-swap, delete, scans, many-row writes, a caller's transaction |
| `GitDurableStore` (a working tree) | authored data that wants git's history, diffs, reverts and hand edits | `DurableStoreCollection`; one commit per change set, one writer per repository |

The same collections run on SQLite and on Postgres: moving a deployment from one to the other is the L3 URL in its
configuration (`l3_store`, below) and, on many machines, an L2. What both stores owe a collection, and what all
three owe, is one contract suite (`packages/core/tests/l3_contract.py`) each store runs unchanged:

- every L3 (Postgres, SQLite, git): a row written whole and read back, the conflict mode a write asks for, the
  compare-and-swap on `date_updated`, deletes, composite keys, equality scans, many rows at once, a transaction
  committing whole or not at all, and a `DurableStoreCollection` over it with L2 and without;
- the schema stores (Postgres, SQLite): typed values from the registered `TableSchema` (UUID, `NUMERIC` as
  `Decimal`, bytes, a naive instant read as UTC), an update rewriting the columns it names and keeping the rest,
  immutable columns, the schema's conflict mode, the NULL-safe and the ordered compare-and-swap, truncate and raw SQL.

## Use

```python
from threetears.core.backends import l3_store
from threetears.core.cache import SQLiteBackend
from threetears.core.collections import NO_L2
from threetears.core.collections.registry import CollectionRegistry

async with l3_store("sqlite:////var/lib/app/state.sqlite") as store:   # or a postgresql:// DSN
    registry = CollectionRegistry()
    registry.configure(l1_backend=SQLiteBackend("app"), l3_pool=store)
    queue = QueueCollection(registry, config, NO_L2)     # a SchemaBackedCollection; NO_L2 on one machine
    await queue.save_entity(queue.create(row))
```

`sqlite:////abs/path` is an absolute path and `sqlite:///rel/path` a relative one (what follows the third slash is
the path as written). `SqliteL3Backend(path)` with `open()` / `close()` (or `async with`) is the same store without
the URL.

## The file

- **Durable.** Write-ahead log, every commit synced (`synchronous=FULL`): a write that returned is in the file after
  the process is killed with SIGKILL (`tests/unit/backends/test_sqlite_l3_survives_kill.py` kills a writer after
  each kind of write and finds it).
- **One process.** `open()` takes an exclusive `flock` on `<path>.lock`; a second process gets
  `SqliteStoreLockedError`. The kernel drops the lock when the holder dies, however it dies.
- **Within the process**, writes take one lock (a statement, or a whole transaction); reads use a second
  connection and see the last commit without waiting for a writer. Statements run on worker threads. A write
  outside the transaction its own task holds is refused rather than left to wait on itself.
- **Format.** The file's `user_version` is the store's format; a file of another format, or one another program
  wrote, is refused (`SqliteStoreFormatError`) rather than misread.
- **Tables.** A collection's `TableSchema` is registered when it is built (`register_schema`), which creates its
  table, or checks the stored one: a new nullable or defaulted column is added; a dropped or retyped column, or a new
  required one, is refused with the difference named. Indexes, foreign keys and unique constraints other than the
  primary key are not created. `server_default` is spelled for `now()`, a quoted literal, a number and
  `true`/`false`; another default is refused at registration.
- **Health.** `await store.health()` answers while the file reads and raises when it does not, for a readiness probe.

## The L1 in front

The L1 is the ordinary SQLite cache, its eviction budget and LRU included: an evicted row is read back from the
file, so no eviction loses one.

## Limits

- The statements are those the Postgres store generates, with `$N` placeholders and `::type` casts rewritten.
  Statements written for Postgres alone (`= ANY($1::uuid[])`, `jsonb` operators, `COPY`, Postgres functions) fail as
  SQLite reports them, so the collection features built on them -- change notes, scope epochs, delta files, Arrow
  copies, partition loads, `bypassing_write` over raw SQL -- are Postgres features.
- A `NUMERIC` is kept as its decimal text: equality filters match the text, so `1.50` and `1.5` are different values
  to a scan's filter.
- One machine. On many machines use Postgres (and L2); a deployment moves by configuration.
